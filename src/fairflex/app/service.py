"""Safe orchestration layer for the FairFlex Demonstrator."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import UTC, datetime
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
from platform import python_version
from time import perf_counter
from typing import Any
from uuid import uuid4

import numpy as np

from .. import __version__
from ..evaluation import summarize_outcome
from ..fair_mpc import CentralizedFairMPC
from ..safety import ACRepairedMPCPolicy
from ..simulation import ChargingSimulation, SimulationResult
from ..v3.lower_tail_mpc import LowerTailFairMPC
from .catalog import ScenarioCatalog, ScenarioDefinition
from .repository import AuditRepository, RunNotFoundError


APP_VERSION = "demonstrator-1.0.0"


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _json_safe(value: Any) -> Any:
    """Convert numerical research objects to strict JSON values."""

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _hash_payload(value: dict[str, Any]) -> str:
    canonical = json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _project_root() -> Path:
    """Return the repository root for the bundled local research release."""

    return Path(__file__).resolve().parents[3]


def _verify_source_derivation(evidence: dict[str, Any]) -> dict[str, Any]:
    """Re-extract the local evidence payload and compare every displayed value.

    Source-file hashes alone cannot prove that a manually edited browser ledger
    still contains the values derived from those files.  The local
    demonstrator has the extraction script and frozen artifacts available, so
    it can perform this bounded read-only check when `/api/evidence` is read.
    """

    builder_path = _project_root() / "scripts" / "build_app_evidence.py"
    try:
        if not builder_path.is_file():
            raise FileNotFoundError("evidence extraction script is unavailable")
        spec = spec_from_file_location("fairflex_runtime_evidence_builder", builder_path)
        if spec is None or spec.loader is None:
            raise RuntimeError("could not load the evidence extraction script")
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        expected = module.build()
        expected_hash = expected.get("source_derived_payload_sha256")
        actual_hash = evidence.get("source_derived_payload_sha256")
        return {
            "status": "verified" if expected == evidence else "mismatch",
            "verified": expected == evidence,
            "expected_sha256": expected_hash,
            "actual_sha256": actual_hash,
        }
    except Exception as exc:  # The UI must not claim verification if packaging is incomplete.
        return {
            "status": "unavailable",
            "verified": False,
            "expected_sha256": None,
            "actual_sha256": evidence.get("source_derived_payload_sha256"),
            "detail": f"{type(exc).__name__}: {str(exc)[:160]}",
        }


def _verify_artifact_ledger(evidence: dict[str, Any]) -> dict[str, Any]:
    """Check that every chart source still matches its frozen SHA-256.

    The application remains usable when a source is absent (for example, in a
    thin demo bundle), but it does not silently call the evidence verified.
    """

    root = _project_root().resolve()
    entries: list[dict[str, Any]] = []
    for source in evidence.get("artifact_ledger", []):
        relative_path = str(source.get("path", ""))
        candidate = (root / relative_path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            entries.append({
                "path": relative_path,
                "purpose": source.get("purpose", "Frozen evidence source"),
                "status": "invalid_path",
                "expected_sha256": source.get("sha256"),
                "actual_sha256": None,
            })
            continue
        if not candidate.is_file():
            entries.append({
                "path": relative_path,
                "purpose": source.get("purpose", "Frozen evidence source"),
                "status": "unavailable",
                "expected_sha256": source.get("sha256"),
                "actual_sha256": None,
            })
            continue
        actual_hash = _file_sha256(candidate)
        expected_hash = str(source.get("sha256", ""))
        entries.append({
            "path": relative_path,
            "purpose": source.get("purpose", "Frozen evidence source"),
            "status": "verified" if actual_hash == expected_hash else "mismatch",
            "expected_sha256": expected_hash,
            "actual_sha256": actual_hash,
        })
    matched = sum(entry["status"] == "verified" for entry in entries)
    source_derived_payload = dict(evidence)
    declared_payload_hash = source_derived_payload.pop("source_derived_payload_sha256", None)
    actual_payload_hash = _hash_payload(source_derived_payload)
    payload_verified = declared_payload_hash == actual_payload_hash
    source_derivation = _verify_source_derivation(evidence)
    source_verified = bool(entries) and matched == len(entries)
    return {
        "status": "verified" if source_verified and payload_verified and source_derivation["verified"] else "attention_required",
        "verified": source_verified and payload_verified and source_derivation["verified"],
        "checked": len(entries),
        "matched": matched,
        "entries": entries,
        "derived_payload": {
            "expected_sha256": declared_payload_hash,
            "actual_sha256": actual_payload_hash,
            "verified": payload_verified,
        },
        "source_derivation": source_derivation,
        "statement": (
            "Every displayed frozen result was freshly re-extracted from locally hash-verified source artifacts."
            if source_verified and payload_verified and source_derivation["verified"]
            else "A frozen evidence source, source-derived payload, or fresh extraction does not match. Do not treat the displayed summary as locally hash-verified."
        ),
    }


def _installed_version(package_name: str) -> str:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "not-installed"


def _execution_manifest(policy_key: str, input_hash: str) -> dict[str, Any]:
    """Capture the settings that affect a reproducible local engine run."""

    controller = (
        {"class": "CentralizedFairMPC", "objective": "lexicographic max-min service, energy, economic tie-break"}
        if policy_key == "v1_default"
        else {"class": "LowerTailFairMPC", "tail_fraction": 0.10, "objective": "lower-tail shortfall, energy, economic tie-break"}
    )
    manifest = {
        "application_version": APP_VERSION,
        "engine_version": __version__,
        "python_version": python_version(),
        "executed_input_sha256": input_hash,
        "controller": controller,
        "safety_wrapper": {"class": "ACRepairedMPCPolicy", "horizon_steps": 4},
        "dependency_versions": {name: _installed_version(name) for name in ("cvxpy", "numpy", "pandapower", "highspy", "osqp")},
        "source_revision": "local working tree (record a Git commit before external release)",
    }
    return {"manifest": manifest, "manifest_sha256": _hash_payload(manifest)}


def _policy_metadata(policy_key: str) -> dict[str, Any]:
    if policy_key == "v1_default":
        return {
            "id": "v1_fair_mpc_ac_demo",
            "label": "V1 Fair MPC",
            "role": "recommended",
            "description": "Lexicographic fair MPC executed through the exact AC-feasibility repair wrapper.",
            "disclosure": "Recommended for this demonstrator. This controlled teaching fixture has no early-unplug forecast, so the historical multi-rate guard is not fitted here.",
        }
    if policy_key == "v3_shadow":
        return {
            "id": "v3_lower_tail_shadow_demo",
            "label": "V3 lower-tail shadow",
            "role": "research_shadow",
            "description": "Lower-tail MPC executed on the same input snapshot. It cannot control the V1 recommendation.",
            "disclosure": "Research shadow mode. This teaching comparison isolates the lower-tail objective; it does not refit the historical CQR guard on this scenario.",
        }
    raise ValueError(f"unknown internal policy key: {policy_key}")


def _make_policy(policy_key: str, simulation: ChargingSimulation) -> ACRepairedMPCPolicy:
    if policy_key == "v1_default":
        controller = CentralizedFairMPC()
    elif policy_key == "v3_shadow":
        controller = LowerTailFairMPC(tail_fraction=0.10)
    else:
        raise ValueError(f"unknown internal policy key: {policy_key}")
    return ACRepairedMPCPolicy(controller, simulation.grid, horizon_steps=4)


def _session_explanations(snapshot: dict[str, Any], result: SimulationResult) -> list[dict[str, Any]]:
    first_step = result.steps[0].allocations_kw if result.steps else {}
    ratio_by_id = result.service_ratios
    explanations: list[dict[str, Any]] = []
    for session in snapshot["sessions"]:
        available_steps = session["planning_deadline_step"] - session["arrival_step"]
        max_energy = available_steps * session["max_power_kw"] * snapshot["step_hours"]
        slack_kwh = max(0.0, max_energy - session["requested_energy_kwh"])
        allocation = float(first_step.get(session["ev_id"], 0.0))
        is_urgent = available_steps <= 1 or slack_kwh <= 1e-8
        reason = (
            f"{session['ev_id']} has only {available_steps} planning step(s) available, so FairFlex reserves immediate capacity."
            if is_urgent
            else f"{session['ev_id']} has {slack_kwh:.2f} kWh of timing slack, so FairFlex can share capacity while protecting tighter deadlines."
        )
        explanations.append(
            {
                "ev_id": session["ev_id"],
                "station_id": session["station_id"],
                "requested_energy_kwh": session["requested_energy_kwh"],
                "allocated_first_step_kw": allocation,
                "service_ratio": float(ratio_by_id[session["ev_id"]]),
                "planning_deadline_step": session["planning_deadline_step"],
                "available_steps": available_steps,
                "slack_kwh": slack_kwh,
                "at_risk": bool(is_urgent or ratio_by_id[session["ev_id"]] < 0.9),
                "reason": reason,
            }
        )
    return explanations


def _run_policy(scenario: ScenarioDefinition, snapshot: dict[str, Any], policy_key: str) -> dict[str, Any]:
    """Run the existing engine on a fresh mutable simulation instance."""

    # Reconstruct from the persisted, complete specification.  Do not call
    # scenario.factory() here: otherwise a later code edit could make the
    # recorded input hash describe something other than the executed replay.
    execution_input = ScenarioCatalog.execution_input(snapshot)
    simulation = ScenarioCatalog.simulation_from_snapshot(snapshot)
    policy = _make_policy(policy_key, simulation)
    started = perf_counter()
    result = simulation.run(policy)
    metrics = summarize_outcome(_policy_metadata(policy_key)["id"], result, runtime_seconds=perf_counter() - started)
    steps = [
        {
            "step": step.step,
            "allocations_kw": _json_safe(dict(step.allocations_kw)),
            "station_powers_kw": _json_safe(dict(step.station_powers_kw)),
            "grid": {
                "safe": step.grid.safe,
                "converged": step.grid.converged,
                "min_voltage_pu": step.grid.min_voltage_pu,
                "max_line_loading_percent": step.grid.max_line_loading_percent,
                "violations": list(step.grid.violations),
            },
        }
        for step in result.steps
    ]
    safety = {
        "safe": result.unsafe_steps == 0,
        "unsafe_steps": result.unsafe_steps,
        "min_voltage_pu": metrics.min_voltage_pu,
        "max_line_loading_percent": metrics.max_line_loading_percent,
        "statement": (
            "Every executed replay step passed the configured AC feasibility checks."
            if result.unsafe_steps == 0
            else "Unsafe replay steps were detected. FairFlex will not present this as a safe recommendation."
        ),
    }
    return _json_safe(
        {
            "policy": _policy_metadata(policy_key),
            "metrics": asdict(metrics),
            "safety": safety,
            "steps": steps,
            "session_outcomes": _session_explanations(snapshot, result),
            "limitations": list(scenario.limitations) + [
                "Simulation/replay safety validation is not physical field deployment.",
                _policy_metadata(policy_key)["disclosure"],
            ],
            "execution_input": {
                "schema_version": execution_input["schema_version"],
                "sha256": _hash_payload(execution_input),
                "verified": True,
            },
        }
    )

class DemonstratorService:
    """Run bounded jobs and preserve a complete browser-safe audit trail."""

    def __init__(self, data_dir: Path) -> None:
        self.catalog = ScenarioCatalog()
        self.repository = AuditRepository(data_dir / "fairflex.db")
        self.repository.reconcile_interrupted(_now())
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fairflex-demo")

    def health(self) -> dict[str, Any]:
        return {
            "status": "ready",
            "app_version": APP_VERSION,
            "engine_version": __version__,
            "offline_demo_ready": True,
            "external_credentials_required": False,
            "statement": "This server runs offline controlled teaching simulations only. It does not control a physical charger.",
        }

    def start_run(self, scenario_id: str, include_v3_shadow: bool) -> dict[str, str]:
        snapshot = self.catalog.snapshot(scenario_id)
        run_id = str(uuid4())
        input_hash = _hash_payload(ScenarioCatalog.execution_input(snapshot))
        self.repository.create_run(
            run_id=run_id,
            scenario_id=scenario_id,
            created_at=_now(),
            input_hash=input_hash,
            scenario=snapshot,
        )
        self._executor.submit(self._execute_v1, run_id, include_v3_shadow)
        return {"run_id": run_id, "status": "queued"}

    def _execute_v1(self, run_id: str, include_v3_shadow: bool) -> None:
        self.repository.mark_running(run_id, _now())
        try:
            record = self.repository.get(run_id)
            scenario = self.catalog.get(record["scenario_id"])
            result = _run_policy(scenario, record["scenario"], "v1_default")
            if result["execution_input"]["sha256"] != record["input_hash"]:
                raise RuntimeError("saved input hash does not match reconstructed execution input")
            result["execution_manifest"] = _execution_manifest("v1_default", record["input_hash"])
            if not result["safety"]["safe"]:
                raise RuntimeError("executed V1 action did not pass safety validation")
            self.repository.complete(run_id, _now(), result)
            if include_v3_shadow:
                claimed, _ = self.repository.claim_comparison(run_id, _now())
                if claimed:
                    self._execute_v3(run_id)
        except Exception as exc:  # Fail closed: no unsafe/stale allocation is returned.
            self.repository.fail(run_id, _now(), "execution_failed", str(exc))

    def start_v3_shadow(self, run_id: str) -> dict[str, str]:
        claimed, detail = self.repository.claim_comparison(run_id, _now())
        if not claimed:
            raise ValueError(detail)
        self._executor.submit(self._execute_v3, run_id)
        return {"run_id": run_id, "status": "queued"}

    def _execute_v3(self, run_id: str) -> None:
        input_hash: str | None = None
        try:
            if not self.repository.mark_comparison_running(run_id, _now()):
                return
            record = self.repository.get(run_id)
            input_hash = record["input_hash"]
            if record["status"] != "succeeded" or record["result"] is None:
                raise ValueError("V3 shadow comparison requires a successful V1 result")
            scenario = self.catalog.get(record["scenario_id"])
            shadow = _run_policy(scenario, record["scenario"], "v3_shadow")
            shadow["execution_manifest"] = _execution_manifest("v3_shadow", input_hash)
            same_input_verified = (
                shadow["execution_input"]["sha256"] == input_hash
                and _hash_payload(ScenarioCatalog.execution_input(record["scenario"])) == input_hash
            )
            comparison = {
                "input_hash": input_hash,
                "same_input_verified": same_input_verified,
                "status": "succeeded",
                "v3": shadow,
                "interpretation": "Research shadow comparison only. V3 cannot replace the V1 recommendation in this application.",
            }
            self.repository.attach_comparison(run_id, _now(), comparison)
        except Exception as exc:
            # Preserve the safe V1 result. The comparison failure is transparent in audit events.
            self.repository.attach_comparison(
                run_id,
                _now(),
                {
                    "input_hash": input_hash,
                    "same_input_verified": False,
                    "status": "failed",
                    "error": "V3 shadow evaluation failed; V1 remains unchanged.",
                    "diagnostic_code": type(exc).__name__,
                },
            )
        finally:
            pass

    def get_run(self, run_id: str) -> dict[str, Any]:
        return self.repository.get(run_id)

    def list_runs(self) -> list[dict[str, Any]]:
        return self.repository.list_runs()

    def evidence(self) -> dict[str, Any]:
        evidence_path = Path(__file__).with_name("frozen_evidence.json")
        raw = evidence_path.read_bytes()
        evidence = json.loads(raw)
        return {
            **evidence,
            "evidence_manifest_sha256": sha256(raw).hexdigest(),
            "artifact_verification": _verify_artifact_ledger(evidence),
        }

    def evidence_report(self) -> str:
        """Export the visible research ledger as a compact Markdown appendix."""

        evidence = self.evidence()
        verification = evidence["artifact_verification"]
        primary = evidence["primary_benchmark"]
        lines = [
            "# FairFlex frozen evidence ledger",
            "",
            "## Scope",
            evidence["scope"],
            "",
            "## Integrity",
            f"- Evidence manifest SHA-256: `{evidence['evidence_manifest_sha256']}`",
            f"- Source artifacts verified: {verification['matched']} / {verification['checked']}",
            f"- Status: {'VERIFIED' if verification['verified'] else 'ATTENTION REQUIRED'}",
            f"- {verification['statement']}",
            "",
            "## Primary baseline study",
            f"- Study: {primary['label']}",
            f"- Site and window: {primary['site']}, {primary['dates']}",
            f"- Sessions: {primary['sessions']}",
            f"- Assumptions: {primary['assumptions']}",
            f"- Inference: {primary['inference_unit']}",
            "",
            "## P10 effects — robust-PV FairFlex minus baseline",
        ]
        for effect in primary["p10_effects"]:
            lines.append(
                f"- {effect['baseline']}: {effect['effect']:+.4f}; 95% CI [{effect['ci_low']:+.4f}, {effect['ci_high']:+.4f}], {effect['days']} matched days"
            )
        lines.extend(["", "## Frozen source artifact ledger"])
        for entry in verification["entries"]:
            lines.append(
                f"- {entry['status'].upper()}: `{entry['path']}` — {entry['purpose']} — expected `{entry['expected_sha256']}`"
            )
        lines.extend(["", "## Limitations"])
        lines.extend(f"- {limitation}" for limitation in evidence["limitations"])
        return "\n".join(lines) + "\n"

    def report(self, run_id: str) -> str:
        record = self.get_run(run_id)
        result = record["result"]
        lines = [
            f"# FairFlex demonstrator audit — {run_id}",
            "",
            "## Scope",
            record["scenario"]["disclosure"],
            "",
            "## Run provenance",
            f"- Scenario: {record['scenario']['label']}",
            f"- Scenario ID: {record['scenario_id']}",
            f"- Input SHA-256: `{record['input_hash']}`",
            f"- Created: {record['created_at']}",
            f"- Status: {record['status']}",
        ]
        if result:
            metrics = result["metrics"]
            manifest_hash = result.get("execution_manifest", {}).get("manifest_sha256", "not available for this older local record")
            lines.extend([
                "",
                "## V1 recommendation",
                f"- Policy: {result['policy']['label']} ({result['policy']['role']})",
                f"- Safety: {'PASS' if result['safety']['safe'] else 'FAIL'}",
                f"- P10 service ratio: {metrics['p10_service_ratio']:.3f}",
                f"- Mean service ratio: {metrics['mean_service_ratio']:.3f}",
                f"- Jain service index: {metrics['jain_service_index']:.3f}",
                f"- Delivered energy: {metrics['delivered_energy_kwh']:.3f} kWh",
                f"- Unsafe steps: {metrics['unsafe_steps']}",
                f"- Execution-manifest SHA-256: `{manifest_hash}`",
                f"- Executed-input SHA-256: `{result.get('execution_input', {}).get('sha256', record['input_hash'])}`",
                "",
                "## Execution manifest",
            ])
            manifest = result.get("execution_manifest", {}).get("manifest", {})
            if manifest:
                controller = manifest.get("controller", {})
                safety_wrapper = manifest.get("safety_wrapper", {})
                lines.extend([
                    f"- Controller: {controller.get('class', 'not recorded')}",
                    f"- Objective: {controller.get('objective', 'not recorded')}",
                    f"- Safety wrapper: {safety_wrapper.get('class', 'not recorded')}",
                    f"- Horizon: {safety_wrapper.get('horizon_steps', 'not recorded')} steps",
                    f"- Engine version: {manifest.get('engine_version', 'not recorded')}",
                    f"- Source revision: {manifest.get('source_revision', 'not recorded')}",
                ])
            lines.extend([
                "",
                "## Per-EV outcome",
            ])
            for session in result["session_outcomes"]:
                lines.append(
                    f"- {session['ev_id']}: first allocation {session['allocated_first_step_kw']:.3f} kW; service ratio {session['service_ratio']:.3f}. {session['reason']}"
                )
            lines.extend(["", "## Limitations"])
            lines.extend(f"- {limitation}" for limitation in result["limitations"])
        if record["comparison"]:
            comparison = record["comparison"]
            lines.extend(["", "## V3 shadow status", comparison.get("interpretation", comparison.get("error", "Available."))])
            if comparison.get("v3"):
                v3_metrics = comparison["v3"]["metrics"]
                lines.extend([
                    f"- V3 P10 service ratio: {v3_metrics['p10_service_ratio']:.3f}",
                    f"- V3 mean service ratio: {v3_metrics['mean_service_ratio']:.3f}",
                    f"- V3 runtime: {v3_metrics['runtime_seconds']:.3f} seconds",
                ])
        lines.extend([
            "",
            "## Local-audit limitation",
            "- This record is stored in local SQLite. It is a reproducibility aid, not a signed or remotely immutable ledger.",
        ])
        return "\n".join(lines) + "\n"

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


__all__ = ["APP_VERSION", "DemonstratorService", "RunNotFoundError"]
