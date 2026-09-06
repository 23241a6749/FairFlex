"""Build the browser-safe FairFlex evidence ledger from frozen result artifacts.

This is intentionally an extraction-only script. It never reruns, selects, or
retunes a policy; it copies pre-existing frozen result summaries into the
application bundle along with source-file SHA-256 hashes.
"""

from __future__ import annotations

import csv
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "src" / "fairflex" / "app" / "frozen_evidence.json"


def _path(relative: str) -> Path:
    return ROOT / relative.replace("/", "\\")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _csv(relative: str) -> list[dict[str, str]]:
    with _path(relative).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _json(relative: str) -> dict[str, Any]:
    return json.loads(_path(relative).read_text(encoding="utf-8"))


def _number(value: str) -> float:
    return float(value)


def _row(rows: list[dict[str, str]], policy: str) -> dict[str, str]:
    return next(row for row in rows if row["policy"] == policy)


def _cohort(
    *,
    label: str,
    site: str,
    dates: str,
    role: str,
    metric_file: str,
    notes: str,
) -> dict[str, Any]:
    rows = _csv(metric_file)
    v1 = _row(rows, "v1_fair_mpc_multirate_guard")
    v3 = _row(rows, "fairflex_uc_hybrid_lower_tail")
    return {
        "cohort": label,
        "site": site,
        "dates": dates,
        "role": role,
        "sessions": int(v1["sessions"]),
        "v1": _number(v1["p10_service_ratio"]),
        "v3": _number(v3["p10_service_ratio"]),
        "v1_energy_kwh": _number(v1["delivered_energy_kwh"]),
        "v3_energy_kwh": _number(v3["delivered_energy_kwh"]),
        "v1_unsafe_steps": int(float(v1["unsafe_steps"])),
        "v3_unsafe_steps": int(float(v3["unsafe_steps"])),
        "notes": notes,
    }


def _artifact(relative: str, purpose: str) -> dict[str, str]:
    return {"path": relative, "sha256": _sha256(_path(relative)), "purpose": purpose}


def _canonical_payload_sha256(payload: dict[str, Any]) -> str:
    """Hash a stable JSON payload without relying on its file formatting."""

    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build() -> dict[str, Any]:
    """Extract the complete browser ledger from fixed, pre-existing artifacts."""
    october_audit = _csv("artifacts/v3/holdout_october_statistical_audit/paired_calendar_day_statistical_audit.csv")
    november_audit = _csv("artifacts/v3/caltech_november_continuous_statistical_audit/paired_calendar_day_statistical_audit.csv")
    marl_audit = _csv("artifacts/v3/safe_marl_final_statistical_audit_endpoint_amendment/safe_marl_seed_ensemble_statistical_audit.csv")
    primary_effects = _csv("artifacts/paper_comparison_jpl_november/evidence_package/primary_paired_comparisons.csv")
    primary_table = _csv("artifacts/paper_comparison_jpl_november/evidence_package/primary_policy_table.csv")
    reliability = _csv("artifacts/paper_comparison_jpl_november/evidence_package/guard_reliability_table.csv")
    sensitivity = _csv("artifacts/v3/caltech_october_frozen_sensitivity/scenario_hybrid_vs_v1.csv")
    calibration = _json("artifacts/v3/holdout_october_cqr_prediction_audit_v3/test_cqr_prediction_summary.json")

    def audit_row(rows: list[dict[str, str]], metric: str) -> dict[str, str]:
        return next(row for row in rows if row["metric"] == metric)

    def inference(label: str, rows: list[dict[str, str]]) -> dict[str, Any]:
        row = audit_row(rows, "p10_service_ratio")
        return {
            "cohort": label,
            "eligible_days": int(row["eligible_calendar_days"]),
            "v3_minus_v1_p10": _number(row["observed_treatment_advantage"]),
            "ci_low": _number(row["advantage_bootstrap_ci_low"]),
            "ci_high": _number(row["advantage_bootstrap_ci_high"]),
            "one_sided_p": _number(row["p_value_one_sided"]),
            "reading": "Inconclusive: the paired-day interval crosses zero.",
        }

    baseline_names = {
        "uncontrolled_shared_cap": "Uncontrolled (capped)",
        "fcfs_shared_cap": "FCFS",
        "edf_shared_cap": "EDF",
        "equal_share_shared_cap": "Equal share",
        "least_laxity_first_shared_cap": "LLF",
        "fair_mpc_ac_no_pv": "FairFlex without PV robustness",
    }
    primary_p10 = [
        {
            "baseline": baseline_names[row["baseline"]],
            "effect": _number(row["observed_treatment_advantage"]),
            "ci_low": _number(row["advantage_ci_low"]),
            "ci_high": _number(row["advantage_ci_high"]),
            "days": int(row["blocks"]),
        }
        for row in primary_effects
        if row["evaluation_group"] == "paper_primary_jpl_november_unseen"
        and row["metric"] == "p10_service_ratio"
        and row["baseline"] in baseline_names
    ]
    policy_rows = [
        {
            "policy": row["policy_display"],
            "sessions": int(row["total_sessions"]),
            "p10": _number(row["p10_service_ratio"]),
            "mean": _number(row["mean_service_ratio"]),
            "jain": _number(row["jain_service_index"]),
            "delivered_energy_kwh": _number(row["total_delivered_energy_kwh"]),
            "unsafe_steps": int(float(row["unsafe_steps"])),
        }
        for row in primary_table
    ]
    sensitivity_labels = {
        "baseline_import_10kw": "10 kW base import",
        "higher_import_12kw": "12 kW import",
        "lower_import_8kw": "8 kW import",
        "higher_pv_45kw": "45 kW AC PV proxy",
        "lower_pv_15kw": "15 kW AC PV proxy",
    }
    payload = {
        "schema_version": 2,
        "scope": "Frozen FairFlex historical ACN evaluation summaries. The dashboard reads this immutable-at-build evidence ledger; it does not rerun, select, or retune research policies.",
        "policy_decision": {
            "default": "Live V1 demo: Centralized Fair MPC + AC repair",
            "shadow": "Live V3 demo: lower-tail MPC + AC repair (teaching-only shadow)",
            "research_v1": "Frozen V1 comparator: Fair MPC + multi-rate deadline guard",
            "research_v3": "Frozen V3: CQR + conservative hybrid deadline envelope + lower-tail MPC",
            "conclusion": "V3 has promising all-session lower-tail estimates, but paired calendar-day tests do not establish a reliable V3 improvement over the frozen V1 comparator.",
        },
        "data_lineage": [
            {
                "layer": "Interactive demonstrator",
                "data": "Bundled deterministic FairFlex teaching fixtures",
                "used_for": "Live V1/V3 mechanics, allocation explanation, and AC-repair demonstration",
                "scope": "2-EV deadline-stress or 5-EV multi-station scenario; not a population-performance estimate.",
            },
            {
                "layer": "Historical charging sessions",
                "data": "ACN-Data session records from Caltech and JPL",
                "used_for": "Frozen chronological trace replay, training/calibration labels, and held-out evaluation",
                "scope": "Connection/disconnect times, declared departure and energy request, station context, delivered energy, and observed charge/current traces where needed.",
            },
            {
                "layer": "Solar and feeder assumptions",
                "data": "Regional NSRDB/PVWatts proxy plus controlled shared-import and IEEE-33 AC sensitivity model",
                "used_for": "Forecast reserve and electrical-feasibility sensitivity evaluation",
                "scope": "Not measured Caltech or JPL rooftop PV or feeder telemetry; every screen must retain this limitation.",
            },
        ],
        "v3_cohorts": [
            _cohort(label="Caltech October frozen holdout", site="Caltech", dates="2019-10-01 to 2019-10-08", role="Frozen temporal holdout", metric_file="artifacts/v3/holdout_october_frozen/test_fairflex_uc_metrics.csv", notes="154 sessions; V3 selection and hyperparameters were frozen before this holdout."),
            _cohort(label="Caltech November continuous", site="Caltech", dates="2019-11-01 to 2019-11-30", role="Later temporal replication", metric_file="artifacts/v3/caltech_november_temporal_replication_continuous/test_fairflex_uc_metrics.csv", notes="615 sessions; continuous replay with calendar-day inference."),
            _cohort(label="JPL October external", site="JPL", dates="2019-10-01 to 2019-10-08", role="External site replication", metric_file="artifacts/v3/jpl_external_october_frozen/test_fairflex_uc_metrics.csv", notes="379 sessions; frozen design with site-calibrated historical guard/CQR, not zero-shot transfer."),
        ],
        "pooled_p10": [],
        "calendar_day_inference": [inference("Caltech October frozen holdout", october_audit), inference("Caltech November continuous", november_audit)],
        "primary_benchmark": {
            "label": "Frozen JPL November V1 baseline comparison",
            "site": "JPL",
            "dates": "2019-11-01 to 2019-11-29 (29 non-empty days)",
            "sessions": 1320,
            "assumptions": "Synthetic 10 kW shared import, regional NSRDB/PVWatts proxy, and the same AC execution-safety check for every policy.",
            "inference_unit": "Matched calendar day; 95% paired day-block bootstrap intervals (5,000 resamples).",
            "p10_effects": primary_p10,
            "policy_table": policy_rows,
        },
        "guard_reliability": [
            {
                "condition": row["condition"].replace("_", " "),
                "days": int(row["days"]),
                "decisions": int(row["decisions"]),
                "coverage": _number(row["observed_coverage"]),
                "ci_low": _number(row["coverage_ci_low"]),
                "ci_high": _number(row["coverage_ci_high"]),
                "target": _number(row["target_coverage"]),
                "mean_buffer_minutes": _number(row["decision_weighted_mean_buffer_minutes"]),
            }
            for row in reliability
        ],
        "cqr_calibration": {
            "site": "Caltech October frozen holdout",
            "sessions": calibration["observed_coverage"]["sessions"],
            "target": calibration["target_coverage"],
            "coverage": calibration["observed_coverage"]["rate"],
            "ci_low": calibration["observed_coverage"]["calendar_day_cluster_bootstrap_95_percent_descriptive"]["ci_low"],
            "ci_high": calibration["observed_coverage"]["calendar_day_cluster_bootstrap_95_percent_descriptive"]["ci_high"],
            "covered_sessions": calibration["observed_coverage"]["covered_sessions"],
            "interpretation": "Coverage is uncertainty calibration, not classifier accuracy. The CQR model is frozen; the observed interval is descriptive and respects day clustering.",
        },
        "sensitivity": [
            {
                "condition": sensitivity_labels[row["scenario"]],
                "p10_difference": _number(row["hybrid_minus_v1_p10_service_ratio"]),
                "early_unplug_p10_difference": _number(row["hybrid_minus_v1_early_unplug_p10_service_ratio"]),
                "shortfall_difference": _number(row["hybrid_minus_v1_expected_shortfall"]),
                "energy_difference_kwh": _number(row["hybrid_minus_v1_delivered_energy_kwh"]),
                "v3_unsafe_steps": int(float(row["hybrid_unsafe_steps"])),
                "v1_unsafe_steps": int(float(row["v1_unsafe_steps"])),
            }
            for row in sensitivity
        ],
        "marl": [
            {
                "policy": row["algorithm"].upper(),
                "daily_p10_difference_vs_v3": _number(row["observed_treatment_advantage"]),
                "ci_low": _number(row["advantage_nested_seed_day_ci_low"]),
                "ci_high": _number(row["advantage_nested_seed_day_ci_high"]),
                "one_sided_p": _number(row["fixed_seed_mean_day_sign_flip_p_value_one_sided"]),
                "eligible_days": int(row["eligible_calendar_days"]),
                "seeds": int(row["predeclared_seed_count"]),
            }
            for row in marl_audit
            if row["metric"] == "p10_service_ratio"
        ],
        "metric_guide": [
            {"metric": "Service ratio", "definition": "Delivered energy divided by requested energy for one EV; 1.0 means the requested energy was delivered.", "direction": "Higher is better."},
            {"metric": "P10 service ratio", "definition": "The lower-tail service ratio at the 10th percentile across a sufficiently large cohort of EV sessions.", "direction": "Higher is better. In a 2/5-EV teaching fixture it is illustrative only, not inferential."},
            {"metric": "Jain service index", "definition": "A 0–1 equality index over service ratios; 1.0 means equal service, not necessarily high service.", "direction": "Higher indicates more equal service."},
            {"metric": "Expected shortfall (worst 10%)", "definition": "Average service deficit within the worst tenth of sessions; the lower-tail MPC uses a convex surrogate for this objective.", "direction": "Lower is better."},
            {"metric": "95% paired day-block interval", "definition": "Bootstrap interval for a matched policy difference using complete calendar days as resampling blocks.", "direction": "If it crosses zero, the small temporal sample does not establish a reliable directional difference."},
            {"metric": "Coverage", "definition": "Fraction of held-out early-unplug events protected by the frozen predictive deadline envelope.", "direction": "Compare with its nominal target; it is calibration, not accuracy."},
        ],
        "findings": [
            {"title": "V1 baseline result", "text": "On frozen JPL November replay (29 non-empty days, 1,320 sessions), robust-PV FairFlex improves P10 over equal share by +0.0204, 95% CI [+0.0061, +0.0339], while Jain equality is lower. This is a lower-tail fairness trade-off, not universal dominance."},
            {"title": "V3 result", "text": "V3’s all-session P10 is higher than guarded V1 in the three descriptive frozen cohorts shown, but the paired calendar-day P10 intervals for Caltech October and November cross zero. Keep V1 as the default recommendation."},
            {"title": "Reliability result", "text": "On JPL November, the selected multi-rate envelope has 0.892 coverage (95% CI [0.875, 0.907]) against a 0.90 target; the fixed global guard is 0.855 (95% CI [0.831, 0.875])."},
            {"title": "Sensitivity result", "text": "V3 improves the prespecified all-session P10 in each listed October sensitivity condition, with zero unsafe replay steps, but gives up 3.45–11.30 kWh of delivered energy versus guarded V1."},
            {"title": "MARL result", "text": "Five-seed MAPPO/IPPO benchmark comparisons do not establish a P10 advantage over V3; both 95% intervals cross zero. They remain benchmarks, not deployment recommendations."},
        ],
        "limitations": [
            "Historical ACN sessions are real, but feeder placement, 10 kW shared import, and PV generation are controlled sensitivity assumptions rather than measured Caltech/JPL feeder or rooftop-PV data.",
            "The interactive Command Center is a deterministic teaching simulation and does not fetch data, train a model, or control a charger.",
            "No result establishes field deployment performance, universal dominance, or a state-of-the-art claim.",
        ],
        "artifact_ledger": [
            _artifact("artifacts/paper_comparison_jpl_november/evidence_package/primary_paired_comparisons.csv", "V1 baseline paired calendar-day effects"),
            _artifact("artifacts/paper_comparison_jpl_november/evidence_package/primary_policy_table.csv", "V1 baseline descriptive policy table"),
            _artifact("artifacts/paper_comparison_jpl_november/evidence_package/guard_reliability_table.csv", "JPL November guard coverage and buffer trade-off"),
            _artifact("artifacts/v3/holdout_october_frozen/test_fairflex_uc_metrics.csv", "Caltech October frozen V1/V3 descriptive cohort metrics"),
            _artifact("artifacts/v3/holdout_october_statistical_audit/paired_calendar_day_statistical_audit.csv", "Caltech October paired-day inference"),
            _artifact("artifacts/v3/caltech_november_temporal_replication_continuous/test_fairflex_uc_metrics.csv", "Caltech November continuous V1/V3 metrics"),
            _artifact("artifacts/v3/caltech_november_continuous_statistical_audit/paired_calendar_day_statistical_audit.csv", "Caltech November paired-day inference"),
            _artifact("artifacts/v3/jpl_external_october_frozen/test_fairflex_uc_metrics.csv", "JPL October external V1/V3 replication metrics"),
            _artifact("artifacts/v3/holdout_october_cqr_prediction_audit_v3/test_cqr_prediction_summary.json", "Caltech October CQR coverage audit"),
            _artifact("artifacts/v3/caltech_october_frozen_sensitivity/scenario_hybrid_vs_v1.csv", "Frozen one-factor V3 sensitivity results"),
            _artifact("artifacts/v3/safe_marl_final_statistical_audit_endpoint_amendment/safe_marl_seed_ensemble_statistical_audit.csv", "Final multi-seed safe-MAPPO/IPPO inference"),
        ],
        "sources": [
            {"label": "ACN-Data documentation", "url": "https://ev.caltech.edu/dataset.html"},
            {"label": "ACN-Sim paper", "url": "https://ev.caltech.edu/assets/pub/ACN_Sim_Open_Source_Simulator.pdf"},
            {"label": "CQR paper", "url": "https://papers.neurips.cc/paper_files/paper/2019/hash/5103c3584b063c431bd1268e9b5e76fb-Abstract.html"},
            {"label": "MAPPO paper", "url": "https://arxiv.org/abs/2103.01955"},
        ],
    }
    # Retain a compatibility view, then bind every displayed field to the
    # source-derived payload. A stale or manually edited bundled JSON can no
    # longer pass the build verification test merely because its CSV hashes
    # still happen to match.
    payload["pooled_p10"] = [
        {"cohort": row["cohort"], "v1": row["v1"], "v3": row["v3"]}
        for row in payload["v3_cohorts"]
    ]
    payload["source_derived_payload_sha256"] = _canonical_payload_sha256(payload)
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build or verify the frozen FairFlex evidence ledger.")
    parser.add_argument("--check", action="store_true", help="fail if the bundled ledger is not exactly source-derived")
    arguments = parser.parse_args()
    payload = build()
    if arguments.check:
        bundled = json.loads(OUTPUT.read_text(encoding="utf-8"))
        if bundled != payload:
            raise SystemExit(
                "Frozen evidence ledger does not exactly match a fresh extraction from its declared source artifacts. Run scripts/build_app_evidence.py only after reviewing the change."
            )
        print(f"Verified {OUTPUT.relative_to(ROOT)} against its frozen source artifacts.")
        raise SystemExit(0)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT.relative_to(ROOT)} with {len(payload['artifact_ledger'])} frozen artifact hashes.")
