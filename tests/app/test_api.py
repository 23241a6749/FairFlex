"""API and engine integration tests for the offline FairFlex Demonstrator."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import subprocess
import sys
from time import monotonic, sleep

from fastapi.testclient import TestClient
import pytest

from fairflex.app.api import create_app


@pytest.fixture()
def client(tmp_path: Path):
    app = create_app(tmp_path / "audit")
    with TestClient(app) as test_client:
        yield test_client
    app.state.fairflex_service.shutdown()


def _wait_for_run(client: TestClient, run_id: str, *, expect_comparison: bool = False) -> dict:
    deadline = monotonic() + 20.0
    latest: dict = {}
    while monotonic() < deadline:
        response = client.get(f"/api/runs/{run_id}")
        assert response.status_code == 200
        latest = response.json()
        if latest["status"] == "failed":
            pytest.fail(f"demo run failed: {latest['error']}")
        comparison_terminal = latest.get("comparison", {}).get("status") in {"succeeded", "failed"} if latest.get("comparison") else False
        if latest["status"] == "succeeded" and (not expect_comparison or comparison_terminal):
            return latest
        sleep(0.1)
    pytest.fail(f"run did not complete: {latest}")


def test_health_and_catalogue_are_offline_and_credential_free(client: TestClient) -> None:
    health = client.get("/api/health")
    assert health.status_code == 200
    body = health.json()
    assert body["offline_demo_ready"] is True
    assert body["external_credentials_required"] is False
    assert "token" not in str(body).lower()

    scenarios = client.get("/api/scenarios")
    assert scenarios.status_code == 200
    assert {scenario["scenario_id"] for scenario in scenarios.json()} == {"deadline-stress", "campus-flow"}

    evidence = client.get("/api/evidence")
    assert evidence.status_code == 200
    evidence_body = evidence.json()
    assert "Frozen FairFlex historical ACN evaluation" in evidence_body["scope"]
    assert len(evidence_body["evidence_manifest_sha256"]) == 64
    assert evidence_body["artifact_verification"]["verified"] is True
    assert evidence_body["artifact_verification"]["matched"] == evidence_body["artifact_verification"]["checked"]
    assert evidence_body["artifact_verification"]["derived_payload"]["verified"] is True
    assert evidence_body["artifact_verification"]["source_derivation"]["verified"] is True

    dashboard = client.get("/")
    assert dashboard.status_code == 200
    assert "fairflex" in dashboard.text.lower()
    assert dashboard.headers["x-content-type-options"] == "nosniff"
    assert "frame-ancestors 'none'" in dashboard.headers["content-security-policy"]


def test_v1_run_is_real_persisted_and_safe(client: TestClient) -> None:
    created = client.post("/api/runs", json={"scenario_id": "deadline-stress"})
    assert created.status_code == 202
    record = _wait_for_run(client, created.json()["run_id"])

    assert record["result"]["policy"]["role"] == "recommended"
    assert record["result"]["safety"]["safe"] is True
    assert record["result"]["safety"]["unsafe_steps"] == 0
    urgent = next(session for session in record["result"]["session_outcomes"] if session["ev_id"] == "urgent")
    assert urgent["allocated_first_step_kw"] > 7.19
    assert urgent["at_risk"] is True
    assert len(record["input_hash"]) == 64
    assert record["result"]["execution_manifest"]["manifest"]["executed_input_sha256"] == record["input_hash"]
    assert record["result"]["execution_input"]["sha256"] == record["input_hash"]
    assert len(record["result"]["execution_manifest"]["manifest_sha256"]) == 64
    assert record["audit_events"][-1]["type"] == "completed"


def test_v3_shadow_uses_saved_identical_input_and_never_changes_v1_role(client: TestClient) -> None:
    created = client.post("/api/runs", json={"scenario_id": "deadline-stress", "include_v3_shadow": True})
    assert created.status_code == 202
    record = _wait_for_run(client, created.json()["run_id"], expect_comparison=True)

    comparison = record["comparison"]
    assert comparison["same_input_verified"] is True
    assert comparison["input_hash"] == record["input_hash"]
    assert comparison["v3"]["policy"]["role"] == "research_shadow"
    assert comparison["v3"]["execution_manifest"]["manifest"]["executed_input_sha256"] == record["input_hash"]
    assert record["result"]["policy"]["role"] == "recommended"


def test_v3_failure_is_audited_without_overwriting_safe_v1(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from fairflex.app import service as service_module

    original_run_policy = service_module._run_policy

    def fail_only_v3(*args, **kwargs):
        if args[2] == "v3_shadow":
            raise RuntimeError("injected V3 failure")
        return original_run_policy(*args, **kwargs)

    monkeypatch.setattr(service_module, "_run_policy", fail_only_v3)
    created = client.post("/api/runs", json={"scenario_id": "deadline-stress"})
    record = _wait_for_run(client, created.json()["run_id"])
    response = client.post(f"/api/runs/{record['run_id']}/comparisons", json={"mode": "v3_shadow"})
    assert response.status_code == 202
    failed_comparison = _wait_for_run(client, record["run_id"], expect_comparison=True)

    assert failed_comparison["status"] == "succeeded"
    assert failed_comparison["result"]["safety"]["safe"] is True
    assert failed_comparison["comparison"]["status"] == "failed"
    assert failed_comparison["comparison"]["input_hash"] == failed_comparison["input_hash"]
    assert failed_comparison["audit_events"][-1]["type"] == "v3_shadow_failed"


def test_v3_shadow_is_claimed_once_under_duplicate_requests(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Repeated browser clicks cannot submit duplicate shadow jobs."""

    from fairflex.app import service as service_module

    original_run_policy = service_module._run_policy

    def slow_v3(*args, **kwargs):
        if args[2] == "v3_shadow":
            sleep(0.25)
        return original_run_policy(*args, **kwargs)

    monkeypatch.setattr(service_module, "_run_policy", slow_v3)
    created = client.post("/api/runs", json={"scenario_id": "deadline-stress"})
    record = _wait_for_run(client, created.json()["run_id"])

    first = client.post(f"/api/runs/{record['run_id']}/comparisons", json={"mode": "v3_shadow"})
    second = client.post(f"/api/runs/{record['run_id']}/comparisons", json={"mode": "v3_shadow"})
    assert first.status_code == 202
    assert second.status_code == 409

    completed = _wait_for_run(client, record["run_id"], expect_comparison=True)
    assert completed["comparison"]["status"] == "succeeded"
    assert sum(event["type"] == "v3_shadow_queued" for event in completed["audit_events"]) == 1


def test_validation_not_found_and_report_export(client: TestClient) -> None:
    assert client.post("/api/runs", json={"scenario_id": "../../.env"}).status_code == 404
    assert client.post("/api/runs", json={"scenario_id": "deadline-stress", "path": "D:/"}).status_code == 422
    assert client.get("/api/runs/not-a-real-run").status_code == 404

    created = client.post("/api/runs", json={"scenario_id": "deadline-stress"})
    run_id = created.json()["run_id"]
    _wait_for_run(client, run_id)
    report = client.get(f"/api/runs/{run_id}/report")
    assert report.status_code == 200
    assert "Controlled teaching scenario" in report.text
    assert "Input SHA-256" in report.text
    assert "Execution-manifest SHA-256" in report.text
    assert "Local-audit limitation" in report.text

    evidence_report = client.get("/api/evidence/report")
    assert evidence_report.status_code == 200
    assert "FairFlex frozen evidence ledger" in evidence_report.text
    assert "Frozen source artifact ledger" in evidence_report.text


def test_persisted_snapshot_not_factory_defines_the_executed_input(client: TestClient) -> None:
    """An audited run remains tied to its saved complete input specification."""

    from fairflex.app import service as service_module

    service = client.app.state.fairflex_service
    original = service.catalog.get("deadline-stress")
    snapshot = service.catalog.snapshot("deadline-stress")

    def poisoned_factory():
        raise AssertionError("a saved run must not call the current scenario factory")

    result = service_module._run_policy(replace(original, factory=poisoned_factory), snapshot, "v1_default")
    expected_hash = service_module._hash_payload(service.catalog.execution_input(snapshot))

    assert result["execution_input"]["verified"] is True
    assert result["execution_input"]["sha256"] == expected_hash
    assert result["metrics"]["worst_service_ratio"] == pytest.approx(1.0)


def test_bundled_evidence_is_a_fresh_source_extraction() -> None:
    """Every browser number must come from the declared frozen artifacts."""

    project_root = Path(__file__).resolve().parents[2]
    check = subprocess.run(
        [sys.executable, "scripts/build_app_evidence.py", "--check"],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert check.returncode == 0, check.stderr or check.stdout
    assert "Verified src" in check.stdout
