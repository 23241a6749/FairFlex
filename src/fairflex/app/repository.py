"""Small SQLite audit store for the offline demonstrator."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class RunNotFoundError(KeyError):
    """Raised when an API route requests an unknown run."""


class AuditRepository:
    """Persist runs and append-only events without storing credentials."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    scenario_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    input_hash TEXT NOT NULL,
                    scenario_json TEXT NOT NULL,
                    result_json TEXT,
                    comparison_json TEXT,
                    error_code TEXT,
                    error_message TEXT
                );
                CREATE TABLE IF NOT EXISTS audit_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    detail_json TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES runs(run_id)
                );
                CREATE INDEX IF NOT EXISTS audit_events_run_id_idx ON audit_events(run_id, event_id);
                """
            )

    def reconcile_interrupted(self, timestamp: str) -> None:
        """Fail incomplete in-memory jobs after a local process restart.

        This demonstrator intentionally has no external job queue.  Persisted
        records must therefore never keep presenting a cancelled worker as
        queued/running after the process that owned it has exited.
        """

        with self._connect() as connection:
            interrupted_runs = connection.execute(
                "SELECT run_id FROM runs WHERE status IN ('queued', 'running')"
            ).fetchall()
            for row in interrupted_runs:
                connection.execute(
                    """
                    UPDATE runs
                    SET status = 'failed', completed_at = ?, error_code = ?, error_message = ?
                    WHERE run_id = ?
                    """,
                    (timestamp, "interrupted_by_restart", "Local demonstrator restarted before this in-memory job finished.", row["run_id"]),
                )
                self._append_event(connection, row["run_id"], timestamp, "interrupted_by_restart", {})

            succeeded_rows = connection.execute(
                "SELECT run_id, comparison_json FROM runs WHERE status = 'succeeded' AND comparison_json IS NOT NULL"
            ).fetchall()
            for row in succeeded_rows:
                comparison = self._loads(row["comparison_json"])
                if comparison.get("status") not in {"queued", "running"}:
                    continue
                comparison.update(
                    {
                        "status": "failed",
                        "error": "V3 shadow was interrupted by a local restart; V1 remains unchanged.",
                        "diagnostic_code": "interrupted_by_restart",
                    }
                )
                connection.execute(
                    "UPDATE runs SET comparison_json = ? WHERE run_id = ?",
                    (self._dumps(comparison), row["run_id"]),
                )
                self._append_event(connection, row["run_id"], timestamp, "v3_shadow_interrupted", {})

    @staticmethod
    def _loads(value: str | None) -> Any:
        return json.loads(value) if value else None

    @staticmethod
    def _dumps(value: Any) -> str:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    def create_run(
        self,
        *,
        run_id: str,
        scenario_id: str,
        created_at: str,
        input_hash: str,
        scenario: dict[str, Any],
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO runs (run_id, scenario_id, status, created_at, input_hash, scenario_json)
                VALUES (?, ?, 'queued', ?, ?, ?)
                """,
                (run_id, scenario_id, created_at, input_hash, self._dumps(scenario)),
            )
            self._append_event(connection, run_id, created_at, "created", {"scenario_id": scenario_id})

    def _append_event(
        self,
        connection: sqlite3.Connection,
        run_id: str,
        created_at: str,
        event_type: str,
        detail: dict[str, Any],
    ) -> None:
        connection.execute(
            "INSERT INTO audit_events (run_id, created_at, event_type, detail_json) VALUES (?, ?, ?, ?)",
            (run_id, created_at, event_type, self._dumps(detail)),
        )

    def mark_running(self, run_id: str, timestamp: str) -> None:
        with self._connect() as connection:
            connection.execute("UPDATE runs SET status = 'running', started_at = ? WHERE run_id = ?", (timestamp, run_id))
            self._append_event(connection, run_id, timestamp, "started", {})

    def complete(self, run_id: str, timestamp: str, result: dict[str, Any]) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE runs SET status = 'succeeded', completed_at = ?, result_json = ? WHERE run_id = ?",
                (timestamp, self._dumps(result), run_id),
            )
            self._append_event(connection, run_id, timestamp, "completed", {"unsafe_steps": result["safety"]["unsafe_steps"]})

    def attach_comparison(self, run_id: str, timestamp: str, comparison: dict[str, Any]) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE runs SET comparison_json = ? WHERE run_id = ?",
                (self._dumps(comparison), run_id),
            )
            self._append_event(
                connection,
                run_id,
                timestamp,
                "v3_shadow_failed" if comparison.get("status") == "failed" else "v3_shadow_completed",
                {"input_hash": comparison.get("input_hash"), "status": comparison.get("status", "succeeded")},
            )

    def claim_comparison(self, run_id: str, timestamp: str) -> tuple[bool, str]:
        """Atomically reserve the one allowed V3 shadow attempt for a run."""

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT status, input_hash, result_json, comparison_json FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is None:
                raise RunNotFoundError(run_id)
            if row["status"] != "succeeded" or row["result_json"] is None:
                return False, "V3 shadow comparison requires a completed V1 run"
            existing = self._loads(row["comparison_json"])
            if existing is not None:
                existing_status = existing.get("status", "succeeded")
                return False, f"V3 shadow is already {existing_status}; this demonstrator records one protected attempt per V1 run"
            comparison = {
                "status": "queued",
                "input_hash": row["input_hash"],
                "same_input_verified": False,
            }
            connection.execute(
                "UPDATE runs SET comparison_json = ? WHERE run_id = ?",
                (self._dumps(comparison), run_id),
            )
            self._append_event(connection, run_id, timestamp, "v3_shadow_queued", {"input_hash": row["input_hash"]})
            return True, "queued"

    def mark_comparison_running(self, run_id: str, timestamp: str) -> bool:
        """Transition a claimed V3 comparison to running exactly once."""

        with self._connect() as connection:
            row = connection.execute("SELECT comparison_json FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            if row is None:
                raise RunNotFoundError(run_id)
            comparison = self._loads(row["comparison_json"])
            if not comparison or comparison.get("status") != "queued":
                return False
            comparison["status"] = "running"
            connection.execute(
                "UPDATE runs SET comparison_json = ? WHERE run_id = ?",
                (self._dumps(comparison), run_id),
            )
            self._append_event(connection, run_id, timestamp, "v3_shadow_started", {"input_hash": comparison.get("input_hash")})
            return True

    def fail(self, run_id: str, timestamp: str, code: str, message: str) -> None:
        safe_message = message[:500]
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE runs
                SET status = 'failed', completed_at = ?, error_code = ?, error_message = ?
                WHERE run_id = ?
                """,
                (timestamp, code, safe_message, run_id),
            )
            self._append_event(connection, run_id, timestamp, "failed", {"code": code})

    def get(self, run_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            if row is None:
                raise RunNotFoundError(run_id)
            events = connection.execute(
                "SELECT created_at, event_type, detail_json FROM audit_events WHERE run_id = ? ORDER BY event_id",
                (run_id,),
            ).fetchall()
        return {
            "run_id": row["run_id"],
            "scenario_id": row["scenario_id"],
            "status": row["status"],
            "created_at": row["created_at"],
            "started_at": row["started_at"],
            "completed_at": row["completed_at"],
            "input_hash": row["input_hash"],
            "scenario": self._loads(row["scenario_json"]),
            "result": self._loads(row["result_json"]),
            "comparison": self._loads(row["comparison_json"]),
            "error": (
                {"code": row["error_code"], "message": row["error_message"]}
                if row["error_code"]
                else None
            ),
            "audit_events": [
                {
                    "at": event["created_at"],
                    "type": event["event_type"],
                    "detail": self._loads(event["detail_json"]),
                }
                for event in events
            ],
        }

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT run_id, scenario_id, status, created_at, completed_at, input_hash, error_code
                FROM runs ORDER BY created_at DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]
