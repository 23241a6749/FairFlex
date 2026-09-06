export type RunStatus = "queued" | "running" | "succeeded" | "failed";

export type Scenario = {
  scenario_id: string;
  label: string;
  subtitle: string;
  kind: string;
  estimated_runtime_seconds: number;
  source: string;
  disclosure: string;
  limitations: string[];
};

export type SessionOutcome = {
  ev_id: string;
  station_id: string;
  requested_energy_kwh: number;
  allocated_first_step_kw: number;
  service_ratio: number;
  planning_deadline_step: number;
  available_steps: number;
  slack_kwh: number;
  at_risk: boolean;
  reason: string;
};

export type Result = {
  policy: { id: string; label: string; role: string; description: string; disclosure: string };
  metrics: Record<string, number | string | null>;
  safety: { safe: boolean; unsafe_steps: number; min_voltage_pu: number | null; max_line_loading_percent: number | null; statement: string };
  steps: Array<{ step: number; allocations_kw: Record<string, number>; station_powers_kw: Record<string, number>; grid: { safe: boolean; min_voltage_pu: number | null; max_line_loading_percent: number | null; violations: string[] } }>;
  session_outcomes: SessionOutcome[];
  limitations: string[];
  execution_input?: { schema_version: number; sha256: string; verified: boolean };
  execution_manifest?: {
    manifest_sha256: string;
    manifest: {
      application_version: string;
      engine_version: string;
      python_version: string;
      executed_input_sha256: string;
      controller: { class: string; objective: string; tail_fraction?: number };
      safety_wrapper: { class: string; horizon_steps: number };
      dependency_versions: Record<string, string>;
      source_revision: string;
    };
  };
};

export type RunRecord = {
  run_id: string;
  scenario_id: string;
  status: RunStatus;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  input_hash: string;
  scenario: Scenario & { stations: Array<{ station_id: string; capacity_kw: number; bus: number }>; sessions: Array<{ ev_id: string; station_id: string; arrival_step: number; departure_step: number; planning_deadline_step: number; requested_energy_kwh: number; max_power_kw: number }>; step_hours: number; simulation_statement: string };
  result: Result | null;
  comparison: { input_hash?: string; same_input_verified: boolean; status?: string; interpretation?: string; error?: string; v3?: Result } | null;
  error: { code: string; message: string } | null;
  audit_events: Array<{ at: string; type: string; detail: Record<string, unknown> }>;
};

export type Evidence = {
  schema_version: number;
  scope: string;
  policy_decision: { default: string; shadow: string; research_v1: string; research_v3: string; conclusion: string };
  pooled_p10: Array<{ cohort: string; v1: number; v3: number }>;
  v3_cohorts: Array<{ cohort: string; site: string; dates: string; sessions: number; role: string; notes: string; v1: number; v3: number; v1_energy_kwh: number; v3_energy_kwh: number; v1_unsafe_steps: number; v3_unsafe_steps: number }>;
  calendar_day_inference: Array<{ cohort: string; eligible_days: number; v3_minus_v1_p10: number; ci_low: number; ci_high: number; one_sided_p: number; reading: string }>;
  primary_benchmark: {
    label: string;
    site: string;
    dates: string;
    sessions: number;
    assumptions: string;
    inference_unit: string;
    p10_effects: Array<{ baseline: string; effect: number; ci_low: number; ci_high: number; days: number }>;
    policy_table: Array<{ policy: string; sessions: number; p10: number; mean: number; jain: number; delivered_energy_kwh: number; unsafe_steps: number }>;
  };
  guard_reliability: Array<{ condition: string; coverage: number; ci_low: number; ci_high: number; target: number; mean_buffer_minutes: number; days: number; decisions: number }>;
  cqr_calibration: { site: string; sessions: number; covered_sessions: number; coverage: number; ci_low: number; ci_high: number; target: number; interpretation: string };
  sensitivity: Array<{ condition: string; p10_difference: number; early_unplug_p10_difference: number; shortfall_difference: number; energy_difference_kwh: number; v1_unsafe_steps: number; v3_unsafe_steps: number }>;
  marl: Array<{ policy: string; daily_p10_difference_vs_v3: number; ci_low: number; ci_high: number; one_sided_p: number; eligible_days: number; seeds: number }>;
  metric_guide: Array<{ metric: string; definition: string; direction: string }>;
  data_lineage: Array<{ layer: string; data: string; used_for: string; scope: string }>;
  findings: Array<{ title: string; text: string }>;
  limitations: string[];
  artifact_ledger: Array<{ path: string; purpose: string; sha256: string }>;
  evidence_manifest_sha256: string;
  source_derived_payload_sha256: string;
  artifact_verification: {
    status: string;
    verified: boolean;
    checked: number;
    matched: number;
    statement: string;
    derived_payload: { expected_sha256: string | null; actual_sha256: string; verified: boolean };
    source_derivation: { status: string; verified: boolean; expected_sha256: string | null; actual_sha256: string | null; detail?: string };
    entries: Array<{ path: string; purpose: string; status: string; expected_sha256: string; actual_sha256: string | null }>;
  };
  sources: Array<{ label: string; url: string }>;
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    ...init,
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload?.detail?.message ?? payload?.detail ?? "The request could not be completed.");
  }
  return response.json() as Promise<T>;
}

export const api = {
  health: () => request<{ status: string; offline_demo_ready: boolean; statement: string }>("/api/health"),
  scenarios: () => request<Scenario[]>("/api/scenarios"),
  evidence: () => request<Evidence>("/api/evidence"),
  runs: () => request<Array<Pick<RunRecord, "run_id" | "scenario_id" | "status" | "created_at" | "completed_at" | "input_hash">>>("/api/runs"),
  run: (runId: string) => request<RunRecord>(`/api/runs/${runId}`),
  startRun: (scenarioId: string, includeV3Shadow: boolean) => request<{ run_id: string; status: RunStatus }>("/api/runs", { method: "POST", body: JSON.stringify({ scenario_id: scenarioId, include_v3_shadow: includeV3Shadow }) }),
  compareV3: (runId: string) => request<{ run_id: string; status: RunStatus }>(`/api/runs/${runId}/comparisons`, { method: "POST", body: JSON.stringify({ mode: "v3_shadow" }) }),
  reportUrl: (runId: string) => `/api/runs/${runId}/report`,
  evidenceReportUrl: () => "/api/evidence/report",
};
