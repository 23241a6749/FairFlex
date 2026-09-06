import { useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  ArrowUpRight,
  BarChart3,
  BatteryCharging,
  CheckCircle2,
  ChevronRight,
  CircleAlert,
  Clock3,
  Database,
  FileDown,
  GitCompareArrows,
  Menu,
  Pause,
  Play,
  RefreshCw,
  ScrollText,
  ShieldCheck,
  Sparkles,
  X,
  Zap,
} from "lucide-react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api, type Evidence, type Result, type RunRecord, type Scenario, type SessionOutcome } from "./api";

type View = "command" | "scenarios" | "compare" | "evidence" | "audit";

const navigation: Array<{ id: View; label: string; icon: typeof Activity }> = [
  { id: "command", label: "Command center", icon: Activity },
  { id: "scenarios", label: "Scenario lab", icon: Sparkles },
  { id: "compare", label: "Compare lab", icon: GitCompareArrows },
  { id: "evidence", label: "Evidence vault", icon: BarChart3 },
  { id: "audit", label: "Audit trail", icon: ScrollText },
];

const format = (value: number | null | undefined, digits = 2) => (value == null ? "—" : value.toFixed(digits));
const ratio = (value: number | null | undefined) => `${format(value == null ? null : value * 100, 0)}%`;
const titleCase = (value: string) => value.replace(/_/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase());

function useMediaQuery(query: string) {
  const [matches, setMatches] = useState(() => typeof window !== "undefined" && window.matchMedia(query).matches);

  useEffect(() => {
    const media = window.matchMedia(query);
    const update = () => setMatches(media.matches);
    update();
    media.addEventListener("change", update);
    return () => media.removeEventListener("change", update);
  }, [query]);

  return matches;
}

function StatusMark({ safe, label }: { safe: boolean; label?: string }) {
  return (
    <span className={`status-mark ${safe ? "safe" : "warning"}`}>
      {safe ? <CheckCircle2 size={15} aria-hidden="true" /> : <CircleAlert size={15} aria-hidden="true" />}
      {label ?? (safe ? "Safe" : "Attention")}
    </span>
  );
}

function Metric({ label, value, detail, emphasis = "cyan" }: { label: string; value: string; detail: string; emphasis?: "cyan" | "violet" | "amber" | "green" }) {
  return (
    <article className={`metric metric-${emphasis}`}>
      <p>{label}</p>
      <strong>{value}</strong>
      <span>{detail}</span>
    </article>
  );
}

function RunButton({ loading, onClick, includeV3 }: { loading: boolean; onClick: () => void; includeV3: boolean }) {
  return (
    <button className="run-button" type="button" onClick={onClick} disabled={loading}>
      {loading ? <RefreshCw size={20} className="spin" aria-hidden="true" /> : <Play size={20} fill="currentColor" aria-hidden="true" />}
      <span>{loading ? "Running the real FairFlex engine…" : "Run V1 recommendation"}</span>
      {includeV3 && !loading ? <small>then V3 shadow</small> : null}
    </button>
  );
}

function ChartTooltip({ active, payload, label }: { active?: boolean; payload?: Array<{ name: string; value: number; color: string }>; label?: string | number }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="chart-tooltip">
      <strong>Step {label}</strong>
      {payload.map((item) => <span key={item.name} style={{ color: item.color }}>{item.name}: {format(item.value)} kW</span>)}
    </div>
  );
}

function CapacityChart({ run, motionReduced }: { run: RunRecord; motionReduced: boolean }) {
  const result = run.result!;
  const capacity = run.scenario.stations.reduce((sum, station) => sum + station.capacity_kw, 0);
  const chartData = result.steps.map((step) => ({
    step: step.step,
    power: Object.values(step.station_powers_kw).reduce((sum, value) => sum + value, 0),
    capacity,
  }));
  return (
    <section className="surface chart-surface" aria-labelledby="capacity-title">
      <div className="surface-heading">
        <div>
          <p className="eyebrow">Decision trace</p>
          <h2 id="capacity-title">Power stays inside the available envelope</h2>
        </div>
        <StatusMark safe={result.safety.safe} label="AC checked" />
      </div>
      <div className="chart-wrap" aria-label="Time series of site power and configured station capacity">
        <ResponsiveContainer width="100%" height={270}>
          <AreaChart data={chartData} margin={{ top: 16, right: 12, left: -18, bottom: 0 }}>
            <defs>
              <linearGradient id="powerFill" x1="0" x2="0" y1="0" y2="1">
                <stop offset="0%" stopColor="#36d8dd" stopOpacity={0.52} />
                <stop offset="100%" stopColor="#36d8dd" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid vertical={false} stroke="#264058" strokeDasharray="3 7" />
            <XAxis dataKey="step" stroke="#8fa8bc" tickLine={false} axisLine={false} tickFormatter={(step) => `T${step}`} />
            <YAxis stroke="#8fa8bc" tickLine={false} axisLine={false} tickFormatter={(value) => `${value} kW`} />
            <Tooltip content={<ChartTooltip />} cursor={{ stroke: "#7a6ef6", strokeWidth: 1 }} />
            <Area type="monotone" dataKey="capacity" stroke="#e7b85c" strokeDasharray="5 5" fill="transparent" name="Configured capacity" isAnimationActive={!motionReduced} />
            <Area type="monotone" dataKey="power" stroke="#36d8dd" strokeWidth={3} fill="url(#powerFill)" name="Executed power" isAnimationActive={!motionReduced} />
          </AreaChart>
        </ResponsiveContainer>
      </div>
      <p className="chart-caption">The cyan trace shows the real engine’s executed station power. The amber line is the configured fixture capacity. Safety comes from AC validation, not from this chart alone.</p>
      <details className="chart-data-table">
        <summary>View capacity-trace data table</summary>
        <div className="table-scroll"><table><thead><tr><th>Step</th><th>Executed power</th><th>Fixture capacity</th></tr></thead><tbody>{chartData.map((row) => <tr key={row.step}><td>T{row.step}</td><td>{format(row.power)} kW</td><td>{format(row.capacity)} kW</td></tr>)}</tbody></table></div>
      </details>
    </section>
  );
}

function SessionTable({ sessions, onSelect }: { sessions: SessionOutcome[]; onSelect: (session: SessionOutcome) => void }) {
  return (
    <section className="surface allocation-surface" aria-labelledby="allocation-title">
      <div className="surface-heading">
        <div>
          <p className="eyebrow">Explainable allocation</p>
          <h2 id="allocation-title">Who receives the next kilowatt, and why?</h2>
        </div>
        <span className="subtle-label">Click an EV for its decision brief</span>
      </div>
      <div className="table-scroll">
        <table>
          <thead><tr><th>EV</th><th>First allocation</th><th>Service</th><th>Deadline</th><th>Decision</th></tr></thead>
          <tbody>
            {sessions.map((session) => (
              <tr key={session.ev_id} className={session.at_risk ? "at-risk-row" : ""}>
                <td><span className="ev-cell"><span className={`ev-dot ${session.at_risk ? "risk" : ""}`} />{session.ev_id}</span></td>
                <td>{format(session.allocated_first_step_kw)} kW</td>
                <td><span className="service-value">{ratio(session.service_ratio)}</span></td>
                <td>T{session.planning_deadline_step} <span className="muted">({session.available_steps} slot{session.available_steps === 1 ? "" : "s"})</span></td>
                <td><button className="table-action" onClick={() => onSelect(session)} type="button" aria-label={`Open decision brief for ${session.ev_id}`}>Why <ChevronRight size={15} /></button></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function DecisionDrawer({ session, onClose }: { session: SessionOutcome; onClose: () => void }) {
  const dialogRef = useRef<HTMLElement>(null);
  const closeRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    const previouslyFocused = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    closeRef.current?.focus();
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        onClose();
        return;
      }
      if (event.key !== "Tab") return;
      const focusable = dialogRef.current?.querySelectorAll<HTMLElement>("button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex='-1'])");
      if (!focusable?.length) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
      if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
    };
    document.addEventListener("keydown", onKeyDown);
    return () => { document.removeEventListener("keydown", onKeyDown); previouslyFocused?.focus(); };
  }, [onClose]);

  return (
    <div className="drawer-backdrop" role="presentation" onMouseDown={onClose}>
      <aside ref={dialogRef} className="decision-drawer" role="dialog" aria-modal="true" aria-labelledby="drawer-title" onMouseDown={(event) => event.stopPropagation()}>
        <button ref={closeRef} className="icon-button close-button" type="button" onClick={onClose} aria-label="Close decision brief"><X size={19} /></button>
        <p className="eyebrow">Decision brief</p>
        <h2 id="drawer-title">{session.ev_id}</h2>
        <StatusMark safe={!session.at_risk} label={session.at_risk ? "Priority review" : "Flexible timing"} />
        <div className="drawer-stats">
          <div><span>First allocation</span><strong>{format(session.allocated_first_step_kw)} kW</strong></div>
          <div><span>Requested energy</span><strong>{format(session.requested_energy_kwh)} kWh</strong></div>
          <div><span>Final service</span><strong>{ratio(session.service_ratio)}</strong></div>
          <div><span>Timing slack</span><strong>{format(session.slack_kwh)} kWh</strong></div>
        </div>
        <div className="drawer-reason"><Zap size={18} aria-hidden="true" /><p>{session.reason}</p></div>
        <p className="drawer-footnote">This explanation describes a controlled simulated decision. It never sends a command to a charger.</p>
      </aside>
    </div>
  );
}

function PresenterPath() {
  const steps = [
    ["01", "Scarcity", "Start the controlled 2-EV case."],
    ["02", "V1 decision", "Show who receives power and why."],
    ["03", "AC check", "Show executed safety validation."],
    ["04", "Same-input V3", "Optional research shadow only."],
    ["05", "Historical evidence", "Use CIs, trade-offs, and negative results."],
    ["06", "Audit", "Export the local reproducibility record."],
  ];
  return (
    <section className="presenter-path" aria-labelledby="presenter-path-title">
      <div className="presenter-heading"><div><p className="eyebrow">Presenter path</p><h2 id="presenter-path-title">A defensible six-step live demonstration</h2></div><span>Teaching simulation → frozen research evidence → local audit</span></div>
      <ol>{steps.map(([number, title, detail]) => <li key={number}><b>{number}</b><strong>{title}</strong><span>{detail}</span></li>)}</ol>
    </section>
  );
}

function EmptyCommand({ scenario, loading, onRun, includeV3, setIncludeV3 }: { scenario?: Scenario; loading: boolean; onRun: () => void; includeV3: boolean; setIncludeV3: (value: boolean) => void }) {
  return (
    <>
      <section className="command-hero" aria-labelledby="command-title">
        <div className="energy-field" aria-hidden="true"><i /><i /><i /></div>
        <div className="hero-copy">
          <p className="eyebrow">Prepared offline demonstration</p>
          <h1 id="command-title">Make constrained charging decisions people can understand.</h1>
          <p>FairFlex uses the actual fairness optimizer and AC feasibility check. Start a deterministic scenario, inspect the allocation, then compare V3 without letting it replace V1.</p>
          <div className="hero-actions">
            <RunButton loading={loading} onClick={onRun} includeV3={includeV3} />
            <label className="shadow-toggle"><input type="checkbox" checked={includeV3} onChange={(event) => setIncludeV3(event.target.checked)} /><span>Run V3 in research shadow mode</span></label>
          </div>
        </div>
        <div className="scenario-orbit">
          <div className="orbit-glow" aria-hidden="true" />
          <div className="orbit-core"><BatteryCharging size={32} aria-hidden="true" /><strong>{scenario?.label ?? "Loading scenario"}</strong><span>{scenario?.subtitle}</span></div>
          <div className="orbit-chip chip-one">V1 default</div>
          <div className="orbit-chip chip-two">AC check on run</div>
          <div className="orbit-chip chip-three">Offline</div>
        </div>
      </section>
      <PresenterPath />
    </>
  );
}

function CommandCenter({ run, scenario, loading, onRun, includeV3, setIncludeV3, onSelectSession, onCompare, motionReduced }: { run: RunRecord | null; scenario?: Scenario; loading: boolean; onRun: () => void; includeV3: boolean; setIncludeV3: (value: boolean) => void; onSelectSession: (session: SessionOutcome) => void; onCompare: () => void; motionReduced: boolean }) {
  if (!run?.result) return <EmptyCommand scenario={scenario} loading={loading} onRun={onRun} includeV3={includeV3} setIncludeV3={setIncludeV3} />;
  const { result } = run;
  const metrics = result.metrics;
  const atRisk = result.session_outcomes.filter((session) => session.at_risk).length;
  return (
    <>
      <section className="run-summary">
        <div className="summary-title"><p className="eyebrow">V1 recommendation completed</p><h1>{run.scenario.label}</h1><p>{run.scenario.disclosure}</p></div>
        <div className="summary-controls"><StatusMark safe={result.safety.safe} label={result.safety.safe ? "Safe execution" : "Unsafe result"} /><button type="button" className="ghost-button" onClick={onCompare} disabled={loading || Boolean(run.comparison)}><GitCompareArrows size={16} />{run.comparison ? "V3 comparison available" : "Run V3 shadow"}</button></div>
      </section>
      <section className="metric-grid" aria-label="V1 scheduling metrics">
        <Metric label="Worst EV service" value={ratio(metrics.worst_service_ratio as number)} detail="Most directly useful teaching-fixture outcome" emphasis="violet" />
        <Metric label="Mean service" value={ratio(metrics.mean_service_ratio as number)} detail="Average requested energy delivered" />
        <Metric label="At-risk EVs" value={String(atRisk)} detail="Tight deadline or low final service" emphasis="amber" />
        <Metric label="Delivered energy" value={`${format(metrics.delivered_energy_kwh as number)} kWh`} detail={`${format(metrics.runtime_seconds as number, 2)} s real solver runtime`} emphasis="green" />
      </section>
      <p className="live-metric-disclosure">Illustrative P10: {ratio(metrics.p10_service_ratio as number)} across n={result.session_outcomes.length} EVs. P10 is a lower-tail research metric only when evaluated over a sufficiently large cohort; use the per-EV outcomes here to understand this teaching fixture.</p>
      <div className="content-grid">
        <CapacityChart run={run} motionReduced={motionReduced} />
        <aside className="decision-brief surface">
          <p className="eyebrow">Decision brief</p>
          <h2>Why V1 made this choice</h2>
          <p>V1 first protects the highest normalized energy shortfall, then maximizes delivered energy, then applies a small economic tie-break. The AC wrapper validates the executed decision.</p>
          <div className="brief-divider" />
          <div className="brief-row"><ShieldCheck size={18} /><span>{result.safety.statement}</span></div>
          <div className="brief-row"><Clock3 size={18} /><span>{atRisk ? `${atRisk} EV needs an explicit timing explanation.` : "No EV was marked at risk in this outcome."}</span></div>
          <div className="brief-row"><Database size={18} /><span>Run ID and input hash are stored in the audit trail.</span></div>
          <a className="report-link" href={api.reportUrl(run.run_id)}><FileDown size={16} />Download audit report</a>
        </aside>
      </div>
      <SessionTable sessions={result.session_outcomes} onSelect={onSelectSession} />
    </>
  );
}

function ScenarioLab({ scenarios, selectedId, onSelect, onRun }: { scenarios: Scenario[]; selectedId: string; onSelect: (id: string) => void; onRun: () => void }) {
  return (
    <section className="page-section scenario-page">
      <div className="page-heading"><p className="eyebrow">Approved inputs only</p><h1>Scenario lab</h1><p>Every scenario is bundled locally. The browser cannot provide a file path, secret, or solver configuration.</p></div>
      <div className="scenario-list">
        {scenarios.map((scenario) => <article className={`scenario-card ${scenario.scenario_id === selectedId ? "selected" : ""}`} key={scenario.scenario_id}>
          <div className="scenario-card-top"><span className="scenario-kind">{titleCase(scenario.kind)}</span>{scenario.scenario_id === selectedId ? <StatusMark safe label="Selected" /> : null}</div>
          <h2>{scenario.label}</h2><p>{scenario.subtitle}</p>
          <dl><div><dt>Runtime budget</dt><dd>≈ {scenario.estimated_runtime_seconds}s</dd></div><div><dt>Source</dt><dd>{scenario.source}</dd></div></dl>
          <p className="disclosure">{scenario.disclosure}</p>
          <div className="scenario-actions"><button className="outline-button" type="button" onClick={() => onSelect(scenario.scenario_id)}>{scenario.scenario_id === selectedId ? "Selected" : "Select scenario"}</button>{scenario.scenario_id === selectedId ? <button className="text-button" type="button" onClick={onRun}>Run V1 <ArrowUpRight size={15} /></button> : null}</div>
        </article>)}
      </div>
    </section>
  );
}

function CompareLab({ run, onCompare, comparisonLoading }: { run: RunRecord | null; onCompare: () => void; comparisonLoading: boolean }) {
  const v1 = run?.result;
  const v3 = run?.comparison?.v3;
  const comparisonStatus = run?.comparison?.status;
  if (!v1) return <section className="empty-page"><GitCompareArrows size={38} /><h1>Compare the same scenario, never two different histories.</h1><p>Run V1 first. FairFlex will preserve the input hash and only then create a V3 research shadow result.</p></section>;
  const rows: Array<[string, (result: Result) => string, string]> = [
    ["Illustrative P10", (result) => ratio(result.metrics.p10_service_ratio as number), `Descriptive only; this fixture has n=${v1.session_outcomes.length} EVs`],
    ["Worst EV service", (result) => ratio(result.metrics.worst_service_ratio as number), "Small-fixture primary outcome: the least-served EV"],
    ["Mean service", (result) => ratio(result.metrics.mean_service_ratio as number), "Average service ratio"],
    ["Jain fairness", (result) => format(result.metrics.jain_service_index as number, 3), "1.000 means equal service"],
    ["Delivered energy", (result) => `${format(result.metrics.delivered_energy_kwh as number)} kWh`, "Total delivered during this replay"],
    ["Solver runtime", (result) => `${format(result.metrics.runtime_seconds as number, 3)} s`, "Wall-clock scheduling and AC-repair runtime"],
    ["Safety", (result) => result.safety.safe ? "PASS" : "FAIL", "Executed feasibility check"],
  ];
  const comparisonBody = !v3 ? (
    comparisonStatus === "failed" ? <div className="truth-card"><CircleAlert size={20} /><p><strong>V3 shadow did not complete.</strong> {run?.comparison?.error ?? "V1 remains the recorded safe recommendation."} The local demonstrator preserves one protected shadow attempt instead of silently retrying a changed execution.</p></div> : comparisonLoading ? <div className="shadow-start"><div><span className="policy-label v1">V1 preserved</span><span className="policy-label v3">V3 {comparisonStatus ?? "queued"}</span><h2>Running the same immutable input in research shadow mode.</h2><p>V1’s result remains safe and unchanged. The audit trail records the V3 lifecycle separately.</p></div><RefreshCw className="spin" size={28} aria-label="V3 shadow is running" /></div> : <div className="shadow-start"><div><span className="policy-label v1">V1 recommended</span><span className="policy-label v3">V3 research shadow</span><h2>Run the shadow comparison on this exact input?</h2><p>FairFlex records the same immutable input hash before it will show a comparison. V3 cannot replace V1’s decision.</p></div><button className="run-button compact" type="button" onClick={onCompare}><GitCompareArrows size={18} />Run V3 shadow</button></div>
  ) : <>
    <div className="comparison-assurance"><StatusMark safe={Boolean(run.comparison?.same_input_verified)} label={run.comparison?.same_input_verified ? "Identical input verified" : "Comparison unavailable"} /><span>{run.comparison?.interpretation ?? run.comparison?.error}</span></div>
    <div className="comparison-table-wrap"><table className="comparison-table"><thead><tr><th>Metric</th><th><span className="policy-label v1">V1 recommended</span></th><th><span className="policy-label v3">V3 research shadow</span></th></tr></thead><tbody>{rows.map(([label, read, explanation]) => <tr key={label}><td><strong>{label}</strong><span>{explanation}</span></td><td>{read(v1)}</td><td>{read(v3)}</td></tr>)}</tbody></table></div>
    <div className="truth-card"><CircleAlert size={20} /><p><strong>Interpretation discipline:</strong> a single teaching scenario demonstrates mechanics, not superiority. The Evidence Vault contains the frozen calendar-day statistical tests that keep V1 as the default.</p></div>
  </>;
  return (
    <section className="page-section compare-page">
      <div className="page-heading"><p className="eyebrow">Policy governance</p><h1>Compare lab</h1><p>V1 controls the demonstrated recommendation. V3 receives the same saved input as a research shadow.</p></div>
      {comparisonBody}
    </section>
  );
}

function ForestPlot({ effects }: { effects: Evidence["primary_benchmark"]["p10_effects"] }) {
  const low = Math.min(-0.01, ...effects.map((effect) => effect.ci_low - 0.005));
  const high = Math.max(0.01, ...effects.map((effect) => effect.ci_high + 0.005));
  const position = (value: number) => `${((value - low) / (high - low)) * 100}%`;
  return (
    <figure className="forest-figure" aria-labelledby="baseline-forest-title">
      <figcaption id="baseline-forest-title">Paired calendar-day P10 effect: robust-PV FairFlex minus each baseline. A confidence interval wholly right of zero supports a positive lower-tail difference in this study. The six displayed intervals are individual contrasts, not a multiplicity-adjusted familywise analysis.</figcaption>
      <div className="forest-axis" aria-hidden="true"><span style={{ left: position(low) }}>{format(low, 2)}</span><span style={{ left: position(0) }}>0</span><span style={{ left: position(high) }}>{format(high, 2)}</span></div>
      <div className="forest-list">
        {effects.map((effect) => <div className="forest-row" key={effect.baseline}>
          <strong>{effect.baseline}</strong>
          <div className="forest-track" role="img" aria-label={`${effect.baseline}: effect ${format(effect.effect, 4)}, 95 percent confidence interval ${format(effect.ci_low, 4)} to ${format(effect.ci_high, 4)}`}>
            <i className="forest-zero" style={{ left: position(0) }} />
            <span className="forest-interval" style={{ left: position(effect.ci_low), width: `calc(${position(effect.ci_high)} - ${position(effect.ci_low)})` }} />
            <b className="forest-point" style={{ left: position(effect.effect) }} />
          </div>
          <span>{effect.effect >= 0 ? "+" : ""}{format(effect.effect, 4)}<small>95% CI [{format(effect.ci_low, 4)}, {format(effect.ci_high, 4)}]</small></span>
        </div>)}
      </div>
      <details className="chart-data-table"><summary>View baseline effect data table</summary><div className="table-scroll"><table><thead><tr><th>Baseline</th><th>FairFlex − baseline P10</th><th>95% CI</th><th>Matched days</th></tr></thead><tbody>{effects.map((effect) => <tr key={effect.baseline}><td>{effect.baseline}</td><td>{effect.effect >= 0 ? "+" : ""}{format(effect.effect, 4)}</td><td>[{format(effect.ci_low, 4)}, {format(effect.ci_high, 4)}]</td><td>{effect.days}</td></tr>)}</tbody></table></div></details>
    </figure>
  );
}

function EvidenceVault({ evidence, motionReduced }: { evidence: Evidence | null; motionReduced: boolean }) {
  if (!evidence) return <section className="empty-page"><RefreshCw className="spin" size={32} /><p>Loading frozen evidence…</p></section>;
  const p10Data = evidence.pooled_p10;
  const verification = evidence.artifact_verification;
  return (
    <section className="page-section evidence-page">
      <div className="page-heading"><p className="eyebrow">Frozen research evidence · schema {evidence.schema_version}</p><h1>Evidence vault</h1><p>{evidence.scope}</p><a className="report-link" href={api.evidenceReportUrl()}><FileDown size={16} />Download evidence ledger</a></div>
      <div className="evidence-decision"><div><span className="policy-label v1">Live default · {evidence.policy_decision.default}</span><span className="policy-label v3">Live shadow · {evidence.policy_decision.shadow}</span><span className="policy-label v1">Research V1 · {evidence.policy_decision.research_v1}</span><span className="policy-label v3">Research V3 · {evidence.policy_decision.research_v3}</span></div><p>{evidence.policy_decision.conclusion}</p></div>
      <section className={`surface integrity-card ${verification.verified ? "verified" : "attention"}`}><div><p className="eyebrow">Evidence integrity</p><h2>{verification.matched} / {verification.checked} frozen source artifacts + derived values verified</h2><p>{verification.statement}</p></div><div className="integrity-hash"><span>Manifest SHA-256</span><code>{evidence.evidence_manifest_sha256}</code><span>Source-derived payload SHA-256</span><code>{evidence.source_derived_payload_sha256}</code><StatusMark safe={verification.verified} label={verification.verified ? "Locally hash-verified" : "Review required"} /></div></section>
      <section className="surface lineage-surface"><div className="surface-heading"><div><p className="eyebrow">Data & method boundary</p><h2>Three data layers — never silently mixed</h2></div><span className="subtle-label">What the browser uses today</span></div><div className="lineage-grid">{evidence.data_lineage.map((item) => <article key={item.layer}><span>{item.layer}</span><strong>{item.data}</strong><p>{item.used_for}</p><small>{item.scope}</small></article>)}</div></section>
      <section className="surface primary-research-surface"><div className="surface-heading"><div><p className="eyebrow">Primary frozen baseline result</p><h2>{evidence.primary_benchmark.label}</h2><p className="surface-subtitle">{evidence.primary_benchmark.site} · {evidence.primary_benchmark.dates} · {evidence.primary_benchmark.sessions.toLocaleString()} sessions</p></div><StatusMark safe label="All policies: 0 unsafe steps" /></div><p className="research-assumptions">{evidence.primary_benchmark.assumptions}</p><ForestPlot effects={evidence.primary_benchmark.p10_effects} /><p className="chart-caption">Inference unit: {evidence.primary_benchmark.inference_unit} The graph reports a lower-tail fairness effect, not universal superiority.</p><details className="chart-data-table"><summary>View complete primary policy table</summary><div className="table-scroll"><table><thead><tr><th>Policy</th><th>P10</th><th>Mean</th><th>Jain</th><th>Energy</th><th>Unsafe steps</th></tr></thead><tbody>{evidence.primary_benchmark.policy_table.map((row) => <tr key={row.policy}><td>{row.policy}</td><td>{format(row.p10, 4)}</td><td>{format(row.mean, 4)}</td><td>{format(row.jain, 4)}</td><td>{format(row.delivered_energy_kwh, 1)} kWh</td><td>{row.unsafe_steps}</td></tr>)}</tbody></table></div></details></section>
      <div className="evidence-layout">
        <section className="surface p10-chart"><div className="surface-heading"><div><p className="eyebrow">Secondary descriptive context</p><h2>All-session P10 by frozen cohort</h2></div><span className="subtle-label">0–1 · higher is better</span></div><ResponsiveContainer width="100%" height={285}><BarChart data={p10Data} margin={{ top: 12, right: 12, left: -16, bottom: 42 }}><CartesianGrid vertical={false} stroke="#264058" strokeDasharray="3 7" /><XAxis dataKey="cohort" angle={-18} textAnchor="end" height={70} tick={{ fill: "#8fa8bc", fontSize: 11 }} axisLine={false} tickLine={false} /><YAxis domain={[0, 1]} tickFormatter={(value) => value.toFixed(2)} tick={{ fill: "#8fa8bc" }} axisLine={false} tickLine={false} /><Tooltip formatter={(value) => typeof value === "number" ? value.toFixed(4) : "—"} /><Legend wrapperStyle={{ fontSize: 12, paddingTop: 8 }} /><Bar dataKey="v1" name="Frozen V1 comparator" fill="#36d8dd" radius={[7, 7, 0, 0]} isAnimationActive={!motionReduced} /><Bar dataKey="v3" name="Frozen V3" fill="#9b82ff" radius={[7, 7, 0, 0]} isAnimationActive={!motionReduced} /></BarChart></ResponsiveContainer><p className="chart-caption">Each bar is an all-session P10 for one stated cohort, not an aggregate meta-analysis.</p><details className="chart-data-table"><summary>View frozen-cohort data table</summary><div className="table-scroll"><table><thead><tr><th>Cohort</th><th>Site / window</th><th>Sessions</th><th>V1</th><th>V3</th></tr></thead><tbody>{evidence.v3_cohorts.map((row) => <tr key={row.cohort}><td>{row.cohort}</td><td>{row.site} · {row.dates}</td><td>{row.sessions}</td><td>{format(row.v1, 4)}</td><td>{format(row.v3, 4)}</td></tr>)}</tbody></table></div></details></section>
        <section className="surface inference-surface"><p className="eyebrow">V3 decision check</p><h2>Paired calendar-day P10 audit</h2>{evidence.calendar_day_inference.map((row) => <article key={row.cohort} className="inference-row"><div><strong>{row.cohort}</strong><span>{row.eligible_days} eligible days</span></div><div><b className={row.v3_minus_v1_p10 >= 0 ? "positive" : "negative"}>{row.v3_minus_v1_p10 >= 0 ? "+" : ""}{format(row.v3_minus_v1_p10, 4)}</b><span>Frozen V3 − V1</span></div><div><span>95% CI [{format(row.ci_low, 4)}, {format(row.ci_high, 4)}] · one-sided p={format(row.one_sided_p, 3)}</span><em>{row.reading}</em></div></article>)}<div className="truth-card compact-truth"><CircleAlert size={19} /><p>Both intervals cross zero. The ethical product decision is to keep V1 as the live recommendation and treat V3 as a research extension.</p></div></section>
      </div>
      <div className="evidence-layout reliability-layout">
        <section className="surface reliability-surface"><div className="surface-heading"><div><p className="eyebrow">Early-unplug guard reliability</p><h2>Coverage versus 90% target</h2></div><span className="subtle-label">JPL November</span></div><div className="coverage-list">{evidence.guard_reliability.map((row) => <article key={row.condition}><div><strong>{row.condition}</strong><span>{format(row.coverage * 100, 1)}% observed coverage</span></div><div className="coverage-track" role="img" aria-label={`${row.condition}: ${format(row.coverage * 100, 1)} percent observed coverage; target ${format(row.target * 100, 0)} percent`}><i className="coverage-target" style={{ left: `${row.target * 100}%` }} /><b style={{ width: `${row.coverage * 100}%` }} /></div><small>95% CI [{format(row.ci_low * 100, 1)}%, {format(row.ci_high * 100, 1)}%] · mean buffer {format(row.mean_buffer_minutes, 1)} min · {row.decisions} decisions</small></article>)}</div></section>
        <section className="surface cqr-surface"><p className="eyebrow">Uncertainty calibration</p><h2>CQR is measured by coverage, not “accuracy”</h2><div className="cqr-score"><strong>{format(evidence.cqr_calibration.coverage * 100, 1)}%</strong><span>observed coverage</span></div><p>{evidence.cqr_calibration.covered_sessions} / {evidence.cqr_calibration.sessions} held-out sessions · target {format(evidence.cqr_calibration.target * 100, 0)}% · 95% CI [{format(evidence.cqr_calibration.ci_low * 100, 1)}%, {format(evidence.cqr_calibration.ci_high * 100, 1)}%]</p><small>{evidence.cqr_calibration.interpretation}</small></section>
      </div>
      <section className="surface sensitivity-surface"><div className="surface-heading"><div><p className="eyebrow">Frozen V3 one-factor sensitivity</p><h2>Fairness improvement has an energy trade-off</h2></div><span className="subtle-label">Descriptive point estimates · V3 − guarded V1</span></div><div className="table-scroll"><table><thead><tr><th>Condition</th><th>P10 difference</th><th>Early-unplug P10</th><th>Worst-tail shortfall</th><th>Energy difference</th><th>Safety</th></tr></thead><tbody>{evidence.sensitivity.map((row) => <tr key={row.condition}><td>{row.condition}</td><td className="positive">+{format(row.p10_difference, 4)}</td><td className={row.early_unplug_p10_difference >= 0 ? "positive" : "negative"}>{row.early_unplug_p10_difference >= 0 ? "+" : ""}{format(row.early_unplug_p10_difference, 4)}</td><td>{format(row.shortfall_difference, 4)}</td><td className="negative">{format(row.energy_difference_kwh, 2)} kWh</td><td>{row.v1_unsafe_steps === 0 && row.v3_unsafe_steps === 0 ? "0 / 0 unsafe" : "Review"}</td></tr>)}</tbody></table></div><p className="chart-caption">These pre-existing one-factor sensitivity estimates do not include separate confidence intervals. They show an explicit design trade-off: V3’s listed conditions improve all-session P10 here but use less delivered energy.</p></section>
      <section className="surface marl-surface"><div className="surface-heading"><div><p className="eyebrow">Benchmark honesty</p><h2>Safe MARL results</h2></div><span className="subtle-label">Safe station-cap benchmark, not deployment controller</span></div><div className="marl-grid">{evidence.marl.map((row) => <div key={row.policy}><strong>{row.policy}</strong><span>Daily P10 difference vs frozen V3</span><b>{format(row.daily_p10_difference_vs_v3, 4)}</b><small>95% CI [{format(row.ci_low, 4)}, {format(row.ci_high, 4)}] · p={format(row.one_sided_p, 3)} · {row.seeds} seeds · {row.eligible_days} matched days</small></div>)}</div></section>
      <div className="methods-grid"><section className="surface metric-guide"><p className="eyebrow">Metrics & methods</p><h2>Read each number correctly</h2>{evidence.metric_guide.map((metric) => <article key={metric.metric}><strong>{metric.metric}</strong><p>{metric.definition}</p><small>{metric.direction}</small></article>)}</section><section className="surface findings-guide"><p className="eyebrow">Findings & limits</p><h2>What the evidence supports</h2>{evidence.findings.map((finding) => <article key={finding.title}><strong>{finding.title}</strong><p>{finding.text}</p></article>)}<div className="limitations-list">{evidence.limitations.map((limitation) => <p key={limitation}><CircleAlert size={14} />{limitation}</p>)}</div></section></div>
      <details className="artifact-ledger"><summary>Inspect every verified frozen source artifact</summary><div className="table-scroll"><table><thead><tr><th>Status</th><th>Purpose</th><th>Path</th><th>SHA-256</th></tr></thead><tbody>{verification.entries.map((entry) => <tr key={entry.path}><td><StatusMark safe={entry.status === "verified"} label={titleCase(entry.status)} /></td><td>{entry.purpose}</td><td><code>{entry.path}</code></td><td><code>{entry.expected_sha256}</code></td></tr>)}</tbody></table></div></details>
      <p className="source-line">Method sources (open in a new tab): {evidence.sources.map((source) => <a key={source.url} href={source.url} target="_blank" rel="noreferrer">{source.label} ↗</a>)}</p>
    </section>
  );
}

function AuditTrail({ history, selectedRun, onSelect }: { history: Array<Pick<RunRecord, "run_id" | "scenario_id" | "status" | "created_at" | "completed_at" | "input_hash">>; selectedRun: RunRecord | null; onSelect: (runId: string) => void }) {
  const manifest = selectedRun?.result?.execution_manifest?.manifest;
  const result = selectedRun?.result;
  return (
    <section className="page-section audit-page"><div className="page-heading"><p className="eyebrow">Reproducible decision history</p><h1>Audit trail</h1><p>Each run stores the exact executed-input SHA-256, controller manifest, feasibility result, and generated report. It never stores API credentials.</p></div>
      <div className="audit-layout"><section className="surface history-list"><h2>Saved runs</h2>{history.length ? history.map((run) => <button key={run.run_id} className={`audit-row ${selectedRun?.run_id === run.run_id ? "active" : ""}`} type="button" onClick={() => onSelect(run.run_id)}><span><strong>{run.scenario_id}</strong><small>{new Date(run.created_at).toLocaleString()}</small></span><span><StatusMark safe={run.status === "succeeded"} label={titleCase(run.status)} /><small>{run.input_hash.slice(0, 12)}…</small></span></button>) : <p className="empty-history">No saved run yet. Run the command-center scenario to create an audit record.</p>}</section>
        <section className="surface run-provenance"><p className="eyebrow">Selected record</p>{selectedRun ? <><h2>{selectedRun.scenario.label}</h2><dl><div><dt>Run ID</dt><dd>{selectedRun.run_id}</dd></div><div><dt>Executed input</dt><dd>{selectedRun.input_hash}</dd></div><div><dt>Status</dt><dd><StatusMark safe={selectedRun.status === "succeeded"} label={titleCase(selectedRun.status)} /></dd></div>{result?.execution_manifest ? <><div><dt>Manifest hash</dt><dd>{result.execution_manifest.manifest_sha256}</dd></div><div><dt>Controller</dt><dd>{manifest?.controller.class} — {manifest?.controller.objective}</dd></div><div><dt>Safety wrapper</dt><dd>{manifest?.safety_wrapper.class} · {manifest?.safety_wrapper.horizon_steps}-step horizon</dd></div><div><dt>Engine / source</dt><dd>{manifest?.engine_version} · {manifest?.source_revision}</dd></div></> : <div><dt>Manifest</dt><dd>Legacy record: no complete execution manifest was stored.</dd></div>}<div><dt>Events</dt><dd>{selectedRun.audit_events.map((event) => <span key={`${event.at}-${event.type}`} className="audit-event">{event.type} · {new Date(event.at).toLocaleTimeString()}</span>)}</dd></div></dl><p className="local-audit-note"><CircleAlert size={14} />Local SQLite record: useful for reproducibility, but not a signed or remotely immutable ledger.</p><a className="report-link" href={api.reportUrl(selectedRun.run_id)}><FileDown size={16} />Download audit report</a></> : <p>Select a saved local audit record to inspect its recorded provenance.</p>}</section></div>
    </section>
  );
}

function App() {
  const [view, setView] = useState<View>("command");
  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const [evidence, setEvidence] = useState<Evidence | null>(null);
  const [history, setHistory] = useState<Array<Pick<RunRecord, "run_id" | "scenario_id" | "status" | "created_at" | "completed_at" | "input_hash">>>([]);
  const [selectedScenarioId, setSelectedScenarioId] = useState("deadline-stress");
  const [run, setRun] = useState<RunRecord | null>(null);
  const [includeV3, setIncludeV3] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [drawerSession, setDrawerSession] = useState<SessionOutcome | null>(null);
  const [menuOpen, setMenuOpen] = useState(false);
  const [motionPaused, setMotionPaused] = useState(false);
  const menuButtonRef = useRef<HTMLButtonElement>(null);
  const sidebarRef = useRef<HTMLElement>(null);
  const compactNavigation = useMediaQuery("(max-width: 760px)");
  const prefersReducedMotion = useMediaQuery("(prefers-reduced-motion: reduce)");
  const motionReduced = motionPaused || prefersReducedMotion;

  const loading = run?.status === "queued" || run?.status === "running";
  const comparisonLoading = run?.comparison?.status === "queued" || run?.comparison?.status === "running";
  const liveStatus = loading
    ? `FairFlex is ${run?.status === "queued" ? "queued" : "running"}${includeV3 ? "; V3 shadow will follow V1." : "."}`
    : comparisonLoading
      ? "V1 is complete. The V3 research shadow is running on the same saved input."
    : run?.result
      ? `V1 recommendation completed for ${run.scenario.label}.`
      : "FairFlex is ready for an offline teaching simulation.";
  const selectedScenario = useMemo(() => scenarios.find((scenario) => scenario.scenario_id === selectedScenarioId), [scenarios, selectedScenarioId]);

  const refreshHistory = () => api.runs().then(setHistory).catch(() => undefined);

  function focusMainContent() {
    window.setTimeout(() => document.getElementById("main-content")?.focus(), 0);
  }

  function changeView(nextView: View) {
    setView(nextView);
    setMenuOpen(false);
    focusMainContent();
  }

  function closeMobileNavigation() {
    setMenuOpen(false);
    window.setTimeout(() => menuButtonRef.current?.focus(), 0);
  }

  useEffect(() => {
    Promise.all([api.scenarios(), api.evidence(), api.health()])
      .then(([nextScenarios, nextEvidence]) => { setScenarios(nextScenarios); setEvidence(nextEvidence); if (!nextScenarios.some((scenario) => scenario.scenario_id === selectedScenarioId)) setSelectedScenarioId(nextScenarios[0]?.scenario_id ?? ""); })
      .catch((reason: Error) => setError(`FairFlex could not reach the local API. ${reason.message}`));
    refreshHistory();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!compactNavigation) {
      setMenuOpen(false);
      return;
    }
    if (!menuOpen) return;
    const focusableSelector = "button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex='-1'])";
    const focusable = () => Array.from(sidebarRef.current?.querySelectorAll<HTMLElement>(focusableSelector) ?? []);
    window.setTimeout(() => focusable()[0]?.focus(), 0);
    const keepFocusInNavigation = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        closeMobileNavigation();
        return;
      }
      if (event.key !== "Tab") return;
      const items = focusable();
      if (!items.length) return;
      const first = items[0];
      const last = items[items.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    document.addEventListener("keydown", keepFocusInNavigation);
    return () => document.removeEventListener("keydown", keepFocusInNavigation);
  }, [compactNavigation, menuOpen]);

  useEffect(() => {
    document.querySelectorAll<HTMLElement>(".table-scroll").forEach((element) => {
      element.tabIndex = 0;
      if (!element.hasAttribute("aria-label")) element.setAttribute("aria-label", "Scrollable data table");
    });
  }, [view, run, evidence]);

  useEffect(() => {
    if (!run || (!loading && !comparisonLoading)) return;
    const interval = window.setInterval(() => {
      api.run(run.run_id).then((record) => { setRun(record); refreshHistory(); }).catch((reason: Error) => setError(reason.message));
    }, 700);
    return () => window.clearInterval(interval);
  }, [run?.run_id, loading, comparisonLoading]);

  async function runV1() {
    setError(null);
    setDrawerSession(null);
    try {
      const accepted = await api.startRun(selectedScenarioId, includeV3);
      const initial = await api.run(accepted.run_id);
      setRun(initial);
      setView("command");
      focusMainContent();
      refreshHistory();
    } catch (reason) { setError(reason instanceof Error ? reason.message : "The scenario could not start."); }
  }

  async function runV3Shadow() {
    if (!run) return;
    setError(null);
    try {
      await api.compareV3(run.run_id);
      setRun(await api.run(run.run_id));
      setView("compare");
      focusMainContent();
    } catch (reason) { setError(reason instanceof Error ? reason.message : "V3 comparison could not start."); }
  }

  async function chooseHistoricRun(runId: string) {
    try { setRun(await api.run(runId)); setView("audit"); focusMainContent(); } catch (reason) { setError(reason instanceof Error ? reason.message : "Could not load this audit record."); }
  }

  return (
    <main className={`app-shell ${motionPaused ? "motion-paused" : ""}`}>
      <a className="skip-link" href="#main-content">Skip to main content</a>
      <aside ref={sidebarRef} className={`sidebar ${menuOpen ? "open" : ""}`} aria-label="FairFlex navigation" inert={compactNavigation && !menuOpen ? true : undefined} aria-hidden={compactNavigation && !menuOpen ? true : undefined}>
        <div className="brand"><div className="brand-mark"><Zap size={19} fill="currentColor" aria-hidden="true" /></div><span>fair<span>flex</span></span></div>
        <nav id="primary-navigation">{navigation.map((item) => { const Icon = item.icon; return <button type="button" key={item.id} className={view === item.id ? "active" : ""} aria-current={view === item.id ? "page" : undefined} onClick={() => changeView(item.id)}><Icon size={18} aria-hidden="true" />{item.label}</button>; })}</nav>
        <div className="sidebar-foot"><StatusMark safe label="Offline-ready" /><p>Demonstrator 1.0<br /><span>Local only</span></p></div>
      </aside>
      {compactNavigation && menuOpen ? <button className="mobile-nav-backdrop" type="button" tabIndex={-1} onClick={closeMobileNavigation} aria-label="Close navigation" /> : null}
      <section className="main-stage" id="main-content" tabIndex={-1}>
        <header className="topbar"><button ref={menuButtonRef} className="icon-button mobile-menu" type="button" onClick={() => setMenuOpen((open) => !open)} aria-label="Toggle navigation" aria-controls="primary-navigation" aria-expanded={menuOpen}><Menu size={20} /></button><div className="topbar-context"><span className="pulse-dot" aria-hidden="true" />Offline controlled teaching simulation <span className="topbar-separator">/</span> no physical charger control</div><div className="topbar-actions"><button className="motion-toggle" type="button" onClick={() => setMotionPaused((paused) => !paused)} aria-pressed={!motionPaused} aria-label={motionPaused ? "Resume ambient visual effects" : "Pause ambient visual effects"} title={motionPaused ? "Resume ambient visual effects" : "Pause ambient visual effects"}>{motionPaused ? <Play size={14} fill="currentColor" aria-hidden="true" /> : <Pause size={14} fill="currentColor" aria-hidden="true" />}<span>{motionPaused ? "Effects paused" : "Ambient effects"}</span></button><button className="topbar-run" type="button" onClick={runV1} disabled={loading || !selectedScenario}><Play size={14} fill="currentColor" />Run scenario</button><span className="live-clock">Local mode</span></div></header>
        {error ? <div className="error-banner" role="alert"><CircleAlert size={18} /><span>{error}</span><button type="button" onClick={() => setError(null)} aria-label="Dismiss message"><X size={17} /></button></div> : null}
        <div className="sr-only" aria-live="polite" aria-atomic="true">{liveStatus}</div>
        <div className="content-stage">
          {view === "command" ? <CommandCenter run={run} scenario={selectedScenario} loading={loading} onRun={runV1} includeV3={includeV3} setIncludeV3={setIncludeV3} onSelectSession={setDrawerSession} onCompare={runV3Shadow} motionReduced={motionReduced} /> : null}
          {view === "scenarios" ? <ScenarioLab scenarios={scenarios} selectedId={selectedScenarioId} onSelect={setSelectedScenarioId} onRun={runV1} /> : null}
          {view === "compare" ? <CompareLab run={run} onCompare={runV3Shadow} comparisonLoading={comparisonLoading} /> : null}
          {view === "evidence" ? <EvidenceVault evidence={evidence} motionReduced={motionReduced} /> : null}
          {view === "audit" ? <AuditTrail history={history} selectedRun={run} onSelect={chooseHistoricRun} /> : null}
        </div>
      </section>
      {drawerSession ? <DecisionDrawer session={drawerSession} onClose={() => setDrawerSession(null)} /> : null}
    </main>
  );
}

export default App;
