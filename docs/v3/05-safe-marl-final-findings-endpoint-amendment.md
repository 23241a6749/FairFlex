# Safe-MARL final findings: endpoint-completed amendment

## Status

The matched Safe-MAPPO and Safe-IPPO benchmark is complete. It compares five
frozen MAPPO seeds and five frozen IPPO seeds with the deterministic
`fairflex_uc_hybrid_lower_tail` (V3) controller on the same 388 valid Caltech
ACN sessions. All policies use the same causal CQR plus multi-rate deadline
guard, station mapping, PV proxy, synthetic feeder limit, lower-tail local
allocator and final AC safety layer.

The original protocol remains unchanged. One session connected before the
frozen 1 January boundary but physically disconnected at 02:30 UTC. The
2019-only solar input therefore ended ten 15-minute steps too early. The
endpoint amendment adds only the same official NSRDB source for 2020 at the
same location and resolution; it does not alter the EV window, models,
checkpoints, controller, metric or seed set. See `04-safe-marl-endpoint-amendment.md`.

## Point-score summary

| Result | V3 | MAPPO seed mean | IPPO seed mean |
|---|---:|---:|---:|
| Pooled all-session P10 service ratio | 0.1545 | 0.1612 | 0.1563 |
| Mean service ratio | 0.4916 | 0.4186 | 0.4081 |
| Delivered energy (kWh) | 3403.1 | 2916.1 | 2798.4 |
| Executed unsafe steps | 0 | 0 for every seed | 0 for every seed |

The pooled P10 point score alone is **not** the registered primary inference.
P10 is a nonlinear quantile, and pooling all sessions gives high-volume days
more influence than averaging one P10 value per calendar day. The predeclared
primary endpoint therefore uses matched arrival-day P10 with at least ten
sessions per day.

## Registered primary result: calendar-day P10

Thirteen matched days met the P10 support rule. The advantage below is
`MARL - V3` after orienting higher P10 as better. The confidence interval is a
crossed bootstrap that resamples both the five frozen training seeds and the
matched days. The sign-flip test is exact because there are only 13 eligible
days (8,192 label swaps); it conditions on the fixed set of five predeclared
seeds.

| Architecture | Mean daily P10 advantage | Nested 95% CI | P(MARL better) | One-sided sign-flip p |
|---|---:|---:|---:|---:|
| MAPPO | -0.0481 | [-0.1367, 0.0209] | 0.1122 | 0.8080 |
| IPPO | -0.0595 | [-0.1548, 0.0168] | 0.0822 | 0.8379 |

Neither confidence interval excludes zero and neither directional test
supports a MARL P10 advantage. This final cohort therefore **does not support
the claim that MAPPO or IPPO improves lower-tail service over V3**.

## Secondary observations

The day-level seed-ensemble analysis also finds that both learned station-cap
policies have lower mean service, lower delivered energy and larger service
deficits than V3 in this scenario. For MAPPO, the mean daily-service advantage
is -0.1143 (nested 95% CI [-0.1556, -0.0768]) and the delivered-energy
advantage is -22.14 kWh/day ([-31.76, -13.96]). IPPO is weaker on the same
measures. Jain fairness differences are inconclusive for both architectures.

There are only two days with at least ten early-unplug sessions, below the
registered minimum of five. Early-unplug P10 is therefore withheld rather than
reported as a misleading significance result.

All executed runs have zero unsafe steps, zero unsafe final actions and zero
AC-repair activations. Thus the comparison isolates service and energy
performance rather than a safety failure.

## Honest interpretation and next action

This is a useful negative architecture result: under a common uncertainty and
safety layer, the deterministic lower-tail MPC remains the stronger primary
method for this registered temporal cohort. MAPPO's training-only centralized
critic does not rescue the learned station-cap layer here; IPPO is also not a
competitive replacement.

Do **not** retune the actor, reward, guard or seeds against this final test.
Any new learned hybrid must be proposed as a new development experiment, then
trained and evaluated on another untouched temporal or site-level holdout.
The project should retain V3 as its primary contribution and present the
MAPPO/IPPO benchmark as a rigorous, transparent architecture ablation.

## Evidence files

- `artifacts/v3/safe_marl_final_mappo_endpoint_amendment/test_matched_safe_marl_metrics.csv`
- `artifacts/v3/safe_marl_final_ippo_endpoint_amendment/test_matched_safe_marl_metrics.csv`
- `artifacts/v3/safe_marl_final_statistical_audit_endpoint_amendment/safe_marl_seed_ensemble_statistical_audit.csv`
- `artifacts/v3/safe_marl_final_statistical_audit_endpoint_amendment/safe_marl_per_seed_statistical_audit.csv`
- `artifacts/v3/safe_marl_final_statistical_audit_endpoint_amendment/safe_marl_daywise_metrics.csv`
