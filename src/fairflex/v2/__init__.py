"""Isolated FairFlex V2 research components.

V2 deliberately imports the stable V1 domain objects but never changes V1
policies, protocol files, artifacts, or output locations.  A V2 result is a
separate experiment and must be reported separately from the frozen V1 study.
"""

from .commitments import (
    CONDITION_GROUPS,
    ConditionAwareEarlyDepartureGuard,
    declared_condition_group,
)
from .comparison import DEVELOPMENT_POLICY_METRICS, development_metric_deltas
from .protocol import ConditionAwareGuardProtocol, load_condition_aware_guard_protocol
from .runtime import ProfiledACRepairedMPCPolicy, V2DecisionRuntime
from .safety import ACRepairedBaselinePolicyV2
from .selection import apply_june_contextual_guard_rule, rank_guard_candidates

__all__ = [
    "CONDITION_GROUPS",
    "DEVELOPMENT_POLICY_METRICS",
    "ConditionAwareEarlyDepartureGuard",
    "declared_condition_group",
    "development_metric_deltas",
    "ConditionAwareGuardProtocol",
    "load_condition_aware_guard_protocol",
    "ProfiledACRepairedMPCPolicy",
    "V2DecisionRuntime",
    "ACRepairedBaselinePolicyV2",
    "apply_june_contextual_guard_rule",
    "rank_guard_candidates",
]
