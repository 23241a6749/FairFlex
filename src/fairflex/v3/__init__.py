"""FairFlex V3: calibrated early-unplug buffers and lower-tail MPC.

V3 deliberately lives beside the frozen V1/V2 modules.  It is an experimental
extension, not a change in the historical baseline definitions.
"""

from .commitments import CausalCQRBufferGuard, conservative_deadline_envelope
from .lower_tail_mpc import LowerTailFairMPC
from .marl_protocol import V3HybridDeadlinePreprocessor, fit_v3_hybrid_deadline_preprocessor

__all__ = [
    "CausalCQRBufferGuard",
    "LowerTailFairMPC",
    "V3HybridDeadlinePreprocessor",
    "conservative_deadline_envelope",
    "fit_v3_hybrid_deadline_preprocessor",
]
