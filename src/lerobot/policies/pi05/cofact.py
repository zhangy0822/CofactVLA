"""Compatibility imports for the original CofactVLA tensor helper names.

New code should import apply_opg from opg and apply_ccr from ccr.
"""

from lerobot.policies.pi05.ccr import apply_ccr
from lerobot.policies.pi05.opg import apply_opg

orthogonal_guidance = apply_opg
reduce_counterfactual_covariance = apply_ccr

__all__ = ["apply_opg", "apply_ccr", "orthogonal_guidance", "reduce_counterfactual_covariance"]
