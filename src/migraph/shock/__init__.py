"""
Shock detection subsystem for migraph.

This module is responsible for detecting:
- novelty
- causal instability
- distributional shift
- reasoning risk

Shock signals control how aggressively the graph is mutated
during inference.
"""

from migraph.shock.shock_detector import ShockDetector
from migraph.shock.shock_score import ShockScore

__all__ = [
    "ShockDetector",
    "ShockScore",
]
