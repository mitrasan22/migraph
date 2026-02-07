"""
Evaluation subsystem for migraph.

Provides metrics to evaluate:
- epistemic confidence calibration
- uncertainty quality
- robustness under graph perturbation
- agent and variant disagreement
"""

from migraph.evaluation.confidence import ConfidenceEvaluator
from migraph.evaluation.metrics import EvaluationMetrics

__all__ = [
    "ConfidenceEvaluator",
    "EvaluationMetrics",
]
