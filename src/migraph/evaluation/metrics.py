from __future__ import annotations

from typing import List, Dict, Any
import numpy as np


class EvaluationMetrics:
    """
    Computes system-level evaluation metrics for migraph.

    These metrics are designed for research analysis,
    not leaderboard optimization.
    """

    def agent_disagreement(self, agent_results: List[Dict[str, Any]]) -> float:
        """
        Measures disagreement across agents.
        """
        scores = [r["score"] for r in agent_results if "score" in r]
        if not scores:
            return 0.0
        return float(np.std(scores))

    def variant_disagreement(self, variant_scores: List[float]) -> float:
        """
        Measures disagreement across graph variants.
        """
        if not variant_scores:
            return 0.0
        return float(np.std(variant_scores))

    def robustness_score(
        self,
        baseline_score: float,
        variant_scores: List[float],
    ) -> float:
        """
        Measures how much performance degrades under perturbation.
        """
        if not variant_scores:
            return baseline_score

        worst_case = min(variant_scores)
        return max(baseline_score - worst_case, 0.0)

    def shock_sensitivity(
        self,
        shocks: List[float],
        confidences: List[float],
    ) -> float:
        """
        Measures how confidence reacts to increasing shock.
        """
        if not shocks or not confidences:
            return 0.0

        return float(np.corrcoef(shocks, confidences)[0, 1])

    def uncertainty_alignment(
        self,
        entropies: List[float],
        disagreements: List[float],
    ) -> float:
        """
        Measures whether uncertainty tracks disagreement.
        """
        if not entropies or not disagreements:
            return 0.0

        return float(np.corrcoef(entropies, disagreements)[0, 1])
