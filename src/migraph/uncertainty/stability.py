from __future__ import annotations

from typing import List, Dict, Any
import numpy as np


class StabilityAnalyzer:
    """
    Computes explanation stability across graph variants and agents.

    Stability measures how invariant reasoning outcomes are under
    structural perturbations.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        variant_results: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Compute stability metrics from variant-level results.

        Each item is expected to contain:
        - score (float)
        - variant (str)
        """
        scores = [r["score"] for r in variant_results if "score" in r]

        if not scores:
            return {
                "stability": 0.0,
                "variance": 0.0,
                "robustness": 0.0,
            }

        mean = float(np.mean(scores))
        std = float(np.std(scores))

        return {
            "stability": max(mean - std, 0.0),
            "variance": std,
            "robustness": self._robustness_ratio(scores),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _robustness_ratio(self, scores: List[float]) -> float:
        """
        Fraction of variants supporting the dominant explanation.
        """
        if not scores:
            return 0.0

        dominant = max(scores)
        agreeing = sum(1 for s in scores if abs(s - dominant) < 0.1)

        return agreeing / len(scores)
