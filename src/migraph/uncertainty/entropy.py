from __future__ import annotations

from typing import List
import math
import numpy as np


class EntropyAnalyzer:
    """
    Computes epistemic entropy of reasoning outcomes.

    High entropy = fragmented explanations.
    Low entropy = convergent reasoning.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self, scores: List[float]) -> float:
        """
        Compute normalized entropy from a list of scores.
        """
        if not scores:
            return 0.0

        probs = self._normalize(scores)
        return self._entropy(probs)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _normalize(self, scores: List[float]) -> List[float]:
        total = sum(scores)
        if total == 0.0:
            return [1.0 / len(scores)] * len(scores)

        return [s / total for s in scores]

    def _entropy(self, probs: List[float]) -> float:
        entropy = 0.0
        for p in probs:
            if p > 0.0:
                entropy -= p * math.log(p)

        # Normalize to [0, 1]
        max_entropy = math.log(len(probs)) if len(probs) > 1 else 1.0
        return entropy / max_entropy
