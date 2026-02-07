from __future__ import annotations

from typing import List, Dict
import numpy as np


class ConfidenceEvaluator:
    """
    Evaluates quality and calibration of epistemic confidence.

    Confidence is expected to correlate with stability
    and anti-correlate with uncertainty and shock.
    """

    def calibration_error(
        self,
        confidences: List[float],
        outcomes: List[bool],
    ) -> float:
        """
        Expected Calibration Error (ECE).

        `outcomes` indicates whether the answer was later
        judged correct / stable / accepted.
        """
        if not confidences or not outcomes:
            return 0.0

        confidences = np.array(confidences)
        outcomes = np.array(outcomes, dtype=float)

        bins = np.linspace(0.0, 1.0, 11)
        error = 0.0

        for i in range(len(bins) - 1):
            mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
            if not np.any(mask):
                continue

            avg_conf = np.mean(confidences[mask])
            avg_out = np.mean(outcomes[mask])
            error += abs(avg_conf - avg_out) * np.sum(mask)

        return float(error / len(confidences))

    def confidence_uncertainty_correlation(
        self,
        confidences: List[float],
        entropies: List[float],
    ) -> float:
        """
        Measures whether higher uncertainty corresponds to lower confidence.
        """
        if not confidences or not entropies:
            return 0.0

        return float(np.corrcoef(confidences, entropies)[0, 1])
