from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class ShockScore:
    """
    Represents the novelty / instability signal for a query.

    Shock is multi-dimensional, not a single heuristic.
    """

    # Overall score used for control decisions
    overall: float

    # Component signals (for auditability & UI)
    novelty: float
    structural_risk: float
    temporal_risk: float
    evidence_fragility: float

    # Optional diagnostics
    details: Dict[str, float]

    def is_high(self, threshold: float) -> bool:
        return self.overall >= threshold
