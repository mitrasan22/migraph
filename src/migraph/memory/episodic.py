from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List
from datetime import datetime
from uuid import uuid4
import numpy as np


@dataclass(frozen=True)
class Episode:
    """
    Immutable epistemic episode.

    Captures what the system believed, how stable it was,
    and under what conditions the belief was formed.
    """

    id: str
    timestamp: datetime

    query: str
    entities: List[str]

    answer: str
    confidence: float
    uncertainty: Dict[str, Any]

    shock: Dict[str, Any]
    graph_variants: List[str]
    dominant_agents: List[str]
    bias: str

    embedding: np.ndarray
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "query": self.query,
            "entities": list(self.entities),
            "answer": self.answer,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "shock": self.shock,
            "graph_variants": list(self.graph_variants),
            "dominant_agents": list(self.dominant_agents),
            "bias": self.bias,
        }

    @staticmethod
    def create(
        *,
        query: str,
        entities: List[str],
        answer: str,
        confidence: float,
        uncertainty: Dict[str, Any],
        shock: Dict[str, Any],
        graph_variants: List[str],
        dominant_agents: List[str],
        bias: str,
        embedding: np.ndarray,
    ) -> "Episode":
        return Episode(
            id=str(uuid4()),
            timestamp=datetime.utcnow(),
            query=query,
            entities=entities,
            answer=answer,
            confidence=confidence,
            uncertainty=uncertainty,
            shock=shock,
            graph_variants=graph_variants,
            dominant_agents=dominant_agents,
            bias=bias,
            embedding=embedding,
        )


class EpisodicMemory:
    """
    Append-only episodic memory store.

    Stores epistemic outcomes, not raw interaction history.
    """

    def __init__(self) -> None:
        self._episodes: List[Episode] = []

    def add(self, episode: Episode) -> None:
        self._episodes.append(episode)

    def all(self) -> List[Episode]:
        return list(self._episodes)

    def filter(
        self,
        *,
        min_confidence: float | None = None,
        max_shock: float | None = None,
        bias: str | None = None,
    ) -> List[Episode]:

        results = self._episodes

        if min_confidence is not None:
            results = [e for e in results if e.confidence >= min_confidence]

        if max_shock is not None:
            results = [
                e for e in results
                if e.shock.get("overall", 0.0) <= max_shock
            ]

        if bias is not None:
            results = [e for e in results if e.bias == bias]

        return results
