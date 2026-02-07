from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict

from migraph.graph.graph_store import GraphStore


@dataclass(frozen=True)
class AgentResult:
    """
    Output of an agent's evaluation over a graph variant.

    - score: final scalar judgment in [0, 1]
    - rationale: structured explanation for the score
    - signals: raw measurements used to derive the score
    """

    agent: str
    graph_variant: str
    score: float
    rationale: Dict[str, Any]
    signals: Dict[str, Any]


class BaseAgent(ABC):
    """
    Abstract base class for epistemic agents.

    Agents:
    - do NOT generate language
    - do NOT call LLMs
    - ONLY evaluate graph variants
    """

    name: str

    @abstractmethod
    def evaluate(
        self,
        graph: GraphStore,
        query_context: Dict[str, Any],
    ) -> AgentResult:
        """
        Evaluate a graph variant under a specific epistemic strategy.

        Implementations MUST:
        - read graph_variant from graph.metadata
        - populate rationale with structured data
        """
        raise NotImplementedError
