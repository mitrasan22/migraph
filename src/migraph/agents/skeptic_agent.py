from __future__ import annotations

from typing import Dict, Any

from migraph.agents.base_agent import BaseAgent, AgentResult
from migraph.graph.graph_store import GraphStore


class SkepticAgent(BaseAgent):
    """
    Stress-tests explanations by penalizing those that collapse
    when assumed edges are removed or invalidated.
    """

    name = "skeptic"

    def evaluate(
        self,
        graph: GraphStore,
        query_context: Dict[str, Any],
    ) -> AgentResult:

        removed_edges = int(query_context.get("edges_removed", 0))

        penalty = removed_edges * 0.1
        score = max(1.0 - penalty, 0.0)

        return AgentResult(
            agent=self.name,
            graph_variant=graph.metadata.get("variant", "unknown"),
            score=score,
            rationale={
                "strategy": "falsification",
                "description": "Penalizes explanations that degrade under edge removal",
                "edges_removed": removed_edges,
                "penalty_per_edge": 0.1,
            },
            signals={
                "edges_removed": removed_edges,
                "penalty": penalty,
            },
        )
