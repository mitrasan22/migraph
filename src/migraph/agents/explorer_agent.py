from __future__ import annotations

from typing import Dict, Any

from migraph.agents.base_agent import BaseAgent, AgentResult
from migraph.graph.graph_store import GraphStore


class ExplorerAgent(BaseAgent):
    """
    Rewards explanations that surface novel, inferred,
    or amplified relationships beyond the base graph.
    """

    name = "explorer"

    def evaluate(
        self,
        graph: GraphStore,
        query_context: Dict[str, Any],
    ) -> AgentResult:

        edges = graph.get_edges()

        exploratory_edges = [
            e
            for e in edges
            if e.provenance in {"inferred", "amplified", "hypothetical"}
        ]

        total_edges = max(len(edges), 1)
        exploratory_ratio = len(exploratory_edges) / total_edges
        score = min(exploratory_ratio, 1.0)

        return AgentResult(
            agent=self.name,
            graph_variant=graph.metadata.get("variant", "unknown"),
            score=score,
            rationale={
                "strategy": "exploration_reward",
                "description": "Rewards discovery of inferred or amplified relationships",
                "exploratory_edges": len(exploratory_edges),
                "total_edges": total_edges,
            },
            signals={
                "exploratory_edge_ratio": exploratory_ratio,
                "exploratory_edge_count": len(exploratory_edges),
            },
        )
