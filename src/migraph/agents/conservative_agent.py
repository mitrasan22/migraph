from __future__ import annotations

from typing import Dict, Any

from migraph.agents.base_agent import BaseAgent, AgentResult
from migraph.graph.graph_store import GraphStore


class ConservativeAgent(BaseAgent):
    """
    Penalizes explanations that rely on weak, hypothetical,
    or low-confidence evidence.
    """

    name = "conservative"

    def evaluate(
        self,
        graph: GraphStore,
        query_context: Dict[str, Any],
    ) -> AgentResult:

        edges = graph.get_edges()

        weak_edges = [
            e for e in edges if e.confidence < 0.5 or e.causal_type == "hypothetical"
        ]

        total_edges = max(len(edges), 1)
        weak_ratio = len(weak_edges) / total_edges
        score = max(1.0 - weak_ratio, 0.0)

        return AgentResult(
            agent=self.name,
            graph_variant=graph.metadata.get("variant", "unknown"),
            score=score,
            rationale={
                "strategy": "risk_averse",
                "description": "Penalizes reliance on weak or hypothetical evidence",
                "weak_edges": len(weak_edges),
                "total_edges": total_edges,
            },
            signals={
                "weak_edge_ratio": weak_ratio,
            },
        )
