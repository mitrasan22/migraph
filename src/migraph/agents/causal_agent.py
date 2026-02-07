from __future__ import annotations

from typing import Dict, Any

from migraph.agents.base_agent import BaseAgent, AgentResult
from migraph.graph.graph_store import GraphStore


class CausalAgent(BaseAgent):
    """
    Evaluates explanations based on the presence and density
    of explicit causal edges in the graph variant.
    """

    name = "causal"

    def evaluate(
        self,
        graph: GraphStore,
        query_context: Dict[str, Any],
    ) -> AgentResult:

        edges = graph.get_edges()

        causal_edges = [e for e in edges if e.causal_type == "causal"]

        total_edges = max(len(edges), 1)
        causal_ratio = len(causal_edges) / total_edges

        return AgentResult(
            agent=self.name,
            graph_variant=graph.metadata.get("variant", "unknown"),
            score=causal_ratio,
            rationale={
                "strategy": "causal_only",
                "description": "Scores higher when explanations rely on explicit causal edges",
                "causal_edges": len(causal_edges),
                "total_edges": total_edges,
            },
            signals={
                "causal_edge_ratio": causal_ratio,
            },
        )
