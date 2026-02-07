from __future__ import annotations

from dataclasses import dataclass
from typing import Set, List

from migraph.graph.graph_store import GraphStore
from migraph.graph.graph_schema import Edge


@dataclass(frozen=True)
class SubgraphResult:
    """
    Result of a bounded graph traversal.

    Defines the epistemic boundary for retrieval.
    """

    nodes: Set[str]
    edges: List[Edge]


class GraphQueryEngine:
    """
    Controlled traversal engine with semantic filtering.

    This engine defines *what evidence is allowed*
    to flow into GraphRAG generation.
    """

    def __init__(self, store: GraphStore) -> None:
        self.store = store

    def k_hop_subgraph(
        self,
        *,
        start: str,
        k: int,
        min_confidence: float = 0.0,
        allow_hypothetical: bool = True,
        temporal_reasoning: bool = False,
        undirected: bool = True,
    ) -> SubgraphResult:

        visited_nodes: Set[str] = set()
        collected_edges: dict[str, Edge] = {}

        frontier: Set[str] = {start}

        for _ in range(k):
            next_frontier: Set[str] = set()

            for node in frontier:
                if node in visited_nodes:
                    continue

                neighbors = set(self.store.neighbors(node))
                if undirected:
                    neighbors.update(self.store.predecessors(node))

                for nbr in neighbors:
                    if self.store.has_edge(node, nbr):
                        edge = self.store.get_edge(node, nbr)
                    elif self.store.has_edge(nbr, node):
                        edge = self.store.get_edge(nbr, node)
                    else:
                        continue

                    if edge.confidence < min_confidence:
                        continue

                    if not allow_hypothetical and edge.causal_type == "hypothetical":
                        continue
                    
                    if temporal_reasoning and edge.end_time is not None:
                        continue

                    collected_edges[edge.id] = edge
                    next_frontier.add(nbr)

                visited_nodes.add(node)

            frontier = next_frontier

        return SubgraphResult(
            nodes=visited_nodes,
            edges=list(collected_edges.values()),
        )
