from __future__ import annotations

from typing import Iterable

from migraph.graph.graph_schema import Node, Edge
from migraph.graph.graph_store import GraphStore


class GraphBuilder:
    """
    Constructs the base knowledge graph from structured inputs.
    """

    def __init__(self, store: GraphStore) -> None:
        self.store = store

    def add_nodes(self, nodes: Iterable[Node]) -> None:
        for node in nodes:
            self.store.add_node(node)

    def add_edges(self, edges: Iterable[Edge]) -> None:
        for edge in edges:
            self.store.add_edge(edge)
