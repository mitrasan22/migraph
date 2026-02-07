from __future__ import annotations

import networkx as nx
from typing import Iterable, List, Dict, Any

from migraph.graph.graph_schema import Node, Edge


class GraphStore:
    """
    Authoritative in-memory graph representation.

    Supports safe cloning and metadata attachment for inference variants.
    """

    def __init__(self) -> None:
        self._graph = nx.DiGraph()
        self.metadata: Dict[str, Any] = {}

    # -------------------- Nodes --------------------

    def add_node(self, node: Node) -> None:
        self._graph.add_node(node.id, data=node)

    def get_node(self, node_id: str) -> Node:
        return self._graph.nodes[node_id]["data"]

    def get_nodes(self) -> List[Node]:
        return [data["data"] for _, data in self._graph.nodes(data=True)]

    # -------------------- Edges --------------------

    def add_edge(self, edge: Edge) -> None:
        self._graph.add_edge(edge.source, edge.target, data=edge)

    def get_edge(self, source: str, target: str) -> Edge:
        return self._graph.edges[source, target]["data"]

    def remove_edge(self, source: str, target: str) -> None:
        if self._graph.has_edge(source, target):
            self._graph.remove_edge(source, target)

    def has_edge(self, source: str, target: str) -> bool:
        return self._graph.has_edge(source, target)

    def edges(self) -> Iterable[Edge]:
        for _, _, data in self._graph.edges(data=True):
            yield data["data"]

    def get_edges(self) -> List[Edge]:
        return list(self.edges())

    def get_edges_between(self, node_ids: Iterable[str]) -> List[Edge]:
        node_set = set(node_ids)
        edges: List[Edge] = []
        for u, v, data in self._graph.edges(data=True):
            if u in node_set and v in node_set:
                edges.append(data["data"])
        return edges

    # -------------------- Traversal --------------------

    def neighbors(self, node_id: str) -> List[str]:
        if node_id not in self._graph:
            return []
        return list(self._graph.successors(node_id))

    def predecessors(self, node_id: str) -> List[str]:
        if node_id not in self._graph:
            return []
        return list(self._graph.predecessors(node_id))

    # -------------------- Analytics --------------------

    def node_count(self) -> int:
        return self._graph.number_of_nodes()

    def edge_count(self) -> int:
        return self._graph.number_of_edges()

    # -------------------- Cloning --------------------

    def clone(self) -> "GraphStore":
        g = GraphStore()
        g._graph = self._graph.copy()
        g.metadata = dict(self.metadata)
        return g
