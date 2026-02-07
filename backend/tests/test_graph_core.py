from migraph.graph.graph_schema import Node, Edge
from migraph.graph.graph_store import GraphStore
from migraph.graph.graph_query import GraphQueryEngine


def _make_edge(source: str, target: str, *, end_time: str | None = None) -> Edge:
    return Edge.create(
        source=source,
        target=target,
        relation="rel",
        weight=0.8,
        confidence=0.9,
        causal_type="causal",
        end_time=end_time,
    )


def test_graph_store_accessors_and_edges_between():
    graph = GraphStore()

    a = Node.create(label="A", type="entity")
    b = Node.create(label="B", type="entity")
    c = Node.create(label="C", type="entity")

    graph.add_node(a)
    graph.add_node(b)
    graph.add_node(c)

    e1 = _make_edge(a.id, b.id)
    e2 = _make_edge(b.id, c.id)
    graph.add_edge(e1)
    graph.add_edge(e2)

    nodes = graph.get_nodes()
    assert len(nodes) == 3
    assert {n.id for n in nodes} == {a.id, b.id, c.id}

    fetched = graph.get_edge(a.id, b.id)
    assert fetched.id == e1.id

    between = graph.get_edges_between({a.id, b.id})
    assert len(between) == 1
    assert between[0].id == e1.id


def test_k_hop_temporal_reasoning_filters_expired_edges():
    graph = GraphStore()

    a = Node.create(label="A", type="entity")
    b = Node.create(label="B", type="entity")
    c = Node.create(label="C", type="entity")
    graph.add_node(a)
    graph.add_node(b)
    graph.add_node(c)

    active = _make_edge(a.id, b.id, end_time=None)
    expired = _make_edge(a.id, c.id, end_time="2020-01-01")

    graph.add_edge(active)
    graph.add_edge(expired)

    engine = GraphQueryEngine(graph)

    without_temporal = engine.k_hop_subgraph(start=a.id, k=1)
    assert len(without_temporal.edges) == 2

    with_temporal = engine.k_hop_subgraph(
        start=a.id,
        k=1,
        temporal_reasoning=True,
    )
    assert len(with_temporal.edges) == 1
    assert with_temporal.edges[0].id == active.id
