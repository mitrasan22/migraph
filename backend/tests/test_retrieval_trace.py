import numpy as np

from migraph.graph.graph_store import GraphStore
from migraph.graph.graph_schema import Node, Edge
from migraph.memory.episodic import EpisodicMemory
from migraph.rag.generator import Generator
from migraph.config.settings import (
    GraphConfig,
    ShockConfig,
    MutationConfig,
    UncertaintyConfig,
    BiasConfig,
    MigraphConfig,
)

from backend.app.services.graphrag_service import GraphRAGService


class DummyEncoder:
    def encode(self, texts):
        return [self._vec(t) for t in texts]

    def encode_one(self, text):
        return self._vec(text)

    def _vec(self, text):
        # Simple deterministic vector based on character codes
        s = sum(ord(c) for c in str(text))
        return np.array([(s % 97) / 97.0, (s % 193) / 193.0], dtype=np.float32)


class DummyGenerator(Generator):
    def generate(self, query, context, *, uncertainty, bias, shock):
        return "ok"

    def stream_generate(self, query, context, *, uncertainty, bias, shock):
        yield "ok"


def _make_config():
    return MigraphConfig(
        graph=GraphConfig(
            max_hops=1,
            allow_weak_edges=True,
            min_edge_confidence=0.0,
            temporal_reasoning=False,
            retrieval_timeout_ms=0.0,
        ),
        shock=ShockConfig(
            enabled=False,
            shock_threshold=0.9,
            novelty_window=10,
            escalation_policy="mutate",
        ),
        mutation=MutationConfig(
            enabled=False,
            max_variants=1,
            edge_decay_factor=0.9,
            allow_edge_removal=False,
            allow_hypothetical_edges=False,
            mutation_strategy="adaptive",
        ),
        uncertainty=UncertaintyConfig(
            enabled=False,
            min_agreement_ratio=0.6,
            entropy_threshold=0.8,
            confidence_mode="hybrid",
        ),
        bias=BiasConfig(
            enabled=False,
            profile="neutral",
            amplification_factor=1.0,
        ),
        entity_max_entities=8,
    )


def test_stream_emits_metadata_for_aapl_query():
    graph = GraphStore()
    node_aapl = Node(id="ticker:aapl", label="AAPL", type="ticker", attributes={})
    node_usa = Node(id="country:usa", label="United States", type="country", attributes={})
    node_burden = Node(
        id="covid_burden:usa",
        label="COVID burden (USA)",
        type="covid_burden",
        attributes={},
    )
    graph.add_node(node_aapl)
    graph.add_node(node_usa)
    graph.add_node(node_burden)
    graph.add_edge(
        Edge.create(
            source="ticker:aapl",
            target="country:usa",
            relation="market_exposure",
            weight=0.9,
            confidence=0.9,
            causal_type="correlational",
        )
    )
    graph.add_edge(
        Edge.create(
            source="country:usa",
            target="covid_burden:usa",
            relation="has_burden",
            weight=0.8,
            confidence=0.8,
            causal_type="correlational",
        )
    )

    service = GraphRAGService(
        base_graph=graph,
        memory=EpisodicMemory(),
        generator=DummyGenerator(),
        embedding_encoder=DummyEncoder(),
        config=_make_config(),
        embeddings_path=None,
    )

    events = list(
        service.run_stream(
            query="How does AAPL's market exposure affects COVID burden in exposed countries?",
            entities=["ticker:aapl"],
            bias=None,
        )
    )

    meta = [e for e in events if e.get("type") == "metadata"]
    assert meta, "metadata event should be emitted"
