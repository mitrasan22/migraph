from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Set
from pathlib import Path
import time
import logging

import numpy as np

from migraph.graph.graph_store import GraphStore
from migraph.graph.graph_query import GraphQueryEngine
from migraph.graph.graph_schema import Edge
from migraph.config.settings import GraphConfig

from migraph.embeddings.encoder import EmbeddingEncoder
from migraph.embeddings.similarity import SimilarityComputer


# ---------------------------------------------------------------------
# Retrieved context
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class RetrievedContext:
    """
    Retrieved subgraph evidence from a single graph variant.
    """

    graph_variant: str
    node_ids: List[str]
    edges: List[Edge]
    metadata: Dict[str, Any]


# ---------------------------------------------------------------------
# Graph retriever with semantic gating
# ---------------------------------------------------------------------

class GraphRetriever:
    """
    Retrieves bounded subgraphs from multiple graph variants.

    Retrieval is:
    - graph-structured (k-hop traversal)
    - confidence-aware
    - temporally aware
    - semantically gated via embeddings
    """

    def __init__(
        self,
        *,
        graph_config: GraphConfig,
        encoder: EmbeddingEncoder,
        min_similarity: float = 0.25,
        embeddings_path: Path | None = None,
    ) -> None:
        self.config = graph_config
        self.encoder = encoder
        self.min_similarity = min_similarity
        self.embeddings_path = embeddings_path
        self._node_embeddings: Dict[str, np.ndarray] | None = None
        self._edge_embeddings: Dict[str, np.ndarray] | None = None
        # Shared cache across instances (per embeddings_path)
        if not hasattr(GraphRetriever, "_GLOBAL_EMBEDDINGS"):
            GraphRetriever._GLOBAL_EMBEDDINGS = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        *,
        graphs: List[GraphStore],
        seed_nodes: List[str],
        query_text: str,
    ) -> List[RetrievedContext]:

        contexts: List[RetrievedContext] = []
        timeout_ms = float(self.config.retrieval_timeout_ms or 0)
        deadline = None
        if timeout_ms > 0:
            deadline = time.perf_counter() + (timeout_ms / 1000.0)
        # Always print a one-line entry so we can confirm this codepath executes.
        logging.getLogger("migraph.retrieval").info(
            "enter retrieve graphs=%s seeds=%s",
            len(graphs),
            len(seed_nodes),
        )

        # Encode query intent once
        t_q = time.perf_counter()
        query_embedding = self._encode_query(query_text, seed_nodes)
        if not self._embeddings_dim_matches(query_embedding):
            logging.getLogger("migraph.retrieval").warning(
                "embedding dim mismatch; skipping semantic gating"
            )
            return contexts
        logging.getLogger("migraph.retrieval").info(
            "query embedding in %.2f ms",
            (time.perf_counter() - t_q) * 1000.0,
        )
        self._ensure_embedding_index()

        for graph in graphs:
            engine = GraphQueryEngine(graph)

            collected_nodes: Set[str] = set()
            collected_edges: Dict[str, Edge] = {}
            timed_out = False

            for seed in seed_nodes:
                if deadline is not None and time.perf_counter() > deadline:
                    timed_out = True
                    logging.getLogger("migraph.retrieval").warning(
                        "timeout reached before seed loop"
                    )
                    break
                t_seed = time.perf_counter()
                subgraph = engine.k_hop_subgraph(
                    start=seed,
                    k=self.config.max_hops,
                    min_confidence=(
                        0.0
                        if self.config.allow_weak_edges
                        else self.config.min_edge_confidence
                    ),
                    temporal_reasoning=self.config.temporal_reasoning,
                    undirected=True,
                )
                logging.getLogger("migraph.retrieval").info(
                    "seed=%s k-hop nodes=%s edges=%s in %.2f ms",
                    seed,
                    len(subgraph.nodes),
                    len(subgraph.edges),
                    (time.perf_counter() - t_seed) * 1000.0,
                )

                # Always keep the seed node
                collected_nodes.add(seed)

                t_sem = time.perf_counter()
                for node_id in subgraph.nodes:
                    if deadline is not None and time.perf_counter() > deadline:
                        timed_out = True
                        logging.getLogger("migraph.retrieval").warning(
                            "timeout during node filter"
                        )
                        break
                    if self._semantic_accept_node(
                        graph=graph,
                        node_id=node_id,
                        query_embedding=query_embedding,
                    ):
                        collected_nodes.add(node_id)

                for edge in subgraph.edges:
                    if deadline is not None and time.perf_counter() > deadline:
                        timed_out = True
                        logging.getLogger("migraph.retrieval").warning(
                            "timeout during edge filter"
                        )
                        break
                    if self._semantic_accept_edge(
                        edge=edge,
                        query_embedding=query_embedding,
                    ):
                        collected_edges[edge.id] = edge
                logging.getLogger("migraph.retrieval").info(
                    "seed=%s semantic filter kept_nodes=%s kept_edges=%s in %.2f ms",
                    seed,
                    len(collected_nodes),
                    len(collected_edges),
                    (time.perf_counter() - t_sem) * 1000.0,
                )

                # Fallback: if nothing passed semantic filter for this seed,
                # keep the raw subgraph nodes/edges (highest-confidence first).
                if not collected_nodes and subgraph.nodes:
                    collected_nodes.update(subgraph.nodes)
                if not collected_edges and subgraph.edges:
                    ranked = sorted(
                        subgraph.edges,
                        key=lambda e: (e.confidence, e.weight),
                        reverse=True,
                    )
                    for edge in ranked[:25]:
                        collected_edges[edge.id] = edge

                if timed_out:
                    break

            contexts.append(
                RetrievedContext(
                    graph_variant=graph.metadata.get("variant", "unknown"),
                    node_ids=list(collected_nodes),
                    edges=list(collected_edges.values()),
                    metadata={
                        "node_count": len(collected_nodes),
                        "edge_count": len(collected_edges),
                        "max_hops": self.config.max_hops,
                        "min_edge_confidence": self.config.min_edge_confidence,
                        "allow_weak_edges": self.config.allow_weak_edges,
                        "temporal_reasoning": self.config.temporal_reasoning,
                        "min_similarity": self.min_similarity,
                        "retrieval_density": (
                            len(collected_edges) / max(len(collected_nodes), 1)
                        ),
                        "timed_out": timed_out,
                        "timeout_ms": timeout_ms,
                    },
                )
            )

        return contexts

    # ------------------------------------------------------------------
    # Semantic gating
    # ------------------------------------------------------------------

    def _encode_query(self, query_text: str, entities: List[str]) -> np.ndarray:
        """
        Encode query intent from seed entities.
        """
        text = f"{query_text} " + " ".join(entities)
        return self.encoder.encode([text])[0]

    def _embeddings_dim_matches(self, query_embedding: np.ndarray) -> bool:
        if not self._node_embeddings:
            return True
        any_vec = next(iter(self._node_embeddings.values()), None)
        if any_vec is None:
            return True
        return query_embedding.shape[0] == any_vec.shape[0]

    def _ensure_embedding_index(self) -> None:
        if self._node_embeddings is not None:
            return
        if self.embeddings_path is None or not self.embeddings_path.exists():
            self._node_embeddings = {}
            self._edge_embeddings = {}
            return
        cache_key = str(self.embeddings_path.resolve())
        cached = GraphRetriever._GLOBAL_EMBEDDINGS.get(cache_key)
        if cached is not None:
            self._node_embeddings = cached["nodes"]
            self._edge_embeddings = cached["edges"]
            return
        try:
            import pandas as pd

            emb_df = pd.read_parquet(self.embeddings_path)
            self._node_embeddings = {}
            self._edge_embeddings = {}
            sample_vec = None
            if "vector" in emb_df.columns and len(emb_df) > 0:
                sample_vec = emb_df.iloc[0]["vector"]
            for _, row in emb_df.iterrows():
                if row.get("kind") == "node":
                    self._node_embeddings[str(row["id"])] = np.asarray(
                        row["vector"], dtype=np.float32
                    )
                elif row.get("kind") == "edge":
                    self._edge_embeddings[str(row["id"])] = np.asarray(
                        row["vector"], dtype=np.float32
                    )
            if sample_vec is not None:
                probe = self.encoder.encode(["dimension_check"])[0]
                if len(probe) != len(sample_vec):
                    self._node_embeddings = {}
                    self._edge_embeddings = {}
                    logging.getLogger("migraph.retrieval").warning(
                        "embeddings dim mismatch; ignoring cached embeddings"
                    )
                    return

            GraphRetriever._GLOBAL_EMBEDDINGS[cache_key] = {
                "nodes": self._node_embeddings,
                "edges": self._edge_embeddings,
            }
            logging.getLogger("migraph.retrieval").info(
                "embeddings index loaded: nodes=%s edges=%s path=%s",
                len(self._node_embeddings),
                len(self._edge_embeddings),
                self.embeddings_path,
            )
        except Exception:
            self._node_embeddings = {}
            self._edge_embeddings = {}

    def _semantic_accept_node(
        self,
        *,
        graph: GraphStore,
        node_id: str,
        query_embedding: np.ndarray,
    ) -> bool:
        """
        Decide whether a node is semantically relevant.
        """
        if self._node_embeddings is not None:
            vec = self._node_embeddings.get(node_id)
            if vec is not None:
                similarity = SimilarityComputer.cosine(query_embedding, vec)
                return similarity >= self.min_similarity
            # If embeddings are available but missing this node, keep it
            # to avoid slow per-node encoding.
            return True

        try:
            node = graph.get_node(node_id)
        except KeyError:
            return False

        text = f"{node.label} {node.type} {node.attributes}"
        node_embedding = self.encoder.encode([text])[0]

        similarity = SimilarityComputer.cosine(
            query_embedding,
            node_embedding,
        )

        return similarity >= self.min_similarity

    def _semantic_accept_edge(
        self,
        *,
        edge: Edge,
        query_embedding: np.ndarray,
    ) -> bool:
        """
        Decide whether an edge is semantically relevant.
        """
        if self._edge_embeddings is not None:
            vec = self._edge_embeddings.get(edge.id)
            if vec is not None:
                similarity = SimilarityComputer.cosine(query_embedding, vec)
                return similarity >= self.min_similarity
            return True

        text = f"{edge.relation} {edge.causal_type} {edge.provenance}"
        edge_embedding = self.encoder.encode([text])[0]

        similarity = SimilarityComputer.cosine(
            query_embedding,
            edge_embedding,
        )

        return similarity >= self.min_similarity
