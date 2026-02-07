from __future__ import annotations

from typing import Iterable, List, Dict
import math
from pathlib import Path
import numpy as np
import re

from migraph.graph.graph_store import GraphStore
from migraph.graph.graph_schema import Edge
from migraph.shock.shock_score import ShockScore
from migraph.config.settings import ShockConfig

from migraph.embeddings.encoder import EmbeddingEncoder
from migraph.embeddings.similarity import SimilarityComputer
import logging


class ShockDetector:
    """
    Detects novelty and instability in reasoning requests.

    This detector intentionally separates:
    - entity novelty
    - semantic novelty (NEW)
    - structural fragility
    - temporal inconsistency
    - evidence weakness

    The result controls mutation intensity but does NOT
    decide escalation behavior.
    """

    def __init__(
        self,
        *,
        graph: GraphStore,
        encoder: EmbeddingEncoder,
        config: ShockConfig,
        embeddings_path: Path | None = None,
    ) -> None:
        self.graph = graph
        self.encoder = encoder
        self.config = config
        self.embeddings_path = embeddings_path
        self._node_embeddings: Dict[str, np.ndarray] | None = None
        self._edge_embeddings: Dict[str, np.ndarray] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, query_entities: Iterable[str]) -> ShockScore:
        """
        Compute a multi-dimensional shock score for a query.
        """

        if not self.config.enabled:
            return ShockScore(
                overall=0.0,
                novelty=0.0,
                structural_risk=0.0,
                temporal_risk=0.0,
                evidence_fragility=0.0,
                details={"disabled": True},
            )

        entity_novelty = self._entity_novelty(query_entities)
        semantic_novelty = self._semantic_novelty(query_entities)

        novelty = max(entity_novelty, semantic_novelty)

        structural = self._structural_risk()
        temporal = self._temporal_risk()
        fragility = self._evidence_fragility()

        overall = self._aggregate(
            novelty,
            structural,
            temporal,
            fragility,
        )

        return ShockScore(
            overall=overall,
            novelty=novelty,
            structural_risk=structural,
            temporal_risk=temporal,
            evidence_fragility=fragility,
            details={
                "entity_novelty": entity_novelty,
                "semantic_novelty": semantic_novelty,
                "structural": structural,
                "temporal": temporal,
                "fragility": fragility,
                "novelty_window": self.config.novelty_window,
            },
        )

    def score_query(
        self,
        query_text: str,
        query_entities: Iterable[str],
    ) -> ShockScore:
        """
        Compute shock using full query text + inferred entities.
        """

        if not self.config.enabled:
            return ShockScore(
                overall=0.0,
                novelty=0.0,
                structural_risk=0.0,
                temporal_risk=0.0,
                evidence_fragility=0.0,
                details={"disabled": True},
            )

        entity_novelty = self._entity_novelty(query_entities)
        semantic_novelty = self._semantic_novelty_text(query_text)
        token_novelty = self._token_novelty(query_text, query_entities)

        novelty = max(entity_novelty, semantic_novelty, token_novelty)

        structural = self._structural_risk()
        temporal = self._temporal_risk()
        fragility = self._evidence_fragility()

        overall = self._aggregate(
            novelty,
            structural,
            temporal,
            fragility,
        )

        return ShockScore(
            overall=overall,
            novelty=novelty,
            structural_risk=structural,
            temporal_risk=temporal,
            evidence_fragility=fragility,
            details={
                "entity_novelty": entity_novelty,
                "semantic_novelty": semantic_novelty,
                "token_novelty": token_novelty,
                "structural": structural,
                "temporal": temporal,
                "fragility": fragility,
                "novelty_window": self.config.novelty_window,
            },
        )

    # ------------------------------------------------------------------
    # Novelty components
    # ------------------------------------------------------------------

    def _entity_novelty(self, entities: Iterable[str]) -> float:
        """
        Measures how unfamiliar the entities are relative to the graph.
        """

        entities = list(entities)[: self.config.novelty_window]

        if not entities:
            return 1.0

        seen = sum(1 for e in entities if e in self.graph._graph)
        return 1.0 - (seen / len(entities))

    def _semantic_novelty(self, entities: Iterable[str]) -> float:
        """
        Measures semantic novelty of the query relative to graph knowledge.
        """

        entities = list(entities)[: self.config.novelty_window]
        if not entities:
            return 1.0

        query_text = " ".join(entities)
        query_embedding = self.encoder.encode_one(query_text)

        # Fast path: use cached embeddings if available
        self._ensure_embedding_index()
        if not self._embeddings_dim_matches(query_embedding):
            return 0.0

        similarities: List[float] = []
        if self._node_embeddings or self._edge_embeddings:
            for vec in (self._node_embeddings or {}).values():
                similarities.append(
                    SimilarityComputer.cosine(query_embedding, vec)
                )
            for vec in (self._edge_embeddings or {}).values():
                similarities.append(
                    SimilarityComputer.cosine(query_embedding, vec)
                )
        else:
            # Slow path: sample a limited number of nodes/edges
            max_samples = max(self.config.novelty_window, 25)
            for node in list(self.graph.get_nodes())[:max_samples]:
                text = f"{node.label} {node.type} {node.attributes}"
                emb = self.encoder.encode_one(text)
                similarities.append(
                    SimilarityComputer.cosine(query_embedding, emb)
                )

            for edge in list(self.graph.get_edges())[:max_samples]:
                text = f"{edge.relation} {edge.causal_type} {edge.provenance}"
                emb = self.encoder.encode_one(text)
                similarities.append(
                    SimilarityComputer.cosine(query_embedding, emb)
                )

        if not similarities:
            return 1.0

        # Novelty = lack of strong semantic match
        return 1.0 - max(similarities)

    def _semantic_novelty_text(self, query_text: str) -> float:
        if not query_text.strip():
            return 1.0

        query_embedding = self.encoder.encode_one(query_text)

        self._ensure_embedding_index()
        if not self._embeddings_dim_matches(query_embedding):
            return 0.0

        similarities: List[float] = []
        if self._node_embeddings or self._edge_embeddings:
            for vec in (self._node_embeddings or {}).values():
                similarities.append(
                    SimilarityComputer.cosine(query_embedding, vec)
                )
            for vec in (self._edge_embeddings or {}).values():
                similarities.append(
                    SimilarityComputer.cosine(query_embedding, vec)
                )
        else:
            max_samples = max(self.config.novelty_window, 25)
            for node in list(self.graph.get_nodes())[:max_samples]:
                text = f"{node.label} {node.type} {node.attributes}"
                emb = self.encoder.encode_one(text)
                similarities.append(
                    SimilarityComputer.cosine(query_embedding, emb)
                )
            for edge in list(self.graph.get_edges())[:max_samples]:
                text = f"{edge.relation} {edge.causal_type} {edge.provenance}"
                emb = self.encoder.encode_one(text)
                similarities.append(
                    SimilarityComputer.cosine(query_embedding, emb)
                )

        if not similarities:
            return 1.0

        return 1.0 - max(similarities)

    def _token_novelty(self, query_text: str, entities: Iterable[str]) -> float:
        tokens = set(re.findall(r"[a-z0-9]+", query_text.lower()))
        if not tokens:
            return 1.0
        entity_text = " ".join(list(entities))
        ent_tokens = set(re.findall(r"[a-z0-9]+", entity_text.lower()))
        if not ent_tokens:
            return 1.0
        coverage = len(tokens & ent_tokens) / max(len(tokens), 1)
        return 1.0 - coverage

    def _ensure_embedding_index(self) -> None:
        if self._node_embeddings is not None:
            return
        if self.embeddings_path is None or not self.embeddings_path.exists():
            self._node_embeddings = {}
            self._edge_embeddings = {}
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
                probe = self.encoder.encode_one("dimension_check")
                if len(probe) != len(sample_vec):
                    self._node_embeddings = {}
                    self._edge_embeddings = {}
                    logging.getLogger("migraph.shock").warning(
                        "embeddings dim mismatch; ignoring cached embeddings"
                    )
                    return

            logging.getLogger("migraph.shock").info(
                "embeddings index loaded: nodes=%s edges=%s path=%s",
                len(self._node_embeddings),
                len(self._edge_embeddings),
                self.embeddings_path,
            )
        except Exception:
            self._node_embeddings = {}
            self._edge_embeddings = {}

    def _embeddings_dim_matches(self, query_embedding: np.ndarray) -> bool:
        if not self._node_embeddings:
            return True
        any_vec = next(iter(self._node_embeddings.values()), None)
        if any_vec is None:
            return True
        return query_embedding.shape[0] == any_vec.shape[0]

    # ------------------------------------------------------------------
    # Risk components (unchanged)
    # ------------------------------------------------------------------

    def _structural_risk(self) -> float:
        edge_count = self.graph.edge_count()
        node_count = self.graph.node_count()

        if node_count == 0:
            return 1.0

        density = edge_count / max(node_count * (node_count - 1), 1)
        return min(abs(density - 0.05) * 5.0, 1.0)

    def _temporal_risk(self) -> float:
        edges: List[Edge] = list(self.graph.get_edges())
        if not edges:
            return 0.0

        expired = sum(1 for e in edges if e.end_time is not None)
        return expired / len(edges)

    def _evidence_fragility(self) -> float:
        edges: List[Edge] = list(self.graph.get_edges())
        if not edges:
            return 1.0

        weak = sum(
            1 for e in edges
            if e.confidence < 0.5 or e.causal_type == "hypothetical"
        )

        return weak / len(edges)

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def _aggregate(
        self,
        novelty: float,
        structural: float,
        temporal: float,
        fragility: float,
    ) -> float:
        """
        Aggregates shock components into a single control signal.

        Uses a conservative geometric mean to amplify
        simultaneous instability signals.
        """

        values = [
            max(novelty, 1e-6),
            max(structural, 1e-6),
            max(temporal, 1e-6),
            max(fragility, 1e-6),
        ]

        log_sum = sum(math.log(v) for v in values)
        return min(math.exp(log_sum / len(values)), 1.0)
