from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any
import re

import numpy as np

from migraph.graph.graph_store import GraphStore
from migraph.embeddings.encoder import EmbeddingEncoder


@dataclass
class EntityInferenceConfig:
    max_entities: int = 5
    max_tfidf_features: int = 50000
    tfidf_stop_words: str | None = "english"
    min_score: float = 0.35


class EntityInferer:
    """
    Production-grade entity inference helper.

    Strategy order:
    1) Embedding similarity (if embeddings.parquet exists)
    2) TF-IDF over node labels
    3) String label match fallback
    """

    def __init__(
        self,
        *,
        graph: GraphStore,
        encoder: EmbeddingEncoder,
        embeddings_path: Path | None,
        config: EntityInferenceConfig | None = None,
    ) -> None:
        self.graph = graph
        self.encoder = encoder
        self.embeddings_path = embeddings_path
        self.config = config or EntityInferenceConfig()
        self._embedding_index: Dict[str, np.ndarray] | None = None
        self._tfidf_index: Dict[str, Any] | None = None
        self.last_info: Dict[str, Any] = {}

    def infer(self, query: str) -> List[str]:
        if not query.strip():
            self.last_info = {
                "method": "none",
                "entities": [],
                "candidates": [],
            }
            return []

        emb_scores = self._score_by_embedding(query)
        tfidf_scores = self._score_by_tfidf(query)
        label_scores = self._score_by_label_match(query)

        if not emb_scores and not tfidf_scores and not label_scores:
            self.last_info = {
                "method": "none",
                "entities": [],
                "candidates": [],
            }
            return []

        emb_norm = self._normalize_scores(emb_scores)
        tfidf_norm = self._normalize_scores(tfidf_scores)
        label_norm = self._normalize_scores(label_scores)

        weights = {
            "embedding": 1.0,
            "tfidf": 0.7,
            "label_match": 0.4,
        }

        merged: Dict[str, Dict[str, float]] = {}
        for node_id, score in emb_norm.items():
            merged.setdefault(node_id, {})["embedding"] = score
        for node_id, score in tfidf_norm.items():
            merged.setdefault(node_id, {})["tfidf"] = score
        for node_id, score in label_norm.items():
            merged.setdefault(node_id, {})["label_match"] = score

        candidates: List[Dict[str, Any]] = []
        for node_id, components in merged.items():
            total = (
                weights["embedding"] * components.get("embedding", 0.0)
                + weights["tfidf"] * components.get("tfidf", 0.0)
                + weights["label_match"] * components.get("label_match", 0.0)
            )
            candidates.append(
                {
                    "id": node_id,
                    "score": float(total),
                    "components": components,
                }
            )

        # Filter candidates that don't appear in the query tokens (generic containment).
        query_tokens = set(
            t for t in re.findall(r"[a-zA-Z0-9_-]+", query.lower()) if len(t) >= 3
        )
        # Tokens that commonly appear in ids/labels and should be covered if present.
        key_tokens = {t for t in query_tokens if t not in {"and", "with", "from", "that", "this", "when", "how"}}

        def is_referenced(node_id: str) -> bool:
            parts = re.findall(r"[a-zA-Z0-9_-]+", node_id.lower())
            return any(p in query_tokens for p in parts)

        filtered = []
        for c in candidates:
            if c["score"] >= self.config.min_score and is_referenced(c["id"]):
                filtered.append(c)

        if filtered:
            candidates = filtered

        # Ensure coverage: for each key token in the question, include at least one
        # candidate that references it (generic, not domain-specific).
        if key_tokens:
            by_token: Dict[str, Dict[str, Any]] = {}
            for c in candidates:
                cid = c["id"].lower()
                for t in key_tokens:
                    if t in cid and t not in by_token:
                        by_token[t] = c
            for t in key_tokens:
                if t not in by_token:
                    # search in full candidate list for a match
                    for c in sorted(candidates, key=lambda x: x["score"], reverse=True):
                        if t in c["id"].lower():
                            by_token[t] = c
                            break
            for c in by_token.values():
                if c not in candidates:
                    candidates.append(c)

        candidates.sort(key=lambda c: c["score"], reverse=True)
        candidates = candidates[: self.config.max_entities]

        self.last_info = {
            "method": "hybrid",
            "entities": [c["id"] for c in candidates],
            "candidates": candidates,
        }
        return candidates

    def _score_by_embedding(self, query: str) -> Dict[str, float]:
        if self._embedding_index is None:
            if self.embeddings_path is None or not self.embeddings_path.exists():
                self._embedding_index = {}
                return {}

            try:
                import pandas as pd

                emb_df = pd.read_parquet(self.embeddings_path)
                emb_df = emb_df[emb_df["kind"] == "node"]
                self._embedding_index = {
                    str(row["id"]): np.asarray(row["vector"], dtype=np.float32)
                    for _, row in emb_df.iterrows()
                }
            except Exception:
                self._embedding_index = {}
                return {}

        if not self._embedding_index:
            return {}

        try:
            q_vec = self.encoder.encode([query])[0]
        except Exception:
            return {}

        # Guard against mismatched embedding dimensions
        any_vec = next(iter(self._embedding_index.values()), None)
        if any_vec is None or q_vec.shape[0] != any_vec.shape[0]:
            return {}

        q_norm = np.linalg.norm(q_vec) + 1e-8
        scores: Dict[str, float] = {}
        for node_id, vec in self._embedding_index.items():
            denom = (np.linalg.norm(vec) + 1e-8) * q_norm
            sim = float(np.dot(q_vec, vec) / denom)
            if sim > 0:
                scores[str(node_id)] = sim

        return scores

    def _score_by_tfidf(self, query: str) -> Dict[str, float]:
        if self._tfidf_index is None:
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.metrics.pairwise import cosine_similarity
            except Exception:
                self._tfidf_index = {}
                return {}

            node_ids: List[str] = []
            labels: List[str] = []
            for node in self.graph.get_nodes():
                node_ids.append(node.id)
                labels.append(str(node.label))

            if not labels:
                self._tfidf_index = {}
                return {}

            vectorizer = TfidfVectorizer(
                lowercase=True,
                stop_words=self.config.tfidf_stop_words,
                max_features=self.config.max_tfidf_features,
            )
            matrix = vectorizer.fit_transform(labels)
            self._tfidf_index = {
                "vectorizer": vectorizer,
                "matrix": matrix,
                "node_ids": node_ids,
                "cosine_similarity": cosine_similarity,
            }

        if not self._tfidf_index:
            return {}

        vectorizer = self._tfidf_index["vectorizer"]
        matrix = self._tfidf_index["matrix"]
        node_ids = self._tfidf_index["node_ids"]
        cosine_similarity = self._tfidf_index["cosine_similarity"]

        try:
            q_vec = vectorizer.transform([query])
        except Exception:
            return {}

        sims = cosine_similarity(q_vec, matrix).flatten()
        if sims.size == 0:
            return {}

        scores: Dict[str, float] = {}
        top_idx = sims.argsort()[::-1][: self.config.max_entities * 5]
        for i in top_idx:
            if sims[i] > 0:
                scores[str(node_ids[i])] = float(sims[i])
        return scores

    def _score_by_label_match(self, query: str) -> Dict[str, float]:
        tokens = [
            t for t in re.findall(r"[a-zA-Z0-9_-]+", query.lower())
            if len(t) >= 3
        ]
        if not tokens:
            return {}

        scores: Dict[str, float] = {}
        for node in self.graph.get_nodes():
            label = str(node.label).lower()
            score = sum(1 for t in tokens if t in label)
            if score > 0:
                scores[str(node.id)] = float(score)

        return scores

    @staticmethod
    def _normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
        if not scores:
            return {}
        max_val = max(scores.values())
        if max_val <= 0:
            return {}
        return {k: float(v / max_val) for k, v in scores.items()}
