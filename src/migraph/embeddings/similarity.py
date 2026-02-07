from __future__ import annotations

import numpy as np
from typing import Iterable, List


class SimilarityComputer:
    """
    Computes similarity between embeddings.
    Provides core similarity metrics and stability-aware aggregation.
    """

    # ------------------------------------------------------------------
    # Core similarities
    # ------------------------------------------------------------------

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute cosine similarity with numerical safety.
        """
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0.0:
            return 0.0
        return float(np.dot(a, b) / denom)

    @staticmethod
    def cosine_batch(
        query: np.ndarray,
        candidates: Iterable[np.ndarray],
    ) -> List[float]:
        """
        Compute cosine similarity between a query and many candidates.
        """
        return [SimilarityComputer.cosine(query, c) for c in candidates]

    # ------------------------------------------------------------------
    # Distance-based reasoning
    # ------------------------------------------------------------------

    @staticmethod
    def normalized_distance(a: np.ndarray, b: np.ndarray) -> float:
        """
        Normalized Euclidean distance in [0, 1].
        """
        dist = np.linalg.norm(a - b)
        return float(dist / (1.0 + dist))

    # ------------------------------------------------------------------
    # Ranking helpers (GraphRAG-critical)
    # ------------------------------------------------------------------

    @staticmethod
    def rank(
        query: np.ndarray,
        candidates: Iterable[np.ndarray],
    ) -> List[int]:
        """
        Rank candidates by descending similarity to query.
        Returns indices.
        """
        scores = SimilarityComputer.cosine_batch(query, candidates)
        return list(np.argsort(scores)[::-1])

    @staticmethod
    def top_k(
        query: np.ndarray,
        candidates: Iterable[np.ndarray],
        k: int,
    ) -> List[int]:
        """
        Return indices of top-k most similar candidates.
        """
        return SimilarityComputer.rank(query, candidates)[:k]

    # ------------------------------------------------------------------
    # Stability-aware similarity
    # ------------------------------------------------------------------

    @staticmethod
    def stable_similarity(
        scores: Iterable[float],
    ) -> float:
        """
        Aggregates multiple similarity scores conservatively.

        Penalizes variance to avoid overconfident similarity.
        """
        scores = list(scores)
        if not scores:
            return 0.0

        mean = float(np.mean(scores))
        std = float(np.std(scores))

        return max(mean - std, 0.0)
