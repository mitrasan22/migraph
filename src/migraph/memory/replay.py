from __future__ import annotations

from typing import List, Dict, Any, Callable, Tuple

from migraph.memory.episodic import Episode
from migraph.embeddings.encoder import EmbeddingEncoder
from migraph.embeddings.similarity import SimilarityComputer


class EpisodicRecall:
    """
    Semantic recall over episodic memory.

    Finds past episodes relevant to a new query.
    """

    def __init__(
        self,
        *,
        encoder: EmbeddingEncoder,
        min_similarity: float = 0.3,
    ) -> None:
        self.encoder = encoder
        self.min_similarity = min_similarity

    def recall(
        self,
        *,
        query: str,
        episodes: List[Episode],
        top_k: int = 5,
    ) -> List[Tuple[Episode, float]]:

        if not episodes:
            return []

        query_embedding = self.encoder.encode_one(query)

        scored: List[Tuple[Episode, float]] = []

        for ep in episodes:
            sim = SimilarityComputer.cosine(
                query_embedding,
                ep.embedding,
            )
            if sim >= self.min_similarity:
                scored.append((ep, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


class MemoryReplayer:
    """
    Replays past epistemic episodes under new conditions.

    Enables:
    - counterfactual memory analysis
    - bias re-evaluation
    - longitudinal belief drift detection
    """

    def __init__(
        self,
        *,
        runtime_fn: Callable[[str, List[str]], Dict[str, Any]],
    ) -> None:
        self.runtime_fn = runtime_fn

    def replay(
        self,
        *,
        episodes: List[Episode],
        override_bias: str | None = None,
    ) -> List[Dict[str, Any]]:

        replays: List[Dict[str, Any]] = []

        for ep in episodes:
            result = self.runtime_fn(
                query=ep.query,
                entities=ep.entities,
            )

            replays.append(
                {
                    "episode_id": ep.id,
                    "original_answer": ep.answer,
                    "original_confidence": ep.confidence,
                    "new_answer": result.get("answer"),
                    "new_confidence": result.get("confidence"),
                    "confidence_delta": (
                        result.get("confidence", 0.0) - ep.confidence
                    ),
                    "override_bias": override_bias,
                }
            )

        return replays
