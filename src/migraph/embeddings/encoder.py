from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List, Dict
import hashlib
import numpy as np


class EmbeddingEncoder(ABC):
    """
    Abstract embedding encoder.

    Concrete implementations may wrap:
    - sentence transformers
    - LLM embedding APIs
    - domain-specific encoders

    This interface intentionally hides model details.
    """

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self._cache: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, texts: Iterable[str]) -> List[np.ndarray]:
        """
        Encode multiple texts into embeddings.

        Uses deterministic caching to avoid recomputation.
        """
        embeddings: List[np.ndarray] = []

        for text in texts:
            key = self._hash(text)
            if key not in self._cache:
                self._cache[key] = self._encode_one(text)
            embeddings.append(self._cache[key])

        return embeddings

    def encode_one(self, text: str) -> np.ndarray:
        """
        Encode a single text into an embedding.
        """
        return self.encode([text])[0]

    # ------------------------------------------------------------------
    # Implementation contract
    # ------------------------------------------------------------------

    @abstractmethod
    def _encode_one(self, text: str) -> np.ndarray:
        """
        Encode a single text into a vector.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _hash(self, text: str) -> str:
        """
        Stable cache key incorporating encoder identity.
        Prevents silent reuse across different models/configs.
        """
        payload = f"{self.__class__.__name__}:{self.dimension}:{text}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
