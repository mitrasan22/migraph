"""
Embedding subsystem for migraph.

Provides semantic encoding and similarity primitives used by:
- graph construction
- semantic retrieval
- shock / novelty detection
- agent reasoning

This module is model-agnostic by design.
"""

from migraph.embeddings.encoder import EmbeddingEncoder
from migraph.embeddings.similarity import SimilarityComputer

__all__ = [
    "EmbeddingEncoder",
    "SimilarityComputer",
]
