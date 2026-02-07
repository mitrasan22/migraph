"""
Retrieval-Augmented Generation (RAG) subsystem for migraph.

This module performs:
- subgraph ensemble retrieval across graph variants
- context construction with provenance
- controlled generation
- epistemically-aware answer synthesis
"""

from migraph.rag.retriever import GraphRetriever, RetrievedContext
from migraph.rag.context_builder import ContextBuilder
from migraph.rag.generator import Generator
from migraph.rag.synthesizer import AnswerSynthesizer

__all__ = [
    "GraphRetriever",
    "RetrievedContext",
    "ContextBuilder",
    "Generator",
    "AnswerSynthesizer",
]
