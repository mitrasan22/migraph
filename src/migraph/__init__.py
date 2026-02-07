"""
migraph
=======

A mutable-inference GraphRAG library that performs shock-aware,
counterfactual, uncertainty-aware, and bias-conditioned reasoning
by dynamically modifying the knowledge graph during inference.

Core idea:
- Reason by editing the graph, not just retrieving from it.

Public API:
- GraphStore
- GraphBuilder
- GraphMutator
- ShockDetector
"""

from migraph.graph.graph_store import GraphStore
from migraph.graph.graph_builder import GraphBuilder
from migraph.graph.graph_mutator import GraphMutator
from migraph.shock.shock_detector import ShockDetector

__all__ = [
    "GraphStore",
    "GraphBuilder",
    "GraphMutator",
    "ShockDetector",
]

__version__ = "0.1.0"
