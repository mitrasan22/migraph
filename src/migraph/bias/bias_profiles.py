from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from migraph.graph.graph_store import GraphStore
from migraph.graph.graph_schema import Edge


class BiasProfile(ABC):
    """
    Abstract bias profile.

    A bias profile defines the *direction* of epistemic distortion.
    The *strength* is supplied externally by policy.
    """

    name: str

    @abstractmethod
    def apply(
        self,
        graph: GraphStore,
        *,
        factor: float,
    ) -> None:
        """
        Apply bias to the graph in-place.
        """
        raise NotImplementedError


class BiasApplier:
    """
    Applies a bias profile with configurable intensity.
    """

    def __init__(
        self,
        *,
        profile: BiasProfile,
        amplification_factor: float = 1.0,
    ) -> None:
        self.profile = profile
        self.factor = amplification_factor

    def apply(self, graph: GraphStore) -> GraphStore:
        biased = graph.clone()
        biased.metadata["bias"] = self.profile.name
        biased.metadata["bias_strength"] = self.factor

        self.profile.apply(biased, factor=self.factor)
        return biased


class NeutralBias(BiasProfile):
    """
    No-op bias profile.
    """

    name = "neutral"

    def apply(
        self,
        graph: GraphStore,
        *,
        factor: float,
    ) -> None:
        return


class RiskAverseBias(BiasProfile):
    """
    Penalizes uncertain and non-causal edges.
    """

    name = "risk_averse"

    def apply(
        self,
        graph: GraphStore,
        *,
        factor: float,
    ) -> None:
        for e in list(graph.get_edges()):
            if not e.mutable:
                continue

            penalty = 0.7 * factor
            if e.confidence < 0.6 or e.causal_type != "causal":
                penalty = 0.5 * factor

            graph.remove_edge(e.source, e.target)
            graph.add_edge(
                Edge(
                    id=e.id,
                    source=e.source,
                    target=e.target,
                    relation=e.relation,
                    weight=max(e.weight * penalty, 0.01),
                    confidence=e.confidence,
                    start_time=e.start_time,
                    end_time=e.end_time,
                    causal_type=e.causal_type,
                    mutable=e.mutable,
                    provenance=e.provenance,
                    tags=e.tags,
                )
            )


class OptimisticBias(BiasProfile):
    """
    Amplifies weak but plausible edges.
    """

    name = "optimistic"

    def apply(
        self,
        graph: GraphStore,
        *,
        factor: float,
    ) -> None:
        for e in list(graph.get_edges()):
            if not e.mutable:
                continue

            boost = 1.2 * factor
            if e.causal_type == "hypothetical":
                boost = 1.4 * factor

            graph.remove_edge(e.source, e.target)
            graph.add_edge(
                Edge(
                    id=e.id,
                    source=e.source,
                    target=e.target,
                    relation=e.relation,
                    weight=min(e.weight * boost, 1.0),
                    confidence=e.confidence,
                    start_time=e.start_time,
                    end_time=e.end_time,
                    causal_type=e.causal_type,
                    mutable=e.mutable,
                    provenance=e.provenance,
                    tags=e.tags,
                )
            )


class SkepticalBias(BiasProfile):
    """
    Removes or strongly downweights edges that lack strong evidence.
    """

    name = "skeptical"

    def apply(
        self,
        graph: GraphStore,
        *,
        factor: float,
    ) -> None:
        for e in list(graph.get_edges()):
            if not e.mutable:
                continue

            if e.confidence < 0.5 or e.causal_type == "hypothetical":
                graph.remove_edge(e.source, e.target)
                continue

            graph.remove_edge(e.source, e.target)
            graph.add_edge(
                Edge(
                    id=e.id,
                    source=e.source,
                    target=e.target,
                    relation=e.relation,
                    weight=max(e.weight * (0.8 * factor), 0.01),
                    confidence=e.confidence,
                    start_time=e.start_time,
                    end_time=e.end_time,
                    causal_type=e.causal_type,
                    mutable=e.mutable,
                    provenance=e.provenance,
                    tags=e.tags,
                )
            )
