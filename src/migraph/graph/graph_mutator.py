from __future__ import annotations

from typing import List

from migraph.graph.graph_store import GraphStore
from migraph.graph.graph_schema import Edge
from migraph.config.settings import MutationConfig
from migraph.bias.bias_profiles import BiasApplier, RiskAverseBias


class GraphMutator:
    """
    Generates epistemically distinct graph variants during inference.
    """

    def __init__(
        self,
        *,
        base_graph: GraphStore,
        config: MutationConfig,
    ) -> None:
        self.base_graph = base_graph
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_variants(self, *, shock_score: float) -> List[GraphStore]:
        """
        Generate graph variants based on mutation policy and shock intensity.
        """

        variants: List[GraphStore] = []

        # ---------------- Baseline ----------------

        base = self.base_graph.clone()
        base.metadata["variant"] = "baseline"
        variants.append(base)

        if not self.config.enabled:
            return variants

        # ---------------- Strategy selection ----------------

        strategy = self.config.mutation_strategy

        if strategy in {"conservative", "adaptive"}:
            variants.append(self._robustness_variant())
            variants.append(self._skeptical_variant())

        if strategy in {"exploratory", "adaptive"}:
            variants.append(self._latent_variant())
            variants.append(self._bias_variant())

        if strategy == "adaptive" and shock_score > 0.6:
            variants.append(self._counterfactual_variant())
            variants.append(self._temporal_variant())

        return variants[: self.config.max_variants]

    # ------------------------------------------------------------------
    # Mutation variants
    # ------------------------------------------------------------------

    def _robustness_variant(self) -> GraphStore:
        g = self.base_graph.clone()
        g.metadata["variant"] = "robustness"

        for e in list(g.get_edges()):
            if e.mutable:
                g.remove_edge(e.source, e.target)
                g.add_edge(e.decay(self.config.edge_decay_factor))

        return g

    def _counterfactual_variant(self) -> GraphStore:
        g = self.base_graph.clone()
        g.metadata["variant"] = "counterfactual"

        if not self.config.allow_edge_removal:
            return g

        for e in list(g.get_edges()):
            if e.mutable and e.causal_type == "correlational":
                g.remove_edge(e.source, e.target)

        return g

    def _skeptical_variant(self) -> GraphStore:
        g = self.base_graph.clone()
        g.metadata["variant"] = "skeptical"

        if not self.config.allow_edge_removal:
            return g

        for e in list(g.get_edges()):
            if e.mutable and e.confidence < 0.5:
                g.remove_edge(e.source, e.target)

        return g

    def _temporal_variant(self) -> GraphStore:
        g = self.base_graph.clone()
        g.metadata["variant"] = "temporal"

        if not self.config.allow_edge_removal:
            return g

        for e in list(g.get_edges()):
            if e.mutable and e.end_time is not None:
                g.remove_edge(e.source, e.target)

        return g

    def _latent_variant(self) -> GraphStore:
        g = self.base_graph.clone()
        g.metadata["variant"] = "latent"

        if not self.config.allow_hypothetical_edges:
            return g

        for e in list(g.get_edges()):
            if e.mutable and e.causal_type == "hypothetical" and e.weight < 0.5:
                g.remove_edge(e.source, e.target)
                g.add_edge(
                    Edge(
                        id=e.id,
                        source=e.source,
                        target=e.target,
                        relation=e.relation,
                        weight=min(e.weight * 1.5, 1.0),
                        confidence=e.confidence,
                        start_time=e.start_time,
                        end_time=e.end_time,
                        causal_type=e.causal_type,
                        mutable=e.mutable,
                        provenance=e.provenance,
                        tags=e.tags,
                    )
                )

        return g

    def _bias_variant(self) -> GraphStore:
        biased = BiasApplier(RiskAverseBias()).apply(self.base_graph)
        biased.metadata["variant"] = "bias:risk_averse"
        return biased
