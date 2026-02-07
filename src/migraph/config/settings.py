from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# ---------------------------------------------------------------------
# Graph traversal & structural reasoning
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class GraphConfig:
    """
    Controls how the knowledge graph is traversed and interpreted
    during retrieval and reasoning.
    """

    max_hops: int
    allow_weak_edges: bool
    min_edge_confidence: float
    temporal_reasoning: bool
    retrieval_timeout_ms: float = 0.0


# ---------------------------------------------------------------------
# Shock / novelty detection
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ShockConfig:
    """
    Controls detection of novelty, instability, and out-of-distribution
    reasoning conditions.
    """

    enabled: bool
    shock_threshold: float
    novelty_window: int
    escalation_policy: Literal["expand", "mutate", "halt"]


# ---------------------------------------------------------------------
# Inference-time graph mutation
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class MutationConfig:
    """
    Controls how and when the graph is modified during inference.
    """

    enabled: bool
    max_variants: int
    edge_decay_factor: float
    allow_edge_removal: bool
    allow_hypothetical_edges: bool
    mutation_strategy: Literal["conservative", "exploratory", "adaptive"]


# ---------------------------------------------------------------------
# Uncertainty & explanation stability
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class UncertaintyConfig:
    """
    Controls how explanation stability and epistemic uncertainty
    are computed across graph variants.
    """

    enabled: bool
    min_agreement_ratio: float
    entropy_threshold: float
    confidence_mode: Literal["stability", "entropy", "hybrid"]


# ---------------------------------------------------------------------
# Bias conditioning
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class BiasConfig:
    """
    Defines how cognitive or risk biases are applied to graph reasoning
    via edge re-weighting policies.
    """

    enabled: bool
    profile: Literal["neutral", "risk_averse", "optimistic", "skeptical"]
    amplification_factor: float


# ---------------------------------------------------------------------
# Answer guardrails
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class AnswerGuardConfig:
    """
    Controls when the system should refuse to answer due to
    weak or irrelevant graph evidence.
    """

    enabled: bool
    min_context_nodes: int
    min_context_edges: int
    min_retrieval_density: float
    min_query_coverage: float
    shock_threshold: float
    min_semantic_novelty: float
    min_token_novelty: float
    min_sentence_overlap: float
    stop_words: list[str]
    min_type_overlap: int


# ---------------------------------------------------------------------
# Root configuration object
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class MigraphConfig:
    """
    Root configuration object for migraph.

    This object is intended to be:
    - constructed explicitly
    - passed through all major subsystems
    - treated as immutable system policy
    """

    graph: GraphConfig
    shock: ShockConfig
    mutation: MutationConfig
    uncertainty: UncertaintyConfig
    bias: BiasConfig
    guard: AnswerGuardConfig
    entity_max_entities: int = 5
