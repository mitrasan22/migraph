from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Literal
from uuid import uuid4


@dataclass(frozen=True)
class Node:
    """
    Semantic entity in the knowledge graph.
    """

    id: str
    label: str
    type: str
    attributes: Dict[str, Any]

    @staticmethod
    def create(
        label: str,
        type: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> "Node":
        return Node(
            id=str(uuid4()),
            label=label,
            type=type,
            attributes=attributes or {},
        )


@dataclass(frozen=True)
class Edge:
    """
    Directed relationship encoding causal, temporal, and epistemic semantics.
    """

    id: str

    source: str
    target: str
    relation: str

    weight: float
    confidence: float

    start_time: Optional[str]
    end_time: Optional[str]

    causal_type: Literal["causal", "correlational", "hypothetical"]

    mutable: bool = True
    provenance: str = "observed"  # observed | inferred | hypothetical
    tags: Optional[Dict[str, Any]] = None

    @staticmethod
    def create(
        source: str,
        target: str,
        relation: str,
        weight: float,
        confidence: float,
        causal_type: Literal["causal", "correlational", "hypothetical"],
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        mutable: bool = True,
        provenance: str = "observed",
        tags: Optional[Dict[str, Any]] = None,
    ) -> "Edge":
        return Edge(
            id=str(uuid4()),
            source=source,
            target=target,
            relation=relation,
            weight=weight,
            confidence=confidence,
            start_time=start_time,
            end_time=end_time,
            causal_type=causal_type,
            mutable=mutable,
            provenance=provenance,
            tags=tags,
        )

    def decay(self, factor: float) -> "Edge":
        return Edge(
            id=self.id,
            source=self.source,
            target=self.target,
            relation=self.relation,
            weight=self.weight * factor,
            confidence=self.confidence,
            start_time=self.start_time,
            end_time=self.end_time,
            causal_type=self.causal_type,
            mutable=self.mutable,
            provenance=self.provenance,
            tags=self.tags,
        )
