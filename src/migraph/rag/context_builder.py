from __future__ import annotations

from typing import Dict, Any, List

from migraph.graph.graph_store import GraphStore
from migraph.rag.retriever import RetrievedContext


class ContextBuilder:
    """
    Builds a structured, epistemically bounded context
    from a retrieved subgraph for GraphRAG generation.

    This is NOT a serializer.
    This is a reasoning contract for the generator.
    """

    def build(
        self,
        graph: GraphStore,
        retrieved: RetrievedContext,
    ) -> Dict[str, Any]:

        max_entities = 20
        max_relations = 20

        entities: List[Dict[str, Any]] = []
        relations: List[Dict[str, Any]] = []

        node_id_set = set(retrieved.node_ids)

        for node_id in retrieved.node_ids[:max_entities]:
            try:
                node = graph.get_node(node_id)
            except KeyError:
                continue

            entities.append(
                {
                    "id": node.id,
                    "label": node.label,
                    "type": node.type,
                    "attributes": node.attributes,
                }
            )

        for edge in graph.get_edges_between(node_id_set)[:max_relations]:
            relations.append(
                {
                    "source": edge.source,
                    "target": edge.target,
                    "relation": edge.relation,
                    "causal_type": edge.causal_type,
                    "strength": {
                        "weight": edge.weight,
                        "confidence": edge.confidence,
                    },
                    "provenance": edge.provenance,
                }
            )

        return {
            "graph_variant": retrieved.graph_variant,
            "entities": entities,
            "relations": relations,
            "retrieval_stats": retrieved.metadata,
            "epistemic_notes": {
                "causal_edges": [r for r in relations if r["causal_type"] == "causal"],
                "correlational_edges": [
                    r for r in relations if r["causal_type"] == "correlational"
                ],
                "weak_evidence_edges": [
                    r for r in relations if r["strength"]["confidence"] < 0.5
                ],
            },
        }
