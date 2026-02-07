from fastapi import APIRouter, Depends

from backend.app.api.schemas import GraphStatsResponse, GraphExportResponse, GraphNode, GraphEdge
import numpy as np
from backend.app.dependencies import get_base_graph

router = APIRouter()


@router.get("/stats", response_model=GraphStatsResponse)
def graph_stats(graph=Depends(get_base_graph)):
    return GraphStatsResponse(
        nodes=graph.node_count(),
        edges=graph.edge_count(),
        metadata=graph.metadata,
    )


@router.get("/export", response_model=GraphExportResponse)
def graph_export(graph=Depends(get_base_graph)):
    def _to_json_safe(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {k: _to_json_safe(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_to_json_safe(v) for v in value]
        return value

    nodes = []

    for node_id, data in graph._graph.nodes(data=True):
        payload = data.get("data")
        if payload is not None:
            nodes.append(
                GraphNode(
                    id=payload.id,
                    label=payload.label,
                    type=payload.type,
                    attributes=_to_json_safe(payload.attributes),
                )
            )
        else:
            nodes.append(
                GraphNode(
                    id=str(node_id),
                    label=str(node_id),
                    type="unknown",
                    attributes={},
                )
            )

    edges = []
    for _, _, data in graph._graph.edges(data=True):
        payload = data.get("data")
        if payload is None:
            continue
        edges.append(
            GraphEdge(
                source=payload.source,
                target=payload.target,
                relation=payload.relation,
                weight=payload.weight,
                confidence=payload.confidence,
                causal_type=payload.causal_type,
                provenance=payload.provenance,
            )
        )

    return GraphExportResponse(nodes=nodes, edges=edges)
