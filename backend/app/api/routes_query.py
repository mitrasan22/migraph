import json
from dataclasses import asdict, is_dataclass
import numpy as np
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from backend.app.api.schemas import QueryRequest, QueryResponse, EpisodeSummary
from backend.app.dependencies import get_graphrag_service, get_memory
from backend.app.services.graphrag_service import GraphRAGService

router = APIRouter()


@router.post("/", response_model=QueryResponse)
def query(
    request: QueryRequest,
    service: GraphRAGService = Depends(get_graphrag_service),
):
    return service.run(
        query=request.query,
        entities=request.entities,
        bias=request.bias,
    )


@router.post("/stream")
def query_stream(
    request: QueryRequest,
    service: GraphRAGService = Depends(get_graphrag_service),
):
    def _to_json_safe(value):
        if is_dataclass(value):
            return _to_json_safe(asdict(value))
        if hasattr(value, "model_dump"):
            return _to_json_safe(value.model_dump())
        if hasattr(value, "dict") and callable(value.dict):
            return _to_json_safe(value.dict())
        if hasattr(value, "__dict__"):
            return _to_json_safe(vars(value))
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {k: _to_json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [_to_json_safe(v) for v in value]
        if hasattr(value, "isoformat"):
            return value.isoformat()
        return value

    def event_stream():
        for event in service.run_stream(
            query=request.query,
            entities=request.entities,
            bias=request.bias,
        ):
            if event["type"] == "chunk":
                yield f"event: chunk\ndata: {event['data']}\n\n"
            elif event["type"] == "metadata":
                payload = json.dumps(_to_json_safe(event["data"]))
                yield f"event: metadata\ndata: {payload}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/history", response_model=list[EpisodeSummary])
def query_history(memory=Depends(get_memory), limit: int = 25):
    episodes = memory.all()[-limit:]
    return [
        EpisodeSummary(
            id=e.id,
            timestamp=e.timestamp.isoformat(),
            query=e.query,
            entities=e.entities,
            answer=e.answer,
            confidence=e.confidence,
            shock_overall=e.shock.get("overall", 0.0),
            bias=e.bias,
        )
        for e in episodes
    ]
