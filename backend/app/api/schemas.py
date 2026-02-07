from typing import List, Dict, Any, Optional
from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str
    entities: List[str]
    bias: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    confidence: float
    uncertainty: Dict[str, Any]
    shock: Dict[str, Any]
    diagnostics: Dict[str, Any] | None = None


class GraphStatsResponse(BaseModel):
    nodes: int
    edges: int
    metadata: Dict[str, Any]


class GraphNode(BaseModel):
    id: str
    label: str
    type: str
    attributes: Dict[str, Any]


class GraphEdge(BaseModel):
    source: str
    target: str
    relation: str
    weight: float
    confidence: float
    causal_type: str
    provenance: str


class GraphExportResponse(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]


class EpisodeSummary(BaseModel):
    id: str
    timestamp: str
    query: str
    entities: List[str]
    answer: str
    confidence: float
    shock_overall: float
    bias: str
