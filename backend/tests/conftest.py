from __future__ import annotations

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from contextlib import asynccontextmanager

from backend.app.main import create_app
from backend.app.config import AppConfig
from backend.app.dependencies import get_graphrag_service
from backend.app.services.graphrag_service import GraphRAGService

from migraph.graph.graph_store import GraphStore
from migraph.memory.episodic import EpisodicMemory
from migraph.rag.generator import Generator
from migraph.embeddings.encoder import EmbeddingEncoder


class DummyEncoder(EmbeddingEncoder):
    def __init__(self, dimension: int = 8) -> None:
        super().__init__(dimension=dimension)

    def _encode_one(self, text: str) -> np.ndarray:
        return np.ones(self.dimension, dtype=float)


class DummyGenerator(Generator):
    def generate(
        self,
        query: str,
        context: dict,
        *,
        uncertainty: dict,
        bias: str,
        shock: dict,
    ) -> str:
        return "dummy answer"

    def stream_generate(
        self,
        query: str,
        context: dict,
        *,
        uncertainty: dict,
        bias: str,
        shock: dict,
    ):
        for chunk in ["dummy ", "answer"]:
            yield chunk


@pytest.fixture()
def memory() -> EpisodicMemory:
    return EpisodicMemory()


@pytest.fixture()
def client(memory: EpisodicMemory):
    @asynccontextmanager
    async def _no_lifespan(_: FastAPI):
        yield

    app = create_app(AppConfig())
    app.router.lifespan_context = _no_lifespan

    def _service_override() -> GraphRAGService:
        config = AppConfig()
        return GraphRAGService(
            base_graph=GraphStore(),
            memory=memory,
            generator=DummyGenerator(),
            embedding_encoder=DummyEncoder(),
            config=config.migraph,
        )

    app.dependency_overrides[get_graphrag_service] = _service_override
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
