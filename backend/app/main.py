from fastapi import FastAPI
from contextlib import asynccontextmanager

from backend.app.config import AppConfig
from backend.app.api.routes_query import router as query_router
from backend.app.api.routes_graph import router as graph_router
from backend.app.dependencies import (
    get_base_graph,
    get_memory,
    get_generator,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifecycle hooks.

    Ensures expensive dependencies are initialized
    once at startup and released cleanly at shutdown.
    """
    # Force initialization
    get_base_graph()
    get_memory()
    get_generator()

    yield

    # Future cleanup hooks go here
    # (GPU memory, tracing, etc.)


def create_app(config: AppConfig) -> FastAPI:
    app = FastAPI(
        title=config.app_name,
        lifespan=lifespan,
    )

    app.include_router(
        query_router,
        prefix=f"{config.api_prefix}/query",
        tags=["query"],
    )

    app.include_router(
        graph_router,
        prefix=f"{config.api_prefix}/graph",
        tags=["graph"],
    )

    return app


config = AppConfig()
app = create_app(config)
