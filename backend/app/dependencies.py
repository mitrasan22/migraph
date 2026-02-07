from functools import lru_cache
import logging
from pathlib import Path
import time

from migraph.graph.graph_store import GraphStore
from migraph.memory.episodic import EpisodicMemory
from migraph.rag.generator import LLMGenerator
from migraph.rag.hf_llama_backend import HuggingFaceLLaMABackend
from migraph.embeddings.hf_encoder import HuggingFaceEmbeddingEncoder

from backend.app.config import AppConfig
from backend.app.services.graphrag_service import GraphRAGService
from backend.app.loaders.graph_loader import load_graph_from_processed


@lru_cache
def get_config() -> AppConfig:
    return AppConfig()


@lru_cache
def get_base_graph() -> GraphStore:
    logger = logging.getLogger("migraph.startup")
    t0 = time.perf_counter()
    graph = GraphStore()
    graph.metadata["source"] = "backend"
    graph.metadata["loaded_from_processed"] = False

    processed_dir = Path("data") / "processed"
    if processed_dir.exists() and not graph.metadata["loaded_from_processed"]:
        try:
            t_enc = time.perf_counter()
            encoder = get_embedding_encoder()
            logger.info(
                "[startup] embedding encoder init in %.3fs",
                time.perf_counter() - t_enc,
            )
            t_load = time.perf_counter()
            config = get_config()
            load_graph_from_processed(
                graph=graph,
                processed_dir=processed_dir,
                encoder=encoder,
                embed_edges=config.embed_edges,
                persist_embeddings=config.persist_embeddings,
                persist_embeddings_if_missing=config.persist_embeddings_if_missing,
                rebuild_embeddings_on_mismatch=config.rebuild_embeddings_on_mismatch,
            )
            logger.info(
                "[startup] graph load in %.3fs",
                time.perf_counter() - t_load,
            )
            graph.metadata["loaded_from_processed"] = True
        except Exception as exc:
            graph.metadata["load_error"] = str(exc)
    logger.info("[startup] get_base_graph total %.3fs", time.perf_counter() - t0)
    return graph


@lru_cache
def get_memory() -> EpisodicMemory:
    return EpisodicMemory()

@lru_cache
def get_embedding_encoder() -> HuggingFaceEmbeddingEncoder:
    config = get_config()

    return HuggingFaceEmbeddingEncoder(
        model_name=config.embedding_model,
        device=config.embedding_device,
    )

@lru_cache
def get_generator() -> LLMGenerator:
    config = get_config()

    t0 = time.perf_counter()
    backend = HuggingFaceLLaMABackend(
        model_name=config.llm_model,
        hf_token=config.hf_token,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        repetition_penalty=config.repetition_penalty,
        no_repeat_ngram_size=config.no_repeat_ngram_size,
    )
    logging.getLogger("migraph.startup").info(
        "[startup] LLM backend init in %.3fs",
        time.perf_counter() - t0,
    )

    return LLMGenerator(backend)


@lru_cache
def get_graphrag_service() -> GraphRAGService:
    config = get_config()

    return GraphRAGService(
        base_graph=get_base_graph(),
        memory=get_memory(),
        generator=get_generator(),
        embedding_encoder=get_embedding_encoder(),
        config=config.migraph,
        embeddings_path=Path(config.embeddings_path),
    )
