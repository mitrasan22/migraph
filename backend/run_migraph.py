import json
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.config import AppConfig  # noqa: E402
from backend.app.dependencies import (  # noqa: E402
    get_base_graph,
    get_embedding_encoder,
    get_generator,
    get_memory,
)
from backend.app.services.graphrag_service import GraphRAGService  # noqa: E402


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger("migraph.run")
    start = time.perf_counter()
    config = AppConfig()

    service = GraphRAGService(
        base_graph=get_base_graph(),
        memory=get_memory(),
        generator=get_generator(),
        embedding_encoder=get_embedding_encoder(),
        config=config.migraph,
        embeddings_path=Path(config.embeddings_path),
    )

    def _to_json_safe(value):
        try:
            import numpy as np
        except Exception:
            np = None
        if np is not None and isinstance(value, np.ndarray):
            return value.tolist()
        if np is not None and isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {k: _to_json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [_to_json_safe(v) for v in value]
        if hasattr(value, "isoformat"):
            return value.isoformat()
        try:
            json.dumps(value)
            return value
        except Exception:
            return str(value)

    def run_query(label: str, q: str) -> None:
        result = service.run(
            query=q,
            entities=[],
            bias=None,
        )
        elapsed = time.perf_counter() - start
        logger.info("[%s] result ready in %.2fs", label, elapsed)
        logger.info(json.dumps(_to_json_safe(result), indent=2))
        shock = (result or {}).get("shock") or {}
        logger.info(
            "[%s] shock overall=%.4f",
            label,
            float(shock.get("overall", 0.0) or 0.0),
        )

        diagnostics = (result or {}).get("diagnostics") or {}
        dom_entities = diagnostics.get("dominant_entities") or []
        dom_edges = diagnostics.get("dominant_edges") or []
        if not dom_entities or not dom_edges:
            logger.warning(
                "[%s] dominant_entities or dominant_edges is empty.",
                label,
            )

    # Shock-aware test: compare a neutral query vs a novelty-heavy query.
    run_query(
        "baseline",
        "How India's unemployment decreased GDP and when did it recovered again?",
    )
    run_query(
        "novel",
        "Explain the economic impact of Martian dust storms on India's GDP recovery timeline.",
    )


if __name__ == "__main__":
    main()
