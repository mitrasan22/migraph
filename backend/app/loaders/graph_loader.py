from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict, Any
import time
import logging

import pandas as pd

from migraph.graph.graph_store import GraphStore
from migraph.graph.graph_schema import Node, Edge
from migraph.embeddings.encoder import EmbeddingEncoder


def load_graph_from_processed(
    *,
    graph: GraphStore,
    processed_dir: Path,
    encoder: Optional[EmbeddingEncoder] = None,
    embed_edges: bool = True,
    persist_embeddings: bool = False,
    persist_embeddings_if_missing: bool = True,
    rebuild_embeddings_on_mismatch: bool = False,
) -> None:
    """
    Load nodes/edges from parquet files into the GraphStore.

    If an encoder is provided, warm the embedding cache by encoding
    node and edge text representations.
    """
    nodes_path = processed_dir / "nodes.parquet"
    edges_path = processed_dir / "edges.parquet"

    if not nodes_path.exists() or not edges_path.exists():
        return

    t0 = time.perf_counter()
    nodes_df = pd.read_parquet(nodes_path)
    t_nodes = time.perf_counter()
    edges_df = pd.read_parquet(edges_path)
    t_edges = time.perf_counter()
    logger = logging.getLogger("migraph.load_graph")
    logger.info(
        "read nodes=%s in %.3fs; read edges=%s in %.3fs",
        len(nodes_df),
        t_nodes - t0,
        len(edges_df),
        t_edges - t_nodes,
    )

    embeddings_rows: List[Dict[str, Any]] = []
    embeddings_path = processed_dir / "embeddings.parquet"
    should_encode = encoder is not None and not embeddings_path.exists()
    if encoder is not None and embeddings_path.exists():
        try:
            emb_df = pd.read_parquet(embeddings_path)
            has_kind = "kind" in emb_df.columns
            has_vector = "vector" in emb_df.columns
            has_id = "id" in emb_df.columns
            sample_vec = None
            if has_vector and len(emb_df) > 0:
                sample_vec = emb_df.iloc[0]["vector"]
            dim_ok = False
            if sample_vec is not None:
                probe = encoder.encode_one("dimension_check")
                dim_ok = len(probe) == len(sample_vec)

            if not (has_kind and has_vector and has_id and dim_ok):
                if rebuild_embeddings_on_mismatch:
                    logger.warning(
                        "embeddings mismatch; rebuilding (kind=%s, vector=%s, id=%s, dim_ok=%s)",
                        has_kind,
                        has_vector,
                        has_id,
                        dim_ok,
                    )
                    try:
                        embeddings_path.unlink()
                    except Exception:
                        pass
                    should_encode = True
                else:
                    logger.warning(
                        "embeddings mismatch; set REBUILD_EMBEDDINGS_ON_MISMATCH=true to rebuild"
                    )
            else:
                logger.info(
                    "embeddings found at %s; skipping encode",
                    embeddings_path,
                )
        except Exception:
            if rebuild_embeddings_on_mismatch:
                logger.warning("embeddings read failed; rebuilding")
                try:
                    embeddings_path.unlink()
                except Exception:
                    pass
                should_encode = True
            else:
                logger.warning(
                    "embeddings read failed; set REBUILD_EMBEDDINGS_ON_MISMATCH=true to rebuild"
                )

    t_build_nodes = time.perf_counter()
    for _, row in nodes_df.iterrows():
        node = Node(
            id=str(row["id"]),
            label=str(row["label"]),
            type=str(row["type"]),
            attributes=row.get("attributes", {}) or {},
        )
        graph.add_node(node)

        if should_encode:
            text = f"{node.label} {node.type} {node.attributes}"
            vec = encoder.encode_one(text)
            embeddings_rows.append(
                {
                    "id": node.id,
                    "kind": "node",
                    "vector": vec.tolist(),
                }
            )
    t_nodes_done = time.perf_counter()
    logger.info(
        "added nodes in %.3fs; embeddings buffered=%s",
        t_nodes_done - t_build_nodes,
        len(embeddings_rows),
    )

    existing_node_ids = set(str(n) for n in nodes_df["id"].tolist())

    t_build_edges = time.perf_counter()
    for _, row in edges_df.iterrows():
        edge = Edge.create(
            source=str(row["source"]),
            target=str(row["target"]),
            relation=str(row["relation"]),
            weight=float(row["weight"]),
            confidence=float(row["confidence"]),
            causal_type=str(row["causal_type"]),
            provenance=str(row.get("provenance", "observed")),
        )

        for missing_id in (edge.source, edge.target):
            if missing_id not in existing_node_ids:
                graph.add_node(
                    Node(
                        id=str(missing_id),
                        label=str(missing_id),
                        type="unknown",
                        attributes={},
                    )
                )
                existing_node_ids.add(missing_id)

        graph.add_edge(edge)

        if should_encode and embed_edges:
            text = f"{edge.relation} {edge.causal_type} {edge.provenance}"
            vec = encoder.encode_one(text)
            edge_id = f"{edge.source}|{edge.relation}|{edge.target}"
            embeddings_rows.append(
                {
                    "id": edge_id,
                    "kind": "edge",
                    "vector": vec.tolist(),
                }
            )
    t_edges_done = time.perf_counter()
    logger.info(
        "added edges in %.3fs; embeddings buffered=%s",
        t_edges_done - t_build_edges,
        len(embeddings_rows),
    )

    if persist_embeddings and embeddings_rows:
        if persist_embeddings_if_missing and embeddings_path.exists():
            logger.info("embeddings already exist at %s", embeddings_path)
            return

        t_emb = time.perf_counter()
        emb_df = pd.DataFrame(embeddings_rows)
        emb_df.to_parquet(embeddings_path, index=False)
        logger.info(
            "saved embeddings: %s rows -> %s",
            len(embeddings_rows),
            embeddings_path,
        )
        logger.info(
            "wrote embeddings in %.3fs",
            time.perf_counter() - t_emb,
        )
