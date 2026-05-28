"""scripts/build_milvus.py

Build the Milvus vector collection from processed chunks.

Usage:
    python scripts/build_milvus.py
    python scripts/build_milvus.py --chunks-file data/processed/chunks.json
    python scripts/build_milvus.py --rebuild

Flags:
    --chunks-file  Path to the chunks JSON file (default: data/processed/chunks.json).
    --rebuild     Drop the collection if it exists, then recreate and insert fresh.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

# Project root is the directory containing this script's parent (scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import get_embedding_config, get_milvus_config, load_config
from src.embedding_client import DashScopeEmbeddingClient
from src.milvus_client import MilvusClient, _compute_idf, compute_bm25_sparse_vector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _load_chunks(chunks_file: Path) -> list[dict[str, Any]]:
    """Load and return the list of chunk dicts from a JSON file."""
    if not chunks_file.is_file():
        raise FileNotFoundError(f"Chunks file not found: {chunks_file}")
    with open(chunks_file, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected chunks file to contain a JSON list, got {type(data).__name__}")
    logger.info("Loaded %d chunks from '%s'", len(data), chunks_file.name)
    return data


def _batch_embed(
    client: DashScopeEmbeddingClient,
    texts: list[str],
    batch_size: int = 100,
) -> list[list[float]]:
    """Generate embedding vectors in batches to avoid API timeouts."""
    vectors: list[list[float]] = []
    total = len(texts)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        logger.info("  Embedding batch %d-%d / %d ...", start + 1, end, total)
        batch_vectors = client.embed_documents(texts[start:end])
        vectors.extend(batch_vectors)
    return vectors


# ----------------------------------------------------------------------
# Main build logic
# ----------------------------------------------------------------------

def build(
    chunks_file: Path,
    rebuild: bool = False,
    milvus_host: str | None = None,
    milvus_port: int | None = None,
    collection_name: str | None = None,
    vector_dim: int | None = None,
) -> None:
    """Build (or rebuild) the Milvus collection from chunk data."""
    # 1. Load config
    config = load_config()
    milvus_cfg = config.get("milvus", {})
    embedding_cfg = config.get("embedding", {})

    host = milvus_host or milvus_cfg.get("host", "localhost")
    port = milvus_port or int(milvus_cfg.get("port", 19530))
    collection = collection_name or milvus_cfg.get("collection_name", "medical_chunks")
    dim = vector_dim or int(milvus_cfg.get("vector_dim", 1536))

    # 2. Load chunks
    chunks = _load_chunks(chunks_file)
    if not chunks:
        logger.warning("No chunks to insert. Exiting.")
        return

    texts = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]

    # 3. Init clients
    logger.info("Initializing DashScopeEmbeddingClient ...")
    embed_client = DashScopeEmbeddingClient(
        api_key=embedding_cfg.get("dashscope_api_key"),
        model_name=embedding_cfg.get("model", "text-embedding-v1"),
    )

    logger.info("Initializing MilvusClient (host=%s, port=%d, collection=%s) ...", host, port, collection)
    milvus_client = MilvusClient(
        host=host,
        port=port,
        collection_name=collection,
        vector_dim=dim,
    )

    with milvus_client:
        # 4. Collection management
        if rebuild:
            logger.info("[--rebuild] Dropping collection '%s' if it exists ...", collection)
            milvus_client.drop_collection()
            logger.info("Collection dropped (or did not exist).")

        if milvus_client.collection_exists():
            logger.info("Collection '%s' already exists.", collection)
        else:
            logger.info("Creating collection '%s' (dim=%d) ...", collection, dim)
            milvus_client.create_collection()
            logger.info("Collection created successfully.")

        # 5. Embed chunks in batches
        total = len(texts)
        logger.info("Generating embeddings for %d chunks (batch_size=100) ...", total)
        t0 = time.perf_counter()
        vectors = _batch_embed(embed_client, texts, batch_size=100)
        embed_elapsed = time.perf_counter() - t0
        logger.info("Embedding done: %d vectors in %.2fs", len(vectors), embed_elapsed)

        if len(vectors) != total:
            raise RuntimeError(
                f"Embedding count mismatch: expected {total}, got {len(vectors)}"
            )

        # 5b. 计算 BM25 稀疏向量
        logger.info("Computing BM25 sparse vectors for %d chunks ...", total)
        t_bm25 = time.perf_counter()
        idf_dict = _compute_idf(texts)
        sparse_vectors = compute_bm25_sparse_vector(texts, idf_dict=idf_dict, single_query=False)
        bm25_elapsed = time.perf_counter() - t_bm25
        logger.info("BM25 sparse vectors computed in %.2fs (%d docs)", bm25_elapsed, len(sparse_vectors))

        # 6. Insert into Milvus
        logger.info("Inserting %d records into Milvus ...", total)
        t1 = time.perf_counter()
        # Milvus insert_chunks expects entities in field order: vector, sparse_vector, text, metadata
        primary_keys = milvus_client.insert_chunks(
            texts=texts,
            metadatas=metadatas,
            vectors=vectors,
            sparse_vectors=sparse_vectors,
        )
        insert_elapsed = time.perf_counter() - t1

        # 7. Stats
        logger.info("=" * 50)
        logger.info("  Collection : %s", collection)
        logger.info("  Total inserted : %d", len(primary_keys))
        logger.info("  Embedding time  : %.2fs", embed_elapsed)
        logger.info("  BM25 time      : %.2fs", bm25_elapsed)
        logger.info("  Insert time     : %.2fs", insert_elapsed)
        logger.info("  Total time      : %.2fs", embed_elapsed + bm25_elapsed + insert_elapsed)
        logger.info("=" * 50)

        # Reload to confirm final entity count
        milvus_client._collection = None  # force re-fetch
        stats = milvus_client.get_collection_stats()
        logger.info(
            "Milvus collection stats after insert: num_entities=%d",
            stats.get("num_entities", "unknown"),
        )


# ----------------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------------

def _resolve_default_chunks() -> Path:
    """Return the default chunks file path relative to project root."""
    return PROJECT_ROOT / "data" / "processed" / "chunks.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the Milvus vector collection from processed chunks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--chunks-file",
        type=Path,
        default=None,
        help="Path to the chunks JSON file (default: data/processed/chunks.json)",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Drop the collection if it exists, then recreate and insert fresh",
    )
    parser.add_argument(
        "--host",
        dest="milvus_host",
        default=None,
        help="Milvus host (overrides config.yml)",
    )
    parser.add_argument(
        "--port",
        dest="milvus_port",
        type=int,
        default=None,
        help="Milvus port (overrides config.yml)",
    )
    parser.add_argument(
        "--collection",
        dest="collection_name",
        default=None,
        help="Collection name (overrides config.yml)",
    )
    parser.add_argument(
        "--dim",
        dest="vector_dim",
        type=int,
        default=None,
        help="Vector dimension (overrides config.yml)",
    )
    args = parser.parse_args()

    chunks_file = args.chunks_file or _resolve_default_chunks()

    try:
        build(
            chunks_file=chunks_file,
            rebuild=args.rebuild,
            milvus_host=args.milvus_host,
            milvus_port=args.milvus_port,
            collection_name=args.collection_name,
            vector_dim=args.vector_dim,
        )
        logger.info("Build completed successfully.")
    except Exception as e:
        logger.error("Build failed: %s", e)
        raise


if __name__ == "__main__":
    main()
