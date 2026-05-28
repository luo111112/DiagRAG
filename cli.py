#!/usr/bin/env python
"""Interactive CLI for DiagRAG medical Q&A.

Usage:
    python cli.py
"""

from __future__ import annotations

import logging
import os
import sys
import warnings
from typing import Any

# Suppress PyMilvus ORM deprecation warnings (ORM API removed in 3.1)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pymilvus")

# Load .env before any other project imports
env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k, v)

from src.config_loader import load_config
from src.embedding_client import DashScopeEmbeddingClient
from src.llm_client import DashScopeLLMClient
from src.milvus_client import MilvusClient
from src.rag_chain import RAGChain, RAGChainError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _print_welcome() -> None:
    border = "=" * 60
    print(f"\n{border}")
    print("  DiagRAG 医学问答交互系统")
    print("  输入 'exit' 退出程序")
    print(f"{border}\n")


def _print_sources(sources: list[dict[str, Any]]) -> None:
    """Print formatted citations."""
    print("\n--- 引用来源 ---")
    if not sources:
        print("  （无）")
        return
    for i, src in enumerate(sources, 1):
        meta = src.get("metadata") or {}
        doc_name = meta.get("source", meta.get("file_name", "未知文档"))
        page = meta.get("page", "-")
        score = src.get("score")
        score_str = f"{score:.4f}" if score is not None else "N/A"
        preview = src.get("text", "")[:80]
        print(f"  [{i}] {doc_name} | 页码: {page} | 相似度: {score_str}")
        print(f"      {preview}...")


def _build_chain() -> RAGChain:
    """Initialize all clients and build the RAG chain."""
    from src.config_loader import get_milvus_config

    logger.info("Loading config...")
    cfg = load_config()
    milvus_cfg = get_milvus_config()

    logger.info("Initializing embedding client...")
    embedding_client = DashScopeEmbeddingClient()

    logger.info("Initializing Milvus client...")
    milvus_client = MilvusClient(
        host=milvus_cfg["host"],
        port=int(milvus_cfg["port"]),
        collection_name=milvus_cfg["collection_name"],
        vector_dim=int(milvus_cfg["vector_dim"]),
    )
    milvus_client.connect()

    logger.info("Initializing LLM client...")
    llm_client = DashScopeLLMClient()

    logger.info("Building RAG chain...")
    chain = RAGChain(
        embedding_client=embedding_client,
        milvus_client=milvus_client,
        llm_client=llm_client,
    )
    logger.info("RAG chain ready.")
    return chain


def main() -> None:
    _print_welcome()

    try:
        chain = _build_chain()
    except Exception as e:
        logger.error("Failed to initialize RAG chain: %s", e)
        print(f"\n[错误] 初始化失败: {e}")
        sys.exit(1)

    print("请输入您的医学问题（或输入 'exit' 退出）：\n")

    while True:
        try:
            question = input("【问题】> ").strip()
        except KeyboardInterrupt:
            print("\n\n收到中断信号，退出程序。")
            break
        except EOFError:
            print("\n\n输入流已关闭，退出程序。")
            break

        if not question:
            print("（请输入问题，或输入 'exit' 退出）\n")
            continue

        if question.lower() in ("exit", "quit", "q"):
            print("再见！")
            break

        print("\n正在检索并生成回答，请稍候...\n")

        try:
            result = chain.answer(question)
            answer = result.get("answer", "")
            sources = result.get("sources", [])

            print("【回答】")
            print(answer)
            _print_sources(sources)
            print()

        except RAGChainError as e:
            logger.error("RAG chain error: %s", e)
            print(f"\n[错误] 回答生成失败: {e}\n")
        except Exception as e:
            logger.exception("Unexpected error during answer generation")
            print(f"\n[错误] 发生未知错误: {e}\n")


if __name__ == "__main__":
    main()
