"""FastAPI REST API for DiagRAG.

Provides /ask and /health endpoints backed by a global RAGChain instance.
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, status
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.config_loader import (
    get_embedding_config,
    get_llm_config,
    get_milvus_config,
    get_retrieval_config,
)
from src.embedding_client import DashScopeEmbeddingClient
from src.llm_client import DashScopeLLMClient
from src.milvus_client import MilvusClient
from src.rag_chain import RAGChain, RAGChainError

logger = logging.getLogger(__name__)


class MilvusConnectionError(Exception):
    """Raised when Milvus cannot be reached after all retry attempts."""


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

rag_chain: RAGChain | None = None


def _build_rag_chain() -> RAGChain:
    """Instantiate all sub-clients and wire them into a RAGChain."""
    embed_cfg = get_embedding_config()
    milvus_cfg = get_milvus_config()
    llm_cfg = get_llm_config()
    retrieval_cfg = get_retrieval_config()

    embed_client = DashScopeEmbeddingClient(
        api_key=embed_cfg.get("dashscope_api_key"),
        model_name=embed_cfg.get("model", "text-embedding-v1"),
    )

    # Milvus gRPC 服务启动较慢，重试连接直到成功
    max_retries, backoff, elapsed = 10, 5.0, 0.0
    for attempt in range(max_retries):
        try:
            milvus_client = MilvusClient(
                host=milvus_cfg.get("host", "localhost"),
                port=int(milvus_cfg.get("port", 19530)),
                collection_name=milvus_cfg.get("collection_name", "medical_chunks"),
                vector_dim=int(milvus_cfg.get("vector_dim", 1536)),
            )
            milvus_client.connect()
            logger.info(
                "Milvus connected successfully after %.1fs (attempt %d/%d).",
                elapsed, attempt + 1, max_retries,
            )
            break
        except Exception as e:
            logger.warning(
                "Milvus 连接失败 (attempt %d/%d): %s，%.1fs 后重试 ...",
                attempt + 1, max_retries, e, backoff,
            )
            if attempt < max_retries - 1:
                time.sleep(backoff)
                elapsed += backoff
                backoff = min(backoff * 1.5, 30.0)
            else:
                raise MilvusConnectionError(
                    f"Milvus 连接在 {max_retries} 次重试后仍未成功，最后一次错误: {e}"
                ) from e

    llm_client = DashScopeLLMClient(
        api_key=llm_cfg.get("dashscope_api_key"),
        model_name=llm_cfg.get("model", "qwen-max"),
        temperature=float(llm_cfg.get("temperature", 0.1)),
        max_tokens=int(llm_cfg.get("max_tokens", 1500)),
    )

    return RAGChain(
        embedding_client=embed_client,
        milvus_client=milvus_client,
        llm_client=llm_client,
        top_k=retrieval_cfg.get("top_k", 5),
    )


class AskRequest(BaseModel):
    """Request body for the /ask endpoint."""

    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=50)


class SourceItem(BaseModel):
    """A single citation returned alongside an answer."""

    text: str
    metadata: dict[str, Any]
    score: float | None


class AskResponse(BaseModel):
    """Response body for the /ask endpoint."""

    answer: str
    sources: list[SourceItem]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise the global RAGChain when the server starts."""
    global rag_chain
    logger.info("Starting DiagRAG API, initialising RAG chain ...")
    try:
        rag_chain = _build_rag_chain()
        logger.info("RAG chain initialised successfully.")
    except Exception as exc:
        logger.error("Failed to initialise RAG chain: %s", exc)
        rag_chain = None
    yield
    rag_chain = None
    logger.info("DiagRAG API shut down.")


app = FastAPI(
    title="DiagRAG API",
    description="Medical diagnostic RAG system powered by Milvus + DashScope Qwen",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Liveness probe — always returns 200 if the process is running."""
    return {"status": "healthy"}


@app.post(
    "/ask",
    response_model=AskResponse,
    responses={503: {"description": "RAG chain not available"}},
)
def ask_endpoint(body: AskRequest) -> AskResponse:
    """Answer a medical question using the RAG pipeline."""
    if rag_chain is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG chain is not initialised. Check server logs.",
        )

    try:
        result = rag_chain.answer(body.question)
    except RAGChainError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG pipeline error: {exc}",
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error in /ask")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {exc}",
        ) from exc

    sources = [
        SourceItem(
            text=src.get("text", "")[:300],
            metadata=src.get("metadata", {}),
            score=src.get("score"),
        )
        for src in result.get("sources", [])
    ]

    return AskResponse(answer=result["answer"], sources=sources)


def _sse_format(event: str, data: Any) -> str:
    """Format a Server-Sent Event line."""
    json_data = json.dumps(jsonable_encoder(data), ensure_ascii=False)
    return f"event: {event}\ndata: {json_data}\n\n"


@app.post(
    "/ask/stream",
    responses={503: {"description": "RAG chain not available"}},
)
def ask_stream_endpoint(body: AskRequest):
    """Stream the LLM answer as SSE while also returning sources."""
    if rag_chain is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG chain is not initialised. Check server logs.",
        )

    try:
        token_gen, sources = rag_chain.stream_answer(body.question)
    except RAGChainError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG pipeline error: {exc}",
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error in /ask/stream")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {exc}",
        ) from exc

    # First send sources so the frontend can display them immediately
    source_items = [
        {
            "text": src.get("text", "")[:300],
            "metadata": src.get("metadata", {}),
            "score": src.get("score"),
        }
        for src in sources
    ]

    def generate():
        yield _sse_format("sources", source_items)
        for token in token_gen:
            yield _sse_format("token", token)
        yield _sse_format("done", {})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )
