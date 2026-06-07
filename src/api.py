"""FastAPI REST API for DiagRAG.

Provides /ask and /health endpoints backed by a global RAGChain instance.
Provides full multi-turn conversation memory support via /conversation/*.
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, status
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.config_loader import (
    get_conversation_config,
    get_embedding_config,
    get_llm_config,
    get_milvus_config,
    get_retrieval_config,
)
from src.embedding_client import DashScopeEmbeddingClient
from src.llm_client import DashScopeLLMClient
from src.milvus_client import MilvusClient
from src.rag_chain import RAGChain, RAGChainError
from src.conversation.kafka_producer import (
    send_message_event,
    send_summary_event,
    send_session_close_event,
)
from kafka.errors import NoBrokersAvailable

logger = logging.getLogger(__name__)


def _safe_kafka_send(fn, *args, **kwargs):
    """Call a Kafka producer function, swallowing NoBrokersAvailable.

    Kafka is an optional async pipeline — it must never crash the API.
    """
    try:
        return fn(*args, **kwargs)
    except NoBrokersAvailable as e:
        logger.warning("/chat: Kafka unavailable, skipping event: %s", e)
        return False


class MilvusConnectionError(Exception):
    """Raised when Milvus cannot be reached after all retry attempts."""


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

rag_chain: RAGChain | None = None


def _build_rag_chain() -> RAGChain:
    embed_cfg = get_embedding_config()
    milvus_cfg = get_milvus_config()
    llm_cfg = get_llm_config()
    retrieval_cfg = get_retrieval_config()

    embed_client = DashScopeEmbeddingClient(
        api_key=embed_cfg.get("dashscope_api_key"),
        model_name=embed_cfg.get("model", "text-embedding-v1"),
    )

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
            logger.info("Milvus connected after %.1fs (attempt %d/%d).", elapsed, attempt + 1, max_retries)
            break
        except Exception as e:
            logger.warning("Milvus 连接失败 (attempt %d/%d): %s", attempt + 1, max_retries, e)
            if attempt < max_retries - 1:
                time.sleep(backoff)
                elapsed += backoff
                backoff = min(backoff * 1.5, 30.0)
            else:
                raise MilvusConnectionError(
                    f"Milvus 连接在 {max_retries} 次重试后仍未成功: {e}"
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


# =============================================================================
# Pydantic models
# =============================================================================

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    session_id: str | None = Field(default=None, description="会话ID，不传则为单轮问答")


class SourceItem(BaseModel):
    text: str
    metadata: dict[str, Any]
    score: float | None


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    session_id: str | None = None
    cache_hit: bool = False
    cache_hit_from: str | None = None


class CreateSessionRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=64)
    title: str = Field(default="", max_length=255)


class CreateSessionResponse(BaseModel):
    session_id: str
    user_id: str
    title: str
    created_at: str


class SendMessageRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=4000)
    metadata: dict[str, Any] | None = Field(default=None)


class MessageItem(BaseModel):
    message_order: int
    role: str
    content: str
    created_at: str


class SendMessageResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    session_id: str
    message_order: int
    cache_hit: bool = False
    cache_hit_from: str | None = None


class HistoryResponse(BaseModel):
    session_id: str
    messages: list[MessageItem]
    summary: str | None


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    session_id: str | None = Field(default=None, description="传则续接会话，不传则自动创建新会话")
    user_id: str | None = Field(default=None, max_length=64, description="创建新会话时必填")
    top_k: int | None = Field(default=None, ge=1, le=50)


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    session_id: str
    message_order: int
    cache_hit: bool = False
    cache_hit_from: str | None = None


class CacheStatsResponse(BaseModel):
    total_requests: int
    cache_hits: int
    redis_hits: int
    milvus_hits: int
    context_filtered: int
    writes: int
    evictions: int
    last_eviction_at: str
    last_reset_at: str
    hit_rate: float
    cache_size: int
    enabled: bool


class CacheEvictRequest(BaseModel):
    n: int = Field(default=100, ge=1, le=10000)


class CacheEvictResponse(BaseModel):
    evicted_ids: list[int]
    evicted_count: int


class CacheClearResponse(BaseModel):
    deleted_count: int


def _sse_format(event: str, data: Any) -> str:
    json_data = json.dumps(jsonable_encoder(data), ensure_ascii=False)
    return f"event: {event}\ndata: {json_data}\n\n"


def _get_semantic_cache():
    if rag_chain is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG chain not initialised.",
        )
    semantic_cache = getattr(rag_chain, "semantic_cache", None)
    if semantic_cache is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Semantic cache is not configured.",
        )
    return semantic_cache


# =============================================================================
# Lifespan
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_chain
    logger.info("Starting DiagRAG API ...")

    # 1. Build RAG chain
    try:
        rag_chain = _build_rag_chain()
        logger.info("RAG chain initialised.")
    except Exception as exc:
        logger.error("Failed to initialise RAG chain: %s", exc)
        rag_chain = None

    # 2. Initialise MySQL schema
    try:
        from src.conversation.mysql_store import init_db
        init_db()
        logger.info("MySQL schema initialised.")
    except Exception as exc:
        logger.error("Failed to initialise MySQL: %s", exc)

    # 3. Start Kafka consumers
    try:
        from src.conversation.kafka_consumer import start_consumers
        consumer_llm = DashScopeLLMClient(
            api_key=get_llm_config().get("dashscope_api_key"),
            model_name=get_llm_config().get("model", "qwen-max"),
        )
        start_consumers(consumer_llm)
        logger.info("Kafka consumers started.")
    except Exception as exc:
        logger.warning("Kafka consumers failed to start: %s", exc)

    yield

    try:
        from src.conversation.kafka_consumer import stop_consumers
        from src.conversation.kafka_producer import flush_producer
        stop_consumers()
        flush_producer()
        logger.info("Kafka consumers stopped.")
    except Exception as exc:
        logger.warning("Error stopping Kafka consumers: %s", exc)

    rag_chain = None
    logger.info("DiagRAG API shut down.")


# =============================================================================
# App
# =============================================================================

app = FastAPI(
    title="DiagRAG API",
    description="Medical diagnostic RAG system with multi-turn conversation memory",
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
    return {"status": "healthy"}


# =============================================================================
# /ask — single turn
# =============================================================================

@app.post("/ask", response_model=AskResponse, responses={503: {"description": "RAG chain not available"}})
def ask_endpoint(body: AskRequest) -> AskResponse:
    if rag_chain is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="RAG chain not initialised.")
    try:
        result = rag_chain.answer(body.question, session_id=None)
    except RAGChainError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"RAG pipeline error: {exc}") from exc
    except Exception as exc:
        logger.exception("Unexpected error in /ask")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Unexpected error: {exc}") from exc
    sources = [
        SourceItem(text=src.get("text", "")[:300], metadata=src.get("metadata", {}), score=src.get("score"))
        for src in result.get("sources", [])
    ]
    return AskResponse(
        answer=result["answer"],
        sources=sources,
        session_id=body.session_id,
        cache_hit=result.get("cache_hit", False),
        cache_hit_from=result.get("cache_hit_from"),
    )


@app.post("/ask/stream", responses={503: {"description": "RAG chain not available"}})
def ask_stream_endpoint(body: AskRequest):
    if rag_chain is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="RAG chain not initialised.")
    try:
        token_gen, sources = rag_chain.stream_answer(body.question)
    except RAGChainError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"RAG pipeline error: {exc}") from exc
    except Exception as exc:
        logger.exception("Unexpected error in /ask/stream")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Unexpected error: {exc}") from exc
    source_items = [
        {"text": src.get("text", "")[:300], "metadata": src.get("metadata", {}), "score": src.get("score")}
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
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# =============================================================================
# /chat — multi-turn (convenience wrapper over /conversation/*)
# =============================================================================

@app.post("/chat", response_model=ChatResponse, responses={503: {"description": "RAG chain not available"}})
def chat_endpoint(body: ChatRequest) -> ChatResponse:
    """Unified multi-turn chat endpoint.

    If ``session_id`` is provided, the message is appended to the existing session.
    Otherwise, a new session is created (requires ``user_id``).

    Flow: write user msg → answer with RAGChain → write assistant msg → check summary trigger
    """
    if rag_chain is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="RAG chain not initialised.")

    try:
        from src.conversation.models import Message, MessageRole
        from src.conversation.mysql_store import (
            create_session, get_session, get_session_messages, get_latest_summary,
            save_message, audit_log,
        )
        from src.conversation.redis_store import (
            write_message, update_session_meta, get_recent_messages,
            warm_cache, get_current_message_order,
        )
        from src.conversation.kafka_producer import send_message_event, send_summary_event
        from src.conversation.summarizer import should_trigger_summary

        cfg_conv = get_conversation_config()
        ttl_days = int(cfg_conv.get("mysql_retention_days", 30))
        interval = int(cfg_conv.get("summary_interval_turns", 5))

        # -- Resolve or create session --
        if body.session_id:
            session = get_session(body.session_id)
            if session is None:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                    detail=f"Session {body.session_id} not found.")
        else:
            if not body.user_id:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                    detail="user_id is required when creating a new session.")
            session = create_session(user_id=body.user_id, title="", ttl_days=ttl_days)
            update_session_meta(
                session_id=session.id, user_id=session.user_id,
                status="active", summary_order=0,
            )
            logger.info("New session created via /chat: session=%s, user=%s",
                        session.id, session.user_id)

        session_id = session.id

        # -- Step 1: write user message to Redis --
        user_msg = Message(
            session_id=session_id, message_order=0, role=MessageRole.USER,
            content=body.question, metadata={"source": "chat_api"},
        )
        message_order = write_message(user_msg)
        user_msg.message_order = message_order
        update_session_meta(
            session_id=session_id, user_id=session.user_id,
            status="active", summary_order=session.summary_order,
        )
        send_message_event(session_id, user_msg)
        logger.info("/chat user message: session=%s, order=%d", session_id, message_order)
    except NoBrokersAvailable as kafka_warn:
        logger.warning("/chat: Kafka unavailable (user msg event skipped): %s", kafka_warn)

        # -- Step 2: warm Redis cache on miss --
        if not get_recent_messages(session_id):
            all_msgs = get_session_messages(session_id)
            if all_msgs:
                warm_cache(session_id, all_msgs)

        latest_summary = get_latest_summary(session_id)
        summary_text = latest_summary.summary_text if latest_summary else ""

        # -- Step 3: answer through RAGChain (includes semantic cache) --
        result = rag_chain.answer(
            question=body.question,
            session_id=session_id,
            session_summary=summary_text,
            user_id=session.user_id,
        )
        answer_text = result["answer"]
        cache_hit_flag = result.get("cache_hit", False)
        cache_hit_from = result.get("cache_hit_from")

        # -- Step 4: write assistant message to Redis + Kafka --
        assistant_msg = Message(
            session_id=session_id, message_order=0, role=MessageRole.ASSISTANT,
            content=answer_text, metadata={"question": body.question},
        )
        asst_order = write_message(assistant_msg)
        assistant_msg.message_order = asst_order
        send_message_event(session_id, assistant_msg)
        logger.info("/chat assistant reply: session=%s, order=%d, len=%d",
                    session_id, asst_order, len(answer_text))

        # -- Step 5: persist to MySQL --
        save_message(user_msg)
        save_message(assistant_msg)

        # -- Step 6: check summary trigger --
        current_order = get_current_message_order(session_id)
        if should_trigger_summary(current_order, session.summary_order, interval):
            msgs_for_summary = get_session_messages(session_id)
            new_msgs = [
                m for m in msgs_for_summary
                if m.message_order > session.summary_order * interval
            ]
            if new_msgs:
                send_summary_event(
                    session_id=session_id,
                    summary_order=session.summary_order + 1,
                    turns_start=session.summary_order * interval + 1,
                    turns_end=current_order,
                    previous_summary=summary_text or None,
                    messages_to_summarize=[
                        {"role": m.role.value, "content": m.content}
                        for m in new_msgs
                    ],
                )
                logger.info("/chat summary queued: session=%s, summary_order=%d",
                            session_id, session.summary_order + 1)

        audit_log(action="CHAT_MESSAGE", operator=session.user_id,
                  session_id=session_id, resource="message",
                  detail={"message_order": message_order})

        # -- Build sources response --
        sources = [
            SourceItem(
                text=(src.get("text", "")[:200] + ("..." if len(src.get("text", "")) > 200 else "")),
                metadata=src.get("metadata", {}),
                score=src.get("score"),
            )
            for src in result.get("sources", [])
        ]

        return ChatResponse(
            answer=answer_text,
            sources=sources,
            session_id=session_id,
            message_order=asst_order,
            cache_hit=cache_hit_flag,
            cache_hit_from=cache_hit_from,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error in /chat")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing chat: {exc}",
        ) from exc


@app.post("/chat/stream", responses={503: {"description": "RAG chain not available"}})
def chat_stream_endpoint(body: ChatRequest):
    """SSE streaming version of /chat for real-time multi-turn responses.

    Events emitted:
        - ``sources``: list of retrieved source chunks (once, before tokens)
        - ``token``: individual LLM tokens as they arrive
        - ``done``: empty sentinel when the stream ends
    """
    if rag_chain is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="RAG chain not initialised.")

    try:
        from src.conversation.models import Message, MessageRole
        from src.conversation.mysql_store import (
            create_session, get_session, get_session_messages, get_latest_summary,
            save_message, audit_log,
        )
        from src.conversation.redis_store import (
            write_message, update_session_meta, get_recent_messages,
            warm_cache, get_current_message_order,
        )
        from src.conversation.kafka_producer import send_message_event, send_summary_event
        from src.conversation.summarizer import should_trigger_summary
        from src.conversation.context_builder import assemble_stream_prompt

        cfg_conv = get_conversation_config()
        ttl_days = int(cfg_conv.get("mysql_retention_days", 30))
        interval = int(cfg_conv.get("summary_interval_turns", 5))

        # -- Resolve or create session --
        if body.session_id:
            session = get_session(body.session_id)
            if session is None:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                    detail=f"Session {body.session_id} not found.")
        else:
            if not body.user_id:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                    detail="user_id is required when creating a new session.")
            session = create_session(user_id=body.user_id, title="", ttl_days=ttl_days)
            update_session_meta(
                session_id=session.id, user_id=session.user_id,
                status="active", summary_order=0,
            )
            logger.info("New session created via /chat/stream: session=%s", session.id)

        session_id = session.id

        # -- Write user message --
        user_msg = Message(
            session_id=session_id, message_order=0, role=MessageRole.USER,
            content=body.question, metadata={"source": "chat_stream_api"},
        )
        message_order = write_message(user_msg)
        user_msg.message_order = message_order
        update_session_meta(
            session_id=session_id, user_id=session.user_id,
            status="active", summary_order=session.summary_order,
        )
        send_message_event(session_id, user_msg)

        # -- Warm cache on miss --
        if not get_recent_messages(session_id):
            all_msgs = get_session_messages(session_id)
            if all_msgs:
                warm_cache(session_id, all_msgs)

        # -- Retrieval --
        top_k = body.top_k if body.top_k else rag_chain.top_k
        try:
            query_vector = rag_chain.embedding_client.embed_text(body.question)
            chunks = rag_chain.milvus_client.search(query_vector=query_vector, top_k=top_k)
            context = "\n\n---\n\n".join(
                f"[文档{i + 1}]\n{c['text']}" for i, c in enumerate(chunks)
            )
        except Exception as e:
            logger.warning("/chat/stream retrieval failed: %s", e)
            chunks = []
            context = ""

        # -- Assemble prompt --
        latest_summary = get_latest_summary(session_id)
        summary_text = latest_summary.summary_text if latest_summary else None
        user_prompt, _, _ = assemble_stream_prompt(
            session_id=session_id, question=body.question, context=context,
        )

        source_items = [
            {
                "text": c["text"][:200] + ("..." if len(c["text"]) > 200 else ""),
                "metadata": c.get("metadata") or {},
                "score": c.get("score"),
            }
            for c in chunks
        ]

        # Use a mutable container so the inner generator can update answer_text
        answer_holder: list[str] = []

        def stream_with_postproc():
            answer_parts: list[str] = []

            try:
                for token in rag_chain.llm_client.stream_generate(
                    prompt=user_prompt, system_prompt=rag_chain.system_prompt,
                ):
                    answer_parts.append(token)
                    yield _sse_format("token", token)

                full_answer = "".join(answer_parts)
                answer_holder.append(full_answer)
            except Exception as e:
                yield _sse_format("error", str(e))
                return

            # After stream finishes: write assistant message
            assistant_msg = Message(
                session_id=session_id, message_order=0, role=MessageRole.ASSISTANT,
                content=full_answer, metadata={"question": body.question},
            )
            asst_order = write_message(assistant_msg)
            assistant_msg.message_order = asst_order
            send_message_event(session_id, assistant_msg)

            save_message(user_msg)
            save_message(assistant_msg)

            # Summary trigger
            current_order = get_current_message_order(session_id)
            if should_trigger_summary(current_order, session.summary_order, interval):
                msgs_for_summary = get_session_messages(session_id)
                new_msgs = [
                    m for m in msgs_for_summary
                    if m.message_order > session.summary_order * interval
                ]
                if new_msgs:
                    send_summary_event(
                        session_id=session_id,
                        summary_order=session.summary_order + 1,
                        turns_start=session.summary_order * interval + 1,
                        turns_end=current_order,
                        previous_summary=summary_text,
                        messages_to_summarize=[
                            {"role": m.role.value, "content": m.content}
                            for m in new_msgs
                        ],
                    )

            audit_log(action="CHAT_STREAM_MESSAGE", operator=session.user_id,
                      session_id=session_id, resource="message",
                      detail={"message_order": message_order})

            yield _sse_format("done", {
                "session_id": session_id,
                "message_order": asst_order,
            })

        def generate():
            yield _sse_format("sources", source_items)
            yield from stream_with_postproc()

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error in /chat/stream")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in chat stream: {exc}",
        ) from exc


# =============================================================================
# Semantic cache endpoints
# =============================================================================

@app.get("/cache/stats", response_model=CacheStatsResponse)
def cache_stats() -> CacheStatsResponse:
    semantic_cache = _get_semantic_cache()
    stats = semantic_cache.get_stats()
    return CacheStatsResponse(
        total_requests=stats.total_requests,
        cache_hits=stats.cache_hits,
        redis_hits=stats.redis_hits,
        milvus_hits=stats.milvus_hits,
        context_filtered=stats.context_filtered,
        writes=stats.writes,
        evictions=stats.evictions,
        last_eviction_at=stats.last_eviction_at,
        last_reset_at=stats.last_reset_at,
        hit_rate=stats.hit_rate(),
        cache_size=semantic_cache.cache_size(),
        enabled=semantic_cache.is_enabled(),
    )


@app.post("/cache/reset-stats", response_model=CacheStatsResponse)
def cache_reset_stats() -> CacheStatsResponse:
    semantic_cache = _get_semantic_cache()
    semantic_cache.reset_stats()
    stats = semantic_cache.get_stats()
    return CacheStatsResponse(
        total_requests=stats.total_requests,
        cache_hits=stats.cache_hits,
        redis_hits=stats.redis_hits,
        milvus_hits=stats.milvus_hits,
        context_filtered=stats.context_filtered,
        writes=stats.writes,
        evictions=stats.evictions,
        last_eviction_at=stats.last_eviction_at,
        last_reset_at=stats.last_reset_at,
        hit_rate=stats.hit_rate(),
        cache_size=semantic_cache.cache_size(),
        enabled=semantic_cache.is_enabled(),
    )


@app.post("/cache/evict", response_model=CacheEvictResponse)
def cache_evict(body: CacheEvictRequest) -> CacheEvictResponse:
    semantic_cache = _get_semantic_cache()
    evicted_ids = semantic_cache.evict_oldest(body.n)
    return CacheEvictResponse(evicted_ids=evicted_ids, evicted_count=len(evicted_ids))


@app.post("/cache/clear", response_model=CacheClearResponse)
def cache_clear() -> CacheClearResponse:
    semantic_cache = _get_semantic_cache()
    deleted_count = semantic_cache.clear_all()
    return CacheClearResponse(deleted_count=deleted_count)


# =============================================================================
# /conversation/* — multi-turn endpoints
# =============================================================================

@app.post("/conversation/create", response_model=CreateSessionResponse, status_code=status.HTTP_201_CREATED)
def create_session(body: CreateSessionRequest):
    """Create a new conversation session."""
    try:
        from src.conversation.mysql_store import create_session as db_create_session, audit_log
        from src.conversation.redis_store import update_session_meta
        from src.conversation.models import SessionStatus

        cfg = get_conversation_config()
        ttl_days = int(cfg.get("mysql_retention_days", 30))
        session = db_create_session(user_id=body.user_id, title=body.title, ttl_days=ttl_days)

        update_session_meta(
            session_id=session.id, user_id=body.user_id,
            status=SessionStatus.ACTIVE.value, summary_order=0,
        )
        audit_log(action="CREATE_SESSION", operator=body.user_id,
                  session_id=session.id, resource="session")

        return CreateSessionResponse(
            session_id=session.id, user_id=session.user_id,
            title=session.title, created_at=session.created_at.isoformat(),
        )
    except Exception as exc:
        logger.exception("Failed to create session")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create session: {exc}",
        ) from exc


@app.post("/conversation/{session_id}/message", response_model=SendMessageResponse)
def send_message(session_id: str, body: SendMessageRequest):
    """Send a message in a multi-turn conversation session.

    Flow: write user msg -> answer with RAGChain -> write assistant msg -> check summary trigger
    """
    if rag_chain is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="RAG chain not initialised.")
    try:
        from src.conversation.models import Message, MessageRole
        from src.conversation.mysql_store import (
            get_session, get_session_messages, get_latest_summary, audit_log,
        )
        from src.conversation.redis_store import (
            write_message, update_session_meta, get_recent_messages,
            warm_cache, get_current_message_order,
        )
        from src.conversation.kafka_producer import send_message_event, send_summary_event
        from src.conversation.summarizer import should_trigger_summary

        # Verify session exists
        session = get_session(session_id)
        if session is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                detail=f"Session {session_id} not found.")

        # Step 1: write user message to Redis
        user_msg = Message(
            session_id=session_id, message_order=0, role=MessageRole.USER,
            content=body.content, metadata=body.metadata,
        )
        message_order = write_message(user_msg)
        user_msg.message_order = message_order
        update_session_meta(
            session_id=session_id, user_id=session.user_id,
            status="active", summary_order=session.summary_order,
        )
        send_message_event(session_id, user_msg)
        logger.info("User message written: session=%s, order=%d", session_id, message_order)

        # Step 2: warm Redis cache if miss
        if not get_recent_messages(session_id):
            all_msgs = get_session_messages(session_id)
            if all_msgs:
                warm_cache(session_id, all_msgs)

        latest_summary = get_latest_summary(session_id)
        summary_text = latest_summary.summary_text if latest_summary else ""

        # Step 3: answer through RAGChain
        result = rag_chain.answer(
            question=body.content,
            session_id=session_id,
            session_summary=summary_text,
            user_id=session.user_id,
        )
        answer_text = result["answer"]
        cache_hit_flag = result.get("cache_hit", False)
        cache_hit_from = result.get("cache_hit_from")

        # Step 4: write assistant message to Redis + Kafka
        assistant_msg = Message(
            session_id=session_id, message_order=0, role=MessageRole.ASSISTANT,
            content=answer_text, metadata={"question": body.content},
        )
        asst_order = write_message(assistant_msg)
        assistant_msg.message_order = asst_order
        send_message_event(session_id, assistant_msg)
        logger.info("Assistant reply written: session=%s, order=%d, len=%d",
                    session_id, asst_order, len(answer_text))

        # Step 5: check summary trigger
        cfg2 = get_conversation_config()
        interval = int(cfg2.get("summary_interval_turns", 5))
        current_order = get_current_message_order(session_id)

        if should_trigger_summary(current_order, session.summary_order, interval):
            msgs_for_summary = get_session_messages(session_id)
            new_msgs = [
                m for m in msgs_for_summary
                if m.message_order > session.summary_order * interval
            ]
            if new_msgs:
                send_summary_event(
                    session_id=session_id,
                    summary_order=session.summary_order + 1,
                    turns_start=session.summary_order * interval + 1,
                    turns_end=current_order,
                    previous_summary=summary_text or None,
                    messages_to_summarize=[
                        {"role": m.role.value, "content": m.content}
                        for m in new_msgs
                    ],
                )
                logger.info("Summary event queued: session=%s, summary_order=%d",
                            session_id, session.summary_order + 1)

        audit_log(action="ADD_MESSAGE", operator=session.user_id,
                  session_id=session_id, resource="message",
                  detail={"message_order": message_order, "role": "user"})

        sources = [
            SourceItem(
                text=src.get("text", "")[:200] + ("..." if len(src.get("text", "")) > 200 else ""),
                metadata=src.get("metadata", {}),
                score=src.get("score"),
            )
            for src in result.get("sources", [])
        ]

        return SendMessageResponse(
            answer=answer_text,
            sources=sources,
            session_id=session_id,
            message_order=asst_order,
            cache_hit=cache_hit_flag,
            cache_hit_from=cache_hit_from,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error in /conversation/{id}/message")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing message: {exc}",
        ) from exc


@app.get("/conversation/{session_id}/history", response_model=HistoryResponse)
def get_history(session_id: str, limit: int = 20):
    """Fetch conversation history for a session."""
    try:
        from src.conversation.mysql_store import get_session, get_session_messages
        from src.conversation.redis_store import get_recent_messages, warm_cache

        session = get_session(session_id)
        if session is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                detail=f"Session {session_id} not found.")

        redis_msgs = get_recent_messages(session_id, limit_turns=limit)
        if redis_msgs:
            messages = redis_msgs
        else:
            messages = get_session_messages(session_id, limit=limit * 2)
            if messages:
                warm_cache(session_id, messages)

        return HistoryResponse(
            session_id=session_id,
            messages=[
                MessageItem(
                    message_order=m.message_order, role=m.role.value,
                    content=m.content, created_at=m.created_at.isoformat(),
                )
                for m in messages
            ],
            summary=session.summary,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error in /conversation/{id}/history")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching history: {exc}",
        ) from exc


@app.post("/conversation/{session_id}/close")
def close_session(session_id: str):
    """Archive a conversation session."""
    try:
        from src.conversation.models import SessionStatus
        from src.conversation.mysql_store import (
            get_session, update_session, get_message_count,
            get_latest_summary, audit_log,
        )
        from src.conversation.kafka_producer import send_session_close_event
        from src.conversation.redis_store import delete_session_cache

        session = get_session(session_id)
        if session is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                detail=f"Session {session_id} not found.")

        total_turns = get_message_count(session_id) // 2
        latest_summary = get_latest_summary(session_id)

        send_session_close_event(
            session_id=session_id, total_turns=total_turns,
            final_summary_generated=latest_summary is not None,
        )
        update_session(session_id=session_id, status=SessionStatus.ARCHIVED)
        delete_session_cache(session_id, session.user_id)
        audit_log(action="CLOSE_SESSION", operator=session.user_id,
                  session_id=session_id, resource="session")

        return {"status": "archived", "session_id": session_id, "total_turns": total_turns}

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error closing session %s", session_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error closing session: {exc}",
        ) from exc


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("src.api:app", host="0.0.0.0", port=port, reload=False)
