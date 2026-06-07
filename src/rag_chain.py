"""RAG chain that orchestrates retrieval and generation for DiagRAG.

处理流程：
    1. 将用户问题嵌入为向量。
    2. 在 Milvus 中搜索 top-k 最相关文档块。
    3. （可选）对文档块进行重排序。
    4. （可选）根据元数据字段过滤文档块。
    5. 将文档块文本拼接为上下文字符串。
    6. 用上下文 + 问题填充 RAG 提示词模板。
    7. 请求 LLM 生成有据可查的回答。
    8. （可选）将结果写入语义缓存。
    9. 返回回答及引用信息和原始检索结果。
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from typing import TYPE_CHECKING, Any

from src.config_loader import (
    get_metadata_filter_config,
    get_preprocessor_config,
    get_reranker_config,
    get_retrieval_config,
    get_semantic_cache_config,
    load_config,
)
from src.generation.prompts import MEDICAL_SYSTEM_PROMPT, RAG_PROMPT_TEMPLATE

if TYPE_CHECKING:
    from src.embedding_client import DashScopeEmbeddingClient
    from src.llm_client import DashScopeLLMClient
    from src.milvus_client import MilvusClient
    from src.retrieval.metadata_filter import MetadataFilter
    from src.retrieval.query_preprocessor import QueryPreprocessor
    from src.retrieval.rerank import LLMRanker
    from src.vectorstore.semantic_cache import SemanticCache

logger = logging.getLogger(__name__)


class RAGChainError(Exception):
    """Raised when any step in the RAG chain fails."""

    pass


class RAGChain:
    """End-to-end Retrieval-Augmented Generation chain for medical Q&A."""

    def __init__(
        self,
        embedding_client: DashScopeEmbeddingClient,
        milvus_client: MilvusClient,
        llm_client: DashScopeLLMClient,
        top_k: int | None = None,
        system_prompt: str | None = None,
        reranker: "LLMRanker | None" = None,
        preprocessor: "QueryPreprocessor | None" = None,
        metadata_filter: "MetadataFilter | None" = None,
        semantic_cache: "SemanticCache | None" = None,
    ) -> None:
        """初始化 RAG chain。

        Args:
            embedding_client: 将文本转换为向量的客户端。
            milvus_client: Milvus 向量数据库操作客户端。
            llm_client: LLM 生成客户端（如 DashScope qwen）。
            top_k: 每次检索返回的文档块数量上限。
                   默认为 ``config["retrieval"]["top_k"]``。
            system_prompt: 传给 LLM 的系统级指令。
                          默认为 ``prompts.MEDICAL_SYSTEM_PROMPT``。
            reranker: 重排序客户端（如 LLMRanker）。当提供且
                      ``config["retrieval"]["enable_rerank"]`` 为 True 时，
                      检索结果会在传给 LLM 前先进行重排序。
                      若为 None，则自动从配置构造。
            preprocessor: 查询预处理器（如 QueryPreprocessor）。
                          提供时，会在检索前对原始查询进行归一化、
                          纠错、改写和/或扩展。
                          若为 None，则自动从配置构造。
            metadata_filter: 基于元数据的后置过滤器（如 MetadataFilter）。
                             提供时，检索到的文档块会在传给 LLM 前
                             根据其存储的元数据字段进行过滤。
                             若为 None，则自动从配置构造。
            semantic_cache: 语义缓存实例（SemanticCache）。
                            提供时，answer() 会在调用 LLM 前先查询缓存，
                            命中则直接返回；LLM 生成完成后会将结果写入缓存。
                            若为 None，则自动从配置构造（需 embedding_client
                            和 milvus_client 均已注入）。
        """
        self.embedding_client = embedding_client
        self.milvus_client = milvus_client
        self.llm_client = llm_client

        if top_k is None:
            cfg = load_config()
            top_k = cfg.get("retrieval", {}).get("top_k", 5)
        self.top_k = top_k

        self.system_prompt = system_prompt or MEDICAL_SYSTEM_PROMPT

        # Reranker: use provided instance, or auto-construct from config
        self.reranker: "LLMRanker | None" = None
        reranker_cfg = get_reranker_config()
        if reranker is not None:
            self.reranker = reranker
            logger.info("RAGChain using provided reranker instance.")
        elif reranker_cfg.get("enable_rerank", False):
            from src.retrieval.rerank import build_ranker
            self.reranker = build_ranker(
                llm_client=llm_client,
                mode=reranker_cfg.get("rerank_mode", "score"),
                top_k=reranker_cfg.get("rerank_top_k", 3),
                enable_bm25=reranker_cfg.get("enable_bm25_blend", False),
                fusion=reranker_cfg.get("rerank_fusion", "rrf"),
            )
            logger.info(
                "RAGChain auto-constructed reranker: mode=%s, rerank_top_k=%d",
                reranker_cfg.get("rerank_mode"), reranker_cfg.get("rerank_top_k"),
            )
        else:
            logger.info("RAGChain reranker disabled (enable_rerank=false).")

        # Query preprocessor: use provided instance, or auto-construct from config
        self.preprocessor: "QueryPreprocessor | None" = None
        preprocessor_cfg = get_preprocessor_config()
        if preprocessor is not None:
            self.preprocessor = preprocessor
            logger.info("RAGChain using provided preprocessor instance.")
        elif preprocessor_cfg.get("enable_rewrite") or preprocessor_cfg.get("enable_expand"):
            from src.retrieval.query_preprocessor import build_preprocessor
            self.preprocessor = build_preprocessor(
                llm_client=llm_client,
                enable_rewrite=preprocessor_cfg.get("enable_rewrite", False),
                enable_expand=preprocessor_cfg.get("enable_expand", False),
                max_variants=preprocessor_cfg.get("max_variants", 2),
                max_expand_terms=preprocessor_cfg.get("max_expand_terms", 4),
            )
            logger.info(
                "RAGChain auto-constructed preprocessor: rewrite=%s, expand=%s",
                preprocessor_cfg.get("enable_rewrite"),
                preprocessor_cfg.get("enable_expand"),
            )
        else:
            logger.info("RAGChain preprocessor disabled (enable_rewrite=expand=false).")

        # ------------------------------------------------------------------
        # 元数据过滤器：优先使用注入的实例，否则从配置自动构造。
        #
        # 启用后，每个检索到的文档块都会依据白名单（所有字段必须匹配）
        # 和黑名单（任意字段命中即丢弃）进行检查。
        # 过滤发生在重排序之后，确保高质量候选块不受影响。
        # ------------------------------------------------------------------
        self.metadata_filter: "MetadataFilter | None" = None
        mf_cfg = get_metadata_filter_config()
        if metadata_filter is not None:
            self.metadata_filter = metadata_filter
            logger.info("RAGChain using provided metadata_filter instance.")
        elif mf_cfg.get("enabled", False):
            from src.retrieval.metadata_filter import MetadataFilter
            self.metadata_filter = MetadataFilter(
                whitelist=mf_cfg.get("whitelist"),
                blacklist=mf_cfg.get("blacklist"),
            )
            logger.info(
                "RAGChain auto-constructed metadata_filter: whitelist=%s, blacklist=%s",
                mf_cfg.get("whitelist"),
                mf_cfg.get("blacklist"),
            )
        else:
            logger.info("RAGChain metadata_filter disabled (enabled=false).")

        # ------------------------------------------------------------------
        # 语义缓存：优先使用注入的实例，否则从配置自动构造。
        #
        # 提供两级缓存（Redis 精确键 + Milvus ANN 向量），在调用 LLM 前
        # 拦截命中，命中后直接返回缓存结果；LLM 生成完成后将结果写入缓存。
        # ------------------------------------------------------------------
        self.semantic_cache: "SemanticCache | None" = None
        cache_cfg = get_semantic_cache_config()
        if semantic_cache is not None:
            self.semantic_cache = semantic_cache
            logger.info("RAGChain using provided SemanticCache instance.")
        elif cache_cfg.get("enabled", False):
            from src.vectorstore.semantic_cache import SemanticCache
            self.semantic_cache = SemanticCache(
                embedding_client=embedding_client,
                milvus_client=milvus_client,
            )
            logger.info("RAGChain auto-constructed SemanticCache.")
        else:
            logger.info("RAGChain SemanticCache disabled (enabled=false).")

        logger.info(
            "RAGChain initialized: top_k=%d, system_prompt_len=%d, "
            "reranker=%s, preprocessor=%s, metadata_filter=%s, semantic_cache=%s",
            self.top_k,
            len(self.system_prompt),
            type(self.reranker).__name__ if self.reranker else "None",
            type(self.preprocessor).__name__ if self.preprocessor else "None",
            type(self.metadata_filter).__name__ if self.metadata_filter else "None",
            type(self.semantic_cache).__name__ if self.semantic_cache else "None",
        )

    def answer(
        self,
        question: str,
        session_id: str | None = None,
        session_summary: str = "",
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """使用检索增强生成回答医学问题。

        完整处理流程：

        1. （可选）查询语义缓存（Redis 精确 + Milvus ANN）—— 命中直接返回。
        2. 预处理查询（归一化 / 改写 / 扩展）—— 可选。
        3. 将（预处理后的）问题嵌入为向量。
        4. 从 Milvus 检索 top-k 最相关文档块。
        5. 对文档块进行重排序（语义 / 关键词相关性）—— 可选。
        6. 根据元数据字段过滤文档块（白名单 / 黑名单）—— 可选。
        7. 用剩余文档块构建上下文字符串。
        8. 请求 LLM 基于上下文生成回答。
        9. （可选）将 LLM 生成结果写入语义缓存。

        Args:
            question: 用户的临床 / 医学问题。
            session_id: 当前会话 ID，用于语义缓存上下文关联（多轮场景）。
            session_summary: 当前会话摘要，用于上下文指纹 Jaccard 比对。
            user_id: 用户 ID（user scope 语义缓存键构建用）。

        Returns:
            包含以下键的字典：
            - ``answer`` (str): LLM 生成的回答（命中缓存时为缓存值）。
            - ``sources`` (list[dict]): 引用列表。
            - ``retrieved_chunks`` (list[dict] | None): Milvus 原始命中结果。
            - ``context`` (str): 喂给 LLM 的上下文字符串。
            - ``reranked`` (bool): 是否启用了重排序。
            - ``reranked_chunks`` (list[dict] | None): 重排序后的文档块列表。
            - ``filtered`` (bool): metadata_filter 是否激活。
            - ``filtered_chunks`` (list[dict] | None): 元数据过滤后的文档块列表。
            - ``preprocessed`` (bool): 是否启用了查询预处理。
            - ``preprocess_result`` (dict | None): 预处理诊断信息。
            - ``cache_hit`` (bool): 是否命中语义缓存。
            - ``cache_hit_from`` (str | None): 命中来源，``"redis_exact"`` 或 ``"milvus_semantic"``。

        Raises:
            RAGChainError: If embedding, retrieval, or generation fails.
        """
        # ------------------------------------------------------------------
        # Step 0 – 语义缓存查询（一级 Redis 精确 + 二级 Milvus ANN）
        # ------------------------------------------------------------------
        if self.semantic_cache is not None and self.semantic_cache.is_enabled():
            cache_hit = self.semantic_cache.get_or_set(
                question=question,
                session_id=session_id,
                session_summary=session_summary,
                user_id=user_id,
            )
            if cache_hit is not None:
                logger.info(
                    "SemanticCache HIT (%s) for question: %s",
                    cache_hit.hit_from, question[:60],
                )
                return {
                    "answer": cache_hit.answer_text,
                    "sources": cache_hit.sources,
                    "retrieved_chunks": None,
                    "context": "",
                    "reranked": False,
                    "reranked_chunks": None,
                    "filtered": False,
                    "filtered_chunks": None,
                    "preprocessed": False,
                    "preprocess_result": None,
                    "cache_hit": True,
                    "cache_hit_from": cache_hit.hit_from,
                }

        # ------------------------------------------------------------------
        # Step 1 – 预处理查询（归一化 / 纠错 / 改写 / 扩展）
        # ------------------------------------------------------------------
        preprocessed = False
        preprocess_result: dict[str, Any] | None = None
        effective_question = question
        if self.preprocessor is not None:
            try:
                preprocess_result = self.preprocessor.process(question)
                effective_question = preprocess_result["final_query"]
                preprocessed = True
                logger.info(
                    "Query preprocessed: %r → %r",
                    question[:50],
                    effective_question[:50],
                )
            except Exception as e:
                logger.warning("Query preprocessing failed, falling back to raw query: %s", e)
                effective_question = question
                preprocessed = False
                preprocess_result = None

        # ------------------------------------------------------------------
        # Step 2 – 将（预处理后的）问题嵌入为向量
        # ------------------------------------------------------------------
        try:
            query_vector = self.embedding_client.embed_text(effective_question)
            logger.debug("Question embedded, vector dim=%d", len(query_vector))
        except Exception as e:
            raise RAGChainError(f"Embedding step failed: {e}") from e

        # ------------------------------------------------------------------
        # Step 3 – 从 Milvus 检索 top-k 最相关文档块
        # ------------------------------------------------------------------
        try:
            chunks = self.milvus_client.search(
                query_vector=query_vector,
                top_k=self.top_k,
            )
            logger.debug("Milvus search returned %d chunks", len(chunks))
        except Exception as e:
            raise RAGChainError(f"Retrieval step failed: {e}") from e

        # ------------------------------------------------------------------
        # Step 4 – 对检索结果进行重排序（可选）
        # ------------------------------------------------------------------
        reranked_chunks: list[dict[str, Any]] | None = None
        reranked = False
        if self.reranker is not None and chunks:
            reranker_cfg = get_reranker_config()
            effective_rerank_top_k = reranker_cfg.get("rerank_top_k", 3)
            try:
                reranked_chunks = self.reranker.rerank(
                    query=effective_question,
                    chunks=chunks,
                    top_k=effective_rerank_top_k,
                )
                # reranker 内部已做 top_k 截断，直接用其结果
                chunks = reranked_chunks
                reranked = True
                logger.info(
                    "Re-ranking applied: %d chunks → %d chunks (mode=%s)",
                    len(reranked_chunks) if reranked_chunks else 0,
                    effective_rerank_top_k,
                    reranker_cfg.get("rerank_mode"),
                )
            except Exception as e:
                # rerank 失败不影响主流程，回退到原始检索结果
                logger.warning("Re-ranking failed, falling back to raw retrieval: %s", e)

        # ------------------------------------------------------------------
        # Step 5 – 根据元数据字段过滤文档块（可选）
        # ------------------------------------------------------------------
        filtered_chunks: list[dict[str, Any]] | None = None
        filtered = False
        if self.metadata_filter is not None and chunks:
            filtered_chunks = self.metadata_filter.filter(chunks)
            filtered = True
            if len(filtered_chunks) < len(chunks):
                logger.info(
                    "Metadata filter: %d chunks → %d chunks (dropped %d)",
                    len(chunks),
                    len(filtered_chunks),
                    len(chunks) - len(filtered_chunks),
                )
            else:
                logger.debug("Metadata filter: all %d chunks passed", len(chunks))
            chunks = filtered_chunks

        # ------------------------------------------------------------------
        # Step 3 – 将检索到的文档块拼接为上下文字符串
        # ------------------------------------------------------------------
        context_parts: list[str] = []
        for i, chunk in enumerate(chunks):
            context_parts.append(
                f"[文档{i + 1}]\n{chunk['text']}"
            )

        context = "\n\n---\n\n".join(context_parts)

        if not context:
            # 无文档块检索到时，不调用 LLM，直接返回空答案
            logger.warning("No chunks retrieved for question: %s", question[:80])
            return {
                "answer": (
                    "当前知识库中缺少与您问题相关的证据，无法给出确切结论。"
                    "请尝试换一种表述，或补充更多临床信息。"
                ),
                "sources": [],
                "retrieved_chunks": None,
                "context": "",
                "reranked": reranked,
                "reranked_chunks": reranked_chunks,
                "filtered": filtered,
                "filtered_chunks": filtered_chunks,
                "preprocessed": preprocessed,
                "preprocess_result": preprocess_result,
                "cache_hit": False,
                "cache_hit_from": None,
            }

        # ------------------------------------------------------------------
        # Step 6 – 用上下文 + 问题填充用户提示词模板
        # ------------------------------------------------------------------
        user_prompt = RAG_PROMPT_TEMPLATE.format(
            context=context,
            question=question,
        )

        # ------------------------------------------------------------------
        # Step 7 – 请求 LLM 生成回答
        # ------------------------------------------------------------------
        try:
            answer_text = self.llm_client.generate(
                prompt=user_prompt,
                system_prompt=self.system_prompt,
            )
            logger.info("LLM generated answer (%d chars)", len(answer_text))
        except Exception as e:
            raise RAGChainError(f"Generation step failed: {e}") from e

        # ------------------------------------------------------------------
        # Step 8 – 从文档块元数据中提取引用信息
        # ------------------------------------------------------------------
        sources: list[dict[str, Any]] = []
        for chunk in chunks:
            meta = chunk.get("metadata") or {}
            sources.append({
                "text": chunk["text"][:200] + ("..." if len(chunk["text"]) > 200 else ""),
                "metadata": meta,
                "score": chunk.get("score"),
            })

        # ------------------------------------------------------------------
        # Step 9 – 将结果写入语义缓存
        # ------------------------------------------------------------------
        if self.semantic_cache is not None and self.semantic_cache.is_enabled():
            self.semantic_cache.write(
                question=effective_question,
                answer_text=answer_text,
                sources=sources,
                query_vector=query_vector,
                session_id=session_id,
                session_summary=session_summary,
                user_id=user_id,
            )

        return {
            "answer": answer_text,
            "sources": sources,
            "retrieved_chunks": chunks,
            "context": context,
            "reranked": reranked,
            "reranked_chunks": reranked_chunks,
            "filtered": filtered,
            "filtered_chunks": filtered_chunks,
            "preprocessed": preprocessed,
            "preprocess_result": preprocess_result,
            "cache_hit": False,
            "cache_hit_from": None,
        }

    def stream_answer(
        self,
        question: str,
        session_id: str | None = None,
        session_summary: str = "",
        user_id: str | None = None,
    ) -> tuple[Generator[str, None, None], list[dict[str, Any]]]:
        """流式返回 LLM 回答，同时返回检索来源。

        流式管线与 ``answer()`` 完全对称（语义缓存查询 → 预处理 → 嵌入 → 检索 →
        重排 → 元数据过滤 → 上下文构建 → LLM 生成 → 缓存写入）。
        当 ``metadata_filter`` 激活时，来源列表会经过元数据过滤。
        语义缓存的写入也在流式完成后执行。

        Args:
            question: 用户的临床 / 医学问题。
            session_id: 当前会话 ID（语义缓存上下文关联用）。
            session_summary: 当前会话摘要（用于上下文指纹 Jaccard 比对）。
            user_id: 用户 ID（user scope 语义缓存键构建用）。

        Returns:
            (token_generator, sources_list) 元组。

        Raises:
            RAGChainError: If embedding or retrieval fails.
        """
        # ------------------------------------------------------------------
        # Step 0 – 预处理查询（与 answer() 对称）
        # ------------------------------------------------------------------
        preprocessed = False
        preprocess_result: dict[str, Any] | None = None
        effective_question = question
        if self.preprocessor is not None:
            try:
                preprocess_result = self.preprocessor.process(question)
                effective_question = preprocess_result["final_query"]
                preprocessed = True
            except Exception as e:
                logger.warning("Query preprocessing failed in stream_answer, using raw query: %s", e)
                effective_question = question
                preprocessed = False
                preprocess_result = None

        try:
            query_vector = self.embedding_client.embed_text(effective_question)
        except Exception as e:
            raise RAGChainError(f"Embedding step failed: {e}") from e

        try:
            chunks = self.milvus_client.search(query_vector=query_vector, top_k=self.top_k)
        except Exception as e:
            raise RAGChainError(f"Retrieval step failed: {e}") from e

        # 重排序步骤（与 answer() 对称）
        reranked_chunks: list[dict[str, Any]] | None = None
        reranked = False
        if self.reranker is not None and chunks:
            reranker_cfg = get_reranker_config()
            effective_rerank_top_k = reranker_cfg.get("rerank_top_k", 3)
            try:
                reranked_chunks = self.reranker.rerank(
                    query=effective_question,
                    chunks=chunks,
                    top_k=effective_rerank_top_k,
                )
                chunks = reranked_chunks
                reranked = True
            except Exception as e:
                logger.warning("Re-ranking failed in stream_answer, using raw retrieval: %s", e)

        # ------------------------------------------------------------------
        # 元数据过滤步骤 — 与 answer() 中的逻辑完全对称。
        # 通过过滤的文档块会继续参与上下文构建和 LLM 生成。
        # ------------------------------------------------------------------
        filtered_chunks: list[dict[str, Any]] | None = None
        filtered = False
        if self.metadata_filter is not None and chunks:
            filtered_chunks = self.metadata_filter.filter(chunks)
            filtered = True
            chunks = filtered_chunks

        context_parts = [f"[文档{i + 1}]\n{chunk['text']}" for i, chunk in enumerate(chunks)]
        context = "\n\n---\n\n".join(context_parts)

        sources: list[dict[str, Any]] = []
        for chunk in chunks:
            meta = chunk.get("metadata") or {}
            sources.append({
                "text": chunk["text"][:200] + ("..." if len(chunk["text"]) > 200 else ""),
                "metadata": meta,
                "score": chunk.get("score"),
            })

        user_prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=effective_question)

        def token_stream():
            try:
                for token in self.llm_client.stream_generate(
                    prompt=user_prompt,
                    system_prompt=self.system_prompt,
                ):
                    yield token
            except Exception as e:
                raise RAGChainError(f"Generation step failed: {e}") from e

        return token_stream(), sources
