"""RAG chain that orchestrates retrieval and generation for DiagRAG.

Flow:
    1. Embed the user question into a vector.
    2. Search Milvus for the top-k most relevant chunks.
    3. Concatenate chunk texts into a context string.
    4. Fill the RAG prompt template with context + question.
    5. Ask the LLM for a grounded answer.
    6. Return the answer together with citations and raw retrieval results.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from src.config_loader import (
    get_preprocessor_config,
    get_reranker_config,
    get_retrieval_config,
    load_config,
)
from src.generation.prompts import MEDICAL_SYSTEM_PROMPT, RAG_PROMPT_TEMPLATE

if TYPE_CHECKING:
    from src.embedding_client import DashScopeEmbeddingClient
    from src.llm_client import DashScopeLLMClient
    from src.milvus_client import MilvusClient
    from src.retrieval.query_preprocessor import QueryPreprocessor
    from src.retrieval.rerank import LLMRanker

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
    ) -> None:
        """Initialize the RAG chain.

        Args:
            embedding_client: Client that turns text into embedding vectors.
            milvus_client: Client for Milvus vector database operations.
            llm_client: Client for LLM generation (e.g. DashScope qwen).
            top_k: Number of chunks to retrieve per question.
                   Defaults to ``config["retrieval"]["top_k"]``.
            system_prompt: System-level instruction passed to the LLM.
                          Defaults to ``prompts.MEDICAL_SYSTEM_PROMPT``.
            reranker: Re-ranking client (e.g. LLMRanker). When provided and
                      ``config["retrieval"]["enable_rerank"]`` is True, retrieval
                      results are re-ranked before being passed to generation.
                      If None, checks config for auto-construction.
            preprocessor: Query preprocessing client (e.g. QueryPreprocessor).
                          When provided, raw user queries are normalized,
                          corrected, rewritten, and/or expanded before retrieval.
                          If None, checks config for auto-construction.
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

        logger.info(
            "RAGChain initialized: top_k=%d, system_prompt_len=%d, "
            "reranker=%s, preprocessor=%s",
            self.top_k,
            len(self.system_prompt),
            type(self.reranker).__name__ if self.reranker else "None",
            type(self.preprocessor).__name__ if self.preprocessor else "None",
        )

    def answer(self, question: str) -> dict[str, Any]:
        """Answer a medical question using retrieval-augmented generation.

        Args:
            question: The user's clinical / medical question.

        Returns:
            A dictionary containing:
            - ``answer`` (str): The LLM-generated answer.
            - ``sources`` (list[dict]): Citation list. Each entry has
              ``text``, ``metadata``, ``score``.
            - ``retrieved_chunks`` (list[dict] | None): Raw Milvus hits,
              or None when retrieval yields no results.
            - ``context`` (str): The context string that was fed to the LLM.
            - ``reranked`` (bool): Whether re-ranking was applied.
            - ``reranked_chunks`` (list[dict] | None): Re-ranked results
              (only present when reranker is active).
            - ``preprocessed`` (bool): Whether query preprocessing was applied.
            - ``preprocess_result`` (dict | None): Full preprocessing diagnostics
              (only present when preprocessor is active).

        Raises:
            RAGChainError: If embedding, retrieval, or generation fails.
        """
        # ------------------------------------------------------------------
        # Step 0 – Preprocess the query (normalize, correct, rewrite, expand)
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
        # Step 1 – Embed the (preprocessed) question
        # ------------------------------------------------------------------
        try:
            query_vector = self.embedding_client.embed_text(effective_question)
            logger.debug("Question embedded, vector dim=%d", len(query_vector))
        except Exception as e:
            raise RAGChainError(f"Embedding step failed: {e}") from e

        # ------------------------------------------------------------------
        # Step 2 – Retrieve top-k relevant chunks from Milvus
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
        # Step 2.5 – Re-rank the retrieved chunks (optional)
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
        # Step 3 – Build the context string from retrieved chunks
        # ------------------------------------------------------------------
        context_parts: list[str] = []
        for i, chunk in enumerate(chunks):
            context_parts.append(
                f"[文档{i + 1}]\n{chunk['text']}"
            )

        context = "\n\n---\n\n".join(context_parts)

        if not context:
            # No chunks retrieved – return a graceful fallback without calling LLM
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
            }

        # ------------------------------------------------------------------
        # Step 4 – Format the user prompt with context + question
        # ------------------------------------------------------------------
        user_prompt = RAG_PROMPT_TEMPLATE.format(
            context=context,
            question=question,
        )

        # ------------------------------------------------------------------
        # Step 5 – Generate the answer via LLM
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
        # Step 6 – Extract sources from chunk metadata
        # ------------------------------------------------------------------
        sources: list[dict[str, Any]] = []
        for chunk in chunks:
            meta = chunk.get("metadata") or {}
            sources.append({
                "text": chunk["text"][:200] + ("..." if len(chunk["text"]) > 200 else ""),
                "metadata": meta,
                "score": chunk.get("score"),
            })

        return {
            "answer": answer_text,
            "sources": sources,
            "retrieved_chunks": chunks,
            "context": context,
            "reranked": reranked,
            "reranked_chunks": reranked_chunks,
            "preprocessed": preprocessed,
            "preprocess_result": preprocess_result,
        }

    def stream_answer(self, question: str) -> tuple[Generator[str, None, None], list[dict[str, Any]]]:
        """Stream the LLM answer while also returning retrieval sources.

        Yields tokens from the LLM and returns source metadata alongside.

        Args:
            question: The user's clinical / medical question.

        Returns:
            A tuple of (token_generator, sources_list).

        Raises:
            RAGChainError: If embedding or retrieval fails.
        """
        # ------------------------------------------------------------------
        # Step 0 – Preprocess the query (mirrors answer())
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

        # Re-ranking step (mirrors answer())
        if self.reranker is not None and chunks:
            reranker_cfg = get_reranker_config()
            effective_rerank_top_k = reranker_cfg.get("rerank_top_k", 3)
            try:
                chunks = self.reranker.rerank(
                    query=effective_question,
                    chunks=chunks,
                    top_k=effective_rerank_top_k,
                )
            except Exception as e:
                logger.warning("Re-ranking failed in stream_answer, using raw retrieval: %s", e)

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
