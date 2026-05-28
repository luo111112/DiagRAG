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

from src.config_loader import load_config
from src.generation.prompts import MEDICAL_SYSTEM_PROMPT, RAG_PROMPT_TEMPLATE

if TYPE_CHECKING:
    from src.embedding_client import DashScopeEmbeddingClient
    from src.llm_client import DashScopeLLMClient
    from src.milvus_client import MilvusClient

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
        """
        self.embedding_client = embedding_client
        self.milvus_client = milvus_client
        self.llm_client = llm_client

        if top_k is None:
            cfg = load_config()
            top_k = cfg.get("retrieval", {}).get("top_k", 5)
        self.top_k = top_k

        self.system_prompt = system_prompt or MEDICAL_SYSTEM_PROMPT

        logger.info(
            "RAGChain initialized: top_k=%d, system_prompt_len=%d",
            self.top_k,
            len(self.system_prompt),
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

        Raises:
            RAGChainError: If embedding, retrieval, or generation fails.
        """
        # ------------------------------------------------------------------
        # Step 1 – Embed the question
        # ------------------------------------------------------------------
        try:
            query_vector = self.embedding_client.embed_text(question)
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
        try:
            query_vector = self.embedding_client.embed_text(question)
        except Exception as e:
            raise RAGChainError(f"Embedding step failed: {e}") from e

        try:
            chunks = self.milvus_client.search(query_vector=query_vector, top_k=self.top_k)
        except Exception as e:
            raise RAGChainError(f"Retrieval step failed: {e}") from e

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

        user_prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)

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
