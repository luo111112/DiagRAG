"""Embedding model wrapper (DashScope text-embedding-v1)."""

from __future__ import annotations

import os
from typing import Annotated

import dashscope
from dashscope import TextEmbedding
from tenacity import retry, stop_after_attempt, wait_exponential

from src.embedding.cache import EmbeddingCache


class Embedder:
    """Wrapper around DashScope TextEmbedding API with optional in-memory cache."""

    def __init__(
        self,
        model_name: str = "text-embedding-v1",
        api_key: str | None = None,
        cache_dir: str | None = None,
        cache_ttl: int = 3600,
    ) -> None:
        self.model_name = model_name
        dashscope.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        self.cache = EmbeddingCache(ttl=cache_ttl, cache_dir=cache_dir)

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _call_api(self, texts: list[str]) -> list[list[float]]:
        """Call DashScope API with retry logic."""
        response = TextEmbedding.call(model=self.model_name, input=texts)
        if response.status_code != 200:
            raise RuntimeError(
                f"DashScope API error: {response.code} {response.message}"
            )
        return [item["embedding"] for item in response.output["embeddings"]]

    def embed_single(self, text: str) -> list[float]:
        """Embed a single text string (uses cache)."""
        cached = self.cache.get(text)
        if cached is not None:
            return cached
        vectors = self._call_api([text])
        self.cache.set(text, vectors[0])
        return vectors[0]

    def embed_batch(
        self, texts: list[str], skip_cache_hits: bool = False
    ) -> list[list[float]]:
        """Embed multiple texts; optionally skip cache reads for bulk ingest."""
        results: list[list[float] | None] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        if not skip_cache_hits:
            for i, text in enumerate(texts):
                cached = self.cache.get(text)
                if cached is not None:
                    results[i] = cached
                else:
                    uncached_indices.append(i)
                    uncached_texts.append(text)
        else:
            uncached_indices = list(range(len(texts)))
            uncached_texts = texts

        if uncached_texts:
            vectors = self._call_api(uncached_texts)
            for idx, vec in zip(uncached_indices, vectors):
                results[idx] = vec
                self.cache.set(texts[idx], vec)

        return results  # type: ignore[return-value]

    def clear_cache(self) -> None:
        """Clear the in-memory embedding cache."""
        self.cache.clear()
