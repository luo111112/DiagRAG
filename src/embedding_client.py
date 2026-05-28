import logging
import os
import time
from typing import List

import dashscope
from dashscope import TextEmbedding

from src.config_loader import get_embedding_config

logger = logging.getLogger(__name__)

# DashScope TextEmbedding batch limit per call
_EMBEDDING_BATCH_SIZE = 25


class EmbeddingError(Exception):
    """Raised when embedding generation fails."""
    pass


class DashScopeEmbeddingClient:
    """Wrapper for DashScope TextEmbedding API."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "text-embedding-v1",
    ):
        if api_key is None:
            config = get_embedding_config()
            api_key = config.get("dashscope_api_key")
        if not api_key:
            raise EmbeddingError(
                "DashScope API key not provided. "
                "Set DASHSCOPE_API_KEY env var or 'dashscope_api_key' in config.yml."
            )
        dashscope.api_key = api_key
        self.model_name = model_name
        logger.info("DashScopeEmbeddingClient initialized with model: %s", model_name)

    def embed_text(self, text: str) -> List[float]:
        """Return embedding vector for a single text string."""
        vectors = self._call_api([text])
        return vectors[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return embedding vectors for multiple texts, handling API limits automatically."""
        if not texts:
            return []

        results: List[List[float]] = []
        for i in range(0, len(texts), _EMBEDDING_BATCH_SIZE):
            batch = texts[i : i + _EMBEDDING_BATCH_SIZE]
            results.extend(self._call_api(batch))
        return results

    def _call_api(self, texts: List[str]) -> List[List[float]]:
        """Call DashScope TextEmbedding API with retry on rate-limit / transient errors."""
        max_retries = 3
        backoff = 2.0  # seconds
        last_err: Exception = EmbeddingError("placeholder")

        for attempt in range(max_retries):
            try:
                response = TextEmbedding.call(
                    model=self.model_name,
                    input=texts,
                )
                if response.status_code == 200:
                    output = response.output
                    if "embeddings" in output:
                        return [item["embedding"] for item in output["embeddings"]]
                    raise EmbeddingError(
                        f"Unexpected API response shape: {output}"
                    )
                elif response.status_code == 429:
                    logger.warning(
                        "Rate limit hit (attempt %d/%d). Retrying in %.1fs...",
                        attempt + 1, max_retries, backoff,
                    )
                else:
                    raise EmbeddingError(
                        f"API returned status {response.status_code}: {response.message}"
                    )
            except (EmbeddingError,) as e:
                last_err = e
                if attempt < max_retries - 1:
                    logger.warning(
                        "Embedding API error (attempt %d/%d): %s. Retrying in %.1fs...",
                        attempt + 1, max_retries, e, backoff,
                    )
                    time.sleep(backoff)
                    backoff *= 2
                continue

        raise EmbeddingError(f"Embedding API failed after {max_retries} attempts: {last_err}") from last_err


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    client = DashScopeEmbeddingClient()
    single = client.embed_text("急性心肌梗死的典型症状是什么？")
    print(f"Single embedding dim={len(single)}, first 5 values: {single[:5]}")

    docs = [
        "急性心肌梗死的诊断标准",
        "心肌梗死的治疗原则",
        "糖尿病的并发症",
    ]
    vectors = client.embed_documents(docs)
    print(f"Batch returned {len(vectors)} vectors, dim={len(vectors[0]) if vectors else 0}")
