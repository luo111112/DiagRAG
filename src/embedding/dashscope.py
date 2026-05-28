"""DashScope API client utilities (embedding + generation)."""

from __future__ import annotations

import os
from typing import Any

import dashscope
from dashscope import Generation, TextEmbedding


def init_dashscope(api_key: str | None = None) -> None:
    """Initialize the DashScope global API key."""
    dashscope.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")


def embed_texts(model: str, texts: list[str]) -> list[list[float]]:
    """Call DashScope text-embedding API, return list of vectors."""
    response = TextEmbedding.call(model=model, input=texts)
    _check_response(response)
    return [item["embedding"] for item in response.output["embeddings"]]


def generate_text(
    model: str,
    prompt: str,
    temperature: float = 0.1,
    max_tokens: int = 1000,
    **kwargs: Any,
) -> str:
    """Call DashScope generation API, return generated text."""
    response = Generation.call(
        model=model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )
    _check_response(response)
    return response.output["text"]


def _check_response(response: Any) -> None:
    """Raise RuntimeError for non-200 responses."""
    if response.status_code != 200:
        raise RuntimeError(
            f"DashScope API error {response.code}: {response.message}"
        )
