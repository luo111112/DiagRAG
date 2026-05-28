import logging
import time
from typing import Generator

import dashscope
from dashscope import Generation

from src.config_loader import get_llm_config

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Raised when LLM generation fails."""
    pass


class DashScopeLLMClient:
    """Wrapper for DashScope Generation API (qwen series)."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "qwen-max",
        temperature: float = 0.1,
        max_tokens: int = 1500,
    ):
        if api_key is None:
            config = get_llm_config()
            api_key = config.get("dashscope_api_key")
            model_name = config.get("model", model_name)
            temperature = float(config.get("temperature", temperature))
            max_tokens = int(config.get("max_tokens", max_tokens))

        if not api_key:
            raise LLMError(
                "DashScope API key not provided. "
                "Set DASHSCOPE_API_KEY env var or 'dashscope_api_key' in config.yml."
            )
        dashscope.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        logger.info(
            "DashScopeLLMClient initialized: model=%s, temperature=%s, max_tokens=%d",
            model_name, temperature, max_tokens,
        )

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        """Generate a response from the LLM.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system message. If the model does not natively
                support system messages, it is prepended to the user prompt.

        Returns:
            The generated text response.

        Raises:
            LLMError: On API errors or after all retries are exhausted.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        max_retries = 3
        backoff = 2.0
        last_err: Exception = LLMError("placeholder")

        for attempt in range(max_retries):
            try:
                response = Generation.call(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    result_format="message",
                )
                if response.status_code == 200:
                    return response.output.choices[0].message.content.strip()
                elif response.status_code == 429:
                    logger.warning(
                        "Rate limit hit (attempt %d/%d). Retrying in %.1fs...",
                        attempt + 1, max_retries, backoff,
                    )
                elif response.status_code == 400 and "system" in str(response.message).lower():
                    logger.warning(
                        "Model does not support system messages natively, "
                        "falling back to prepending system prompt."
                    )
                    messages = [
                        {"role": "user", "content": f"{system_prompt}\n\n{prompt}"}
                    ]
                    response = Generation.call(
                        model=self.model_name,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        result_format="message",
                    )
                    if response.status_code == 200:
                        return response.output.choices[0].message.content.strip()
                    raise LLMError(
                        f"API returned status {response.status_code}: {response.message}"
                    )
                else:
                    raise LLMError(
                        f"API returned status {response.status_code}: {response.message}"
                    )
            except LLMError as e:
                last_err = e
                if attempt < max_retries - 1:
                    logger.warning(
                        "LLM API error (attempt %d/%d): %s. Retrying in %.1fs...",
                        attempt + 1, max_retries, e, backoff,
                    )
                    time.sleep(backoff)
                    backoff *= 2
                continue

        raise LLMError(
            f"LLM generation failed after {max_retries} attempts: {last_err}"
        ) from last_err

    def stream_generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> Generator[str, None, None]:
        """Stream tokens from the LLM using DashScope Generation streaming API.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system message.

        Yields:
            Token strings as they arrive from the API.

        Raises:
            LLMError: On unrecoverable API errors.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = Generation.call(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            result_format="message",
            stream=True,
            stream_options={"incremental_output": True},
        )

        prev_len = 0
        for item in response:
            if item.status_code == 200:
                output = item.output
                if output is None:
                    continue

                # item.output is a dict in dashscope >= 1.20
                if isinstance(output, dict):
                    choices = output.get("choices")
                    if choices:
                        choice = choices[0] if isinstance(choices, list) else choices
                        # incremental_output=True: token may be in delta.content
                        delta = choice.get("delta") if isinstance(choice, dict) else getattr(choice, "delta", None)
                        if delta is not None:
                            content = delta.get("content") if isinstance(delta, dict) else getattr(delta, "content", None)
                            if content:
                                yield content
                            prev_len = 0
                            continue
                        # or token may be in message.content (incremental mode, updated each chunk)
                        msg = choice.get("message") if isinstance(choice, dict) else getattr(choice, "message", None)
                        if msg is not None:
                            content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
                            if content:
                                if len(content) > prev_len:
                                    yield content[prev_len:]
                                    prev_len = len(content)
                                elif prev_len == 0:
                                    yield content
                            continue
                    elif "text" in output:
                        content = output["text"]
                        if content:
                            yield content
                    prev_len = 0
                    continue

                # Fallback: treat output as a DashScope response object with .choices
                if hasattr(output, "choices"):
                    choice = output.choices[0]
                    # incremental_output: try delta first, then message.content
                    delta = getattr(choice, "delta", None)
                    if delta is not None:
                        content = delta.get("content") if isinstance(delta, dict) else getattr(delta, "content", None)
                        if content:
                            yield content
                        prev_len = 0
                        continue
                    # message.content incremental mode (dashscope >= 1.25)
                    msg = getattr(choice, "message", None)
                    if msg is not None:
                        content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
                        if content:
                            if len(content) > prev_len:
                                yield content[prev_len:]
                                prev_len = len(content)
                            elif prev_len == 0:
                                yield content
                elif hasattr(output, "text"):
                    content = output.text or ""
                    if content:
                        yield content
                prev_len = 0
            else:
                raise LLMError(
                    f"Streaming error: status {item.status_code}, message: {getattr(item, 'message', '')}"
                )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    client = DashScopeLLMClient()
    response = client.generate(
        prompt="请简要介绍一下急性心肌梗死的典型症状。",
        system_prompt="你是一个医学知识助手，请用简洁专业的语言回答。",
    )
    print("Response:", response)
