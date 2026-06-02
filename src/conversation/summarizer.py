"""Summary generator for the conversation memory system (medium-term memory)."""

from __future__ import annotations

import logging
from typing import Any

from src.config_loader import get_conversation_config
from src.llm_client import DashScopeLLMClient

logger = logging.getLogger(__name__)


# =============================================================================
# Prompt template
# =============================================================================

SUMMARY_PROMPT_TEMPLATE = """你是一个医学对话摘要助手。请将以下医学对话历史压缩为一段结构化摘要。

【上一次摘要】
{previous_summary}

【待摘要的新对话（第{turns_start}轮到第{turns_end}轮）】
{messages_text}

---
请将以上新对话摘要为 200 字以内的结构化文本，必须包含以下四个部分：

**病史摘要**：患者主诉和症状（关键词）
**诊断方向**：已考虑的鉴别诊断方向
**已查项目**：已建议或已完成的检查项目
**待解决问题**：当前尚未解决或需要进一步明确的临床问题

格式要求：
- 四个部分用换行分隔，每部分以 **标题** 开头
- 仅输出摘要文本，不要输出任何解释或说明文字
- 使用简体中文
"""


# =============================================================================
# Prompt builder
# =============================================================================

def build_summary_prompt(
    previous_summary: str | None,
    turns_start: int,
    turns_end: int,
    messages: list[dict[str, str]],
) -> str:
    """Build the prompt for the summary LLM call.

    Args:
        previous_summary: The previous summary text (or None for the first summary).
        turns_start: Starting turn number (1-indexed).
        turns_end: Ending turn number (inclusive).
        messages: List of message dicts with keys: role, content.

    Returns:
        The fully assembled prompt string.
    """
    prev_text = previous_summary if previous_summary else "（首轮，无历史摘要）"

    messages_lines = []
    for m in messages:
        role_label = "用户" if m.get("role") == "user" else "助手"
        messages_lines.append(f"{role_label}：{m.get('content', '')}")

    messages_text = "\n".join(messages_lines)

    return SUMMARY_PROMPT_TEMPLATE.format(
        previous_summary=prev_text,
        turns_start=turns_start,
        turns_end=turns_end,
        messages_text=messages_text,
    )


# =============================================================================
# Summary generation
# =============================================================================

def generate_summary_from_event(
    llm_client: DashScopeLLMClient,
    previous_summary: str | None,
    turns_start: int,
    turns_end: int,
    messages: list[dict[str, str]],
    system_prompt: str = "你是一个医学对话摘要助手。",
) -> str:
    """Generate a summary by calling the LLM.

    Args:
        llm_client: Initialised LLM client.
        previous_summary: Previous summary text (or None).
        turns_start: Starting turn number.
        turns_end: Ending turn number.
        messages: List of message dicts to summarise.
        system_prompt: System prompt for the summary LLM.

    Returns:
        The generated summary text.

    Raises:
        Exception: If the LLM call fails.
    """
    cfg = get_conversation_config()
    model = cfg.get("summary_llm_model", "qwen-plus")
    temperature = float(cfg.get("summary_llm_temperature", 0.1))
    max_tokens = int(cfg.get("summary_max_tokens", 500))

    prompt = build_summary_prompt(
        previous_summary=previous_summary,
        turns_start=turns_start,
        turns_end=turns_end,
        messages=messages,
    )

    logger.info(
        "Generating summary: turns=%d-%d, model=%s, messages_count=%d",
        turns_start, turns_end, model, len(messages),
    )

    # Build a temporary LLM client with the summary model config
    from src.llm_client import DashScopeLLMClient
    summary_llm = DashScopeLLMClient(
        model_name=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    summary_text = summary_llm.generate(prompt=prompt, system_prompt=system_prompt)

    if not summary_text or not summary_text.strip():
        raise ValueError("LLM returned an empty summary.")

    logger.info(
        "Summary generated: turns=%d-%d, length=%d chars",
        turns_start, turns_end, len(summary_text),
    )
    return summary_text.strip()


# =============================================================================
# Trigger condition
# =============================================================================

def should_trigger_summary(
    current_message_order: int,
    last_summary_order: int,
    summary_interval: int | None = None,
) -> bool:
    """Determine whether a summary should be triggered after a new message.

    A summary is triggered when the number of new messages (since the last
    summary) reaches the configured interval.

    Args:
        current_message_order: The message_order of the most recent message.
        last_summary_order: The message_order at which the last summary was generated.
        summary_interval: Override the interval from config (defaults to config value).

    Returns:
        True if a summary should be triggered, False otherwise.
    """
    if summary_interval is None:
        cfg = get_conversation_config()
        summary_interval = int(cfg.get("summary_interval_turns", 5))

    messages_since_last_summary = current_message_order - last_summary_order
    return messages_since_last_summary >= summary_interval
