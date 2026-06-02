"""Context builder for assembling LLM prompts with conversation history."""

from __future__ import annotations

import logging
from typing import Any

from src.conversation.models import Message
from src.conversation.mysql_store import (
    get_latest_summary,
    get_session,
    get_session_messages,
)
from src.conversation.redis_store import (
    get_recent_messages,
    warm_cache,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Prompt template
# =============================================================================

CONVERSATION_RAG_PROMPT_TEMPLATE = """【历史摘要】
{summary}

---

【近期对话】
{conversation_history}

---

【医学知识片段】

{context}

---

【用户问题】

{question}

---

请基于以上信息，按照以下步骤分析并回答：

**第一步 —— 症状分析**：识别问题中涉及的主要症状、体征或实验室指标，说明其临床意义。

**第二步 —— 鉴别诊断方向**：结合知识片段，列出可能的鉴别诊断方向，并简要说明支持点与不支持点。

**第三步 —— 检查与建议**：指出需要进一步哪些检查以明确诊断，以及初步处置建议。

**第四步 —— 输出格式**：将完整回答按以下 JSON Schema 组织输出，不要在 JSON 之外输出任何其他内容：

{{
  "analysis": "string,  第一步的症状分析结果",
  "differential_diagnosis": [
    {{
      "diagnosis": "string,  可能的诊断名称",
      "supporting": "string,  支持点",
      "against": "string  不支持点"
    }}
  ],
  "suggested_exams": ["string,  建议的检查项目"],
  "preliminary_advice": "string,  初步处置建议",
  "references": [
    {{
      "source": "string,  文档名称",
      "relevance": "number,  相关度评分（0-1）"
    }}
  ],
  "safety_alert": "string | null,  若涉及急危重症则显示安全提示，否则为 null"
}}

**格式约束**：
- 回答必须是一个合法 JSON 对象，不要添加 markdown 代码块标记。
- 所有字段必须存在，不可省略。
- 若某字段无内容，填入空字符串或空数组。
- 相关度评分保留两位小数。
"""


# =============================================================================
# Context loader
# =============================================================================

def build_llm_context(session_id: str) -> dict[str, Any]:
    """Load conversation context for LLM prompt injection.

    Reads from Redis first (hot path). On a cache miss, falls back to MySQL
    and repopulates the Redis cache.

    Args:
        session_id: The conversation session ID.

    Returns:
        A dict with keys:
            - messages: list[Message], most recent messages (oldest first)
            - summary: str | None, the latest summary text
            - summary_order: int, the summary order number
            - redis_hit: bool, whether the data came from Redis
    """
    # 1. Try Redis hot path
    messages = get_recent_messages(session_id)
    redis_hit = len(messages) > 0

    if redis_hit:
        session = get_session(session_id)
        return {
            "messages": messages,
            "summary": session.summary if session else None,
            "summary_order": session.summary_order if session else 0,
            "redis_hit": True,
        }

    # 2. Cache miss — fall back to MySQL
    logger.debug("Redis cache miss for session %s, loading from MySQL.", session_id)
    all_messages = get_session_messages(session_id)

    # Warm Redis cache with the loaded messages
    if all_messages:
        warm_cache(session_id, all_messages)

    session = get_session(session_id)

    return {
        "messages": all_messages,
        "summary": session.summary if session else None,
        "summary_order": session.summary_order if session else 0,
        "redis_hit": False,
    }


# =============================================================================
# Prompt assembly
# =============================================================================

def assemble_conversation_prompt(
    session_id: str,
    question: str,
    context: str,
) -> dict[str, str]:
    """Assemble a complete LLM prompt with conversation history and medical context.

    Args:
        session_id: Conversation session ID.
        question: The user's current question.
        context: The retrieved medical knowledge chunks as a string.

    Returns:
        A dict with keys:
            - user_prompt: The assembled user prompt string.
            - summary: The session summary (or None).
            - redis_hit: Whether Redis was hit.
    """
    ctx = build_llm_context(session_id)
    messages = ctx["messages"]
    summary = ctx["summary"]

    # Build conversation history string
    history_parts = []
    for msg in messages:
        role_label = "用户" if msg.role.value == "user" else "助手"
        history_parts.append(f"{role_label}：{msg.content}")

    history_text = "\n".join(history_parts) if history_parts else "（暂无历史对话）"

    # Build summary section
    summary_text = summary if summary else "（该会话暂无历史摘要）"

    user_prompt = CONVERSATION_RAG_PROMPT_TEMPLATE.format(
        summary=summary_text,
        conversation_history=history_text,
        context=context,
        question=question,
    )

    return {
        "user_prompt": user_prompt,
        "summary": summary,
        "redis_hit": ctx["redis_hit"],
    }


def assemble_stream_prompt(
    session_id: str,
    question: str,
    context: str,
) -> tuple[str, str | None, bool]:
    """Stream-compatible version of assemble_conversation_prompt.

    Returns (user_prompt, summary, redis_hit).
    """
    result = assemble_conversation_prompt(session_id, question, context)
    return result["user_prompt"], result["summary"], result["redis_hit"]


# =============================================================================
# Utility: fetch messages only (no prompt assembly)
# =============================================================================

def get_conversation_history(session_id: str, limit_turns: int = 20) -> list[Message]:
    """Fetch recent messages for a session, using Redis first.

    Args:
        session_id: Session ID.
        limit_turns: Maximum number of turns to return.

    Returns:
        List of Message objects (oldest first).
    """
    messages = get_recent_messages(session_id, limit_turns=limit_turns)
    if messages:
        return messages
    # Fall back to MySQL
    db_messages = get_session_messages(session_id, limit=limit_turns * 2)
    if db_messages:
        warm_cache(session_id, db_messages)
    return db_messages
