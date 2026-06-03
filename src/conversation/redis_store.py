"""Redis cache layer for the conversation memory system (short-term memory)."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

import redis

from src.config_loader import get_conversation_config
from src.conversation.models import Message

logger = logging.getLogger(__name__)


# =============================================================================
# Key patterns
# =============================================================================

_SESSION_HASH_KEY = "session:{session_id}"           # Hash
_MESSAGES_LIST_KEY = "messages:{session_id}"        # List
_USER_SESSIONS_KEY = "user:sessions:{user_id}"      # Sorted Set
_MESSAGE_COUNTER_KEY = "msg:counter:{session_id}"    # String (atomic counter)


# =============================================================================
# Redis client (lazy singleton)
# =============================================================================

_redis_client: redis.Redis | None = None


def get_redis_client() -> redis.Redis:
    """Return the global Redis client, creating it on first call."""
    global _redis_client
    if _redis_client is None:
        cfg = get_conversation_config()
        redis_cfg = cfg["redis"]
        password = redis_cfg.get("password") or None
        _redis_client = redis.Redis(
            host=redis_cfg.get("host", "localhost"),
            port=int(redis_cfg.get("port", 6379)),
            db=int(redis_cfg.get("db", 0)),
            password=password,
            decode_responses=True,
            socket_timeout=5.0,
            socket_connect_timeout=5.0,
            retry_on_timeout=True,
        )
        logger.info("Redis client created: %s:%s/%d",
                    redis_cfg.get("host"), redis_cfg.get("port"), redis_cfg.get("db", 0))
    return _redis_client


# =============================================================================
# Message write (Pipeline)
# =============================================================================

def write_message(message: Message) -> int:
    """Write a message to Redis using a Pipeline.

    Performs the following atomically via Pipeline:
        1. INCR message order counter
        2. RPUSH message JSON to the messages list
        3. LTRIM to keep only the last 20 entries
        4. HSET session metadata (updated_at, last_message_order, summary_order)
        5. EXPIRE session hash (refresh TTL)
        6. EXPIRE messages list (refresh TTL)

    Returns the message_order that was used (either generated or the existing one).
    """
    cfg = get_conversation_config()
    ttl_seconds = int(cfg.get("redis_ttl_days", 7) * 86400)
    max_messages = int(cfg.get("redis_cache_turns", 10)) * 2  # 2 messages per turn

    client = get_redis_client()
    pipe = client.pipeline(transaction=True)

    # Serialize message BEFORE assigning message_order from pipeline
    # (pipeline commands return lazy chain objects, not real values)
    msg_json = message.to_json_str()

    # 1. Atomically generate message_order
    counter_key = _MESSAGE_COUNTER_KEY.format(session_id=message.session_id)
    pipe.incr(counter_key)

    msg_list_key = _MESSAGES_LIST_KEY.format(session_id=message.session_id)
    pipe.rpush(msg_list_key, msg_json)

    # 3. Keep only the most recent N messages
    pipe.ltrim(msg_list_key, -max_messages, -1)

    # 4. Update session hash
    session_key = _SESSION_HASH_KEY.format(session_id=message.session_id)
    pipe.hset(session_key, mapping={
        "updated_at": datetime.now().isoformat(),
        "user_id": message.session_id.split("-")[0],  # placeholder; caller should set properly
        "status": "active",
    })

    # 5 & 6. Refresh TTLs
    pipe.expire(session_key, ttl_seconds)
    pipe.expire(msg_list_key, ttl_seconds)

    # 7. Add session to user's sorted set (score = now timestamp)
    # Note: user_id should be set by caller via update_session_meta
    user_sessions_key = _USER_SESSIONS_KEY.format(user_id=message.session_id.split("-")[0])
    pipe.zadd(user_sessions_key, {message.session_id: datetime.now().timestamp()})

    results = pipe.execute()
    message_order = int(results[0])  # INCR result is at index 0

    # Update message_order AFTER execute() — now it's a real integer
    message.message_order = message_order

    # Update session hash with the real message_order
    get_redis_client().hset(session_key, "last_message_order", message_order)

    logger.debug(
        "Redis write_message: session=%s, order=%s, ttl=%ds",
        message.session_id, message_order, ttl_seconds,
    )
    return message_order


def update_session_meta(
    session_id: str,
    user_id: str,
    status: str = "active",
    summary_order: int = 0,
) -> None:
    """Update or create the session metadata hash in Redis."""
    cfg = get_conversation_config()
    ttl_seconds = int(cfg.get("redis_ttl_days", 7) * 86400)

    client = get_redis_client()
    session_key = _SESSION_HASH_KEY.format(session_id=session_id)
    user_sessions_key = _USER_SESSIONS_KEY.format(user_id=user_id)

    pipe = client.pipeline(transaction=True)
    pipe.hset(session_key, mapping={
        "user_id": user_id,
        "status": status,
        "updated_at": datetime.now().isoformat(),
        "summary_order": str(summary_order),
    })
    pipe.expire(session_key, ttl_seconds)
    pipe.zadd(user_sessions_key, {session_id: datetime.now().timestamp()})
    pipe.execute()


# =============================================================================
# Message read
# =============================================================================

def get_recent_messages(session_id: str, limit_turns: int | None = None) -> list[Message]:
    """Read the most recent messages for a session from Redis.

    Args:
        session_id: Session ID.
        limit_turns: If specified, return at most this many turns
                      (i.e. limit_turns * 2 messages).

    Returns:
        List of Message objects, oldest first.
    """
    cfg = get_conversation_config()
    if limit_turns is None:
        limit_turns = cfg.get("redis_cache_turns", 10)
    max_messages = limit_turns * 2

    client = get_redis_client()
    msg_list_key = _MESSAGES_LIST_KEY.format(session_id=session_id)

    raw = client.lrange(msg_list_key, -max_messages, -1)
    messages = []
    for item in raw:
        try:
            messages.append(Message.from_json_str(item))
        except Exception as e:
            logger.warning("Failed to deserialize message from Redis: %s", e)
    return messages


def get_all_messages(session_id: str) -> list[Message]:
    """Read all cached messages for a session (up to the LTRIM limit)."""
    client = get_redis_client()
    msg_list_key = _MESSAGES_LIST_KEY.format(session_id=session_id)
    raw = client.lrange(msg_list_key, 0, -1)
    messages = []
    for item in raw:
        try:
            messages.append(Message.from_json_str(item))
        except Exception as e:
            logger.warning("Failed to deserialize message from Redis: %s", e)
    return messages


# =============================================================================
# Session metadata
# =============================================================================

def get_session_meta(session_id: str) -> dict[str, Any] | None:
    """Read the session metadata hash from Redis. Returns None if not found."""
    client = get_redis_client()
    session_key = _SESSION_HASH_KEY.format(session_id=session_id)
    data = client.hgetall(session_key)
    if not data:
        return None
    return data


def get_current_message_order(session_id: str) -> int:
    """Get the current message order counter for a session.

    Returns 0 if the counter does not exist.
    """
    client = get_redis_client()
    counter_key = _MESSAGE_COUNTER_KEY.format(session_id=session_id)
    val = client.get(counter_key)
    return int(val) if val else 0


# =============================================================================
# Cache warm-up (cold start from MySQL)
# =============================================================================

def warm_cache(session_id: str, messages: list[Message]) -> None:
    """Populate Redis cache from a list of messages (e.g. loaded from MySQL).

    Clears any existing cache entries for the session first, then repopulates
    with the provided messages.
    """
    cfg = get_conversation_config()
    ttl_seconds = int(cfg.get("redis_ttl_days", 7) * 86400)
    max_messages = int(cfg.get("redis_cache_turns", 10)) * 2

    client = get_redis_client()
    msg_list_key = _MESSAGES_LIST_KEY.format(session_id=session_id)
    session_key = _SESSION_HASH_KEY.format(session_id=session_id)

    pipe = client.pipeline(transaction=True)

    # Clear existing
    pipe.delete(msg_list_key)

    # Repopulate (most recent first in the list, but we RPUSH so they end up correct)
    for msg in messages[-max_messages:]:
        pipe.rpush(msg_list_key, msg.to_json_str())

    pipe.expire(msg_list_key, ttl_seconds)

    # Sync the counter to the max message_order seen
    if messages:
        max_order = max(m.message_order for m in messages)
        counter_key = _MESSAGE_COUNTER_KEY.format(session_id=session_id)
        pipe.set(counter_key, max_order)
        pipe.expire(counter_key, ttl_seconds)

    pipe.execute()
    logger.info("Redis cache warmed for session %s with %d messages", session_id, len(messages))


# =============================================================================
# Cache invalidation
# =============================================================================

def delete_session_cache(session_id: str, user_id: str | None = None) -> None:
    """Delete all Redis keys associated with a session."""
    client = get_redis_client()
    msg_list_key = _MESSAGES_LIST_KEY.format(session_id=session_id)
    session_key = _SESSION_HASH_KEY.format(session_id=session_id)
    counter_key = _MESSAGE_COUNTER_KEY.format(session_id=session_id)

    keys_to_delete = [msg_list_key, session_key, counter_key]
    if user_id:
        user_sessions_key = _USER_SESSIONS_KEY.format(user_id=user_id)
        client.zrem(user_sessions_key, session_id)

    client.delete(*keys_to_delete)
    logger.debug("Redis cache deleted for session %s", session_id)


# =============================================================================
# TTL refresh
# =============================================================================

def refresh_ttl(session_id: str) -> None:
    """Refresh the TTL on session and message list keys."""
    cfg = get_conversation_config()
    ttl_seconds = int(cfg.get("redis_ttl_days", 7) * 86400)

    client = get_redis_client()
    msg_list_key = _MESSAGES_LIST_KEY.format(session_id=session_id)
    session_key = _SESSION_HASH_KEY.format(session_id=session_id)

    pipe = client.pipeline()
    pipe.expire(session_key, ttl_seconds)
    pipe.expire(msg_list_key, ttl_seconds)
    pipe.execute()
