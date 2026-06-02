"""Kafka producer layer for the conversation memory system."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta

from kafka import KafkaProducer
from kafka.errors import KafkaError

from src.config_loader import get_conversation_config
from src.conversation.models import (
    ConversationEvent,
    Message,
    build_message_event,
    build_session_close_event,
    build_summary_event,
)
from src.conversation.mysql_store import write_to_retry_queue

logger = logging.getLogger(__name__)


# =============================================================================
# Producer singleton
# =============================================================================

_producer: KafkaProducer | None = None

_BASE_RETRY_INTERVAL = 10  # seconds


def _calc_next_retry(retry_count: int) -> datetime:
    """Exponential backoff capped at 30 minutes."""
    interval = min(_BASE_RETRY_INTERVAL * (2 ** retry_count), 1800)
    return datetime.now() + timedelta(seconds=interval)


def get_producer() -> KafkaProducer:
    """Return the global KafkaProducer, creating it on first call.

    Configured with idempotence=True and acks='all' for exactly-once semantics.
    """
    global _producer
    if _producer is None:
        cfg = get_conversation_config()
        kafka_cfg = cfg["kafka"]
        _producer = KafkaProducer(
            bootstrap_servers=kafka_cfg.get("bootstrap_servers", "localhost:9092"),
            acks=kafka_cfg.get("acks", "all"),
            retries=int(kafka_cfg.get("retries", 3)),
            enable_idempotence=True,
            key_serializer=lambda k: k.encode() if k else None,
            value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
            max_block_ms=5000,
        )
        logger.info("KafkaProducer created: %s", kafka_cfg.get("bootstrap_servers"))
    return _producer


def _get_topic() -> str:
    cfg = get_conversation_config()
    return cfg.get("kafka", {}).get("topic", "rag-conversation-events")


# =============================================================================
# Core send logic with fallback to MySQL retry queue
# =============================================================================

def _send_event(event: ConversationEvent) -> bool:
    """Attempt to send a Kafka event. On failure, write to MySQL retry queue.

    Returns True if sent successfully, False if written to the retry queue.
    """
    producer = get_producer()
    topic = _get_topic()
    try:
        future = producer.send(
            topic=topic,
            key=event.session_id.encode("utf-8"),
            value=event.to_dict(),
        )
        future.get(timeout=5)
        logger.debug(
            "Kafka event sent: event_id=%s, event_type=%s, session_id=%s",
            event.event_id, event.event_type.value, event.session_id,
        )
        return True
    except KafkaError as e:
        logger.warning(
            "Kafka send failed for event %s (%s): %s. Writing to MySQL retry queue.",
            event.event_id, event.event_type.value, e,
        )
        _write_fallback(event)
        return False
    except Exception as e:
        logger.warning(
            "Unexpected Kafka send error for event %s (%s): %s. Writing to MySQL retry queue.",
            event.event_id, event.event_type.value, e,
        )
        _write_fallback(event)
        return False


def _write_fallback(event: ConversationEvent) -> None:
    """Write an event to the MySQL retry queue when Kafka is unavailable."""
    try:
        write_to_retry_queue(
            event_id=event.event_id,
            event_type=event.event_type.value,
            session_id=event.session_id,
            payload=event.payload,
            next_retry_at=_calc_next_retry(0),
        )
    except Exception as fallback_err:
        # Last resort: at least log the event so it can be manually recovered
        logger.error(
            "FATAL: Failed to write event to retry queue as well. "
            "Event may be lost. event_id=%s, event_type=%s, error=%s",
            event.event_id, event.event_type.value, fallback_err,
        )


# =============================================================================
# Public API
# =============================================================================

def send_message_event(session_id: str, message: Message) -> bool:
    """Send a message_add event to Kafka.

    Returns True if sent successfully, False if written to retry queue.
    """
    event = build_message_event(session_id, message)
    return _send_event(event)


def send_summary_event(
    session_id: str,
    summary_order: int,
    turns_start: int,
    turns_end: int,
    previous_summary: str | None,
    messages_to_summarize: list[dict[str, str]],
) -> bool:
    """Send a generate_summary event to Kafka.

    Returns True if sent successfully, False if written to retry queue.
    """
    event = build_summary_event(
        session_id=session_id,
        summary_order=summary_order,
        turns_start=turns_start,
        turns_end=turns_end,
        previous_summary=previous_summary,
        messages_to_summarize=messages_to_summarize,
    )
    return _send_event(event)


def send_session_close_event(session_id: str, total_turns: int, final_summary_generated: bool) -> bool:
    """Send a session_close event to Kafka.

    Returns True if sent successfully, False if written to retry queue.
    """
    event = build_session_close_event(
        session_id=session_id,
        total_turns=total_turns,
        final_summary_generated=final_summary_generated,
    )
    return _send_event(event)


def flush_producer() -> None:
    """Flush and close the global Kafka producer."""
    global _producer
    if _producer is not None:
        _producer.flush()
        _producer.close()
        _producer = None
        logger.info("KafkaProducer closed.")
