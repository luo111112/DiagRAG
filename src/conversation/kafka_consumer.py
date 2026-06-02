"""Kafka consumer layer for the conversation memory system."""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime

from kafka import KafkaConsumer
from kafka.errors import KafkaError

from src.config_loader import get_conversation_config
from src.conversation.models import (
    ConversationEvent,
    KafkaEventType,
    Message,
    MessageRole,
    SummaryRecord,
)
from src.conversation.mysql_store import (
    get_pending_retry_events,
    get_session_messages,
    mark_retry_failed,
    mark_retry_sent,
    save_message,
    save_summary,
    update_session,
)
from src.conversation.summarizer import generate_summary_from_event
from src.llm_client import DashScopeLLMClient

logger = logging.getLogger(__name__)

_RETRY_WORKER_INTERVAL = 30  # seconds


# =============================================================================
# Base consumer
# =============================================================================

class _BaseConsumer:
    """Base Kafka consumer with common logic (manual offset commit)."""

    def __init__(self, topics: list[str], group_id: str):
        cfg = get_conversation_config()
        kafka_cfg = cfg["kafka"]
        self._consumer = KafkaConsumer(
            *topics,
            bootstrap_servers=kafka_cfg.get("bootstrap_servers", "localhost:9092"),
            group_id=group_id,
            enable_auto_commit=False,
            auto_offset_reset="earliest",
            max_poll_records=100,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        )
        self._running = False
        self._thread: threading.Thread | None = None
        logger.info("_BaseConsumer initialised: topics=%s, group_id=%s", topics, group_id)

    def _on_message(self, event: ConversationEvent) -> None:
        """Override this in subclasses to handle a deserialised event."""
        raise NotImplementedError

    def start(self, daemon: bool = True) -> None:
        """Start consuming in a background thread."""
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=daemon)
        self._thread.start()
        logger.info("%s started in background thread.", self.__class__.__name__)

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._consumer.close()
        logger.info("%s stopped.", self.__class__.__name__)

    def _run(self) -> None:
        for message in self._consumer:
            if not self._running:
                break
            try:
                raw = message.value
                event = ConversationEvent.from_dict(raw)
                self._on_message(event)
                self._consumer.commit()
            except Exception as e:
                logger.exception("Error processing message in %s: %s",
                                self.__class__.__name__, e)


# =============================================================================
# Message consumer
# =============================================================================

class MessageConsumer(_BaseConsumer):
    """Consumes message_add events and writes them to MySQL messages table."""

    def __init__(self):
        cfg = get_conversation_config()
        topic = cfg.get("kafka", {}).get("topic", "rag-conversation-events")
        group = cfg.get("kafka", {}).get("consumer_group", "rag-msg-consumer")
        super().__init__(topics=[topic], group_id=group)

    def _on_message(self, event: ConversationEvent) -> None:
        if event.event_type != KafkaEventType.MESSAGE_ADD:
            return

        payload = event.payload
        message = Message(
            id=payload.get("message_id", ""),
            session_id=event.session_id,
            message_order=payload.get("message_order", 0),
            role=MessageRole(payload.get("role", "user")),
            content=payload.get("content", ""),
            metadata=payload.get("metadata"),
            created_at=event.timestamp,
        )

        try:
            inserted = save_message(message)
            if inserted:
                logger.debug(
                    "Message persisted: id=%s, session=%s, order=%d",
                    message.id, message.session_id, message.message_order,
                )
            else:
                logger.debug(
                    "Message already existed (idempotent skip): id=%s",
                    message.id,
                )
        except Exception as e:
            logger.error("Failed to save message %s to MySQL: %s", message.id, e)
            raise


# =============================================================================
# Summary consumer
# =============================================================================

class SummaryConsumer(_BaseConsumer):
    """Consumes generate_summary events and generates/updates summaries in MySQL."""

    def __init__(self, llm_client: DashScopeLLMClient):
        self._llm = llm_client
        cfg = get_conversation_config()
        topic = cfg.get("kafka", {}).get("topic", "rag-conversation-events")
        group = cfg.get("kafka", {}).get("consumer_group", "rag-summary-consumer")
        super().__init__(topics=[topic], group_id=group)

    def _on_message(self, event: ConversationEvent) -> None:
        if event.event_type != KafkaEventType.GENERATE_SUMMARY:
            return

        payload = event.payload
        summary_order = payload.get("summary_order", 0)
        turns_start = payload.get("turns_start", 0)
        turns_end = payload.get("turns_end", 0)
        previous_summary = payload.get("previous_summary")
        messages_to_summarize = payload.get("messages_to_summarize", [])

        try:
            summary_text = generate_summary_from_event(
                llm_client=self._llm,
                previous_summary=previous_summary,
                turns_start=turns_start,
                turns_end=turns_end,
                messages=messages_to_summarize,
            )

            record = SummaryRecord(
                id=None,
                session_id=event.session_id,
                summary_order=summary_order,
                turns_start=turns_start,
                turns_end=turns_end,
                summary_text=summary_text,
                created_at=datetime.now(),
            )
            save_summary(record)

            # Update sessions table with latest summary
            update_session(
                session_id=event.session_id,
                summary=summary_text,
                summary_order=summary_order,
            )
            logger.info(
                "Summary generated and saved: session=%s, order=%d, turns=%d-%d, len=%d",
                event.session_id, summary_order, turns_start, turns_end, len(summary_text),
            )

        except Exception as e:
            logger.error("Failed to generate/save summary for session %s: %s",
                        event.session_id, e)
            raise


# =============================================================================
# Session close consumer
# =============================================================================

class SessionCloseConsumer(_BaseConsumer):
    """Consumes session_close events and updates session status in MySQL."""

    def __init__(self):
        from src.conversation.models import SessionStatus
        self._SessionStatus = SessionStatus
        cfg = get_conversation_config()
        topic = cfg.get("kafka", {}).get("topic", "rag-conversation-events")
        super().__init__(topics=[topic], group_id="rag-session-consumer")

    def _on_message(self, event: ConversationEvent) -> None:
        if event.event_type != KafkaEventType.SESSION_CLOSE:
            return
        try:
            update_session(session_id=event.session_id, status=self._SessionStatus.ARCHIVED)
            logger.info("Session archived: id=%s", event.session_id)
        except Exception as e:
            logger.error("Failed to archive session %s: %s", event.session_id, e)
            raise


# =============================================================================
# Retry worker
# =============================================================================

class RetryWorker:
    """Background thread that polls MySQL retry queue and re-sends failed Kafka events."""

    def __init__(self, interval: int = _RETRY_WORKER_INTERVAL):
        self._interval = interval
        self._running = False
        self._thread: threading.Thread | None = None
        self._producer = None  # lazy

    def _get_producer(self):
        from src.conversation.kafka_producer import get_producer
        if self._producer is None:
            self._producer = get_producer()
        return self._producer

    def start(self, daemon: bool = True) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=daemon)
        self._thread.start()
        logger.info("RetryWorker started (interval=%ds).", self._interval)

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=10)
        logger.info("RetryWorker stopped.")

    def _run(self) -> None:
        while self._running:
            try:
                self._process_batch()
            except Exception as e:
                logger.exception("RetryWorker error: %s", e)
            time.sleep(self._interval)

    def _process_batch(self) -> None:
        events = get_pending_retry_events(limit=100)
        if not events:
            return

        logger.debug("RetryWorker: processing %d pending events.", len(events))
        topic = get_conversation_config().get("kafka", {}).get(
            "topic", "rag-conversation-events"
        )
        producer = self._get_producer()

        for row in events:
            queue_id: int = row["id"]
            event_id: str = row["event_id"]
            event_type: str = row["event_type"]
            session_id: str = row["session_id"]
            payload: dict = row["payload"]
            retry_count: int = int(row.get("retry_count", 0))

            full_event = {
                "event_id": event_id,
                "event_type": event_type,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "payload": payload,
            }

            try:
                future = producer.send(
                    topic=topic,
                    key=session_id.encode("utf-8"),
                    value=full_event,
                )
                future.get(timeout=5)
                mark_retry_sent(queue_id)
                logger.info(
                    "Retry succeeded: event_id=%s (attempt %d)",
                    event_id, retry_count + 1,
                )
            except KafkaError as e:
                new_retry_count = retry_count + 1
                from src.conversation.kafka_producer import _calc_next_retry
                from src.conversation.mysql_store import write_to_retry_queue

                write_to_retry_queue(
                    event_id=event_id,
                    event_type=event_type,
                    session_id=session_id,
                    payload=payload,
                    next_retry_at=_calc_next_retry(new_retry_count),
                )
                # Mark old row as updated (retry_count bumped via ON DUPLICATE KEY)
                if new_retry_count >= 5:
                    mark_retry_failed(queue_id, str(e))
                    logger.error(
                        "Retry permanently failed after %d attempts: event_id=%s, error=%s",
                        new_retry_count, event_id, e,
                    )
                else:
                    logger.warning(
                        "Retry failed again: event_id=%s (attempt %d), error=%s",
                        event_id, new_retry_count, e,
                    )
            except Exception as e:
                logger.error("Unexpected error in RetryWorker for event %s: %s", event_id, e)


# =============================================================================
# Start all consumers (main entry point)
# =============================================================================

_consumer_instances: list = []


def start_consumers(llm_client: DashScopeLLMClient) -> None:
    """Start all Kafka consumers and the retry worker.

    This should be called once at application startup (e.g. in api.py lifespan).

    Args:
        llm_client: LLM client used for summary generation.
    """
    global _consumer_instances

    # Initialise MySQL schema
    from src.conversation.mysql_store import init_db
    init_db()

    msg_consumer = MessageConsumer()
    msg_consumer.start()
    _consumer_instances.append(msg_consumer)

    summary_consumer = SummaryConsumer(llm_client)
    summary_consumer.start()
    _consumer_instances.append(summary_consumer)

    session_consumer = SessionCloseConsumer()
    session_consumer.start()
    _consumer_instances.append(session_consumer)

    retry_worker = RetryWorker()
    retry_worker.start()
    _consumer_instances.append(retry_worker)

    logger.info("All conversation consumers started.")


def stop_consumers() -> None:
    """Stop all running consumers. Call at application shutdown."""
    global _consumer_instances
    for inst in _consumer_instances:
        try:
            inst.stop()
        except Exception as e:
            logger.warning("Error stopping %s: %s", type(inst).__name__, e)
    _consumer_instances.clear()
    logger.info("All conversation consumers stopped.")
