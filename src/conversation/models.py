"""Data models for the conversation memory system."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


# =============================================================================
# Enums
# =============================================================================


class MessageRole(str, Enum):
    """Role of a message in a conversation."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class SessionStatus(str, Enum):
    """Lifecycle status of a conversation session."""

    ACTIVE = "active"      # 活跃会话
    ARCHIVED = "archived"  # 已归档
    DELETED = "deleted"    # 已删除


class KafkaEventType(str, Enum):
    """Types of events sent to Kafka."""

    MESSAGE_ADD = "message_add"
    GENERATE_SUMMARY = "generate_summary"
    SESSION_CLOSE = "session_close"


class RetryStatus(str, Enum):
    """Processing status of a Kafka retry queue entry."""

    PENDING = "PENDING"
    SENT = "SENT"
    FAILED = "FAILED"


# =============================================================================
# Core domain models
# =============================================================================


@dataclass
class Message:
    """A single message within a conversation session."""

    session_id: str
    message_order: int
    role: MessageRole
    content: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: dict[str, Any] | None = None
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        """Sanitize metadata at construction time to drop non-JSON-serializable values."""
        if self.metadata is not None:
            self.metadata = self._serialize_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "message_order": self.message_order,
            "role": self.role.value,
            "content": self.content,
            "metadata": self._serialize_metadata(self.metadata),
            "created_at": self.created_at.isoformat(),
        }

    @staticmethod
    def _serialize_metadata(meta: dict[str, Any] | None) -> dict[str, Any]:
        """Safely serialize metadata, dropping non-JSON-serializable values."""
        if meta is None:
            return {}
        result: dict[str, Any] = {}
        for k, v in meta.items():
            try:
                import json
                json.dumps(v)
                result[k] = v
            except Exception:
                try:
                    result[k] = repr(v)
                except Exception:
                    pass
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        role_str = data.get("role", "user")
        if isinstance(role_str, MessageRole):
            role = role_str
        else:
            role = MessageRole(role_str)
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            session_id=data["session_id"],
            message_order=data["message_order"],
            role=role,
            content=data["content"],
            metadata=data.get("metadata"),
            created_at=(
                datetime.fromisoformat(data["created_at"])
                if isinstance(data.get("created_at"), str)
                else (data.get("created_at") or datetime.now())
            ),
        )

    def to_json_str(self) -> str:
        import json

        def _safe_default(obj: Any) -> str:
            return repr(obj)

        return json.dumps(self.to_dict(), ensure_ascii=False, default=_safe_default)

    @classmethod
    def from_json_str(cls, s: str) -> "Message":
        import json

        return cls.from_dict(json.loads(s))


@dataclass
class Session:
    """Metadata for a single conversation session."""

    id: str
    user_id: str
    title: str = ""
    status: SessionStatus = SessionStatus.ACTIVE
    summary: str | None = None
    summary_order: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    expired_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "title": self.title,
            "status": self.status.value,
            "summary": self.summary,
            "summary_order": self.summary_order,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "expired_at": self.expired_at.isoformat() if self.expired_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Session":
        status_str = data.get("status", "active")
        if isinstance(status_str, SessionStatus):
            status = status_str
        else:
            status = SessionStatus(status_str)
        return cls(
            id=data["id"],
            user_id=data["user_id"],
            title=data.get("title", ""),
            status=status,
            summary=data.get("summary"),
            summary_order=int(data.get("summary_order", 0)),
            created_at=(
                datetime.fromisoformat(data["created_at"])
                if isinstance(data.get("created_at"), str)
                else (data.get("created_at") or datetime.now())
            ),
            updated_at=(
                datetime.fromisoformat(data["updated_at"])
                if isinstance(data.get("updated_at"), str)
                else (data.get("updated_at") or datetime.now())
            ),
            expired_at=(
                datetime.fromisoformat(data["expired_at"])
                if isinstance(data.get("expired_at"), str) and data.get("expired_at")
                else data.get("expired_at")
            ),
        )


@dataclass
class SummaryRecord:
    """A periodic summary of a conversation session."""

    id: int | None
    session_id: str
    summary_order: int
    turns_start: int
    turns_end: int
    summary_text: str
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "summary_order": self.summary_order,
            "turns_start": self.turns_start,
            "turns_end": self.turns_end,
            "summary_text": self.summary_text,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SummaryRecord":
        return cls(
            id=data.get("id"),
            session_id=data["session_id"],
            summary_order=int(data["summary_order"]),
            turns_start=int(data["turns_start"]),
            turns_end=int(data["turns_end"]),
            summary_text=data["summary_text"],
            created_at=(
                datetime.fromisoformat(data["created_at"])
                if isinstance(data.get("created_at"), str)
                else (data.get("created_at") or datetime.now())
            ),
        )


# =============================================================================
# Kafka event models
# =============================================================================


@dataclass
class ConversationEvent:
    """A Kafka event for the conversation system."""

    event_id: str
    event_type: KafkaEventType
    session_id: str
    timestamp: datetime
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConversationEvent":
        event_type_str = data.get("event_type", "message_add")
        if isinstance(event_type_str, KafkaEventType):
            event_type = event_type_str
        else:
            event_type = KafkaEventType(event_type_str)
        return cls(
            event_id=data.get("event_id", str(uuid.uuid4())),
            event_type=event_type,
            session_id=data["session_id"],
            timestamp=(
                datetime.fromisoformat(data["timestamp"])
                if isinstance(data.get("timestamp"), str)
                else (data.get("timestamp") or datetime.now())
            ),
            payload=data.get("payload", {}),
        )


def build_message_event(session_id: str, message: Message) -> ConversationEvent:
    """Factory: build a message_add event from a Message object."""
    return ConversationEvent(
        event_id=str(uuid.uuid4()),
        event_type=KafkaEventType.MESSAGE_ADD,
        session_id=session_id,
        timestamp=message.created_at,
        payload={
            "message_id": message.id,
            "message_order": message.message_order,
            "role": message.role.value,
            "content": message.content,
            "metadata": message.metadata or {},
        },
    )


def build_summary_event(
    session_id: str,
    summary_order: int,
    turns_start: int,
    turns_end: int,
    previous_summary: str | None,
    messages_to_summarize: list[dict[str, str]],
) -> ConversationEvent:
    """Factory: build a generate_summary event."""
    return ConversationEvent(
        event_id=str(uuid.uuid4()),
        event_type=KafkaEventType.GENERATE_SUMMARY,
        session_id=session_id,
        timestamp=datetime.now(),
        payload={
            "summary_order": summary_order,
            "turns_start": turns_start,
            "turns_end": turns_end,
            "previous_summary": previous_summary,
            "messages_to_summarize": messages_to_summarize,
        },
    )


def build_session_close_event(session_id: str, total_turns: int, final_summary_generated: bool) -> ConversationEvent:
    """Factory: build a session_close event."""
    return ConversationEvent(
        event_id=str(uuid.uuid4()),
        event_type=KafkaEventType.SESSION_CLOSE,
        session_id=session_id,
        timestamp=datetime.now(),
        payload={
            "total_turns": total_turns,
            "final_summary_generated": final_summary_generated,
        },
    )
