"""MySQL persistence layer for the conversation memory system."""

from __future__ import annotations

import json
import logging
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any, Generator

import mysql.connector
from mysql.connector import pooling

from src.config_loader import get_conversation_config
from src.conversation.models import (
    Message,
    MessageRole,
    RetryStatus,
    Session,
    SessionStatus,
    SummaryRecord,
)

logger = logging.getLogger(__name__)


# =============================================================================
# SQL DDL
# =============================================================================

_DDL_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS sessions (
        id            VARCHAR(36) PRIMARY KEY,
        user_id       VARCHAR(64) NOT NULL,
        title         VARCHAR(255) DEFAULT '',
        status        ENUM('active','archived','deleted') NOT NULL DEFAULT 'active',
        summary       TEXT,
        summary_order INT UNSIGNED DEFAULT 0,
        created_at    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        expired_at    DATETIME,
        INDEX idx_user_updated (user_id, updated_at DESC),
        INDEX idx_expired (expired_at, status)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS messages (
        id             BIGINT AUTO_INCREMENT PRIMARY KEY,
        session_id     VARCHAR(36) NOT NULL,
        message_order  INT UNSIGNED NOT NULL,
        role           ENUM('user','assistant','system') NOT NULL,
        content        TEXT NOT NULL,
        metadata       JSON,
        created_at     DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
        UNIQUE INDEX idx_unique_msg (session_id, message_order),
        INDEX idx_session_time (session_id, created_at)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS session_summaries (
        id             BIGINT AUTO_INCREMENT PRIMARY KEY,
        session_id     VARCHAR(36) NOT NULL,
        summary_order  INT UNSIGNED NOT NULL,
        turns_start    INT UNSIGNED NOT NULL,
        turns_end      INT UNSIGNED NOT NULL,
        summary_text   TEXT NOT NULL,
        created_at     DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
        INDEX idx_session_order (session_id, summary_order)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS kafka_retry_queue (
        id             BIGINT AUTO_INCREMENT PRIMARY KEY,
        event_id       VARCHAR(36) NOT NULL,
        event_type     VARCHAR(32) NOT NULL,
        session_id     VARCHAR(36) NOT NULL,
        payload        JSON NOT NULL,
        retry_count    INT UNSIGNED DEFAULT 0,
        max_retries    INT UNSIGNED DEFAULT 5,
        status         ENUM('PENDING','SENT','FAILED') NOT NULL DEFAULT 'PENDING',
        next_retry_at  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        last_error     TEXT,
        created_at     DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at     DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        INDEX idx_status_retry (status, next_retry_at),
        INDEX idx_event_id (event_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS audit_log (
        id           BIGINT AUTO_INCREMENT PRIMARY KEY,
        operator     VARCHAR(64),
        action       VARCHAR(32) NOT NULL,
        session_id   VARCHAR(36),
        resource     VARCHAR(64),
        detail       JSON,
        ip_address   VARCHAR(45),
        created_at   DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_session (session_id),
        INDEX idx_time (created_at)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
]


# =============================================================================
# Connection pool
# =============================================================================

_pool: pooling.MySQLConnectionPool | None = None


def _get_pool() -> pooling.MySQLConnectionPool:
    global _pool
    if _pool is None:
        cfg = get_conversation_config()
        mysql_cfg = cfg["mysql"]
        _pool = pooling.MySQLConnectionPool(
            pool_name="diagrag_conv",
            pool_size=mysql_cfg.get("pool_size", 10),
            pool_reset_session=True,
            host=mysql_cfg.get("host", "localhost"),
            port=int(mysql_cfg.get("port", 3306)),
            user=mysql_cfg.get("user", "root"),
            password=mysql_cfg.get("password", ""),
            database=mysql_cfg.get("database", "diagrag"),
            charset="utf8mb4",
            collation="utf8mb4_unicode_ci",
            autocommit=False,
        )
        logger.info("MySQL connection pool created: %s:%s/%s",
                    mysql_cfg.get("host"), mysql_cfg.get("port"), mysql_cfg.get("database"))
    return _pool


@contextmanager
def get_connection() -> Generator:
    """Context manager that yields a MySQL connection from the pool.

    The connection is automatically returned to the pool on exit.
    If autocommit is False (default), caller must commit() explicitly.
    """
    pool = _get_pool()
    conn = pool.get_connection()
    try:
        yield conn
    finally:
        conn.close()


# =============================================================================
# Database initialisation
# =============================================================================

def init_db() -> None:
    """Run all DDL statements to create tables if they do not exist."""
    logger.info("Initialising MySQL database tables ...")
    with get_connection() as conn:
        cursor = conn.cursor()
        try:
            for ddl in _DDL_STATEMENTS:
                for statement in ddl.strip().split(";"):
                    stmt = statement.strip()
                    if stmt:
                        cursor.execute(stmt)
            conn.commit()
            logger.info("All database tables initialised successfully.")
        finally:
            cursor.close()


# =============================================================================
# Session operations
# =============================================================================

def create_session(
    user_id: str,
    title: str = "",
    ttl_days: int = 30,
) -> Session:
    """Create a new conversation session and persist it to MySQL.

    Args:
        user_id: User identifier.
        title: Session title (defaults to empty, can be set later).
        ttl_days: Number of days until expiration (for expired_at).

    Returns:
        The created Session object.
    """
    now = datetime.now()
    session = Session(
        id=str(uuid.uuid4()),
        user_id=user_id,
        title=title,
        status=SessionStatus.ACTIVE,
        summary=None,
        summary_order=0,
        created_at=now,
        updated_at=now,
        expired_at=now + timedelta(days=ttl_days),
    )
    with get_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                INSERT INTO sessions (id, user_id, title, status, summary, summary_order,
                                      created_at, updated_at, expired_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    session.id,
                    session.user_id,
                    session.title,
                    session.status.value,
                    session.summary,
                    session.summary_order,
                    session.created_at,
                    session.updated_at,
                    session.expired_at,
                ),
            )
            conn.commit()
        finally:
            cursor.close()
    logger.debug("Session created: id=%s, user_id=%s", session.id, session.user_id)
    return session


def get_session(session_id: str) -> Session | None:
    """Fetch a session by ID. Returns None if not found."""
    with get_connection() as conn:
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute(
                "SELECT * FROM sessions WHERE id = %s",
                (session_id,),
            )
            row = cursor.fetchone()
        finally:
            cursor.close()
    if row is None:
        return None
    return Session.from_dict(row)


def update_session(
    session_id: str,
    summary: str | None = None,
    summary_order: int | None = None,
    status: SessionStatus | None = None,
    title: str | None = None,
    updated_at: datetime | None = None,
) -> None:
    """Update mutable fields on a session. Only provided fields are updated."""
    set_clauses: list[str] = []
    values: list = []

    if summary is not None:
        set_clauses.append("summary = %s")
        values.append(summary)
    if summary_order is not None:
        set_clauses.append("summary_order = %s")
        values.append(summary_order)
    if status is not None:
        set_clauses.append("status = %s")
        values.append(status.value)
    if title is not None:
        set_clauses.append("title = %s")
        values.append(title)
    if updated_at is not None:
        set_clauses.append("updated_at = %s")
        values.append(updated_at)

    if not set_clauses:
        return

    values.append(session_id)
    sql = f"UPDATE sessions SET {', '.join(set_clauses)} WHERE id = %s"
    with get_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(sql, values)
            conn.commit()
        finally:
            cursor.close()


# =============================================================================
# Message operations
# =============================================================================

def save_message(message: Message) -> bool:
    """Persist a single message using idempotent INSERT ON DUPLICATE KEY UPDATE.

    Returns True if the message was inserted (new), False if it already existed.
    Raises on unexpected database errors.
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                INSERT INTO messages
                    (session_id, message_order, role, content, metadata, created_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE content = content
                """,
                (
                    message.session_id,
                    message.message_order,
                    message.role.value,
                    message.content,
                    json.dumps(message.metadata) if message.metadata else None,
                    message.created_at,
                ),
            )
            conn.commit()
            affected = cursor.rowcount
            return affected > 0
        finally:
            cursor.close()


def get_session_messages(
    session_id: str,
    limit: int | None = None,
    offset: int = 0,
) -> list[Message]:
    """Fetch all messages for a session ordered by message_order ASC.

    Args:
        session_id: Session ID.
        limit: Maximum number of messages to return (None = all).
        offset: Number of messages to skip from the start.

    Returns:
        List of Message objects.
    """
    sql = """
        SELECT id, session_id, message_order, role, content, metadata, created_at
        FROM messages
        WHERE session_id = %s
        ORDER BY message_order ASC
    """
    params: list[Any] = [session_id]
    if limit is not None:
        sql += " LIMIT %s OFFSET %s"
        params.extend([limit, offset])

    with get_connection() as conn:
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute(sql, params)
            rows = cursor.fetchall()
        finally:
            cursor.close()

    messages: list[Message] = []
    for row in rows:
        metadata = row.get("metadata")
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        row["metadata"] = metadata
        messages.append(Message.from_dict(row))
    return messages


# =============================================================================
# Summary operations
# =============================================================================

def save_summary(record: SummaryRecord) -> None:
    """Append a new summary record for a session."""
    with get_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                INSERT INTO session_summaries
                    (session_id, summary_order, turns_start, turns_end, summary_text, created_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    record.session_id,
                    record.summary_order,
                    record.turns_start,
                    record.turns_end,
                    record.summary_text,
                    record.created_at,
                ),
            )
            conn.commit()
        finally:
            cursor.close()
    logger.debug("Summary saved: session=%s, order=%d", record.session_id, record.summary_order)


def get_latest_summary(session_id: str) -> SummaryRecord | None:
    """Fetch the most recent summary for a session."""
    with get_connection() as conn:
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute(
                """
                SELECT id, session_id, summary_order, turns_start, turns_end,
                       summary_text, created_at
                FROM session_summaries
                WHERE session_id = %s
                ORDER BY summary_order DESC
                LIMIT 1
                """,
                (session_id,),
            )
            row = cursor.fetchone()
        finally:
            cursor.close()
    if row is None:
        return None
    return SummaryRecord.from_dict(row)


def get_all_summaries(session_id: str) -> list[SummaryRecord]:
    """Fetch all summary records for a session in chronological order."""
    with get_connection() as conn:
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute(
                """
                SELECT id, session_id, summary_order, turns_start, turns_end,
                       summary_text, created_at
                FROM session_summaries
                WHERE session_id = %s
                ORDER BY summary_order ASC
                """,
                (session_id,),
            )
            rows = cursor.fetchall()
        finally:
            cursor.close()
    return [SummaryRecord.from_dict(row) for row in rows]


# =============================================================================
# Kafka retry queue
# =============================================================================

def write_to_retry_queue(
    event_id: str,
    event_type: str,
    session_id: str,
    payload: dict[str, Any],
    next_retry_at: datetime | None = None,
) -> None:
    """Insert a Kafka send failure event into the retry queue.

    Uses INSERT ... ON DUPLICATE KEY UPDATE so that a duplicate event_id
    (from a prior failed attempt) simply resets the status to PENDING.
    """
    if next_retry_at is None:
        next_retry_at = datetime.now()
    with get_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                INSERT INTO kafka_retry_queue
                    (event_id, event_type, session_id, payload, next_retry_at)
                VALUES (%s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    retry_count = retry_count,
                    next_retry_at = VALUES(next_retry_at),
                    status = 'PENDING',
                    last_error = NULL
                """,
                (
                    event_id,
                    event_type,
                    session_id,
                    json.dumps(payload),
                    next_retry_at,
                ),
            )
            conn.commit()
        finally:
            cursor.close()
    logger.debug("Event written to retry queue: event_id=%s, event_type=%s", event_id, event_type)


def get_pending_retry_events(limit: int = 100) -> list[dict[str, Any]]:
    """Fetch pending Kafka events that are due for retry."""
    with get_connection() as conn:
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute(
                """
                SELECT id, event_id, event_type, session_id, payload,
                       retry_count, max_retries, last_error
                FROM kafka_retry_queue
                WHERE status = 'PENDING'
                  AND next_retry_at <= NOW()
                  AND retry_count < max_retries
                ORDER BY next_retry_at ASC
                LIMIT %s
                """,
                (limit,),
            )
            rows = cursor.fetchall()
        finally:
            cursor.close()
    result = []
    for row in rows:
        row["payload"] = json.loads(row["payload"]) if isinstance(row["payload"], str) else row["payload"]
        result.append(row)
    return result


def mark_retry_sent(queue_id: int) -> None:
    """Mark a retry queue entry as successfully sent."""
    with get_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(
                "UPDATE kafka_retry_queue SET status = 'SENT', updated_at = NOW() WHERE id = %s",
                (queue_id,),
            )
            conn.commit()
        finally:
            cursor.close()


def mark_retry_failed(queue_id: int, error: str) -> None:
    """Mark a retry queue entry as permanently failed after exhausting retries."""
    with get_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                UPDATE kafka_retry_queue
                SET status = 'FAILED', last_error = %s, updated_at = NOW()
                WHERE id = %s
                """,
                (error, queue_id),
            )
            conn.commit()
        finally:
            cursor.close()


# =============================================================================
# Audit log
# =============================================================================

def audit_log(
    action: str,
    operator: str | None = None,
    session_id: str | None = None,
    resource: str | None = None,
    detail: dict[str, Any] | None = None,
    ip_address: str | None = None,
) -> None:
    """Append a record to the audit log table."""
    with get_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                INSERT INTO audit_log (operator, action, session_id, resource, detail, ip_address)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    operator,
                    action,
                    session_id,
                    resource,
                    json.dumps(detail) if detail else None,
                    ip_address,
                ),
            )
            conn.commit()
        finally:
            cursor.close()


# =============================================================================
# Utility
# =============================================================================

def get_message_count(session_id: str) -> int:
    """Return the total number of messages in a session."""
    with get_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id = %s",
                (session_id,),
            )
            row = cursor.fetchone()
            return int(row[0]) if row else 0
        finally:
            cursor.close()
