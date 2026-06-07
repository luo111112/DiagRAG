"""语义缓存真实服务集成测试。

运行方式：
    # 仅运行本文件
    python -m pytest tests/test_semantic_cache_integration.py -v

前提：
    1. FastAPI 服务已启动（默认 http://127.0.0.1:8000）
    2. Redis / Milvus / MySQL 可用
    3. DASHSCOPE_API_KEY 有效
    4. 知识库已导入可检索数据

说明：
    - 本测试会优先走真实 HTTP API 验证缓存行为
    - 若本地服务不可用，会自动 skip
    - 若依赖不可用，也会自动 skip
"""

from __future__ import annotations

import time
import uuid
from typing import Any

import pytest
import requests

BASE_URL = "http://127.0.0.1:8000"
TIMEOUT = 120
TEST_QUESTION = "急性心肌梗死的典型症状有哪些？"


# ---------------------------------------------------------------------------
# 依赖检查
# ---------------------------------------------------------------------------

def _service_available() -> bool:
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


SERVICE_OK = _service_available()


@pytest.fixture
def unique_user_id() -> str:
    return f"cache-it-user-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def unique_session_title() -> str:
    return f"缓存测试会话-{uuid.uuid4().hex[:6]}"


def _post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    resp = requests.post(f"{BASE_URL}{path}", json=payload, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def _get(path: str) -> dict[str, Any]:
    resp = requests.get(f"{BASE_URL}{path}", timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


@pytest.mark.skipif(not SERVICE_OK, reason="需要本地 FastAPI 服务运行在 http://127.0.0.1:8000")
class TestSemanticCacheAPIIntegration:
    """通过真实 HTTP API 验证缓存行为。"""

    def test_chat_endpoint_cache_cycle(self, unique_user_id: str):
        """/chat 第一次写缓存，第二次相同问题命中缓存。"""
        payload = {
            "question": TEST_QUESTION,
            "user_id": unique_user_id,
        }

        first = _post("/chat", payload)
        second = _post("/chat", payload)

        assert "answer" in first
        assert "session_id" in first
        assert first["answer"] != ""
        assert first["cache_hit"] is False

        assert second["answer"] != ""
        assert second["cache_hit"] is True
        assert second["cache_hit_from"] in {"redis_exact", "milvus_semantic"}

    def test_conversation_endpoint_cache_cycle(self, unique_user_id: str, unique_session_title: str):
        """/conversation/{id}/message 两次相同问题应在第二次命中缓存。"""
        create_payload = {
            "user_id": unique_user_id,
            "title": unique_session_title,
        }
        session = _post("/conversation/create", create_payload)
        session_id = session["session_id"]

        message_payload = {"content": TEST_QUESTION}

        first = _post(f"/conversation/{session_id}/message", message_payload)
        second = _post(f"/conversation/{session_id}/message", message_payload)

        assert first["session_id"] == session_id
        assert first["answer"] != ""
        assert first["cache_hit"] is False

        assert second["session_id"] == session_id
        assert second["answer"] != ""
        assert second["cache_hit"] is True
        assert second["cache_hit_from"] in {"redis_exact", "milvus_semantic"}

    def test_cache_stats_endpoint(self, unique_user_id: str):
        """调用缓存后，/cache/stats 应返回可解析的统计信息。"""
        _post("/chat", {"question": TEST_QUESTION, "user_id": unique_user_id})
        _post("/chat", {"question": TEST_QUESTION, "user_id": unique_user_id})

        stats = _get("/cache/stats")

        assert "total_requests" in stats
        assert "cache_hits" in stats
        assert "redis_hits" in stats
        assert "milvus_hits" in stats
        assert "writes" in stats
        assert "hit_rate" in stats
        assert "enabled" in stats
        assert isinstance(stats["enabled"], bool)

    def test_cache_reset_stats_endpoint(self, unique_user_id: str):
        """/cache/reset-stats 应返回归零后的统计。"""
        _post("/chat", {"question": TEST_QUESTION, "user_id": unique_user_id})
        reset = _post("/cache/reset-stats", {})

        assert reset["total_requests"] == 0
        assert reset["cache_hits"] == 0
        assert reset["redis_hits"] == 0
        assert reset["milvus_hits"] == 0
        assert reset["writes"] == 0

    def test_cache_clear_endpoint(self, unique_user_id: str):
        """/cache/clear 应能清空 Milvus 语义缓存。"""
        _post("/chat", {"question": TEST_QUESTION, "user_id": unique_user_id})
        cleared = _post("/cache/clear", {})

        assert "deleted_count" in cleared
        assert isinstance(cleared["deleted_count"], int)

    def test_cache_evict_endpoint(self, unique_user_id: str):
        """/cache/evict 应返回驱逐结果结构。"""
        for i in range(3):
            _post("/chat", {"question": f"{TEST_QUESTION}-{i}", "user_id": unique_user_id})
            time.sleep(0.1)

        evicted = _post("/cache/evict", {"n": 1})

        assert "evicted_ids" in evicted
        assert "evicted_count" in evicted
        assert evicted["evicted_count"] >= 0

    def test_conversation_history_endpoint(self, unique_user_id: str, unique_session_title: str):
        """发送消息后，/conversation/{id}/history 应返回消息历史。"""
        session = _post("/conversation/create", {"user_id": unique_user_id, "title": unique_session_title})
        session_id = session["session_id"]

        _post(f"/conversation/{session_id}/message", {"content": TEST_QUESTION})
        history = _get(f"/conversation/{session_id}/history")

        assert history["session_id"] == session_id
        assert "messages" in history
        assert isinstance(history["messages"], list)
        assert len(history["messages"]) >= 2  # user + assistant

    def test_chat_response_time_improves_on_cache_hit(self, unique_user_id: str):
        """第二次相同问题命中缓存时，通常应不慢于首次调用。"""
        payload = {"question": TEST_QUESTION, "user_id": unique_user_id}

        t1 = time.perf_counter()
        first = _post("/chat", payload)
        elapsed1 = time.perf_counter() - t1

        t2 = time.perf_counter()
        second = _post("/chat", payload)
        elapsed2 = time.perf_counter() - t2

        assert first["cache_hit"] is False
        assert second["cache_hit"] is True
        # 允许偶发波动，但命中缓存一般不会更慢太多
        assert elapsed2 <= elapsed1 * 1.5
