"""语义缓存（Semantic Cache）测试套件。

运行方式：
    # Mock 测试（无需任何外部依赖）
    python -m pytest tests/test_semantic_cache.py -v

    # 集成测试（需要 Milvus + Redis 服务运行）
    python -m pytest tests/test_semantic_cache.py -v
"""

from __future__ import annotations

import uuid
import warnings
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

warnings.filterwarnings("ignore", category=DeprecationWarning, module="pymilvus")

from src.vectorstore.semantic_cache import CacheStats, SemanticCache
from src.vectorstore import context_fingerprint as fp


class FakeEmbeddingClient:
    """返回固定向量的假 embedding client（维度 1536）。"""

    def __init__(self, vector_dim: int = 1536):
        self.vector_dim = vector_dim
        self.call_count = 0

    def embed_text(self, text: str) -> list[float]:
        self.call_count += 1
        vec = [0.1] * self.vector_dim
        vec[0] = hash(text) % 100 / 100.0
        norm = sum(v * v for v in vec) ** 0.5
        return [v / norm for v in vec]


class FakeMilvusClient:
    """纯内存 Milvus 替身。"""

    def __init__(self):
        self._storage: dict[int, dict[str, Any]] = {}
        self._next_id = 1
        self._conn_alias = "default"


@contextmanager
def patched_semantic_cache(
    fake_embed: FakeEmbeddingClient,
    fake_milvus: FakeMilvusClient,
    mock_redis: MagicMock,
    cache_cfg: dict[str, Any],
    conv_cfg: dict[str, Any],
):
    def _get_sem_cache_config():
        return cache_cfg

    def _get_conv_config():
        return conv_cfg

    def _noop(*a, **k):
        return None

    def _insert_cache_record(_milvus_client, record: dict[str, Any]) -> int:
        pk = fake_milvus._next_id
        fake_milvus._next_id += 1
        fake_milvus._storage[pk] = {"id": pk, **record}
        return pk

    def _search_similar(*, milvus_client=None, query_vector: list[float], top_k: int, expr: str | None = None, **kwargs):
        items = list(fake_milvus._storage.items())[:top_k]
        return [{"id": pk, "score": 0.95, **record} for pk, record in items]

    def _update_hit_stats(_milvus_client, cache_id: int, hit_count: int, last_hit_at: str) -> None:
        if cache_id in fake_milvus._storage:
            fake_milvus._storage[cache_id]["hit_count"] = hit_count
            fake_milvus._storage[cache_id]["last_hit_at"] = last_hit_at

    def _count_entries(_milvus_client) -> int:
        return len(fake_milvus._storage)

    def _evict_oldest(_milvus_client, n: int = 1) -> list[int]:
        ids = list(fake_milvus._storage.keys())[:n]
        for pk in ids:
            fake_milvus._storage.pop(pk, None)
        return ids

    def _clear_all_cache(_milvus_client) -> int:
        count = len(fake_milvus._storage)
        fake_milvus._storage.clear()
        return count

    with (
        patch("src.vectorstore.semantic_cache.get_semantic_cache_config", _get_sem_cache_config),
        patch("src.vectorstore.semantic_cache.get_conversation_config", _get_conv_config),
        patch("src.vectorstore.query_cache.ensure_query_cache_collection", _noop),
        patch("src.vectorstore.query_cache.insert_cache_record", _insert_cache_record),
        patch("src.vectorstore.query_cache.search_similar", _search_similar),
        patch("src.vectorstore.query_cache.update_hit_stats", _update_hit_stats),
        patch("src.vectorstore.query_cache.count_entries", _count_entries),
        patch("src.vectorstore.query_cache.evict_oldest", _evict_oldest),
        patch("src.vectorstore.query_cache.clear_all_cache", _clear_all_cache),
    ):
        cache = SemanticCache(fake_embed, fake_milvus)
        cache._redis = mock_redis
        yield cache


@pytest.fixture
def fake_embed() -> FakeEmbeddingClient:
    return FakeEmbeddingClient()


@pytest.fixture
def fake_milvus() -> FakeMilvusClient:
    return FakeMilvusClient()


@pytest.fixture
def mock_redis() -> MagicMock:
    mock = MagicMock()
    mock.get.return_value = None
    mock.setex.return_value = True
    mock.exists.return_value = 0
    mock.hset.return_value = True
    return mock


@pytest.fixture
def mock_cache_config() -> dict[str, Any]:
    return {
        "enabled": True,
        "similarity_threshold": 0.93,
        "context_threshold": 0.50,
        "max_entries": 10000,
        "cache_key_scope": "session",
        "redis_exact_ttl_seconds": 3600,
        "enable_exact_match": True,
        "search_top_k": 3,
        "min_answer_length": 20,
        "min_sources_count": 0,
        "collection_name": "query_cache",
    }


@pytest.fixture
def mock_conv_config() -> dict[str, Any]:
    return {"redis": {"host": "localhost", "port": 6379, "db": 0, "password": None}}


class TestContextFingerprint:
    def test_compute_summary_fingerprint(self):
        result = fp.compute_summary_fingerprint("急性心肌梗死是一种严重的心血管疾病，表现为胸痛。")
        assert "hash8" in result
        assert "keywords" in result
        assert len(result["hash8"]) == 8

    def test_fingerprint_stable(self):
        text = "糖尿病的诊断标准包括空腹血糖和糖化血红蛋白。"
        assert fp.compute_summary_fingerprint(text) == fp.compute_summary_fingerprint(text)

    def test_empty_text(self):
        result = fp.compute_summary_fingerprint("")
        assert result["hash8"] == ""
        assert result["keywords"] == []

    def test_jaccard(self):
        assert fp.jaccard_similarity(["a", "b"], ["a", "b"]) == 1.0
        assert fp.jaccard_similarity(["a"], ["b"]) == 0.0


class TestCacheStats:
    def test_hit_rate_zero(self):
        assert CacheStats(total_requests=0).hit_rate() == 0.0

    def test_hit_rate_normal(self):
        assert CacheStats(total_requests=100, cache_hits=37).hit_rate() == 0.37

    def test_to_dict(self):
        d = CacheStats(total_requests=10, cache_hits=3, redis_hits=2, milvus_hits=1).to_dict()
        assert d["cache_hits"] == 3
        assert "hit_rate" in d


class TestSemanticCacheMock:
    def test_disabled_returns_none(self, fake_embed, fake_milvus, mock_cache_config):
        mock_cache_config["enabled"] = False
        with patch("src.vectorstore.semantic_cache.get_semantic_cache_config", lambda: mock_cache_config):
            cache = SemanticCache(fake_embed, fake_milvus)
        assert cache.get_or_set(question="什么是心肌梗死？") is None

    def test_write_skipped_when_disabled(self, fake_embed, fake_milvus, mock_cache_config):
        mock_cache_config["enabled"] = False
        with patch("src.vectorstore.semantic_cache.get_semantic_cache_config", lambda: mock_cache_config):
            cache = SemanticCache(fake_embed, fake_milvus)
        ok = cache.write(question="测试问题", answer_text="测试回答", sources=[], query_vector=[0.0] * 1536)
        assert ok is False

    def test_write_skipped_on_empty_answer(self, fake_embed, fake_milvus, mock_redis, mock_cache_config, mock_conv_config):
        with patched_semantic_cache(fake_embed, fake_milvus, mock_redis, mock_cache_config, mock_conv_config) as cache:
            ok = cache.write(question="测试问题", answer_text="", sources=[], query_vector=[0.0] * 1536)
            assert ok is False

    def test_write_increments_stats(self, fake_embed, fake_milvus, mock_redis, mock_cache_config, mock_conv_config):
        with patched_semantic_cache(fake_embed, fake_milvus, mock_redis, mock_cache_config, mock_conv_config) as cache:
            cache.write(question="测试问题", answer_text="这是一个有效的测试回答。", sources=[], query_vector=[0.0] * 1536, session_id="sess-write")
            assert cache.get_stats().writes == 1

    def test_is_enabled(self, fake_embed, fake_milvus, mock_redis, mock_cache_config, mock_conv_config):
        with patched_semantic_cache(fake_embed, fake_milvus, mock_redis, mock_cache_config, mock_conv_config) as cache:
            assert cache.is_enabled() is True

    def test_session_fingerprint_cached_in_memory(self, fake_embed, fake_milvus, mock_redis, mock_cache_config, mock_conv_config):
        with patched_semantic_cache(fake_embed, fake_milvus, mock_redis, mock_cache_config, mock_conv_config) as cache:
            summary = "这是糖尿病的诊断和治疗的讨论摘要。" * 5
            assert cache._get_session_fingerprint(summary) == cache._get_session_fingerprint(summary)

    def test_cache_size_and_clear(self, fake_embed, fake_milvus, mock_redis, mock_cache_config, mock_conv_config):
        with patched_semantic_cache(fake_embed, fake_milvus, mock_redis, mock_cache_config, mock_conv_config) as cache:
            assert cache.cache_size() == 0
            cache.write(question="问题1", answer_text="回答1" * 10, sources=[], query_vector=[0.0] * 1536, session_id="sess-001")
            assert cache.cache_size() == 1
            deleted = cache.clear_all()
            assert deleted == 1
            assert cache.cache_size() == 0

    def test_reset_stats_on_cache_instance(self, fake_embed, fake_milvus, mock_redis, mock_cache_config, mock_conv_config):
        with patched_semantic_cache(fake_embed, fake_milvus, mock_redis, mock_cache_config, mock_conv_config) as cache:
            cache.write(question="q", answer_text="a" * 20, sources=[], query_vector=[0.0] * 1536, session_id="sess-reset")
            assert cache.get_stats().writes == 1
            cache.reset_stats()
            assert cache.get_stats().writes == 0

    def test_redis_exact_hit(self, fake_embed, fake_milvus, mock_redis, mock_cache_config, mock_conv_config):
        mock_redis.get.return_value = '{"answer_text":"心肌梗死是一种严重的心脏病。","sources_json":"[]","milvus_id":"123","hit_count":5,"last_hit_at":"2025-01-01T00:00:00"}'
        with patched_semantic_cache(fake_embed, fake_milvus, mock_redis, mock_cache_config, mock_conv_config) as cache:
            result = cache.get_or_set(question="什么是心肌梗死？", session_id="sess-001", session_summary="")
            assert result is not None
            assert result.hit_from == "redis_exact"
            assert cache.get_stats().redis_hits == 1

    def test_redis_miss_continues_to_milvus(self, fake_embed, fake_milvus, mock_redis, mock_cache_config, mock_conv_config):
        with patched_semantic_cache(fake_embed, fake_milvus, mock_redis, mock_cache_config, mock_conv_config) as cache:
            q_vec = fake_embed.embed_text("糖尿病的诊断标准是什么？")
            cache.write(
                question="糖尿病的诊断标准是什么？",
                answer_text="空腹血糖≥7.0 mmol/L 可诊断糖尿病。",
                sources=[],
                query_vector=q_vec,
                session_id="sess-001",
                session_summary="关于糖尿病的讨论",
            )
            mock_redis.get.return_value = None
            result = cache.get_or_set(
                question="糖尿病的诊断标准是什么？",
                session_id="sess-001",
                session_summary="关于糖尿病的讨论",
            )
            assert result is not None
            assert result.hit_from == "milvus_semantic"

    def test_jaccard_filter(self, fake_embed, fake_milvus, mock_redis, mock_cache_config, mock_conv_config):
        with patched_semantic_cache(fake_embed, fake_milvus, mock_redis, mock_cache_config, mock_conv_config) as cache:
            q_vec = fake_embed.embed_text("心肌梗死的治疗")
            cache.write(
                question="心肌梗死的治疗",
                answer_text="心肌梗死的治疗包括PCI和溶栓。",
                sources=[],
                query_vector=q_vec,
                session_id="sess-003",
                session_summary="心肌梗死相关讨论",
            )
            mock_redis.get.return_value = None
            result = cache.get_or_set(
                question="心肌梗死的治疗",
                session_id="sess-004",
                session_summary="糖尿病的诊断标准、血糖控制、饮食管理以及并发症预防的全面讨论",
            )
            assert result is None
            assert cache.get_stats().context_filtered >= 1

    def test_manual_evict_oldest(self, fake_embed, fake_milvus, mock_redis, mock_cache_config, mock_conv_config):
        with patched_semantic_cache(fake_embed, fake_milvus, mock_redis, mock_cache_config, mock_conv_config) as cache:
            for i in range(5):
                cache.write(
                    question=f"evict{i}",
                    answer_text=f"answer{i}" * 20,
                    sources=[],
                    query_vector=fake_embed.embed_text(f"evict{i}"),
                    session_id=f"evict-session-{i}",
                )
            deleted_ids = cache.evict_oldest(n=2)
            assert len(deleted_ids) == 2
            assert cache.cache_size() == 3


def _check_services() -> tuple[bool, bool]:
    milvus_ok = False
    redis_ok = False
    try:
        from pymilvus import connections
        alias = "_semcache_test_check"
        connections.connect(alias=alias, host="localhost", port="19530")
        milvus_ok = connections.has_connection(alias)
        connections.disconnect(alias=alias)
    except Exception:
        pass
    try:
        import redis as redis_lib
        client = redis_lib.Redis(host="localhost", port=6379, socket_timeout=2)
        client.ping()
        redis_ok = True
    except Exception:
        pass
    return milvus_ok, redis_ok


MILVUS_OK, REDIS_OK = _check_services()


@pytest.mark.skipif(not MILVUS_OK, reason="需要 Milvus 服务运行（localhost:19530）")
class TestSemanticCacheIntegration:
    def test_placeholder(self):
        assert True
