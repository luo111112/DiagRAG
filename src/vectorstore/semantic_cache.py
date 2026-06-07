"""语义缓存（Semantic Cache）主逻辑。

采用 Redis（精确键）+ Milvus（ANN 向量）二级架构：

  一级  Redis 精确键匹配
    └── 键格式：semcache:{scope}:{question_sha256[:16]}:{session_id}:{summary_hash}
    └── 值：JSON { answer_text, sources_json, milvus_id, hit_count, last_hit_at }
    └── TTL：redis_exact_ttl_seconds（默认 3600s）

  二级  Milvus ANN 语义搜索
    └── Collection：query_cache（见 query_cache.py）
    └── 命中条件：余弦相似度 ≥ similarity_threshold
    └── 命中后：还需通过上下文指纹 Jaccard 比对（context_threshold）

写入时机（主动缓存）：
  - RAGChain.answer() 返回 answer 后，且 answer 非空、sources 非零，
    主动将 (query_text, query_vector, answer, sources) 写入缓存。

查询时机（被动命中）：
  - RAGChain.answer() 调用前，先查语义缓存：
      1. Redis 精确键 → 命中则直接返回（一级命中）
      2. Milvus ANN 搜索 → 命中则进一步校验上下文指纹 → 返回（二级命中）
  - 查询成功后立即更新 hit_count 和 last_hit_at。

淘汰策略（被动 + 主动）：
  - 被动：Milvus 查询时检查 max_entries 超量 → 按 last_hit_at 淘汰最旧记录
  - 主动：定时任务（每日凌晨）或手动触发可调用 evict_oldest_cache(n=100)

对外暴露：
  - SemanticCache 类：RAGChain 侧直接持有的缓存实例
  - CacheStats：缓存统计信息
  - is_cache_hit()：判断某次 answer() 是否走了缓存（通过 answer_id）
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

import redis

from src.config_loader import (
    get_conversation_config,
    get_semantic_cache_config,
)
from src.conversation.models import CacheHit
from src.vectorstore import context_fingerprint as fp
from src.vectorstore import query_cache as qc

if TYPE_CHECKING:
    from src.embedding_client import DashScopeEmbeddingClient
    from src.milvus_client import MilvusClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class CacheStats:
    """语义缓存运行时统计。"""

    total_requests: int = 0      # 累计请求数（answer 调用次数）
    cache_hits: int = 0          # 累计命中次数（Redis + Milvus）
    redis_hits: int = 0          # Redis 一级命中次数
    milvus_hits: int = 0         # Milvus 二级命中次数
    context_filtered: int = 0    # 因上下文指纹不匹配被过滤的次数
    writes: int = 0              # 累计写入次数
    evictions: int = 0           # 累计淘汰记录数
    last_eviction_at: str = ""   # 最近一次淘汰时间（ISO 字符串）
    last_reset_at: str = ""      # 上次重置时间（ISO 字符串）

    def hit_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return round(self.cache_hits / self.total_requests, 4)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "redis_hits": self.redis_hits,
            "milvus_hits": self.milvus_hits,
            "context_filtered": self.context_filtered,
            "writes": self.writes,
            "evictions": self.evictions,
            "last_eviction_at": self.last_eviction_at,
            "last_reset_at": self.last_reset_at,
            "hit_rate": self.hit_rate(),
        }


# ---------------------------------------------------------------------------
# Redis 精确缓存键管理
# ---------------------------------------------------------------------------

_REDIS_VALUE_KEYS = frozenset([
    "answer_text",
    "sources_json",
    "milvus_id",
    "hit_count",
    "last_hit_at",
])


def _redis_key_for_question(
    question: str,
    session_id: str | None,
    summary_hash: str,
    scope: str,
) -> str:
    """构建 Redis 精确缓存键。"""
    return fp.build_cache_key(
        question=question,
        session_id=session_id,
        summary_hash=summary_hash,
        scope=scope,
    )


# ---------------------------------------------------------------------------
# SemanticCache 主类
# ---------------------------------------------------------------------------


class SemanticCache:
    """语义缓存：Redis（一级精确）+ Milvus（二级语义 ANN）二级架构。"""

    def __init__(
        self,
        embedding_client: "DashScopeEmbeddingClient",
        milvus_client: "MilvusClient",
    ) -> None:
        """初始化语义缓存。

        Args:
            embedding_client: 用于将查询文本编码为向量的客户端。
            milvus_client: MilvusClient 实例，持有 query_cache Collection 连接。
        """
        self._embed = embedding_client
        self._milvus = milvus_client
        self._cfg = get_semantic_cache_config()
        self._enabled = self._cfg.get("enabled", False)
        self._ttl_seconds = int(self._cfg.get("redis_exact_ttl_seconds", 3600))

        # Redis 客户端（延迟初始化）
        self._redis: redis.Redis | None = None

        # 运行时统计
        self._stats = CacheStats()

        # 会话摘要哈希缓存（会话级别，避免重复调用 fingerprint）
        # key: session_id, value: {"hash8": str, "keywords": list[str]}
        self._session_fingerprint_cache: dict[str, dict[str, Any]] = {}

        if self._enabled:
            qc.ensure_query_cache_collection(milvus_client)
            logger.info(
                "SemanticCache enabled: threshold=%.2f, context_threshold=%.2f, "
                "scope=%s, redis_ttl=%ds, max_entries=%d",
                self._cfg.get("similarity_threshold", 0.93),
                self._cfg.get("context_threshold", 0.50),
                self._cfg.get("cache_key_scope", "session"),
                self._ttl_seconds,
                self._cfg.get("max_entries", 10000),
            )
        else:
            logger.info("SemanticCache disabled (enabled=false).")

    # ------------------------------------------------------------------
    # Redis 客户端（延迟初始化）
    # ------------------------------------------------------------------

    @property
    def _redis_client(self) -> redis.Redis:
        if self._redis is None:
            conv_cfg = get_conversation_config()
            redis_cfg = conv_cfg.get("redis", {})
            password = redis_cfg.get("password") or None
            self._redis = redis.Redis(
                host=redis_cfg.get("host", "localhost"),
                port=int(redis_cfg.get("port", 6379)),
                db=int(redis_cfg.get("db", 0)),
                password=password,
                decode_responses=True,
                socket_timeout=5.0,
                socket_connect_timeout=5.0,
                retry_on_timeout=True,
            )
        return self._redis

    # ------------------------------------------------------------------
    # 会话指纹（带内存缓存，避免重复计算）
    # ------------------------------------------------------------------

    def _get_session_fingerprint(self, session_summary: str) -> dict[str, Any]:
        """获取会话摘要的指纹（带内存缓存）。"""
        if not session_summary:
            return {"hash8": "", "keywords": []}

        # 简单用文本前 128 字符作缓存键（摘要长度固定，够用）
        cache_key = hashlib.md5(session_summary.encode("utf-8")).hexdigest()[:16]

        if cache_key not in self._session_fingerprint_cache:
            fingerprint = fp.compute_summary_fingerprint(session_summary)
            self._session_fingerprint_cache[cache_key] = fingerprint

        return self._session_fingerprint_cache[cache_key]

    # ------------------------------------------------------------------
    # 公开 API：查询（对外调用的主入口）
    # ------------------------------------------------------------------

    def get_or_set(
        self,
        question: str,
        session_id: str | None = None,
        session_summary: str = "",
        user_id: str | None = None,
        llm_call_fn: callable | None = None,
    ) -> CacheHit | None:
        """查询缓存，命中则返回 CacheHit；未命中则返回 None。

        此方法不执行 LLM 调用，调用方应在 None 时自行调用 LLM，
        然后再用 ``write()`` 写入缓存。

        查询流程：
          1. 归一化问题文本 → 精确键查询 Redis（一级命中直接返回）
          2. 生成问题向量 → ANN 搜索 Milvus query_cache（二级）
          3. 二级命中后 → 上下文指纹 Jaccard 比对 → 通过则返回

        Args:
            question: 原始用户问题。
            session_id: 当前会话 ID（session scope 需提供）。
            session_summary: 当前会话摘要（用于上下文指纹比对）。
            user_id: 用户 ID（user scope 需提供）。
            llm_call_fn: 未使用，保留向后兼容。

        Returns:
            CacheHit：缓存命中。
            None：未命中（需调用 LLM）。
        """
        if not self._enabled:
            return None

        self._stats.total_requests += 1

        # ---------- 一级：Redis 精确键 ----------
        hit = self._try_redis_exact(question, session_id, session_summary)
        if hit is not None:
            self._stats.cache_hits += 1
            self._stats.redis_hits += 1
            logger.debug("SemanticCache REDIS hit: %s", question[:40])
            return hit

        # ---------- 二级：Milvus ANN ----------
        hit = self._try_milvus_semantic(question, session_id, session_summary)
        if hit is not None:
            self._stats.cache_hits += 1
            self._stats.milvus_hits += 1
            logger.info(
                "SemanticCache MILVUS hit (score=%.4f): %s",
                getattr(hit, "score", 0.0),
                question[:40],
            )
            return hit

        return None

    # ------------------------------------------------------------------
    # 一级：Redis 精确键查询
    # ------------------------------------------------------------------

    def _try_redis_exact(
        self,
        question: str,
        session_id: str | None,
        session_summary: str,
    ) -> CacheHit | None:
        """Redis 精确键匹配（一级缓存）。"""
        if not self._cfg.get("enable_exact_match", True):
            return None

        try:
            scope = self._cfg.get("cache_key_scope", "session")
            summary_fp = self._get_session_fingerprint(session_summary)
            cache_key = _redis_key_for_question(
                question=question,
                session_id=session_id,
                summary_hash=summary_fp["hash8"],
                scope=scope,
            )

            raw = self._redis_client.get(cache_key)
            if not raw:
                return None

            data = json.loads(raw)

            # 更新命中统计
            new_hit_count = int(data.get("hit_count", 0)) + 1
            last_hit_at = datetime.now().isoformat()
            self._redis_client.hset(
                cache_key,
                mapping={"hit_count": new_hit_count, "last_hit_at": last_hit_at},
            )

            # 同步更新 Milvus 中的 hit_count（异步失败不影响主流程）
            milvus_id = data.get("milvus_id")
            if milvus_id is not None:
                try:
                    qc.update_hit_stats(self._milvus, milvus_id, new_hit_count, last_hit_at)
                except Exception as e:
                    logger.warning("Failed to sync hit_count to Milvus: %s", e)

            sources: list[dict] = []
            if data.get("sources_json"):
                try:
                    sources = json.loads(data["sources_json"])
                except json.JSONDecodeError:
                    pass

            return CacheHit(
                answer_text=data.get("answer_text", ""),
                sources=sources,
                cache_entry_id=int(milvus_id) if milvus_id is not None else -1,
                hit_from="redis_exact",
            )

        except Exception as e:
            logger.warning("Redis exact cache lookup failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # 二级：Milvus ANN 语义搜索
    # ------------------------------------------------------------------

    def _try_milvus_semantic(
        self,
        question: str,
        session_id: str | None,
        session_summary: str,
    ) -> CacheHit | None:
        """Milvus ANN 语义搜索 + 上下文指纹校验（二级缓存）。"""
        try:
            # 1. 生成查询向量
            query_vector = self._embed.embed_text(question)

            # 2. ANN 搜索
            search_top_k = int(self._cfg.get("search_top_k", 3))
            results = qc.search_similar(
                milvus_client=self._milvus,
                query_vector=query_vector,
                top_k=search_top_k,
            )

            if not results:
                return None

            # 3. 逐一检查命中结果
            similarity_threshold = float(self._cfg.get("similarity_threshold", 0.93))
            context_threshold = float(self._cfg.get("context_threshold", 0.50))
            min_answer_len = int(self._cfg.get("min_answer_length", 20))
            min_sources = int(self._cfg.get("min_sources_count", 0))

            for hit in results:
                score: float = hit.get("score", 0.0)
                if score < similarity_threshold:
                    continue

                answer_text = hit.get("answer_text", "")
                if len(answer_text) < min_answer_len:
                    self._stats.context_filtered += 1
                    continue

                sources: list[dict] = []
                if hit.get("sources_json"):
                    try:
                        sources = json.loads(hit["sources_json"])
                    except json.JSONDecodeError:
                        pass

                if len(sources) < min_sources:
                    self._stats.context_filtered += 1
                    continue

                # 4. 上下文指纹 Jaccard 比对（多轮场景）
                if session_summary and hit.get("summary_text"):
                    current_fp = fp.compute_summary_fingerprint(session_summary)
                    cached_fp = fp.compute_summary_fingerprint(hit["summary_text"])
                    similarity = fp.jaccard_similarity(
                        current_fp["keywords"],
                        cached_fp["keywords"],
                    )
                    if similarity < context_threshold:
                        self._stats.context_filtered += 1
                        logger.debug(
                            "Context Jaccard too low (%.4f < %.2f), skipping cache entry %s",
                            similarity,
                            context_threshold,
                            hit["id"],
                        )
                        continue

                # 5. 更新 Milvus hit_count
                new_hit_count = int(hit.get("hit_count", 0)) + 1
                last_hit_at = datetime.now().isoformat()
                try:
                    qc.update_hit_stats(self._milvus, hit["id"], new_hit_count, last_hit_at)
                except Exception as e:
                    logger.warning("Failed to update Milvus hit_count: %s", e)

                # 6. 同步更新 Redis 键（若存在）
                self._sync_redis_from_milvus(hit, new_hit_count, last_hit_at)

                return CacheHit(
                    answer_text=answer_text,
                    sources=sources,
                    cache_entry_id=hit["id"],
                    hit_from="milvus_semantic",
                )

            return None

        except Exception as e:
            logger.warning("Milvus semantic cache lookup failed: %s", e)
            return None

    def _sync_redis_from_milvus(
        self,
        milvus_hit: dict[str, Any],
        hit_count: int,
        last_hit_at: str,
    ) -> None:
        """将 Milvus 命中结果同步写入 Redis 精确键（若尚未存在）。"""
        try:
            scope = self._cfg.get("cache_key_scope", "session")
            summary_fp = self._get_session_fingerprint(milvus_hit.get("summary_text", ""))

            if scope == "session":
                cache_key = _redis_key_for_question(
                    question=milvus_hit.get("query_text", ""),
                    session_id=milvus_hit.get("session_id"),
                    summary_hash=summary_fp["hash8"],
                    scope=scope,
                )
            elif scope == "user":
                # 从 session_id 提取 user_id（约定 user_id 是 session_id 前缀）
                session_id_str = milvus_hit.get("session_id", "")
                user_id = session_id_str.split("-")[0] if session_id_str else ""
                cache_key = _redis_key_for_question(
                    question=milvus_hit.get("query_text", ""),
                    session_id=None,
                    summary_hash="",
                    scope=scope,
                )
            else:
                cache_key = _redis_key_for_question(
                    question=milvus_hit.get("query_text", ""),
                    session_id=None,
                    summary_hash="",
                    scope=scope,
                )

            # 仅当键不存在时才写入（EX 保证后续不覆盖）
            exists = self._redis_client.exists(cache_key)
            if exists:
                return

            redis_value = json.dumps({
                "answer_text": milvus_hit.get("answer_text", ""),
                "sources_json": milvus_hit.get("sources_json", ""),
                "milvus_id": milvus_hit["id"],
                "hit_count": hit_count,
                "last_hit_at": last_hit_at,
            })
            self._redis_client.setex(cache_key, self._ttl_seconds, redis_value)

        except Exception as e:
            logger.warning("Failed to sync Redis from Milvus hit: %s", e)

    # ------------------------------------------------------------------
    # 公开 API：写入缓存
    # ------------------------------------------------------------------

    def write(
        self,
        question: str,
        answer_text: str,
        sources: list[dict],
        query_vector: list[float],
        session_id: str | None = None,
        session_summary: str = "",
        user_id: str | None = None,
    ) -> bool:
        """将 LLM 生成结果写入语义缓存（Redis + Milvus）。

        写入条件（全部满足才写入）：
          - enabled == True
          - answer_text 非空
          - sources 数量 ≥ min_sources_count（默认 0）

        Args:
            question: 归一化后的原始问题文本。
            answer_text: LLM 生成的回答文本。
            sources: 引用来源列表。
            query_vector: 问题的稠密向量。
            session_id: 当前会话 ID。
            session_summary: 当前会话摘要（用于后续多轮命中时的上下文指纹比对）。
            user_id: 用户 ID（user scope 时使用）。

        Returns:
            True：写入成功（含部分成功）。
            False：未写入（被禁用或不满足写入条件）。
        """
        if not self._enabled:
            return False

        if not answer_text:
            logger.debug("SemanticCache write skipped: answer_text is empty.")
            return False

        min_sources = int(self._cfg.get("min_sources_count", 0))
        if len(sources) < min_sources:
            logger.debug(
                "SemanticCache write skipped: sources count %d < min_sources_count %d.",
                len(sources), min_sources,
            )
            return False

        try:
            scope = self._cfg.get("cache_key_scope", "session")
            summary_fp = self._get_session_fingerprint(session_summary)
            summary_text = session_summary[:2000]  # 截断以防超 Milvus VARCHAR 限制
            summary_hash = summary_fp["hash8"]

            # ---------- Milvus 写入 ----------
            sources_json = json.dumps(sources, ensure_ascii=False)
            now_iso = datetime.now().isoformat()

            milvus_record = {
                "query_vector": query_vector,
                "query_text": question[:4000],
                "answer_text": answer_text[:8000],
                "sources_json": sources_json[:4000],
                "session_id": (session_id or "_single")[:64],
                "summary_hash": summary_hash[:16],
                "summary_text": summary_text[:2000],
                "hit_count": 0,
                "created_at": now_iso,
                "last_hit_at": now_iso,
            }

            milvus_id = qc.insert_cache_record(self._milvus, milvus_record)

            # ---------- Redis 写入（精确键） ----------
            cache_key = _redis_key_for_question(
                question=question,
                session_id=session_id,
                summary_hash=summary_hash,
                scope=scope,
            )
            redis_value = json.dumps({
                "answer_text": answer_text[:8000],
                "sources_json": sources_json[:4000],
                "milvus_id": milvus_id,
                "hit_count": 0,
                "last_hit_at": now_iso,
            })
            self._redis_client.setex(cache_key, self._ttl_seconds, redis_value)

            # ---------- 容量检查：超量则淘汰最旧记录 ----------
            self._check_and_evict()

            self._stats.writes += 1
            logger.info(
                "SemanticCache write success: milvus_id=%s, question=%r, sources=%d",
                milvus_id, question[:40], len(sources),
            )
            return True

        except Exception as e:
            logger.error("SemanticCache write failed: %s", e)
            return False

    # ------------------------------------------------------------------
    # 容量管理
    # ------------------------------------------------------------------

    def _check_and_evict(self) -> None:
        """检查缓存容量，超出 max_entries 则淘汰最旧记录。"""
        try:
            max_entries = int(self._cfg.get("max_entries", 10000))
            current_count = qc.count_entries(self._milvus)

            if current_count < max_entries:
                return

            excess = current_count - max_entries + 1
            deleted_ids = qc.evict_oldest(self._milvus, n=excess)
            if deleted_ids:
                self._stats.evictions += len(deleted_ids)
                self._stats.last_eviction_at = datetime.now().isoformat()
                logger.info(
                    "SemanticCache evicted %d oldest entries (current=%d, max=%d).",
                    len(deleted_ids), current_count, max_entries,
                )

        except Exception as e:
            logger.warning("SemanticCache eviction check failed: %s", e)

    def evict_oldest(self, n: int = 100) -> list[int]:
        """主动淘汰最旧的 n 条缓存记录。

        Args:
            n: 淘汰数量，默认 100。

        Returns:
            被删除记录的 Milvus ID 列表。
        """
        deleted_ids = qc.evict_oldest(self._milvus, n=n)
        if deleted_ids:
            self._stats.evictions += len(deleted_ids)
            self._stats.last_eviction_at = datetime.now().isoformat()
        return deleted_ids

    # ------------------------------------------------------------------
    # 统计 & 管理
    # ------------------------------------------------------------------

    def get_stats(self) -> CacheStats:
        """返回当前缓存统计信息（拷贝）。"""
        d = self._stats.to_dict()
        d.pop("hit_rate", None)
        return CacheStats(**d)

    def reset_stats(self) -> None:
        """重置运行时统计计数器。"""
        self._stats = CacheStats(last_reset_at=datetime.now().isoformat())

    def is_enabled(self) -> bool:
        """返回 enabled 状态。"""
        return self._enabled

    def cache_size(self) -> int:
        """返回 Milvus query_cache 中的记录总数。"""
        try:
            return qc.count_entries(self._milvus)
        except Exception:
            return -1

    def clear_all(self) -> int:
        """清空所有缓存（Milvus + Redis）。

        Returns:
            Milvus 中被删除的记录数量。
        """
        deleted_count = qc.clear_all_cache(self._milvus)
        # Redis 精确键无法高效枚举删除，保持 Redis 靠 TTL 自然过期
        logger.info("SemanticCache cleared: %d Milvus entries removed.", deleted_count)
        return deleted_count
