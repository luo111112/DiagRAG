"""Milvus query_cache Collection 管理。

负责 semantic_cache 的 Milvus 存储层：
  - Collection Schema 定义
  - 创建 / 删除 Collection
  - 插入、搜索、更新、删除缓存记录
  - 按 last_hit_at 淘汰最旧记录

Collection Schema：
  - id             INT64    auto_id, 主键
  - query_vector   FLOAT_VECTOR  dim=1536（语义搜索用）
  - query_text     VARCHAR  max_length=4000
  - answer_text    VARCHAR  max_length=8000
  - sources_json   VARCHAR  max_length=4000
  - session_id     VARCHAR  max_length=64
  - summary_hash   VARCHAR  max_length=16（SHA256 前8位）
  - summary_text   VARCHAR  max_length=2000（用于 Jaccard 比对）
  - hit_count      INT64
  - created_at     VARCHAR  max_length=32
  - last_hit_at    VARCHAR  max_length=32
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema
from pymilvus.exceptions import MilvusException

from src.config_loader import get_milvus_config, get_semantic_cache_config

if TYPE_CHECKING:
    from src.milvus_client import MilvusClient

logger = logging.getLogger(__name__)

_QUERY_CACHE_COLLECTION = "query_cache"


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------


def _build_cache_schema(vector_dim: int) -> CollectionSchema:
    """构建 query_cache Collection 的 Schema。"""
    fields = [
        FieldSchema(
            name="id",
            dtype=DataType.INT64,
            is_primary=True,
            auto_id=True,
            description="主键，自增 ID",
        ),
        FieldSchema(
            name="query_vector",
            dtype=DataType.FLOAT_VECTOR,
            dim=vector_dim,
            description="问题文本的稠密向量（语义搜索用）",
        ),
        FieldSchema(
            name="query_text",
            dtype=DataType.VARCHAR,
            max_length=4000,
            description="归一化后的原始问题文本",
        ),
        FieldSchema(
            name="answer_text",
            dtype=DataType.VARCHAR,
            max_length=8000,
            description="LLM 生成的完整回答",
        ),
        FieldSchema(
            name="sources_json",
            dtype=DataType.VARCHAR,
            max_length=4000,
            description="来源列表的 JSON 字符串",
        ),
        FieldSchema(
            name="session_id",
            dtype=DataType.VARCHAR,
            max_length=64,
            description="写入时的会话ID（单轮为 _single）",
        ),
        FieldSchema(
            name="summary_hash",
            dtype=DataType.VARCHAR,
            max_length=16,
            description="写入时会话摘要的 SHA256 前8位",
        ),
        FieldSchema(
            name="summary_text",
            dtype=DataType.VARCHAR,
            max_length=2000,
            description="写入时会话摘要原文（用于 Jaccard 比对）",
        ),
        FieldSchema(
            name="hit_count",
            dtype=DataType.INT64,
            description="累计命中次数",
        ),
        FieldSchema(
            name="created_at",
            dtype=DataType.VARCHAR,
            max_length=32,
            description="创建时间 ISO 字符串",
        ),
        FieldSchema(
            name="last_hit_at",
            dtype=DataType.VARCHAR,
            max_length=32,
            description="最近命中时间 ISO 字符串",
        ),
    ]
    return CollectionSchema(
        fields=fields,
        description="DiagRAG 语义缓存 query_cache Collection",
    )


def _index_params() -> dict[str, Any]:
    """构建 query_vector 字段的索引参数（HNSW，高召回率优先）。"""
    return {
        "index_type": "HNSW",
        "metric_type": "IP",
        "params": {"M": 16, "efConstruction": 200},
    }


# ---------------------------------------------------------------------------
# Collection lifecycle
# ---------------------------------------------------------------------------


def collection_exists(milvus_client: "MilvusClient") -> bool:
    """检查 query_cache Collection 是否存在。"""
    from pymilvus import utility

    cfg = get_semantic_cache_config()
    collection_name = cfg.get("collection_name", _QUERY_CACHE_COLLECTION)
    return utility.has_collection(collection_name, using=milvus_client._conn_alias)


def ensure_query_cache_collection(
    milvus_client: "MilvusClient",
    vector_dim: int | None = None,
) -> None:
    """确保 query_cache Collection 存在，不存在则创建并建立 HNSW 索引。

    Args:
        milvus_client: MilvusClient 实例。
        vector_dim: 向量维度，若为 None 则从配置读取。
    """
    cfg = get_semantic_cache_config()
    collection_name = cfg.get("collection_name", _QUERY_CACHE_COLLECTION)

    if vector_dim is None:
        vector_dim = int(get_milvus_config().get("vector_dim", 1536))

    from pymilvus import utility

    if utility.has_collection(collection_name, using=milvus_client._conn_alias):
        logger.info("query_cache Collection '%s' already exists.", collection_name)
        return

    schema = _build_cache_schema(vector_dim)
    collection = Collection(
        name=collection_name,
        schema=schema,
        using=milvus_client._conn_alias,
    )

    # 建立 HNSW 索引
    index_name = f"{collection_name}_vector_idx"
    collection.create_index(
        field_name="query_vector",
        index_params=_index_params(),
        index_name=index_name,
    )
    collection.flush()
    logger.info(
        "query_cache Collection '%s' created with HNSW index (dim=%d).",
        collection_name, vector_dim,
    )


def drop_query_cache_collection(milvus_client: "MilvusClient") -> None:
    """删除 query_cache Collection（用于测试或重置）。"""
    from pymilvus import utility

    cfg = get_semantic_cache_config()
    collection_name = cfg.get("collection_name", _QUERY_CACHE_COLLECTION)

    if not utility.has_collection(collection_name, using=milvus_client._conn_alias):
        logger.info("query_cache Collection '%s' does not exist, nothing to drop.", collection_name)
        return

    utility.drop_collection(collection_name, using=milvus_client._conn_alias)
    logger.info("query_cache Collection '%s' dropped.", collection_name)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_collection(milvus_client: "MilvusClient") -> Collection:
    """获取或加载 query_cache Collection 实例。"""
    cfg = get_semantic_cache_config()
    collection_name = cfg.get("collection_name", _QUERY_CACHE_COLLECTION)

    collection = Collection(collection_name, using=milvus_client._conn_alias)
    collection.load()
    return collection


def _collection_name() -> str:
    return get_semantic_cache_config().get("collection_name", _QUERY_CACHE_COLLECTION)


# ---------------------------------------------------------------------------
# CRUD operations
# ---------------------------------------------------------------------------


def insert_cache_record(
    milvus_client: "MilvusClient",
    record: dict[str, Any],
) -> int:
    """向 query_cache Collection 插入一条缓存记录。

    Args:
        milvus_client: MilvusClient 实例。
        record: 包含所有字段的字典（除 id 外），字段名须与 Schema 一致。

    Returns:
        Milvus entity id（int）。

    Raises:
        MilvusException: 插入失败时抛出。
    """
    collection = _get_collection(milvus_client)

    entities = [
        record["query_vector"],
        record["query_text"],
        record["answer_text"],
        record["sources_json"],
        record["session_id"],
        record["summary_hash"],
        record["summary_text"],
        record.get("hit_count", 0),
        record.get("created_at", datetime.now().isoformat()),
        record.get("last_hit_at", datetime.now().isoformat()),
    ]

    try:
        result = collection.insert([entities])
        collection.flush()
        pk = result.primary_keys[0]
        logger.debug("Cache record inserted: id=%s, query=%r", pk, record["query_text"][:40])
        return pk  # type: ignore[return-value]
    except MilvusException as e:
        raise MilvusException(
            code=e.code if hasattr(e, "code") else -1,
            message=f"插入 query_cache 记录失败: {e}",
        ) from e


def update_hit_stats(
    milvus_client: "MilvusClient",
    cache_id: int,
    hit_count: int,
    last_hit_at: str,
) -> None:
    """更新指定缓存记录的命中统计。

    Args:
        milvus_client: MilvusClient 实例。
        cache_id: 缓存记录的主键 ID。
        hit_count: 更新后的累计命中次数。
        last_hit_at: 更新后的最近命中时间（ISO 字符串）。
    """
    collection = _get_collection(milvus_client)
    try:
        collection.update(
            filter_expression=f'id == {cache_id}',
            set_fields={
                "hit_count": hit_count,
                "last_hit_at": last_hit_at,
            },
        )
        collection.flush()
        logger.debug("Cache hit stats updated: id=%s, hit_count=%d", cache_id, hit_count)
    except MilvusException as e:
        logger.warning("更新 query_cache hit_stats 失败: id=%s, error=%s", cache_id, e)


def delete_cache_entry(
    milvus_client: "MilvusClient",
    cache_id: int,
) -> None:
    """按主键 ID 删除一条缓存记录。"""
    collection = _get_collection(milvus_client)
    try:
        collection.delete(f"id == {cache_id}")
        collection.flush()
        logger.debug("Cache entry deleted: id=%s", cache_id)
    except MilvusException as e:
        logger.warning("删除 query_cache 记录失败: id=%s, error=%s", cache_id, e)


def search_similar(
    milvus_client: "MilvusClient",
    query_vector: list[float],
    top_k: int,
    expr: str | None = None,
) -> list[dict[str, Any]]:
    """在 query_cache Collection 中进行 ANN 向量搜索。

    Args:
        milvus_client: MilvusClient 实例。
        query_vector: 查询向量。
        top_k: 返回的最近邻数量。
        expr: 可选，Milvus 标量过滤表达式（如 session_id 过滤）。

    Returns:
        结果列表，每个元素包含 id、query_vector、query_text、answer_text、
        sources_json、session_id、summary_hash、summary_text、hit_count、
        created_at、last_hit_at、score。
    """
    collection = _get_collection(milvus_client)

    search_params = {
        "metric_type": "IP",
        "params": {"ef": 128},
    }

    try:
        results = collection.search(
            data=[query_vector],
            anns_field="query_vector",
            param=search_params,
            limit=top_k,
            output_fields=[
                "id", "query_vector", "query_text", "answer_text",
                "sources_json", "session_id", "summary_hash", "summary_text",
                "hit_count", "created_at", "last_hit_at",
            ],
            expr=expr,
        )

        hits = results[0] if results else []
        output: list[dict[str, Any]] = []

        for hit in hits:
            entity = hit.entity
            output.append({
                "id": hit.id,
                "score": hit.score,
                "query_vector": entity.get("query_vector"),
                "query_text": entity.get("query_text", ""),
                "answer_text": entity.get("answer_text", ""),
                "sources_json": entity.get("sources_json", ""),
                "session_id": entity.get("session_id", ""),
                "summary_hash": entity.get("summary_hash", ""),
                "summary_text": entity.get("summary_text", ""),
                "hit_count": entity.get("hit_count", 0),
                "created_at": entity.get("created_at", ""),
                "last_hit_at": entity.get("last_hit_at", ""),
            })

        return output

    except MilvusException as e:
        logger.error("query_cache 语义搜索失败: %s", e)
        return []


def count_entries(milvus_client: "MilvusClient") -> int:
    """返回 query_cache Collection 中的记录总数。"""
    try:
        collection = _get_collection(milvus_client)
        return collection.num_entities
    except MilvusException:
        return 0


def evict_oldest(milvus_client: "MilvusClient", n: int = 1) -> list[int]:
    """按 last_hit_at 淘汰最旧的 n 条缓存记录。

    Args:
        milvus_client: MilvusClient 实例。
        n: 淘汰数量。

    Returns:
        被删除记录的 ID 列表。
    """
    collection = _get_collection(milvus_client)
    deleted_ids: list[int] = []

    try:
        results = collection.query(
            expr="",
            output_fields=["id", "last_hit_at"],
            limit=n,
            sort=["last_hit_at", "ASC"],
        )
        if not results:
            return []

        for record in results:
            pk = record["id"]
            collection.delete(f"id == {pk}")
            deleted_ids.append(pk)

        if deleted_ids:
            collection.flush()
            logger.info("Evicted %d oldest cache entries: ids=%s", len(deleted_ids), deleted_ids)

        return deleted_ids

    except MilvusException as e:
        logger.warning("淘汰 query_cache 最旧记录失败: %s", e)
        return []


def clear_all_cache(milvus_client: "MilvusClient") -> int:
    """清空 query_cache Collection 中的所有记录。

    Returns:
        被删除的记录数量。
    """
    try:
        collection = _get_collection(milvus_client)
        count = collection.num_entities
        if count > 0:
            collection.delete("id >= 0")
            collection.flush()
            logger.info("Cleared all %d entries from query_cache.", count)
        return count
    except MilvusException as e:
        logger.warning("清空 query_cache 失败: %s", e)
        return 0
