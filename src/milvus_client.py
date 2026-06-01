"""Milvus vector database client for DiagRAG.

支持稠密向量（语义）+ 稀疏向量（BM25 关键词）混合检索。
"""

from __future__ import annotations

import logging
from typing import Any

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)
from pymilvus.exceptions import MilvusException

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# BM25 / Sparse Vector Helpers
# ----------------------------------------------------------------------


def _tokenize_chinese(text: str) -> list[str]:
    """中文分词，基于 jieba 精确分词，小写化。

    英文词统一转为小写；中文词保留原样（BM25 对英文大小写不敏感）。
    移除空字符串。
    """
    import jieba

    tokens = jieba.lcut(text)
    return [t.lower() for t in tokens if t.strip()]


def _compute_idf(corpus: list[str]) -> dict[str, float]:
    """根据语料库计算每个词项的 IDF 值。"""
    import math

    N = len(corpus)
    df: dict[str, int] = {}
    for doc in corpus:
        tokens = set(_tokenize_chinese(doc))
        for token in tokens:
            df[token] = df.get(token, 0) + 1

    return {term: math.log((N - doc_freq + 0.5) / (doc_freq + 0.5) + 1) for term, doc_freq in df.items()}


def compute_bm25_sparse_vector(
    text_or_corpus: str | list[str],
    idf_dict: dict[str, float] | None = None,
    single_query: bool = False,
) -> dict[int, float] | list[dict[int, float]]:
    """为单条查询文本或文档语料库计算 BM25 稀疏向量。

    返回格式为 Milvus SparseFloatVector：dict[term_index -> bm25_score]。
    term_index = hash(token) & 0xFFFFFFFF，与 Milvus 内部 hash 方式对齐。

    Args:
        text_or_corpus: 单条查询文本（single_query=True）或文档列表（single_query=False）。
        idf_dict: 预计算的 IDF 字典。若为 None 且 single_query=False，
                  自动用 corpus 计算 IDF（适用于索引构建）。
        single_query: True = 查询模式，False = 文档模式。

    Returns:
        single_query=True:  dict[int, float]  — 单条稀疏向量。
        single_query=False: list[dict[int, float]] — 文档稀疏向量列表。
    """
    import math

    k1, b = 1.5, 0.75

    if single_query:
        tokens = _tokenize_chinese(text_or_corpus)  # type: ignore[assignment]
        if idf_dict is None or not tokens:
            return {}
        doc_len = len(tokens)
        avgdl = max(doc_len, 1)
        seen: set[str] = set()
        scores: dict[int, float] = {}
        for token in tokens:
            if token in seen:
                continue
            seen.add(token)
            tf = sum(1 for t in tokens if t == token)
            idf_val = idf_dict.get(token, 0)
            score = idf_val * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avgdl))
            term_idx = hash(token) & 0xFFFFFFFF
            if score > 0:
                scores[term_idx] = round(score, 6)
        return scores

    corpus: list[str] = text_or_corpus  # type: ignore[assignment]
    N = len(corpus)
    avgdl = sum(len(_tokenize_chinese(d)) for d in corpus) / N or 1
    if idf_dict is None:
        idf_dict = _compute_idf(corpus)

    result: list[dict[int, float]] = []
    for doc in corpus:
        tokens = _tokenize_chinese(doc)
        doc_len = len(tokens) or 1
        freq = {t: sum(1 for x in tokens if x == t) for t in set(tokens)}
        scores: dict[int, float] = {}
        for token, tf in freq.items():
            idf_val = idf_dict.get(token, 0)
            score = idf_val * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avgdl))
            term_idx = hash(token) & 0xFFFFFFFF
            if score > 0:
                scores[term_idx] = round(score, 6)
        result.append(scores)

    return result


# ----------------------------------------------------------------------
# RRF Fusion
# ----------------------------------------------------------------------


def _rrf_fusion(
    results_a: list[dict[str, Any]],
    results_b: list[dict[str, Any]],
    top_k: int = 5,
    score_threshold: float | None = None,
    k: int = 60,
) -> list[dict[str, Any]]:
    """倒数排名融合（RRF），将两路检索结果合并排序。

    RRF score = Σ 1 / (k + rank_i)，k 越大各路权重越均衡（默认 60）。
    """
    rrf_scores: dict[int, dict[str, Any]] = {}

    for rank, hit in enumerate(results_a):
        entity_id = hit["id"]
        entry = rrf_scores.get(entity_id)
        if entry is None:
            rrf_scores[entity_id] = hit
            hit["rrf_score"] = 1 / (k + rank + 1)
        else:
            entry["rrf_score"] = entry.get("rrf_score", 0) + 1 / (k + rank + 1)

    for rank, hit in enumerate(results_b):
        entity_id = hit["id"]
        entry = rrf_scores.get(entity_id)
        if entry is None:
            rrf_scores[entity_id] = hit
            hit["rrf_score"] = 1 / (k + rank + 1)
        else:
            entry["rrf_score"] = entry.get("rrf_score", 0) + 1 / (k + rank + 1)

    fused = sorted(rrf_scores.values(), key=lambda x: x.get("rrf_score", 0), reverse=True)

    output: list[dict[str, Any]] = []
    for hit in fused[:top_k]:
        if score_threshold is None or hit.get("rrf_score", 0) >= score_threshold:
            output.append(hit)

    return output


class MilvusClient:
    """Milvus 向量数据库操作封装。

    提供连接管理、Collection 创建、向量插入与相似度检索等核心操作。
    """

    def __init__(
        self,
        host: str,
        port: int,
        collection_name: str,
        vector_dim: int,
        index_type: str = "IVF_FLAT",
        metric_type: str = "IP",
    ) -> None:
        """初始化 MilvusClient。

        Args:
            host: Milvus 服务地址。
            port: Milvus 服务端口。
            collection_name: Collection 名称。
            vector_dim: 向量维度。
            index_type: 索引类型，支持 ``"IVF_FLAT"`` 或 ``"HNSW"``。
            metric_type: 距离度量类型，``"IP"``（内积）或 ``"L2"``（欧氏距离）。
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.vector_dim = vector_dim
        self.index_type = index_type.upper()
        self.metric_type = metric_type.upper()
        self._conn_alias: str = f"milvus_{collection_name}"
        self._collection: Collection | None = None

    # ------------------------------------------------------------------
    # 连接管理
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """建立到 Milvus 服务的连接。

        Raises:
            MilvusException: 连接失败时抛出。
        """
        try:
            connections.connect(
                alias=self._conn_alias,
                host=self.host,
                port=str(self.port),
            )
        except MilvusException as e:
            raise MilvusException(
                code=e.code if hasattr(e, "code") else -1,
                message=f"Milvus 连接失败 [host={self.host}, port={self.port}]: {e}",
            ) from e

    def disconnect(self) -> None:
        """关闭连接并释放资源。"""
        if connections.has_connection(self._conn_alias):
            connections.disconnect(alias=self._conn_alias)
        self._collection = None

    # ------------------------------------------------------------------
    # Collection 管理
    # ------------------------------------------------------------------

    def collection_exists(self) -> bool:
        """检查 Collection 是否已存在。

        Returns:
            存在返回 ``True``，否则返回 ``False``。
        """
        return utility.has_collection(self.collection_name, using=self._conn_alias)

    def create_collection(self) -> None:
        """创建 Collection（若不存在）并建立向量索引。

        Schema 定义：
            - ``id``           — INT64, 主键, 自增
            - ``vector``       — FLOAT_VECTOR, dim=vector_dim（稠密向量，语义搜索）
            - ``sparse_vector``— SPARSE_FLOAT_VECTOR（BM25 稀疏向量，关键词搜索）
            - ``text``         — VARCHAR, max_length=65535
            - ``metadata``     — JSON

        Raises:
            MilvusException: Collection 已存在或创建失败时抛出。
        """
        if self.collection_exists():
            raise MilvusException(
                code=200,
                message=f"Collection '{self.collection_name}' 已存在，无需重复创建。",
            )

        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=True,
                description="主键，自增 ID",
            ),
            FieldSchema(
                name="vector",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.vector_dim,
                description="文本向量嵌入（稠密，语义搜索）",
            ),
            FieldSchema(
                name="sparse_vector",
                dtype=DataType.SPARSE_FLOAT_VECTOR,
                description="BM25 稀疏向量（关键词搜索）",
            ),
            FieldSchema(
                name="text",
                dtype=DataType.VARCHAR,
                max_length=65535,
                description="原始文本内容",
            ),
            FieldSchema(
                name="metadata",
                dtype=DataType.JSON,
                description="元数据（来源文件、页码等）",
            ),
        ]

        schema = CollectionSchema(
            fields=fields,
            description="DiagRAG 医学诊断知识库 Collection",
        )

        collection = Collection(
            name=self.collection_name,
            schema=schema,
            using=self._conn_alias,
        )

        # 构建索引参数
        if self.index_type == "HNSW":
            index_params = {
                "index_type": "HNSW",
                "metric_type": self.metric_type,
                "params": {"M": 16, "efConstruction": 200},
            }
        else:
            # 默认 IVF_FLAT
            nlist = max(128, self.vector_dim * 4)
            index_params = {
                "index_type": "IVF_FLAT",
                "metric_type": self.metric_type,
                "params": {"nlist": nlist},
            }

        index_name = f"{self.collection_name}_vector_idx"
        try:
            collection.create_index(
                field_name="vector",
                index_params=index_params,
                index_name=index_name,
            )
            # 稀疏向量（BM25）索引
            sparse_index_name = f"{self.collection_name}_sparse_idx"
            collection.create_index(
                field_name="sparse_vector",
                index_params={
                    "index_type": "SPARSE_INVERTED_INDEX",
                    "metric_type": "IP",
                    "params": {},
                },
                index_name=sparse_index_name,
            )
            collection.flush()
        except MilvusException as e:
            raise MilvusException(
                code=e.code if hasattr(e, "code") else -1,
                message=f"创建索引失败: {e}",
            ) from e

    def drop_collection(self) -> None:
        """删除 Collection（若存在）。

        Raises:
            MilvusException: 删除失败时抛出。
        """
        if not self.collection_exists():
            return

        try:
            utility.drop_collection(self.collection_name, using=self._conn_alias)
            self._collection = None
        except MilvusException as e:
            raise MilvusException(
                code=e.code if hasattr(e, "code") else -1,
                message=f"删除 Collection '{self.collection_name}' 失败: {e}",
            ) from e

    def _get_collection(self) -> Collection:
        """获取或加载 Collection 实例。"""
        if self._collection is not None:
            return self._collection

        if not self.collection_exists():
            raise MilvusException(
                code=100,
                message=f"Collection '{self.collection_name}' 不存在，请先调用 create_collection()。",
            )

        collection = Collection(self.collection_name, using=self._conn_alias)
        collection.load()
        self._collection = collection
        return collection

    # ------------------------------------------------------------------
    # 数据操作
    # ------------------------------------------------------------------

    def insert_chunks(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]],
        vectors: list[list[float]],
        sparse_vectors: list[dict[int, float]] | None = None,
    ) -> list[int]:
        """批量插入向量块数据。

        Args:
            texts: 文本内容列表。
            metadatas: 对应的元数据字典列表。
            vectors: 对应的稠密向量嵌入列表。
            sparse_vectors: 对应的稀疏 BM25 向量列表（term_index -> score）。
                           若为 None，则 sparse_vector 字段留空（查询时不可用）。

        Returns:
            插入记录的 ID 列表。

        Raises:
            MilvusException: 插入失败时抛出。
            ValueError: 三个列表长度不一致时抛出。
        """
        if not (len(texts) == len(metadatas) == len(vectors)):
            raise ValueError(
                f"texts、metadatas、vectors 长度不一致: "
                f"{len(texts)}, {len(metadatas)}, {len(vectors)}",
            )
        if sparse_vectors is not None and len(sparse_vectors) != len(texts):
            raise ValueError(
                f"sparse_vectors 长度与 texts 不一致: "
                f"{len(sparse_vectors)} vs {len(texts)}",
            )

        collection = self._get_collection()

        # entities 顺序须与 schema 字段顺序一致（id 为 auto_id 不传入）
        # sparse_vectors 为 None 时，用空 dict 填充 sparse_vector 字段
        if sparse_vectors is None:
            sparse_vectors = [{} for _ in texts]  # type: ignore[assignment]
        entities: list[Any] = [vectors, sparse_vectors, texts, metadatas]

        try:
            result = collection.insert(entities)
            collection.flush()
            return result.primary_keys  # type: ignore[return-value]
        except MilvusException as e:
            raise MilvusException(
                code=e.code if hasattr(e, "code") else -1,
                message=f"批量插入失败（{len(texts)} 条）: {e}",
            ) from e

    def bm25_search(
        self,
        query_text: str,
        top_k: int = 5,
        score_threshold: float | None = None,
        expr: str | None = None,
    ) -> list[dict[str, Any]]:
        """BM25 关键词检索。

        Args:
            query_text: 查询文本（分词后计算 BM25 得分）。
            top_k: 返回的最近邻数量。
            score_threshold: 可选，分数过滤阈值。
            expr: 可选，Milvus 标量过滤表达式。

        Returns:
            结果列表，每个元素包含 ``id``、``text``、``metadata``、``score``。
        """
        # 计算查询文本的稀疏向量
        sparse_vector = compute_bm25_sparse_vector(query_text, single_query=True)
        if not sparse_vector:
            logger.warning("BM25 查询文本分词结果为空，返回空列表。")
            return []

        collection = self._get_collection()
        search_params = {
            "metric_type": "IP",
            "params": {},
        }

        try:
            results = collection.search(
                data=[sparse_vector],
                anns_field="sparse_vector",
                param=search_params,
                limit=top_k,
                output_fields=["id", "text", "metadata"],
                expr=expr,
            )

            hits = results[0] if results else []
            output: list[dict[str, Any]] = []

            for hit in hits:
                record: dict[str, Any] = {
                    "id": hit.id,
                    "text": hit.entity.get("text", ""),
                    "metadata": hit.entity.get("metadata", {}),
                    "score": hit.score,
                }
                if score_threshold is None or record["score"] >= score_threshold:
                    output.append(record)

            return output

        except MilvusException as e:
            logger.error("BM25 检索失败: %s", e)
            return []

    def hybrid_search(
        self,
        query_text: str,
        query_vector: list[float],
        top_k: int = 5,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
        score_threshold: float | None = None,
        expr: str | None = None,
    ) -> list[dict[str, Any]]:
        """混合检索：向量语义搜索 + BM25 关键词搜索，RRF 融合排名。

        Args:
            query_text: 查询文本（用于 BM25）。
            query_vector: 查询向量（用于语义搜索）。
            top_k: 每路召回数量，最终返回融合后的 top_k 条。
            vector_weight: 向量搜索权重（已废弃，RRF 融合等权重）。
            bm25_weight: BM25 权重（已废弃，RRF 融合等权重）。
            score_threshold: 可选，分数过滤阈值。
            expr: 可选，Milvus 标量过滤表达式。

        Returns:
            结果列表，RRF 融合后按综合得分排序。
        """
        vector_results = self.search(
            query_vector=query_vector,
            top_k=top_k,
            expr=expr,
        )
        bm25_results = self.bm25_search(
            query_text=query_text,
            top_k=top_k,
            expr=expr,
        )
        return _rrf_fusion(vector_results, bm25_results, top_k, score_threshold)

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        score_threshold: float | None = None,
        expr: str | None = None,
    ) -> list[dict[str, Any]]:
        """向量相似度检索。

        Args:
            query_vector: 查询向量。
            top_k: 返回的最近邻数量。
            score_threshold: 可选，分数过滤阈值（仅返回 score >= threshold 的结果）。
            expr: 可选，Milvus 标量过滤表达式。

        Returns:
            结果列表，每个元素包含 ``id``、``text``、``metadata``、``score``。
        """
        collection = self._get_collection()

        search_params: dict[str, Any]
        if self.index_type == "HNSW":
            search_params = {"metric_type": self.metric_type, "params": {"ef": 128}}
        else:
            search_params = {"metric_type": self.metric_type, "params": {"nprobe": 16}}

        try:
            results = collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=top_k,
                output_fields=["id", "text", "metadata"],
                expr=expr,
            )

            hits = results[0] if results else []
            output: list[dict[str, Any]] = []

            for hit in hits:
                record: dict[str, Any] = {
                    "id": hit.id,
                    "text": hit.entity.get("text", ""),
                    "metadata": hit.entity.get("metadata", {}),
                    "score": hit.score,
                }
                if score_threshold is None or record["score"] >= score_threshold:
                    output.append(record)

            return output

        except MilvusException as e:
            raise MilvusException(
                code=e.code if hasattr(e, "code") else -1,
                message=f"向量检索失败: {e}",
            ) from e

    def get_collection_stats(self) -> dict[str, Any]:
        """返回 Collection 的统计信息（实体数量）。"""
        collection = self._get_collection()
        try:
            stats = collection.num_entities
            return {"collection_name": self.collection_name, "num_entities": stats}
        except MilvusException:
            return {"collection_name": self.collection_name, "num_entities": -1}

    def __enter__(self) -> "MilvusClient":
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.disconnect()
