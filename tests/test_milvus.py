"""DiagRAG Milvus 向量数据库模块测试套件。

运行命令（需 Milvus 服务运行）：
    pytest tests/test_milvus.py -v

若 Milvus 未启动，相关测试会被自动跳过（依赖 conftest.py 的 milvus_available fixture）。

DashScope embedding 测试若无有效 API key 也会自动跳过。
"""

import random
import time
import uuid
from typing import Generator

import pytest

from src.config_loader import get_embedding_config, get_milvus_config
from src.embedding_client import DashScopeEmbeddingClient
from src.milvus_client import MilvusClient

# -------------------------------------------------------------------------- #
# Fixtures
# -------------------------------------------------------------------------- #


@pytest.fixture
def test_collection_name() -> str:
    """生成唯一的测试 Collection 名称，测试结束后可安全清理。"""
    return f"test_{int(time.time())}_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def milvus_client(test_collection_name: str, milvus_config: dict) -> Generator[MilvusClient, None, None]:
    """创建 MilvusClient 实例并建立连接，测试结束后清理测试 Collection。"""
    cfg = milvus_config
    client = MilvusClient(
        host=cfg["host"],
        port=cfg["port"],
        collection_name=test_collection_name,
        vector_dim=cfg.get("vector_dim", 1536),
    )
    client.connect()
    yield client
    # 清理：删除测试 Collection（若存在）
    client.drop_collection()
    client.disconnect()


@pytest.fixture
def fake_vectors_and_texts() -> tuple[list[list[float]], list[str], list[dict]]:
    """构造 3 条虚构医学文本及对应的 1536 维向量（随机值）。"""
    dim = 1536
    texts = [
        "急性心肌梗死（AMI）的典型症状为胸骨后压榨性疼痛，可向左肩、左臂或颈部放射。",
        "糖尿病的诊断标准为空腹血糖≥7.0 mmol/L，或OGTT 2h血糖≥11.1 mmol/L。",
        "社区获得性肺炎（CAP）的经验性治疗首选大环内酯类或呼吸喹诺酮类抗生素。",
    ]
    vectors = [[random.random() for _ in range(dim)] for _ in texts]
    # 向量归一化（内积度量下搜索更稳定）
    for vec in vectors:
        norm = sum(v * v for v in vec) ** 0.5
        vec[:] = [v / norm for v in vec]
    metadatas = [
        {"source": "指南_急性心肌梗死.txt", "page": 1, "chunk_index": 0},
        {"source": "指南_糖尿病.txt", "page": 2, "chunk_index": 0},
        {"source": "指南_肺炎.txt", "page": 3, "chunk_index": 0},
    ]
    return vectors, texts, metadatas


# -------------------------------------------------------------------------- #
# test_milvus_connection
# -------------------------------------------------------------------------- #

@pytest.mark.milvus
class TestMilvusConnection:
    """Milvus 服务连接测试。"""

    def test_milvus_connection(self, milvus_config: dict):
        """验证能够成功连接到 config.yml 中配置的 Milvus 服务。"""
        from pymilvus import connections

        alias = f"test_conn_{uuid.uuid4().hex[:6]}"
        connections.connect(
            alias=alias,
            host=milvus_config["host"],
            port=str(milvus_config["port"]),
        )
        try:
            assert connections.has_connection(alias), "应成功建立连接"
        finally:
            connections.disconnect(alias=alias)

    def test_milvus_connection_via_client(self, milvus_client: MilvusClient):
        """通过 MilvusClient.connect() 建立连接并确认 collection_exists 可调用。"""
        # milvus_client fixture 已通过 connect() 建立连接
        # 验证 collection_exists 方法可正常调用（返回 False，因为 Collection 尚未创建）
        assert milvus_client.collection_exists() is False, "新建 Client 的 collection 应不存在"


# -------------------------------------------------------------------------- #
# test_create_and_drop_collection
# -------------------------------------------------------------------------- #

@pytest.mark.milvus
class TestCollectionCreateDrop:
    """Collection 创建与删除测试。"""

    def test_create_and_drop_collection(self, milvus_client: MilvusClient, test_collection_name: str):
        """创建测试 Collection，验证 exists，返回 True；再删除，验证 exists，返回 False。"""
        # 创建前应不存在
        assert milvus_client.collection_exists() is False

        # 创建
        milvus_client.create_collection()
        assert milvus_client.collection_exists() is True, f"Collection '{test_collection_name}' 应已创建"

        # 删除
        milvus_client.drop_collection()
        assert milvus_client.collection_exists() is False, f"Collection '{test_collection_name}' 应已删除"

    def test_create_collection_idempotent(self, milvus_client: MilvusClient):
        """重复创建同一 Collection 应抛出 MilvusException。"""
        from pymilvus import MilvusException

        milvus_client.create_collection()
        with pytest.raises(MilvusException, match="已存在"):
            milvus_client.create_collection()

    def test_drop_nonexistent_collection(self, milvus_client: MilvusClient):
        """删除不存在的 Collection 应静默成功（不抛异常）。"""
        assert milvus_client.collection_exists() is False
        milvus_client.drop_collection()  # 不应抛异常
        assert milvus_client.collection_exists() is False


# -------------------------------------------------------------------------- #
# test_insert_and_search
# -------------------------------------------------------------------------- #

@pytest.mark.milvus
class TestInsertAndSearch:
    """向量插入与相似度搜索测试。"""

    def test_insert_and_search(
        self,
        milvus_client: MilvusClient,
        fake_vectors_and_texts: tuple[list[list[float]], list[str], list[dict]],
    ):
        """插入 3 条虚构文本和向量，搜索向量，验证返回结果数量正确。"""
        vectors, texts, metadatas = fake_vectors_and_texts

        milvus_client.create_collection()
        milvus_client.insert_chunks(texts=texts, metadatas=metadatas, vectors=vectors)

        # 以第一个向量作为查询向量（内积度量），top_k=3 应返回全部 3 条
        results = milvus_client.search(query_vector=vectors[0], top_k=3)

        assert len(results) == 3, f"top_k=3 应返回 3 条结果，实际返回 {len(results)} 条"

        # 验证结果包含必要字段
        for r in results:
            assert "id" in r
            assert "text" in r
            assert "metadata" in r
            assert "score" in r

    def test_search_top_k_respected(
        self,
        milvus_client: MilvusClient,
        fake_vectors_and_texts: tuple[list[list[float]], list[str], list[dict]],
    ):
        """验证 search 参数 top_k 被正确遵守（不超过请求数量）。"""
        vectors, texts, metadatas = fake_vectors_and_texts

        milvus_client.create_collection()
        milvus_client.insert_chunks(texts=texts, metadatas=metadatas, vectors=vectors)

        # 只请求 2 条
        results = milvus_client.search(query_vector=vectors[0], top_k=2)
        assert len(results) == 2, f"top_k=2 应返回 2 条结果，实际返回 {len(results)} 条"

    def test_search_score_threshold(
        self,
        milvus_client: MilvusClient,
        fake_vectors_and_texts: tuple[list[list[float]], list[str], list[dict]],
    ):
        """验证 score_threshold 过滤功能。"""
        vectors, texts, metadatas = fake_vectors_and_texts

        milvus_client.create_collection()
        milvus_client.insert_chunks(texts=texts, metadatas=metadatas, vectors=vectors)

        results_all = milvus_client.search(query_vector=vectors[0], top_k=3)
        results_filtered = milvus_client.search(
            query_vector=vectors[0], top_k=3, score_threshold=0.99
        )

        # 设置极高阈值应返回更少结果
        assert len(results_filtered) <= len(results_all)
        if results_filtered:
            for r in results_filtered:
                assert r["score"] >= 0.99

    def test_get_collection_stats(
        self,
        milvus_client: MilvusClient,
        fake_vectors_and_texts: tuple[list[list[float]], list[str], list[dict]],
    ):
        """验证 get_collection_stats 返回正确实体数量。"""
        vectors, texts, metadatas = fake_vectors_and_texts

        milvus_client.create_collection()
        milvus_client.insert_chunks(texts=texts, metadatas=metadatas, vectors=vectors)

        stats = milvus_client.get_collection_stats()
        assert stats["num_entities"] == 3


# -------------------------------------------------------------------------- #
# test_embedding_client
# -------------------------------------------------------------------------- #

@pytest.mark.embedding
class TestEmbeddingClient:
    """DashScope Embedding API 测试。"""

    def test_embedding_dimensions(self, embedding_api_key: str | None):
        """调用 DashScope API，验证返回向量维度为 1536（或模型指定维度）。无有效 key 时跳过。"""
        skip_if_no_api_key(embedding_api_key)

        client = DashScopeEmbeddingClient(api_key=embedding_api_key)
        text = "急性心肌梗死的典型症状是胸骨后压榨性疼痛。"
        vector = client.embed_text(text)

        assert isinstance(vector, list), "embedding 结果应为 list"
        assert len(vector) > 0, "embedding 结果不应为空"
        assert all(isinstance(v, (int, float)) for v in vector), "embedding 值应为数值类型"
        # text-embedding-v1 模型输出 1536 维向量
        assert len(vector) == 1536, f"text-embedding-v1 应返回 1536 维向量，实际维度为 {len(vector)}"

    def test_embed_documents(self, embedding_api_key: str | None):
        """验证批量嵌入多条文本，返回向量数量与输入文本数量一致。"""
        skip_if_no_api_key(embedding_api_key)

        client = DashScopeEmbeddingClient(api_key=embedding_api_key)
        texts = [
            "糖尿病的诊断标准",
            "心肌梗死的治疗原则",
            "社区获得性肺炎的临床表现",
        ]
        vectors = client.embed_documents(texts)

        assert len(vectors) == len(texts), f"应返回 {len(texts)} 条向量，实际返回 {len(vectors)} 条"
        for v in vectors:
            assert len(v) == 1536, f"每条向量维度应为 1536，实际为 {len(v)}"

    def test_embedding_consistency(self, embedding_api_key: str | None):
        """同一文本两次调用应返回完全相同（或数值近似）的向量。"""
        skip_if_no_api_key(embedding_api_key)

        client = DashScopeEmbeddingClient(api_key=embedding_api_key)
        text = "急性心肌梗死"

        vec1 = client.embed_text(text)
        vec2 = client.embed_text(text)

        assert len(vec1) == len(vec2)
        assert vec1 == vec2, "相同文本的 embedding 应完全一致"


def skip_if_no_api_key(api_key: str | None) -> None:
    """若 api_key 为空或无效，抛出 SkipTest。"""
    if not api_key:
        pytest.skip("DASHSCOPE_API_KEY 环境变量未设置或配置文件中 dashscope_api_key 为空")
    if api_key.startswith("${") or api_key == "":
        pytest.skip("dashscope_api_key 未配置有效值（仍为占位符）")
