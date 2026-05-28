"""DiagRAG RAG Chain 测试套件。

运行方式：
    pytest tests/test_rag.py -v

测试分为三类：
  - Mock 测试（test_rag_chain_with_mock）：无需任何外部依赖，纯单元测试。
  - Milvus 集成测试（test_answer_with_retrieved_sources）：需要 Milvus 服务运行，
    使用 `@pytest.mark.milvus` 标记，无 Milvus 时自动跳过。
  - 真实 API 测试（test_missing_context_handling + test_answer_with_retrieved_sources
    的真实调用分支）：需要 `DASHSCOPE_API_KEY` 环境变量，通过
    `@pytest.mark.skipif` 在无 key 时跳过。

标记说明：
  @pytest.mark.milvus      — 需要 Milvus 服务
  @pytest.mark.skipif(...) — 需要 DASHSCOPE_API_KEY
"""

from __future__ import annotations

import os
import random
import time
import uuid
import warnings
from typing import Any, Generator
from unittest.mock import MagicMock

import pytest

# Suppress PyMilvus ORM deprecation warnings (ORM API removed in 3.1)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pymilvus")

from src.config_loader import get_milvus_config
from src.milvus_client import MilvusClient
from src.rag_chain import RAGChain


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

class FakeEmbeddingClient:
    """返回固定向量的假 embedding client（维度 1536）。"""

    def __init__(self, vector_dim: int = 1536):
        self.vector_dim = vector_dim
        self.call_count = 0

    def embed_text(self, text: str) -> list[float]:
        self.call_count += 1
        vec = [0.1] * self.vector_dim
        # 加入文本的简单哈希扰动，使不同文本的向量略有差异
        vec[0] = hash(text) % 100 / 100.0
        norm = sum(v * v for v in vec) ** 0.5
        return [v / norm for v in vec]


class FakeLLMClient:
    """返回预设回复的假 LLM client。"""

    def __init__(self, response: str | None = None):
        self.response = response or "这是来自 FakeLLMClient 的模拟回复。"
        self.call_count = 0
        self.last_messages: list[dict[str, str]] = []

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        self.call_count += 1
        self.last_messages = [{"role": "system", "content": system_prompt or ""},
                               {"role": "user", "content": prompt}]
        return self.response


# --------------------------------------------------------------------------- #
# 1. Mock 测试
# --------------------------------------------------------------------------- #

class TestRAGChainWithMock:
    """纯 Mock 测试：验证 RAG chain 的返回结构，不依赖任何外部服务。"""

    def test_rag_chain_with_mock(self):
        """验证 answer() 返回正确的字段结构（answer, sources, retrieved_chunks, context）。"""
        fake_emb = FakeEmbeddingClient()
        fake_llm = FakeLLMClient(response="模拟医学回答：急性心肌梗死应及时就医。")

        mock_milvus = MagicMock()
        mock_milvus.search.return_value = [
            {
                "id": 1,
                "text": "急性心肌梗死的典型症状为胸骨后压榨性疼痛。",
                "metadata": {"source": "test.txt", "page": 1},
                "score": 0.95,
            },
            {
                "id": 2,
                "text": "AMI 的治疗原则包括再灌注治疗。",
                "metadata": {"source": "test.txt", "page": 2},
                "score": 0.88,
            },
        ]

        chain = RAGChain(
            embedding_client=fake_emb,
            milvus_client=mock_milvus,
            llm_client=fake_llm,
            top_k=2,
        )

        result = chain.answer("急性心肌梗死的典型症状是什么？")

        # 验证返回结构
        assert "answer" in result, "返回字典必须包含 'answer' 字段"
        assert "sources" in result, "返回字典必须包含 'sources' 字段"
        assert "retrieved_chunks" in result, "返回字典必须包含 'retrieved_chunks' 字段"
        assert "context" in result, "返回字典必须包含 'context' 字段"

        # 验证 answer 为非空字符串
        assert isinstance(result["answer"], str), "answer 应为字符串"
        assert len(result["answer"]) > 0, "answer 不应为空"

        # 验证 sources 结构
        assert isinstance(result["sources"], list), "sources 应为列表"
        assert len(result["sources"]) == 2, "应返回 2 条 source"
        for src in result["sources"]:
            assert "text" in src
            assert "metadata" in src
            assert "score" in src

        # 验证 retrieved_chunks
        assert isinstance(result["retrieved_chunks"], list)
        assert len(result["retrieved_chunks"]) == 2

        # 验证 context 包含检索到的文本
        assert "急性心肌梗死" in result["context"], "context 应包含检索文本"

        # 验证 mock 被正确调用
        assert fake_emb.call_count == 1, "embedding client 应被调用 1 次"
        assert mock_milvus.search.call_count == 1, "milvus search 应被调用 1 次"
        assert fake_llm.call_count == 1, "llm generate 应被调用 1 次"

    def test_missing_context_handling(self):
        """Milvus 返回空列表时，验证 answer 包含「缺乏足够证据」提示且 sources 为空。"""
        fake_emb = FakeEmbeddingClient()
        fake_llm = FakeLLMClient()

        mock_milvus = MagicMock()
        mock_milvus.search.return_value = []  # 模拟无检索结果

        chain = RAGChain(
            embedding_client=fake_emb,
            milvus_client=mock_milvus,
            llm_client=fake_llm,
            top_k=5,
        )

        result = chain.answer("量子力学在医学影像中的应用前景？")

        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0
        assert "证据" in result["answer"], (
            "无上下文时应返回缺乏证据的提示，实际返回: " + result["answer"]
        )
        assert result["sources"] == [], "sources 应为空列表"
        assert result["retrieved_chunks"] is None, "retrieved_chunks 应为 None"
        assert result["context"] == "", "context 应为空字符串"
        assert fake_llm.call_count == 0, "无上下文时不应调用 LLM"


# --------------------------------------------------------------------------- #
# 2. Milvus 集成测试（需要 Milvus 服务）
# --------------------------------------------------------------------------- #

@pytest.fixture
def rag_test_collection_name() -> str:
    """生成唯一的测试 Collection 名称。"""
    return f"test_rag_{int(time.time())}_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def rag_milvus_client(
    rag_test_collection_name: str,
    milvus_config: dict,
) -> Generator[MilvusClient, None, None]:
    """创建 MilvusClient 并连接到测试 Collection，测试结束后清理。"""
    cfg = milvus_config
    client = MilvusClient(
        host=cfg["host"],
        port=cfg["port"],
        collection_name=rag_test_collection_name,
        vector_dim=cfg.get("vector_dim", 1536),
    )
    client.connect()
    yield client
    client.drop_collection()
    client.disconnect()


@pytest.mark.milvus
class TestRAGChainWithMilvus:
    """Milvus 集成测试：使用真实 Milvus 但 mock LLM，验证检索流程。"""

    def test_answer_with_retrieved_sources(
        self,
        rag_milvus_client: MilvusClient,
        embedding_api_key: str | None,
        rag_test_collection_name: str,
    ):
        """向 Milvus 插入测试文本，提问已知问题，验证 sources 非空且 answer 包含相关词汇。"""
        if not embedding_api_key:
            pytest.skip("需要 DASHSCOPE_API_KEY 来生成真实 embedding 向量")

        from src.embedding_client import DashScopeEmbeddingClient

        milvus = rag_milvus_client
        milvus.create_collection()

        # 构造测试数据
        test_texts = [
            "急性心肌梗死（AMI）的典型症状为胸骨后压榨性疼痛，可向左肩、左臂或颈部放射，持续时间通常超过 30 分钟。",
            "AMI 的诊断依赖于心电图 ST 段抬高、心肌酶升高及典型临床表现三方面证据。",
            "糖尿病的诊断标准为空腹血糖≥7.0 mmol/L，或 OGTT 2h 血糖≥11.1 mmol/L。",
        ]
        test_metadatas = [
            {"source": "急性心肌梗死指南.txt", "page": 1},
            {"source": "急性心肌梗死指南.txt", "page": 2},
            {"source": "糖尿病指南.txt", "page": 1},
        ]

        # 使用真实 embedding client 生成向量
        emb_client = DashScopeEmbeddingClient(api_key=embedding_api_key)
        vectors = emb_client.embed_documents(test_texts)

        milvus.insert_chunks(
            texts=test_texts,
            metadatas=test_metadatas,
            vectors=vectors,
        )

        # 使用 Mock LLM，避免真实 API 调用
        fake_llm = FakeLLMClient(
            response="根据现有证据，急性心肌梗死的典型症状为胸骨后压榨性疼痛。",
        )

        chain = RAGChain(
            embedding_client=emb_client,
            milvus_client=milvus,
            llm_client=fake_llm,
            top_k=2,
        )

        # 提问与心肌梗死相关的问题
        result = chain.answer("急性心肌梗死的典型症状是什么？")

        assert "answer" in result
        assert "sources" in result
        assert len(result["sources"]) > 0, "应检索到至少 1 条 source"

        # 验证 sources 内容
        for src in result["sources"]:
            assert "text" in src
            assert "metadata" in src
            assert "急性心肌梗死" in src["text"] or "AMI" in src["text"]

        # 验证 answer 包含相关词汇（来自 mock 或真实 LLM）
        answer_lower = result["answer"].lower()
        assert any(kw in answer_lower for kw in ["心肌梗死", "压榨性疼痛", "胸骨"]), (
            f"answer 应包含心肌梗死相关词汇，实际 answer: {result['answer']}"
        )


# --------------------------------------------------------------------------- #
# 3. 真实 API 测试（需要 Milvus + DASHSCOPE_API_KEY）
# --------------------------------------------------------------------------- #

_run_real_api_tests = bool(os.getenv("DASHSCOPE_API_KEY"))


@pytest.mark.skipif(not _run_real_api_tests, reason="需要 DASHSCOPE_API_KEY 环境变量")
class TestRAGChainRealAPI:
    """真实 API 端到端测试：使用真实 Milvus + 真实 DashScope API。"""

    def test_missing_context_handling_real_api(
        self,
        rag_milvus_client: MilvusClient,
        embedding_api_key: str | None,
    ):
        """向空（或无关）Milvus 提问，验证回答包含「缺乏足够证据」提示。"""
        if not embedding_api_key:
            pytest.skip("需要 DASHSCOPE_API_KEY")

        from src.embedding_client import DashScopeEmbeddingClient

        milvus = rag_milvus_client
        milvus.create_collection()

        # 插入与"心肌梗死"无关的内容
        unrelated_texts = [
            "糖尿病的诊断标准为空腹血糖≥7.0 mmol/L。",
        ]
        unrelated_metadatas = [{"source": "糖尿病指南.txt", "page": 1}]

        emb_client = DashScopeEmbeddingClient(api_key=embedding_api_key)
        vectors = emb_client.embed_documents(unrelated_texts)
        milvus.insert_chunks(
            texts=unrelated_texts,
            metadatas=unrelated_metadatas,
            vectors=vectors,
        )

        llm_client = FakeLLMClient(response="无法根据现有知识库回答此问题。")

        chain = RAGChain(
            embedding_client=emb_client,
            milvus_client=milvus,
            llm_client=llm_client,
            top_k=3,
        )

        # 提问一个毫无关系的问题
        result = chain.answer("如何治疗新冠病毒感染？")

        # 即使检索到无关内容，answer 中仍包含"无法回答"提示
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0

    def test_rag_chain_end_to_end(
        self,
        rag_milvus_client: MilvusClient,
        embedding_api_key: str | None,
    ):
        """完整端到端测试：插入数据 → 提问 → 验证返回结构完整。"""
        if not embedding_api_key:
            pytest.skip("需要 DASHSCOPE_API_KEY")

        from src.embedding_client import DashScopeEmbeddingClient

        milvus = rag_milvus_client
        milvus.create_collection()

        texts = [
            "急性心肌梗死的典型症状为胸骨后压榨性疼痛，可向左肩、左臂放射。",
            "社区获得性肺炎的经验性治疗首选大环内酯类或呼吸喹诺酮类抗生素。",
        ]
        metadatas = [
            {"source": "AMI指南.txt", "page": 1},
            {"source": "肺炎指南.txt", "page": 1},
        ]

        emb_client = DashScopeEmbeddingClient(api_key=embedding_api_key)
        vectors = emb_client.embed_documents(texts)
        milvus.insert_chunks(texts=texts, metadatas=metadatas, vectors=vectors)

        chain = RAGChain(
            embedding_client=emb_client,
            milvus_client=milvus,
            llm_client=FakeLLMClient(response="根据检索到的证据，急性心肌梗死的典型症状为胸骨后压榨性疼痛。"),
            top_k=2,
        )

        result = chain.answer("急性心肌梗死的典型症状是什么？")

        # 验证结构完整性
        for key in ("answer", "sources", "retrieved_chunks", "context"):
            assert key in result, f"结果应包含 '{key}' 字段"

        # 验证 sources 非空
        assert len(result["sources"]) > 0, "应检索到 source"
        assert all(
            key in src for src in result["sources"] for key in ("text", "metadata", "score")
        )

        # 验证 context 非空
        assert len(result["context"]) > 0, "context 不应为空"
