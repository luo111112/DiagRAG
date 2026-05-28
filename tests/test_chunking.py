# 运行命令：pytest tests/test_chunking.py -v
"""
DiagRAG 文档分块模块测试套件。

覆盖 document_loader、text_splitter、chunk_pipeline 三个核心组件。
"""

import json
import tempfile
from pathlib import Path

import pytest

from src.chunk_pipeline import ChunkPipeline
from src.document_loader import DocumentLoader
from src.text_splitter import TextSplitter


# =============================================================================
# test_document_loader_txt
# =============================================================================

class TestDocumentLoader:
    """DocumentLoader 测试。"""

    def test_document_loader_txt(self, tmp_path: Path):
        """用临时 txt 文件验证 load_document 返回正确结构和内容。"""
        loader = DocumentLoader()

        txt_file = tmp_path / "sample.txt"
        txt_file.write_text("这是第一段。\n\n这是第二段。", encoding="utf-8")

        docs = loader.load_document(txt_file)

        assert len(docs) == 1, "txt 文件应返回单个文档片段"
        assert "page_content" in docs[0]
        assert "metadata" in docs[0]
        assert docs[0]["page_content"] == "这是第一段。\n\n这是第二段。"
        assert docs[0]["metadata"]["source"] == str(txt_file)
        assert docs[0]["metadata"]["file_type"] == "txt"
        assert docs[0]["metadata"]["doc_name"] == "sample.txt"

    def test_document_loader_md(self, tmp_path: Path):
        """用临时 md 文件验证 load_document 返回正确结构和内容。"""
        loader = DocumentLoader()

        md_file = tmp_path / "sample.md"
        md_file.write_text("# 标题\n\n正文内容。", encoding="utf-8")

        docs = loader.load_document(md_file)

        assert len(docs) == 1
        assert docs[0]["metadata"]["file_type"] == "md"
        assert docs[0]["metadata"]["doc_name"] == "sample.md"

    def test_document_loader_unsupported(self, tmp_path: Path):
        """不支持的文件类型应抛出 ValueError。"""
        loader = DocumentLoader()
        bad_file = tmp_path / "doc.docx"
        bad_file.write_text("fake", encoding="utf-8")

        with pytest.raises(ValueError, match="不支持的文件类型"):
            loader.load_document(bad_file)

    def test_document_loader_not_found(self):
        """不存在的文件应抛出 FileNotFoundError。"""
        loader = DocumentLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_document("this_file_does_not_exist_123.txt")


# =============================================================================
# test_text_splitter_basic
# =============================================================================

class TestTextSplitter:
    """TextSplitter 测试。"""

    def test_text_splitter_basic(self):
        """简单文本分割，验证 chunk 数量和长度。"""
        splitter = TextSplitter(chunk_size=20, chunk_overlap=5, min_chunk_length=5)

        docs = [
            {
                "page_content": "医学诊断是临床工作的重要环节。医生需要根据患者的症状、体征以及辅助检查结果进行综合分析，从而做出准确的诊断。",
                "metadata": {"source": "test.txt", "doc_name": "test.txt"},
            }
        ]

        chunks = splitter.split_documents(docs)

        assert len(chunks) >= 1, "应至少产生 1 个 chunk"
        for chunk in chunks:
            assert "text" in chunk
            assert "metadata" in chunk
            assert len(chunk["text"]) >= 5, "每个 chunk 长度应 >= min_chunk_length"

    def test_text_splitter_chunks_have_metadata(self):
        """每个 chunk 的 metadata 应包含 chunk_id 和 chunk_index。"""
        splitter = TextSplitter(chunk_size=30, chunk_overlap=5, min_chunk_length=5)

        docs = [
            {
                "page_content": "第一段内容。\n第二段内容。\n第三段内容。",
                "metadata": {"source": "test.txt", "doc_name": "test.txt"},
            }
        ]

        chunks = splitter.split_documents(docs)

        assert len(chunks) > 0
        chunk_ids = [c["metadata"]["chunk_id"] for c in chunks]
        chunk_indices = [c["metadata"]["chunk_index"] for c in chunks]
        assert chunk_ids == list(range(len(chunks))), "chunk_id 应全局递增"
        assert chunk_indices == list(range(len(chunks))), "chunk_index 应从 0 开始"

    def test_text_splitter_respects_min_length(self):
        """短于 min_chunk_length 的片段应被丢弃。"""
        splitter = TextSplitter(chunk_size=200, chunk_overlap=0, min_chunk_length=20)

        docs = [
            {
                "page_content": "短文本",
                "metadata": {"source": "short.txt", "doc_name": "short.txt"},
            }
        ]

        chunks = splitter.split_documents(docs)
        assert len(chunks) == 0, "短于 min_chunk_length 的文本不应产生 chunk"


# =============================================================================
# test_chunk_pipeline_integration
# =============================================================================

class TestChunkPipeline:
    """ChunkPipeline 端到端集成测试。"""

    def test_chunk_pipeline_integration(self, tmp_path: Path):
        """临时目录含一个 txt 和一个 md，运行 pipeline，检查 chunks 元数据完整。"""
        # 准备测试文件
        txt_file = tmp_path / "doc1.txt"
        txt_file.write_text(
            "医学诊断是临床工作的重要环节。医生需要根据患者的症状进行综合分析。"
            "实验室检查包括血常规、尿常规等辅助手段。",
            encoding="utf-8",
        )

        md_file = tmp_path / "doc2.md"
        md_file.write_text(
            "# 诊断流程\n\n第一步是收集病史。第二步是体格检查。第三步是辅助检查。",
            encoding="utf-8",
        )

        # 运行 pipeline
        pipeline = ChunkPipeline(
            chunk_size=50,
            chunk_overlap=10,
            min_chunk_length=10,
        )
        chunks = pipeline.run(tmp_path)

        assert len(chunks) > 0, "pipeline 应产生至少一个 chunk"

        # 验证元数据字段
        for chunk in chunks:
            meta = chunk["metadata"]
            assert "source" in meta, "metadata 应包含 source"
            assert "doc_name" in meta, "metadata 应包含 doc_name"
            assert "chunk_id" in meta, "metadata 应包含 chunk_id"
            assert "chunk_index" in meta, "metadata 应包含 chunk_index"

        # 验证 source 和 doc_name 来自真实文件
        doc_names = {c["metadata"]["doc_name"] for c in chunks}
        assert "doc1.txt" in doc_names
        assert "doc2.md" in doc_names

    def test_pipeline_run_and_save(self, tmp_path: Path):
        """run_and_save 应正确生成 JSON 文件。"""
        txt_file = tmp_path / "doc.txt"
        txt_file.write_text("这是一段足够长的测试文本，用于验证 pipeline 的保存功能是否正常。", encoding="utf-8")

        output_file = tmp_path / "chunks_output.json"
        pipeline = ChunkPipeline(chunk_size=30, chunk_overlap=5, min_chunk_length=5)
        chunks = pipeline.run_and_save(tmp_path, output_file)

        assert output_file.exists(), "输出文件应被创建"
        with open(output_file, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded == chunks, "加载的 JSON 内容应与返回的 chunks 一致"
