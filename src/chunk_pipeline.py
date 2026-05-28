"""文档分块管道：将目录中的文档加载并切分为 chunks，支持保存为 JSON。"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# 配置加载器（若暂未实现则使用硬编码默认值）
try:
    from src.config_loader import load_config
except ImportError:  # pragma: no cover
    load_config = None  # type: ignore

from src.document_loader import DocumentLoader
from src.text_splitter import TextSplitter


# ─────────────────────────────────────────────────────────────────────────────
# 日志配置
# ─────────────────────────────────────────────────────────────────────────────

_log = logging.getLogger(__name__)
if not _log.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s — %(message)s")
    )
    _log.addHandler(_handler)
    _log.setLevel(logging.INFO)


# ─────────────────────────────────────────────────────────────────────────────
# 默认 chunking 配置（config_loader 不可用时使用）
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_CHUNK_SIZE = 500
_DEFAULT_CHUNK_OVERLAP = 50
_DEFAULT_MIN_CHUNK_LENGTH = 20


# ─────────────────────────────────────────────────────────────────────────────
# ChunkPipeline
# ─────────────────────────────────────────────────────────────────────────────

class ChunkPipeline:
    """
    文档分块管道。

    负责将指定目录中的文档加载并切分为 chunks，
    同时将结果序列化为 JSON 文件供后续向量化和检索使用。

    Parameters
    ----------
    chunk_size : int, optional
        每个 chunk 的最大字符数（默认从 config.yml 的 chunking.chunk_size 读取）。
    chunk_overlap : int, optional
        相邻 chunk 之间的重叠字符数。
    min_chunk_length : int, optional
        丢弃长度小于此值的 chunk。
    config_path : str, optional
        config.yml 的路径（仅当 config_loader 可用时有效）。
    """

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        min_chunk_length: Optional[int] = None,
        config_path: Optional[str] = None,
    ) -> None:
        """
        初始化 ChunkPipeline。

        优先级：构造函数显式参数 > config.yml > 硬编码默认值。
        同时创建 DocumentLoader 和 TextSplitter 实例，供 run() 方法调用。
        """
        # 从 config.yml 读取 chunking 配置（若调用方未显式覆盖）。
        # config.yml 为主配置源，构造函数参数用于运行时覆盖，两者均缺省时使用硬编码默认值，
        # 保证即在 config_loader 不可用的情况下也能正常运行。
        if load_config is not None:
            try:
                cfg = load_config(config_path)
                ck_cfg = cfg.get("chunking", {})
            except Exception as exc:  # pragma: no cover
                _log.warning("加载 config.yml 失败，使用默认值: %s", exc)
                ck_cfg = {}
        else:
            ck_cfg = {}

        self.chunk_size: int = (
            chunk_size
            if chunk_size is not None
            else ck_cfg.get("chunk_size", _DEFAULT_CHUNK_SIZE)
        )
        self.chunk_overlap: int = (
            chunk_overlap
            if chunk_overlap is not None
            else ck_cfg.get("chunk_overlap", _DEFAULT_CHUNK_OVERLAP)
        )
        self.min_chunk_length: int = (
            min_chunk_length
            if min_chunk_length is not None
            else ck_cfg.get("min_chunk_length", _DEFAULT_MIN_CHUNK_LENGTH)
        )

        self._loader = DocumentLoader()
        self._splitter = TextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            min_chunk_length=self.min_chunk_length,
        )

        _log.info(
            "ChunkPipeline 初始化完成: chunk_size=%d, chunk_overlap=%d, min_chunk_length=%d",
            self.chunk_size,
            self.chunk_overlap,
            self.min_chunk_length,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 公开 API
    # ─────────────────────────────────────────────────────────────────────────

    def run(self, directory_path: str | Path) -> List[Dict[str, Any]]:
        """
        遍历目录、加载文档并切分为 chunks。

        依次调用 DocumentLoader.load_directory（加载）和 TextSplitter.split_documents（切分），
        两阶段均设异常捕获，保证任意阶段失败都能向上抛出便于排查。
        返回的 chunks 可直接用于向量化和存储。

        Parameters
        ----------
        directory_path : str | Path
            待处理的目录路径。

        Returns
        -------
        List[Dict[str, Any]]
            chunks 列表，每个元素为 ``{"text": str, "metadata": dict}``。
        """
        _log.info("开始处理目录: %s", directory_path)
        try:
            documents = self._loader.load_directory(directory_path)
        except Exception as exc:
            _log.error("加载文档目录失败: %s", exc)
            raise

        _log.info("成功加载 %d 个文档片段，开始切分 ...", len(documents))
        try:
            chunks = self._splitter.split_documents(documents)
        except Exception as exc:
            _log.error("文本切分失败: %s", exc)
            raise

        _log.info("切分完成，共生成 %d 个 chunks", len(chunks))
        return chunks

    def save_chunks(
        self,
        chunks: List[Dict[str, Any]],
        output_path: str | Path,
    ) -> None:
        """
        将 chunks 列表保存为 JSON 文件。

        确保输出目录存在（自动创建父目录），以 UTF-8 编码写出格式化 JSON，
        方便人工查阅和调试；序列化结果可被向量数据库直接加载使用。

        Parameters
        ----------
        chunks : List[Dict[str, Any]]
            chunks 列表，每个元素必须包含 ``text`` 和 ``metadata`` 键。
        output_path : str | Path
            输出 JSON 文件路径。

        Raises
        ------
        IOError
            文件写入失败时抛出。
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        _log.info("保存 chunks 至: %s", path)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            _log.error("写入 JSON 文件失败: %s", exc)
            raise

        _log.info("写入完成: %d chunks → %s", len(chunks), path)

    def run_and_save(
        self,
        directory_path: str | Path,
        output_path: str | Path,
    ) -> List[Dict[str, Any]]:
        """
        便捷方法：依次调用 ``run()`` → ``save_chunks()``。

        将"加载→切分→持久化"三步合为一次调用，
        CLI 脚本和快速测试场景使用该方法最方便。

        Parameters
        ----------
        directory_path : str | Path
            待处理的目录路径。
        output_path : str | Path
            输出 JSON 文件路径。

        Returns
        -------
        List[Dict[str, Any]]
            生成的 chunks 列表。
        """
        chunks = self.run(directory_path)
        self.save_chunks(chunks, output_path)
        return chunks


# =============================================================================
# 入口点（可直接运行）
# =============================================================================
# 支持直接运行本文件进行快速测试或调试：python -m src.chunk_pipeline <dir> -o chunks.json
# 命令行参数定义在此处，运行时按需覆盖 config.yml 中的默认值。
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DiagRAG 文档分块管道")
    # 支持直接运行本脚本进行快速测试或调试：python scripts/generate_chunks.py <dir> -o chunks.json
    # 命令行参数定义在此处，运行时按需覆盖 config.yml 中的默认值。
    parser.add_argument(
        "directory",
        type=str,
        help="待处理的文档目录路径",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="chunks.json",
        help="输出 JSON 文件路径（默认: chunks.json）",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="覆盖 config.yml 中的 chunk_size",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="覆盖 config.yml 中的 chunk_overlap",
    )
    parser.add_argument(
        "--min-chunk-length",
        type=int,
        default=None,
        help="覆盖 config.yml 中的 min_chunk_length",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    pipeline = ChunkPipeline(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        min_chunk_length=args.min_chunk_length,
    )
    pipeline.run_and_save(args.directory, args.output)
