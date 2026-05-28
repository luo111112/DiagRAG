"""CLI script: 使用 ChunkPipeline 将 input-dir 中的文档切分为 chunks 并保存为 JSON。"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# 将项目根目录加入路径，以便直接运行本脚本时能 import src 模块
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunk_pipeline import ChunkPipeline

# ─────────────────────────────────────────────────────────────────────────────
# 日志配置
# ─────────────────────────────────────────────────────────────────────────────

_log = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    """
    构建 CLI 参数解析器。

    定义脚本可接受的命令行选项（chunk 大小、重叠、输入输出路径等），
    config.yml 中的值作为默认值，CLI 参数用于覆盖默认，提升脚本灵活性。
    """
    parser = argparse.ArgumentParser(
        description="DiagRAG 文档分块脚本 — 读取 config.yml 作为默认值，CLI 参数覆盖。"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="每个 chunk 的最大字符数（默认从 config.yml 的 chunking.chunk_size 读取）",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="相邻 chunk 之间的重叠字符数（默认从 config.yml 的 chunking.chunk_overlap 读取）",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        dest="min_length",
        default=None,
        help="丢弃长度小于此值的 chunk（默认从 config.yml 的 chunking.min_chunk_length 读取）",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/",
        help="待处理的源文档目录（默认: data/）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/chunks.json",
        help="输出 JSON 文件路径（默认: data/processed/chunks.json）",
    )
    return parser


def main() -> None:
    """
    脚本主入口：执行完整分块流程并输出统计摘要。

    依次完成：参数解析 → Pipeline 初始化 → 目录检查 → 执行 run_and_save →
    生成 chunks_stats.json 统计文件 → 打印人类可读的汇总报告。
    """
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        stream=sys.stdout,
    )

    # 初始化 pipeline（config.yml 自动作为默认值，CLI 参数覆盖）
    pipeline = ChunkPipeline(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        min_chunk_length=args.min_length,
    )

    input_path = Path(args.input_dir)
    if not input_path.is_dir():
        _log.error("输入目录不存在或不是有效目录: %s", input_path)
        sys.exit(1)

    # 确保输出目录存在
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 执行处理
    chunks = pipeline.run_and_save(input_path, output_path)

    # ── 统计信息 ──────────────────────────────────────────────────────────────
    # 从 pipeline 的内部日志输出中我们已经知道文档数量，
    # 这里通过读取 loader 的结果文件来精确统计。
    # ChunkPipeline.run() 会将 documents 存入 _documents，
    # 但不对外暴露；因此我们在 pipeline 外部重新统计一次。
    # 实际上我们已经有了 chunks 数量，只需额外统计源文件数量即可。
    # 由于 DocumentLoader 返回的是文档片段列表而非文件列表，
    # 我们直接统计 input_dir 中的文件数（排除子目录中的同名结果文件）。
    import os

    supported_exts = {".txt", ".md", ".pdf", ".docx", ".html"}
    file_count = sum(
        1
        for f in input_path.rglob("*")
        if f.is_file() and f.suffix.lower() in supported_exts
    )

    avg_chunks = round(len(chunks) / file_count, 2) if file_count > 0 else 0

    # 保存 chunks 的同时，额外保存一份元数据统计文件
    stats_path = output_path.parent / "chunks_stats.json"
    stats = {
        "input_dir": str(input_path.resolve()),
        "output_file": str(output_path.resolve()),
        "files_processed": file_count,
        "chunks_generated": len(chunks),
        "avg_chunks_per_file": avg_chunks,
        "chunk_size": pipeline.chunk_size,
        "chunk_overlap": pipeline.chunk_overlap,
        "min_chunk_length": pipeline.min_chunk_length,
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # 输出统计摘要
    print()
    print("=" * 60)
    print("  ChunkPipeline 执行完毕")
    print("=" * 60)
    print(f"  处理文件数     : {file_count}")
    print(f"  生成 chunks 数: {len(chunks)}")
    print(f"  平均每文件     : {avg_chunks} chunks")
    print(f"  输出文件       : {output_path.resolve()}")
    print(f"  统计文件       : {stats_path.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
