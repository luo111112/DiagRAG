"""Re-ranking components for retrieval result post-processing.

本模块提供两种重排序策略：

1. **LLMRanker**（默认）：基于 DashScope Qwen LLM 的交叉编码器风格重排序。
   将 query 和每个 doc 拼接后送入 LLM，由 LLM 评分排序。
   优势：深度语义理解，适合医学等专业领域。

2. **BM25Reranker**：基于 BM25 词项匹配的轻量级重排序。
   补充向量检索无法捕捉的精确词项匹配。

两种策略的结果可通过加权融合（RRF 或线性组合）得到最终排序。
"""

from __future__ import annotations

import logging
import math
import re
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.llm_client import DashScopeLLMClient

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# BM25 Reranker (lightweight, no API needed)
# ----------------------------------------------------------------------


class BM25Reranker:
    """基于 BM25 词项匹配的轻量级重排序器。

    在向量语义检索的基础上，补充精确词项匹配得分，
    特别适合医学术语（如"心肌梗死"、"ST段抬高"）的精确匹配。
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        """初始化 BM25Reranker。

        Args:
            k1: 词频饱和参数，值越大对高频词越敏感（默认 1.5）。
            b: 文档长度归一化参数（默认 0.75）。
        """
        self.k1 = k1
        self.b = b
        self._idf: dict[str, float] = {}
        self._avgdl: float = 0.0

    def _tokenize(self, text: str) -> list[str]:
        """中文分词，基于 jieba 精确分词，小写化。"""
        import jieba

        tokens = jieba.lcut(text)
        return [t.lower() for t in tokens if t.strip()]

    def fit(self, corpus: list[str]) -> "BM25Reranker":
        """用文档语料库拟合 IDF 参数。

        Args:
            corpus: 文档文本列表。

        Returns:
            返回 self，支持链式调用。
        """
        N = len(corpus)
        df: dict[str, int] = {}
        total_len = 0

        for doc in corpus:
            tokens = set(self._tokenize(doc))
            total_len += len(tokens)
            for token in tokens:
                df[token] = df.get(token, 0) + 1

        self._avgdl = total_len / N if N > 0 else 1.0
        self._idf = {
            term: math.log((N - doc_freq + 0.5) / (doc_freq + 0.5) + 1)
            for term, doc_freq in df.items()
        }
        logger.info("BM25Reranker fitted on %d docs, vocabulary size=%d", N, len(self._idf))
        return self

    def score(self, query: str, document: str) -> float:
        """计算 query 对 document 的 BM25 相关性得分。

        Args:
            query: 查询文本。
            document: 候选文档文本。

        Returns:
            BM25 得分（越高越相关）。
        """
        if not self._idf:
            logger.warning("BM25Reranker not fitted yet, returning 0.0")
            return 0.0

        q_tokens = self._tokenize(query)
        d_tokens = self._tokenize(document)
        doc_len = len(d_tokens) or 1

        freq: dict[str, int] = {}
        for token in d_tokens:
            freq[token] = freq.get(token, 0) + 1

        score = 0.0
        for token in q_tokens:
            if token in freq:
                tf = freq[token]
                idf = self._idf.get(token, 0)
                score += idf * (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * doc_len / self._avgdl)
                )
        return round(score, 6)

    def rerank(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """对检索结果按 BM25 得分重排序。

        Args:
            query: 查询文本。
            chunks: 检索结果列表，每个元素需包含 ``text`` 字段。
            top_k: 最终返回的 top_k 条，None=返回全部。

        Returns:
            重排序后的 chunk 列表，每条追加 ``bm25_score`` 字段。
        """
        scored = []
        for chunk in chunks:
            text = chunk.get("text", "")
            bm25_score = self.score(query, text)
            scored_chunk = {**chunk, "bm25_score": bm25_score}
            scored.append(scored_chunk)

        scored.sort(key=lambda x: x["bm25_score"], reverse=True)
        logger.debug("BM25 reranked %d chunks", len(scored))
        return scored[:top_k] if top_k else scored


# ----------------------------------------------------------------------
# LLM Ranker (cross-encoder style, uses Qwen)
# ----------------------------------------------------------------------


class LLMRanker:
    """基于 LLM（DashScope Qwen）的交叉编码器风格重排序器。

    原理：将 query 和每个候选 doc 拼接为提示词，
    让 LLM 从语义相关性角度判断排序。
    适用于需要深度语义理解的专业领域（如医学诊断问答）。

    支持三种打分模式：
    - ``rank``: 仅输出排序列表（最快，零样本排序）
    - ``score``: 输出 0-10 分的数值评分（较慢，更精确）
    - ``score_with_reason``: 输出评分 + 理由（最慢，最可解释）
    """

    # 提示词模板（预留在类中，方便子类覆写）
    RANK_PROMPT_TEMPLATE = (
        "你是一个专业的医学信息检索评估专家。\n"
        "请根据以下【查询】与【文档】的语义相关性，对候选文档进行排序。\n"
        "相关性高的文档应该排在前面。\n"
        "只输出排序结果，格式为：\n"
        "1. [文档编号]\n"
        "2. [文档编号]\n"
        "（以此类推）\n"
        "\n"
        "【查询】：{query}\n"
        "\n"
        "【候选文档】：\n"
        "{docs}\n"
        "\n"
        "排序结果："
    )

    SCORE_PROMPT_TEMPLATE = (
        "你是一个专业的医学信息检索评估专家。\n"
        "请为以下【查询】与每篇【文档】的相关性打分（0-10分），"
        "0分=完全不相关，10分=高度相关。\n"
        "只输出分数，格式为：\n"
        "文档1: X分\n"
        "文档2: Y分\n"
        "（以此类推）\n"
        "\n"
        "【查询】：{query}\n"
        "\n"
        "【候选文档】：\n"
        "{docs}\n"
    )

    SCORE_REASON_PROMPT_TEMPLATE = (
        "你是一个专业的医学信息检索评估专家。\n"
        "请为以下【查询】与每篇【文档】的相关性打分（0-10分）并说明理由。\n"
        "0分=完全不相关，10分=高度相关。\n"
        "输出格式为：\n"
        "文档1: X分（理由：...）\n"
        "文档2: Y分（理由：...）\n"
        "\n"
        "【查询】：{query}\n"
        "\n"
        "【候选文档】：\n"
        "{docs}\n"
    )

    def __init__(
        self,
        llm_client: "DashScopeLLMClient",
        mode: str = "score",
        top_k: int = 5,
        enable_reason: bool = False,
        max_docs_per_call: int = 10,
        fallback_to_rank: bool = True,
    ) -> None:
        """初始化 LLM 重排序器。

        Args:
            llm_client: LLM 客户端（DashScopeLLMClient）。
            mode: 评分模式。``"rank"``=仅排序，``"score"``=数值评分，
                  ``"score_with_reason"``=评分+理由。
            top_k: 重排序后返回的 top_k 条。
            enable_reason: 是否在结果中包含 LLM 的评分理由（仅 mode=score 时有效）。
            max_docs_per_call: 单次 API 调用最多处理的文档数（控制 token 消耗）。
            fallback_to_rank: 当 score 模式解析失败时，是否降级为 rank 模式。
        """
        valid_modes = {"rank", "score", "score_with_reason"}
        if mode not in valid_modes:
            raise ValueError(f"mode 必须是 {valid_modes} 之一，当前值为 {mode!r}")

        self.llm_client = llm_client
        self.mode = mode
        self.top_k = top_k
        self.enable_reason = enable_reason
        self.max_docs_per_call = max_docs_per_call
        self.fallback_to_rank = fallback_to_rank

        actual_mode = mode
        if enable_reason and mode == "score":
            actual_mode = "score_with_reason"

        self._prompt_template = {
            "rank": self.RANK_PROMPT_TEMPLATE,
            "score": self.SCORE_PROMPT_TEMPLATE,
            "score_with_reason": self.SCORE_REASON_PROMPT_TEMPLATE,
        }[actual_mode]

        logger.info(
            "LLMRanker initialized: mode=%s, top_k=%d, max_docs_per_call=%d",
            actual_mode, top_k, max_docs_per_call,
        )

    def _build_docs_text(self, chunks: list[dict[str, Any]]) -> str:
        """将 chunks 列表格式化为提示词中的文档描述文本。"""
        lines: list[str] = []
        for i, chunk in enumerate(chunks, 1):
            text = chunk.get("text", "")
            # 截断过长文档，避免超出 token 限制
            truncated = text[:600] + "..." if len(text) > 600 else text
            meta = chunk.get("metadata") or {}
            source = meta.get("source_file", "未知来源")
            lines.append(f"[文档{i}]（来源：{source}）\n{truncated}")
        return "\n\n".join(lines)

    def _parse_rank_response(
        self, response: str, num_docs: int
    ) -> list[tuple[int, float]]:
        """从 rank 模式响应中解析文档编号顺序，返回 [(doc_idx_0based, score), ...]."""
        # 匹配形如 "1." "2." 或 "[文档1]" 的行
        pattern = re.compile(r"(?:^\s*(?:\d+[\.\)]\s*)?(?:\[?文档?(\d+)\]?)|(?:^\s*(\d+)[\.\)])", re.MULTILINE)
        order: list[int] = []
        for m in pattern.finditer(response):
            doc_num = m.group(1) or m.group(2)
            if doc_num:
                idx = int(doc_num) - 1  # 转 0-based
                if 0 <= idx < num_docs:
                    order.append(idx)

        if not order:
            logger.warning("无法解析 rank 响应，返回原文顺序: %s", response[:200])
            return [(i, float(num_docs - i)) for i in range(num_docs)]

        # 排名靠前得分高
        return [(idx, float(num_docs - rank)) for rank, idx in enumerate(order)]

    def _parse_score_response(
        self, response: str, num_docs: int, with_reason: bool = False
    ) -> list[tuple[int, float, str | None]]:
        """从 score 模式响应中解析分数，返回 [(doc_idx_0based, score, reason_or_none), ...]."""
        reason: str | None = None
        if with_reason:
            # 匹配 "文档X: Y分（理由：...）" 或 "文档 X: Y 分"
            pattern = re.compile(
                r"文档\s*(\d+)\s*[:：]\s*(\d+(?:\.\d+)?)\s*(?:分)?\s*(?:（理由[：:]\s*([^）]+)）)?",
                re.MULTILINE,
            )
        else:
            # 匹配 "文档X: Y分" 或 "文档 X: Y"
            pattern = re.compile(
                r"文档\s*(\d+)\s*[:：]\s*(\d+(?:\.\d+)?)\s*(?:分)?",
                re.MULTILINE,
            )

        results: dict[int, tuple[float, str | None]] = {}
        for m in pattern.finditer(response):
            idx = int(m.group(1)) - 1  # 转 0-based
            score = float(m.group(2))
            reason = m.group(3) if with_reason else None
            if 0 <= idx < num_docs:
                results[idx] = (min(max(score, 0.0), 10.0), reason)

        if not results:
            logger.warning("无法解析 score 响应中的文档编号，返回原文顺序: %s", response[:200])
            return [(i, 5.0, None) for i in range(num_docs)]

        # 补全未找到的文档（给予中等偏低分）
        for i in range(num_docs):
            if i not in results:
                results[i] = (2.0, None)

        return [(idx, score, reason) for idx, (score, reason) in sorted(results.items())]

    def rerank(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """对候选文档进行 LLM 重排序。

        Args:
            query: 查询文本。
            chunks: 检索结果列表，每个元素需包含 ``text`` 字段。
            top_k: 最终返回的条数，None=使用初始化时的 self.top_k。

        Returns:
            重排序后的 chunk 列表，每条追加 ``llm_score`` 字段。
            mode=score_with_reason 时还会包含 ``llm_reason`` 字段。
        """
        if not chunks:
            return []

        effective_top_k = top_k if top_k is not None else self.top_k

        # 分批处理（避免单次 prompt 过长）
        all_scores: list[tuple[int, float, str | None]] = []
        batch_size = min(self.max_docs_per_call, len(chunks))

        for batch_start in range(0, len(chunks), batch_size):
            batch_chunks = chunks[batch_start : batch_start + batch_size]
            docs_text = self._build_docs_text(batch_chunks)

            prompt = self._prompt_template.format(query=query, docs=docs_text)
            mode = self.mode
            if self.enable_reason and mode == "score":
                mode = "score_with_reason"

            try:
                response = self.llm_client.generate(
                    prompt=prompt,
                    system_prompt=None,
                )
                logger.debug(
                    "LLM rerank batch [%d-%d], mode=%s, response_len=%d",
                    batch_start, batch_start + len(batch_chunks) - 1,
                    mode, len(response),
                )
            except Exception as e:
                logger.error("LLM rerank 调用失败: %s，使用原文顺序", e)
                for i, chunk in enumerate(batch_chunks):
                    all_scores.append((batch_start + i, 0.0, None))
                continue

            # 解析响应
            if mode == "rank":
                raw = self._parse_rank_response(response, len(batch_chunks))
                all_scores.extend((batch_start + idx, score, None) for idx, score in raw)
            else:
                with_reason = mode == "score_with_reason"
                raw = self._parse_score_response(response, len(batch_chunks), with_reason)
                all_scores.extend((batch_start + idx, score, reason) for idx, score, reason in raw)

        # 全局排序（按 llm_score 从高到低）
        all_scores.sort(key=lambda x: x[1], reverse=True)

        # 重新构建 chunks，追加 llm_score 和可选的 llm_reason
        reranked: list[dict[str, Any]] = []
        for orig_idx, llm_score, llm_reason in all_scores[:effective_top_k]:
            chunk = {**chunks[orig_idx], "llm_score": llm_score}
            if llm_reason:
                chunk["llm_reason"] = llm_reason
            reranked.append(chunk)

        logger.info(
            "LLM reranked %d chunks, top score=%.2f",
            len(chunks), all_scores[0][1] if all_scores else 0.0,
        )
        return reranked


# ----------------------------------------------------------------------
# Hybrid Reranker (BM25 + LLM fusion)
# ----------------------------------------------------------------------


class HybridReranker:
    """混合重排序器：融合 BM25Reranker 与 LLMRanker 的排序结果。

    支持两种融合策略：
    - ``rrf``: 倒数排名融合（Reciprocal Rank Fusion），各排序器权重均衡。
    - ``score_linear``: 线性加权，需要两者的绝对分数可比。

    融合公式（RRF）::

        score_final(d) = sum_i 1 / (k + rank_i(d))

    其中 k=60，rank_i(d) 为文档 d 在第 i 个排序器中的排名（从 1 开始）。
    """

    def __init__(
        self,
        llm_ranker: LLMRanker,
        bm25_reranker: BM25Reranker | None = None,
        fusion: str = "rrf",
        rrf_k: int = 60,
        llm_weight: float = 0.7,
        bm25_weight: float = 0.3,
    ) -> None:
        """初始化混合重排序器。

        Args:
            llm_ranker: LLM 重排序器实例。
            bm25_reranker: BM25 重排序器实例（可选，None 时退化为纯 LLM 排序）。
            fusion: 融合策略。``"rrf"``=倒数排名融合，``"linear"``=线性加权。
            rrf_k: RRF 融合参数，k 越大各路权重越均衡。
            llm_weight: 线性加权模式下 LLM 得分权重。
            bm25_weight: 线性加权模式下 BM25 得分权重。
        """
        valid_fusions = {"rrf", "linear"}
        if fusion not in valid_fusions:
            raise ValueError(f"fusion 必须是 {valid_fusions} 之一，当前值为 {fusion!r}")
        if abs(llm_weight + bm25_weight - 1.0) > 1e-9:
            logger.warning(
                "llm_weight + bm25_weight = %.2f，线性加权模式下已归一化处理",
                llm_weight + bm25_weight,
            )

        self.llm_ranker = llm_ranker
        self.bm25_reranker = bm25_reranker
        self.fusion = fusion
        self.rrf_k = rrf_k
        self.llm_weight = llm_weight
        self.bm25_weight = bm25_weight

        logger.info(
            "HybridReranker initialized: fusion=%s, llm_weight=%.2f, bm25_weight=%.2f",
            fusion, llm_weight, bm25_weight,
        )

    def rerank(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """混合重排序：先各自排序，再融合。

        Args:
            query: 查询文本。
            chunks: 候选文档列表。
            top_k: 返回的 top_k 条，None=使用 LLM ranker 的 top_k。

        Returns:
            重排序后的文档列表，包含 ``hybrid_score`` 字段。
            若两路均失败则返回原文顺序。
        """
        if not chunks:
            return []

        effective_top_k = top_k if top_k is not None else self.llm_ranker.top_k

        # Step 1: LLM 排序（带 llm_score）
        llm_reranked = self.llm_ranker.rerank(query, chunks, top_k=len(chunks))

        # Step 2: BM25 排序（可选，带 bm25_score）
        if self.bm25_reranker is not None:
            bm25_reranked = self.bm25_reranker.rerank(query, chunks, top_k=len(chunks))
        else:
            # 无 BM25 时，按 llm_score 排序后直接返回
            for i, chunk in enumerate(llm_reranked):
                chunk["hybrid_score"] = float(len(llm_reranked) - i)
            result = llm_reranked[:effective_top_k]
            logger.info("Hybrid rerank: BM25 unavailable, returning LLM-only order")
            return result

        # Step 3: 建立 doc id -> 原始 index 映射
        doc_id_map: dict[int, int] = {}
        for i, chunk in enumerate(chunks):
            doc_id_map[i] = i

        # 建立两路的排名映射
        llm_rank: dict[int, int] = {}
        for rank, chunk in enumerate(llm_reranked, 1):
            orig = doc_id_map.get(id(chunk))
            if orig is None:
                # 按 text 匹配找原始索引
                for j, c in enumerate(chunks):
                    if c.get("text") == chunk.get("text"):
                        orig = j
                        break
                else:
                    orig = rank - 1
            llm_rank[orig] = rank

        bm25_rank: dict[int, int] = {}
        for rank, chunk in enumerate(bm25_reranked, 1):
            for j, c in enumerate(chunks):
                if c.get("text") == chunk.get("text"):
                    bm25_rank[j] = rank
                    break
            else:
                bm25_rank[rank - 1] = rank

        # Step 4: 融合
        hybrid_scores: dict[int, float] = {}
        for j in range(len(chunks)):
            if self.fusion == "rrf":
                score = (
                    1.0 / (self.rrf_k + llm_rank.get(j, len(chunks) + 1))
                    + 1.0 / (self.rrf_k + bm25_rank.get(j, len(chunks) + 1))
                )
            else:  # linear
                llm_s = llm_reranked[j]["llm_score"] / 10.0 if j < len(llm_reranked) else 0.0
                bm25_s = bm25_reranked[j]["bm25_score"] if j < len(bm25_reranked) else 0.0
                max_bm25 = max(c["bm25_score"] for c in bm25_reranked) if bm25_reranked else 1.0
                bm25_s_norm = bm25_s / max_bm25 if max_bm25 > 0 else 0.0
                score = self.llm_weight * llm_s + self.bm25_weight * bm25_s_norm

            hybrid_scores[j] = score

        # Step 5: 按 hybrid_score 排序，构建最终结果
        sorted_indices = sorted(hybrid_scores, key=hybrid_scores.get, reverse=True)

        result: list[dict[str, Any]] = []
        for j in sorted_indices[:effective_top_k]:
            chunk = {**chunks[j]}
            chunk["hybrid_score"] = round(hybrid_scores[j], 6)
            # 携带两路分数方便调试
            if j < len(llm_reranked):
                chunk["llm_score"] = llm_reranked[j].get("llm_score")
                if "llm_reason" in llm_reranked[j]:
                    chunk["llm_reason"] = llm_reranked[j]["llm_reason"]
            if j < len(bm25_reranked):
                chunk["bm25_score"] = bm25_reranked[j].get("bm25_score")
            result.append(chunk)

        logger.info(
            "Hybrid reranked %d chunks, fusion=%s, top hybrid_score=%.4f",
            len(chunks), self.fusion,
            hybrid_scores[sorted_indices[0]] if sorted_indices else 0.0,
        )
        return result


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------


def build_ranker(
    llm_client: "DashScopeLLMClient",
    mode: str = "score",
    top_k: int = 5,
    enable_bm25: bool = False,
    fusion: str = "rrf",
) -> LLMRanker | HybridReranker:
    """构建重排序器的工厂函数。

    Args:
        llm_client: LLM 客户端。
        mode: LLM 评分模式。
        top_k: 返回的 top_k 条。
        enable_bm25: 是否启用 BM25 混合排序（False=纯 LLM 排序）。
        fusion: 融合策略（rrf 或 linear）。

    Returns:
        LLMRanker（enable_bm25=False）或 HybridReranker（enable_bm25=True）。
    """
    llm_ranker = LLMRanker(
        llm_client=llm_client,
        mode=mode,
        top_k=top_k,
    )

    if not enable_bm25:
        logger.info("Using LLMRanker (mode=%s, top_k=%d)", mode, top_k)
        return llm_ranker

    bm25_reranker = BM25Reranker()
    hybrid = HybridReranker(
        llm_ranker=llm_ranker,
        bm25_reranker=bm25_reranker,
        fusion=fusion,
    )
    logger.info("Using HybridReranker (fusion=%s, top_k=%d)", fusion, top_k)
    return hybrid
