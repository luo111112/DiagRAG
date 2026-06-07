"""上下文指纹（Context Fingerprint）计算模块。

用于语义缓存的多轮场景：
  - 将会话摘要转化为 Jaccard 关键词指纹，在 Milvus 查询时
    过滤掉上下文关联性过低的缓存条目，降低"答非所问"风险。
  - 同时导出 SHA256 哈希值作为精确匹配键（Redis 一级缓存）。

算法：
  1. 分句 + 去停用词 + 统一小写。
  2. 关键词集合 = {term | term not in STOP_WORDS}。
  3. Jaccard(现有指纹, 缓存指纹) = |交集| / |并集|。
  4. 哈希 = SHA256(归一化文本) 前 8 位。

依赖：仅标准库，无外部依赖。
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Any

# ---------------------------------------------------------------------------
# 停用词表（可按业务领域扩展）
# ---------------------------------------------------------------------------

_STOP_WORDS: set[str] = {
    # 通用中文停用词（精选高频功能词，保留诊断相关实词）
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
    "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有",
    "看", "好", "自己", "这", "那", "他", "她", "它", "们", "这个", "那个",
    "什么", "怎么", "为什么", "哪", "哪里", "哪些", "谁", "多少", "几", "怎样",
    "可以", "可能", "能", "应该", "需要", "必须", "把", "被", "让", "给",
    "但是", "而", "所以", "因为", "如果", "虽然", "还是", "或者", "而且",
    "然后", "接着", "最后", "首先", "其次", "再次", "另外", "总之",
    "很", "非常", "特别", "比较", "相当", "极其", "更", "最", "太", "真",
    "只", "仅仅", "才", "已", "已经", "曾", "曾经", "正在", "将", "将要",
    "还", "再", "又", "也", "都", "并", "且", "以及", "之", "等", "等等",
    "一下", "一点", "一些", "那个", "这样", "那样", "这么", "那么",
    "请问", "问一下", "我想", "我想问", "帮我", "帮我查", "请问一下",
    "请问", "请问您", "打扰", "麻烦", "谢谢", "好的", "嗯", "啊",
    "请", "稍", "等一下", "稍等", "现在", "目前", "当前", "今天", "明天",
    "昨天", "前", "后", "前天", "后天", "以内", "之前", "之后",
    # 英文停用词（常见）
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "dare",
    "and", "or", "but", "if", "else", "because", "as", "until", "while",
    "of", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below",
    "to", "from", "up", "down", "in", "out", "on", "off", "over", "under",
    "again", "further", "then", "once", "here", "there", "when", "where",
    "why", "how", "all", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than",
    "too", "very", "s", "t", "just", "don", "now",
}

# 标点符号正则（用于分句）
_SENTENCE_SPLITTER = re.compile(r"[。！？；\n]+")
# 非字母/数字/汉字字符（用于 token 清洗）
_TOKEN_CLEANER = re.compile(r"[^\w\u4e00-\u9fff]+")


# ---------------------------------------------------------------------------
# 公开 API
# ---------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    """对输入文本进行统一归一化。

    处理步骤：
      1. Unicode NFKC 规范化（全角转半角等）。
      2. 小写化。
      3. 去除多余空白。
      4. 去除首尾空白。
    """
    if not text:
        return ""
    # NFKC 规范化（全角字母/数字转半角）
    normalized = unicodedata.normalize("NFKC", text)
    # 小写
    normalized = normalized.lower()
    # 合并内部多余空白
    normalized = " ".join(normalized.split())
    return normalized.strip()


def extract_keywords(text: str) -> set[str]:
    """从文本中提取关键词集合（去停用词）。

    处理步骤：
      1. normalize_text 归一化。
      2. _SENTENCE_SPLITTER 分句。
      3. _TOKEN_CLEANER 将每句切分为 token。
      4. 过滤停用词表。
      5. 过滤单字符（保留中文词和英文词）。

    Args:
        text: 原始文本。

    Returns:
        关键词集合（去重）。
    """
    if not text:
        return set()

    text = normalize_text(text)
    keywords: set[str] = set()

    # 分句
    sentences = _SENTENCE_SPLITTER.split(text)
    for sentence in sentences:
        # 清洗并切分
        cleaned = _TOKEN_CLEANER.sub(" ", sentence)
        tokens = cleaned.split()
        for token in tokens:
            token = token.strip()
            if not token:
                continue
            # 过滤：停用词 / 单字符（英文）/ 太短的（长度==1）
            if token in _STOP_WORDS:
                continue
            if len(token) == 1:
                continue
            keywords.add(token)

    return keywords


def compute_summary_fingerprint(summary_text: str) -> dict[str, Any]:
    """计算摘要的完整指纹信息。

    返回值包含：
      - `keywords`: 关键词集合（list，用于 Jaccard 比对）。
      - `hash8`: SHA256 前8位十六进制字符串（用于 Redis 精确匹配）。
      - `keyword_count`: 关键词数量。

    Args:
        summary_text: 会话摘要原文。

    Returns:
        包含 fingerprint 信息的字典。
    """
    if not summary_text:
        return {
            "keywords": [],
            "hash8": "",
            "keyword_count": 0,
        }

    normalized = normalize_text(summary_text)
    keywords = extract_keywords(summary_text)

    # SHA256 前8位（16个十六进制字符）
    sha256_hex = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    hash8 = sha256_hex[:8]

    return {
        "keywords": sorted(keywords),          # list 便于 JSON 序列化
        "hash8": hash8,
        "keyword_count": len(keywords),
    }


def jaccard_similarity(keywords_a: list[str], keywords_b: list[str]) -> float:
    """计算两个关键词集合的 Jaccard 相似度。

    Jaccard(A, B) = |A ∩ B| / |A ∪ B|
    空集合约定返回 0.0。

    Args:
        keywords_a: 关键词列表 A（去重后）。
        keywords_b: 关键词列表 B（去重后）。

    Returns:
        0.0 ~ 1.0 的浮点数相似度。
    """
    set_a = set(keywords_a)
    set_b = set(keywords_b)

    if not set_a or not set_b:
        return 0.0

    intersection = len(set_a & set_b)
    union = len(set_a | set_b)

    if union == 0:
        return 0.0

    return intersection / union


def check_context_match(
    current_summary_text: str,
    cached_summary_text: str,
    threshold: float = 0.50,
) -> tuple[bool, float]:
    """检查当前会话上下文与缓存记录的上下文是否足够相似。

    Args:
        current_summary_text: 当前会话的摘要文本。
        cached_summary_text: 缓存记录中的摘要文本。
        threshold: Jaccard 相似度阈值，默认 0.50。

    Returns:
        (is_match, similarity_score)。
          - is_match = True  表示上下文匹配，缓存条目可复用。
          - is_match = False 表示上下文不匹配，缓存条目不适用。
    """
    current_fingerprint = compute_summary_fingerprint(current_summary_text)
    cached_fingerprint = compute_summary_fingerprint(cached_summary_text)

    similarity = jaccard_similarity(
        current_fingerprint["keywords"],
        cached_fingerprint["keywords"],
    )

    return similarity >= threshold, round(similarity, 4)


def build_cache_key(
    question: str,
    session_id: str | None = None,
    user_id: str | None = None,
    summary_hash: str = "",
    scope: str = "session",
) -> str:
    """构建语义缓存键。

    格式规则（scope 控制包含哪些字段）：
      - session : "semcache:{scope}:{question_sha256[:16]}:{session_id}:{summary_hash}"
      - user    : "semcache:{scope}:{question_sha256[:16]}:{user_id}"
      - question: "semcache:{scope}:{question_sha256[:16]}"

    Args:
        question: 归一化后的问题文本。
        session_id: 会话 ID（session scope 时必须提供）。
        user_id: 用户 ID（user scope 时提供）。
        summary_hash: 会话摘要 SHA256 前8位。
        scope: cache_key_scope，取值 session | user | question。

    Returns:
        缓存键字符串。
    """
    q_sha = hashlib.sha256(normalize_text(question).encode("utf-8")).hexdigest()[:16]

    if scope == "session":
        if not session_id:
            raise ValueError("session scope requires session_id")
        sid = session_id.replace(":", "_")
        sh = summary_hash.replace(":", "_")
        return f"semcache:session:{q_sha}:{sid}:{sh}"
    elif scope == "user":
        if not user_id:
            raise ValueError("user scope requires user_id")
        uid = user_id.replace(":", "_")
        return f"semcache:user:{q_sha}:{uid}"
    else:
        return f"semcache:question:{q_sha}"


# ---------------------------------------------------------------------------
# 内部工具（导出用于单测）
# ---------------------------------------------------------------------------

__all__ = [
    "normalize_text",
    "extract_keywords",
    "compute_summary_fingerprint",
    "jaccard_similarity",
    "check_context_match",
    "build_cache_key",
    "_STOP_WORDS",
]
