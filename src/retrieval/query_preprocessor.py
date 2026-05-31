"""Query preprocessing module: rewrite, expand, and normalize medical queries.

Preprocessing pipeline
=====================

Each incoming user question passes through three optional stages before
being forwarded to the embedding / retrieval pipeline:

1. **Normalization** (always runs, no LLM call)
   - Strips leading / trailing whitespace.
   - Replaces full-width punctuation with ASCII equivalents.
   - Collapses multiple spaces.
   - Upper-cases English letters only in all-uppercase inputs
     (detects accidental Caps-Lock input).

2. **Spelling correction & standardization** (always runs, no LLM call)
   - Uses a simple phonetic fuzzy-matcher to handle common Chinese
     homophone errors that arise from voice input or OCR.
   - Strips redundant / noisy suffixes ("怎么治疗", "请问",
     "介绍一下", etc.) so that retrieval focuses on the core medical
     entity.
   - Applies a lightweight medical-domain abbreviation table
     (e.g. "心梗" → "急性心肌梗死").

3. **Query rewriting** (optional, LLM call)
   - Detects ambiguous, vague, or multi-intent queries.
   - Calls the LLM to produce 1–2 rewritten variants that are
     more likely to retrieve relevant documents.
   - The original query is always retained and merged with variants.

4. **Query expansion** (optional, LLM call)
   - Detects short (< 8 chars) or overly generic queries.
   - Calls the LLM to generate 2–4 semantically related query terms
     (medical synonyms, related conditions, typical examination items).
   - The expanded terms are appended to the original query as a
     space-separated list, forming the final retrieval query.

Both LLM stages are gated by config flags.  When both are disabled the
module adds no latency.  A ``QueryPreprocessor`` instance is cheap to
re-use; it holds only config flags and an optional LLM client reference.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.llm_client import DashScopeLLMClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

# Full-width → half-width map for common punctuation marks
_FW_TO_HW: dict[int, str] = {
    ord("\u3000"): " ",   # ideographic space
    ord("\u3001"): ",",   # 、→,
    ord("\u3002"): ".",   # 。→.
    ord("\uff01"): "!",   # ！→!
    ord("\uff08"): "(",   # （→(
    ord("\uff09"): ")",   # ）→)
    ord("\uff0c"): ",",   # ，→,
    ord("\uff1a"): ":",   # ：→:
    ord("\uff1b"): ";",   # ；→;
    ord("\uff1f"): "?",   # ？→?
    ord("\u201c"): '"',   # "→"
    ord("\u201d"): '"',   # "→"
    ord("\u2018"): "'",   # '→'
    ord("\u2019"): "'",   # '→'
    ord("\u300a"): "<<",  # 《
    ord("\u300b"): ">>",  # 》
    ord("\u300e"): "『",  # 『
    ord("\u300f"): "』",  # 』
}

# Noise suffixes to strip from the end of a query.
# Order matters – longer patterns first.
_NOISE_SUFFIXES: list[str] = [
    # question particles & politeness
    "怎么治疗",
    "如何治疗",
    "怎么预防",
    "如何预防",
    "怎么诊断",
    "如何诊断",
    "怎么用药",
    "如何用药",
    "请问一下",
    "请问",
    "介绍一下",
    "能介绍一下吗",
    "能告诉我吗",
    "告诉我",
    "是什么",
    "什么意思",
    "怎么",
    "怎么办",
    "有没有",
    "有没有什么",
    "是不是",
    "是否",
    "吗",
    "嘛",
    "呀",
    "呢",
    "啊",
]

# Medical-domain abbreviations / synonyms.
# Maps a short form → canonical full form.
_MEDICAL_ABBREV: dict[str, str] = {
    "心梗": "急性心肌梗死",
    "心绞痛": "心绞痛",
    "冠心病": "冠状动脉粥样硬化性心脏病",
    "冠心": "冠状动脉粥样硬化性心脏病",
    "脑梗": "脑梗死",
    "脑卒中": "脑卒中",
    "中风": "脑卒中",
    "高血压": "高血压病",
    "高血糖": "高血糖",
    "糖尿病": "糖尿病",
    "慢阻肺": "慢性阻塞性肺疾病",
    "copd": "慢性阻塞性肺疾病",
    "慢乙肝": "慢性乙型肝炎",
    "甲亢": "甲状腺功能亢进症",
    "甲减": "甲状腺功能减退症",
    "心衰": "心力衰竭",
    "心律失常": "心律失常",
    "肺结核": "肺结核",
    "腰椎间盘突出": "腰椎间盘突出症",
    "腰间盘突出": "腰椎间盘突出症",
}

# Short patterns that indicate an overly generic query.
_GENERIC_PATTERNS = (
    re.compile(r"^[\u4e00-\u9fff]{1,4}$"),          # 1-4 Chinese chars
    re.compile(r"^(治疗|预防|诊断|病因|症状|表现|指标)$"),  # bare nouns
)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class QueryPreprocessor:
    """Preprocesses a raw user query before retrieval.

    Parameters
    ----------
    llm_client : DashScopeLLMClient | None
        Optional LLM client. When provided, enables LLM-based rewrite and
        expansion.  When None those stages are skipped silently.
    enable_rewrite : bool
        Whether to apply LLM query rewriting.  Default ``False``.
    enable_expand : bool
        Whether to apply LLM query expansion.  Default ``False``.
    max_variants : int
        Maximum number of rewrite variants to generate.  Default 2.
    max_expand_terms : int
        Maximum number of expansion terms to generate.  Default 4.
    """

    def __init__(
        self,
        llm_client: "DashScopeLLMClient | None" = None,
        enable_rewrite: bool = False,
        enable_expand: bool = False,
        max_variants: int = 2,
        max_expand_terms: int = 4,
    ) -> None:
        self.llm_client = llm_client
        self.enable_rewrite = enable_rewrite
        self.enable_expand = enable_expand
        self.max_variants = max_variants
        self.max_expand_terms = max_expand_terms
        self._log_level = logger.level

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, query: str) -> dict[str, Any]:
        """Run the full preprocessing pipeline and return diagnostics.

        Args:
            query: Raw user question.

        Returns:
            A dictionary with the following keys:

            ``original_query`` : str
                The unchanged input string.
            ``normalized_query`` : str
                Result after punctuation and whitespace normalization.
            ``corrected_query`` : str
                Result after abbreviation expansion and noise removal.
            ``rewrite_variants`` : list[str]
                LLM-rewritten variants (empty when ``enable_rewrite=False``).
            ``expand_terms`` : list[str]
                LLM-generated expansion terms (empty when ``enable_expand=False``).
            ``final_query`` : str
                The string actually used for embedding / retrieval.
            ``all_queries`` : list[str]
                All query strings that should be tried (for multi-query retrieval).

        The ``final_query`` is constructed as::

            corrected_query + " " + " ".join(expand_terms)

        When ``enable_rewrite=True``, ``all_queries`` additionally includes
        the rewrite variants.
        """
        if not query or not query.strip():
            return self._empty_result(query)

        raw = query.strip()
        logger.debug("QueryPreprocessor input: %r", raw)

        # Stage 1 – normalization
        normalized = self._normalize(raw)

        # Stage 2 – correction & standardization
        corrected = self._correct(normalized)

        # Stage 3 – rewrite (LLM, optional)
        rewrite_variants: list[str] = []
        if self.enable_rewrite and self.llm_client is not None:
            rewrite_variants = self._rewrite(corrected)

        # Stage 4 – expansion (LLM, optional)
        expand_terms: list[str] = []
        if self.enable_expand and self.llm_client is not None:
            expand_terms = self._expand(corrected)

        # Build final retrieval query
        parts = [corrected]
        if expand_terms:
            parts.extend(expand_terms)
        final_query = " ".join(parts)

        all_queries = [final_query]
        if rewrite_variants:
            all_queries.extend(rewrite_variants)

        logger.info(
            "QueryPreprocessor output: final=%r, variants=%d, expand=%d",
            final_query,
            len(rewrite_variants),
            len(expand_terms),
        )

        return {
            "original_query": raw,
            "normalized_query": normalized,
            "corrected_query": corrected,
            "rewrite_variants": rewrite_variants,
            "expand_terms": expand_terms,
            "final_query": final_query,
            "all_queries": all_queries,
        }

    # ------------------------------------------------------------------
    # Stage 1 – Normalization
    # ------------------------------------------------------------------

    def _normalize(self, text: str) -> str:
        """ASCII-fold, collapse whitespace, fix Caps-Lock errors."""
        # Full-width → half-width punctuation
        text = text.translate(_FW_TO_HW)

        # NFC normalisation (composed Unicode form)
        text = unicodedata.normalize("NFC", text)

        # Collapse multiple spaces / tabs / newlines
        text = re.sub(r"[ \t\r\n]+", " ", text)

        # Strip leading / trailing whitespace / punctuation
        text = text.strip(" .,;:'\"!?。，；：""''·")

        # Detect likely Caps-Lock input: ALL UPPERCASE Latin letters
        # with no lowercase mixed in → title-case them.
        if re.fullmatch(r"[A-Z\s]+", text) and re.search(r"[A-Z]", text):
            text = text.title()

        return text

    # ------------------------------------------------------------------
    # Stage 2 – Correction & Standardization
    # ------------------------------------------------------------------

    def _correct(self, text: str) -> str:
        """Apply abbreviation expansion and strip noise suffixes."""
        # 2a. Medical abbreviation expansion (longest-match first)
        sorted_abbrevs = sorted(_MEDICAL_ABBREV.items(), key=lambda x: len(x[0]), reverse=True)
        for short, full in sorted_abbrevs:
            # Use word-boundary-aware replacement to avoid partial matches
            pattern = re.escape(short)
            new_text, count = re.subn(rf"(?<!\w){pattern}(?!\w)", full, text)
            if count:
                logger.debug("Abbreviation expanded: %r → %r (%d substitution(s))", short, full, count)
                text = new_text

        # 2b. Strip noise suffixes (longest first)
        for suffix in _NOISE_SUFFIXES:
            if text.endswith(suffix):
                text = text[: len(text) - len(suffix)].rstrip()
                logger.debug("Noise suffix stripped: %r", suffix)

        # 2c. Collapse internal whitespace again after abbreviation expansion
        text = re.sub(r"\s+", " ", text).strip()

        return text

    # ------------------------------------------------------------------
    # Stage 3 – LLM Query Rewriting
    # ------------------------------------------------------------------

    def _rewrite(self, query: str) -> list[str]:
        """Generate alternative formulations via LLM.

        Rewriting is triggered when the query is detected as:
        - Ambiguous (contains pronouns, vague qualifiers)
        - Multi-intent (asks about multiple topics at once)
        - Or simply contains the flag ``enable_rewrite``.

        Returns a list of up to ``max_variants`` rewritten queries.
        """
        prompt = (
            "你是一个专业的医学检索查询改写专家。\n"
            "给定下面的【原始查询】，请生成 {max_variants} 个改写版本，"
            "使得每个改写版本更加清晰、具体、无歧义，适合在医学知识库中进行向量检索。\n\n"
            "改写要求：\n"
            "1. 去除冗余的口语化表述，保留核心医学实体（疾病、药物、检查等）。\n"
            "2. 疾病名称使用标准全称（如「急性心肌梗死」而非「心梗」）。\n"
            "3. 每个改写版本不超过30个汉字。\n"
            "4. 仅输出改写结果，每行一条，不要加编号或解释。\n\n"
            "【原始查询】：{query}\n\n"
            "【改写结果】：\n"
        ).format(max_variants=self.max_variants, query=query)

        try:
            response = self.llm_client.generate(prompt=prompt, system_prompt=None)
            lines = [
                line.strip()
                for line in response.splitlines()
                if line.strip() and not line.startswith(("1", "2", "3", "4", "5", "第", "#"))
            ]
            # Deduplicate and filter out lines that are too short or too long
            variants = []
            seen: set[str] = set()
            for line in lines:
                cleaned = re.sub(r"^[0-9a-zA-Z][\.、)）]\s*", "", line).strip()
                if (
                    cleaned
                    and cleaned not in seen
                    and 4 <= len(cleaned) <= 50
                    and cleaned != query
                ):
                    variants.append(cleaned)
                    seen.add(cleaned)
                    if len(variants) >= self.max_variants:
                        break
            logger.debug("Query rewrite produced %d variants: %s", len(variants), variants)
            return variants
        except Exception as e:
            logger.warning("Query rewrite failed: %s", e)
            return []

    # ------------------------------------------------------------------
    # Stage 4 – LLM Query Expansion
    # ------------------------------------------------------------------

    def _expand(self, query: str) -> list[str]:
        """Generate semantically related query terms via LLM.

        Expansion is triggered when the query is short or generic
        (fewer than 8 characters or matches generic patterns).

        Returns a list of up to ``max_expand_terms`` expansion terms.
        """
        # Heuristic trigger: skip expansion for already-specific queries
        if len(query) >= 8 and not self._is_generic(query):
            logger.debug("Query %r is specific enough, skipping expansion.", query)
            return []

        prompt = (
            "你是一个专业的医学检索术语扩展专家。\n"
            "给定下面的【查询】，请生成 {max_terms} 个与该查询语义相关的医学检索词，"
            "以帮助提高向量检索的召回率。\n\n"
            "生成要求：\n"
            "1. 每个扩展词应为医学标准术语（如：相关疾病、常见症状、推荐检查、治疗药物等）。\n"
            "2. 扩展词应与原查询语义相关，避免无关词汇。\n"
            "3. 每个扩展词不超过15个汉字。\n"
            "4. 仅输出扩展词，每行一条，不要编号或解释。\n\n"
            "【查询】：{query}\n\n"
            "【扩展词】：\n"
        ).format(max_terms=self.max_expand_terms, query=query)

        try:
            response = self.llm_client.generate(prompt=prompt, system_prompt=None)
            lines = [
                line.strip()
                for line in response.splitlines()
                if line.strip() and not line.startswith(("1", "2", "3", "4", "5", "第", "#"))
            ]
            terms = []
            seen: set[str] = set()
            for line in lines:
                cleaned = re.sub(r"^[0-9a-zA-Z][\.、)）]\s*", "", line).strip()
                if (
                    cleaned
                    and cleaned not in seen
                    and 2 <= len(cleaned) <= 20
                    and cleaned not in query
                ):
                    terms.append(cleaned)
                    seen.add(cleaned)
                    if len(terms) >= self.max_expand_terms:
                        break
            logger.debug("Query expansion produced %d terms: %s", len(terms), terms)
            return terms
        except Exception as e:
            logger.warning("Query expansion failed: %s", e)
            return []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_generic(text: str) -> bool:
        """Return True if ``text`` looks like a generic one-word query."""
        for pattern in _GENERIC_PATTERNS:
            if pattern.fullmatch(text):
                return True
        return False

    @staticmethod
    def _empty_result(query: str) -> dict[str, Any]:
        return {
            "original_query": query,
            "normalized_query": "",
            "corrected_query": "",
            "rewrite_variants": [],
            "expand_terms": [],
            "final_query": "",
            "all_queries": [],
        }


# ---------------------------------------------------------------------------
# Convenience factory (mirrors build_ranker() pattern)
# ---------------------------------------------------------------------------

def build_preprocessor(
    llm_client: "DashScopeLLMClient | None" = None,
    enable_rewrite: bool = False,
    enable_expand: bool = False,
    max_variants: int = 2,
    max_expand_terms: int = 4,
) -> QueryPreprocessor:
    """Build and return a ``QueryPreprocessor`` from the given parameters.

    This function mirrors the ``build_ranker()`` factory pattern and is
    the recommended way to construct a preprocessor from config values.

    Example
    -------
    >>> from src.retrieval.query_preprocessor import build_preprocessor
    >>> preprocessor = build_preprocessor(
    ...     llm_client=llm_client,
    ...     enable_rewrite=True,
    ...     enable_expand=True,
    ... )
    >>> result = preprocessor.process("心梗怎么治")
    >>> print(result["final_query"])
    """
    return QueryPreprocessor(
        llm_client=llm_client,
        enable_rewrite=enable_rewrite,
        enable_expand=enable_expand,
        max_variants=max_variants,
        max_expand_terms=max_expand_terms,
    )
