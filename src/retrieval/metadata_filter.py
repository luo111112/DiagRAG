"""检索结果元数据过滤器。

在向量检索（和可选的重排序）之后，根据文档块中存储的元数据字段对候选结果进行过滤。
支持两种互补的过滤模式：

- **whitelist（白名单）** — 当且仅当文档块的每个白名单字段都匹配（字段值在允许列表中）时予以保留。
- **blacklist（黑名单）** — 当任意黑名单字段匹配（字段值在排除列表中）时丢弃该文档块。

两种模式可以同时生效：文档块必须通过白名单（若非空）**并且**不被黑名单命中，才能最终保留。
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MetadataFilter
# ---------------------------------------------------------------------------

class MetadataFilter:
    """根据存储的元数据字段对检索到的文档块进行过滤。

    管线定位
    ========
    过滤发生在**检索**和（可选的）**重排序**之后。
    这种设计保持向量搜索 / 重排逻辑完全不变，同时为调用方提供了一种简洁的方式，
    通过文档属性（如来源文件、疾病类型、作者、发表年份等）来约束候选集。

    两种过滤模式可以同时使用：

    whitelist（白名单）
        文档块的每个白名单字段都必须命中对应的允许值列表。
        字段为空或不存在永远不会满足白名单规则。
        白名单字典为空 → 不施加白名单限制。

    blacklist（黑名单）
        一旦任意黑名单字段命中对应的排除值列表，该文档块立即被丢弃。
        字段为空或不存在时跳过黑名单检查。
        黑名单字典为空 → 不施加黑名单限制。

    两种模式可同时生效。文档块只有同时满足以下条件才能保留：
    通过白名单（若白名单非空）**且**不在黑名单中。

    类型强制转换
    ===========
    元数据以 JSON 格式存储，数值型字段可能以 ``int`` 或 ``float`` 而非 ``str`` 的形式传入。
    ``_passes_whitelist`` 和 ``_hits_blacklist`` 均处理了这种情况，
    会在比较前将数值强制转换为字符串。因此配置项
    ``{"year": ["2024"]}`` 可以匹配 ``metadata["year"] = 2024``。

    示例
    -----
    >>> mf = MetadataFilter(
    ...     whitelist={"source_file": ["急性心肌梗死诊疗指南"]},
    ...     blacklist={"doc_type": ["test"]},
    ... )
    >>> filtered = mf.filter(chunks)   # chunks 为包含 "metadata" 键的字典列表

    另请参见
    --------
    build_metadata_filter : 从配置值构造过滤器的工厂函数。
    """

    def __init__(
        self,
        whitelist: dict[str, list[str]] | None = None,
        blacklist: dict[str, list[str]] | None = None,
    ) -> None:
        """使用白名单和/或黑名单规则初始化过滤器。

        Args:
            whitelist: 字段 → 允许值的映射。详见类文档。
            blacklist: 字段 → 排除值的映射。详见类文档。
        """
        self.whitelist = whitelist or {}
        self.blacklist = blacklist or {}

        if self.whitelist:
            logger.info(
                "MetadataFilter 白名单已激活: %d 个字段 — %s",
                len(self.whitelist),
                {k: len(v) for k, v in self.whitelist.items()},
            )
        if self.blacklist:
            logger.info(
                "MetadataFilter 黑名单已激活: %d 个字段 — %s",
                len(self.blacklist),
                {k: len(v) for k, v in self.blacklist.items()},
            )

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def filter(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """对文档块列表应用白名单和黑名单过滤。

        Args:
            chunks: 检索结果列表，每个元素应包含 ``metadata`` 键（值为 dict 或 None）。

        Returns:
            过滤后的文档块列表，保留被保留下来的块的原始顺序。
        """
        if not chunks:
            return []

        # 当未配置任何过滤规则时，直接返回原列表
        if not self.whitelist and not self.blacklist:
            return list(chunks)

        kept, dropped = [], []
        for chunk in chunks:
            meta = chunk.get("metadata") or {}
            if self._passes_whitelist(meta) and not self._hits_blacklist(meta):
                kept.append(chunk)
            else:
                dropped.append(chunk)

        if dropped:
            logger.debug(
                "MetadataFilter: %d/%d 个文档块被过滤（白名单=%s, 黑名单=%s）",
                len(dropped),
                len(chunks),
                bool(self.whitelist),
                bool(self.blacklist),
            )
        return kept

    # ------------------------------------------------------------------
    # 内部辅助方法
    # ------------------------------------------------------------------

    def _passes_whitelist(self, meta: dict[str, Any]) -> bool:
        """当元数据满足所有白名单规则时返回 True（每个字段内部为 OR 逻辑）。"""
        if not self.whitelist:
            return True

        for field, allowed in self.whitelist.items():
            actual = meta.get(field)
            # 字段为空或不存在：白名单永远不会匹配
            if actual is None:
                return False
            # 类型强制转换：允许 int/float 与 str 形式的配置值匹配
            if isinstance(actual, (int, float)):
                str_allowed = [str(v) for v in allowed]
                if str(actual) not in str_allowed and str(actual) not in allowed:
                    return False
            elif actual not in allowed:
                return False
        return True

    def _hits_blacklist(self, meta: dict[str, Any]) -> bool:
        """当元数据命中任意黑名单规则时返回 True（每个字段内部为 OR 逻辑）。"""
        if not self.blacklist:
            return False

        for field, excluded in self.blacklist.items():
            actual = meta.get(field)
            if actual is None:
                continue
            if isinstance(actual, (int, float)):
                str_excluded = [str(v) for v in excluded]
                if str(actual) in str_excluded or actual in excluded:
                    return True
            elif actual in excluded:
                return True
        return False


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------

def build_metadata_filter(
    whitelist: dict[str, list[str]] | None = None,
    blacklist: dict[str, list[str]] | None = None,
) -> MetadataFilter:
    """根据给定参数构建并返回 ``MetadataFilter`` 实例。

    示例
    -----
    >>> from src.retrieval.metadata_filter import build_metadata_filter
    >>> mf = build_metadata_filter(
    ...     whitelist={"source_file": ["急性心肌梗死诊疗指南"]},
    ...     blacklist={"disease_type": ["测试数据"]},
    ... )
    >>> filtered = mf.filter(chunks)
    """
    return MetadataFilter(whitelist=whitelist, blacklist=blacklist)
