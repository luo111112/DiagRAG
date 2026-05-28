"""文本分割器：将文档切分为固定大小的 chunk，支持 LangChain 回退。"""

from __future__ import annotations

from typing import Any, Dict, List

# LangChain 依赖声明
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # pragma: no cover
    RecursiveCharacterTextSplitter = None  # type: ignore


class TextSplitter:
    """统一的文本分割接口，内部优先使用 LangChain，否则降级为纯 Python 实现。"""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        min_chunk_length: int = 20,
        separators: List[str] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        chunk_size : int
            每个 chunk 的最大字符数（默认 500）。
        chunk_overlap : int
            相邻 chunk 之间的重叠字符数（默认 50）。
        min_chunk_length : int
            丢弃长度小于此值的 chunk（默认 20）。
        separators : List[str] | None
            递归分割的优先级字符列表，默认为 ["\n\n", "\n", "。", "；", "，", " ", ""]。
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_length = min_chunk_length
        self._separators = separators or ["\n\n", "\n", "。", "；", "，", " ", ""]

        # 优先使用 LangChain（更智能的语义切分）；不可用时降级为纯 Python 实现。
        # chunk_size / chunk_overlap 控制切分粒度，separators 定义递归切分优先级。
        self._lc_splitter: RecursiveCharacterTextSplitter | None = None
        if RecursiveCharacterTextSplitter is not None:
            self._lc_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=self._separators,
                length_function=len,
            )

    # ────────────────────────────────────────────────────────────────────────
    # 公开 API
    # ────────────────────────────────────────────────────────────────────────

    def split_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        分割文档列表，返回 chunk 列表。

        对每个文档按优先级 separators 递归切分，超过 chunk_size 的段落自动降级；
        丢弃长度小于 min_chunk_length 的过短片段，避免无效 embedding；
        全局 chunk_id 保证跨文档唯一，chunk_index 记录同文档内序号，
        为后续向量检索和溯源提供依据。

        Parameters
        ----------
        documents : List[Dict[str, Any]]
            输入文档列表，格式与 ``DocumentLoader`` 输出一致，每个字典必须包含：
            - ``page_content`` (str)：文档文本内容
            - ``metadata`` (dict)：文档元数据

        Returns
        -------
        List[Dict[str, Any]]
            chunk 列表，每个 chunk 包含：
            - ``text`` (str)：分割后的文本片段
            - ``metadata`` (dict)：原始 metadata 加上 ``chunk_id``（全局递增，从 0 开始）
              和 ``chunk_index``（当前文档内的序号，从 0 开始）
            - 长度小于 ``min_chunk_length`` 的 chunk 会被丢弃
        """
        chunks: List[Dict[str, Any]] = []
        global_chunk_id = 0

        for doc_index, doc in enumerate(documents):
            page_content: str = doc.get("page_content", "")
            metadata: Dict[str, Any] = dict(doc.get("metadata", {}))

            if self._lc_splitter is not None:
                raw_chunks = self._lc_splitter.split_text(page_content)
            else:
                raw_chunks = self._python_split(page_content)

            for local_index, text in enumerate(raw_chunks):
                if len(text) < self.min_chunk_length:
                    continue
                chunks.append(
                    {
                        "text": text,
                        "metadata": {
                            **metadata,
                            "chunk_id": global_chunk_id,
                            "chunk_index": local_index,
                        },
                    }
                )
                global_chunk_id += 1

        return chunks

    # ────────────────────────────────────────────────────────────────────────
    # 内部实现（LangChain 不可用时的降级方案）
    # ────────────────────────────────────────────────────────────────────────

    def _python_split(self, text: str) -> List[str]:
        """
        纯 Python 实现的递归文本分割入口。

        当 LangChain 不可用时作为唯一切分引擎使用。
        按 self._separators 优先级从高到低尝试切分，
        若某级无法产生有效结果则递归降级至下一级。
        """
        return self._split_recursive(text, 0)

    def _split_recursive(self, text: str, sep_index: int) -> List[str]:
        """
        递归分割的核心逻辑。

        用当前优先级（sep_index）的分隔符尝试切分文本：buffer 累积不足 chunk_size 的段落，
        超过时固化 buffer 并对超长段落递归降级至更低优先级分隔符。
        若最终仅产生 1 个 chunk，说明该分隔符无法有效切分，同样降级，
        直至降至空字符串（固定大小硬切），确保任意文本都能被处理。
        """
        separator = self._separators[sep_index]
        if not text:
            return []

        # 末级分隔符（空字符串）：按固定大小硬切
        if separator == "":
            return self._fixed_split(text)

        parts = text.split(separator)
        result: List[str] = []
        buffer = ""

        for part in parts:
            # 先把 buffer 与当前 part 拼接
            candidate = (buffer + separator + part) if buffer else part

            if len(candidate) <= self.chunk_size:
                buffer = candidate
            else:
                if buffer:
                    result.append(buffer)
                    buffer = ""
                # 若单段本身已超上限，递归降级
                if len(part) > self.chunk_size:
                    sub_parts = self._split_recursive(part, sep_index + 1)
                    result.extend(sub_parts[:-1] if sub_parts else [])
                    buffer = sub_parts[-1] if sub_parts else ""
                else:
                    buffer = part

        if buffer:
            result.append(buffer)

        # 如果产生的 chunk 数过少（整体未有效分割），降级到下一级
        if len(result) <= 1 and sep_index < len(self._separators) - 1:
            return self._split_recursive(text, sep_index + 1)

        return result

    def _fixed_split(self, text: str) -> List[str]:
        """
        固定大小硬切——递归分割的最后兜底方案。

        按 step = chunk_size - chunk_overlap 步长滑动窗口截取文本，
        每个窗口长度为 chunk_size，窗口之间重叠 chunk_overlap 个字符，
        保证语义边界丢失时仍能均匀切分且相邻 chunk 有上下文衔接。
        """
        chunks: List[str] = []
        start = 0
        total = len(text)
        step = self.chunk_size - self.chunk_overlap

        if step <= 0:
            step = self.chunk_size

        while start < total:
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += step

        return chunks
