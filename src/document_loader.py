"""文档加载器：支持 PDF、TXT、Markdown 文件的加载与分片。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

try:
    from langchain_community.document_loaders import PyPDFLoader
except ImportError:  # pragma: no cover
    PyPDFLoader = None  # type: ignore

try:
    import pypdf
except ImportError:  # pragma: no cover
    pypdf = None  # type: ignore


class DocumentLoader:
    """统一的文档加载接口，支持 PDF / TXT / Markdown 三种格式。"""

    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}

    # ────────────────────────────────────────────────────────────────────────
    # 公开 API
    # ────────────────────────────────────────────────────────────────────────

    def load_document(self, file_path: str | Path) -> List[Dict[str, Any]]:
        """
        加载单个文件，返回文档片段列表。

        根据文件扩展名分发到对应加载器（PDF → _load_pdf，文本 → _load_text），
        保证输出格式统一（均含 page_content 和 metadata），便于下游处理。

        Parameters
        ----------
        file_path : str | Path
            待加载文件的路径。

        Returns
        -------
        List[Dict[str, Any]]
            每个片段为一个字典，必须包含键 ``page_content`` 和 ``metadata``。
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")

        suffix = path.suffix.lower()
        if suffix not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"不支持的文件类型: {suffix}，仅支持 {self.SUPPORTED_EXTENSIONS}"
            )

        if suffix == ".pdf":
            return self._load_pdf(path)
        elif suffix in {".txt", ".md"}:
            return self._load_text(path)

        return []  # unreachable

    def load_directory(
        self, directory_path: str | Path, recursive: bool = True
    ) -> List[Dict[str, Any]]:
        """
        遍历目录，加载所有支持的文档文件。

        通过 glob 模式递归/非递归枚举目录下所有文件，逐一调用 load_document，
        将结果合并为单一列表返回；单个文件加载失败不影响其他文件，
        保证批量处理时的鲁棒性。

        Parameters
        ----------
        directory_path : str | Path
            目标目录路径。
        recursive : bool
            是否递归扫描子目录，默认为 True。

        Returns
        -------
        List[Dict[str, Any]]
            所有文档片段的合并列表。
        """
        dir_path = Path(directory_path)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"目录不存在或非目录: {dir_path}")

        pattern = "**/*" if recursive else "*"
        all_docs: List[Dict[str, Any]] = []
        for file_path in sorted(dir_path.glob(pattern)):
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                try:
                    all_docs.extend(self.load_document(file_path))
                except Exception as exc:
                    # 跳过单个文件的加载错误，避免污染整批文档
                    print(f"[DocumentLoader] 加载失败 {file_path}: {exc}")
        return all_docs

    # ────────────────────────────────────────────────────────────────────────
    # 内部实现
    # ────────────────────────────────────────────────────────────────────────

    def _load_pdf(self, path: Path) -> List[Dict[str, Any]]:
        """
        使用 pypdf / langchain 加载 PDF。

        优先使用 langchain_community 的 PyPDFLoader（提取更稳定）；
        不可用时降级为纯 pypdf 实现，按页拆分为独立片段，
        每片段附带页码、文件路径等元数据，便于后续精确溯源。
        """
        docs: List[Dict[str, Any]] = []

        if PyPDFLoader is not None:
            loader = PyPDFLoader(str(path))
            raw_docs = loader.load()
            for doc in raw_docs:
                docs.append(
                    {
                        "page_content": doc.page_content,
                        "metadata": {
                            "source": str(path),
                            "page": doc.metadata.get("page", None),
                            "file_type": "pdf",
                            "doc_name": path.name,
                        },
                    }
                )
            return docs

        if pypdf is not None:
            with open(path, "rb") as f:
                reader = pypdf.PdfReader(f)
                for page_num, page in enumerate(reader.pages, start=1):
                    text = page.extract_text() or ""
                    if text.strip():
                        docs.append(
                            {
                                "page_content": text,
                                "metadata": {
                                    "source": str(path),
                                    "page": page_num,
                                    "file_type": "pdf",
                                    "doc_name": path.name,
                                },
                            }
                        )
            return docs

        raise RuntimeError(
            "加载 PDF 需要 pypdf 或 langchain_community，请运行: pip install pypdf"
        )

    def _load_text(self, path: Path) -> List[Dict[str, Any]]:
        """
        加载 TXT / Markdown 文件，全部内容作为单个片段返回。

        以 UTF-8 读取完整文件内容，file_type 根据后缀判断（.md → "md"，.txt → "txt"），
        metadata 中固定 page=1（文本文件无分页概念），确保输出格式与 PDF 一致。
        """
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        file_type = "md" if path.suffix.lower() == ".md" else "txt"
        return [
            {
                "page_content": content,
                "metadata": {
                    "source": str(path),
                    "page": 1,
                    "file_type": file_type,
                    "doc_name": path.name,
                },
            }
        ]
