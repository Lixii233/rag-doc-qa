"""文档解析：把 PDF / Word / 文本文件读成带页码的纯文本。

对每种格式都做了降级处理：
- PDF 优先用 PyMuPDF（`fitz`）提取，效果对中文排版更好；失败则退回 pypdf。
- Word 用 python-docx，除正文段落外还会读取表格单元格。
- 文本类文件做编码嗅探，兼容 UTF-8 / GBK（Windows 下常见）。
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from .schemas import Doc, Page

SUPPORTED_SUFFIXES = {".pdf", ".docx", ".txt", ".md", ".markdown", ".csv"}

_ZERO_WIDTH = re.compile(r"[\u200b-\u200f\u202a-\u202e\ufeff]")


class UnsupportedFileError(ValueError):
    """文件类型不在支持范围内。"""


class EmptyDocumentError(ValueError):
    """文件解析后没有任何文本（例如扫描版 PDF）。"""


# --------------------------------------------------------------------------- 文本清洗
def normalize_text(text: str) -> str:
    """统一换行、去掉零宽字符、压缩多余空行。"""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _ZERO_WIDTH.sub("", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _read_plain(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "big5", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


_SENTENCE_END = tuple("。！？!?；;：:")
_LIST_START = re.compile(r"^\s*(?:\d+[.、)]|[（(]\d+[)）]|[一二三四五六七八九十]+、|第[一二三四五六七八九十]+[章节条])")


def _looks_like_heading(line: str) -> bool:
    """短、无标点、以“一、/1./第X章”开头 —— PDF 里的小标题。"""
    stripped = line.strip()
    if not stripped or len(stripped) > 24:
        return False
    return bool(_LIST_START.match(stripped)) or not re.search(r"[，。；：、,.;:]", stripped)


def reflow_lines(text: str) -> str:
    """把 PDF 里被版式硬折的行接回句子。

    PDF 抽出来的文本是“视觉行”而不是“段落”：一句话会被切成好几行，
    直接切片就会出现「期末考试及」「相关安排的通知」这种半截句子。
    这里按“上一行没写完、下一行不是新条目”把行接起来，中英文分别处理连接符。
    """
    lines = [line.rstrip() for line in text.split("\n")]
    merged: list[str] = []
    for raw in lines:
        line = raw.strip()
        if not line:
            if merged and merged[-1] != "":
                merged.append("")
            continue
        if not merged or merged[-1] == "":
            merged.append(line)
            continue

        prev = merged[-1]
        unfinished = prev.rstrip()[-1] not in _SENTENCE_END
        starts_new = bool(_LIST_START.match(line))
        salutation = line.endswith("：") or line.endswith(":")
        heading_before = _looks_like_heading(prev)
        # 首行如果是标题，可能被版式折成两三行：标题行通常不带句末标点，可以放心接上
        title_continuation = (
            len(merged) == 1
            and not re.search(r"[。！？!?]", prev)
            and not re.search(r"[。！？!?]", line)
        )

        if unfinished and not starts_new and not salutation and (not heading_before or title_continuation):
            joiner = " " if (prev[-1].isascii() and prev[-1].isalnum() and line[0].isascii() and line[0].isalnum()) else ""
            merged[-1] = prev + joiner + line
        else:
            merged.append(line)

    return "\n".join(merged).strip()


# --------------------------------------------------------------------------- 各格式解析
def _load_pdf(path: Path) -> list[Page]:
    pages: list[Page] = []
    try:
        try:
            import pymupdf as fitz  # PyMuPDF >= 1.24 的新包名
        except ImportError:  # 老版本只有 fitz
            import fitz

        with fitz.open(path) as pdf:
            for number, page in enumerate(pdf, start=1):
                text = reflow_lines(page.get_text("text"))
                pages.append(Page(number=number, text=normalize_text(text)))
        if any(p.text for p in pages):
            return pages
    except Exception:  # noqa: BLE001 - 缺依赖或文件损坏时退回 pypdf
        pages = []

    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return [
        Page(number=i, text=normalize_text(reflow_lines(page.extract_text() or "")))
        for i, page in enumerate(reader.pages, start=1)
    ]


def _load_docx(path: Path) -> list[Page]:
    from docx import Document as DocxDocument

    document = DocxDocument(str(path))
    blocks: list[str] = [p.text.strip() for p in document.paragraphs if p.text.strip()]

    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                blocks.append(" | ".join(cells))

    return [Page(number=1, text=normalize_text("\n".join(blocks)))]


def load_document(path: str | Path) -> Doc:
    """解析单个文件。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise UnsupportedFileError(f"暂不支持 {suffix or '无后缀'} 文件：{path.name}")

    if suffix == ".pdf":
        pages = _load_pdf(path)
    elif suffix == ".docx":
        pages = _load_docx(path)
    else:
        pages = [Page(number=1, text=normalize_text(_read_plain(path)))]

    pages = [p for p in pages if p.text.strip()]
    if not pages:
        raise EmptyDocumentError(f"{path.name} 中没有解析到文字（可能是扫描件，需要先做 OCR）")

    digest = hashlib.sha1(path.read_bytes()).hexdigest()[:12]
    return Doc(
        doc_id=digest,
        source=path.name,
        path=str(path),
        pages=pages,
        char_count=sum(len(p.text) for p in pages),
    )


def iter_files(directory: str | Path, recursive: bool = True) -> list[Path]:
    """列出目录下所有受支持的文件。"""
    directory = Path(directory)
    if not directory.exists():
        return []
    pattern = "**/*" if recursive else "*"
    files = [
        p for p in directory.glob(pattern)
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES and not p.name.startswith("~$")
    ]
    return sorted(files)


def load_directory(directory: str | Path, recursive: bool = True) -> tuple[list[Doc], list[tuple[str, str]]]:
    """批量解析目录，返回 (成功列表, [(文件名, 失败原因)])。"""
    docs: list[Doc] = []
    errors: list[tuple[str, str]] = []
    for path in iter_files(directory, recursive=recursive):
        try:
            docs.append(load_document(path))
        except Exception as exc:  # noqa: BLE001 - 单个文件失败不影响整体入库
            errors.append((path.name, f"{type(exc).__name__}: {exc}"))
    return docs, errors


if __name__ == "__main__":  # python -m rag.loader <文件或目录>
    import sys

    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "data" / "documents"
    if target.is_dir():
        loaded, failures = load_directory(target)
        print(f"解析成功 {len(loaded)} 个文件，失败 {len(failures)} 个")
        for doc in loaded:
            print(f"  {doc.source:30s} {doc.page_count} 页 / {doc.char_count} 字")
        for name, reason in failures:
            print(f"  [失败] {name}: {reason}")
    else:
        doc = load_document(target)
        print(f"{doc.source}: {doc.page_count} 页 / {doc.char_count} 字")
        print(doc.pages[0].text[:400])
