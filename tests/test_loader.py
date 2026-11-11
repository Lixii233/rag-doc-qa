"""文档解析测试：md / txt / docx 与异常路径。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from docx import Document

from rag.loader import SUPPORTED_SUFFIXES, load_document, load_directory, normalize_text, reflow_lines


class LoaderTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write(self, name: str, text: str, encoding: str = "utf-8") -> Path:
        path = self.dir / name
        path.write_text(text, encoding=encoding)
        return path

    def test_markdown(self) -> None:
        path = self._write("规则.md", "# 标题\n\n第一段内容。\n\n第二段内容。\n")
        doc = load_document(path)
        self.assertEqual(doc.source, "规则.md")
        self.assertIn("第一段内容", "".join(page.text for page in doc.pages))
        self.assertGreater(doc.char_count, 0)

    def test_plain_text_gbk(self) -> None:
        path = self._write("通知.txt", "宿舍熄灯时间为 23:00。", encoding="gb18030")
        doc = load_document(path)
        self.assertIn("熄灯", "".join(page.text for page in doc.pages))

    def test_docx_with_table(self) -> None:
        path = self.dir / "手册.docx"
        document = Document()
        document.add_paragraph("校园卡补办需要 3 个工作日。")
        table = document.add_table(rows=1, cols=2)
        table.cell(0, 0).text = "窗口"
        table.cell(0, 1).text = "行政楼一楼"
        document.save(path)

        doc = load_document(path)
        text = "".join(page.text for page in doc.pages)
        self.assertIn("3 个工作日", text)
        self.assertIn("行政楼一楼", text, "表格内容也应被解析")

    def test_directory_skips_unsupported(self) -> None:
        self._write("a.md", "内容 A")
        (self.dir / "b.exe").write_bytes(b"\x00\x01")
        docs, errors = load_directory(self.dir)
        self.assertEqual(len(docs), 1)
        self.assertEqual(errors, [])

    def test_unsupported_suffix_raises(self) -> None:
        path = self.dir / "x.zip"
        path.write_bytes(b"PK")
        with self.assertRaises(ValueError):
            load_document(path)

    def test_missing_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_document(self.dir / "不存在.md")

    def test_normalize_text(self) -> None:
        raw = "第一行\r\n\r\n\r\n\r\n第二行\u200b  \t "
        normalized = normalize_text(raw)
        self.assertNotIn("\r", normalized)
        self.assertNotIn("\u200b", normalized)
        self.assertIn("第一行\n\n第二行", normalized)
        self.assertEqual(normalized, normalized.strip())

    def test_reflow_joins_wrapped_pdf_lines(self) -> None:
        raw = "\n".join(
            [
                "关于期末考试安排的通知",
                "各学院、各位同学：",
                "1. 期末考试统一安排在 2027 年 1 月 4 日至 1 月 15 日进",
                "行，每天三个时段。",
                "2. 考生须持学生证入场。",
            ]
        )
        text = reflow_lines(raw)
        lines = text.splitlines()
        self.assertEqual(lines[0], "关于期末考试安排的通知", "标题应独立成行")
        self.assertEqual(lines[1], "各学院、各位同学：", "称呼语不应被并到标题里")
        self.assertIn("1 月 15 日进行，每天三个时段。", text, "被版式折断的句子应接回原句")
        self.assertNotIn("进\n行", text)
        self.assertEqual(lines[-1], "2. 考生须持学生证入场。", "新条目另起一行")

    def test_supported_suffixes_declared(self) -> None:
        self.assertTrue({".pdf", ".docx", ".txt", ".md"}.issubset(SUPPORTED_SUFFIXES))


if __name__ == "__main__":
    unittest.main()
