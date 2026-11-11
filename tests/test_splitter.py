"""切片器测试：长度约束、重叠、偏移量、边界情况。"""

from __future__ import annotations

import unittest

from rag.splitter import RecursiveCharacterTextSplitter


class RecursiveSplitterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40)
        self.text = "校园卡丢失后，请携带学生证到行政楼一楼服务中心办理补办手续。" * 30

    def test_chunks_never_exceed_size(self) -> None:
        chunks = self.splitter.split_text(self.text)
        self.assertGreater(len(chunks), 1, "长文本应被切成多个片段")
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 200, f"片段超长：{len(chunk)} 字")
            self.assertTrue(chunk.strip(), "不应产生空白片段")

    def test_neighbours_overlap(self) -> None:
        pieces = self.splitter.split_text_with_offsets(self.text)
        self.assertGreater(len(pieces), 1)
        for prev, cur in zip(pieces, pieces[1:]):
            self.assertLess(cur.start, prev.end, "相邻片段之间应保留重叠")
            self.assertGreaterEqual(prev.end - cur.start, 0)

    def test_offsets_point_back_to_source(self) -> None:
        for piece in self.splitter.split_text_with_offsets(self.text):
            self.assertEqual(self.text[piece.start:piece.end].strip(), piece.text.strip())

    def test_hard_split_without_separators(self) -> None:
        chunks = self.splitter.split_text("甲" * 1000)
        self.assertTrue(all(len(c) <= 200 for c in chunks))
        self.assertGreaterEqual(sum(len(c) for c in chunks), 1000, "硬切不应丢字")

    def test_english_and_code_splitting(self) -> None:
        code = "def foo():\n    return 1\n\n" * 40
        chunks = self.splitter.split_text(code)
        self.assertTrue(all(len(c) <= 200 for c in chunks))

    def test_blank_and_short_text(self) -> None:
        self.assertEqual(self.splitter.split_text("   \n\n  "), [])
        self.assertEqual(self.splitter.split_text("短句。"), ["短句。"])

    def test_invalid_params(self) -> None:
        with self.assertRaises(ValueError):
            RecursiveCharacterTextSplitter(chunk_size=10, chunk_overlap=20)


if __name__ == "__main__":
    unittest.main()
