"""生成与离线摘录测试：标题过滤、稀有词加权、摘录引用编号。"""

from __future__ import annotations

import unittest

from rag.generator import _char_bigrams, _lexical_scores, _split_sentences, extractive_answer
from rag.schemas import Chunk


def make_chunk(text: str, chunk_id: str = "c1", source: str = "校园卡.md", index: int = 0) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        source=source,
        page=1,
        chunk_index=index,
        doc_id="doc1",
        start=0,
        end=len(text),
    )


class ExtractiveTest(unittest.TestCase):
    def test_split_sentences_drops_headings(self) -> None:
        text = (
            "# 校园卡使用与补办指南\n"
            "## 三、挂失与补办\n"
            "1. 卡片遗失后请立即挂失。\n"
            "2. 补办需本人持身份证到卡务中心办理，工本费 15 元。\n"
        )
        sentences = _split_sentences(text)
        joined = "\n".join(sentences)
        self.assertNotIn("校园卡使用与补办指南", joined, "Markdown 标题不应参与摘录")
        self.assertNotIn("三、挂失与补办", joined, "中文小标题不应参与摘录")
        self.assertTrue(any("工本费 15 元" in s for s in sentences))

    def test_lexical_prefers_rare_terms(self) -> None:
        """查询里的稀有词（补办）应比句句都有的词（校园卡）更能拉开分差。"""
        pool = [
            (0, 0, "校园卡是校内唯一的身份凭证。", _char_bigrams("校园卡是校内唯一的身份凭证。"), make_chunk("x")),
            (0, 1, "校园卡余额不计利息，毕业时退还。", _char_bigrams("校园卡余额不计利息，毕业时退还。"), make_chunk("x")),
            (0, 2, "补办需持身份证到卡务中心办理。", _char_bigrams("补办需持身份证到卡务中心办理。"), make_chunk("x")),
        ]
        scores = _lexical_scores("校园卡丢了怎么补办？", pool)
        self.assertEqual(max(range(3), key=lambda i: scores[i]), 2, "含“补办”的句子应得分最高")

    def test_extractive_answer_uses_citations_and_document_order(self) -> None:
        chunk_a = make_chunk(
            "## 三、挂失与补办\n1. 卡片遗失后请立即挂失。\n2. 补办需持身份证到卡务中心办理，工本费 15 元。",
            chunk_id="a",
            index=0,
        )
        chunk_b = make_chunk("## 五、联系方式\n卡务中心电话 0769-8888 6699。", chunk_id="b", index=1)
        answer = extractive_answer("校园卡丢了怎么补办？", [chunk_a, chunk_b], embedder=None)

        self.assertIn("未配置大模型", answer)
        self.assertIn("工本费 15 元", answer)
        self.assertIn("立即挂失", answer)
        self.assertIn("[1]", answer)
        self.assertLess(answer.index("立即挂失"), answer.index("工本费 15 元"), "应按原文顺序排版")

    def test_extractive_answer_without_sentences(self) -> None:
        answer = extractive_answer("随便问问", [make_chunk("# 只有标题")], embedder=None)
        self.assertIn("无法提炼", answer)


if __name__ == "__main__":
    unittest.main()
