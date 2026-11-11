"""端到端测试：建库 → 检索 → 生成（离线模式，不依赖网络与 API Key）。"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from rag.config import load_settings
from rag.pipeline import RAGPipeline
from rag.schemas import Route

SAMPLE = """# 校园卡使用指南

## 补办流程
校园卡丢失后，请先通过校园卡 App 挂失，然后携带学生证到行政楼一楼服务中心办理补办，
补办工本费 20 元，一般 3 个工作日后可取卡。

## 充值方式
可以通过校园卡 App、微信小程序或自助充值机充值，单次充值上限 500 元。
"""


class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.mkdtemp(prefix="rag-test-")
        root = Path(cls._tmp)
        docs = root / "documents"
        docs.mkdir()
        (docs / "校园卡使用指南.md").write_text(SAMPLE, encoding="utf-8")
        (docs / "宿舍管理规定.md").write_text(
            "# 宿舍管理\n\n本科生宿舍熄灯时间为 23:00，周五周六延长至 24:00。禁止使用大功率电器。",
            encoding="utf-8",
        )

        base = load_settings()
        cls.settings = replace(
            base,
            data_dir=root,
            docs_dir=docs,
            upload_dir=root / "uploads",
            index_dir=root / "index",
            collection_name="test_collection",
            embedding_backend="hash",
            embedding_dim=256,
            llm_provider="offline",
            llm_api_key="",
            chunk_size=120,
            chunk_overlap=20,
            top_k=3,
            similarity_threshold=0.0,  # 让检索一定进入 ANSWER 分支，专门测生成链路
        )
        cls.pipeline = RAGPipeline(cls.settings)
        cls.stats = cls.pipeline.index_directory()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.pipeline.reset()
        shutil.rmtree(cls._tmp, ignore_errors=True)

    def test_index_built(self) -> None:
        self.assertEqual(self.stats["documents"], 2)
        self.assertGreater(self.stats["chunks"], 0)
        self.assertEqual(self.pipeline.store.count(), self.stats["chunks"])

    def test_sources_listed(self) -> None:
        names = [name for name, _ in self.pipeline.store.sources()]
        self.assertIn("校园卡使用指南.md", names)
        self.assertIn("宿舍管理规定.md", names)

    def test_search_hits_expected_document(self) -> None:
        result = self.pipeline.search("校园卡丢了怎么补办", top_k=3)
        self.assertIs(result.route, Route.ANSWER)
        self.assertEqual(result.hits[0].source, "校园卡使用指南.md")
        self.assertGreater(result.hits[0].score, 0.0)

    def test_ask_returns_answer_with_citation(self) -> None:
        answer = self.pipeline.ask("校园卡补办要多少钱")
        self.assertIs(answer.route, Route.ANSWER)
        self.assertTrue(answer.text.strip())
        self.assertTrue(answer.hits)
        self.assertIn("校园卡使用指南.md", [hit.source for hit in answer.hits])

    def test_stream_events_order(self) -> None:
        events = list(self.pipeline.stream_answer("宿舍几点熄灯"))
        self.assertEqual(events[0]["type"], "retrieval")
        self.assertEqual(events[-1]["type"], "done")
        tokens = [e["text"] for e in events if e["type"] == "token"]
        self.assertTrue(tokens, "应当逐 token 输出")
        self.assertEqual("".join(tokens), events[-1]["answer"].text)

    def test_refuse_when_threshold_too_high(self) -> None:
        answer = self.pipeline.ask("请帮我写一段快速排序", threshold=0.99)
        self.assertIs(answer.route, Route.REFUSE)
        self.assertTrue(answer.text.strip(), "拒答也要给出说明而不是空串")

    def test_upload_and_index(self) -> None:
        stats = self.pipeline.index_uploads([("临时通知.md", "# 通知\n\n停电时间：周四 9:00-12:00。".encode())])
        self.assertEqual(stats["documents"], 1)
        self.assertGreater(stats["chunks"], 0)
        result = self.pipeline.search("什么时候停电", top_k=3)
        self.assertEqual(result.hits[0].source, "临时通知.md")
        self.assertTrue(list(self.pipeline.settings.upload_dir.glob("*临时通知.md")))

    def test_stats_shape(self) -> None:
        stats = self.pipeline.stats()
        for key in ("chunks", "sources", "embedder", "llm", "chunk_size", "threshold", "top_k"):
            self.assertIn(key, stats)
        self.assertFalse(stats["embedder_fallback"] is None)
        self.assertIsNotNone(stats["meta"])


if __name__ == "__main__":
    unittest.main()
