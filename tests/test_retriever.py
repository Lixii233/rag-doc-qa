"""阈值分流测试：高相关 / 弱相关 / 不相关三条分支是否走对。"""

from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from rag.config import load_settings
from rag.retriever import Retriever
from rag.schemas import Chunk, Route


class FakeEmbedder:
    """检索逻辑测试不需要真实语义，返回固定向量即可。"""

    name = "fake"
    dim = 8
    is_fallback = True
    status = "fake embedder"

    def embed_documents(self, texts):
        return np.ones((len(texts), self.dim), dtype=np.float32)

    def embed_query(self, text):
        return np.ones(self.dim, dtype=np.float32)


class FakeStore:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores

    def query(self, embedding, top_k=4):  # noqa: ARG002
        return [
            Chunk(
                chunk_id=f"c{i}",
                text=f"片段{i}",
                source=f"文档{i}.md",
                page=1,
                chunk_index=i,
                doc_id=f"d{i}",
                start=0,
                end=4,
                score=score,
            )
            for i, score in enumerate(self.scores[:top_k])
        ]


def make_retriever(scores: list[float], **overrides) -> Retriever:
    settings = replace(load_settings(), similarity_threshold=0.35, weak_ratio=0.85, **overrides)
    return Retriever(store=FakeStore(scores), embedder=FakeEmbedder(), settings=settings)


class RoutingTest(unittest.TestCase):
    def test_high_similarity_answers(self) -> None:
        result = make_retriever([0.72, 0.55, 0.40, 0.20]).search("图书馆能借几本书")
        self.assertIs(result.route, Route.ANSWER)
        self.assertEqual(len(result.hits), 3, "只应保留超过阈值的片段")
        self.assertTrue(all(hit.score >= 0.35 for hit in result.hits))
        self.assertIn("0.72", result.reason)

    def test_weak_similarity_clarifies(self) -> None:
        result = make_retriever([0.32, 0.30, 0.29]).search("那个东西在哪办")
        self.assertIs(result.route, Route.CLARIFY)
        self.assertEqual(len(result.hits), 1, "追问时只带最接近的一条作为提示")
        self.assertLess(result.candidates[0].score, 0.35)

    def test_low_similarity_refuses(self) -> None:
        result = make_retriever([0.12, 0.08]).search("帮我写一段快速排序")
        self.assertIs(result.route, Route.REFUSE)
        self.assertEqual(result.hits, [])

    def test_threshold_override_per_query(self) -> None:
        retriever = make_retriever([0.60])
        self.assertIs(retriever.search("问题", threshold=0.8).route, Route.REFUSE)
        self.assertIs(retriever.search("问题", threshold=0.5).route, Route.ANSWER)

    def test_general_strategy(self) -> None:
        retriever = make_retriever([0.12], low_relevance_strategy="general")
        self.assertIs(retriever.search("问题").route, Route.GENERAL)

    def test_empty_store(self) -> None:
        result = make_retriever([]).search("任何问题")
        self.assertIs(result.route, Route.REFUSE)
        self.assertIn("空的", result.reason)

    def test_blank_query_raises(self) -> None:
        with self.assertRaises(ValueError):
            make_retriever([0.5]).search("   ")

    def test_explain_output(self) -> None:
        text = make_retriever([0.72, 0.20]).explain("图书馆能借几本书")
        self.assertIn("route=answer", text)
        self.assertIn("文档0.md", text)


if __name__ == "__main__":
    unittest.main()
