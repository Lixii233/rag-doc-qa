"""向量化测试：兜底哈希向量的确定性与语义区分度。"""

from __future__ import annotations

import unittest

import numpy as np

from rag.embeddings import HashingEmbedder, get_embedder


class HashingEmbedderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.embedder = HashingEmbedder(dim=256)

    def test_dimension_and_normalization(self) -> None:
        vectors = self.embedder.embed_documents(["图书馆借阅规则", "宿舍熄灯时间"])
        self.assertEqual(vectors.shape, (2, 256))
        norms = np.linalg.norm(vectors, axis=1)
        np.testing.assert_allclose(norms, np.ones(2), atol=1e-5)

    def test_deterministic(self) -> None:
        a = self.embedder.embed_query("校园卡补办流程")
        b = self.embedder.embed_query("校园卡补办流程")
        np.testing.assert_allclose(a, b)

    def test_similar_query_scores_higher(self) -> None:
        doc = self.embedder.embed_query("校园卡丢失后请到行政楼一楼服务中心补办，需三个工作日")
        same_topic = self.embedder.embed_query("校园卡丢了去哪里补办")
        other_topic = self.embedder.embed_query("实验室危险化学品双人双锁管理")
        self.assertGreater(float(doc @ same_topic), float(doc @ other_topic))

    def test_empty_batch(self) -> None:
        vectors = self.embedder.embed_documents([])
        self.assertEqual(vectors.shape, (0, 256))

    def test_fallback_flag(self) -> None:
        self.assertTrue(self.embedder.is_fallback)


class FactoryTest(unittest.TestCase):
    def test_hash_backend(self) -> None:
        embedder = get_embedder(backend="hash", dim=128)
        self.assertEqual(embedder.dim, 128)
        self.assertTrue(embedder.is_fallback)

    def test_unknown_backend_never_crashes(self) -> None:
        embedder = get_embedder(backend="不存在的后端", dim=64)
        self.assertEqual(embedder.dim, 64, "未知后端应降级到哈希向量而不是抛异常")


if __name__ == "__main__":
    unittest.main()
