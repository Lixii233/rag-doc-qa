"""向量化模型。

默认使用 BGE（BAAI/bge-small-zh-v1.5），这是中文检索里性价比很高的小模型：
- 768 维以内、CPU 也能跑、对中文短句相似度效果好；
- 检索时应给 query 加指令前缀（BGE 官方建议），文档侧不加。

如果本地没有模型 / 没装 sentence-transformers，会自动降级到
`HashingEmbedder`（字符 n-gram 哈希 + 次线性词频 + L2 归一化）。
降级实现没有语义泛化能力，但能保证项目在完全离线、零依赖的环境里跑通全流程。
"""

from __future__ import annotations

import hashlib
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

# 国内网络直连 huggingface 经常超时，默认走镜像；可用环境变量覆盖
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# 模型缓存放到项目内，避免用户目录权限问题，也方便整体拷贝/清理
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[1] / ".hf-cache"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


class BaseEmbedder(ABC):
    """所有向量化实现的统一接口。"""

    name: str = "base"
    dim: int = 0
    is_fallback: bool = False

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """文档侧向量，返回 (n, dim) 的 float32 数组。"""

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        """查询侧向量，返回 (dim,) 的 float32 数组。"""

    def embed_query_batch(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self.embed_query(t) for t in texts]) if texts else np.zeros((0, self.dim), dtype=np.float32)

    @property
    def status(self) -> str:
        return f"{self.name} (dim={self.dim})"


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (matrix / norms).astype(np.float32)


class BGEEmbedder(BaseEmbedder):
    """基于 sentence-transformers 的 BGE 模型。"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5", device: str | None = None, batch_size: int = 32) -> None:
        from sentence_transformers import SentenceTransformer  # 延迟导入，避免无谓的重依赖

        # 关掉 transformers 的 "Loading weights" 进度条与告警，演示时控制台更干净
        try:
            from transformers.utils import logging as hf_logging

            hf_logging.disable_progress_bar()
            hf_logging.set_verbosity_error()
        except Exception:  # noqa: BLE001 - 老版本 transformers 没有这些开关
            pass

        self.name = model_name
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name, device=device)
        # sentence-transformers 6.x 改了方法名，这里兼容新旧版本
        getter = getattr(self._model, "get_embedding_dimension", None) or self._model.get_sentence_embedding_dimension
        self.dim = int(getter() or 512)
        # 仅中文 bge 系列需要 query 指令
        self.query_instruction = BGE_QUERY_INSTRUCTION if "zh" in model_name.lower() else ""

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return self._encode(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([self.query_instruction + text])[0]


class HashingEmbedder(BaseEmbedder):
    """零依赖兜底向量化：字符 n-gram 哈希 + 次线性词频 + L2 归一化。"""

    def __init__(self, dim: int = 512) -> None:
        self.name = "hashing-char-ngram"
        self.dim = dim
        self.is_fallback = True
        self._token_pattern = re.compile(r"[\u4e00-\u9fff]|[a-zA-Z0-9]+")

    def _tokens(self, text: str) -> list[str]:
        chars = self._token_pattern.findall(text.lower())
        grams = list(chars)
        grams += [chars[i] + chars[i + 1] for i in range(len(chars) - 1)]
        grams += [chars[i] + chars[i + 1] + chars[i + 2] for i in range(len(chars) - 2)]
        return grams

    def _bucket(self, token: str) -> tuple[int, float]:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        return value % self.dim, 1.0 if (value >> 63) & 1 else -1.0

    def _embed_one(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        counts: dict[str, int] = {}
        for token in self._tokens(text):
            counts[token] = counts.get(token, 0) + 1
        for token, count in counts.items():
            index, sign = self._bucket(token)
            vector[index] += sign * (1.0 + np.log(count))
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self._embed_one(t) for t in texts]).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed_one(text)


_CACHE: dict[str, BaseEmbedder] = {}


def get_embedder(backend: str = "auto", model_name: str = "BAAI/bge-small-zh-v1.5", dim: int = 512) -> BaseEmbedder:
    """按配置返回向量化模型，失败时自动降级（结果按 key 缓存）。"""
    key = f"{backend}|{model_name}|{dim}"
    if key in _CACHE:
        return _CACHE[key]

    embedder: BaseEmbedder
    if backend == "hash":
        embedder = HashingEmbedder(dim)
    elif backend in {"auto", "bge"}:
        try:
            embedder = BGEEmbedder(model_name)
        except Exception as exc:  # noqa: BLE001 - 任何原因都降级，保证可用
            print(f"[embedding] BGE 加载失败（{type(exc).__name__}: {exc}），降级为哈希向量。")
            embedder = HashingEmbedder(dim)
    else:
        print(f"[embedding] 未识别的向量后端 {backend!r}，改用哈希向量。")
        embedder = HashingEmbedder(dim)

    _CACHE[key] = embedder
    return embedder


if __name__ == "__main__":  # python -m rag.embeddings
    emb = get_embedder()
    print("模型:", emb.status)
    docs = ["图书馆开放时间为周一至周五 8:00-22:00。", "宿舍晚上 23:30 熄灯。", "奖学金每学年评定一次。"]
    vectors = emb.embed_documents(docs)
    query = emb.embed_query("图书馆几点关门？")
    scores = vectors @ query
    for text, score in sorted(zip(docs, scores), key=lambda x: -x[1]):
        print(f"{score:+.3f}  {text}")
