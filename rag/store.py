"""ChromaDB 持久化向量库封装。

设计要点
--------
- **cosine 距离**：建库时声明 `hnsw:space=cosine`，相似度 = 1 - 距离。
- **幂等入库**：chunk_id 由 `doc_id + 页码 + 序号` 决定，重复入库只会覆盖，不会产生重复片段。
- **自带向量**：写入/查询都显式传 embedding，不依赖 Chroma 内置的默认模型（避免联网下载 ONNX 模型）。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .schemas import Chunk


class ChromaStore:
    def __init__(self, persist_dir: str | Path, collection_name: str = "campus_docs") -> None:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,  # 向量由我们自己算
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------ 写入
    def upsert(self, chunks: list[Chunk], embeddings: np.ndarray, batch_size: int = 256) -> int:
        if not chunks:
            return 0
        if len(chunks) != len(embeddings):
            raise ValueError(f"chunk 数({len(chunks)}) 与向量数({len(embeddings)}) 不一致")

        written = 0
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            vectors = embeddings[start:start + batch_size]
            self._collection.upsert(
                ids=[c.chunk_id for c in batch],
                embeddings=[v.tolist() for v in vectors],
                documents=[c.text for c in batch],
                metadatas=[c.to_metadata() for c in batch],
            )
            written += len(batch)
        return written

    # ------------------------------------------------------------------ 查询
    def query(self, embedding: np.ndarray, top_k: int = 4) -> list[Chunk]:
        if self.count() == 0:
            return []
        result = self._collection.query(
            query_embeddings=[np.asarray(embedding, dtype=np.float32).tolist()],
            n_results=min(top_k, self.count()),
            include=["documents", "metadatas", "distances"],
        )
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        ids = (result.get("ids") or [[]])[0]

        chunks: list[Chunk] = []
        for chunk_id, text, meta, distance in zip(ids, documents, metadatas, distances):
            meta = meta or {}
            # cosine 距离 ∈ [0, 2]，相似度 ∈ [-1, 1]；本项目向量均已归一化
            score = float(1.0 - distance)
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=text or "",
                    source=str(meta.get("source", "未知文档")),
                    page=int(meta.get("page", 1)),
                    chunk_index=int(meta.get("chunk_index", 0)),
                    doc_id=str(meta.get("doc_id", "")),
                    start=int(meta.get("start", 0)),
                    end=int(meta.get("end", 0)),
                    score=round(max(-1.0, min(1.0, score)), 4),
                )
            )
        return chunks

    # ------------------------------------------------------------------ 维护
    def count(self) -> int:
        return int(self._collection.count())

    def sources(self) -> list[tuple[str, int]]:
        """返回 [(文件名, 片段数)]，用于前端展示知识库构成。"""
        if self.count() == 0:
            return []
        data = self._collection.get(include=["metadatas"])
        counter: dict[str, int] = {}
        for meta in data.get("metadatas") or []:
            name = str((meta or {}).get("source", "未知文档"))
            counter[name] = counter.get(name, 0) + 1
        return sorted(counter.items(), key=lambda x: -x[1])

    def delete_source(self, source: str) -> None:
        self._collection.delete(where={"source": source})

    def reset(self) -> None:
        """清空整个集合（重建索引时使用）。"""
        try:
            self._client.delete_collection(self.collection_name)
        except Exception:  # noqa: BLE001 - 集合不存在时忽略
            pass
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )


def make_chunk_id(source: str, doc_id: str, page: int, index: int) -> str:
    raw = f"{source}|{doc_id}|{page}|{index}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:24]
