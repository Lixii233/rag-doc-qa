"""端到端流程编排：解析 → 切片 → 向量化 → 入库 → 检索 → 生成。"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import AsyncIterator, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from .config import Settings, load_settings
from .embeddings import BaseEmbedder, get_embedder
from .generator import AnswerGenerator
from .loader import load_directory, load_document
from .retriever import Retriever
from .schemas import Answer, Chunk, Doc, Route, SearchResult
from .splitter import RecursiveCharacterTextSplitter
from .store import ChromaStore, make_chunk_id

ProgressFn = Callable[[str, float], None] | None


@dataclass
class IndexMeta:
    """记录建库时的关键参数，用于判断是否需要重建索引。"""

    embedder: str
    dim: int
    chunk_size: int
    chunk_overlap: int
    count: int
    updated_at: str
    collection: str


def _noop(message: str, ratio: float) -> None:  # pragma: no cover - 默认回调
    return None


class RAGPipeline:
    """一个实例对应一套“知识库 + 检索 + 生成”的组合。"""

    def __init__(self, settings: Settings | None = None, embedder: BaseEmbedder | None = None) -> None:
        self.settings = settings or load_settings()
        self.settings.ensure_dirs()
        self.embedder = embedder or get_embedder(
            backend=self.settings.embedding_backend,
            model_name=self.settings.embedding_model,
            dim=self.settings.embedding_dim,
        )
        self.store = ChromaStore(self.settings.chroma_dir, self.settings.collection_name)
        self.retriever = Retriever(self.store, self.embedder, self.settings)
        self.generator = AnswerGenerator(self.settings, embedder=self.embedder)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )

    # ------------------------------------------------------------------ 索引
    def make_chunks(self, docs: list[Doc]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for doc in docs:
            for page in doc.pages:
                for index, piece in enumerate(self.splitter.split_text_with_offsets(page.text)):
                    chunks.append(
                        Chunk(
                            chunk_id=make_chunk_id(doc.source, doc.doc_id, page.number, index),
                            text=piece.text,
                            source=doc.source,
                            page=page.number,
                            chunk_index=index,
                            doc_id=doc.doc_id,
                            start=piece.start,
                            end=piece.end,
                        )
                    )
        return chunks

    def index_docs(
        self,
        docs: list[Doc],
        replace: bool = False,
        progress: ProgressFn = None,
    ) -> dict:
        progress = progress or _noop
        if not docs:
            return {"chunks": 0, "documents": 0}

        if replace:
            progress("清空旧索引…", 0.02)
            self.store.reset()

        self._auto_reset_if_incompatible()

        progress(f"切分 {len(docs)} 篇文档…", 0.1)
        chunks = self.make_chunks(docs)
        if not chunks:
            return {"chunks": 0, "documents": len(docs)}

        progress(f"向量化 {len(chunks)} 个片段…", 0.3)
        started = time.time()
        vectors = self.embedder.embed_documents([c.text for c in chunks])

        progress("写入向量库…", 0.8)
        written = self.store.upsert(chunks, vectors)
        elapsed = time.time() - started

        self._write_meta(len(chunks))
        progress("完成", 1.0)
        return {
            "documents": len(docs),
            "chunks": written,
            "seconds": round(elapsed, 2),
            "total_in_store": self.store.count(),
        }

    def index_directory(self, directory: str | Path | None = None, replace: bool = False, progress: ProgressFn = None) -> dict:
        progress = progress or _noop
        directory = Path(directory) if directory else self.settings.docs_dir
        progress(f"解析目录 {directory.name}…", 0.05)
        docs, errors = load_directory(directory)
        stats = self.index_docs(docs, replace=replace, progress=progress)
        stats["errors"] = errors
        stats["dir"] = str(directory)
        return stats

    def index_files(self, paths: list[str | Path], replace: bool = False, progress: ProgressFn = None) -> dict:
        progress = progress or _noop
        docs: list[Doc] = []
        errors: list[tuple[str, str]] = []
        for path in paths:
            try:
                docs.append(load_document(path))
            except Exception as exc:  # noqa: BLE001
                errors.append((Path(path).name, f"{type(exc).__name__}: {exc}"))
        stats = self.index_docs(docs, replace=replace, progress=progress)
        stats["errors"] = errors
        return stats

    def save_upload(self, filename: str, content: bytes) -> Path:
        """把上传的文件落到 data/uploads；重名时加内容指纹，避免互相覆盖。"""
        safe_name = Path(filename).name
        target = self.settings.upload_dir / safe_name
        if target.exists() and target.read_bytes() != content:
            digest = hashlib.sha1(content).hexdigest()[:6]
            target = self.settings.upload_dir / f"{Path(safe_name).stem}_{digest}{Path(safe_name).suffix}"
        target.write_bytes(content)
        return target

    def index_uploads(self, files: list[tuple[str, bytes]], replace: bool = False, progress: ProgressFn = None) -> dict:
        paths = [self.save_upload(name, content) for name, content in files]
        return self.index_files(paths, replace=replace, progress=progress)

    # ------------------------------------------------------------------ 查询
    def search(self, question: str, top_k: int | None = None, threshold: float | None = None) -> SearchResult:
        return self.retriever.search(question, top_k=top_k, threshold=threshold)

    def ask(
        self,
        question: str,
        top_k: int | None = None,
        threshold: float | None = None,
        history: list[tuple[str, str]] | None = None,
    ) -> Answer:
        result = self.search(question, top_k=top_k, threshold=threshold)
        text = "".join(self.generator.stream_answer(question, result, history, self.store.sources()))
        return Answer(
            question=question,
            text=text,
            route=result.route,
            reason=result.reason,
            hits=result.hits,
        )

    def stream_answer(
        self,
        question: str,
        top_k: int | None = None,
        threshold: float | None = None,
        history: list[tuple[str, str]] | None = None,
    ) -> Iterator[dict]:
        """事件流：retrieval → token* → done，方便 UI / CLI 逐步渲染。"""
        result = self.search(question, top_k=top_k, threshold=threshold)
        yield {"type": "retrieval", "result": result}

        collected: list[str] = []
        for piece in self.generator.stream_answer(question, result, history, self.store.sources()):
            collected.append(piece)
            yield {"type": "token", "text": piece}

        yield {
            "type": "done",
            "answer": Answer(
                question=question,
                text="".join(collected),
                route=result.route,
                reason=result.reason,
                hits=result.hits,
            ),
        }

    async def astream_answer(
        self,
        question: str,
        top_k: int | None = None,
        threshold: float | None = None,
        history: list[tuple[str, str]] | None = None,
    ) -> AsyncIterator[dict]:
        """异步版本：检索在函数里同步完成（很快），生成走异步流。"""
        result = self.search(question, top_k=top_k, threshold=threshold)
        yield {"type": "retrieval", "result": result}

        collected: list[str] = []
        async for piece in self.generator.astream_answer(question, result, history, self.store.sources()):
            collected.append(piece)
            yield {"type": "token", "text": piece}

        yield {
            "type": "done",
            "answer": Answer(
                question=question,
                text="".join(collected),
                route=result.route,
                reason=result.reason,
                hits=result.hits,
            ),
        }

    # ------------------------------------------------------------------ 状态 / 维护
    def stats(self) -> dict:
        meta = self._read_meta()
        return {
            "chunks": self.store.count(),
            "sources": self.store.sources(),
            "embedder": self.embedder.status,
            "embedder_fallback": self.embedder.is_fallback,
            "llm": self.generator.status,
            "chunk_size": self.settings.chunk_size,
            "chunk_overlap": self.settings.chunk_overlap,
            "threshold": self.settings.similarity_threshold,
            "top_k": self.settings.top_k,
            "meta": asdict(meta) if meta else None,
        }

    def reset(self) -> None:
        self.store.reset()
        meta_path = self._meta_path()
        if meta_path.exists():
            meta_path.unlink()

    # ------------------------------------------------------------------ 内部
    def _meta_path(self) -> Path:
        return self.settings.index_dir / "meta.json"

    def _read_meta(self) -> IndexMeta | None:
        path = self._meta_path()
        if not path.exists():
            return None
        try:
            return IndexMeta(**json.loads(path.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001
            return None

    def _write_meta(self, count: int) -> None:
        meta = IndexMeta(
            embedder=self.embedder.name,
            dim=self.embedder.dim,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            count=self.store.count(),
            updated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            collection=self.settings.collection_name,
        )
        self._meta_path().write_text(json.dumps(asdict(meta), ensure_ascii=False, indent=2), encoding="utf-8")

    def _auto_reset_if_incompatible(self) -> None:
        """切换向量模型或切片参数后，旧向量维度/语义已不匹配，自动重建。"""
        meta = self._read_meta()
        if not meta or self.store.count() == 0:
            return
        reasons = []
        if meta.dim != self.embedder.dim:
            reasons.append(f"向量维度 {meta.dim}→{self.embedder.dim}")
        if meta.embedder != self.embedder.name:
            reasons.append(f"向量模型 {meta.embedder}→{self.embedder.name}")
        if meta.chunk_size != self.settings.chunk_size or meta.chunk_overlap != self.settings.chunk_overlap:
            reasons.append(f"切片参数 {meta.chunk_size}/{meta.chunk_overlap}→{self.settings.chunk_size}/{self.settings.chunk_overlap}")
        if reasons:
            print(f"[pipeline] 索引参数变化（{'；'.join(reasons)}），自动清空旧索引。")
            self.store.reset()


__all__ = ["RAGPipeline", "IndexMeta", "Route"]
