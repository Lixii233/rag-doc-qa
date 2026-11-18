"""检索与“相似度阈值分流”。

流程：query → 向量化 → Chroma top-k → 按阈值判定走哪条分支
     ├── 高相关：把（过滤后的）片段交给大模型生成答案
     ├── 弱相关：先追问 / 提示改写，并把最接近的文档名抛给用户
     └── 不相关：直接说明知识库范围，不硬编答案
阈值判定是控制“幻觉”的第一道闸门：宁可说不知道，也不让模型对无关片段自由发挥。
"""

from __future__ import annotations

from .config import Settings
from .embeddings import BaseEmbedder
from .schemas import Chunk, Route, SearchResult
from .store import ChromaStore


class Retriever:
    def __init__(self, store: ChromaStore, embedder: BaseEmbedder, settings: Settings) -> None:
        self.store = store
        self.embedder = embedder
        self.settings = settings

    def search(
        self,
        query: str,
        top_k: int | None = None,
        threshold: float | None = None,
    ) -> SearchResult:
        query = (query or "").strip()
        if not query:
            raise ValueError("问题不能为空")

        top_k = top_k or self.settings.top_k
        threshold = self.settings.similarity_threshold if threshold is None else threshold

        vector = self.embedder.embed_query(query)
        candidates = self.store.query(vector, top_k=top_k)

        for chunk in candidates:  # Chroma 已按距离升序返回，这里显式排序更稳妥
            chunk.score = round(chunk.score, 4)
        candidates.sort(key=lambda c: -c.score)

        route, reason, hits = self._route(candidates, threshold)
        return SearchResult(
            query=query,
            hits=hits,
            candidates=candidates,
            route=route,
            reason=reason,
            threshold=threshold,
        )

    # ------------------------------------------------------------------ 分流逻辑
    def _route(self, candidates: list[Chunk], threshold: float) -> tuple[Route, str, list[Chunk]]:
        if not candidates:
            return Route.REFUSE, "知识库还是空的，请先上传文档并建立索引。", []

        best = candidates[0]
        strategy = self.settings.low_relevance_strategy
        weak_floor = threshold * self.settings.weak_ratio

        if best.score >= threshold:
            hits = [c for c in candidates if c.score >= threshold]
            return (
                Route.ANSWER,
                f"命中 {len(hits)} 个片段，最高相似度 {best.score:.3f} ≥ 阈值 {threshold:.2f}",
                hits,
            )

        if strategy == "general":
            return (
                Route.GENERAL,
                f"最高相似度仅 {best.score:.3f} < 阈值 {threshold:.2f}，按配置改用通用模式作答（无文档依据）",
                [],
            )

        if strategy == "clarify" and best.score >= weak_floor:
            return (
                Route.CLARIFY,
                f"最高相似度 {best.score:.3f} 低于阈值 {threshold:.2f}，但接近临界值，先追问确认",
                [best],
            )

        return (
            Route.REFUSE,
            f"最高相似度仅 {best.score:.3f} < 阈值 {threshold:.2f}，判定为知识库未收录",
            [],
        )

    # ------------------------------------------------------------------ 调试用
    def explain(self, query: str, top_k: int | None = None) -> str:
        result = self.search(query, top_k=top_k)
        lines = [f"Q: {query}", f"route={result.route.value}  {result.reason}", ""]
        for i, chunk in enumerate(result.candidates, 1):
            flag = "✅" if chunk.score >= result.threshold else "·"
            lines.append(f"{flag} {chunk.score:.3f}  {chunk.source} p.{chunk.page}  {chunk.text[:50]!r}")
        return "\n".join(lines)
