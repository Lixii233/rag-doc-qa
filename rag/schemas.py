"""跨模块共享的数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


@dataclass
class Page:
    """文档中的一页（PDF 按页，docx/txt 视为单页）。"""

    number: int
    text: str


@dataclass
class Doc:
    """解析后的一篇原始文档。"""

    doc_id: str
    source: str  # 展示用文件名
    path: str
    pages: list[Page] = field(default_factory=list)
    char_count: int = 0

    @property
    def page_count(self) -> int:
        return len(self.pages)


@dataclass
class Chunk:
    """切片后的一个知识片段（对应向量库中的一行）。"""

    chunk_id: str
    text: str
    source: str
    page: int
    chunk_index: int
    doc_id: str
    start: int = 0
    end: int = 0
    score: float = 0.0

    def to_metadata(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "doc_id": self.doc_id,
            "page": self.page,
            "chunk_index": self.chunk_index,
            "start": self.start,
            "end": self.end,
        }

    def as_citation(self, index: int) -> str:
        return f"[{index}] 《{self.source}》第 {self.page} 页 · 相似度 {self.score:.3f}"


class Route(str, Enum):
    """一次问答的检索分流结果。"""

    ANSWER = "answer"        # 检索结果足够相关，正常生成
    CLARIFY = "clarify"      # 相关度偏低，先追问 / 提示改写
    REFUSE = "refuse"        # 完全没有相关内容，拒答并说明知识库范围
    GENERAL = "general"      # 无资料但用户允许“无依据”作答


@dataclass
class SearchResult:
    query: str
    hits: list[Chunk]
    candidates: list[Chunk]
    route: Route
    reason: str
    threshold: float

    @property
    def best_score(self) -> float:
        return self.candidates[0].score if self.candidates else 0.0


@dataclass
class Answer:
    """一次完整的问答结果。"""

    question: str
    text: str
    route: Route
    reason: str
    hits: list[Chunk]
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def sources(self) -> list[str]:
        return [chunk.as_citation(i + 1) for i, chunk in enumerate(self.hits)]
