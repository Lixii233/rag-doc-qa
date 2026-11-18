"""RAG 校园文档问答 Agent。

模块划分
--------
- ``loader``     : PDF / Word / 文本解析，保留页码
- ``splitter``   : 递归字符切分（500 / 50，带重叠与偏移）
- ``embeddings`` : BGE 向量化，缺模型时降级为哈希向量
- ``store``      : ChromaDB 持久化向量库（cosine）
- ``retriever``  : 相似度检索 + 阈值分流（answer / clarify / refuse）
- ``generator``  : 提示词、流式生成、离线摘录兜底
- ``pipeline``   : 端到端编排
"""

from .config import Settings, load_settings
from .pipeline import RAGPipeline
from .schemas import Answer, Chunk, Doc, Page, Route, SearchResult

__all__ = [
    "Answer",
    "Chunk",
    "Doc",
    "Page",
    "RAGPipeline",
    "Route",
    "SearchResult",
    "Settings",
    "load_settings",
]
