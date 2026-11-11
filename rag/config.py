"""全局配置。

所有可调参数集中在这里，通过环境变量 / `.env` 覆盖，避免散落在各个模块中。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]  # 项目根目录
load_dotenv(ROOT / ".env")


def _env_str(key: str, default: str) -> str:
    value = os.getenv(key)
    return default if value is None or value.strip() == "" else value.strip()


def _env_int(key: str, default: int) -> int:
    try:
        return int(_env_str(key, str(default)))
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(_env_str(key, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """一次运行期的全部配置。"""

    # 目录
    root_dir: Path = ROOT
    data_dir: Path = ROOT / "data"
    docs_dir: Path = ROOT / "data" / "documents"
    samples_dir: Path = ROOT / "data" / "samples"
    upload_dir: Path = ROOT / "data" / "uploads"
    index_dir: Path = ROOT / "data" / "index"

    # 切分参数（对应简历中的 500 / 50）
    chunk_size: int = 500
    chunk_overlap: int = 50

    # 检索参数
    top_k: int = 4
    similarity_threshold: float = 0.35
    weak_ratio: float = 0.85  # top1 落在 [threshold*weak_ratio, threshold) 视为“弱相关”，走追问分流

    # 向量化
    embedding_backend: str = "auto"  # auto | bge | hash
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_dim: int = 512  # 仅 hash 兜底时使用
    collection_name: str = "campus_docs"

    # 大模型
    llm_provider: str = "auto"  # auto | deepseek | openai | offline
    llm_api_key: str = ""
    llm_base_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-chat"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 1024

    # 低相关度时的兜底策略：clarify(追问) | general(无资料直答) | refuse(直接拒答)
    low_relevance_strategy: str = "clarify"

    extra: dict = field(default_factory=dict)

    # ---------- 派生属性 ----------
    @property
    def chroma_dir(self) -> Path:
        return self.index_dir / "chroma"

    @property
    def has_llm_key(self) -> bool:
        return bool(self.llm_api_key)

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.docs_dir, self.samples_dir, self.upload_dir, self.index_dir, self.chroma_dir):
            path.mkdir(parents=True, exist_ok=True)


def _detect_provider() -> tuple[str, str, str]:
    """按优先级挑选大模型供应商，返回 (provider, api_key, base_url)。"""
    explicit = _env_str("LLM_PROVIDER", "auto").lower()
    deepseek_key = _env_str("DEEPSEEK_API_KEY", "")
    openai_key = _env_str("OPENAI_API_KEY", "")
    base_url = _env_str("LLM_BASE_URL", "")

    if explicit == "offline":
        return "offline", "", ""

    if explicit in {"deepseek", "openai"}:
        key = deepseek_key if explicit == "deepseek" else openai_key
        default_base = "https://api.deepseek.com" if explicit == "deepseek" else "https://api.openai.com/v1"
        return explicit, key, base_url or default_base

    if deepseek_key:
        return "deepseek", deepseek_key, base_url or "https://api.deepseek.com"
    if openai_key:
        return "openai", openai_key, base_url or "https://api.openai.com/v1"
    return "offline", "", ""


@lru_cache(maxsize=1)
def load_settings() -> Settings:
    provider, api_key, base_url = _detect_provider()

    settings = Settings(
        chunk_size=_env_int("CHUNK_SIZE", 500),
        chunk_overlap=_env_int("CHUNK_OVERLAP", 50),
        top_k=_env_int("TOP_K", 4),
        similarity_threshold=_env_float("SIM_THRESHOLD", 0.35),
        weak_ratio=_env_float("WEAK_RATIO", 0.85),
        embedding_backend=_env_str("EMBEDDING_BACKEND", "auto").lower(),
        embedding_model=_env_str("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"),
        embedding_dim=_env_int("EMBEDDING_DIM", 512),
        collection_name=_env_str("COLLECTION_NAME", "campus_docs"),
        llm_provider=provider,
        llm_api_key=api_key,
        llm_base_url=base_url,
        llm_model=_env_str("LLM_MODEL", "deepseek-chat" if provider != "openai" else "gpt-4o-mini"),
        llm_temperature=_env_float("LLM_TEMPERATURE", 0.2),
        llm_max_tokens=_env_int("LLM_MAX_TOKENS", 1024),
        low_relevance_strategy=_env_str("LOW_RELEVANCE_STRATEGY", "clarify").lower(),
    )
    settings.ensure_dirs()
    return settings


if __name__ == "__main__":  # 便于快速自检：python -m rag.config
    s = load_settings()
    for f in ("chunk_size", "chunk_overlap", "top_k", "similarity_threshold", "embedding_backend",
              "embedding_model", "llm_provider", "llm_model", "llm_base_url"):
        print(f"{f:22s} = {getattr(s, f)}")
    print(f"{'llm_api_key':22s} = {'***' if s.llm_api_key else '(空，将使用离线摘录模式)'}")
    print(f"{'chroma_dir':22s} = {s.chroma_dir}")
