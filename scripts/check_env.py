"""环境自检：依赖版本 / 向量模型是否可用 / 向量库能否读写 / 大模型是否配置。

用法：
    python scripts/check_env.py           # 全量自检（会尝试加载 BGE 模型）
    python scripts/check_env.py --fast    # 跳过模型下载，只看依赖与配置
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OK = "[OK]"
WARN = "[!!]"
BAD = "[NG]"


def _version(module: str) -> str:
    try:
        mod = __import__(module)
    except Exception as exc:  # noqa: BLE001
        return f"{BAD} 未安装（{type(exc).__name__}）"
    return getattr(mod, "__version__", "unknown")


def check_dependencies() -> bool:
    print("== 依赖 ==")
    print(f"  Python            : {platform.python_version()} ({platform.machine()})")
    optional_bad = False
    for name, label, required in [
        ("streamlit", "Streamlit", True),
        ("chromadb", "ChromaDB", True),
        ("pypdf", "pypdf", True),
        ("fitz", "PyMuPDF", False),
        ("docx", "python-docx", True),
        ("openai", "openai", True),
        ("sentence_transformers", "sentence-transformers", False),
        ("torch", "torch", False),
    ]:
        version = _version(name)
        flag = BAD if version.startswith(BAD) else OK
        if version.startswith(BAD) and not required:
            optional_bad = True
        print(f"  {label:22s}: {flag} {version}")
    if optional_bad:
        print(f"  {WARN}sentence-transformers / torch 缺失时只能使用哈希向量兜底（语义检索会明显变差）")
    return True


def check_embedder(fast: bool) -> None:
    print("\n== 向量模型 ==")
    from rag.embeddings import get_embedder

    if fast:
        print(f"  {WARN}--fast 模式跳过模型加载")
        return

    started = time.time()
    embedder = get_embedder(backend="auto")
    print(f"  使用模型 : {embedder.status}")
    print(f"  向量维度 : {embedder.dim}")
    print(f"  加载耗时 : {time.time() - started:.1f}s")
    if embedder.is_fallback:
        print(f"  {WARN}已降级为哈希向量：检索能用，但语义泛化能力弱于 BGE")
        return

    import numpy as np

    docs = ["图书馆本科生一次最多可借 10 本书", "学生宿舍 23:00 熄灯", "实验室危险化学品双人双锁"]
    query = "宿舍几点熄灯"
    vectors = embedder.embed_documents(docs)
    q = embedder.embed_query(query)
    scores = (vectors @ q).tolist()
    best = int(np.argmax(scores))
    print(f"  语义检索 : “{query}” → {docs[best]}（相似度 {scores[best]:.3f}）")
    print(f"  {OK if best == 1 else WARN}命中期望文档" if best == 1 else f"  {WARN}未命中期望文档")


def check_store() -> None:
    print("\n== 向量库 ==")
    from rag import load_settings
    from rag.store import ChromaStore

    settings = load_settings()
    store = ChromaStore(settings.chroma_dir, settings.collection_name)
    count = store.count()
    print(f"  目录     : {settings.chroma_dir}")
    print(f"  片段数量 : {count}" + ("" if count else "（空库，先运行 scripts/build_index.py）"))
    for name, pieces in store.sources()[:10]:
        print(f"    · {name}（{pieces} 段）")


def check_llm() -> None:
    print("\n== 大模型 ==")
    from rag import load_settings
    from rag.generator import LLMClient

    settings = load_settings()
    client = LLMClient(settings)
    print(f"  状态     : {client.status}")
    print(f"  接口地址 : {settings.llm_base_url or '(未设置)'}")
    if not client.available:
        print(f"  {WARN}未配置 Key，将使用离线摘录模式（答案直接摘自原文）")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="项目环境自检")
    parser.add_argument("--fast", action="store_true", help="跳过向量模型加载")
    args = parser.parse_args(argv)

    check_dependencies()
    check_embedder(args.fast)
    check_store()
    check_llm()
    print("\n完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
