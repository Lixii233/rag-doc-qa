"""命令行建库：解析 → 切片 → 向量化 → 写入 ChromaDB。

用法：
    python scripts/build_index.py                 # 建 data/documents + data/samples 的索引（增量覆盖）
    python scripts/build_index.py --reset         # 先清空再建
    python scripts/build_index.py 某文件.pdf       # 只索引指定文件
    python scripts/build_index.py --chunk-size 300 --chunk-overlap 30
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag import RAGPipeline, load_settings  # noqa: E402
from rag.loader import load_directory, load_document  # noqa: E402


def _bar(message: str, ratio: float) -> None:
    width = 28
    filled = int(width * ratio)
    print(f"\r  [{('#' * filled).ljust(width)}] {ratio * 100:3.0f}%  {message:<24}", end="", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="构建校园文档知识库索引")
    parser.add_argument(
        "target",
        nargs="?",
        default=None,
        help="目录或文件路径，默认索引 data/documents 与 data/samples",
    )
    parser.add_argument("--reset", action="store_true", help="清空前先删除已有索引")
    parser.add_argument("--chunk-size", type=int, default=None, help="切片长度（默认 500）")
    parser.add_argument("--chunk-overlap", type=int, default=None, help="切片重叠（默认 50）")
    parser.add_argument("--hash", action="store_true", help="强制使用哈希向量（不下载 BGE 模型）")
    args = parser.parse_args(argv)

    settings = load_settings()
    overrides = {}
    if args.chunk_size:
        overrides["chunk_size"] = args.chunk_size
    if args.chunk_overlap is not None:
        overrides["chunk_overlap"] = args.chunk_overlap
    if args.hash:
        overrides["embedding_backend"] = "hash"
    if overrides:
        settings = replace(settings, **overrides)

    pipeline = RAGPipeline(settings)
    print(f"向量模型：{pipeline.embedder.status}")
    print(f"大模型　：{pipeline.generator.status}")

    started = time.time()
    if args.target:
        target = Path(args.target)
        if target.is_file():
            stats = pipeline.index_files([target], replace=args.reset, progress=_bar)
        else:
            stats = pipeline.index_directory(target, replace=args.reset, progress=_bar)
    else:
        # 默认把“手写文档”和“生成的示例 PDF”一起入库；一次解析再统一切片，
        # 避免分两次调用时 --reset 把前一批清掉。
        docs, errors = [], []
        for directory in (settings.docs_dir, settings.samples_dir):
            if not directory.exists():
                continue
            found, failed = load_directory(directory)
            docs.extend(found)
            errors.extend(failed)
        _bar(f"解析 {len(docs)} 篇文档…", 0.05)
        stats = pipeline.index_docs(docs, replace=args.reset, progress=_bar)
        stats["errors"] = errors
    print()

    elapsed = time.time() - started
    print(f"\n完成：{stats['documents']} 篇文档 → {stats['chunks']} 个片段，用时 {elapsed:.1f}s")
    print(f"库内共 {stats['total_in_store']} 个片段，目录：{settings.chroma_dir}")

    for name, message in stats.get("errors", []):
        print(f"  [!!] 解析失败 {name}：{message}")

    sources = pipeline.store.sources()
    if sources:
        print("\n知识库内容：")
        for name, count in sources:
            print(f"  - {name}（{count} 片段）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
