"""命令行入口：建库 / 单次提问 / 连续对话。

    python cli.py build            # 把 data/documents 下的文档建索引
    python cli.py build --rebuild  # 清空后重建
    python cli.py ask "图书馆能借几本书？"
    python cli.py chat             # 进入连续对话
    python cli.py serve            # 启动 Streamlit 界面
"""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
from pathlib import Path

from rag import RAGPipeline, load_settings
from rag.schemas import Route

ROUTE_TEXT = {
    Route.ANSWER: "🟢 命中知识库",
    Route.CLARIFY: "🟡 相关度偏低 → 追问",
    Route.REFUSE: "🔴 知识库未收录 → 拒答",
    Route.GENERAL: "🔵 无资料 → 通用作答",
}


def cmd_build(args: argparse.Namespace) -> int:
    pipeline = RAGPipeline(load_settings())
    print(f"向量模型: {pipeline.embedder.status}")
    print(f"文档目录: {args.dir}")

    def progress(message: str, ratio: float) -> None:
        bar = "█" * int(ratio * 24)
        print(f"\r[{bar:<24}] {ratio * 100:5.1f}%  {message}", end="", flush=True)

    stats = pipeline.index_directory(args.dir, replace=args.rebuild, progress=progress)
    print()
    print(f"完成：{stats['documents']} 篇文档 → {stats['chunks']} 个片段，耗时 {stats['seconds']}s")
    print(f"库内片段总数：{stats['total_in_store']}")
    for name, reason in stats.get("errors", []):
        print(f"  [跳过] {name}: {reason}")
    for name, count in pipeline.store.sources():
        print(f"  · {name}（{count} 段）")
    return 0


def _print_answer(pipeline: RAGPipeline, question: str, top_k: int | None, threshold: float | None, show: bool) -> None:
    events = pipeline.stream_answer(question, top_k=top_k, threshold=threshold)
    first = next(events)
    result = first["result"]
    print(f"\n{ROUTE_TEXT[result.route]}  {result.reason}\n")
    print("答：", end="", flush=True)
    for event in events:
        if event["type"] == "token":
            print(event["text"], end="", flush=True)
    print("\n")

    if result.hits:
        print("依据：")
        for i, chunk in enumerate(result.hits, start=1):
            print(f"  [{i}] 《{chunk.source}》p.{chunk.page}  相似度 {chunk.score:.3f}")
            print(f"      {chunk.text[:90].replace(chr(10), ' ')}…")

    if show:
        print("\n候选片段：")
        for i, chunk in enumerate(result.candidates, start=1):
            mark = "✓" if chunk.score >= result.threshold else " "
            print(f"  {mark} {i}. {chunk.score:.3f}  {chunk.source} p.{chunk.page}  {chunk.text[:50]!r}")


def cmd_ask(args: argparse.Namespace) -> int:
    pipeline = RAGPipeline(load_settings())
    if pipeline.store.count() == 0:
        print("知识库为空，先运行：python cli.py build")
        return 1
    _print_answer(pipeline, args.question, args.top_k, args.threshold, args.show_candidates)
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    pipeline = RAGPipeline(load_settings())
    if pipeline.store.count() == 0:
        print("知识库为空，先运行：python cli.py build")
        return 1
    print(f"知识库：{pipeline.store.count()} 个片段 / {len(pipeline.store.sources())} 篇文档")
    print(f"模型：{pipeline.generator.status}    输入 exit 退出\n")

    history: list[tuple[str, str]] = []
    while True:
        try:
            question = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"exit", "quit", "q"}:
            break
        if not question:
            continue

        events = pipeline.stream_answer(question, top_k=args.top_k, threshold=args.threshold, history=history)
        result = next(events)["result"]
        print(f"\n{ROUTE_TEXT[result.route]}  {result.reason}")
        print("助手：", end="", flush=True)
        buffer: list[str] = []
        for event in events:
            if event["type"] == "token":
                buffer.append(event["text"])
                print(event["text"], end="", flush=True)
        print("\n")
        if result.hits:
            for i, chunk in enumerate(result.hits, start=1):
                print(f"  [{i}] 《{chunk.source}》p.{chunk.page} · {chunk.score:.3f}")
            print()
        history.append((question, "".join(buffer)))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    app = Path(__file__).resolve().parent / "app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app), "--server.port", str(args.port)]
    print("启动：", " ".join(cmd))
    return subprocess.call(cmd)


def cmd_async_demo(args: argparse.Namespace) -> int:
    """演示异步流式接口（与 Web 端共用同一套 pipeline）。"""
    pipeline = RAGPipeline(load_settings())

    async def run() -> None:
        async for event in pipeline.astream_answer(args.question, top_k=args.top_k, threshold=args.threshold):
            if event["type"] == "retrieval":
                result = event["result"]
                print(f"{ROUTE_TEXT[result.route]}  {result.reason}\n")
                print("答：", end="", flush=True)
            elif event["type"] == "token":
                print(event["text"], end="", flush=True)
        print()

    asyncio.run(run())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="校园文档智能问答 Agent（RAG）")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="解析文档并建立索引")
    p_build.add_argument("--dir", default=str(load_settings().docs_dir), help="文档目录")
    p_build.add_argument("--rebuild", action="store_true", help="先清空再建")
    p_build.set_defaults(func=cmd_build)

    p_ask = sub.add_parser("ask", help="单次提问")
    p_ask.add_argument("question")
    p_ask.add_argument("--top-k", type=int, default=None)
    p_ask.add_argument("--threshold", type=float, default=None)
    p_ask.add_argument("--show-candidates", action="store_true")
    p_ask.set_defaults(func=cmd_ask)

    p_chat = sub.add_parser("chat", help="连续对话")
    p_chat.add_argument("--top-k", type=int, default=None)
    p_chat.add_argument("--threshold", type=float, default=None)
    p_chat.set_defaults(func=cmd_chat)

    p_serve = sub.add_parser("serve", help="启动 Streamlit 界面")
    p_serve.add_argument("--port", type=int, default=8501)
    p_serve.set_defaults(func=cmd_serve)

    p_async = sub.add_parser("async-demo", help="演示异步流式接口")
    p_async.add_argument("question")
    p_async.add_argument("--top-k", type=int, default=None)
    p_async.add_argument("--threshold", type=float, default=None)
    p_async.set_defaults(func=cmd_async_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
