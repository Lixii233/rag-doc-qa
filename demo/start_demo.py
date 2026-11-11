"""演示启动器：检查依赖 → 没索引就自动建库 → 起 Streamlit → 自动打开浏览器。

双击项目根目录的 `启动演示.bat` 即可，等价于：

    python demo/start_demo.py                     # 默认 8501 端口，自动开浏览器
    python demo/start_demo.py --check-only        # 只做环境自检，不起服务
    python demo/start_demo.py --rebuild           # 强制重建索引后再起
"""

from __future__ import annotations

import argparse
import importlib.util
import socket
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BANNER = r"""
+--------------------------------------------------------------+
|        校园 / 个人文档智能问答 Agent  ·  演示模式             |
|        PDF/Word/文本 -> 切片 -> BGE 向量 -> Chroma -> LLM     |
+--------------------------------------------------------------+
"""


def health_ok(port: int, timeout: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=timeout) as response:
            return response.status == 200
    except Exception:  # noqa: BLE001 - 端口没起来就是 False
        return False


def port_free(port: int) -> bool:
    with socket.socket() as sock:
        return sock.connect_ex(("127.0.0.1", port)) != 0


def check_deps() -> list[str]:
    missing: list[str] = []
    for module, package in (
        ("streamlit", "streamlit"),
        ("chromadb", "chromadb"),
        ("sentence_transformers", "sentence-transformers"),
        ("openai", "openai"),
        ("pypdf", "pypdf"),
        ("docx", "python-docx"),
    ):
        # find_spec 只查安装位置、不真正 import：避免为了自检把 torch 加载一遍（省十几秒）
        try:
            found = importlib.util.find_spec(module) is not None
        except Exception:  # noqa: BLE001
            found = False
        if not found:
            missing.append(package)
    return missing


def build_index_if_needed(rebuild: bool) -> None:
    from rag import RAGPipeline, load_settings

    settings = load_settings()
    pipeline = RAGPipeline(settings)
    if rebuild or pipeline.store.count() == 0:
        print("[..] 正在建立索引（首次运行需要加载向量模型，约 30 秒）…")
        docs = []
        for directory in (settings.docs_dir, settings.samples_dir):
            loaded, errors = pipeline.load_directory(directory)
            docs.extend(loaded)
            for name, reason in errors:
                print(f"     [跳过] {name}: {reason}")
        if not docs:
            print(f"[!!] {settings.docs_dir} 下没有可用的文档，先放几篇 .md / .pdf / .docx 再试")
            raise SystemExit(1)
        stats = pipeline.index_docs(docs, replace=True)
        print(f"[OK] 索引完成：{stats['documents']} 篇文档 → {stats['chunks']} 个片段，耗时 {stats['seconds']}s")
    else:
        sources = pipeline.store.sources()
        print(f"[OK] 已有索引：{pipeline.store.count()} 个片段 / {len(sources)} 篇文档")
    print(f"[OK] 向量模型：{pipeline.embedder.status}")
    print(f"[OK] 大模型：{pipeline.generator.status}")
    if not settings.has_llm_key:
        print("[..] 未配置 API Key，将使用离线摘录模式（答案摘自原文，仍可完整演示）")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="演示启动器")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    parser.add_argument("--check-only", action="store_true", help="只做自检，不启动服务")
    parser.add_argument("--rebuild", action="store_true", help="强制重建索引")
    args = parser.parse_args(argv)

    print(BANNER)
    url = f"http://localhost:{args.port}"
    if health_ok(args.port):
        # 已经有一个演示窗口在跑：不要再建库（Chroma 目录被占用会卡住），直接开浏览器
        print(f"[OK] 检测到 {url} 已在运行，直接打开浏览器")
        if not args.no_browser:
            webbrowser.open(url)
        return 0

    missing = check_deps()
    if missing:
        print("[!!] 缺少依赖：" + "、".join(missing))
        print("     安装：.venv\\Scripts\\python.exe -m pip install -r requirements.txt")
        return 1
    print("[OK] 依赖齐全")

    build_index_if_needed(args.rebuild)
    if args.check_only:
        print("\n自检完成。去掉 --check-only 即可启动演示。")
        return 0

    if not port_free(args.port):
        print(f"[!!] 端口 {args.port} 被其他程序占用，换一个：--port 8502")
        return 1

    import subprocess

    command = [
        sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"),
        "--server.port", str(args.port),
        "--server.headless", "true",
        "--server.fileWatcherType", "none",
        "--browser.gatherUsageStats", "false",
    ]
    print(f"[..] 启动界面：{url}")
    process = subprocess.Popen(command, cwd=str(ROOT))

    def open_when_ready() -> None:
        for _ in range(120):
            if health_ok(args.port):
                print(f"[OK] 界面已就绪：{url}    （按 Ctrl+C 停止）")
                if not args.no_browser:
                    webbrowser.open(url)
                return
            time.sleep(0.5)
        print("[!!] 界面启动超时，请看上面的日志")

    threading.Thread(target=open_when_ready, daemon=True).start()
    try:
        return process.wait()
    except KeyboardInterrupt:
        process.terminate()
        print("\n已停止演示服务")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
