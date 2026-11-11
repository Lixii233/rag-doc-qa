"""用 Chrome DevTools 协议给 Streamlit 页面截图（可顺带提一个问题再截）。

命令行里 `chrome --screenshot=xx.png` 对 Streamlit 这种「页面加载完才开始渲染」的
单页应用不生效，截出来常常是空白。这里改成：启动无头 Chrome → 连 CDP → 等页面真正
渲染完（轮询 DOM 出现侧栏）→ 截图；加 `--ask` 时还会往对话框里输入问题并回车，截到回答。

用法：
    python scripts/capture_ui.py --out data/ui_home.png
    python scripts/capture_ui.py --out data/ui_answer.png --ask "校园卡丢了怎么补办？"
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CHROME_CANDIDATES = [
    Path(os.environ.get("PROGRAMFILES", r"C:\Program Files")) / "Google/Chrome/Application/chrome.exe",
    Path(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")) / "Google/Chrome/Application/chrome.exe",
    Path(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")) / "Microsoft/Edge/Application/msedge.exe",
]


def find_browser() -> Path:
    for path in CHROME_CANDIDATES:
        if path.exists():
            return path
    raise SystemExit("没找到 Chrome / Edge，无法截图")


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class CDP:
    """极简 CDP 客户端：只实现我们需要的 send / recv。"""

    def __init__(self, ws) -> None:
        self.ws = ws
        self._id = 0

    async def call(self, method: str, **params) -> dict:
        self._id += 1
        message_id = self._id
        await self.ws.send(json.dumps({"id": message_id, "method": method, "params": params}))
        while True:
            raw = await self.ws.recv()
            data = json.loads(raw)
            if data.get("id") == message_id:
                if "error" in data:
                    raise RuntimeError(f"{method} 失败：{data['error']}")
                return data.get("result", {})

    async def evaluate(self, expression: str):
        result = await self.call(
            "Runtime.evaluate", expression=expression, returnByValue=True, awaitPromise=True
        )
        return result.get("result", {}).get("value")


async def wait_for(ws, cdp: CDP, expression: str, timeout: float, interval: float = 0.5) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if await cdp.evaluate(expression):
                return True
        except Exception:  # noqa: BLE001 - 页面还在导航时 evaluate 会短暂失败
            pass
        await asyncio.sleep(interval)
    return False


async def capture(
    url: str,
    out: Path,
    question: str | None,
    wait: float,
    cdp_url: str | None = None,
    size: tuple[int, int] = (1500, 1150),
) -> None:
    import websockets

    if cdp_url:
        # 连接一个已经在运行的 Chrome（`chrome --remote-debugging-port=9333`），
        # 用在沙箱不允许本进程再拉起浏览器子进程的环境里。
        base = cdp_url.rstrip("/")
        request = urllib.request.Request(f"{base}/json/new?{url}", method="PUT")
        with urllib.request.urlopen(request, timeout=5) as response:
            page_ws = json.load(response)["webSocketDebuggerUrl"]

        async with websockets.connect(page_ws, max_size=64 * 1024 * 1024) as ws:
            await _shoot(ws, out, question, wait, size)
        return

    browser = find_browser()
    port = free_port()
    profile = Path(tempfile.mkdtemp(prefix="cdp-profile-"))
    log_path = profile / "chrome.log"
    log_file = log_path.open("w", encoding="utf-8", errors="replace")
    chrome = subprocess.Popen(
        [
            str(browser),
            "--headless=new",
            "--disable-gpu",
            "--no-first-run",
            "--no-default-browser-check",
            "--hide-scrollbars",
            f"--remote-debugging-port={port}",
            f"--user-data-dir={profile}",
            f"--window-size={size[0]},{size[1]}",
            "about:blank",
        ],
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )

    try:
        endpoint = f"http://127.0.0.1:{port}/json/version"
        for _ in range(80):
            try:
                with urllib.request.urlopen(endpoint, timeout=1) as response:
                    browser_ws = json.load(response)["webSocketDebuggerUrl"]
                break
            except Exception:  # noqa: BLE001
                if chrome.poll() is not None:
                    break
                await asyncio.sleep(0.3)
        else:
            browser_ws = None
        if "browser_ws" not in locals() or not browser_ws:
            log_file.flush()
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-800:]
            raise SystemExit(f"Chrome 没起来（调试端口 {port} 不可用）。日志：\n{tail}")

        # 新建一个标签页，绑定到目标页面
        request = urllib.request.Request(f"http://127.0.0.1:{port}/json/new?{url}", method="PUT")
        with urllib.request.urlopen(request, timeout=5) as response:
            page_ws = json.load(response)["webSocketDebuggerUrl"]

        async with websockets.connect(page_ws, max_size=64 * 1024 * 1024) as ws:
            await _shoot(ws, out, question, wait, size)
    finally:
        chrome.terminate()
        try:
            chrome.wait(timeout=10)
        except subprocess.TimeoutExpired:
            chrome.kill()
        log_file.close()
        shutil.rmtree(profile, ignore_errors=True)


async def _shoot(ws, out: Path, question: str | None, wait: float, size: tuple[int, int]) -> None:
    """等页面渲染 →（可选）提一个问题 → 截图。"""
    cdp = CDP(ws)
    await cdp.call("Page.enable")
    await cdp.call("Runtime.enable")
    await cdp.call(
        "Emulation.setDeviceMetricsOverride",
        width=size[0], height=size[1], deviceScaleFactor=1, mobile=False,
    )

    rendered = await wait_for(
        ws,
        cdp,
        "!!document.querySelector('[data-testid=\"stSidebar\"]') && !document.querySelector('[data-testid=\"stStatusWidget\"]')",
        timeout=wait,
    )
    if not rendered:
        print("[!!] 等待渲染超时，仍尝试截图")
    await asyncio.sleep(1.5)

    if question:
        focused = await cdp.evaluate(
            "(() => { const el = document.querySelector('[data-testid=\"stChatInputTextArea\"]');"
            " if (!el) return false; el.focus(); return true; })()"
        )
        if not focused:
            raise SystemExit("没找到对话框输入框")
        await cdp.call("Input.insertText", text=question)
        await asyncio.sleep(0.3)
        for event in ("keyDown", "keyUp"):
            await cdp.call(
                "Input.dispatchKeyEvent", type=event, key="Enter", code="Enter",
                windowsVirtualKeyCode=13, nativeVirtualKeyCode=13,
            )
        done = await wait_for(
            ws,
            cdp,
            "!document.querySelector('[data-testid=\"stStatusWidget\"]')"
            " && (() => { const m = document.querySelectorAll('[data-testid=\"stChatMessage\"]');"
            " return m.length > 0 && !!m[m.length - 1].querySelector('details'); })()",
            timeout=min(wait, 40),
        )
        if not done:
            print("[!!] 等待回答超时，仍尝试截图")
        await asyncio.sleep(1.0)

    result = await cdp.call("Page.captureScreenshot", format="png", captureBeyondViewport=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(base64.b64decode(result["data"]))
    print(f"[OK] 截图已保存：{out}（{out.stat().st_size / 1024:.0f} KB）")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="给 Streamlit 页面截图")
    parser.add_argument("--url", default="http://127.0.0.1:8501", help="页面地址")
    parser.add_argument("--out", default=str(ROOT / "data" / "ui.png"), help="输出 PNG 路径")
    parser.add_argument("--ask", default=None, help="截图前先提一个问题")
    parser.add_argument("--wait", type=float, default=120, help="等待页面渲染的秒数上限")
    parser.add_argument("--width", type=int, default=1500, help="视口宽度")
    parser.add_argument("--height", type=int, default=1150, help="视口高度（调大可一屏截到完整回答）")
    parser.add_argument(
        "--cdp-url",
        default=None,
        help="连接已在运行的 Chrome，例如 http://127.0.0.1:9333（不自己拉起浏览器）",
    )
    args = parser.parse_args(argv)

    asyncio.run(capture(args.url, Path(args.out), args.ask, args.wait, args.cdp_url, (args.width, args.height)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
