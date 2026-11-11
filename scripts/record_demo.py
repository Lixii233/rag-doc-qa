"""录一段演示动图：提问 → 流式作答 → 展开检索依据，用 CDP 逐帧截屏再合成 GIF。

前置：先起一个带调试端口的无头 Chrome（和项目里的截图脚本共用一套做法）

    chrome --headless=new --remote-debugging-port=9333 --user-data-dir=<临时目录> about:blank

然后：

    python scripts/record_demo.py --cdp-url http://127.0.0.1:9333 \
        --question "校园卡丢了怎么补办？" --out demo/demo.gif

不传 `--cdp-url` 时脚本会自己拉起 Chrome（部分沙箱环境不允许子进程管道，会起不来）。
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class CDP:
    def __init__(self, ws) -> None:
        self.ws = ws
        self._id = 0

    async def call(self, method: str, **params) -> dict:
        self._id += 1
        message_id = self._id
        await self.ws.send(json.dumps({"id": message_id, "method": method, "params": params}))
        while True:
            data = json.loads(await self.ws.recv())
            if data.get("id") == message_id:
                if "error" in data:
                    raise RuntimeError(f"{method} 失败：{data['error']}")
                return data.get("result", {})

    async def evaluate(self, expression: str):
        result = await self.call("Runtime.evaluate", expression=expression, returnByValue=True, awaitPromise=True)
        return result.get("result", {}).get("value")


async def _wait(cdp: CDP, expression: str, timeout: float, interval: float = 0.3) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if await cdp.evaluate(expression):
                return True
        except Exception:  # noqa: BLE001 - 导航过程中 evaluate 会短暂失败
            pass
        await asyncio.sleep(interval)
    return False


async def _shoot(cdp: CDP, fmt: str = "png", quality: int = 80) -> bytes:
    params = {"captureBeyondViewport": False}
    if fmt == "jpeg":
        params.update({"format": "jpeg", "quality": quality})
    else:
        params.update({"format": "png"})
    result = await cdp.call("Page.captureScreenshot", **params)
    return base64.b64decode(result["data"])


async def record(cdp_url: str, url: str, question: str, out: Path, frames_dir: Path,
                 size: tuple[int, int], interval: float, max_frames: int,
                 fmt: str = "png", quality: int = 80, playback_ms: int = 0) -> int:
    import websockets

    base = cdp_url.rstrip("/")
    request = urllib.request.Request(f"{base}/json/new?{url}", method="PUT")
    with urllib.request.urlopen(request, timeout=5) as response:
        page_ws = json.load(response)["webSocketDebuggerUrl"]

    frames: list[bytes] = []
    timings: list[float] = []

    async with websockets.connect(page_ws, max_size=128 * 1024 * 1024) as ws:
        cdp = CDP(ws)

        async def shoot() -> bytes:
            started = time.time()
            data = await _shoot(cdp, fmt, quality)
            timings.append(time.time() - started)
            return data

        await cdp.call("Page.enable")
        await cdp.call("Runtime.enable")
        await cdp.call("Emulation.setDeviceMetricsOverride",
                       width=size[0], height=size[1], deviceScaleFactor=1, mobile=False)

        ok = await _wait(cdp, "!!document.querySelector('[data-testid=\"stSidebar\"]')"
                              " && document.body.innerText.includes('试试这些问题')"
                              " && !document.querySelector('[data-testid=\"stStatusWidget\"]')", timeout=180)
        print(f"[{'OK' if ok else '!!'}] 首页渲染{'完成' if ok else '超时'}")
        await asyncio.sleep(1.0)
        frames.append(await shoot())  # 首帧：首页

        ready = await _wait(cdp, "!!document.querySelector('[data-testid=\"stChatInputTextArea\"]')", timeout=60)
        if not ready:
            raise SystemExit("对话框输入框一直没出现，页面结构可能变了")
        focused = await cdp.evaluate(
            "(() => { const el = document.querySelector('[data-testid=\"stChatInputTextArea\"]');"
            " if (!el) return false; el.focus(); return true; })()"
        )
        if not focused:
            raise SystemExit("没找到对话框输入框，页面结构可能变了")
        await cdp.call("Input.insertText", text=question)
        await asyncio.sleep(0.4)
        frames.append(await shoot())  # 第二帧：问题已输入
        for event in ("keyDown", "keyUp"):
            await cdp.call("Input.dispatchKeyEvent", type=event, key="Enter", code="Enter",
                           windowsVirtualKeyCode=13, nativeVirtualKeyCode=13)

        # 逐帧截图：一直录到回答结束，再多录一小段让「检索依据」稳定显示
        start = time.time()
        while len(frames) < max_frames and time.time() - start < 120:
            await asyncio.sleep(interval)
            frames.append(await shoot())
            running = await cdp.evaluate("!!document.querySelector('[data-testid=\"stStatusWidget\"]')")
            answered = await cdp.evaluate("document.body.innerText.includes('依据')")
            if not running and answered and time.time() - start > 0.8:
                # 页面会自动滚到最底部（那里是长段原文），手动滚回最后一条回答的开头再收尾
                moved = await cdp.evaluate(
                    "(() => { const nodes = document.querySelectorAll('[data-testid=\"stChatMessage\"]');"
                    " const last = nodes[nodes.length - 1];"
                    " if (!last) return 'no-msg';"
                    " let el = last.parentElement;"
                    " while (el && el !== document.body && el.scrollHeight <= el.clientHeight + 4) el = el.parentElement;"
                    " if (!el) return 'no-scroller';"
                    " const delta = last.getBoundingClientRect().top - el.getBoundingClientRect().top - 24;"
                    " el.scrollTop += delta;"
                    " return el.tagName + ' ' + Math.round(el.scrollTop) + '/' + el.scrollHeight; })()"
                )
                print(f"[..] 收尾滚动：{moved}")
                for _ in range(3):
                    await asyncio.sleep(interval)
                    frames.append(await shoot())
                break

    print(f"[OK] 采集 {len(frames)} 帧，单帧耗时 {(sum(timings) / len(timings) * 1000):.0f} ms")

    frames_dir.mkdir(parents=True, exist_ok=True)
    suffix = "jpg" if fmt == "jpeg" else "png"
    for index, data in enumerate(frames):
        (frames_dir / f"frame_{index:03d}.{suffix}").write_bytes(data)

    from PIL import Image

    images: list[Image.Image] = []
    for data in frames:
        image = Image.open(io.BytesIO(data)).convert("RGB")
        width = 900
        images.append(image.resize((width, int(image.height * width / image.width)), Image.LANCZOS))

    duration = max(80, playback_ms or int(interval * 1000))
    durations = [1200] + [duration] * (len(images) - 2) + [2600] if len(images) > 1 else [1200]
    out.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        out,
        save_all=True,
        append_images=images[1:],
        duration=durations,  # 首帧与末帧多停一会儿，看清首页和最终答案
        loop=0,
        optimize=True,
    )
    print(f"[OK] 动图已保存：{out}（{out.stat().st_size / 1024 / 1024:.1f} MB，{len(images)} 帧）")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="录制 Streamlit 问答演示动图")
    parser.add_argument("--cdp-url", default="http://127.0.0.1:9333", help="已运行的 Chrome 调试地址")
    parser.add_argument("--url", default="http://127.0.0.1:8501", help="页面地址")
    parser.add_argument("--question", default="校园卡丢了怎么补办？")
    parser.add_argument("--out", default=str(ROOT / "demo" / "demo.gif"))
    parser.add_argument("--frames-dir", default=str(ROOT / "demo" / "frames"))
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--interval", type=float, default=0.25, help="两帧之间的间隔（秒）")
    parser.add_argument("--max-frames", type=int, default=40)
    parser.add_argument("--format", choices=["png", "jpeg"], default="png", help="截屏格式，png 清晰但慢")
    parser.add_argument("--quality", type=int, default=80, help="jpeg 质量")
    parser.add_argument("--playback-ms", type=int, default=0, help="GIF 中间帧的停留时长（毫秒），默认跟随 interval")
    args = parser.parse_args(argv)

    return asyncio.run(record(
        args.cdp_url, args.url, args.question, Path(args.out), Path(args.frames_dir),
        (args.width, args.height), args.interval, args.max_frames, args.format, args.quality,
        args.playback_ms,
    ))


if __name__ == "__main__":
    raise SystemExit(main())
