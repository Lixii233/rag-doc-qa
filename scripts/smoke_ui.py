"""界面冒烟测试：不开浏览器，直接跑一遍 pages 逻辑。

Streamlit 官方的 `AppTest` 会把 app.py 当成脚本执行一遍，可以检查有没有异常、
渲染了哪些控件，还能模拟输入。适合在没有图形环境（或沙箱禁止启动浏览器）时做验证。

用法：python scripts/smoke_ui.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", str(ROOT / ".hf-cache"))

from streamlit.testing.v1 import AppTest  # noqa: E402


def main() -> int:
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
    app.run()

    if app.exception:
        print("[NG] 界面运行时报错：")
        for item in app.exception:
            print("   ", item.value)
        return 1

    title = app.title[0].value if app.title else "(无标题)"
    print(f"[OK] 页面标题：{title}")
    print(f"[OK] markdown 块 {len(app.markdown)} 个，sidebar 控件 {len(app.sidebar)} 个")

    labels = [getattr(w, "label", "") or getattr(w, "value", "") for w in app.sidebar]
    print("     侧栏控件：" + "、".join(str(x) for x in labels if x)[:120])

    if not app.chat_input:
        print("[!!] 没有找到 chat_input，跳过提问测试")
        return 0

    app.chat_input[0].set_value("校园卡丢了怎么补办？").run()
    if app.exception:
        print("[NG] 提问时报错：")
        for item in app.exception:
            print("   ", item.value)
        return 1

    answer = "\n".join(m.value for m in app.markdown)
    marker = "知识库原文摘录" if "知识库原文摘录" in answer else ("根据" if "根据" in answer else "")
    print(f"[OK] 提问后 markdown 长度 {len(answer)}，命中标记：{marker or '(直接回答)'}")
    print("[OK] 会话状态 history 条数：", len(app.session_state.get("history", [])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
