# demo —— 一键演示包

给需要"看一眼就懂"的场合准备：不用配环境、不用敲命令，双击就跑；另有录屏动图和三态截图可直接贴进 PPT / README。

## 目录内容

| 文件 | 用途 |
| --- | --- |
| `start_demo.py` | 一键启动器：查依赖 → 索引为空则自动建库 → 起 Streamlit → 自动打开浏览器 |
| `demo.gif` | 真实浏览器录屏（19 帧）：首页 → 提问 → 逐字流式 → 展开检索依据 |
| `screenshots/01_home.png` | 首页：侧栏参数、21 片段 / 7 文档、向量模型与大模型状态 |
| `screenshots/02_answer.png` | 🟢 命中态：「校园卡丢了怎么补办？」最高相似度 0.719，带 `[1][2]` 引用 + 4 段依据 |
| `screenshots/03_clarify.png` | 🟡 追问态：「今天北京的天气怎么样？」0.306 低于阈值但接近临界值 |
| `screenshots/04_refuse.png` | 🔴 拒答态：「推荐几部好看的科幻电影」0.235，并列出知识库现有 7 篇文档 |
| `演示脚本.md` | 30 秒 / 90 秒讲解词、三个问题的预期输出、面试追问 Q&A、翻车预案 |

## 怎么用

双击项目根目录的 **`启动演示.bat`**（Windows），或者：

```bash
python demo/start_demo.py            # 起服务并打开浏览器
python demo/start_demo.py --rebuild  # 强制重建索引后再启动
python demo/start_demo.py --check-only  # 只做环境自检，不启动
python demo/start_demo.py --port 8600 --no-browser
```

启动器会打印：知识库文档数 / 片段数 / 建库耗时 / 向量模型 / 大模型状态，然后给出 `http://localhost:8501`。

> 需要 Python 与 `.venv` 里已装依赖（见主 [README](../README.md) 第 2 节）；缺依赖时脚本会直接告诉你哪几个没装。

## 想自己重录 GIF？

`scripts/record_demo.py` 用 Chrome DevTools 协议逐帧截屏再合成动图（Windows 沙箱下浏览器需要放宽权限）：

```bash
# 1) 先起一个带调试端口的 Chrome
chrome --headless=new --remote-debugging-port=9333 --user-data-dir=.\data\cdp_demo about:blank

# 2) 录一段：自动填问题、等流式结束、逐帧截图、合成 GIF
.venv\Scripts\python.exe scripts\record_demo.py ^
  --cdp-url http://127.0.0.1:9333 --question "校园卡丢了怎么补办？" ^
  --out demo\demo.gif --frames-dir demo\_frames --format jpeg ^
  --width 1200 --height 760 --interval 0.05 --playback-ms 130
```

`scripts/capture_ui.py` 是它的单帧版，用来出截图：
`--ask "问题" --out demo\screenshots\xx.png --width 1400 --height 1500`。
