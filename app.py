"""Streamlit 界面：上传文档 → 建库 → 对话式问答（流式输出 + 检索依据）。

启动：streamlit run app.py
"""

from __future__ import annotations

import time

import streamlit as st

from rag import RAGPipeline, load_settings
from rag.schemas import Route

st.set_page_config(page_title="校园文档智能问答 Agent", page_icon="📚", layout="wide")

ROUTE_BADGE = {
    Route.ANSWER: ("🟢", "高相关，基于资料作答"),
    Route.CLARIFY: ("🟡", "相关度偏低，已转为追问"),
    Route.REFUSE: ("🔴", "知识库未收录，已拒答"),
    Route.GENERAL: ("🔵", "无资料依据，通用模式作答"),
}


@st.cache_resource(show_spinner="正在加载向量模型与索引…")
def get_pipeline() -> RAGPipeline:
    return RAGPipeline(load_settings())


def init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "history" not in st.session_state:  # [(问题, 回答)]，作为多轮上下文喂给大模型
        st.session_state.history = []
    if "last_result" not in st.session_state:
        st.session_state.last_result = None


def human_size(count: int) -> str:
    return f"{count / 1000:.1f}k" if count >= 1000 else str(count)


def render_sources(hits, expanded: bool = False) -> None:
    if not hits:
        return
    with st.expander(f"📎 检索依据（{len(hits)} 段）", expanded=expanded):
        # 展开时用一个定高滚动区域：否则四段原文会把页面拉得很长，回答被顶出视口
        area = st.container(height=360) if expanded else st.container()
        with area:
            for i, chunk in enumerate(hits, start=1):
                st.markdown(
                    f"**[{i}] 《{chunk.source}》第 {chunk.page} 页** · 相似度 `{chunk.score:.3f}`"
                )
                st.caption(chunk.text[:260] + ("……" if len(chunk.text) > 260 else ""))
                st.divider()


def index_with_progress(pipeline: RAGPipeline, files=None, rebuild: bool = False) -> None:
    bar = st.progress(0.0, text="准备中…")

    def progress(message: str, ratio: float) -> None:
        bar.progress(min(max(ratio, 0.0), 1.0), text=message)

    try:
        if files:
            payload = [(f.name, f.getvalue()) for f in files]
            stats = pipeline.index_uploads(payload, replace=False, progress=progress)
        else:
            stats = pipeline.index_directory(replace=rebuild, progress=progress)
    except Exception as exc:  # noqa: BLE001
        bar.empty()
        st.error(f"建库失败：{type(exc).__name__}: {exc}")
        return

    bar.empty()
    st.success(
        f"索引完成：{stats.get('documents', 0)} 篇文档 / {stats.get('chunks', 0)} 个片段，"
        f"用时 {stats.get('seconds', 0)}s"
    )
    for name, reason in stats.get("errors", []):
        st.warning(f"跳过 {name}：{reason}")


def sidebar(pipeline: RAGPipeline) -> dict:
    settings = pipeline.settings
    with st.sidebar:
        st.title("📚 校园文档问答")
        st.caption("PDF / Word / 文本 → 切片 → BGE 向量 → Chroma 检索 → 大模型作答")

        st.subheader("1. 添加文档")
        uploaded = st.file_uploader(
            "上传文件（可多选）",
            type=["pdf", "docx", "txt", "md"],
            accept_multiple_files=True,
            help="上传后点“建立索引”，文件会保存到 data/uploads 并入库。",
        )
        col_a, col_b = st.columns(2)
        with col_a:
            build = st.button("建立索引", type="primary", use_container_width=True)
        with col_b:
            rebuild = st.button("重建全部", use_container_width=True)
        if st.button("清空索引", use_container_width=True):
            pipeline.reset()
            st.session_state.messages = []
            st.success("已清空索引")
            time.sleep(0.4)
            st.rerun()

        if build:
            index_with_progress(pipeline, files=uploaded)
        if rebuild:
            index_with_progress(pipeline, rebuild=True)

        st.subheader("2. 检索参数")
        top_k = st.slider("召回片段数 top_k", 1, 8, settings.top_k, help="每次送给大模型的片段数量")
        threshold = st.slider(
            "相似度阈值", 0.0, 1.0, float(settings.similarity_threshold), 0.01,
            help="top1 低于该值时不直接作答：略低则追问，明显偏低则拒答。",
        )

        st.subheader("3. 运行状态")
        stats = pipeline.stats()
        col1, col2 = st.columns(2)
        col1.metric("知识库片段", human_size(stats["chunks"]))
        col2.metric("文档数", len(stats["sources"]))
        st.caption(f"向量模型：`{stats['embedder']}`")
        st.caption(f"大模型：`{stats['llm']}`")
        if stats["embedder_fallback"]:
            st.info("当前使用离线哈希向量（未加载到 BGE 模型），语义检索能力有限。", icon="⚠️")

        if stats["sources"]:
            with st.expander("📄 知识库文档"):
                for name, count in stats["sources"]:
                    st.caption(f"· {name}（{count} 段）")

        with st.expander("🔍 调试：候选片段分数"):
            if st.session_state.last_result:
                result = st.session_state.last_result
                st.caption(f"route = `{result.route.value}` · {result.reason}")
                st.dataframe(
                    [
                        {
                            "排名": i,
                            "相似度": round(c.score, 4),
                            "文档": c.source,
                            "页码": c.page,
                            "片段": c.text[:60],
                        }
                        for i, c in enumerate(result.candidates, start=1)
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.caption("先提一个问题。")

        if not settings.has_llm_key:
            st.warning(
                "未检测到 API Key，当前为**离线摘录模式**：答案直接摘自原文，不会调用大模型。\n\n"
                "在项目根目录创建 `.env` 并填入 `DEEPSEEK_API_KEY=sk-...` 后重启即可切换为大模型生成。",
                icon="🔑",
            )

    return {"top_k": top_k, "threshold": threshold}


def render_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                route: Route | None = message.get("route")
                if route:
                    icon, label = ROUTE_BADGE[route]
                    st.caption(f"{icon} {label}")
                st.markdown(message["content"])
                render_sources(message.get("hits", []))
            else:
                st.markdown(message["content"])


def main() -> None:
    init_state()
    pipeline = get_pipeline()
    params = sidebar(pipeline)

    st.title("基于 RAG 的校园 / 个人文档智能问答 Agent")
    st.caption(
        "把分散的校园通知、制度文件、个人笔记丢进知识库，用自然语言直接提问；"
        "检索不到依据时会主动追问或拒答，而不是硬编一个答案。"
    )

    render_history()

    if not st.session_state.messages:
        st.markdown("#### 试试这些问题")
        samples = [
            "图书馆的开放时间和借阅期限是怎么规定的？",
            "宿舍晚归和用电有什么要求？",
            "国家励志奖学金需要满足哪些条件？",
            "校园卡丢了怎么补办？",
        ]
        cols = st.columns(2)
        for i, sample in enumerate(samples):
            if cols[i % 2].button(sample, use_container_width=True, key=f"sample_{i}"):
                st.session_state.pending = sample
                st.rerun()

    question = st.chat_input("输入你的问题，例如：图书馆借书能借多久？")
    if not question and st.session_state.get("pending"):
        question = st.session_state.pop("pending")

    if question:
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("assistant"):
            events = pipeline.stream_answer(
                question,
                top_k=params["top_k"],
                threshold=params["threshold"],
                history=st.session_state.history[-3:],
            )
            first = next(events)
            result = first["result"]
            st.session_state.last_result = result

            icon, label = ROUTE_BADGE[result.route]
            st.caption(f"{icon} {label} · {result.reason}")

            placeholder = st.empty()
            text = ""
            for event in events:
                if event["type"] == "token":
                    text += event["text"]
                    placeholder.markdown(text + "▌")
            placeholder.markdown(text)

            render_sources(result.hits, expanded=result.route == Route.ANSWER)
            if result.route in (Route.CLARIFY, Route.REFUSE):
                st.caption("提示：换一种更具体的问法，或先在左侧补充相关文档。")

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": text,
                "route": result.route,
                "hits": result.hits,
            }
        )
        st.session_state.history.append((question, text))


if __name__ == "__main__":
    main()
