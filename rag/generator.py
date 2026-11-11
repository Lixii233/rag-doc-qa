"""答案生成：提示词构造 + 大模型流式调用 + 离线摘录兜底。

三种输出通道共用同一个 `stream_answer` 接口，界面层不需要关心底层是
DeepSeek、OpenAI 兼容服务，还是没配 Key 时的离线摘录模式。
"""

from __future__ import annotations

import math
import re
import time
from collections.abc import AsyncIterator, Iterator

from .config import Settings
from .schemas import Chunk, Route, SearchResult

SYSTEM_PROMPT = (
    "你是「校园 / 个人文档智能问答助手」，负责根据给定的资料片段回答用户问题。\n"
    "必须遵守：\n"
    "1. 只依据【参考资料】作答，不得编造资料中没有的数字、时间、流程或结论。\n"
    "2. 如果资料不足以回答，直接说明缺少什么信息，不要猜测。\n"
    "3. 回答用中文，先给结论再给要点，必要时分条列出。\n"
    "4. 每个关键结论后用 [序号] 标注来源，例如 [1]，序号对应参考资料编号。\n"
    "5. 保持简洁，默认不超过 300 字。"
)

CONTEXT_BUDGET = 6000  # 拼进提示词的资料字符上限，超出则截断尾部片段

_SENTENCE_RE = re.compile(r"[^。！？!?\n]+[。！？!?]?")


# --------------------------------------------------------------------------- 提示词
def build_context(hits: list[Chunk], budget: int = CONTEXT_BUDGET) -> str:
    blocks: list[str] = []
    used = 0
    for index, chunk in enumerate(hits, start=1):
        header = f"[{index}] 《{chunk.source}》第 {chunk.page} 页（相似度 {chunk.score:.3f}）"
        body = chunk.text.strip()
        remaining = budget - used - len(header)
        if remaining <= 60:
            break
        if len(body) > remaining:
            body = body[:remaining].rstrip() + "……"
        block = f"{header}\n{body}"
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


def build_messages(question: str, hits: list[Chunk], history: list[tuple[str, str]] | None = None) -> list[dict]:
    context = build_context(hits) or "（无可用资料）"
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    for prev_q, prev_a in (history or [])[-3:]:
        messages.append({"role": "user", "content": prev_q})
        messages.append({"role": "assistant", "content": prev_a[:400]})

    messages.append({
        "role": "user",
        "content": f"【参考资料】\n{context}\n\n【问题】\n{question}\n\n请基于参考资料回答，并在结论后标注来源序号。",
    })
    return messages


# --------------------------------------------------------------------------- 大模型客户端
class LLMClient:
    """OpenAI 协议兼容的流式客户端（DeepSeek / OpenAI / 本地 vLLM 均可）。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = None
        self._async_client = None
        if settings.has_llm_key:
            from openai import AsyncOpenAI, OpenAI

            self._client = OpenAI(
                api_key=settings.llm_api_key,
                base_url=settings.llm_base_url,
                timeout=60.0,
                max_retries=1,
            )
            self._async_client = AsyncOpenAI(
                api_key=settings.llm_api_key,
                base_url=settings.llm_base_url,
                timeout=60.0,
                max_retries=1,
            )

    @property
    def available(self) -> bool:
        return self._client is not None

    @property
    def status(self) -> str:
        if self.available:
            return f"{self.settings.llm_provider} / {self.settings.llm_model}"
        return "离线摘录模式（未配置 API Key）"

    def _params(self, messages: list[dict], stream: bool) -> dict:
        return {
            "model": self.settings.llm_model,
            "messages": messages,
            "temperature": self.settings.llm_temperature,
            "max_tokens": self.settings.llm_max_tokens,
            "stream": stream,
        }

    def stream(self, messages: list[dict]) -> Iterator[str]:
        if not self.available:
            raise RuntimeError("未配置大模型 API Key")
        response = self._client.chat.completions.create(**self._params(messages, stream=True))
        for event in response:
            if not event.choices:
                continue
            delta = event.choices[0].delta
            piece = getattr(delta, "content", None)
            if piece:
                yield piece

    def complete(self, messages: list[dict]) -> str:
        return "".join(self.stream(messages))

    async def astream(self, messages: list[dict]) -> AsyncIterator[str]:
        if not self._async_client:
            raise RuntimeError("未配置大模型 API Key")
        response = await self._async_client.chat.completions.create(**self._params(messages, stream=True))
        async for event in response:
            if not event.choices:
                continue
            piece = getattr(event.choices[0].delta, "content", None)
            if piece:
                yield piece


# --------------------------------------------------------------------------- 离线摘录
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s")
_SECTION_RE = re.compile(r"^\s*[一二三四五六七八九十]+、[^。！？]{0,12}$")


def _split_sentences(text: str) -> list[str]:
    """按行切句，顺手丢掉 Markdown 标题和“一、xxx”这类栏目名，避免摘录出无信息的标题。"""
    sentences: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or _HEADING_RE.match(stripped) or _SECTION_RE.match(stripped):
            continue
        for piece in _SENTENCE_RE.findall(stripped):
            piece = piece.strip()
            if len(piece) >= 10:  # 太短的句子（如“见附件”）没有摘录价值
                sentences.append(piece)
    return sentences


def _char_bigrams(text: str) -> set[str]:
    cleaned = re.sub(r"[\s\W_]+", "", text.lower())
    return {cleaned[i:i + 2] for i in range(len(cleaned) - 1)} or {cleaned}


_LIST_MARKER_RE = re.compile(r"^\s*(?:[（(]\d{1,2}[）)]|\d{1,2}[.、])(?!\d)\s*")


def _strip_list_marker(sentence: str) -> str:
    """去掉原文的“1. ”“（2）”这类编号：摘录输出自己会加项目符号，否则会变成“- 1. …”。"""
    stripped = _LIST_MARKER_RE.sub("", sentence, count=1).strip()
    return stripped or sentence


def _semantic_scores(question: str, sentences: list[str], embedder: object | None) -> list[float] | None:
    """用向量模型给每个候选句子打分；没有模型或调用失败时返回 None 交给字面策略。"""
    if embedder is None or not sentences:
        return None
    try:
        query_vec = embedder.embed_query(question)
        rows = embedder.embed_documents(sentences)
    except Exception:  # noqa: BLE001 - 摘录是兜底路径，任何异常都退回字面打分
        return None
    query_norm = math.sqrt(sum(v * v for v in query_vec)) or 1.0
    scores: list[float] = []
    for row in rows:
        row_norm = math.sqrt(sum(v * v for v in row)) or 1.0
        scores.append(sum(a * b for a, b in zip(query_vec, row)) / (query_norm * row_norm))
    return scores


def _lexical_scores(question: str, pool: list[tuple[int, int, str, set[str], Chunk]]) -> list[float]:
    """idf 加权的 bigram 命中分：常见词降权，只在少数句子里出现的词提权。

    权重用 idf 的平方（`log((N+1)/df) ** 2`），“补办”这种只在 3 句里出现的词
    比“校园”这种句句都有的词高一个数量级，避免摘出来的都是泛泛而谈的句子。
    """
    query_grams = _char_bigrams(question)
    total = len(pool)
    frequency: dict[str, int] = {}
    for _, _, _, grams, _ in pool:
        for gram in query_grams & grams:
            frequency[gram] = frequency.get(gram, 0) + 1
    weight = {gram: math.log((total + 1) / (frequency.get(gram, 0) + 1)) ** 2 for gram in query_grams}
    # 过半句子都含有的 bigram（如“图书馆”“校园卡”）在这种问答里等同停用词，直接不计分
    common = {gram for gram in query_grams if frequency.get(gram, 0) > total * 0.5}
    if len(common) < len(query_grams):
        for gram in common:
            weight.pop(gram, None)

    scores: list[float] = []
    for _rank, _position, sentence, grams, _chunk in pool:
        matched = sum(weight.get(gram, 0.0) for gram in query_grams & grams)
        value = matched / (1.0 + math.log(1 + len(grams)) / 4)  # 轻微惩罚长句，避免一大段话霸榜
        value += 0.05 if re.search(r"\d", sentence) else 0.0  # 带数字的句子通常信息量更大
        scores.append(value)
    return scores


def extractive_answer(
    question: str,
    hits: list[Chunk],
    limit: int = 4,
    embedder: object | None = None,
) -> str:
    """无大模型时的兜底：挑与问题最贴近的原文句子，保证“有据可查”。

    排序用「语义相似度 + 关键词命中」混合打分：
    - 语义分来自向量模型（句子级 cosine），能抓住换了说法的同一件事；
    - 关键词分用 idf 加权的 bigram，「补办」「熄灯」这类少见的词权重高，
      「怎么」「校园」这类几乎句句都有的词权重低，避免摘出泛泛而谈的句子。
    """
    pool = [
        (rank, position, sentence, _char_bigrams(sentence), chunk)
        for rank, chunk in enumerate(hits)
        for position, sentence in enumerate(_split_sentences(chunk.text))
    ]
    if not pool:
        return "根据检索到的片段，暂时无法提炼出与问题直接相关的句子，建议换个问法或补充资料。"

    lexical = _lexical_scores(question, pool)
    semantic = _semantic_scores(question, [item[2] for item in pool], embedder)
    if semantic is None:
        base = lexical
    else:
        # 语义为主、字面为辅：字面分只当“同义句之间”的排序依据，
        # 权重过大时会被“图书馆”这类偶然重合的子串带偏。
        peak = max(lexical) or 1.0
        base = [0.85 * sim + 0.15 * score / peak for sim, score in zip(semantic, lexical)]

    scored: list[tuple[float, str, Chunk, int]] = []
    for index, ((rank, position, sentence, _, chunk), value) in enumerate(zip(pool, base)):
        value += 0.02 if position == 0 else 0.0  # 片段首句往往是主题句
        value -= rank * 0.02  # 排序越靠后的片段越不优先
        scored.append((value, sentence, chunk, index))

    scored.sort(key=lambda item: -item[0])
    picked: list[tuple[int, str, Chunk]] = []
    seen: set[str] = set()
    for _, sentence, chunk, index in scored:
        if sentence in seen:
            continue
        if len(picked) < limit:
            picked.append((index, sentence, chunk))
            seen.add(sentence)

    if not picked:
        return "根据检索到的片段，暂时无法提炼出与问题直接相关的句子，建议换个问法或补充资料。"

    picked.sort(key=lambda item: item[0])  # 按原文顺序排列，读起来才像一段说明而不是打散的关键词
    citation_index = {chunk.chunk_id: i + 1 for i, chunk in enumerate(hits)}
    lines = ["（当前未配置大模型 API Key，以下为知识库原文摘录，供参考）", ""]
    for _, sentence, chunk in picked:
        lines.append(f"- {_strip_list_marker(sentence)} [{citation_index.get(chunk.chunk_id, 1)}]")
    return "\n".join(lines)


# --------------------------------------------------------------------------- 统一入口
class AnswerGenerator:
    def __init__(self, settings: Settings, embedder: object | None = None) -> None:
        self.settings = settings
        self.embedder = embedder  # 离线摘录时用它做句子级语义排序
        self.llm = LLMClient(settings)

    @property
    def status(self) -> str:
        return self.llm.status

    # ---------------- 分流话术 ----------------
    def clarify_message(self, result: SearchResult) -> str:
        best = result.candidates[0] if result.candidates else None
        hint = ""
        if best:
            hint = (
                f"\n\n我找到的最接近的资料是《{best.source}》第 {best.page} 页"
                f"（相似度 {best.score:.3f}，低于阈值 {result.threshold:.2f}）：\n\n> {best.text[:120]}……"
            )
        return (
            "我不太确定你的具体场景，直接回答可能会有偏差。能不能再补充一点信息？"
            "比如问的是哪一项业务、哪个时间范围，或者换成更具体的说法。"
            f"{hint}"
        )

    def refuse_message(self, result: SearchResult, sources: list[tuple[str, int]] | None = None) -> str:
        listed = sources or []
        names = "、".join(f"《{name}》" for name, _ in listed[:6])
        if len(listed) > 6:
            names = f"{names} 等 {len(listed)} 篇文档"
        scope = f"目前知识库收录了：{names}。" if names else "目前知识库还没有收录文档。"
        best = result.candidates[0] if result.candidates else None
        detail = f"（最高相似度仅 {best.score:.3f}，未达到 {result.threshold:.2f} 的阈值）" if best else ""
        return (
            f"这个问题在现有资料里找不到依据{detail}，为了避免编造，我先不做回答。\n\n"
            f"{scope}\n\n你可以：补充上传相关文档后重建索引，或者换个更贴近资料的说法再问一次。"
        )

    # ---------------- 同步流式 ----------------
    def stream_answer(
        self,
        question: str,
        result: SearchResult,
        history: list[tuple[str, str]] | None = None,
        sources: list[tuple[str, int]] | None = None,
    ) -> Iterator[str]:
        if result.route == Route.CLARIFY:
            yield from _fake_stream(self.clarify_message(result))
            return
        if result.route == Route.REFUSE:
            yield from _fake_stream(self.refuse_message(result, sources))
            return

        if result.route == Route.GENERAL:
            yield from _fake_stream("资料中没有直接依据，以下回答基于通用知识，仅供参考：\n\n")

        if not self.llm.available or not result.hits:
            yield from _fake_stream(extractive_answer(question, result.hits, embedder=self.embedder))
            return

        try:
            yield from self.llm.stream(build_messages(question, result.hits, history))
        except Exception as exc:  # noqa: BLE001 - 网络/鉴权异常时降级，不让界面白屏
            yield f"\n\n[调用大模型失败：{type(exc).__name__}: {exc}]\n已降级为摘录模式：\n\n"
            yield from _fake_stream(extractive_answer(question, result.hits, embedder=self.embedder))

    # ---------------- 异步流式 ----------------
    async def astream_answer(
        self,
        question: str,
        result: SearchResult,
        history: list[tuple[str, str]] | None = None,
        sources: list[tuple[str, int]] | None = None,
    ) -> AsyncIterator[str]:
        if result.route in (Route.CLARIFY, Route.REFUSE) or not self.llm.available or not result.hits:
            for piece in self.stream_answer(question, result, history, sources):
                yield piece
            return

        if result.route == Route.GENERAL:
            yield "资料中没有直接依据，以下回答基于通用知识，仅供参考：\n\n"

        try:
            async for piece in self.llm.astream(build_messages(question, result.hits, history)):
                yield piece
        except Exception as exc:  # noqa: BLE001
            yield f"\n\n[调用大模型失败：{type(exc).__name__}: {exc}]\n已降级为摘录模式：\n\n"
            async for piece in _afake_stream(extractive_answer(question, result.hits, embedder=self.embedder)):
                yield piece


def _fake_stream(text: str, chunk_size: int = 4, delay: float = 0.02) -> Iterator[str]:
    """把整段文本切成小片吐出来，让离线模式也有打字机效果（约 200 字/秒）。"""
    for i in range(0, len(text), chunk_size):
        yield text[i:i + chunk_size]
        time.sleep(delay)


async def _afake_stream(text: str, chunk_size: int = 4, delay: float = 0.02):
    import asyncio

    for i in range(0, len(text), chunk_size):
        yield text[i:i + chunk_size]
        await asyncio.sleep(delay)
