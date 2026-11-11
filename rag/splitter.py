"""递归字符切分器（RecursiveCharacterTextSplitter 思路）。

设计要点
--------
1. **按语义层级递归**：优先用空行切，其次换行，再次中文句末标点（。！？；），
   最后退化为按字符硬切。这样能尽量保证一个 chunk 内部语义完整。
2. **中文友好**：标点作为分隔符时保留在前一段末尾（lookbehind 切分），
   不会出现“。下一句”这种以标点开头的片段。
3. **带偏移量**：同时返回每个 chunk 在原文中的 [start, end)，便于前端高亮原文位置。
4. **重叠**：相邻 chunk 之间保留 `chunk_overlap` 个字符，避免答案被切断在边界上。

默认参数与项目说明一致：chunk_size=500，chunk_overlap=50。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

DEFAULT_SEPARATORS: tuple[str, ...] = (
    "\n\n",     # 段落
    "\n",       # 换行
    "。", "！", "？", "；",   # 中文句末标点
    "，", "、",              # 中文句中标点
    ". ", "! ", "? ", "; ",  # 英文句末标点
    ", ",                    # 英文句中标点
    " ",                     # 空格
    "",                      # 逐字符兜底
)


@dataclass
class _Piece:
    """切分过程中的最小片段，保留原文偏移。"""

    start: int
    text: str

    @property
    def end(self) -> int:
        return self.start + len(self.text)


class RecursiveCharacterTextSplitter:
    """把长文本递归地切成不超过 `chunk_size` 的片段。"""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: tuple[str, ...] | list[str] | None = None,
        strip_whitespace: bool = True,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size 必须大于 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap 不能为负数")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap 必须小于 chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = tuple(separators) if separators else DEFAULT_SEPARATORS
        self.strip_whitespace = strip_whitespace

    # ------------------------------------------------------------------ 公共 API
    def split_text(self, text: str) -> list[str]:
        return [piece.text for piece in self.split_text_with_offsets(text)]

    def split_text_with_offsets(self, text: str) -> list[_Piece]:
        if not text or not text.strip():
            return []
        pieces = self._split(text, 0, self.separators)
        merged = self._merge(pieces)
        if not self.strip_whitespace:
            return merged
        return [p for p in merged if p.text.strip()]

    # ------------------------------------------------------------------ 递归切分
    def _split(self, text: str, base: int, separators: tuple[str, ...]) -> list[_Piece]:
        """在当前文本上挑一个分隔符切分，过长的片段继续用下一级分隔符递归。"""
        separator = "" if separators else ""
        next_separators: tuple[str, ...] = ()
        for idx, candidate in enumerate(separators):
            if candidate == "":
                separator, next_separators = "", separators[idx + 1:]
                break
            if candidate in text:
                separator, next_separators = candidate, separators[idx + 1:]
                break

        raw_pieces = self._split_on_separator(text, base, separator)

        good: list[_Piece] = []
        final: list[_Piece] = []
        for piece in raw_pieces:
            if len(piece.text) <= self.chunk_size:
                good.append(piece)
                continue
            # 太长了：先把已攒下的片段合并输出，再对这一段用更细的分隔符递归
            if good:
                final.extend(self._merge(good))
                good = []
            if separator == "" or not next_separators:
                # 已经退到逐字符，无法再切：只能硬切
                final.extend(self._hard_split(piece))
            else:
                final.extend(self._split(piece.text, piece.start, next_separators))
        if good:
            final.extend(self._merge(good))
        return final

    def _split_on_separator(self, text: str, base: int, separator: str) -> list[_Piece]:
        """按分隔符切分，分隔符保留在前一个片段末尾。"""
        if separator == "":
            return [_Piece(base + i, ch) for i, ch in enumerate(text)]

        pattern = f"(?<={re.escape(separator)})"
        parts = [p for p in re.split(pattern, text) if p != ""]
        pieces: list[_Piece] = []
        cursor = 0
        for part in parts:
            pieces.append(_Piece(base + cursor, part))
            cursor += len(part)
        return pieces

    def _hard_split(self, piece: _Piece) -> list[_Piece]:
        """兜底：按固定窗口硬切，保证每个片段不超过 chunk_size。"""
        step = max(1, self.chunk_size - self.chunk_overlap)
        return [
            _Piece(piece.start + i, piece.text[i:i + self.chunk_size])
            for i in range(0, len(piece.text), step)
        ]

    # ------------------------------------------------------------------ 合并 + 重叠
    def _merge(self, pieces: list[_Piece]) -> list[_Piece]:
        """把相邻小片段拼成不超过 chunk_size 的块，块之间保留 overlap。"""
        if not pieces:
            return []

        chunks: list[_Piece] = []
        window: list[_Piece] = []
        total = 0

        for piece in pieces:
            length = len(piece.text)
            if window and total + length > self.chunk_size:
                chunks.append(_merge_window(window))
                # 从窗口头部丢弃片段，直到剩余长度 <= overlap
                while window and (total > self.chunk_overlap or total + length > self.chunk_size):
                    total -= len(window[0].text)
                    window.pop(0)
            window.append(piece)
            total += length

        if window:
            chunks.append(_merge_window(window))
        return chunks


def _merge_window(window: list[_Piece]) -> _Piece:
    start = window[0].start
    text = "".join(p.text for p in window)
    return _Piece(start, text)


def split_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    separators: tuple[str, ...] | list[str] | None = None,
) -> list[str]:
    """便捷函数：一行完成切分。"""
    splitter = RecursiveCharacterTextSplitter(chunk_size, chunk_overlap, separators)
    return splitter.split_text(text)


if __name__ == "__main__":  # 快速演示：python -m rag.splitter
    demo = "第一段内容。" * 60 + "\n\n" + "第二段内容，包含一些细节。" * 40
    result = RecursiveCharacterTextSplitter().split_text(demo)
    print(f"原文 {len(demo)} 字，切成 {len(result)} 段")
    for i, chunk in enumerate(result, 1):
        print(f"--- chunk {i} ({len(chunk)} 字) ---")
        print(chunk[:80].replace("\n", " ") + ("..." if len(chunk) > 80 else ""))
