"""轻量意图前置判断。

只做一件事：识别“根本不需要检索”的问题（寒暄、致谢、问助手本身能力）。
判定刻意保守——宁可漏判让它去检索，也不要把“校园卡怎么用”这类真问题误判成闲聊。
"""

from __future__ import annotations

_GREETING = ("你好", "您好", "hi", "hello", "嗨", "在吗", "有人吗", "早上好", "下午好", "晚上好")
_META = ("你是谁", "你能做什么", "你能干什么", "你会什么", "介绍一下你自己", "介绍下你自己")
_THANKS = ("谢谢", "多谢", "感谢", "thanks")


def normalize(question: str) -> str:
    return (question or "").strip().lower().rstrip("？?！!。.~ ")


def is_smalltalk(question: str) -> bool:
    """寒暄 / 致谢 / 问助手身份能力时返回 True（这类问题不该去查向量库）。"""
    q = normalize(question)
    if not q:
        return False
    if len(q) <= 8 and any(word in q for word in _GREETING):
        return True
    if len(q) <= 12 and any(word in q for word in _META):
        return True
    if len(q) <= 6 and any(word in q for word in _THANKS):
        return True
    return False
