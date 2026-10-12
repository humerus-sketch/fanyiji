"""Strict Chinese presentation policy, shared by models and the posting boundary."""
from __future__ import annotations

import unicodedata


class ChineseOutputError(ValueError):
    """Non-Chinese output cannot be displayed or posted as a Chinese translation."""


def _is_han(char: str) -> bool:
    name = unicodedata.name(char, "")
    return char == "〇" or name.startswith(("CJK UNIFIED IDEOGRAPH-", "CJK COMPATIBILITY IDEOGRAPH-"))


def is_chinese_output(value: object) -> bool:
    """Allow Han, decimal digits, whitespace and normal subtitle punctuation only.

    Han characters shared by Chinese/Japanese cannot establish language by
    themselves; meaning is handled by translation. Never silently strip a name.
    """
    if not isinstance(value, str) or not value.strip():
        return False
    for char in value:
        category = unicodedata.category(char)
        if _is_han(char) or category == "Nd" or char in " \t\r\n":
            continue
        # Japanese script marks such as the prolonged sound mark are letters,
        # so punctuation allowance cannot accidentally accept them.
        if category.startswith("P") or char in "%％￥¥$℃°+-=×÷":
            continue
        return False
    return True


def ensure_chinese_output(value: str, target_lang: str) -> str:
    if target_lang == "zh" and value.strip() and not is_chinese_output(value):
        raise ChineseOutputError("译文含非中文字符，须转换为中文后才能显示或发送")
    return value
