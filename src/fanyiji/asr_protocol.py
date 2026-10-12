"""Fail closed on recognizer control tokens reaching ordinary subtitle text."""
from __future__ import annotations

import html
import re


_MARKER = re.compile(r"<\s*/?\s*asr_text\b|<\|(?:im_start|im_end|endoftext|audio[^<>\r\n]*)\|>", re.IGNORECASE)
_NULL_PREFIX = re.compile(r"^\s*[【\[\"']?\s*(?:language|语言)\s*[:：]?\s*(?:None|null)\b", re.IGNORECASE)


class AsrProtocolError(RuntimeError):
    """A malformed recognition result; never a generation-limit retry."""


def contains_asr_protocol(text: object) -> bool:
    if not isinstance(text, str):
        return False
    normalized = html.unescape(text)
    return bool(_MARKER.search(normalized) or _NULL_PREFIX.match(normalized))


def require_subtitle_text(text: str) -> str:
    if contains_asr_protocol(text):
        raise AsrProtocolError("识别控制标记混入正文，该片段已拒绝翻译或外发。")
    return text
