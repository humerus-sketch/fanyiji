"""Small source-text history and conservative incomplete-sentence hints.

Hints request a bounded wait or one local repair; they are not grammatical
truth and must never discard a short reply or wait indefinitely for a sentence.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import re
import threading
import time
from typing import Callable

from .livestream_prompt import CONTEXT_CHARACTERS


MAX_CONTEXT_CHARACTERS = CONTEXT_CHARACTERS
MAX_CONTEXT_TOKENS = 256
_ENDING_PUNCTUATION = re.compile(r"[\s。．.!！?？、,，;；:：…\"'「」『』（）()【】\[\]]+$")
_SHORT_REPLIES = frozenset({
    "はい", "いいえ", "ええ", "うん", "ううん", "そう", "そうです", "そうですね",
    "なるほど", "了解", "了解です", "ありがとう", "ありがとうございます", "もしもし",
    "こんにちは", "こんばんは", "おはよう", "おはようございます", "そうなの", "そうなんだ",
    "何の", "誰の", "どこの",
    "是", "不是", "是的", "否", "好", "好的", "嗯", "对", "不对", "没有", "有",
    "可以", "不可以", "行", "不行", "谢谢", "明白", "知道了", "收到", "不用", "不了",
    "谁的", "为什么", "什么", "怎么",
})
_JA_OPEN_ENDINGS = (
    "けれども", "けれど", "けど", "ので", "のに", "だったら", "なら", "という", "っていう",
    "なければ", "れば", "たら", "だら",
    # These particular tails can precede a negation in the next ASR segment.
    # A plain 'そう' remains a valid short answer; do not infer missing words.
    "別にそう", "別にそういう",
    "たとえば", "例えば", "それから", "それで", "そして", "つまり", "でも", "まず",
    "が", "を", "に", "で", "と", "の", "は",
)
_ZH_OPEN_ENDINGS = (
    "如果", "假如", "要是", "因为", "由于", "虽然", "尽管", "但是", "不过", "所以",
    "然后", "例如", "比如", "也就是说", "至于", "关于", "以及", "的", "和", "与",
)
_ZH_CONDITION_PREFIXES = ("如果", "假如", "要是", "除非")
_ZH_MAIN_CLAUSE_MARKERS = ("就", "那么", "则", "否则", "才", "请")
_CLAUSE_PUNCTUATION = re.compile(r"[。!！?？、,，;；:：\r\n]|(?<!\d)[.．]|[.．](?!\d)")
_QUOTES_OR_BRACKETS = frozenset("\"'「」『』“”‘’（）()【】[]{}〈〉《》")


def _explicit_zh_condition_fragment(value: str) -> bool:
    # This adds explicit condition-only candidates, not all comma-ended text.
    # Broad main-clause blockers favor missed joins over swallowing a complete
    # conditional instruction, including an unpunctuated '如果…请…'.
    return (value.startswith(_ZH_CONDITION_PREFIXES)
            and not _CLAUSE_PUNCTUATION.search(value)
            and not any(marker in value for marker in _ZH_MAIN_CLAUSE_MARKERS))


def bounded_source_context(text: str, *, token_counter: Callable[[str], int] | None = None,
                           max_characters: int = MAX_CONTEXT_CHARACTERS,
                           max_tokens: int = MAX_CONTEXT_TOKENS) -> str:
    """Keep the newest reference text; current text must not use this helper."""
    if not isinstance(text, str):
        raise TypeError("context 必须是字符串")
    if type(max_characters) is not int or max_characters <= 0 or type(max_tokens) is not int or max_tokens <= 0:
        raise ValueError("上下文预算必须为正整数")
    result = text.strip()
    if len(result) > max_characters:
        cut = len(result) - max_characters
        # A newline is an explicit source-segment boundary. Do not introduce
        # a broken older segment when complete newer segments still fit.
        if result[cut - 1] != "\n":
            boundary = result.find("\n", cut)
            if boundary >= 0:
                cut = boundary + 1
        result = result[cut:].lstrip()
    if token_counter is None:
        return result
    # Token counts are not strictly monotonic at a character boundary. Recount
    # every retained suffix, with at most logarithmically many counter calls.
    while result:
        try:
            count = token_counter(result)
        except Exception:
            return result  # The documented fallback remains character-bounded.
        if type(count) is not int or count < 0:
            return result
        if count <= max_tokens:
            return result
        cut = max(1, len(result) // 2)
        if "\n" in result:
            if result[cut - 1] != "\n":
                boundary = result.find("\n", cut)
                # Try the complete latest line before truncating that line.
                cut = (boundary if boundary >= 0 else result.rfind("\n")) + 1
        result = result[cut:].lstrip()
    return ""


@dataclass(frozen=True)
class _SourceEntry:
    text: str
    timestamp: float


class SourceTextContext:
    """At most three original ASR segments per channel and language direction.

    Call ``get`` before remembering the current segment. Call ``clear`` on
    capture discontinuity, failures, stop, or model/direction changes.
    """

    def __init__(self, *, max_segments: int = 3, ttl_seconds: float = 20.0,
                 token_counter: Callable[[str], int] | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        if type(max_segments) is not int or not 1 <= max_segments <= 3:
            raise ValueError("前文仅保留 1 到 3 段")
        if not math.isfinite(ttl_seconds) or ttl_seconds <= 0:
            raise ValueError("前文有效时间必须为正数")
        self.max_segments = max_segments
        self.ttl_seconds = ttl_seconds
        self._token_counter = token_counter
        self._clock = clock
        self._entries: dict[tuple[str, tuple[str, str]], deque[_SourceEntry]] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _key(channel: str, direction: tuple[str, str]) -> tuple[str, tuple[str, str]]:
        if not isinstance(channel, str) or not channel:
            raise ValueError("channel 必须为非空字符串")
        if (not isinstance(direction, tuple) or len(direction) != 2
                or any(language not in ("zh", "ja") for language in direction)
                or direction[0] == direction[1]):
            raise ValueError("direction 必须为 (zh, ja) 或 (ja, zh)")
        return channel, direction

    def _expire(self, key: tuple[str, tuple[str, str]], now: float) -> None:
        # Capture clocks can arrive out of order across a fair two-source queue.
        # One source's older job must not expire another source's newer history.
        entries = self._entries.get(key)
        if entries is None:
            return
        # A reversed clock must not expose a future reference behind a valid
        # older entry in the same channel's history.
        while entries and now < entries[-1].timestamp:
            entries.pop()
        while entries and (now - entries[0].timestamp >= self.ttl_seconds or now < entries[0].timestamp):
            entries.popleft()
        if not entries:
            self._entries.pop(key)

    def get(self, channel: str, direction: tuple[str, str], *, now: float | None = None) -> str:
        key = self._key(channel, direction)
        with self._lock:
            self._expire(key, self._clock() if now is None else now)
            selected = ""
            for entry in reversed(self._entries.get(key, ())):
                candidate = entry.text + "\n" + selected if selected else entry.text
                fits = len(candidate) <= MAX_CONTEXT_CHARACTERS
                if fits and self._token_counter is not None:
                    try:
                        count = self._token_counter(candidate)
                        if type(count) is int and count >= 0:
                            fits = count <= MAX_CONTEXT_TOKENS
                    except Exception:
                        pass  # The documented fallback is the character budget.
                if not fits:
                    # Never return a sliced older segment. Only an oversized
                    # latest segment falls back to a bounded reference suffix.
                    return selected or bounded_source_context(entry.text, token_counter=self._token_counter)
                selected = candidate
            return selected

    def remember(self, channel: str, direction: tuple[str, str], text: str, *, now: float | None = None) -> None:
        key = self._key(channel, direction)
        if not isinstance(text, str):
            raise TypeError("context 必须是字符串")
        reference = text.strip()
        with self._lock:
            timestamp = self._clock() if now is None else now
            self._expire(key, timestamp)
            if reference:
                self._entries.setdefault(key, deque(maxlen=self.max_segments)).append(_SourceEntry(reference, timestamp))

    def clear(self, channel: str | None = None) -> None:
        with self._lock:
            if channel is None:
                self._entries.clear()
            else:
                for key in tuple(self._entries):
                    if key[0] == channel:
                        self._entries.pop(key)


def looks_incomplete(text: str, language: str) -> bool:
    """A continuation candidate even if ASR added a period, never a verdict."""
    if language not in ("zh", "ja"):
        raise ValueError("只支持 zh 或 ja")
    if not isinstance(text, str):
        raise TypeError("text 必须是字符串")
    value = _ENDING_PUNCTUATION.sub("", text.strip()).strip()
    if not value or value in _SHORT_REPLIES:
        return False
    if language == "zh" and value.startswith(_ZH_CONDITION_PREFIXES):
        return _explicit_zh_condition_fragment(value)
    endings = _JA_OPEN_ENDINGS if language == "ja" else _ZH_OPEN_ENDINGS
    return value.endswith(endings)


def join_continuation_text(previous: str, current: str, language: str) -> str:
    """Normalize only an eligible fragment's artificial terminal boundary.

    Caller must first verify source continuity, pending TTL and language. This
    helper neither decides that two speakers/utterances belong together nor
    rewrites lexical content. Quotes, brackets and meaningful punctuation keep
    the original newline boundary when the safe case cannot be established.
    """
    if not isinstance(previous, str) or not isinstance(current, str):
        raise TypeError("previous 和 current 必须是字符串")
    if language not in ("zh", "ja"):
        raise ValueError("只支持 zh 或 ja")
    unchanged = previous + "\n" + current
    left, right = previous.rstrip(), current.lstrip()
    if not left or not right or not looks_incomplete(left, language):
        return unchanged
    # A quote/bracket boundary can change the scope of the following words.
    # Do not remove its period even when it resembles an ASR boundary.
    if any(mark in left for mark in _QUOTES_OR_BRACKETS):
        return unchanged
    if left.endswith(("?", "？", "!", "！", "…", "‥", "⋯")):
        return unchanged
    if re.search(r"[。．.]{2,}$", left):
        return unchanged
    if left[-1] in "。．.":
        return left[:-1] + right
    if left[-1] in ",，、":
        return left + right
    return unchanged
