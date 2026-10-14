"""Separate recognition languages from a channel's fixed translation target."""
from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class AsrResult:
    text: str
    detected_language: str | None = None
    detection_source: str = "unknown"


def validate_asr_language(value: str) -> str:
    if not isinstance(value, str) or value not in ("zh", "ja", "en", "auto"):
        raise ValueError("识别语言仅支持 zh、ja、en 或 auto")
    return value


def validate_translation_source(value: str) -> str:
    if not isinstance(value, str) or value not in ("zh", "ja", "en", "mixed"):
        raise ValueError("翻译源语言仅支持 zh、ja、en 或 mixed")
    return value


def normalize_detected_language(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    return {
        "zh": "zh", "chinese": "zh", "中文": "zh",
        "ja": "ja", "japanese": "ja", "日语": "ja",
        "en": "en", "english": "en", "英语": "en",
    }.get(value.strip().casefold())


def continuation_language(text: str, primary_language: str,
                          detected_language: str | None = None, *, automatic: bool = False) -> str | None:
    """A conservative grammar hint, never a claim about detected audio language.

    Ambiguous Han-only text and substantial English clauses keep their segment
    boundary instead of borrowing Japanese particle rules from the channel.
    """
    if primary_language not in ("zh", "ja"):
        raise ValueError("频道主要语言仅支持 zh 或 ja")
    if not automatic:
        return primary_language
    detected = normalize_detected_language(detected_language)
    if detected == "en" or len(re.findall(r"[A-Za-z]+(?:['’-][A-Za-z]+)*", text)) >= 3:
        return None
    has_kana = bool(re.search(r"[\u3040-\u30ff\uff66-\uff9f]", text))
    if has_kana:
        return None if detected == "zh" else "ja"
    if not re.search(r"[\u3400-\u9fff]", text):
        return None
    if detected in ("zh", "ja"):
        return detected
    return "zh" if primary_language == "zh" else None


def context_language(text: str, primary_language: str,
                     detected_language: str | None = None, *, automatic: bool = False) -> str | None:
    """Keep clear English context without applying CJK continuation grammar."""
    grammar = continuation_language(text, primary_language, detected_language, automatic=automatic)
    if grammar is not None or not automatic:
        return grammar
    if re.search(r"[\u3400-\u9fff\u3040-\u30ff\uff66-\uff9f]", text):
        return None
    detected = normalize_detected_language(detected_language)
    if detected in ("zh", "ja"):
        return None
    if detected == "en" or len(re.findall(r"[A-Za-z]+(?:['’-][A-Za-z]+)*", text)) >= 3:
        return "en"
    return None
