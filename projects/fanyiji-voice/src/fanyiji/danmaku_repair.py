"""Choose local wording tests without treating substrings as forbidden words."""
from __future__ import annotations

import re

from .translation_output import is_chinese_output


def repair_terms(text: str, dictionary_hits=()) -> list[str]:
    body = text[1:-1].strip() if text.startswith("【") and text.endswith("】") else text.strip()
    terms = []

    def add(term):
        if (isinstance(term, str) and 2 <= len(term) <= 12 and term in body
                and term != body and is_chinese_output(term)
                and any("\u4e00" <= char <= "\u9fff" for char in term)
                and term not in terms):
            terms.append(term)

    for hit in dictionary_hits:
        add(hit.get("term") if isinstance(hit, dict) else hit)
    phrases = re.findall(r"[\u4e00-\u9fff]+", body)
    for phrase in phrases:
        add(phrase)
    for phrase in phrases:
        if len(phrase) > 12:
            for start in range(0, len(phrase), 12):
                add(phrase[start:start + 12])
    # These are test spans, not a dictionary or a claim about platform policy.
    # Keep the surrounding sentence intact while trying one local change.
    for width in (2, 3, 4):
        for phrase in phrases:
            for start in range(max(0, len(phrase) - width + 1)):
                add(phrase[start:start + width])
    return terms


def edits_match_focus(edits: list[dict], focus: str) -> bool:
    if len(edits) != 1 or not focus:
        return False
    term = edits[0]["term"]
    # A one-character difference is expanded to its neighbouring word by the
    # existing diff guard, so allow that one-character boundary extension.
    return term in focus or (focus in term and len(term) <= len(focus) + 1)


def repair_context(text: str, round_number: int, failed_plans: list[dict], dictionary_hits=()) -> dict:
    tried_terms = {edit["term"] for plan in failed_plans for edit in plan.get("edits", [])}
    terms = repair_terms(text, dictionary_hits) if is_chinese_output(text) else []
    untried = [term for term in terms if term not in tried_terms]
    pool = untried or terms
    focus = pool[(round_number - 1) % len(pool)] if pool else ""
    return {"strategy": "word_by_word", "round": round_number, "focus_term": focus,
            "failed_revisions": [
                {"revised_text": plan["revised_text"], "message": plan.get("message", "")}
                for plan in failed_plans[-12:]]}
