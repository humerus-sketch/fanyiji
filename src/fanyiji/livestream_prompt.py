"""Shared, bounded Japanese livestream guidance for both translation engines."""
from __future__ import annotations

from collections.abc import Iterable
import re


PROFILE_ID = "japanese_livestream_v1"
HY_PROMPT_FORMAT = "hy-mt2-production-with-source-context-v2"
LEGACY_HY_PROMPT_FORMAT = "hy-mt2-production-with-source-context"
CONTEXT_CHARACTERS = 384
HY_CONTEXT_CHARACTERS = CONTEXT_CHARACTERS
LEGACY_HY_CONTEXT_CHARACTERS = 256
MAX_TERM_HINTS = 3
_CHINESE_OUTPUT_RULES = (
    "译文全部简体中文，禁日文、英文及其他外语文字。"
    "人名昵称游戏名用通行中文名，否则按读音音译为中文；"
    "称呼译中文，不猜身份，不删名称或外语内容。"
    "保留数字、否定、条件、不确定性、空格和中文标点。"
)
_RULES = (
    "直播字幕：用自然简体中文口语，只译当前原文。保留否定、条件、改口、疑问及犹豫。"
    "仅省略无意义停顿，不删有意义的语气词。不补全缺失句尾，不猜人物身份或性别；"
    "按语境译游戏用语，不强套网络梗。"
    "保留引述和玩笑，不改成事实；使用正常标点。"
)
# These are conditional meanings, never replacements applied to ASR text.
_TERMS = (
    (re.compile("ナイス"), "ナイス：漂亮/干得好；确属射击表现时才译好枪。"),
    (re.compile("ワンチャン"), "ワンチャン：有机会/说不定，保留否定。"),
    (re.compile("ガチ"), "ガチ：真的/认真/动真格，按语境。"),
    (re.compile("台パン"), "台パン：拍桌子/砸桌子。"),
    (re.compile("エイム"), "エイム：瞄准/枪法，按语境。"),
    (re.compile("スパチャ|スーパーチャット"), "スパチャ：付费留言，不译成普通聊天。"),
    (re.compile("切り抜き"), "切り抜き：直播语境译切片，其他语境保留本义。"),
    (re.compile("リスナー"), "リスナー：直播语境译观众/听众。"),
    (re.compile(r"お疲れ|おつかれ|(?:^|[\s、。！？!?])おつ(?:$|[\s、。！？!?])"),
     "おつ/お疲れ：辛苦了，不据此断言主播已下播。"),
)


def chinese_output_rules(target_lang: str) -> str:
    """Apply to every Chinese target without changing reverse translation."""
    return _CHINESE_OUTPUT_RULES if target_lang == "zh" else ""


def livestream_term_hints(source_lang: str, target_lang: str,
                          source_texts: Iterable[str] = ()) -> str:
    """Application-owned meanings only; no source text or persistent rules."""
    if target_lang != "zh" or source_lang not in ("ja", "mixed"):
        return ""
    if isinstance(source_texts, str):
        source_texts = (source_texts,)
    matches: dict[int, tuple[int, int]] = {}
    for text_index, text in enumerate(source_texts):
        if not isinstance(text, str):
            raise TypeError("source_texts 必须仅包含字符串")
        for term_index, (pattern, _) in enumerate(_TERMS):
            found = pattern.search(text)
            if found is not None:
                matches.setdefault(term_index, (text_index, found.start()))
    selected = sorted(matches, key=matches.get)[:MAX_TERM_HINTS]
    return " ".join(_TERMS[index][1] for index in selected)


def livestream_rules(source_lang: str, target_lang: str,
                     source_texts: Iterable[str] = ()) -> str:
    """Share current Chinese livestream rules across inference and training."""
    if target_lang != "zh" or source_lang not in ("ja", "mixed"):
        return ""
    hints = livestream_term_hints(source_lang, target_lang, source_texts)
    rules = _RULES + chinese_output_rules(target_lang)
    return rules + "\n仅在相应含义成立时参考术语：" + hints if hints else rules


def hy_translation_prompt(source: str, source_lang: str, target_lang: str,
                          context: str = "", *, profile_enabled: bool = True) -> str:
    """One Hy template for inference/export, with explicit legacy compatibility."""
    if target_lang not in ("zh", "ja"):
        raise ValueError("翻译目标仅支持 zh 或 ja")
    language = "简体中文" if target_lang == "zh" else "日语"
    rules = livestream_rules(source_lang, target_lang, (source,)) if profile_enabled else ""
    # profile_enabled=False reconstructs saved legacy training prompts exactly.
    # All current inference prompts enforce the Chinese target for every source.
    if profile_enabled and not rules:
        rules = chinese_output_rules(target_lang)
    rules = "\n" + rules if rules else ""
    mixed = (
        "原文可能同时包含日语、中文和英语，也可能在一句内切换语言。"
        f"所有当前原文都统一译为{language}，不得随原文语言改变翻译目标。"
        "已经是目标语言的部分保留原意，不得将中文误当日语解释。"
        + ("保持人名和游戏专有名词的含义并写成中文，" if target_lang == "zh" and profile_enabled
           else "谨慎保留人名和游戏专有名词，")
        + "保持数字、否定、语气和不确定性；不得补写未识别到的内容。"
    ) if source_lang == "mixed" else ""
    if context:
        return (f"〖背景信息：同一通道前文原文，仅用于理解，不翻译〗\n{context}\n\n"
                f"请结合背景信息将〖待翻译文本〗翻译为{language}。只输出当前待翻译文本的译文，"
                "不要重复翻译背景信息，不要解释。保留当前原文的人名、数字、币种、否定和条件关系；"
                "背景只用于理解指代和话题，不得从背景添加、替换或推断当前原文没有的事实。"
                f"{mixed}{rules}\n\n〖待翻译文本〗\n{source}")
    if source_lang == "mixed":
        return (f"将以下混合语言文本翻译为{language}。只输出当前原文的译文，不要解释。"
                f"{mixed}{rules}\n\n{source}")
    return f"将以下文本翻译为{language}，注意只需要输出翻译后的结果，不要额外解释：{rules}\n\n{source}"
