"""Analysis isolation and rewrite guards; all analyzers and CLI transports are fake."""
import os
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pytest
from PySide6.QtWidgets import QApplication

from fanyiji.danmaku_ai import (DANMAKU_AI_INSTRUCTIONS, DANMAKU_AI_OUTPUT_SCHEMA,
    DanmakuAIAnalyzer, DanmakuAIValidationError, derive_local_edits, sanitize_failure_data, validate_analysis_result)
from fanyiji.codex_engine import CodexTranslationEngine


ORIGINAL = "【首先 大家请冷静一点 蕾莲 谢谢你的留言】"


def recommendation(**changes):
    result = {"action": "rewrite", "revised_text": "首先 大家请冷静一点 蕾莲 谢谢你的评论",
              "reason": "尝试普通同义词 不据此认定平台禁词", "suspected_cause": "unknown",
              "edits": [{"term": "留言", "replacement": "评论"}]}
    result.update(changes)
    return result


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def wait(app, predicate, seconds=2):
    deadline = time.monotonic() + seconds
    while not predicate() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.005)
    app.processEvents()
    assert predicate()


def test_safe_input_never_forwards_credentials_or_identity_even_in_nested_fields():
    result = sanitize_failure_data({"text": ORIGINAL, "room": "private-room", "Cookie": "private-cookie",
        "receipt": {"outcome": "platform_f", "code": 0, "Cookie": "private-nested-cookie",
                    "http_status": 200, "headers": {"Authorization": "private-auth"}},
        "diagnosis": {"category": "unknown", "reason": "platform_f", "account": "private-account"},
        "dictionary_hits": [{"term": "留言", "account": "private-account"}],
        "confirmed_rules": [{"term": "评价", "action": "remove", "Cookie": "private-cookie"}]})
    assert result == {"text": ORIGINAL, "receipt": {"outcome": "platform_f", "code": 0, "http_status": 200},
        "diagnosis": {"category": "unknown", "reason": "platform_f"}, "dictionary_hits": ["留言"],
        "confirmed_rules": [{"term": "评价", "action": "remove", "replacement": ""}]}
    assert "private" not in repr(result)


def test_simple_local_edit_is_valid_with_a_chinese_name():
    assert validate_analysis_result(recommendation(), ORIGINAL) == recommendation()


def test_runtime_schema_returns_only_the_corrected_sentence():
    assert DANMAKU_AI_OUTPUT_SCHEMA["required"] == ["revised_text"]
    assert set(DANMAKU_AI_OUTPUT_SCHEMA["properties"]) == {"revised_text"}
    assert DANMAKU_AI_OUTPUT_SCHEMA["additionalProperties"] is False
    assert "suggestion, diagnosis, reasoning, explanation" in DANMAKU_AI_INSTRUCTIONS


def test_minimal_cli_result_derives_a_local_term_without_returning_explanations():
    result = validate_analysis_result({"revised_text": recommendation()["revised_text"]}, ORIGINAL)
    assert result["action"] == "rewrite"
    assert result["edits"] == [{"term": "留言", "replacement": "评论"}]
    assert result["suspected_cause"] == "unknown"
    assert "本地词级差异" in result["reason"]


def test_unchanged_sentence_keeps_original_and_never_learns_whole_sentence():
    result = validate_analysis_result({"revised_text": ORIGINAL[1:-1]}, ORIGINAL)
    assert result["action"] == "retry_original"
    assert result["revised_text"] == "" and result["edits"] == []


def test_partial_character_difference_extends_to_a_local_word():
    assert derive_local_edits("谢谢你的评价 我们继续", "谢谢你的评语 我们继续") == [
        {"term": "评价", "replacement": "评语"}]


def test_multiple_local_differences_are_replayed_exactly():
    result = validate_analysis_result({"revised_text": "谢谢你的感想 继续交流意见"}, "谢谢你的评价 继续交流留言")
    assert result["edits"] == [{"term": "评价", "replacement": "感想"}, {"term": "留言", "replacement": "意见"}]


@pytest.mark.parametrize("original,revised", [
    ("谢谢你的评价 我有10个留言", "谢谢你的感想 我有11个留言"),
    ("我不能接受这个留言", "我可以接受这个评论"),
    ("Mio谢谢你的留言", "Miko谢谢你的评论"),
    ("レレン谢谢你的留言", "レナ谢谢你的评论"),
    ("张老师谢谢留言", "李老师谢谢评论"),
    ("这句话完整需要保留实际意思", "全部修改成另一句不同表达"),
    ("谢谢", "多谢"),
    ("首先 大家请冷静一点 レレン 谢谢你的留言", "请大家先冷静一下 レレン 谢谢你的评论"),
    ("留言 留言 都是留言", "评论 留言 都是留言"),
])
def test_minimal_result_rejects_unsafe_nonlocal_or_inconsistent_diff(original, revised):
    with pytest.raises(DanmakuAIValidationError):
        validate_analysis_result({"revised_text": revised}, original)


@pytest.mark.parametrize("value", [{"revised_text": "谢谢你的评论", "explanation": "extra"},
    {"revised_text": ""}, {"revised_text": "a" * 201}, {"revised_text": "【谢谢你的评论】"},
    {"revised_text": "<script>谢谢你的评论</script>"}])
def test_minimal_result_requires_the_exact_plain_schema(value):
    with pytest.raises(DanmakuAIValidationError):
        validate_analysis_result(value, "谢谢你的留言")


@pytest.mark.parametrize("changes", [
    {"extra": "unexpected"}, {"action": []}, {"suspected_cause": []},
    {"suspected_cause": "environment"}, {"revised_text": ""}, {"revised_text": "x" * 201},
    {"revised_text": "<script>send</script>"}, {"revised_text": "```text```"},
    {"revised_text": "隐形\u200b文字"}, {"reason": "a\nb"},
    {"revised_text": "【首先 大家请冷静一点 レレン 谢谢你的评论】"},
    {"edits": []}, {"edits": [{"term": "留", "replacement": "评"}]},
    {"edits": [{"term": "不存在", "replacement": "评论"}]},
    {"edits": [{"term": "留言", "replacement": "留言"}]},
    {"edits": [{"term": "留言", "replacement": "评论", "cause": "extra"}]},
    {"edits": [{"term": "留言", "replacement": "评论"}, {"term": "留言", "replacement": "发言"}]},
    {"edits": [{"term": "首先 大家请冷静一点 レレン 谢谢你的留言", "replacement": "评论"}]},
    {"revised_text": "请大家先冷静一下 レレン 谢谢你的评论"},
    {"revised_text": "首先 大家请冷静一点 レレン 谢谢你的评论",
     "edits": [{"term": "蕾莲", "replacement": "レレン"}, {"term": "留言", "replacement": "评论"}]},
])
def test_invalid_or_nonlocal_rewrite_is_rejected(changes):
    with pytest.raises(DanmakuAIValidationError):
        validate_analysis_result(recommendation(**changes), ORIGINAL)


@pytest.mark.parametrize("action", ["wait", "manual", "retry_original"])
def test_nonrewrite_requires_no_modified_text_or_edits(action):
    result = recommendation(action=action, revised_text="", edits=[])
    assert validate_analysis_result(result, ORIGINAL) == result
    with pytest.raises(DanmakuAIValidationError):
        validate_analysis_result(recommendation(action=action), ORIGINAL)


@pytest.mark.parametrize("original,revised,term,replacement,error", [
    ("我有10次留言", "我有11次留言", "10次", "11次", "numbers_changed"),
    ("我有10次留言", "我有次数留言", "10次", "次数", "numbers_changed"),
    ("我不能接受这个留言", "我可以接受这个留言", "不能", "可以", "negation_changed"),
    ("我没有说这个留言", "我说了这个留言", "没有说", "说了", "negation_changed"),
    ("Mio谢谢你的留言", "Miko谢谢你的留言", "Mio", "Miko", "non_chinese_output"),
    ("张三老师谢谢留言", "李四老师谢谢留言", "张三", "李四", "numbers_changed"),
    ("张老师谢谢留言", "李老师谢谢留言", "张老师", "李老师", "names_changed"),
    ("我只有两次留言", "我只有三次留言", "两次", "三次", "numbers_changed"),
    ("现在为-10度留言", "现在为10度留言", "-10度", "10度", "numbers_changed"),
    ("这个10%留言", "这个10留言", "10%", "10", "numbers_changed"),
])
def test_numbers_negation_and_names_are_preserved(original, revised, term, replacement, error):
    with pytest.raises(DanmakuAIValidationError, match=error):
        validate_analysis_result(recommendation(revised_text=revised,
            edits=[{"term": term, "replacement": replacement}]), original)


def test_replacement_is_single_pass_without_cascading():
    original = "谢谢留言和评论"
    result = recommendation(revised_text="谢谢评论和发言",
        edits=[{"term": "留言", "replacement": "评论"}, {"term": "评论", "replacement": "发言"}])
    assert validate_analysis_result(result, original)["revised_text"] == "谢谢评论和发言"


def test_longest_match_cannot_learn_an_edit_that_was_not_actually_applied():
    result = recommendation(revised_text="谢谢发言", edits=[
        {"term": "留言", "replacement": "评论"}, {"term": "你的留言", "replacement": "发言"}])
    with pytest.raises(DanmakuAIValidationError, match="unapplied_edit"):
        validate_analysis_result(result, "谢谢你的留言")


def test_only_filler_can_be_removed_by_local_edit_contract():
    # Semantic suitability is instructed in the prompt; structural checks preserve
    # names/numbers/negation and ensure that only the specified substring is removed.
    result = recommendation(revised_text="谢谢你的留言", edits=[{"term": "啊啊", "replacement": ""}])
    assert validate_analysis_result(result, "啊啊谢谢你的留言") == result
    assert "meaningless repeated filler" in DANMAKU_AI_INSTRUCTIONS


def test_historical_foreign_names_become_only_local_chinese_transliteration_rules():
    original = "【首先 大家请冷静一点 レレン 谢谢你的留言 等一下 クちゃん也 谢谢你的留言】"
    revised = "首先 大家请冷静一点 蕾莲 谢谢你的留言 等一下 小库也 谢谢你的留言"
    result = validate_analysis_result({"revised_text": revised}, original)
    assert result["revised_text"] == revised
    assert result["edits"] == [{"term": "レレン", "replacement": "蕾莲"},
                                {"term": "クちゃん", "replacement": "小库"}]
    assert result["suspected_cause"] == "unknown"


@pytest.mark.parametrize("original,revised,term,replacement", [
    ("Mio小姐 谢谢你的留言", "米奥小姐 谢谢你的留言", "Mio", "米奥"),
    ("Mio2 谢谢你的留言", "米奥2 谢谢你的留言", "Mio", "米奥"),
    ("ﾚﾚﾝ 谢谢你的留言", "蕾莲 谢谢你的留言", "ﾚﾚﾝ", "蕾莲"),
    ("Спасибо 谢谢你的留言", "谢谢 谢谢你的留言", "Спасибо", "谢谢"),
])
def test_bounded_foreign_word_translation_preserves_existing_chinese(original, revised, term, replacement):
    result = validate_analysis_result({"revised_text": revised}, original)
    assert result["edits"] == [{"term": term, "replacement": replacement}]


@pytest.mark.parametrize("revised", [
    "首先 大家请冷静一点 レレン 谢谢你的留言 等一下 小库也 谢谢你的留言",
    "首先 大家请冷静一点 蕾莲 谢谢你的留言 等一下 クちゃん也 谢谢你的留言",
    "首先 大家请冷静一点 蕾莲 谢谢你的留言 等一下 Ku也 谢谢你的留言",
])
def test_historical_repair_rejects_any_foreign_remnant(revised):
    original = "首先 大家请冷静一点 レレン 谢谢你的留言 等一下 クちゃん也 谢谢你的留言"
    with pytest.raises(DanmakuAIValidationError, match="non_chinese_output"):
        validate_analysis_result({"revised_text": revised}, original)


@pytest.mark.parametrize("original,revised,term,replacement,error", [
    ("Mio 我有10次机会", "米奥 我有11次机会", "10", "11", "numbers_changed"),
    ("Mio 我不能这样做", "米奥 我能够这样做", "不能", "能够", "negation_changed"),
    ("Mio 张老师谢谢留言", "米奥 李老师谢谢留言", "张老师", "李老师", "names_changed"),
])
def test_chinese_transliteration_does_not_relax_facts_or_existing_chinese_names(original, revised, term, replacement, error):
    with pytest.raises(DanmakuAIValidationError, match=error):
        validate_analysis_result(recommendation(revised_text=revised, edits=[
            {"term": "Mio", "replacement": "米奥"}, {"term": term, "replacement": replacement}]), original)


@pytest.mark.parametrize("revised,edits", [
    ("谢谢你的留言", [{"term": "レレン", "replacement": ""}]),
    ("谢谢你的留言", [{"term": "レレン 谢谢", "replacement": "谢谢"}]),
])
def test_foreign_name_cannot_be_dropped_instead_of_translated(revised, edits):
    original = "レレン谢谢你的留言" if edits[0]["term"] == "レレン" else "レレン 谢谢你的留言"
    with pytest.raises(DanmakuAIValidationError, match="foreign_name"):
        validate_analysis_result(recommendation(revised_text=revised, edits=edits), original)


def test_foreign_original_cannot_be_returned_as_a_retryable_sentence():
    original = "レレン 谢谢你的留言"
    with pytest.raises(DanmakuAIValidationError, match="non_chinese_output"):
        validate_analysis_result({"revised_text": original}, original)
    with pytest.raises(DanmakuAIValidationError, match="non_chinese_output"):
        validate_analysis_result(recommendation(action="retry_original", revised_text="", edits=[]), original)


def test_foreign_name_conversion_preserves_an_already_chinese_nickname_without_address_suffix():
    original = "Mio 蕾莲谢谢留言"
    with pytest.raises(DanmakuAIValidationError, match="chinese_text_changed"):
        validate_analysis_result(recommendation(revised_text="米奥 雷伦谢谢留言", edits=[
            {"term": "Mio", "replacement": "米奥"}, {"term": "蕾莲", "replacement": "雷伦"}]), original)


@pytest.mark.parametrize("original,revised", [
    ("Mio Cia Rio Dia 谢谢你的留言", "米奥 西娅 里奥 迪娅 谢谢你的留言"),
    ("ExtremelyLongUserName 谢谢你的留言", "某个用户 谢谢你的留言"),
    ("A 谢谢你的留言", "诶 谢谢你的留言"),
])
def test_foreign_conversion_cannot_relax_word_level_learning_bounds(original, revised):
    with pytest.raises(DanmakuAIValidationError):
        validate_analysis_result({"revised_text": revised}, original)


def test_worker_is_lazy_serial_background_and_uses_current_settings(app, monkeypatch):
    monkeypatch.setattr(CodexTranslationEngine, "__init__", lambda *args, **kwargs: pytest.fail("real CLI engine created"))
    entered, release = threading.Event(), threading.Event()
    calls = []
    def fake(data, settings, cancellation):
        calls.append((threading.get_ident(), data, settings))
        entered.set()
        release.wait(2)
        return recommendation()
    analyzer = DanmakuAIAnalyzer(analyzer=fake)
    answers = []
    analyzer.finished.connect(lambda *args: answers.append(args))
    assert not calls and not analyzer.busy
    assert analyzer.submit("head:8:0", {"text": ORIGINAL}, {"codex_model": "chosen-model", "codex_effort": "low", "codex_speed": "fast"})
    assert entered.wait(1)
    assert analyzer.busy
    assert not analyzer.submit("second", {"text": ORIGINAL}, {})
    release.set()
    wait(app, lambda: bool(answers))
    assert calls[0][0] != threading.get_ident()
    assert calls[0][2] == {"model": "chosen-model", "effort": "low", "speed": "fast"}
    assert answers == [("head:8:0", recommendation(), "")]
    assert not analyzer.busy and analyzer.close()


@pytest.mark.parametrize("failure,expected", [("invalid", "ai_invalid_result"), ("exception", "ai_unavailable"), ("timeout", "ai_timeout")])
def test_worker_emits_safe_error_and_never_leaks_exception_text(app, failure, expected):
    def fake(data, settings, cancellation):
        if failure == "invalid":
            return {"private": "bad-result"}
        if failure == "exception":
            raise RuntimeError("private-cookie-or-url")
        cancellation.wait(1)
        return recommendation()
    analyzer = DanmakuAIAnalyzer(analyzer=fake, timeout_seconds=0.05)
    answers = []
    analyzer.finished.connect(lambda *args: answers.append(args))
    analyzer.submit("test", {"text": ORIGINAL}, {})
    wait(app, lambda: bool(answers))
    assert answers == [("test", None, expected)]
    assert "private" not in repr(answers)
    assert analyzer.close()


def test_close_cancels_active_isolated_engine_and_suppresses_late_result(app):
    started = threading.Event()
    instances = []
    class FakeEngine:
        def __init__(self, **options):
            self.cancelled = threading.Event()
            self.closed = False
            instances.append(self)
        def analyze_danmaku_failure(self, data, cancellation_event):
            started.set()
            self.cancelled.wait(1)
            return recommendation()
        def cancel_pending(self):
            self.cancelled.set()
        def close(self):
            self.closed = True
            self.cancel_pending()
    analyzer = DanmakuAIAnalyzer(engine_factory=FakeEngine)
    answers = []
    analyzer.finished.connect(lambda *args: answers.append(args))
    analyzer.submit("close-task", {"text": ORIGINAL}, {})
    assert started.wait(1)
    assert analyzer.close()
    app.processEvents()
    assert instances[0].closed and not answers
    assert not analyzer.submit("new-task", {"text": ORIGINAL}, {})


def test_old_cancellation_token_does_not_cancel_new_task(app):
    analyzer = DanmakuAIAnalyzer(analyzer=lambda *args: recommendation())
    answers = []
    analyzer.finished.connect(lambda *args: answers.append(args))
    analyzer.submit("first", {"text": ORIGINAL}, {})
    wait(app, lambda: len(answers) == 1)
    analyzer._interrupt(token="old-token", timeout=True)
    analyzer.submit("second", {"text": ORIGINAL}, {})
    analyzer.cancel("first")
    wait(app, lambda: len(answers) == 2)
    assert all(not answer[2] for answer in answers)
    assert analyzer.close()


def test_public_engine_method_uses_safe_schema_and_nonreused_analysis_context(monkeypatch):
    calls = []
    def payload(self, data, schema, instructions, cancellation, decoder, **options):
        calls.append((data, schema, instructions, options))
        return decoder({"revised_text": recommendation()["revised_text"]})
    monkeypatch.setattr(CodexTranslationEngine, "_translate_payload", payload)
    engine = CodexTranslationEngine()
    result = engine.analyze_danmaku_failure({"text": ORIGINAL, "Cookie": "private-cookie"})
    assert result["revised_text"] == recommendation()["revised_text"]
    assert result["edits"] == recommendation()["edits"]
    data, schema, instructions, options = calls[0]
    assert data == {"source_language": "zh", "target_language": "zh", "failure": {"text": ORIGINAL, "receipt": {}}}
    assert schema == DANMAKU_AI_OUTPUT_SCHEMA
    assert options == {"conversation_key": "danmaku_failure_analysis", "reuse": False}
    assert "No tools" in instructions and "does not prove" in instructions
    engine.close()


def test_cancelled_public_analysis_does_not_start_transport(monkeypatch):
    monkeypatch.setattr(CodexTranslationEngine, "_translate_payload", lambda *args, **kwargs: pytest.fail("transport started"))
    cancellation = threading.Event()
    cancellation.set()
    engine = CodexTranslationEngine()
    with pytest.raises(RuntimeError):
        engine.analyze_danmaku_failure({"text": ORIGINAL}, cancellation)
    engine.close()
