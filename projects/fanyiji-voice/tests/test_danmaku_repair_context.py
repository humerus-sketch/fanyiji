"""Local repair selection and the credential-free failure feedback boundary."""
import pytest

from fanyiji.danmaku_ai import DanmakuAIValidationError, sanitize_failure_data
from fanyiji.danmaku_repair import edits_match_focus, repair_context, repair_terms


ORIGINAL = "【谢谢你的评价 我会继续努力】"


@pytest.mark.parametrize("term,focus,expected", [
    ("评价", "评价", True), ("评价", "你的评价", True),
    ("评价", "继续", False), ("意外评价", "评价", False),
    ("意评价", "评价", True), ("评价", "", False),
])
def test_one_local_edit_must_belong_to_the_selected_span(term, focus, expected):
    assert edits_match_focus([{"term": term, "replacement": "感想"}], focus) is expected


def test_two_simultaneous_word_edits_cannot_pass_the_focus_guard():
    assert not edits_match_focus([{"term": "评价"}, {"term": "继续"}], "评价继续")


def test_dictionary_candidates_first_without_treating_the_sentence_as_a_word():
    terms = repair_terms(ORIGINAL, [{"term": "评价"}, {"term": "不存在"}, {"term": "评价"}])
    assert terms[0] == "评价"
    assert len(terms) == len(set(terms))
    assert all(2 <= len(term) <= 12 and term in ORIGINAL and "【" not in term for term in terms)
    assert "谢谢你的评价 我会继续努力" not in terms


def test_already_attempted_terms_move_behind_untried_test_spans():
    history = [{"revised_text": "谢谢你的感想 我会继续努力", "message": "测试未接受",
                "edits": [{"term": "评价", "replacement": "感想"}]}]
    context = repair_context(ORIGINAL, 2, history, ["评价"])
    assert context["focus_term"] != "评价"
    assert context["failed_revisions"] == [{"revised_text": history[0]["revised_text"], "message": "测试未接受"}]
    assert context["round"] == 2


def test_unchanged_round_moves_to_a_different_local_span():
    first = repair_context(ORIGINAL, 1, [])
    second = repair_context(ORIGINAL, 2, [])
    assert first["focus_term"] != second["focus_term"]
    assert first["focus_term"] in ORIGINAL and second["focus_term"] in ORIGINAL


def test_foreign_names_get_the_existing_complete_chinese_conversion_before_word_tests():
    assert repair_context("レレン谢谢你的留言 クちゃん也谢谢", 1, [])["focus_term"] == ""


def test_cli_feedback_bounded_but_local_attempt_count_can_continue_after_three():
    history = [{"revised_text": "谢谢你的感想 我会继续努力", "message": "测试未接受", "edits": []}] * 20
    context = repair_context(ORIGINAL, 21, history)
    assert len(context["failed_revisions"]) == 12
    assert sanitize_failure_data({"text": ORIGINAL, "repair_context": context})["repair_context"]["round"] == 21


def test_repair_context_never_forwards_accounts_cookies_or_extra_nested_fields():
    context = {"strategy": "word_by_word", "round": 4, "focus_term": "评价",
               "Cookie": "secret-cookie", "account": "private-account",
               "failed_revisions": [{"revised_text": "谢谢你的感想 我会继续努力", "message": "未接受",
                                     "Authorization": "secret-token"}]}
    safe = sanitize_failure_data({"text": ORIGINAL, "repair_context": context})
    assert "secret" not in repr(safe) and "private" not in repr(safe)
    assert set(safe["repair_context"]) == {"strategy", "round", "focus_term", "failed_revisions"}


@pytest.mark.parametrize("patch", [
    {"strategy": "arbitrary"}, {"round": 0}, {"round": True}, {"round": "4"},
    {"focus_term": "并不存在"}, {"focus_term": "<script>"},
    {"failed_revisions": "not-a-list"},
    {"failed_revisions": [{"revised_text": "<script>不允许内容</script>"}]},
    {"failed_revisions": [{"revised_text": "谢谢你的感想 我会继续努力"}] * 13},
])
def test_malformed_repair_feedback_is_rejected_locally(patch):
    context = {"strategy": "word_by_word", "round": 4, "focus_term": "评价", "failed_revisions": []}
    context.update(patch)
    with pytest.raises(DanmakuAIValidationError):
        sanitize_failure_data({"text": ORIGINAL, "repair_context": context})
