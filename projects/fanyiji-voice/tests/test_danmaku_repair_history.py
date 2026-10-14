"""Repair history is durable and never permits replaying an uncertain POST."""
import copy
import json

import pytest

from fanyiji import danmaku_filter
from fanyiji.danmaku_filter import DanmakuContentFilter, DanmakuFilterError


SOURCE_ROOM = "21399689"
TEST_ROOM = "1698225"
ORIGINAL = "【谢谢大家的评价我会继续努力】"


@pytest.fixture
def history_filter(tmp_path):
    current = DanmakuContentFilter(tmp_path / "filter.json")
    candidate = current.record_failure(SOURCE_ROOM, ORIGINAL, "platform_f", queue_id=44)
    return current, candidate["id"]


def stage(current, candidate_id, replacement):
    revised = ORIGINAL[1:-1].replace("评价", replacement)
    current.save_validation_plan(candidate_id, revised,
                                 edits=[{"term": "评价", "replacement": replacement}])
    return current.validation_plan(candidate_id)


def reject(current, candidate_id, *, error="platform_f", code=0, http_status=200):
    current.configure_validation(candidate_id, TEST_ROOM, formal_room=SOURCE_ROOM)
    plan = current.validation_plan(candidate_id)
    current.update_validation(candidate_id, state="testing", request_id=plan["validation_id"])
    current.record_validation_receipt(candidate_id, request_id=plan["validation_id"],
        before_count=0, receipt_accepted=False, code=code, http_status=http_status, error=error)
    current.update_validation(candidate_id, state="failed", message="测试未接受")
    return current.validation_plan(candidate_id)


def test_legacy_candidate_loads_empty_history_without_changing_version(history_filter):
    current, candidate_id = history_filter
    document = json.loads(current.path.read_text(encoding="utf-8"))
    document["candidates"][0].pop("repair_history")
    current.path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    restored = DanmakuContentFilter(current.path)
    assert restored.candidates()[0]["repair_history"] == []
    assert restored.failed_validation_plans(candidate_id) == []
    restored.persist()
    assert json.loads(current.path.read_text(encoding="utf-8"))["version"] == 1


def test_multiple_rejected_corrections_survive_restart_in_order_without_rules(history_filter):
    current, candidate_id = history_filter
    rejected = []
    for replacement in ("感想", "看法", "意见", "观点"):
        stage(current, candidate_id, replacement)
        rejected.append(reject(current, candidate_id))
        current = DanmakuContentFilter(current.path)
    candidate = current.candidates()[0]
    assert candidate["repair_history"] == rejected[:-1]
    assert candidate["last_validation"] == rejected[-2]
    assert candidate["validation"] == rejected[-1]
    assert current.failed_validation_plans(candidate_id) == rejected
    assert not current.rules()


def test_old_history_candidate_is_not_resubmitted_even_with_broader_edit(history_filter):
    current, candidate_id = history_filter
    stage(current, candidate_id, "感想")
    reject(current, candidate_id)
    stage(current, candidate_id, "看法")
    reject(current, candidate_id)
    expected = current.candidates()[0]
    stored = current.path.read_bytes()
    current.save_validation_plan(candidate_id, ORIGINAL[1:-1].replace("评价", "感想"),
        edits=[{"term": "大家的评价", "replacement": "大家的感想"}])
    assert current.candidates()[0] == expected
    assert current.path.read_bytes() == stored


def test_returned_history_is_a_deep_copy(history_filter):
    current, candidate_id = history_filter
    stage(current, candidate_id, "感想")
    expected = reject(current, candidate_id)
    returned = current.failed_validation_plans(candidate_id)
    returned[0]["receipt"]["error"] = "other"
    returned[0]["edits"][0]["replacement"] = "私自修改"
    assert current.failed_validation_plans(candidate_id) == [expected]


@pytest.mark.parametrize("code,http_status,error", [
    (None, None, "other"), (0, 200, "other"), (0, None, "platform_f"),
    (0, 200, "platform_rejected"),
])
def test_unproven_rejection_is_not_advertised_as_retryable(history_filter, code, http_status, error):
    current, candidate_id = history_filter
    stage(current, candidate_id, "感想")
    reject(current, candidate_id, code=code, http_status=http_status, error=error)
    stored = current.path.read_bytes()
    assert current.failed_validation_plans(candidate_id) == []
    with pytest.raises(DanmakuFilterError, match="只能继续核验"):
        stage(current, candidate_id, "看法")
    assert current.path.read_bytes() == stored
    assert current.candidates()[0]["repair_history"] == []


def test_unsubmitted_failed_plan_allows_a_different_local_correction(history_filter):
    current, candidate_id = history_filter
    stage(current, candidate_id, "感想")
    current.update_validation(candidate_id, state="failed", message="本地正文限制未通过")
    failed = current.validation_plan(candidate_id)
    assert current.failed_validation_plans(candidate_id) == [failed]
    stage(current, candidate_id, "看法")
    assert current.candidates()[0]["repair_history"] == [failed]
    assert current.validation_plan(candidate_id)["request_id"] == ""
    assert not current.rules()


def test_uncertain_post_remains_protected_while_old_failed_history_remains_visible(history_filter):
    current, candidate_id = history_filter
    stage(current, candidate_id, "感想")
    failed = reject(current, candidate_id)
    stage(current, candidate_id, "看法")
    current.configure_validation(candidate_id, TEST_ROOM, formal_room=SOURCE_ROOM)
    current.update_validation(candidate_id, state="testing", request_id="f" * 32)
    expected = current.validation_plan(candidate_id)
    assert current.failed_validation_plans(candidate_id) == [failed]
    with pytest.raises(DanmakuFilterError, match="只能继续核验"):
        stage(current, candidate_id, "意见")
    assert current.validation_plan(candidate_id) == expected
    assert current.candidates()[0]["repair_history"] == [failed]


def test_atomic_history_save_failure_keeps_previous_plan_and_existing_rules(history_filter, monkeypatch):
    current, candidate_id = history_filter
    rule = current.add_rule("旧词", "旧表达")
    stage(current, candidate_id, "感想")
    first = reject(current, candidate_id)
    stage(current, candidate_id, "看法")
    second = reject(current, candidate_id)
    expected = current.candidates()[0]
    stored = current.path.read_bytes()
    with monkeypatch.context() as patcher:
        def cannot_replace(*args):
            raise OSError("simulated disk failure")
        patcher.setattr(danmaku_filter.os, "replace", cannot_replace)
        with pytest.raises(DanmakuFilterError, match="原规则未修改"):
            stage(current, candidate_id, "意见")
    assert current.path.read_bytes() == stored
    assert current.candidates()[0] == expected
    restored = DanmakuContentFilter(current.path)
    assert restored.failed_validation_plans(candidate_id) == [first, second]
    assert restored.rules() == [rule]
    stage(restored, candidate_id, "意见")
    assert restored.candidates()[0]["repair_history"] == [first, second]


def test_old_last_validation_is_archived_when_a_new_plan_replaces_it(history_filter):
    current, candidate_id = history_filter
    stage(current, candidate_id, "感想")
    first = reject(current, candidate_id)
    stage(current, candidate_id, "看法")
    second = reject(current, candidate_id)
    document = json.loads(current.path.read_text(encoding="utf-8"))
    document["candidates"][0].pop("repair_history")
    current.path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    restored = DanmakuContentFilter(current.path)
    assert restored.failed_validation_plans(candidate_id) == [first, second]
    stage(restored, candidate_id, "意见")
    assert restored.candidates()[0]["repair_history"] == [first, second]
    assert restored.candidates()[0]["last_validation"] == second


def test_reopened_success_retains_history_and_global_word_rule(history_filter):
    current, candidate_id = history_filter
    stage(current, candidate_id, "感想")
    failed = reject(current, candidate_id)
    stage(current, candidate_id, "看法")
    current.configure_validation(candidate_id, TEST_ROOM, formal_room=SOURCE_ROOM)
    plan = current.validation_plan(candidate_id)
    request_id = plan["validation_id"]
    current.update_validation(candidate_id, state="testing", request_id=request_id)
    current.record_validation_receipt(candidate_id, request_id=request_id, before_count=0,
        receipt_accepted=True, code=0, http_status=200)
    current.confirm_validation(candidate_id, evidence={
        "test_room": TEST_ROOM, "validation_id": plan["validation_id"], "request_id": request_id,
        "observed_at": "2026-10-13T11:00:00.000+08:00", "receipt_accepted": True,
        "page_observed": True, "before_count": 0, "occurrence_count": 1,
        "code": 0, "http_status": 200, "error": "none"})
    passed = current.validation_plan(candidate_id)
    rules = current.rules()
    current.record_failure(SOURCE_ROOM, ORIGINAL, "platform_f", queue_id=45)
    restored = DanmakuContentFilter(current.path)
    assert restored.candidates()[0]["repair_history"] == [failed, passed]
    assert restored.candidates()[0]["last_validation"] == passed
    assert restored.validation_plan(candidate_id) is None
    assert restored.failed_validation_plans(candidate_id) == [failed]
    assert restored.rules() == rules
    assert rules[0]["room"] == "" and rules[0]["enabled"]


@pytest.mark.parametrize("bad_history", [None, {}, "history", [None]])
def test_invalid_history_is_rejected_without_replacing_file(history_filter, bad_history):
    current, _ = history_filter
    document = json.loads(current.path.read_text(encoding="utf-8"))
    document["candidates"][0]["repair_history"] = bad_history
    current.path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    stored = current.path.read_bytes()
    with pytest.raises(DanmakuFilterError, match="损坏或格式无效"):
        DanmakuContentFilter(current.path)
    assert current.path.read_bytes() == stored


def test_history_plan_is_validated_against_original_and_rejects_duplicate_ids(history_filter):
    current, candidate_id = history_filter
    stage(current, candidate_id, "感想")
    failed = reject(current, candidate_id)
    document = json.loads(current.path.read_text(encoding="utf-8"))
    document["candidates"][0]["repair_history"] = [failed, copy.deepcopy(failed)]
    with pytest.raises(DanmakuFilterError, match="重复方案"):
        current._validate(document)
    document["candidates"][0]["repair_history"] = [copy.deepcopy(failed)]
    document["candidates"][0]["repair_history"][0]["original_text"] = "另一条原句"
    with pytest.raises(DanmakuFilterError):
        current._validate(document)


def test_failed_history_requires_existing_candidate(history_filter):
    current, _ = history_filter
    with pytest.raises(DanmakuFilterError, match="不存在"):
        current.failed_validation_plans("f" * 32)
