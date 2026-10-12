"""Local filter tests; no real user data, model calls or external requests."""
import json
from datetime import datetime, timedelta

import pytest

from fanyiji import danmaku_filter
from fanyiji.danmaku_filter import DanmakuContentFilter, DanmakuFilterError


@pytest.fixture
def content_filter(tmp_path):
    return DanmakuContentFilter(tmp_path / "filter.json")


def test_default_path_is_user_data_not_package(tmp_path, monkeypatch):
    monkeypatch.setattr(danmaku_filter, "data_root", lambda: tmp_path)
    current = DanmakuContentFilter()
    assert current.path == tmp_path / "danmaku-content-filter.json"
    assert current.enabled
    assert current.rules() == current.candidates() == []
    assert not current.path.exists()


def learning_evidence(original="【不过 谢谢 获得评价】", revised="【不过 谢谢 获得感想】", **overrides):
    return {"queue_id": 44, "original_text": original, "current_text": revised,
            "failure_count": 3, "observed_at": "2026-10-11T12:00:00.000+08:00",
            "receipt_accepted": True, "page_observed": True, **overrides}


def legacy_observed_rule(content_filter):
    rule = content_filter.add_rule("评价", "感想", "1698225")
    payload = json.loads(content_filter.path.read_text(encoding="utf-8"))
    evidence = learning_evidence()
    payload["rules"][0].update(origin="ai_success_observed", evidence={
        "queue_id": 44, "failed_digest": danmaku_filter.hashlib.sha256(evidence["original_text"].encode()).hexdigest(),
        "revised_digest": danmaku_filter.hashlib.sha256(evidence["current_text"].encode()).hexdigest(),
        "failure_count": 3, "learned_at": rule["created_at"], "observed_at": evidence["observed_at"], "confirmed_cause": False})
    content_filter.path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    restored = DanmakuContentFilter(content_filter.path)
    return restored, restored.rules()[0]


def test_successful_local_edits_learn_empirical_room_rules_and_digest_evidence(content_filter):
    observation = content_filter.record_failure("1698225", "【不过 谢谢 获得评价】", "platform_f", queue_id=44)
    learned = content_filter.learn_successful_edits("1698225", [{"term": "评价", "replacement": "感想"}],
                                                    evidence=learning_evidence())
    assert learned == [] and content_filter.rules() == []
    plan = content_filter.validation_plan(observation["id"])
    assert plan["state"] == "waiting_room" and plan["evidence"] is None
    assert plan["original_text"] == observation["text"] and plan["revised_text"] == "不过 谢谢 获得感想"
    assert content_filter.candidates()[0]["status"] == "waiting_validation"
    assert content_filter.apply("【评价】", "1698225") == ("【评价】", 0)
    assert content_filter.apply("【评价】", "21399689") == ("【评价】", 0)
    assert DanmakuContentFilter(content_filter.path).validation_plan(observation["id"]) == plan


def test_successful_deletion_learns_remove_action_without_neutral_default(content_filter):
    evidence = learning_evidence(revised="【不过 谢谢 获得】")
    row = content_filter.record_failure("1698225", evidence["original_text"], "platform_f", queue_id=44)
    assert content_filter.learn_successful_edits("1698225", [{"term": "评价", "replacement": ""}], evidence=evidence) == []
    assert content_filter.validation_plan(row["id"])["revised_text"] == "不过 谢谢 获得"
    assert content_filter.rules() == []


def test_learning_replays_multiple_edits_once_without_cascading(content_filter, monkeypatch):
    original, revised = "【甲词 乙词 甲词】", "【乙词 丙词 乙词】"
    row = content_filter.record_failure("1698225", original, "platform_f", queue_id=44)
    original_replace = danmaku_filter.os.replace
    writes = []
    def count_replace(source, destination):
        writes.append(destination)
        return original_replace(source, destination)
    monkeypatch.setattr(danmaku_filter.os, "replace", count_replace)
    learned = content_filter.learn_successful_edits("1698225",
        [{"term": "甲词", "replacement": "乙词"}, {"term": "乙词", "replacement": "丙词"}],
        evidence=learning_evidence(original, revised))
    assert learned == [] and len(writes) == 1
    assert content_filter.validation_plan(row["id"])["revised_text"] == "乙词 丙词 乙词"
    assert content_filter.apply(original, "1698225") == (original, 0)


@pytest.mark.parametrize("scope", ["1698225", ""])
@pytest.mark.parametrize("enabled", [True, False])
def test_learning_never_overrides_applicable_manual_rule(content_filter, scope, enabled):
    original = content_filter.add_rule("评价", "人工表达", scope, enabled=enabled)
    before, revision = content_filter.path.read_bytes(), content_filter.revision
    assert content_filter.learn_successful_edits("1698225", [{"term": "评价", "replacement": "感想"}],
                                                evidence=learning_evidence()) == []
    assert content_filter.rules() == [original]
    assert content_filter.path.read_bytes() == before and content_filter.revision == revision


def test_rule_in_different_room_does_not_prevent_local_learning(content_filter):
    other = content_filter.add_rule("评价", "另一房间说法", "21399689")
    row = content_filter.record_failure("1698225", learning_evidence()["original_text"], "platform_f", queue_id=44)
    learned = content_filter.learn_successful_edits("1698225", [{"term": "评价", "replacement": "感想"}],
                                                    evidence=learning_evidence())
    assert learned == [] and content_filter.rules() == [other]
    assert content_filter.validation_plan(row["id"])["state"] == "waiting_room"
    assert content_filter.apply("评价", "21399689") == ("另一房间说法", 1)


@pytest.mark.parametrize("room,original", [("21399689", "【不过 谢谢 获得评价】"), ("1698225", "【其他语句 评价】")])
def test_learning_cannot_borrow_known_queue_evidence_from_another_room_or_text(content_filter, room, original):
    candidate = content_filter.record_failure("1698225", "【不过 谢谢 获得评价】", "platform_f", queue_id=44)
    before = content_filter.path.read_bytes()
    with pytest.raises(DanmakuFilterError, match="失败房间或原正文"):
        content_filter.learn_successful_edits(room, [{"term": "评价", "replacement": "感想"}],
            evidence=learning_evidence(original, original.replace("评价", "感想")))
    assert content_filter.path.read_bytes() == before
    assert content_filter.rules() == [] and content_filter.candidates() == [candidate]


def test_duplicate_learning_does_not_refresh_evidence_or_write_file(content_filter):
    content_filter, rule = legacy_observed_rule(content_filter)
    before = content_filter.path.read_bytes()
    assert content_filter.learn_successful_edits("1698225", [{"term": "评价", "replacement": "说法"}],
        evidence=learning_evidence(revised="【不过 谢谢 获得说法】", queue_id=45)) == []
    assert content_filter.path.read_bytes() == before and content_filter.rules() == [rule]


def test_manual_edit_of_learned_rule_converts_origin_but_toggle_keeps_observation(content_filter):
    content_filter, rule = legacy_observed_rule(content_filter)
    toggled = content_filter.update_rule(rule["id"], enabled=False)
    assert toggled["origin"] == "ai_success_observed" and toggled["evidence"] == rule["evidence"]
    changed = content_filter.update_rule(rule["id"], replacement="用户新表达")
    assert changed["origin"] == "user" and changed["evidence"] is None
    assert changed["created_at"] == rule["created_at"]


def test_origin_metadata_does_not_change_semantic_revision_or_signatures(content_filter):
    content_filter, rule = legacy_observed_rule(content_filter)
    revision, signatures = content_filter.revision, content_filter.rule_signatures("1698225")
    source = json.loads(content_filter.path.read_text(encoding="utf-8"))
    source["rules"][0].update(origin="user", evidence=None, enabled=False)
    content_filter.path.write_text(json.dumps(source, ensure_ascii=False), encoding="utf-8")
    restored = DanmakuContentFilter(content_filter.path)
    assert restored.revision == revision and restored.rule_signatures("1698225") == signatures
    assert restored.rules()[0]["id"] == rule["id"]


def test_legacy_rules_default_to_user_origin_without_persistence_write(content_filter):
    original = content_filter.add_rule("评价", "感想", "1698225")
    source = json.loads(content_filter.path.read_text(encoding="utf-8"))
    source["rules"][0].pop("origin")
    source["rules"][0].pop("evidence")
    legacy = json.dumps(source, ensure_ascii=False)
    content_filter.path.write_text(legacy, encoding="utf-8")
    restored = DanmakuContentFilter(content_filter.path)
    assert restored.rules() == [original]
    assert restored.path.read_text(encoding="utf-8") == legacy


@pytest.mark.parametrize("overrides", [
    {"queue_id": None}, {"queue_id": True}, {"queue_id": 0}, {"queue_id": "44"},
    {"failure_count": 0}, {"failure_count": True}, {"failure_count": 2**63},
    {"receipt_accepted": False}, {"receipt_accepted": 1}, {"page_observed": False}, {"page_observed": 1},
    {"observed_at": "2026-10-11T12:00:00+00:00"}, {"observed_at": None},
    {"current_text": "【不过 谢谢 获得评价】"}, {"current_text": "【 】"}])
def test_learning_requires_distinct_corrected_post_and_page_observation(content_filter, overrides):
    with pytest.raises(DanmakuFilterError):
        content_filter.learn_successful_edits("1698225", [{"term": "评价", "replacement": "感想"}],
                                             evidence=learning_evidence(**overrides))
    assert content_filter.rules() == [] and not content_filter.path.exists()


@pytest.mark.parametrize("room", ["", None, "*", "１２３", "0", "https://live.bilibili.com/1698225"])
def test_learning_never_creates_global_or_unscoped_rules(content_filter, room):
    with pytest.raises(DanmakuFilterError):
        content_filter.learn_successful_edits(room, [{"term": "评价", "replacement": "感想"}], evidence=learning_evidence())
    assert not content_filter.path.exists()


@pytest.mark.parametrize("edits", [[], {}, [{"term": "评价"}], [{"term": "评价", "replacement": "感想", "extra": True}],
    [{"term": "评", "replacement": "感"}], [{"term": "😀", "replacement": "😊"}],
    [{"term": "评价", "replacement": "评价"}], [{"term": "未出现", "replacement": "中性"}],
    [{"term": "评价", "replacement": "字" * 13}], [{"term": "评价", "replacement": "【感想】"}],
    [{"term": "评价", "replacement": "a\x00b"}], [{"term": "评价", "replacement": "感想"}] * 2,
    [{"term": "评价", "replacement": "感想"}] * 9])
def test_learning_rejects_nonlocal_unsafe_or_duplicate_edits(content_filter, edits):
    with pytest.raises(DanmakuFilterError):
        content_filter.learn_successful_edits("1698225", edits, evidence=learning_evidence())
    assert not content_filter.path.exists()


@pytest.mark.parametrize("original,term,revised", [
    ("【评价】", "评价", "【感想】"), ("评价", "评价", "感想"),
    ("【很长很长的经验编辑字词太多】", "很长很长的经验编辑字词太多", "【感想】")])
def test_learning_cannot_store_entire_sentence_or_long_phrase(content_filter, original, term, revised):
    with pytest.raises(DanmakuFilterError):
        content_filter.learn_successful_edits("1698225", [{"term": term, "replacement": "感想"}],
                                             evidence=learning_evidence(original, revised))
    assert not content_filter.path.exists()


@pytest.mark.parametrize("revised", ["【不过 谢谢 获得 感想】", "【谢谢 获得感想】", "【不过 谢谢 获得说法】"])
def test_learning_requires_exact_replay_of_every_observed_character(content_filter, revised):
    with pytest.raises(DanmakuFilterError, match="单次重放"):
        content_filter.learn_successful_edits("1698225", [{"term": "评价", "replacement": "感想"}],
                                             evidence=learning_evidence(revised=revised))
    assert not content_filter.path.exists()


def test_learning_does_not_store_shadowed_edit_without_evidence_of_use(content_filter):
    with pytest.raises(DanmakuFilterError, match="单次重放"):
        content_filter.learn_successful_edits("1698225",
            [{"term": "获得评价", "replacement": "收到感想"}, {"term": "评价", "replacement": "感想"}],
            evidence=learning_evidence(revised="【不过 谢谢 收到感想】"))
    assert not content_filter.path.exists()


def test_successful_deletion_learning_matches_sender_whitespace_folding(content_filter):
    original, revised = "【你好 えっと 大家】", "【你好 大家】"
    row = content_filter.record_failure("1698225", original, "platform_f", queue_id=44)
    learned = content_filter.learn_successful_edits("1698225", [{"term": "えっと", "replacement": ""}],
                                                   evidence=learning_evidence(original, revised))
    assert learned == [] and content_filter.rules() == []
    assert content_filter.validation_plan(row["id"])["revised_text"] == "你好 大家"


def test_successful_learning_whitespace_folding_does_not_ignore_nonspace_characters(content_filter):
    with pytest.raises(DanmakuFilterError, match="单次重放"):
        content_filter.learn_successful_edits("1698225", [{"term": "えっと", "replacement": ""}],
            evidence=learning_evidence("【你好 えっと 大家】", "【你好 大家呢】"))
    assert not content_filter.path.exists()


def test_ai_suggestions_have_analysis_provenance_without_fabricated_delivery_evidence(content_filter):
    candidate = content_filter.record_failure("1698225", "【谢谢 获得评价】", "platform_f", queue_id=44)
    saved = content_filter.save_ai_suggestions("1698225", candidate["text"],
        [{"term": "评价", "replacement": "感想"}], candidate_id=candidate["id"])
    assert saved == [] and content_filter.rules() == []
    plan = content_filter.validation_plan(candidate["id"])
    assert plan["original_text"] == candidate["text"] and plan["evidence"] is None
    assert plan["revised_text"] == "谢谢 获得感想" and plan["state"] == "waiting_room"
    assert content_filter.candidates()[0]["status"] == "waiting_validation"
    assert content_filter.apply("【评价】", "1698225") == ("【评价】", 0)
    assert content_filter.apply("【评价】", "21399689") == ("【评价】", 0)
    assert DanmakuContentFilter(content_filter.path).validation_plan(candidate["id"]) == plan


@pytest.mark.parametrize("room,text,candidate_id", [
    ("21399689", "【谢谢 获得评价】", None), ("1698225", "【另一句 评价】", None),
    ("", "【谢谢 获得评价】", None), ("1698225", "【谢谢 获得评价】", "f" * 32)])
def test_ai_suggestions_cannot_attach_to_another_candidate_room_or_original(content_filter, room, text, candidate_id):
    candidate = content_filter.record_failure("1698225", "【谢谢 获得评价】", "platform_f")
    before = content_filter.path.read_bytes()
    with pytest.raises(DanmakuFilterError, match="房间或原文"):
        content_filter.save_ai_suggestions(room, text, [{"term": "评价", "replacement": "感想"}],
                                          candidate_id=candidate_id or candidate["id"])
    assert content_filter.path.read_bytes() == before and content_filter.rules() == []


@pytest.mark.parametrize("scope,enabled", [("1698225", True), ("1698225", False), ("", True), ("", False)])
def test_ai_suggestions_do_not_override_applicable_manual_choices(content_filter, scope, enabled):
    existing = content_filter.add_rule("评价", "人工表达", scope, enabled=enabled)
    candidate = content_filter.record_failure("1698225", "【谢谢 获得评价】", "platform_f")
    before = content_filter.path.read_bytes()
    assert content_filter.save_ai_suggestions("1698225", candidate["text"],
        [{"term": "评价", "replacement": "感想"}], candidate_id=candidate["id"]) == []
    assert content_filter.rules() == [existing]
    assert content_filter.validation_plan(candidate["id"])["state"] == "waiting_room"


def test_ai_suggestions_allow_local_deletion_with_space_folding(content_filter):
    candidate = content_filter.record_failure("1698225", "【你好 えっと 大家】", "platform_f")
    assert content_filter.save_ai_suggestions("1698225", candidate["text"],
        [{"term": "えっと", "replacement": ""}], candidate_id=candidate["id"]) == []
    assert content_filter.validation_plan(candidate["id"])["revised_text"] == "你好 大家"
    assert content_filter.rules() == []


@pytest.mark.parametrize("edits", [[{"term": "评", "replacement": "感"}],
    [{"term": "评价", "replacement": "评价"}], [{"term": "未出现", "replacement": "中性"}],
    [{"term": "评价", "replacement": "字" * 13}], [{"term": "评价", "replacement": "【感想】"}],
    [{"term": "谢谢 获得评价", "replacement": "其他内容"}]])
def test_ai_suggestions_keep_short_local_edit_guards(content_filter, edits):
    candidate = content_filter.record_failure("1698225", "【谢谢 获得评价】", "platform_f")
    before = content_filter.path.read_bytes()
    with pytest.raises(DanmakuFilterError):
        content_filter.save_ai_suggestions("1698225", candidate["text"], edits, candidate_id=candidate["id"])
    assert content_filter.path.read_bytes() == before and content_filter.rules() == []


def test_ai_suggestions_cannot_remove_every_body_word(content_filter):
    candidate = content_filter.record_failure("1698225", "【甲词 乙词】", "platform_f")
    with pytest.raises(DanmakuFilterError, match="不能删除全部正文"):
        content_filter.save_ai_suggestions("1698225", candidate["text"],
            [{"term": "甲词", "replacement": ""}, {"term": "乙词", "replacement": ""}], candidate_id=candidate["id"])
    assert content_filter.rules() == []


def test_ai_suggestions_save_batch_atomically_and_preserve_candidate_until_completion(content_filter, monkeypatch):
    candidate = content_filter.record_failure("1698225", "【甲词 乙词】", "platform_f")
    before = content_filter.path.read_bytes()
    def fail_replace(*_args):
        raise PermissionError("test file locked")
    monkeypatch.setattr(danmaku_filter.os, "replace", fail_replace)
    with pytest.raises(DanmakuFilterError, match="原规则未修改"):
        content_filter.save_ai_suggestions("1698225", candidate["text"],
            [{"term": "甲词", "replacement": "乙词"}, {"term": "乙词", "replacement": "丙词"}],
            candidate_id=candidate["id"])
    assert content_filter.rules() == [] and content_filter.candidates() == [candidate]
    assert content_filter.path.read_bytes() == before


@pytest.mark.parametrize("resolution", ["ai_no_local_edit", "ai_rules_saved"])
def test_complete_ai_review_uses_analysis_completion_without_page_delivery_claim(content_filter, resolution):
    candidate = content_filter.record_failure("1698225", "【谢谢 获得评价】", "platform_f")
    diagnosis = {"category": "ai_review", "action": "analysis_complete", "message": "AI 已完成本地分析 原因尚未证实"}
    resolved = content_filter.complete_ai_review(candidate["id"], resolution, diagnosis=diagnosis)
    assert resolved["status"] == "waiting_validation" and resolved["resolution"] == ""
    assert resolved["analysis_resolution"] == resolution
    assert resolved["diagnosis"] == diagnosis and resolved["reason"] == "platform_f"
    assert content_filter.list_candidates(include_resolved=False) == [resolved]
    assert content_filter.rules() == []
    assert resolved["last_resolution"] is None


def test_set_candidate_status_explicit_analysis_resolution_is_idempotent(content_filter):
    candidate = content_filter.record_failure("1698225", "【完整失败句】", "platform_f")
    resolved = content_filter.set_candidate_status(candidate["id"], "resolved", resolution="ai_no_local_edit")
    before = content_filter.path.read_bytes()
    assert content_filter.set_candidate_status(candidate["id"], "resolved", resolution="ai_no_local_edit") == resolved
    assert content_filter.path.read_bytes() == before


@pytest.mark.parametrize("resolution", ["page_observed", "delivered", "", None])
def test_complete_ai_review_rejects_nonanalysis_resolution_codes(content_filter, resolution):
    candidate = content_filter.record_failure("1698225", "【完整失败句】", "platform_f")
    before = content_filter.path.read_bytes()
    with pytest.raises(DanmakuFilterError):
        content_filter.complete_ai_review(candidate["id"], resolution)
    assert content_filter.path.read_bytes() == before


@pytest.mark.parametrize("field,value", [("candidate_id", "wrong"), ("source_digest", "not digest"),
    ("confirmed_cause", True), ("learned_at", "2026-10-11T12:00:00+00:00")])
def test_invalid_ai_analysis_evidence_does_not_load_as_successful_observation(content_filter, field, value):
    candidate = content_filter.record_failure("1698225", "【谢谢 获得评价】", "platform_f")
    rule = content_filter.add_rule("评价", "感想", "1698225")
    source = json.loads(content_filter.path.read_text(encoding="utf-8"))
    source["rules"][0].update(origin="ai_analysis", evidence={"candidate_id": candidate["id"],
        "source_digest": danmaku_filter.hashlib.sha256(candidate["text"].encode()).hexdigest(),
        "learned_at": rule["created_at"], "confirmed_cause": False})
    source["rules"][0]["evidence"][field] = value
    content_filter.path.write_text(json.dumps(source, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(DanmakuFilterError, match="损坏或格式无效"):
        DanmakuContentFilter(content_filter.path)


def staged_validation(content_filter):
    candidate = content_filter.record_failure("1698225", "【谢谢 获得评价】", "platform_f", queue_id=44)
    content_filter.save_validation_plan(candidate["id"], "【谢谢 获得感想】", edits=[{"term": "评价", "replacement": "感想"}])
    content_filter.configure_validation(candidate["id"], "1698225", formal_room="21399689")
    content_filter.update_validation(candidate["id"], state="testing", request_id="e" * 32)
    return candidate, content_filter.validation_plan(candidate["id"])


def validation_evidence(plan, **overrides):
    return {"test_room": plan["test_room"], "validation_id": plan["validation_id"], "request_id": plan["request_id"],
        "observed_at": "2026-10-11T12:00:00.000+08:00", "receipt_accepted": True, "page_observed": True,
        "before_count": 1, "occurrence_count": 2, "code": 0, "http_status": 200, "error": "none", **overrides}


def test_validation_plan_and_unanswered_request_survive_restart_without_rules(content_filter):
    candidate, plan = staged_validation(content_filter)
    assert content_filter.rules() == [] and plan["state"] == "testing" and plan["receipt"] is None
    assert plan["attempts"] == 0 and plan["source_room"] == "1698225" and plan["formal_room"] == "21399689"
    restored = DanmakuContentFilter(content_filter.path)
    assert restored.validation_plan(candidate["id"]) == plan
    with pytest.raises(DanmakuFilterError, match="不能重新提交"):
        restored.update_validation(candidate["id"], state="testing", request_id="f" * 32)
    assert restored.validation_plan(candidate["id"]) == plan


def test_validation_saves_rules_only_after_persisted_receipt_and_new_page_count(content_filter):
    candidate, plan = staged_validation(content_filter)
    content_filter.record_validation_receipt(candidate["id"], request_id=plan["request_id"], before_count=1,
                                              receipt_accepted=True, code=0, http_status=200)
    assert content_filter.rules() == []
    resolved = content_filter.confirm_validation(candidate["id"], evidence=validation_evidence(plan))
    assert resolved["status"] == "resolved" and resolved["resolution"] == "test_room_validated"
    assert resolved["validation"]["state"] == "passed" and resolved["validation"]["attempts"] == 1
    rule = content_filter.rules()[0]
    assert rule["enabled"] and rule["origin"] == "ai_analysis" and rule["validation"]["test_room"] == "1698225"
    assert rule["evidence"]["confirmed_cause"] is False and rule["validation"]["confirmed_cause"] is False
    assert rule["evidence"]["candidate_id"] == candidate["id"]
    assert content_filter.apply("评价", "1698225") == ("感想", 1)
    assert content_filter.apply("评价", "21399689") == ("感想", 1)
    assert rule["room"] == ""
    assert rule["validation"]["formal_room"] == "21399689"
    assert resolved["room"] == "1698225"
    assert DanmakuContentFilter(content_filter.path).candidates()[0] == resolved
    before = content_filter.path.read_bytes()
    assert content_filter.confirm_validation(candidate["id"], evidence=validation_evidence(plan)) == resolved
    assert content_filter.path.read_bytes() == before


def test_new_actual_failure_reopens_passed_plan_without_deleting_learned_rules(content_filter):
    candidate, plan = staged_validation(content_filter)
    content_filter.record_validation_receipt(candidate["id"], request_id=plan["request_id"], before_count=1,
                                              receipt_accepted=True, code=0, http_status=200)
    resolved = content_filter.confirm_validation(candidate["id"], evidence=validation_evidence(plan))
    rules, revision = content_filter.rules(), content_filter.revision
    reopened = content_filter.record_failure(candidate["room"], candidate["text"], "platform_f", queue_id=45)
    assert reopened["id"] == candidate["id"]
    assert reopened["count"] == candidate["count"] + 1
    assert reopened["queue_id"] == 45 and reopened["status"] == "active"
    assert reopened["resolved_at"] is None and reopened["resolution"] == ""
    assert reopened["validation"] is None
    assert reopened["last_validation"] == resolved["validation"]
    assert reopened["last_resolution"] == resolved["last_resolution"]
    assert content_filter.rules() == rules and content_filter.revision == revision
    restored = DanmakuContentFilter(content_filter.path)
    assert restored.candidates()[0] == reopened
    new = restored.save_validation_plan(candidate["id"], "谢谢 获得看法",
                                          edits=[{"term": "评价", "replacement": "看法"}])
    assert new["validation"]["state"] == "waiting_room"
    assert new["last_validation"] == resolved["validation"]
    assert restored.rules() == rules
    restored.configure_validation(candidate["id"], "1698225", formal_room="21399689")
    assert restored.rules() == rules and restored.revision == revision
    assert restored.apply("评价", "777777") == ("感想", 1)
    pending = restored.update_validation(candidate["id"], state="testing", request_id="e" * 32)
    restored.record_validation_receipt(candidate["id"], request_id=pending["validation"]["request_id"],
                                      before_count=0, receipt_accepted=False, code=0, http_status=200, error="platform_f")
    restored.update_validation(candidate["id"], state="failed")
    assert restored.rules() == rules and restored.candidates()[0]["last_validation"] == resolved["validation"]
    assert DanmakuContentFilter(restored.path).rules() == rules


@pytest.mark.parametrize("state,receipt", [
    ("testing", None),
    ("testing", {"receipt_accepted": False, "code": None, "http_status": None, "error": "result_unknown"}),
    ("failed", {"receipt_accepted": False, "code": 0, "http_status": 200, "error": "platform_f"}),
])
def test_new_actual_failure_keeps_unfinished_validation_plan_for_same_sentence(content_filter, state, receipt):
    candidate, plan = staged_validation(content_filter)
    if receipt:
        content_filter.record_validation_receipt(candidate["id"], request_id=plan["request_id"], before_count=1, **receipt)
    content_filter.update_validation(candidate["id"], state=state, request_id=plan["request_id"])
    previous = content_filter.validation_plan(candidate["id"])
    failed = content_filter.record_failure(candidate["room"], candidate["text"], "platform_f", queue_id=45)
    assert failed["status"] == "active"
    assert failed["validation"] == previous
    assert failed["last_validation"] is None


def test_validation_cannot_be_completed_by_unpersisted_callback_claim(content_filter):
    candidate, plan = staged_validation(content_filter)
    before = content_filter.path.read_bytes()
    with pytest.raises(DanmakuFilterError, match="持久化回执"):
        content_filter.confirm_validation(candidate["id"], evidence=validation_evidence(plan))
    assert content_filter.path.read_bytes() == before and content_filter.rules() == []


@pytest.mark.parametrize("override", [
    {"before_count": 0}, {"before_count": True}, {"occurrence_count": 1}, {"occurrence_count": True},
    {"test_room": "21399689"}, {"test_room": "111"}, {"request_id": "f" * 32}, {"validation_id": "f" * 32},
    {"receipt_accepted": False}, {"page_observed": False}, {"code": False}, {"code": 1},
    {"http_status": 429}, {"error": "platform_f"}])
def test_validation_rejects_wrong_room_request_baseline_or_failed_response(content_filter, override):
    candidate, plan = staged_validation(content_filter)
    content_filter.record_validation_receipt(candidate["id"], request_id=plan["request_id"], before_count=1,
                                              receipt_accepted=True, code=0, http_status=200)
    before = content_filter.path.read_bytes()
    with pytest.raises(DanmakuFilterError):
        content_filter.confirm_validation(candidate["id"], evidence=validation_evidence(plan, **override))
    assert content_filter.path.read_bytes() == before and content_filter.rules() == []


def test_validation_failed_post_receipt_cannot_be_upgraded_by_page_presence_alone(content_filter):
    candidate, plan = staged_validation(content_filter)
    content_filter.record_validation_receipt(candidate["id"], request_id=plan["request_id"], before_count=1,
                                              receipt_accepted=False, code=0, http_status=200, error="platform_f")
    content_filter.update_validation(candidate["id"], state="failed", message="专用房间返回 f")
    with pytest.raises(DanmakuFilterError, match="持久化回执"):
        content_filter.confirm_validation(candidate["id"], evidence=validation_evidence(plan))
    assert content_filter.rules() == []


@pytest.mark.parametrize("code,http_status,error", [
    (0, 200, "platform_f"), (-403, 200, "platform_rejected"),
    (None, 429, "platform_rejected"),
])
def test_explicit_rejected_plan_can_stage_different_sentence_and_preserve_audit(content_filter, code, http_status, error):
    manual = content_filter.add_rule("评价", "人工表达", "1698225", enabled=False)
    candidate, plan = staged_validation(content_filter)
    content_filter.record_validation_receipt(candidate["id"], request_id=plan["request_id"], before_count=1,
                                              receipt_accepted=False, code=code, http_status=http_status, error=error)
    content_filter.update_validation(candidate["id"], state="failed", message="本方案明确被拒绝")
    old = content_filter.validation_plan(candidate["id"])
    restored = DanmakuContentFilter(content_filter.path)
    revised = restored.save_validation_plan(candidate["id"], "谢谢 获得看法",
                                               edits=[{"term": "评价", "replacement": "看法"}])
    assert revised["last_validation"] == old
    current = revised["validation"]
    assert current["revised_text"] == "谢谢 获得看法"
    assert current["state"] == "waiting_room"
    assert current["attempts"] == 0 and current["receipt"] is None and current["request_id"] == ""
    assert current["validation_id"] != old["validation_id"]
    assert current["evidence"] is None and current["validated_rule_ids"] == []
    assert restored.rules() == [manual]
    reloaded = DanmakuContentFilter(content_filter.path)
    assert reloaded.candidates()[0]["last_validation"] == old
    assert reloaded.validation_plan(candidate["id"]) == current
    reloaded.configure_validation(candidate["id"], "1698225", formal_room="21399689")
    reloaded.update_validation(candidate["id"], state="testing", request_id="f" * 32)
    assert reloaded.validation_plan(candidate["id"])["request_id"] == "f" * 32


def test_same_rejected_sentence_keeps_failed_plan_without_resubmission(content_filter):
    candidate, plan = staged_validation(content_filter)
    content_filter.record_validation_receipt(candidate["id"], request_id=plan["request_id"], before_count=1,
                                              receipt_accepted=False, code=0, http_status=200, error="platform_f")
    content_filter.update_validation(candidate["id"], state="failed")
    before = content_filter.path.read_bytes()
    # A broader spelling of the same edit still produces the same sentence.
    result = content_filter.save_validation_plan(candidate["id"], "谢谢 获得感想",
                                                  edits=[{"term": "获得评价", "replacement": "获得感想"}])
    assert result["validation"]["state"] == "failed"
    assert result["validation"]["attempts"] == 1
    assert result["last_validation"] is None
    assert content_filter.path.read_bytes() == before
    with pytest.raises(DanmakuFilterError, match="继续只读核验"):
        content_filter.update_validation(candidate["id"], state="testing", request_id="f" * 32)


@pytest.mark.parametrize("state,accepted,code,http_status,error,record", [
    ("testing", False, None, None, "result_unknown", False),
    ("failed", False, None, None, "result_unknown", False),
    ("failed", False, None, None, "result_unknown", True),
    ("failed", True, 0, 200, "none", True),
    ("failed", False, 0, 200, "other", True),
    ("failed", False, 0, 200, "platform_rejected", True),
    ("failed", False, 0, None, "platform_f", True),
])
def test_unknown_or_unproven_rejection_cannot_replace_submitted_plan(content_filter, state, accepted, code, http_status, error, record):
    candidate, plan = staged_validation(content_filter)
    if record:
        content_filter.record_validation_receipt(candidate["id"], request_id=plan["request_id"], before_count=1,
                                                  receipt_accepted=accepted, code=code, http_status=http_status, error=error)
    content_filter.update_validation(candidate["id"], state=state, request_id=plan["request_id"])
    before = content_filter.path.read_bytes()
    with pytest.raises(DanmakuFilterError, match="只能继续核验"):
        content_filter.save_validation_plan(candidate["id"], "谢谢 获得看法",
                                               edits=[{"term": "评价", "replacement": "看法"}])
    assert content_filter.path.read_bytes() == before
    assert content_filter.candidates()[0]["last_validation"] is None


def test_failed_plan_replacement_save_error_preserves_original_and_audit(content_filter, monkeypatch):
    candidate, plan = staged_validation(content_filter)
    content_filter.record_validation_receipt(candidate["id"], request_id=plan["request_id"], before_count=1,
                                              receipt_accepted=False, code=0, http_status=200, error="platform_f")
    content_filter.update_validation(candidate["id"], state="failed")
    before = content_filter.path.read_bytes()
    def fail(*args):
        raise OSError("temporary save failure")
    monkeypatch.setattr(danmaku_filter.os, "replace", fail)
    with pytest.raises(DanmakuFilterError, match="原规则未修改"):
        content_filter.save_validation_plan(candidate["id"], "谢谢 获得看法",
                                               edits=[{"term": "评价", "replacement": "看法"}])
    assert content_filter.path.read_bytes() == before
    assert content_filter.candidates()[0]["last_validation"] is None


def test_validation_no_submission_does_not_count_post_or_block_new_request(content_filter):
    candidate, plan = staged_validation(content_filter)
    content_filter.record_validation_receipt(candidate["id"], request_id=plan["request_id"], before_count=None,
                                              receipt_accepted=False, request_started=False)
    restored = DanmakuContentFilter(content_filter.path)
    ready = restored.validation_plan(candidate["id"])
    assert ready["state"] == "ready" and ready["attempts"] == 0
    restored.update_validation(candidate["id"], state="testing", request_id="f" * 32)
    assert restored.validation_plan(candidate["id"])["receipt"] is None


def test_validation_accepted_baseline_is_durable_and_cannot_be_rewritten(content_filter):
    candidate, plan = staged_validation(content_filter)
    content_filter.record_validation_receipt(candidate["id"], request_id=plan["request_id"], before_count=1,
                                              receipt_accepted=True, code=0, http_status=200)
    restored = DanmakuContentFilter(content_filter.path)
    before = restored.path.read_bytes()
    with pytest.raises(DanmakuFilterError, match="基线不能改写"):
        restored.record_validation_receipt(candidate["id"], request_id=plan["request_id"], before_count=0,
                                           receipt_accepted=True, code=0, http_status=200)
    assert restored.path.read_bytes() == before
    assert restored.confirm_validation(candidate["id"], evidence=validation_evidence(plan))["validation"]["state"] == "passed"


def test_validation_success_does_not_overwrite_manual_rule(content_filter):
    manual = content_filter.add_rule("评价", "人工表达", "1698225", enabled=False)
    candidate, plan = staged_validation(content_filter)
    content_filter.record_validation_receipt(candidate["id"], request_id=plan["request_id"], before_count=1,
                                              receipt_accepted=True, code=0, http_status=200)
    content_filter.confirm_validation(candidate["id"], evidence=validation_evidence(plan))
    assert content_filter.rules() == [manual]


def test_validation_config_rejects_active_formal_room_but_preserves_historical_source(content_filter):
    row = content_filter.record_failure("1698225", "【谢谢 获得评价】", "platform_f")
    content_filter.save_ai_suggestions("1698225", row["text"], [{"term": "评价", "replacement": "感想"}], candidate_id=row["id"])
    with pytest.raises(DanmakuFilterError, match="正式发送房间分开"):
        content_filter.configure_validation(row["id"], "1698225", formal_room="1698225")
    configured = content_filter.configure_validation(row["id"], "1698225", formal_room="")
    assert configured["validation"]["source_room"] == "1698225" and configured["validation"]["formal_room"] == ""


def test_old_ai_analysis_completion_migrates_to_waiting_validation(content_filter):
    row = content_filter.record_failure("1698225", "【谢谢 获得评价】", "platform_f")
    source = json.loads(content_filter.path.read_text(encoding="utf-8"))
    source["candidates"][0].update(status="resolved", resolution="ai_rules_saved", resolved_at=row["last_time"],
                                  last_resolution={"at": row["last_time"], "resolution": "ai_rules_saved", "queue_id": None})
    content_filter.path.write_text(json.dumps(source, ensure_ascii=False), encoding="utf-8")
    restored = DanmakuContentFilter(content_filter.path).candidates()[0]
    assert restored["status"] == "waiting_validation" and restored["resolution"] == "" and restored["resolved_at"] is None
    assert restored["analysis_resolution"] == "ai_rules_saved" and restored["last_resolution"]["resolution"] == "ai_rules_saved"


def test_legacy_unvalidated_ai_rule_is_disabled_and_cannot_be_enabled(content_filter):
    restored, rule = legacy_observed_rule(content_filter)
    assert not rule["enabled"] and rule["validation"] is None
    with pytest.raises(DanmakuFilterError, match="尚未通过专用测试房间"):
        restored.update_rule(rule["id"], enabled=True)
    assert restored.apply("评价", "1698225") == ("评价", 0)


def test_empty_ai_edits_do_not_create_validation_plan_or_rules(content_filter):
    row = content_filter.record_failure("1698225", "【谢谢 获得评价】", "platform_f")
    before = content_filter.path.read_bytes()
    assert content_filter.save_ai_suggestions("1698225", row["text"], [], candidate_id=row["id"]) == []
    assert content_filter.validation_plan(row["id"]) is None
    assert content_filter.rules() == [] and content_filter.path.read_bytes() == before


def test_validation_success_is_atomic_with_enabling_rule_and_archiving_candidate(content_filter, monkeypatch):
    row, plan = staged_validation(content_filter)
    content_filter.record_validation_receipt(row["id"], request_id=plan["request_id"], before_count=1,
                                              receipt_accepted=True, code=0, http_status=200)
    before = content_filter.path.read_bytes()
    def fail_replace(*_args):
        raise PermissionError("test file locked")
    monkeypatch.setattr(danmaku_filter.os, "replace", fail_replace)
    with pytest.raises(DanmakuFilterError, match="原规则未修改"):
        content_filter.confirm_validation(row["id"], evidence=validation_evidence(plan))
    assert content_filter.path.read_bytes() == before and content_filter.rules() == []
    assert content_filter.validation_plan(row["id"])["state"] == "testing"


def test_foreign_correction_cannot_learn_rules_even_with_accepted_receipt_and_page_observation(content_filter):
    row = content_filter.record_failure("1698225", "【レレン 谢谢你的评价】", "platform_f")
    content_filter.save_validation_plan(row["id"], "レレン 谢谢你的感想",
        edits=[{"term": "评价", "replacement": "感想"}])
    content_filter.configure_validation(row["id"], "1698225", formal_room="")
    content_filter.update_validation(row["id"], state="testing", request_id="e" * 32)
    plan = content_filter.validation_plan(row["id"])
    content_filter.record_validation_receipt(row["id"], request_id=plan["request_id"], before_count=1,
        receipt_accepted=True, code=0, http_status=200)
    before = content_filter.path.read_bytes()
    with pytest.raises(DanmakuFilterError, match="全中文"):
        content_filter.confirm_validation(row["id"], evidence=validation_evidence(plan))
    assert content_filter.path.read_bytes() == before
    assert content_filter.rules() == []
    assert content_filter.validation_plan(row["id"])["state"] == "testing"


@pytest.mark.parametrize("original,revised,edits,message", [
    ("【谢谢 获得评价】", "谢谢 获得感想", [{"term": "获得评价", "replacement": "获得感想"}], "重新计算"),
    ("【我没有买东西】", "我已经买东西", [{"term": "没有", "replacement": "已经"}], "语义安全")])
def test_validation_recomputes_minimal_and_safe_diff_before_creating_rules(content_filter, original, revised, edits, message):
    row = content_filter.record_failure("1698225", original, "platform_f", queue_id=44)
    content_filter.save_validation_plan(row["id"], revised, edits=edits)
    content_filter.configure_validation(row["id"], "1698225", formal_room="")
    content_filter.update_validation(row["id"], state="testing", request_id="e" * 32)
    plan = content_filter.validation_plan(row["id"])
    content_filter.record_validation_receipt(row["id"], request_id=plan["request_id"], before_count=1,
                                              receipt_accepted=True, code=0, http_status=200)
    before = content_filter.path.read_bytes()
    with pytest.raises(DanmakuFilterError, match=message):
        content_filter.confirm_validation(row["id"], evidence=validation_evidence(plan))
    assert content_filter.path.read_bytes() == before and content_filter.rules() == []
    assert content_filter.validation_plan(row["id"])["state"] == "testing"


def test_wrapped_failure_and_plain_ai_result_have_exactly_one_runtime_wrapper(content_filter):
    row = content_filter.record_failure("1698225", "【不过 谢谢 获得评价】", "platform_f")
    staged = content_filter.save_validation_plan(row["id"], "不过 谢谢 获得感想",
                                                edits=[{"term": "评价", "replacement": "感想"}])
    plan = staged["validation"]
    assert plan["original_text"] == "【不过 谢谢 获得评价】"
    assert plan["revised_text"] == "不过 谢谢 获得感想"
    assert "【" + plan["revised_text"] + "】" == "【不过 谢谢 获得感想】"
    assert content_filter.rules() == []


def test_analysis_completion_never_bypasses_pending_validation(content_filter):
    row, _ = staged_validation(content_filter)
    completed = content_filter.complete_ai_review(row["id"], "ai_rules_saved")
    assert completed["status"] == "waiting_validation" and completed["resolution"] == ""
    assert content_filter.rules() == []
    with pytest.raises(DanmakuFilterError, match="尚未通过"):
        content_filter.set_candidate_status(row["id"], "resolved", resolution="test_room_validated")


def test_multiple_learning_edits_roll_back_together_on_atomic_write_failure(content_filter, monkeypatch):
    existing = content_filter.add_rule("原词", "原表达", "1698225")
    content_filter.record_failure("1698225", "【甲词 乙词】", "platform_f", queue_id=44)
    before, revision = content_filter.path.read_bytes(), content_filter.revision
    def fail_replace(*_args):
        raise PermissionError("test file locked")
    monkeypatch.setattr(danmaku_filter.os, "replace", fail_replace)
    with pytest.raises(DanmakuFilterError, match="原规则未修改"):
        content_filter.learn_successful_edits("1698225",
            [{"term": "甲词", "replacement": "乙词"}, {"term": "乙词", "replacement": "丙词"}],
            evidence=learning_evidence("【甲词 乙词】", "【乙词 丙词】"))
    assert content_filter.rules() == [existing] and content_filter.revision == revision
    assert content_filter.path.read_bytes() == before


@pytest.mark.parametrize("field,value", [("origin", "official_ban"), ("origin", []), ("evidence", None),
    ("evidence.confirmed_cause", True), ("evidence.queue_id", False), ("evidence.failure_count", 0),
    ("evidence.failed_digest", "bad"), ("evidence.observed_at", "2026-10-11T12:00:00+00:00"),
    ("room", "invalid"), ("term", "单"), ("term", "很长的经验词条" * 2)])
def test_corrupt_learning_origin_or_evidence_is_not_silently_loaded(content_filter, field, value):
    content_filter, _ = legacy_observed_rule(content_filter)
    source = json.loads(content_filter.path.read_text(encoding="utf-8"))
    row = source["rules"][0]
    if "." in field:
        row["evidence"][field.split(".", 1)[1]] = value
    else:
        row[field] = value
    content_filter.path.write_text(json.dumps(source, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(DanmakuFilterError, match="损坏或格式无效"):
        DanmakuContentFilter(content_filter.path)


def test_longest_original_match_and_no_cascading(content_filter):
    content_filter.add_rule("异常", "中性")
    content_filter.add_rule("异常内容", "正常话题")
    content_filter.add_rule("正常话题", "不应继续替换")
    content_filter.add_rule("中性", "也不应继续替换")
    assert content_filter.apply("异常内容与异常和中性") == ("正常话题与中性和也不应继续替换", 3)


def test_cyclic_rules_cannot_loop(content_filter):
    content_filter.add_rule("甲", "乙")
    content_filter.add_rule("乙", "甲")
    assert content_filter.apply("甲乙甲") == ("乙甲乙", 3)


def test_expanding_rule_matches_each_original_occurrence_once(content_filter):
    content_filter.add_rule("a", "aa")
    assert content_filter.apply("aaa") == ("aaaaaa", 3)


def test_regex_syntax_is_literal(content_filter):
    content_filter.add_rule(".*[词]", "表达")
    assert content_filter.apply("普通内容 .*[词] 内容") == ("普通内容 表达 内容", 1)


def test_unicode_and_brackets_are_preserved(content_filter):
    content_filter.add_rule("日本語😀", "日语")
    assert content_filter.apply("【日本語😀 中日英ABC 日本語😀】", "1698225") == ("【日语 中日英ABC 日语】", 2)


def test_new_rules_are_global_and_same_term_cannot_be_added_for_another_room(content_filter):
    content_filter.add_rule("某词", "通用说法")
    before = content_filter.path.read_bytes()
    with pytest.raises(DanmakuFilterError, match="已存在相同过滤词，请编辑原规则"):
        content_filter.add_rule("某词", "本房说法", "1698225")
    assert content_filter.path.read_bytes() == before
    content_filter.add_rule("限定词", "本房替换", "1698225")
    assert content_filter.apply("某词 限定词", "1698225") == ("通用说法 本房替换", 2)
    assert content_filter.apply("某词 限定词", "21399689") == ("通用说法 本房替换", 2)
    assert content_filter.apply("某词 限定词") == ("通用说法 本房替换", 2)


def test_disabled_rule_and_master_switch(content_filter):
    first = content_filter.add_rule("词语", "中性说法", enabled=False)
    assert content_filter.apply("词语") == ("词语", 0)
    content_filter.update_rule(first["id"], enabled=True)
    assert content_filter.apply("词语") == ("中性说法", 1)
    content_filter.set_enabled(False)
    assert content_filter.apply("词语") == ("词语", 0)
    restored = DanmakuContentFilter(content_filter.path)
    assert not restored.enabled
    restored.set_enabled(True)
    assert restored.apply("词语") == ("中性说法", 1)


@pytest.mark.parametrize("reason", ["platform_f", "platform_rejected", "result_unknown"])
def test_failure_candidates_never_activate_rules(content_filter, reason):
    candidate = content_filter.record_failure("1698225", "【完全正常的直播译文】", reason)
    assert candidate["count"] == 1
    assert candidate["reason"] == reason
    assert content_filter.rules() == []
    assert content_filter.apply(candidate["text"], "1698225") == (candidate["text"], 0)
    assert datetime.fromisoformat(candidate["first_time"]).utcoffset() == timedelta(hours=8)
    assert datetime.fromisoformat(candidate["last_time"]).utcoffset() == timedelta(hours=8)


def test_repeated_same_failure_counts_single_candidate_per_room(content_filter):
    first = content_filter.record_failure("1698225", "【同一条译文】", "platform_f")
    second = content_filter.record_failure("1698225", "【同一条译文】", "platform_rejected")
    content_filter.record_failure("21399689", "【同一条译文】", "platform_f")
    assert second["id"] == first["id"]
    assert second["count"] == 2
    assert second["first_time"] == first["first_time"]
    assert second["reason"] == "platform_rejected"
    assert len(content_filter.candidates()) == 2
    assert content_filter.candidates("1698225") == [second]
    assert content_filter.list_candidates() == content_filter.candidates()


def test_legacy_candidates_migrate_as_active_without_rewriting_file(content_filter):
    original = content_filter.record_failure("1698225", "【完整失败句】", "platform_f")
    source = json.loads(content_filter.path.read_text(encoding="utf-8"))
    for key in ("status", "queue_id", "diagnosis", "resolved_at", "resolution", "last_resolution"):
        source["candidates"][0].pop(key)
    legacy = json.dumps(source, ensure_ascii=False)
    content_filter.path.write_text(legacy, encoding="utf-8")
    restored = DanmakuContentFilter(content_filter.path)
    candidate = restored.candidates()[0]
    assert candidate["id"] == original["id"] and candidate["status"] == "active"
    assert candidate["queue_id"] is candidate["diagnosis"] is candidate["resolved_at"] is None
    assert candidate["resolution"] == "" and candidate["last_resolution"] is None
    assert restored.path.read_text(encoding="utf-8") == legacy


def test_failure_diagnosis_and_queue_link_are_persisted_without_rules(content_filter):
    diagnosis = {"category": "platform_f", "action": "retry", "message": "平台返回 f 原因未确定"}
    row = content_filter.record_failure("1698225", "【完整失败句】", "platform_f", queue_id=44,
                                        diagnosis=diagnosis)
    diagnosis["message"] = "outside mutation"
    assert row["status"] == "active" and row["queue_id"] == 44
    assert row["diagnosis"]["message"] == "平台返回 f 原因未确定"
    assert content_filter.rules() == []
    assert DanmakuContentFilter(content_filter.path).candidates() == [row]


def test_recovery_status_counts_and_history_default_are_compatible(content_filter):
    rows = [content_filter.record_failure("1698225", f"【失败句 {index}】", "platform_f", queue_id=index + 1)
            for index in range(len(danmaku_filter.CANDIDATE_STATUSES))]
    for row, status in zip(rows, danmaku_filter.CANDIDATE_STATUSES):
        content_filter.set_candidate_status(row["id"], status)
    other = content_filter.record_failure("21399689", "【其他房间】", "platform_f")
    assert content_filter.status_counts("1698225") == dict.fromkeys(danmaku_filter.CANDIDATE_STATUSES, 1)
    assert content_filter.status_counts()["active"] == 2
    assert len(content_filter.list_candidates()) == 7
    assert len(content_filter.list_candidates(include_resolved=False)) == 6
    assert len(content_filter.candidates("1698225", include_resolved=False)) == 5
    assert content_filter.status_counts("123") == dict.fromkeys(danmaku_filter.CANDIDATE_STATUSES, 0)
    assert other in content_filter.list_candidates()


def test_page_observed_archives_exact_queue_without_diagnosing_banned_words(content_filter):
    row = content_filter.record_failure("1698225", "【完整失败句】", "platform_f", queue_id=44)
    diagnosis = {"category": "platform_f", "action": "retry", "message": "原因未确定 按原顺序重试"}
    recovering = content_filter.update_recovery("1698225", row["text"], 44, diagnosis=diagnosis)
    assert recovering["status"] == "recovering" and recovering["diagnosis"] == diagnosis
    resolved = content_filter.resolve_success("1698225", row["text"], 44)
    assert resolved["status"] == "resolved" and resolved["resolution"] == "page_observed"
    assert resolved["reason"] == "platform_f" and resolved["count"] == 1
    assert resolved["last_resolution"] == {"at": resolved["resolved_at"], "resolution": "page_observed", "queue_id": 44}
    assert datetime.fromisoformat(resolved["resolved_at"]).utcoffset() == timedelta(hours=8)
    assert content_filter.list_candidates(include_resolved=False) == []
    assert content_filter.rules() == []
    assert content_filter.apply(row["text"], "1698225") == (row["text"], 0)
    assert DanmakuContentFilter(content_filter.path).candidates() == [resolved]


@pytest.mark.parametrize("operation", ["update_recovery", "resolve_success"])
@pytest.mark.parametrize("room,text,queue_id", [
    ("21399689", "【完整失败句】", 44), ("1698225", "【另一个失败句】", 44),
    ("1698225", "【完整失败句】", 45), ("1698225", "【完整失败句】", None)])
def test_recovery_never_matches_another_room_text_or_linked_queue(content_filter, operation, room, text, queue_id):
    row = content_filter.record_failure("1698225", "【完整失败句】", "platform_f", queue_id=44)
    before = content_filter.path.read_bytes()
    assert getattr(content_filter, operation)(room, text, queue_id) is None
    assert content_filter.path.read_bytes() == before
    assert content_filter.candidates() == [row]


@pytest.mark.parametrize("operation", ["update_recovery", "resolve_success"])
def test_recovery_attaches_exact_legacy_observation_to_queue(content_filter, operation):
    row = content_filter.record_failure("1698225", "【完整失败句】", "platform_f")
    updated = getattr(content_filter, operation)("1698225", row["text"], 44)
    assert updated["id"] == row["id"] and updated["queue_id"] == 44
    assert updated["status"] == ("recovering" if operation == "update_recovery" else "resolved")


def test_new_failure_reopens_record_and_preserves_prior_recovery_history(content_filter):
    first = content_filter.record_failure("1698225", "【完整失败句】", "platform_f", queue_id=44)
    resolved = content_filter.resolve_success("1698225", first["text"], 44)
    again = content_filter.record_failure("1698225", first["text"], "platform_rejected", queue_id=45)
    assert again["id"] == first["id"] and again["first_time"] == first["first_time"]
    assert again["status"] == "active" and again["count"] == 2 and again["queue_id"] == 45
    assert again["resolved_at"] is None and again["resolution"] == ""
    assert again["last_resolution"] == resolved["last_resolution"]
    assert again["diagnosis"] is None
    assert content_filter.resolve_success("1698225", first["text"], 44) is None
    assert content_filter.list_candidates(include_resolved=False) == [again]


def test_recovery_poll_cannot_reopen_archived_observation(content_filter):
    row = content_filter.record_failure("1698225", "【完整失败句】", "platform_f", queue_id=44)
    resolved = content_filter.resolve_success("1698225", row["text"], 44)
    before = content_filter.path.read_bytes()
    assert content_filter.update_recovery("1698225", row["text"], 44) == resolved
    assert content_filter.path.read_bytes() == before


def test_recovery_updates_are_idempotent_and_do_not_invalidate_rule_revision(content_filter, monkeypatch):
    row = content_filter.record_failure("1698225", "【完整失败句】", "platform_f", queue_id=44)
    revision = content_filter.revision
    original = danmaku_filter.os.replace
    writes = []
    def count_replace(source, destination):
        writes.append(destination)
        return original(source, destination)
    monkeypatch.setattr(danmaku_filter.os, "replace", count_replace)
    content_filter.update_recovery("1698225", row["text"], 44)
    content_filter.update_recovery("1698225", row["text"], 44)
    content_filter.set_candidate_status(row["id"], "recovering")
    assert len(writes) == 1
    resolved = content_filter.resolve_success("1698225", row["text"], 44)
    assert content_filter.resolve_success("1698225", row["text"], 44) == resolved
    assert len(writes) == 2 and content_filter.revision == revision


@pytest.mark.parametrize("queue_id", [0, -1, True, False, "44", 2**63, 1.5])
def test_invalid_candidate_queue_link_is_rejected_without_persisting(content_filter, queue_id):
    with pytest.raises(DanmakuFilterError):
        content_filter.record_failure("1698225", "【完整失败句】", "platform_f", queue_id=queue_id)
    assert not content_filter.path.exists()


@pytest.mark.parametrize("diagnosis", [[], "guess", {}, {"category": "f", "action": "retry"},
    {"category": "f", "action": "retry", "message": "说明", "token": "not allowed"},
    {"category": "F", "action": "retry", "message": "说明"},
    {"category": "f", "action": True, "message": "说明"},
    {"category": "f", "action": "retry", "message": "a\x00b"},
    {"category": "f", "action": "retry", "message": "a" * 513}])
def test_invalid_candidate_diagnosis_is_rejected_without_persisting(content_filter, diagnosis):
    with pytest.raises(DanmakuFilterError):
        content_filter.record_failure("1698225", "【完整失败句】", "platform_f", diagnosis=diagnosis)
    assert not content_filter.path.exists()


@pytest.mark.parametrize("operation", ["set_candidate_status", "update_recovery", "resolve_success"])
def test_recovery_atomic_failure_keeps_original_observation(content_filter, monkeypatch, operation):
    row = content_filter.record_failure("1698225", "【完整失败句】", "platform_f", queue_id=44)
    before, revision = content_filter.path.read_bytes(), content_filter.revision
    def fail_replace(*_args):
        raise PermissionError("test file locked")
    monkeypatch.setattr(danmaku_filter.os, "replace", fail_replace)
    with pytest.raises(DanmakuFilterError, match="原规则未修改"):
        if operation == "set_candidate_status":
            content_filter.set_candidate_status(row["id"], "action_required")
        else:
            getattr(content_filter, operation)("1698225", row["text"], 44)
    assert content_filter.path.read_bytes() == before
    assert content_filter.candidates() == [row] and content_filter.revision == revision


@pytest.mark.parametrize("field,value", [("status", "unknown"), ("status", True), ("queue_id", 0),
    ("queue_id", True), ("diagnosis", []), ("resolved_at", "2026-10-11T12:00:00+00:00"),
    ("resolution", "source guaranteed"), ("status", "resolved"),
    ("last_resolution", {"at": "2026-10-11T12:00:00+08:00", "resolution": "page_observed", "queue_id": -1})])
def test_corrupt_candidate_recovery_fields_cannot_silently_load(content_filter, field, value):
    content_filter.record_failure("1698225", "【完整失败句】", "platform_f", queue_id=44)
    source = json.loads(content_filter.path.read_text(encoding="utf-8"))
    source["candidates"][0][field] = value
    content_filter.path.write_text(json.dumps(source, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(DanmakuFilterError, match="损坏或格式无效"):
        DanmakuContentFilter(content_filter.path)


def test_candidate_requires_manual_selected_term_to_promote(content_filter):
    candidate = content_filter.record_failure("1698225", "【待确认词出现在这条字幕】", "platform_f")
    with pytest.raises(DanmakuFilterError, match="必须来自"):
        content_filter.promote_candidate(candidate["id"], "凭空猜测")
    assert content_filter.rules() == []
    rule = content_filter.promote_candidate(candidate["id"], "待确认词", "中性内容")
    assert rule["room"] == ""
    assert content_filter.apply(candidate["text"], "1698225") == ("【中性内容出现在这条字幕】", 1)
    assert content_filter.apply(candidate["text"], "21399689")[1] == 1
    assert content_filter.candidates() == [candidate]


def test_promote_can_explicitly_select_global_scope(content_filter):
    candidate = content_filter.record_failure("1698225", "【一个词语】", "platform_f")
    rule = content_filter.promote_candidate(candidate["id"], "词语", room="")
    assert rule["room"] == ""
    assert content_filter.apply("词语", "21399689") == ("已过滤", 1)


def test_promote_disabled_rule_is_one_atomic_write(content_filter, monkeypatch):
    candidate = content_filter.record_failure("1698225", "【一个词语】", "platform_f")
    original = danmaku_filter.os.replace
    writes = []
    def count_replace(source, destination):
        writes.append(destination)
        return original(source, destination)
    monkeypatch.setattr(danmaku_filter.os, "replace", count_replace)
    row = content_filter.promote_candidate(candidate["id"], "词语", enabled=False)
    assert not row["enabled"]
    assert len(writes) == 1
    assert content_filter.apply("词语", "1698225") == ("词语", 0)
    assert not DanmakuContentFilter(content_filter.path).rules()[0]["enabled"]


def test_persistence_and_returned_lists_are_independent(content_filter):
    rule = content_filter.add_rule("本地词", "本地表达", "1698225")
    candidate = content_filter.record_failure("1698225", "【本地词句子】", "platform_f")
    content_filter.rules()[0]["term"] = "篡改"
    content_filter.candidates()[0]["count"] = 999
    rule["replacement"] = "篡改"
    candidate["reason"] = "篡改"
    restored = DanmakuContentFilter(content_filter.path)
    assert restored.list_rules()[0]["replacement"] == "本地表达"
    assert restored.candidates()[0]["count"] == 1
    assert restored.candidates()[0]["reason"] == "platform_f"
    assert restored.apply("本地词", "1698225") == ("本地表达", 1)
    assert "本地词" in content_filter.path.read_text(encoding="utf-8")


def test_edit_remove_duplicate_and_missing_records(content_filter):
    row = content_filter.add_rule("词语", "正常", "1698225")
    with pytest.raises(DanmakuFilterError, match="相同过滤词"):
        content_filter.add_rule("词语", "其他", "1698225")
    another = content_filter.add_rule("另词", "正常", "1698225")
    with pytest.raises(DanmakuFilterError, match="相同过滤词"):
        content_filter.update_rule(another["id"], term="词语")
    changed = content_filter.update_rule(row["id"], term="改词", replacement="新表达", room="21399689")
    assert changed["created_at"] == row["created_at"]
    assert content_filter.apply("改词", "21399689") == ("新表达", 1)
    assert content_filter.remove_rule(row["id"])
    assert not content_filter.remove_rule(row["id"])
    with pytest.raises(DanmakuFilterError, match="不存在"):
        content_filter.update_rule(row["id"], term="改词")
    candidate = content_filter.record_failure("1698225", "【失败内容】", "platform_f")
    assert content_filter.remove_candidate(candidate["id"])
    assert not content_filter.remove_candidate(candidate["id"])
    with pytest.raises(DanmakuFilterError, match="不存在"):
        content_filter.promote_candidate(candidate["id"], "内容")


@pytest.mark.parametrize("replacement", ["", "   "])
def test_empty_replacement_uses_safe_neutral_default(content_filter, replacement):
    row = content_filter.add_rule("词", replacement)
    assert row["replacement"] == "已过滤"
    assert content_filter.apply("【词】") == ("【已过滤】", 1)
    assert content_filter.update_rule(row["id"], replacement=replacement)["replacement"] == "已过滤"


@pytest.mark.parametrize("term", ["", "   ", "a\x00b", "a\nb", "a\u202eb", "a\ud800b", "a" * 81, "😀" * 41, 1, None])
def test_invalid_term_cannot_change_or_create_file(content_filter, term):
    with pytest.raises(DanmakuFilterError):
        content_filter.add_rule(term)
    assert content_filter.rules() == []
    assert not content_filter.path.exists()


@pytest.mark.parametrize("replacement", ["a\x00b", "a\nb", "a\u202eb", "a\ud800b", "a" * 81, "😀" * 41, None, 1])
def test_unsafe_replacement_is_rejected(content_filter, replacement):
    with pytest.raises(DanmakuFilterError):
        content_filter.add_rule("词", replacement)
    assert content_filter.rules() == []


def test_exact_utf16_limit_accepts_emoji(content_filter):
    content_filter.add_rule("😀" * 40, "字" * 80)
    assert content_filter.apply("😀" * 40) == ("字" * 80, 1)


@pytest.mark.parametrize("room", ["0", "-1", "https://live.bilibili.com/1698225", "１２３", "12x", 1698225, True])
def test_invalid_room_does_not_persist(content_filter, room):
    with pytest.raises(DanmakuFilterError):
        content_filter.add_rule("词", room=room)
    assert not content_filter.path.exists()


@pytest.mark.parametrize("payload", ["{broken", "[]", '{"version":2}', '{"version":true}',
    '{"version":1,"enabled":"yes","rules":[],"candidates":[]}'])
def test_corrupt_json_raises_explicit_error(tmp_path, payload):
    path = tmp_path / "filter.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(DanmakuFilterError, match="损坏或格式无效"):
        DanmakuContentFilter(path)
    assert path.read_text(encoding="utf-8") == payload


def test_invalid_utf8_has_readable_error(tmp_path):
    path = tmp_path / "filter.json"
    path.write_bytes(b"\xff\xfe")
    with pytest.raises(DanmakuFilterError, match="无法读取"):
        DanmakuContentFilter(path)


@pytest.mark.parametrize("operation", ["add", "update", "remove", "candidate", "toggle"])
def test_atomic_replace_failure_keeps_disk_and_memory(content_filter, monkeypatch, operation):
    row = content_filter.add_rule("原词", "原表达")
    before_disk = content_filter.path.read_bytes()
    before_rules = content_filter.rules()
    before_revision = content_filter.revision
    def fail_replace(*_args):
        raise PermissionError("test file locked")
    monkeypatch.setattr(danmaku_filter.os, "replace", fail_replace)
    with pytest.raises(DanmakuFilterError, match="原规则未修改"):
        if operation == "add":
            content_filter.add_rule("新词", "新表达")
        elif operation == "update":
            content_filter.update_rule(row["id"], replacement="新表达")
        elif operation == "remove":
            content_filter.remove_rule(row["id"])
        elif operation == "candidate":
            content_filter.record_failure("1698225", "【失败词】", "platform_f")
        else:
            content_filter.set_enabled(False)
    assert content_filter.path.read_bytes() == before_disk
    assert content_filter.rules() == before_rules
    assert content_filter.candidates() == []
    assert content_filter.revision == before_revision
    assert content_filter.apply("原词") == ("原表达", 1)
    assert content_filter.filter_error == content_filter.last_error
    assert "test file locked" in content_filter.last_error
    assert not list(content_filter.path.parent.glob("*.tmp"))


def test_successful_save_clears_previous_io_error(content_filter, monkeypatch):
    original = danmaku_filter.os.replace
    def fail_replace(*_args):
        raise PermissionError("test file locked")
    monkeypatch.setattr(danmaku_filter.os, "replace", fail_replace)
    with pytest.raises(DanmakuFilterError):
        content_filter.add_rule("词")
    monkeypatch.setattr(danmaku_filter.os, "replace", original)
    content_filter.add_rule("词")
    assert content_filter.last_error == ""


def test_revision_ignores_observation_and_identity_only_changes(content_filter):
    row = content_filter.add_rule("词", "表达")
    version = content_filter.revision
    content_filter.record_failure("1698225", "【词】", "platform_f")
    content_filter.update_rule(row["id"], term="词", replacement="表达")
    assert content_filter.revision == version
    content_filter.remove_rule(row["id"])
    content_filter.add_rule("词", "表达")
    assert content_filter.revision == version
    content_filter.set_enabled(False)
    assert content_filter.revision != version


@pytest.mark.parametrize("field,value", [("enabled", 1), ("count", True), ("count", 0), ("last_time", "2026-01-01T00:00:00+00:00")])
def test_structurally_corrupt_records_cannot_silently_load(content_filter, field, value):
    content_filter.add_rule("词")
    content_filter.record_failure("1698225", "【词】", "platform_f")
    source = json.loads(content_filter.path.read_text(encoding="utf-8"))
    if field == "enabled":
        source["rules"][0][field] = value
    else:
        source["candidates"][0][field] = value
    content_filter.path.write_text(json.dumps(source, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(DanmakuFilterError, match="损坏或格式无效"):
        DanmakuContentFilter(content_filter.path)


def write_detection_dictionary(path, terms=("测试词",)):
    payload = {"version": 1, "sources": [{"id": "fixture", "name": "Fixture",
        "url": "https://example.invalid", "revision": "test", "license": "MIT", "license_text": "fixture"}],
        "entries": [{"term": term, "category": "fixture", "source_id": "fixture"} for term in terms],
        "allowlist": []}
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_failed_sentence_is_automatically_detected_and_never_creates_replacement(content_filter):
    write_detection_dictionary(content_filter.detector_path)
    content_filter.reload_detector()
    candidate = content_filter.record_failure("1698225", "【出现测试词的完整句子】", "platform_f")
    analysis = candidate["analysis"]
    assert analysis["status"] == "checked" and analysis["summary_code"] == "advisory_hits"
    assert analysis["hits"][0]["term"] == "测试词"
    assert analysis["hits"][0]["matched"] == "测试词"
    assert datetime.fromisoformat(analysis["checked_at"]).utcoffset() == timedelta(hours=8)
    assert content_filter.rules() == []
    assert content_filter.apply(candidate["text"], "1698225") == (candidate["text"], 0)
    assert DanmakuContentFilter(content_filter.path).candidates()[0]["analysis"] == analysis


def test_repeat_f_scans_once_per_dictionary_and_rule_revision(content_filter, monkeypatch):
    write_detection_dictionary(content_filter.detector_path)
    content_filter.reload_detector()
    calls = []
    real_detect = content_filter._detector.detect
    def detect(text, room=""):
        calls.append(text)
        return real_detect(text, room)
    monkeypatch.setattr(content_filter._detector, "detect", detect)
    first = content_filter.record_failure("1698225", "【测试词】", "platform_f")
    for _ in range(5):
        content_filter.record_failure("1698225", "【测试词】", "platform_f")
    assert calls == ["【测试词】"]
    assert content_filter.candidates()[0]["analysis"] == first["analysis"]
    assert content_filter.candidates()[0]["count"] == 6
    assert content_filter.analyze_candidate(first["id"]) == first["analysis"]
    assert content_filter.analyze_candidates()[0]["analysis"] == first["analysis"]
    assert len(calls) == 1
    content_filter.add_rule("测试词", "普通表达")
    after_rule = content_filter.record_failure("1698225", "【测试词】", "platform_f")
    assert len(calls) == 2
    assert after_rule["analysis"]["known_rules"] == [{"term": "测试词", "replacement": "普通表达", "action": "replace"}]
    write_detection_dictionary(content_filter.detector_path, ("新词",))
    content_filter.reload_detector()
    after_dictionary = content_filter.record_failure("1698225", "【测试词】", "platform_f")
    assert len(calls) == 3 and after_dictionary["analysis"]["summary_code"] == "no_hits"
    content_filter.analyze_candidate(first["id"], force=True)
    assert len(calls) == 4


def test_unavailable_or_bad_dictionary_never_blocks_rules_or_failure_recording(content_filter):
    missing = content_filter.record_failure("1698225", "【完全正常的句子】", "platform_f")
    assert missing["analysis"]["summary_code"] == "no_wordlist"
    content_filter.detector_path.write_text("{broken", encoding="utf-8")
    assert content_filter.reload_detector()["status"] == "error"
    failed = content_filter.record_failure("1698225", "【完全正常的句子】", "platform_f")
    assert failed["analysis"]["status"] == "error" and failed["count"] == 2
    content_filter.add_rule("正常", "普通")
    assert content_filter.apply("正常") == ("普通", 1)
    assert content_filter.last_error == ""


def test_legacy_candidates_are_backfilled_in_one_write(content_filter, monkeypatch):
    first = content_filter.record_failure("1698225", "【第一条失败句】", "platform_f")
    content_filter.record_failure("1698225", "【第二条失败句】", "platform_f")
    source = json.loads(content_filter.path.read_text(encoding="utf-8"))
    for row in source["candidates"]:
        row.pop("analysis")
    content_filter.path.write_text(json.dumps(source, ensure_ascii=False), encoding="utf-8")
    restored = DanmakuContentFilter(content_filter.path)
    writes = []
    real_save = restored._save
    def save(state):
        writes.append(1)
        real_save(state)
    monkeypatch.setattr(restored, "_save", save)
    assert len(restored.analyze_candidates()) == 2 and len(writes) == 1
    restored.analyze_candidates()
    assert len(writes) == 1
    assert restored.candidates()[0]["id"] == first["id"]


def test_allowlist_edit_reanalyzes_saved_candidates_without_activating_rules(content_filter):
    write_detection_dictionary(content_filter.detector_path)
    content_filter.reload_detector()
    candidate = content_filter.record_failure("1698225", "【测试词】", "platform_f")
    content_filter.set_allowlist(["测试词"])
    current = content_filter.candidates()[0]
    assert current["id"] == candidate["id"] and current["count"] == 1
    assert current["analysis"]["summary_code"] == "no_hits"
    assert content_filter.get_allowlist() == ["测试词"]
    assert content_filter.detector_entries()[0]["term"] == "测试词"
    assert content_filter.detector_sources()[0]["id"] == "fixture"
    assert content_filter.rules() == []


def test_cached_analysis_survives_restart_without_repeat_scan(content_filter, monkeypatch):
    write_detection_dictionary(content_filter.detector_path)
    content_filter.reload_detector()
    candidate = content_filter.record_failure("1698225", "【测试词】", "platform_f")
    restored = DanmakuContentFilter(content_filter.path)
    def unexpected_detect(_text, _room=""):
        raise AssertionError("cached analysis should not scan again")
    monkeypatch.setattr(restored._detector, "detect", unexpected_detect)
    assert restored.analyze_candidates()[0]["analysis"] == candidate["analysis"]
    assert restored.record_failure("1698225", "【测试词】", "platform_f")["count"] == 2


@pytest.mark.parametrize("mutation", [
    lambda analysis: analysis.update(engine="unknown"),
    lambda analysis: analysis.update(checked_at="2026-01-01T00:00:00+00:00"),
    lambda analysis: analysis.update(known_rules=["not a rule"]),
    lambda analysis: analysis["hits"][0].update(start=-1),
    lambda analysis: analysis["hits"][0].update(matched="wrong text"),
])
def test_invalid_imported_analysis_is_not_silently_trusted(content_filter, mutation):
    write_detection_dictionary(content_filter.detector_path)
    content_filter.reload_detector()
    content_filter.record_failure("1698225", "【测试词】", "platform_f")
    source = json.loads(content_filter.path.read_text(encoding="utf-8"))
    mutation(source["candidates"][0]["analysis"])
    content_filter.path.write_text(json.dumps(source, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(DanmakuFilterError, match="损坏或格式无效"):
        DanmakuContentFilter(content_filter.path)


def test_known_rules_include_enabled_rules_from_all_former_room_scopes(content_filter):
    content_filter.add_rule("全局", "通用")
    content_filter.add_rule("本房", "房间内容", "1698225")
    content_filter.add_rule("其他", "其他内容", "21399689")
    content_filter.add_rule("停用", "未启用", enabled=False)
    result = content_filter.detect("全局 本房 其他 停用", "1698225")
    assert result["known_rules"] == [{"term": "全局", "replacement": "通用", "action": "replace"},
                                     {"term": "本房", "replacement": "房间内容", "action": "replace"},
                                     {"term": "其他", "replacement": "其他内容", "action": "replace"}]
    assert content_filter.detect("全局 本房 其他 停用", "21399689")["known_rules"] == result["known_rules"]
    content_filter.set_enabled(False)
    assert content_filter.detect("全局 本房", "1698225")["known_rules"] == []


def test_room_scoped_candidate_analysis_persists_scope_and_accepts_legacy_hits(content_filter):
    write_detection_dictionary(content_filter.detector_path, ("评价",))
    payload = json.loads(content_filter.detector_path.read_text(encoding="utf-8"))
    payload["entries"][0]["room"] = "1698225"
    content_filter.detector_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    content_filter.reload_detector()
    observed = content_filter.record_failure("1698225", "【评价】", "platform_f")
    unrelated = content_filter.record_failure("21399689", "【评价】", "platform_f")
    assert observed["analysis"]["hits"][0]["room"] == "1698225"
    assert unrelated["analysis"]["hits"] == []
    restored = DanmakuContentFilter(content_filter.path)
    assert restored.candidates()[0]["analysis"] == observed["analysis"]
    legacy = json.loads(content_filter.path.read_text(encoding="utf-8"))
    legacy["candidates"][0]["analysis"]["hits"][0].pop("room")
    content_filter.path.write_text(json.dumps(legacy, ensure_ascii=False), encoding="utf-8")
    assert DanmakuContentFilter(content_filter.path).candidates()[0]["analysis"]["hits"][0]["room"] == ""
    assert content_filter.rules() == []


def test_explicit_remove_action_deletes_literal_word_in_all_rooms(content_filter):
    rule = content_filter.add_rule("评价", "ignored replacement", "1698225", action="remove")
    assert rule["action"] == "remove" and rule["replacement"] == ""
    assert content_filter.apply("【不过 谢谢 恭喜获得评价 谢谢】", "1698225") == ("【不过 谢谢 恭喜获得 谢谢】", 1)
    assert content_filter.apply("【不过 谢谢 恭喜获得评价 谢谢】", "21399689") == ("【不过 谢谢 恭喜获得 谢谢】", 1)
    restored = DanmakuContentFilter(content_filter.path)
    assert restored.rules()[0]["action"] == "remove" and restored.rules()[0]["replacement"] == ""
    assert restored.apply("评价评价", "1698225") == ("", 2)


def test_remove_respects_longest_match_and_single_original_scan_in_all_rooms(content_filter):
    content_filter.add_rule("甲乙", action="remove")
    content_filter.add_rule("乙丙", "改写")
    content_filter.add_rule("甲", "丙")
    content_filter.add_rule("丙", action="remove")
    assert content_filter.apply("甲乙丙 甲") == (" 丙", 3)
    row = content_filter.add_rule("评价", "通用表达")
    content_filter.update_rule(row["id"], room="1698225", action="remove")
    assert content_filter.apply("评价", "1698225") == ("", 1)
    assert content_filter.apply("评价", "21399689") == ("", 1)


def test_update_switches_action_and_changes_revision_without_resetting_remove_on_other_edits(content_filter):
    row = content_filter.add_rule("评价", "普通表达")
    original_revision = content_filter.revision
    removed = content_filter.update_rule(row["id"], action="remove")
    assert removed["action"] == "remove" and removed["replacement"] == ""
    removed_revision = content_filter.revision
    assert removed_revision != original_revision
    content_filter.update_rule(row["id"], enabled=False)
    unchanged_action = content_filter.update_rule(row["id"], enabled=True, replacement="仍然忽略")
    assert unchanged_action["action"] == "remove" and unchanged_action["replacement"] == ""
    assert content_filter.revision == removed_revision
    replaced = content_filter.update_rule(row["id"], action="replace")
    assert replaced["replacement"] == "已过滤"
    assert content_filter.apply("评价") == ("已过滤", 1)
    assert content_filter.revision != removed_revision


def test_legacy_empty_replace_stays_neutral_default_and_missing_action_defaults_to_replace(content_filter):
    row = content_filter.add_rule("旧词", "")
    source = json.loads(content_filter.path.read_text(encoding="utf-8"))
    source["rules"][0].pop("action")
    source["rules"][0]["replacement"] = ""
    content_filter.path.write_text(json.dumps(source, ensure_ascii=False), encoding="utf-8")
    restored = DanmakuContentFilter(content_filter.path)
    assert restored.rules()[0]["id"] == row["id"]
    assert restored.rules()[0]["action"] == "replace"
    assert restored.rules()[0]["replacement"] == "已过滤"
    assert restored.apply("旧词") == ("已过滤", 1)


def test_promote_remove_keeps_candidate_and_analysis_explicitly_describes_removal(content_filter):
    candidate = content_filter.record_failure("1698225", "【评价与普通内容】", "platform_f")
    row = content_filter.promote_candidate(candidate["id"], "评价", action="remove")
    assert row["room"] == "" and row["replacement"] == "" and row["action"] == "remove"
    assert content_filter.candidates()[0]["text"] == candidate["text"]
    analysis = content_filter.analyze_candidate(candidate["id"])
    assert analysis["known_rules"] == [{"term": "评价", "replacement": "", "action": "remove"}]
    assert DanmakuContentFilter(content_filter.path).candidates()[0]["analysis"] == analysis


@pytest.mark.parametrize("action", ["", "drop", None, 1, True, ["remove"]])
def test_invalid_action_cannot_change_rules(content_filter, action):
    rule = content_filter.add_rule("原词", "原内容")
    before = content_filter.path.read_bytes()
    with pytest.raises(DanmakuFilterError, match="过滤动作"):
        content_filter.add_rule("新词", action=action)
    with pytest.raises(DanmakuFilterError, match="过滤动作"):
        content_filter.update_rule(rule["id"], action=action)
    assert content_filter.path.read_bytes() == before
    assert content_filter.apply("原词") == ("原内容", 1)


def test_remove_action_atomic_save_failure_preserves_original_rule(content_filter, monkeypatch):
    rule = content_filter.add_rule("评价", "原表达")
    before = content_filter.path.read_bytes()
    revision = content_filter.revision
    def fail_replace(*_args):
        raise PermissionError("test locked")
    monkeypatch.setattr(danmaku_filter.os, "replace", fail_replace)
    with pytest.raises(DanmakuFilterError, match="原规则未修改"):
        content_filter.update_rule(rule["id"], action="remove")
    assert content_filter.path.read_bytes() == before and content_filter.revision == revision
    assert content_filter.rules()[0]["action"] == "replace"
    assert content_filter.apply("评价") == ("原表达", 1)


def test_rule_signatures_are_semantic_and_stable_across_id_time_and_reload(content_filter):
    first = content_filter.add_rule("旧词", "普通表达", "1698225")
    signature = content_filter.rule_signatures("1698225")
    assert len(signature) == 1 and len(signature[0]) == 64
    content_filter.update_rule(first["id"], term="旧词", replacement="普通表达")
    assert content_filter.rule_signatures("1698225") == signature
    content_filter.remove_rule(first["id"])
    recreated = content_filter.add_rule("旧词", "普通表达", "1698225")
    assert recreated["id"] != first["id"]
    assert content_filter.rule_signatures("1698225") == signature
    assert DanmakuContentFilter(content_filter.path).rule_signatures("1698225") == signature
    content_filter.update_rule(recreated["id"], action="remove")
    assert content_filter.rule_signatures("1698225") != signature
    assert content_filter.rule_signatures("21399689") == content_filter.rule_signatures("1698225")


def test_signatures_and_exclusion_are_shared_by_all_rooms(content_filter):
    content_filter.add_rule("同词", "全局表达")
    global_signatures = content_filter.rule_signatures("21399689")
    room_signatures = content_filter.rule_signatures("1698225")
    assert len(room_signatures) == len(global_signatures) == 1
    assert room_signatures == global_signatures
    assert content_filter.apply("同词", "1698225", excluded_rule_signatures=room_signatures) == ("同词", 0)
    assert content_filter.apply("同词", "1698225", excluded_rule_signatures=global_signatures) == ("同词", 0)
    assert content_filter.apply("同词", "21399689", excluded_rule_signatures=room_signatures) == ("同词", 0)


def test_excluded_prior_semantics_prevent_cross_version_cascade_after_unrelated_rule_edit(content_filter):
    content_filter.add_rule("A", "B")
    second = content_filter.add_rule("B", "C")
    baseline = content_filter.rule_signatures()
    original_final, hits = content_filter.apply("A")
    assert (original_final, hits) == ("B", 1)
    content_filter.add_rule("无关词", "无关表达")
    assert content_filter.apply(original_final, excluded_rule_signatures=baseline) == ("B", 0)
    assert content_filter.apply("无关词", excluded_rule_signatures=baseline) == ("无关表达", 1)
    content_filter.update_rule(second["id"], replacement="明确修改的表达")
    assert content_filter.apply(original_final, excluded_rule_signatures=baseline) == ("明确修改的表达", 1)


def test_disabled_master_and_disabled_rules_produce_no_signatures(content_filter):
    row = content_filter.add_rule("词", action="remove", enabled=False)
    assert content_filter.rule_signatures() == []
    content_filter.update_rule(row["id"], enabled=True)
    signatures = content_filter.rule_signatures()
    assert signatures
    content_filter.set_enabled(False)
    assert content_filter.rule_signatures() == []
    assert content_filter.apply("词", excluded_rule_signatures=signatures) == ("词", 0)


@pytest.mark.parametrize("signatures", [None, "a" * 64, ["bad"], [None], ["A" * 64], [1]])
def test_invalid_excluded_signature_list_is_rejected(content_filter, signatures):
    content_filter.add_rule("词", "普通")
    with pytest.raises(DanmakuFilterError, match="签名"):
        content_filter.apply("词", excluded_rule_signatures=signatures)


def legacy_rule_record(term, replacement, room, *, enabled=True, updated_at="2026-10-11T12:00:00.000+08:00"):
    return {"id": danmaku_filter.uuid.uuid4().hex, "term": term, "replacement": replacement,
            "action": "replace", "room": room, "enabled": enabled,
            "created_at": "2026-10-11T11:00:00.000+08:00", "updated_at": updated_at,
            "origin": "user", "evidence": None, "validation": None}


def write_legacy_rules(content_filter, rules):
    content_filter.path.write_text(json.dumps({"version": 1, "enabled": True,
        "rules": rules, "candidates": []}, ensure_ascii=False), encoding="utf-8")
    return DanmakuContentFilter(content_filter.path)


def test_legacy_room_rules_migrate_in_memory_and_persist_idempotently_with_full_audit(content_filter):
    originals = [legacy_rule_record("评价", "感想", "1698225"),
                 legacy_rule_record("风险词", "通用表达", "21399689")]
    restored = write_legacy_rules(content_filter, originals)
    untouched = content_filter.path.read_bytes()
    assert all(row["room"] == "" for row in restored.rules())
    assert restored.apply("评价 风险词", "999999") == ("感想 通用表达", 2)
    assert content_filter.path.read_bytes() == untouched
    restored.persist()
    persisted = content_filter.path.read_bytes()
    saved = json.loads(persisted)
    assert saved["scope_migration"]["legacy_rules"] == originals
    reloaded = DanmakuContentFilter(content_filter.path)
    assert reloaded.rules() == restored.rules()
    reloaded.persist()
    assert content_filter.path.read_bytes() == persisted
    assert [row["id"] for row in restored.rules()] == [row["id"] for row in originals]


@pytest.mark.parametrize("old_global,newer_enabled", [(True, True), (True, False), (False, False)])
def test_legacy_conflicts_keep_deliberate_global_or_latest_manual_choice(content_filter, old_global, newer_enabled):
    older = legacy_rule_record("同词", "旧表达", "" if old_global else "111", enabled=False)
    newer = legacy_rule_record("同词", "新表达", "222", enabled=newer_enabled,
                               updated_at="2026-10-11T13:00:00.000+08:00")
    originals = [older, newer]
    restored = write_legacy_rules(content_filter, originals)
    expected = older if old_global else newer
    assert len(restored.rules()) == 1 and restored.rules()[0]["id"] == expected["id"]
    assert restored.apply("同词", "333") == ((expected["replacement"], 1) if expected["enabled"] else ("同词", 0))
    restored.persist()
    assert json.loads(content_filter.path.read_text(encoding="utf-8"))["scope_migration"]["legacy_rules"] == originals


def test_legacy_conflict_resolution_is_independent_of_file_order(content_filter):
    records = [legacy_rule_record("同词", "旧表达", "111"),
               legacy_rule_record("同词", "新表达", "222", updated_at="2026-10-11T13:00:00.000+08:00")]
    first = write_legacy_rules(content_filter, records).rules()
    second = write_legacy_rules(content_filter, list(reversed(records))).rules()
    assert first == second


def test_legacy_validated_ai_migrates_globally_without_changing_proof_or_candidate(content_filter):
    candidate, plan = staged_validation(content_filter)
    content_filter.record_validation_receipt(candidate["id"], request_id=plan["request_id"], before_count=1,
                                             receipt_accepted=True, code=0, http_status=200)
    resolved = content_filter.confirm_validation(candidate["id"], evidence=validation_evidence(plan))
    document = json.loads(content_filter.path.read_text(encoding="utf-8"))
    original = document["rules"][0]
    original["room"] = "1698225"
    content_filter.path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    restored = DanmakuContentFilter(content_filter.path)
    rule = restored.rules()[0]
    assert rule["room"] == "" and rule["id"] == original["id"]
    assert rule["evidence"] == original["evidence"] and rule["validation"] == original["validation"]
    assert restored.candidates()[0] == resolved
    assert restored.apply("评价", "987654321") == ("感想", 1)
    assert resolved["validation"]["validated_rule_ids"] == [rule["id"]]
    restored.persist()
    saved = json.loads(content_filter.path.read_text(encoding="utf-8"))
    assert saved["scope_migration"]["legacy_rules"] == [original]


def test_manual_disabled_rule_wins_over_newer_enabled_validated_ai_without_losing_evidence(content_filter):
    candidate, plan = staged_validation(content_filter)
    content_filter.record_validation_receipt(candidate["id"], request_id=plan["request_id"], before_count=1,
                                             receipt_accepted=True, code=0, http_status=200)
    content_filter.confirm_validation(candidate["id"], evidence=validation_evidence(plan))
    document = json.loads(content_filter.path.read_text(encoding="utf-8"))
    learned = document["rules"][0]
    learned["room"] = "1698225"
    manual = legacy_rule_record("评价", "人工表达", "21399689", enabled=False)
    document["rules"].append(manual)
    original_candidate = document["candidates"][0]
    content_filter.path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    restored = DanmakuContentFilter(content_filter.path)
    assert restored.rules()[0]["id"] == manual["id"] and not restored.rules()[0]["enabled"]
    assert restored.apply("评价", "1698225") == ("评价", 0)
    assert restored.candidates()[0] == original_candidate
    restored.persist()
    archived = json.loads(content_filter.path.read_text(encoding="utf-8"))["scope_migration"]["legacy_rules"]
    assert archived == [learned, manual]
    assert original_candidate["validation"]["validated_rule_ids"] == [learned["id"]]


def test_global_migration_keeps_unvalidated_ai_disabled_and_original_record_for_audit(content_filter):
    content_filter, _ = legacy_observed_rule(content_filter)
    document = json.loads(content_filter.path.read_text(encoding="utf-8"))
    document["rules"][0].update(room="1698225", enabled=True)
    original = dict(document["rules"][0])
    content_filter.path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    restored = DanmakuContentFilter(content_filter.path)
    assert not restored.rules()[0]["enabled"]
    assert restored.apply("评价", "21399689") == ("评价", 0)
    with pytest.raises(DanmakuFilterError, match="尚未通过"):
        restored.update_rule(original["id"], enabled=True)
    restored.persist()
    assert json.loads(content_filter.path.read_text(encoding="utf-8"))["scope_migration"]["legacy_rules"] == [original]


def test_legacy_signatures_preserve_applied_rule_history_and_prevent_cascade_after_scope_migration(content_filter):
    records = [legacy_rule_record("甲词", "乙词", "1698225"),
               legacy_rule_record("乙词", "丙词", "1698225")]
    signatures = [DanmakuContentFilter._rule_signature(row) for row in records]
    restored = write_legacy_rules(content_filter, records)
    assert restored.apply("甲词", "1698225") == ("乙词", 1)
    assert restored.apply("乙词", "1698225", excluded_rule_signatures=signatures) == ("乙词", 0)
    restored.add_rule("无关词", "另一个表达")
    assert restored.apply("乙词", "1698225", excluded_rule_signatures=signatures) == ("乙词", 0)
    restored.persist()
    reloaded = DanmakuContentFilter(content_filter.path)
    assert reloaded.apply("乙词", "1698225", excluded_rule_signatures=signatures) == ("乙词", 0)
    reloaded.update_rule(records[1]["id"], replacement="明确的新表达")
    assert "signature_aliases" not in reloaded.rules()[1]
    assert reloaded.apply("乙词", "1698225", excluded_rule_signatures=signatures) == ("明确的新表达", 1)


def test_conflict_aliases_cover_equal_semantics_and_never_exclude_different_winning_replacement(content_filter):
    same = [legacy_rule_record("同词", "通用表达", "111"), legacy_rule_record("同词", "通用表达", "222")]
    restored = write_legacy_rules(content_filter, same)
    for old in same:
        assert restored.apply("同词", "333", excluded_rule_signatures=[restored._rule_signature(old)]) == ("同词", 0)
    different = [legacy_rule_record("同词", "旧表达", "111"),
                 legacy_rule_record("同词", "新表达", "222", updated_at="2026-10-11T13:00:00.000+08:00")]
    restored = write_legacy_rules(content_filter, different)
    assert restored.apply("同词", "333", excluded_rule_signatures=[restored._rule_signature(different[0])]) == ("新表达", 1)


def test_changing_only_legacy_room_argument_does_not_reset_validated_origin_or_evidence(content_filter):
    candidate, plan = staged_validation(content_filter)
    content_filter.record_validation_receipt(candidate["id"], request_id=plan["request_id"], before_count=1,
                                             receipt_accepted=True, code=0, http_status=200)
    content_filter.confirm_validation(candidate["id"], evidence=validation_evidence(plan))
    original = content_filter.rules()[0]
    changed = content_filter.update_rule(original["id"], room="999")
    assert changed["room"] == "" and changed["origin"] == original["origin"]
    assert changed["evidence"] == original["evidence"] and changed["validation"] == original["validation"]


@pytest.mark.parametrize("migration", [[], {"version": True, "migrated_at": "bad", "legacy_rules": []},
    {"version": 1, "migrated_at": "2026-10-11T12:00:00.000+08:00", "legacy_rules": [{}]}])
def test_corrupt_migration_audit_fails_without_overwriting_original_file(content_filter, migration):
    source = {"version": 1, "enabled": True, "rules": [], "candidates": [], "scope_migration": migration}
    content_filter.path.write_text(json.dumps(source), encoding="utf-8")
    before = content_filter.path.read_bytes()
    with pytest.raises(DanmakuFilterError, match="损坏或格式无效"):
        DanmakuContentFilter(content_filter.path)
    assert content_filter.path.read_bytes() == before
