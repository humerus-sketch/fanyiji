"""Isolated validation uses fake browser replies and real durable candidate storage."""
from unittest.mock import Mock, patch

import pytest

from test_danmaku_service import app, FakeBrowser, enqueue
from fanyiji.danmaku_filter import DanmakuContentFilter
from fanyiji.danmaku_service import DanmakuService
from fanyiji.danmaku_validation import DanmakuValidationRunner
from fanyiji.danmaku_ai import validate_analysis_result


FORMAL_ROOM = "21399689"
TEST_ROOM = "1698225"
ORIGINAL = "感谢你的评价我会继续努力"
REVISED = "感谢你的感想我会继续努力"


@pytest.fixture
def validation(app, tmp_path):
    content_filter = DanmakuContentFilter(tmp_path / "filter.json")
    candidate = content_filter.record_failure(FORMAL_ROOM, ORIGINAL, "platform_f")
    content_filter.save_validation_plan(candidate["id"], REVISED,
        edits=[{"term": "评价", "replacement": "感想"}])
    browser = FakeBrowser()
    browser.open = Mock(side_effect=AssertionError("Validation must not open a browser automatically"))
    now, logs = [100.0], []
    runner = DanmakuValidationRunner(browser, content_filter, clock=lambda: now[0],
        log=lambda kind, **fields: logs.append({"kind": kind, **fields}))
    runner.set_enabled(True)
    yield runner, browser, content_filter, candidate["id"], now, logs
    runner.close()


def tick(runner, *, test_room=TEST_ROOM, formal_room=FORMAL_ROOM, interval=5.0, earliest_post=0.0):
    runner.tick(test_room, formal_room, interval, earliest_post=earliest_post)


def begin_send(runner, browser):
    tick(runner)
    assert browser.requests[-1]["operation"] == "login"
    assert browser.requests[-1]["room"] == TEST_ROOM
    browser.reply(is_login=True, code=0, error="none", max_weighted_length=80)
    tick(runner)
    assert browser.requests[-1]["operation"] == "send"
    assert browser.requests[-1]["room"] == TEST_ROOM
    assert browser.requests[-1]["text"] == "【" + REVISED + "】"


def reply_receipt(browser, *, baseline=0, outcome="receipt_accepted", **updates):
    browser.reply(outcome=outcome, code=0, http_status=200, error="none", before_count=baseline, **updates)


def begin_check(runner, browser, now):
    now[0] += 5.0
    tick(runner)
    assert browser.requests[-1]["operation"] == "check"


def reply_page(browser, *, count=1, **updates):
    payload = {"present": count > 0, "room_matches": True, "chat_ready": True,
               "error": "none", "occurrence_count": count}
    payload.update(updates)
    browser.reply(**payload)


def sends(browser):
    return [request for request in browser.requests if request["operation"] == "send"]


@pytest.mark.parametrize("test_room,formal_room", [("", FORMAL_ROOM), (FORMAL_ROOM, FORMAL_ROOM),
    ("0021399689", FORMAL_ROOM)])
def test_missing_or_same_test_room_never_dispatches(validation, test_room, formal_room):
    runner, browser, content_filter, candidate_id, now, _ = validation
    for _ in range(3):
        tick(runner, test_room=test_room, formal_room=formal_room)
        now[0] += 10.0
    assert browser.requests == []
    assert runner.state == "waiting_validation"
    assert content_filter.validation_plan(candidate_id)["state"] == "waiting_room"
    assert content_filter.rules() == []
    browser.open.assert_not_called()


def test_historical_foreign_names_only_learn_two_rules_after_chinese_delivery_validation(app, tmp_path):
    original = "【レレン 谢谢 クちゃん 谢谢留言】"
    revised = "蕾莲 谢谢 小库 谢谢留言"
    content_filter = DanmakuContentFilter(tmp_path / "historical.json")
    candidate = content_filter.record_failure(TEST_ROOM, original, "platform_f")
    analysis = validate_analysis_result({"revised_text": revised}, original)
    content_filter.save_validation_plan(candidate["id"], revised, edits=analysis["edits"])
    browser, now = FakeBrowser(), [100.0]
    runner = DanmakuValidationRunner(browser, content_filter, clock=lambda: now[0])
    runner.set_enabled(True)
    try:
        runner.tick(TEST_ROOM, "", 3.5)
        assert browser.requests[-1]["operation"] == "login"
        browser.reply(is_login=True, code=0, error="none", max_weighted_length=80)
        runner.tick(TEST_ROOM, "", 3.5)
        assert browser.requests[-1]["operation"] == "send"
        assert browser.requests[-1]["text"] == "【" + revised + "】"
        reply_receipt(browser, baseline=1)
        assert content_filter.rules() == []
        now[0] += 3.5
        runner.tick(TEST_ROOM, "", 3.5)
        assert browser.requests[-1]["operation"] == "check"
        reply_page(browser, count=2)
        rules = content_filter.rules()
        assert [(rule["term"], rule["replacement"]) for rule in rules] == [
            ("レレン", "蕾莲"), ("クちゃん", "小库")]
        assert all(rule["enabled"] and rule["validation"]["page_observed"] for rule in rules)
        assert all(rule["validation"]["confirmed_cause"] is False for rule in rules)
        assert content_filter.candidates()[0]["status"] == "resolved"
        restored = DanmakuContentFilter(content_filter.path)
        assert restored.rules() == rules
        assert len(sends(browser)) == 1
    finally:
        runner.close()


def test_validation_plan_with_foreign_remnant_is_not_posted(app, tmp_path):
    content_filter = DanmakuContentFilter(tmp_path / "foreign.json")
    candidate = content_filter.record_failure(FORMAL_ROOM, "レレン 谢谢你的评价", "platform_f")
    content_filter.save_validation_plan(candidate["id"], "レレン 谢谢你的感想",
        edits=[{"term": "评价", "replacement": "感想"}])
    browser = FakeBrowser()
    runner = DanmakuValidationRunner(browser, content_filter)
    runner.set_enabled(True)
    try:
        tick(runner)
        assert browser.requests == []
        assert content_filter.validation_plan(candidate["id"])["state"] == "failed"
        assert "外语" in content_filter.validation_plan(candidate["id"])["message"]
        assert content_filter.rules() == []
    finally:
        runner.close()


def test_legacy_foreign_validation_post_is_only_readonly_checked(app, tmp_path):
    content_filter = DanmakuContentFilter(tmp_path / "old-foreign.json")
    candidate = content_filter.record_failure(FORMAL_ROOM, "レレン 谢谢你的评价", "platform_f")
    content_filter.save_validation_plan(candidate["id"], "レレン 谢谢你的感想",
        edits=[{"term": "评价", "replacement": "感想"}])
    content_filter.configure_validation(candidate["id"], TEST_ROOM, formal_room=FORMAL_ROOM)
    content_filter.update_validation(candidate["id"], state="testing", request_id="e" * 32)
    browser = FakeBrowser()
    runner = DanmakuValidationRunner(browser, content_filter)
    runner.set_enabled(True)
    try:
        tick(runner)
        assert browser.requests[-1]["operation"] == "login"
        browser.reply(is_login=True, code=0, error="none", max_weighted_length=80)
        tick(runner)
        assert browser.requests[-1]["operation"] == "check"
        assert browser.requests[-1]["text"] == "【レレン 谢谢你的感想】"
        assert sends(browser) == []
        assert content_filter.rules() == []
    finally:
        runner.close()


def test_preparing_and_receiving_normal_acknowledgement_does_not_create_rules_or_archive(validation):
    runner, browser, content_filter, candidate_id, now, _ = validation
    assert content_filter.rules() == []
    begin_send(runner, browser)
    reply_receipt(browser)
    plan = content_filter.validation_plan(candidate_id)
    assert plan["state"] == "testing"
    assert plan["receipt"]["receipt_accepted"] is True
    assert content_filter.candidates()[0]["status"] == "waiting_validation"
    assert content_filter.rules() == []
    loaded = DanmakuContentFilter(content_filter.path)
    assert loaded.validation_plan(candidate_id)["receipt"] == plan["receipt"]
    begin_check(runner, browser, now)
    reply_page(browser, count=0)
    assert content_filter.rules() == []
    assert content_filter.validation_plan(candidate_id)["state"] == "testing"
    browser.open.assert_not_called()


@pytest.mark.parametrize("baseline", [0, 1, 3])
def test_fresh_page_count_and_persisted_receipt_enable_only_local_rule(validation, baseline):
    runner, browser, content_filter, candidate_id, now, logs = validation
    begin_send(runner, browser)
    request_id = browser.requests[-1]["request_id"]
    reply_receipt(browser, baseline=baseline)
    begin_check(runner, browser, now)
    reply_page(browser, count=baseline + 1)
    assert runner.state == "validation_passed"
    plan = content_filter.validation_plan(candidate_id)
    assert plan["state"] == "passed"
    assert plan["evidence"]["request_id"] == request_id
    assert plan["evidence"]["test_room"] == TEST_ROOM
    assert plan["evidence"]["formal_room"] == FORMAL_ROOM
    assert plan["evidence"]["before_count"] == baseline
    assert plan["evidence"]["occurrence_count"] == baseline + 1
    assert plan["evidence"]["confirmed_cause"] is False
    candidate = content_filter.candidates()[0]
    assert candidate["status"] == "resolved"
    assert candidate["resolution"] == "test_room_validated"
    rule, = content_filter.rules()
    assert rule["term"] == "评价"
    assert rule["replacement"] == "感想"
    assert rule["room"] == ""
    assert rule["enabled"] is True
    assert rule["origin"] == "ai_analysis"
    assert rule["validation"]["test_room"] == TEST_ROOM
    assert content_filter.apply(ORIGINAL, TEST_ROOM) == (REVISED, 1)
    assert content_filter.apply(ORIGINAL, FORMAL_ROOM) == (REVISED, 1)
    assert content_filter.apply(ORIGINAL, "21399689") == (REVISED, 1)
    assert any(row["kind"] == "validation_passed" for row in logs)
    assert len(sends(browser)) == 1
    browser.open.assert_not_called()


@pytest.mark.parametrize("page", [
    {"occurrence_count": 1}, {"occurrence_count": 0, "present": False},
    {"room_matches": False}, {"chat_ready": False}, {"present": False},
    {"error": "navigation_changed"}, {"occurrence_count": True}, {"occurrence_count": "2"}])
def test_old_or_inconsistent_page_observations_do_not_validate_or_write_rules(validation, page):
    runner, browser, content_filter, candidate_id, now, _ = validation
    begin_send(runner, browser)
    reply_receipt(browser, baseline=1)
    begin_check(runner, browser, now)
    reply_page(browser, count=2, **page)
    assert content_filter.rules() == []
    assert content_filter.validation_plan(candidate_id)["state"] == "testing"
    assert content_filter.candidates()[0]["status"] == "waiting_validation"
    assert len(sends(browser)) == 1


@pytest.mark.parametrize("outcome", ["result_unknown", "receipt_unconfirmed"])
def test_unknown_post_stays_readonly_even_after_matching_text_appears(validation, outcome):
    runner, browser, content_filter, candidate_id, now, _ = validation
    begin_send(runner, browser)
    reply_receipt(browser, outcome=outcome)
    for _ in range(4):
        begin_check(runner, browser, now)
        reply_page(browser, count=4)
    assert content_filter.validation_plan(candidate_id)["state"] == "testing"
    assert content_filter.validation_plan(candidate_id)["receipt"]["receipt_accepted"] is False
    assert content_filter.rules() == []
    assert len(sends(browser)) == 1
    browser.open.assert_not_called()


@pytest.mark.parametrize("outcome", ["platform_f", "platform_rejected"])
def test_f_and_explicit_rejection_fail_validation_without_creating_rules(validation, outcome):
    runner, browser, content_filter, candidate_id, now, _ = validation
    begin_send(runner, browser)
    reply_receipt(browser, outcome=outcome)
    for _ in range(4):
        now[0] += 10.0
        tick(runner)
    assert runner.state == "validation_failed"
    assert content_filter.validation_plan(candidate_id)["state"] == "failed"
    assert content_filter.rules() == []
    assert content_filter.candidates()[0]["status"] == "waiting_validation"
    assert len(sends(browser)) == 1


@pytest.mark.parametrize("accepted_receipt", [True, False])
def test_restart_of_testing_plan_only_checks_and_never_reposts(validation, accepted_receipt):
    runner, browser, content_filter, candidate_id, now, _ = validation
    begin_send(runner, browser)
    reply_receipt(browser, outcome="receipt_accepted" if accepted_receipt else "result_unknown")
    runner.close()
    reopened = DanmakuContentFilter(content_filter.path)
    replacement = DanmakuValidationRunner(browser, reopened, clock=lambda: now[0])
    replacement.set_enabled(True)
    try:
        tick(replacement)
        assert browser.requests[-1]["operation"] == "login"
        browser.reply(is_login=True, code=0, error="none")
        tick(replacement)
        assert browser.requests[-1]["operation"] == "check"
        reply_page(browser, count=1)
        assert len(sends(browser)) == 1
        assert bool(reopened.rules()) is accepted_receipt
        assert reopened.validation_plan(candidate_id)["state"] == ("passed" if accepted_receipt else "testing")
        browser.open.assert_not_called()
    finally:
        replacement.close()


def test_timeout_without_post_receipt_never_reposts_or_learns(validation):
    runner, browser, content_filter, candidate_id, now, _ = validation
    begin_send(runner, browser)
    now[0] += 46.0
    tick(runner)
    assert not runner.busy
    assert content_filter.validation_plan(candidate_id)["receipt"] is None
    now[0] += 5.0
    tick(runner)
    assert browser.requests[-1]["operation"] == "login"
    browser.reply(is_login=True, code=0, error="none")
    tick(runner)
    assert browser.requests[-1]["operation"] == "check"
    reply_page(browser, count=5)
    assert len(sends(browser)) == 1
    assert content_filter.rules() == []


def test_runner_uses_one_inflight_request_and_respects_shared_post_interval(validation):
    runner, browser, content_filter, candidate_id, now, _ = validation
    tick(runner)
    for _ in range(5):
        tick(runner)
    assert len(browser.requests) == 1
    browser.reply(is_login=True, code=0, error="none", max_weighted_length=80)
    tick(runner, earliest_post=105.0)
    assert len(sends(browser)) == 0
    now[0] = 104.999
    tick(runner, earliest_post=105.0)
    assert len(sends(browser)) == 0
    now[0] = 105.0
    tick(runner, earliest_post=105.0)
    assert len(sends(browser)) == 1
    for _ in range(5):
        tick(runner)
    assert len(sends(browser)) == 1
    reply_receipt(browser)
    now[0] = 109.999
    tick(runner)
    assert browser.requests[-1]["operation"] == "send"
    now[0] = 110.0
    tick(runner)
    assert browser.requests[-1]["operation"] == "check"


def test_foreign_request_ids_and_operations_never_complete_validation(validation):
    runner, browser, content_filter, candidate_id, now, _ = validation
    begin_send(runner, browser)
    request = browser.requests[-1]
    browser.danmaku_result.emit({"request_id": "foreign", "operation": "send",
        "result": {"outcome": "receipt_accepted", "code": 0, "http_status": 200}})
    browser.danmaku_result.emit({"request_id": request["request_id"], "operation": "check",
        "result": {"present": True, "room_matches": True, "chat_ready": True, "occurrence_count": 1}})
    assert runner.busy
    assert content_filter.validation_plan(candidate_id)["receipt"] is None
    assert content_filter.rules() == []


def test_real_wrapped_failed_sentence_accepts_plain_ai_output_and_sends_one_wrapper(app, tmp_path):
    content_filter = DanmakuContentFilter(tmp_path / "filter.json")
    candidate = content_filter.record_failure(FORMAL_ROOM, "【" + ORIGINAL + "】", "platform_f")
    content_filter.save_validation_plan(candidate["id"], REVISED,
        edits=[{"term": "评价", "replacement": "感想"}])
    plan = content_filter.validation_plan(candidate["id"])
    assert plan["original_text"] == "【" + ORIGINAL + "】"
    assert plan["revised_text"] == REVISED
    browser = FakeBrowser()
    runner = DanmakuValidationRunner(browser, content_filter, clock=lambda: 100.0)
    runner.set_enabled(True)
    try:
        begin_send(runner, browser)
        assert browser.requests[-1]["text"] == "【" + REVISED + "】"
        assert "【【" not in browser.requests[-1]["text"]
    finally:
        runner.close()


@pytest.mark.parametrize("baseline", [None, True, -1, "0"])
def test_missing_or_invalid_baseline_cannot_crash_or_learn_after_normal_receipt(validation, baseline):
    runner, browser, content_filter, candidate_id, now, _ = validation
    begin_send(runner, browser)
    request = browser.requests[-1]
    runner._on_result({"request_id": request["request_id"], "operation": "send", "result": {
        "outcome": "receipt_accepted", "code": 0, "http_status": 200, "error": "none", "before_count": baseline}})
    assert content_filter.rules() == []
    assert content_filter.validation_plan(candidate_id)["state"] == "testing"
    begin_check(runner, browser, now)
    reply_page(browser, count=5)
    assert content_filter.rules() == []
    assert len(sends(browser)) == 1


@pytest.mark.parametrize("storage_method", ["record_validation_receipt", "confirm_validation"])
def test_storage_error_keeps_testing_state_and_never_reposts(validation, storage_method):
    runner, browser, content_filter, candidate_id, now, logs = validation
    begin_send(runner, browser)
    payload = {"outcome": "receipt_accepted", "code": 0, "http_status": 200, "error": "none", "before_count": 0}
    if storage_method == "confirm_validation":
        reply_receipt(browser)
        begin_check(runner, browser, now)
        payload = {"present": True, "room_matches": True, "chat_ready": True, "error": "none", "occurrence_count": 1}
    request = browser.requests[-1]
    with patch.object(content_filter, storage_method, side_effect=OSError("PRIVATE_STORAGE_ERROR")):
        runner._on_result({"request_id": request["request_id"], "operation": request["operation"], "result": payload})
    assert content_filter.validation_plan(candidate_id)["state"] == "testing"
    assert content_filter.rules() == []
    assert runner.state == "validation_storage_error"
    assert runner.busy
    assert "PRIVATE_STORAGE_ERROR" not in str(logs)
    previous_requests = len(browser.requests)
    now[0] += 4.999
    tick(runner)
    assert len(browser.requests) == previous_requests
    now[0] += 0.001
    tick(runner)
    assert not runner.busy
    assert len(browser.requests) == previous_requests
    if storage_method == "record_validation_receipt":
        assert content_filter.validation_plan(candidate_id)["receipt"]["receipt_accepted"] is True
        begin_check(runner, browser, now)
        reply_page(browser)
    assert content_filter.validation_plan(candidate_id)["state"] == "passed"
    assert len(content_filter.rules()) == 1
    assert len(sends(browser)) == 1
    browser.open.assert_not_called()


def test_failed_validation_does_not_starve_the_next_independent_test(validation):
    runner, browser, content_filter, candidate_id, now, _ = validation
    next_original = ORIGINAL + "谢谢"
    next_revised = REVISED + "谢谢"
    next_candidate = content_filter.record_failure(FORMAL_ROOM, next_original, "platform_f")
    content_filter.save_validation_plan(next_candidate["id"], next_revised,
        edits=[{"term": "评价", "replacement": "感想"}])
    begin_send(runner, browser)
    reply_receipt(browser, outcome="platform_f")
    now[0] += 5.0
    tick(runner)
    assert browser.requests[-1]["operation"] == "send"
    assert browser.requests[-1]["text"] == "【" + next_revised + "】"
    assert content_filter.validation_plan(candidate_id)["state"] == "failed"
    assert content_filter.rules() == []


@pytest.mark.parametrize("rooms", [("1698226", FORMAL_ROOM), (TEST_ROOM, "21399690")])
def test_room_change_ignores_inflight_reply_and_keeps_uncertain_test_without_reposting(validation, rooms):
    runner, browser, content_filter, candidate_id, now, _ = validation
    begin_send(runner, browser)
    runner.set_rooms(*rooms)
    reply_receipt(browser)
    plan = content_filter.validation_plan(candidate_id)
    assert plan["state"] == "testing"
    assert plan["receipt"] is None
    assert plan["test_room"] == TEST_ROOM
    assert plan["formal_room"] == FORMAL_ROOM
    for _ in range(2):
        now[0] += 10.0
        tick(runner, test_room=rooms[0], formal_room=rooms[1])
    assert content_filter.rules() == []
    assert len(sends(browser)) == 1
    browser.open.assert_not_called()


def test_global_next_send_at_keeps_post_interval_after_receipt(validation):
    runner, browser, content_filter, candidate_id, now, _ = validation
    begin_send(runner, browser)
    reply_receipt(browser)
    assert runner.next_send_at == now[0] + 5.0
    now[0] += 5.0
    tick(runner)
    reply_page(browser)
    assert runner.next_send_at == now[0] + 5.0
    assert content_filter.validation_plan(candidate_id)["state"] == "passed"
    assert len(sends(browser)) == 1


def test_production_service_waits_for_validation_post_interval_on_shared_browser(validation, tmp_path):
    runner, browser, content_filter, candidate_id, now, _ = validation
    service = DanmakuService(browser, store_path=tmp_path / "queue.sqlite3", filter_path=content_filter.path)
    service._clock = lambda: now[0]
    service._validator = runner
    service.configure(True, FORMAL_ROOM, interval_seconds=5.0, validation_room=TEST_ROOM, repair_enabled=True)
    service.timer.stop()
    service.begin_session(1)
    service._apply_limits({"max_weighted_length": 80})
    service._login_room = FORMAL_ROOM
    enqueue(service, "正式字幕应等待测试请求间隔")
    try:
        begin_send(runner, browser)
        service.tick()
        assert len(sends(browser)) == 1
        reply_receipt(browser)
        service.tick()
        assert service.snapshot()["state"] == "validating"
        assert len(sends(browser)) == 1
        now[0] += 4.999
        service.tick()
        assert len(sends(browser)) == 1
        now[0] += 0.001
        service.tick()
        assert len(sends(browser)) == 2
        assert browser.requests[-1]["room"] == FORMAL_ROOM
        assert browser.requests[-1]["text"] == "【正式字幕应等待测试请求间隔】"
        browser.open.assert_not_called()
    finally:
        service.close()


def add_next_plan(content_filter):
    next_original, next_revised = ORIGINAL + "谢谢", REVISED + "谢谢"
    candidate = content_filter.record_failure(FORMAL_ROOM, next_original, "platform_f")
    content_filter.save_validation_plan(candidate["id"], next_revised,
        edits=[{"term": "评价", "replacement": "感想"}])
    return candidate["id"], next_revised


@pytest.mark.parametrize("outcome", ["result_unknown", "receipt_unconfirmed"])
def test_uncertain_test_keeps_readonly_verification_without_starving_another_plan(validation, outcome):
    runner, browser, content_filter, candidate_id, now, _ = validation
    next_id, next_revised = add_next_plan(content_filter)
    begin_send(runner, browser)
    reply_receipt(browser, outcome=outcome)
    now[0] += 5.0
    tick(runner)
    assert browser.requests[-1]["operation"] == "send"
    assert browser.requests[-1]["text"] == "【" + next_revised + "】"
    reply_receipt(browser)
    begin_check(runner, browser, now)
    assert browser.requests[-1]["text"] == "【" + REVISED + "】"
    reply_page(browser, count=0)
    begin_check(runner, browser, now)
    assert browser.requests[-1]["text"] == "【" + next_revised + "】"
    reply_page(browser)
    assert content_filter.validation_plan(next_id)["state"] == "passed"
    assert content_filter.validation_plan(candidate_id)["state"] == "testing"
    assert content_filter.candidates()[0]["status"] == "waiting_validation"
    assert len(sends(browser)) == 2
    assert [request["text"] for request in sends(browser)].count("【" + REVISED + "】") == 1
    browser.open.assert_not_called()


@pytest.mark.parametrize("rooms", [("1698226", FORMAL_ROOM), (TEST_ROOM, "21399690")])
def test_test_for_old_room_configuration_does_not_starve_a_plan_for_current_rooms(validation, rooms):
    runner, browser, content_filter, candidate_id, now, _ = validation
    begin_send(runner, browser)
    reply_receipt(browser, outcome="result_unknown")
    old_plan = content_filter.validation_plan(candidate_id)
    next_id, next_revised = add_next_plan(content_filter)
    now[0] += 5.0
    tick(runner, test_room=rooms[0], formal_room=rooms[1])
    assert browser.requests[-1]["operation"] == "login"
    assert browser.requests[-1]["room"] == rooms[0]
    browser.reply(is_login=True, code=0, error="none", max_weighted_length=80)
    tick(runner, test_room=rooms[0], formal_room=rooms[1])
    assert browser.requests[-1]["operation"] == "send"
    assert browser.requests[-1]["text"] == "【" + next_revised + "】"
    reply_receipt(browser)
    now[0] += 5.0
    previous_requests = len(browser.requests)
    tick(runner, test_room=rooms[0], formal_room=rooms[1])
    assert runner.state == "validation_configuration_changed"
    assert len(browser.requests) == previous_requests
    tick(runner, test_room=rooms[0], formal_room=rooms[1])
    assert browser.requests[-1]["operation"] == "check"
    assert browser.requests[-1]["room"] == rooms[0]
    reply_page(browser)
    assert content_filter.validation_plan(next_id)["state"] == "passed"
    assert content_filter.validation_plan(candidate_id) == old_plan
    assert len(sends(browser)) == 2
    browser.open.assert_not_called()


@pytest.mark.parametrize("punctuation", ["!", "。", "？"])
def test_punctuation_in_ai_text_fails_locally_without_dispatching_a_test(validation, punctuation):
    runner, browser, content_filter, candidate_id, now, _ = validation
    content_filter.save_validation_plan(candidate_id, REVISED.replace("感想", "感想" + punctuation),
        edits=[{"term": "评价", "replacement": "感想" + punctuation}])
    for _ in range(3):
        tick(runner)
        now[0] += 5.0
    assert content_filter.validation_plan(candidate_id)["state"] == "failed"
    assert content_filter.rules() == []
    assert browser.requests == []
    assert runner.state == "validation_failed"
    browser.open.assert_not_called()
