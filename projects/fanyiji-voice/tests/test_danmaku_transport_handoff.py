"""Only retire transport state after both durable failure stores acknowledge it."""
import pytest
from unittest.mock import Mock
from test_danmaku_service import app, setup_service, enqueue, login


def fail_unknown(browser):
    browser.reply(outcome="result_unknown", error="network_unknown", code=None)


def prepare(setup_service):
    sender, browser, now = setup_service
    browser.retire_danmaku_submission = Mock()
    enqueue(sender, text="首次未知字幕", segment="first")
    enqueue(sender, text="后续正常字幕", segment="next")
    login(sender, browser, now)
    return sender, browser, now


def test_retirement_occurs_after_failure_candidate_and_fifo_handoff(setup_service):
    sender, browser, _ = prepare(setup_service)
    seen = []
    def retire(room, text):
        seen.append((room, text, sender.store.head()["text"],
                     any(row["text"] == text for row in sender.content_filter.candidates())))
    browser.retire_danmaku_submission.side_effect = retire
    fail_unknown(browser)
    assert seen == [("1698225", "【首次未知字幕】", "【后续正常字幕】", True)]


@pytest.mark.parametrize("failure", ["filter", "queue"])
def test_write_failure_never_retires_the_original_submission(setup_service, monkeypatch, failure):
    sender, browser, _ = prepare(setup_service)
    def fail(*args, **kwargs):
        raise OSError("synthetic private details never logged")
    monkeypatch.setattr(sender.content_filter if failure == "filter" else sender.store,
                        "record_failure" if failure == "filter" else "defer_failed_head", fail)
    fail_unknown(browser)
    browser.retire_danmaku_submission.assert_not_called()
    assert sender.store.head()["text"] == "【首次未知字幕】"


def test_retirement_exception_does_not_undo_the_durable_handoff_or_leak_details(setup_service, caplog):
    sender, browser, now = prepare(setup_service)
    browser.retire_danmaku_submission.side_effect = OSError("synthetic secret retirement detail")
    fail_unknown(browser)
    assert sender.store.head()["text"] == "【后续正常字幕】"
    assert len(sender.content_filter.candidates()) == 1
    assert "synthetic secret retirement detail" not in caplog.text
    now[0] += 4
    sender.tick()
    assert browser.requests[-1]["text"] == "【后续正常字幕】"
