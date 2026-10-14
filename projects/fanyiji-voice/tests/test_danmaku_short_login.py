"""Two-character captions send independently; single characters wait in FIFO."""

import pytest

from fanyiji.danmaku_service import DanmakuService
from fanyiji.danmaku_protocol import danmaku_utf16_length, danmaku_weighted_length, danmaku_body_length
from test_danmaku_service import FakeBrowser, app, enqueue, finish


@pytest.fixture
def unknown_limit_service(app, tmp_path):
    browser = FakeBrowser()
    service = DanmakuService(browser, store_path=tmp_path / "queue.sqlite3")
    now = [100.0]
    service._clock = lambda: now[0]
    service.configure(True, "1698225")
    service.timer.stop()
    service.begin_session(1)
    yield service, browser, now
    service.close()


def test_two_character_first_utterance_probes_login_then_sends_without_next_sentence(unknown_limit_service):
    service, browser, now = unknown_limit_service
    enqueue(service, "谢谢", "short")
    first = service.store.head()
    assert danmaku_utf16_length(first["text"]) <= 10
    service.tick()
    assert [row["operation"] for row in browser.requests] == ["login"]
    assert browser.requests[0]["text"] == ""
    browser.reply(is_login=True, code=0, error="none", max_weighted_length=80)
    service.tick()
    assert browser.requests[-1]["operation"] == "send"
    assert browser.requests[-1]["text"] == "【谢谢】"
    enqueue(service, "继续欢迎大家来到直播间", "next")
    head = service.store.head()
    assert head["id"] == first["id"]
    assert head["text"] == "【谢谢】"
    assert service.snapshot()["queued"] == 2
    assert danmaku_body_length(browser.requests[-1]["text"]) == 2
    assert danmaku_utf16_length(browser.requests[-1]["text"]) <= 40
    assert danmaku_weighted_length(browser.requests[-1]["text"]) <= 40
    finish(service, browser, now)
    service.tick()
    assert browser.requests[-1]["text"] == "【继续欢迎大家来到直播间】"


def test_known_low_account_cap_sends_short_head_before_following_caption(unknown_limit_service):
    service, browser, now = unknown_limit_service
    enqueue(service, "谢谢大家", "short")
    first = service.store.head()
    service.tick()
    browser.reply(is_login=True, code=0, error="none", max_weighted_length=20)
    enqueue(service, "后续字幕先等前句确认", "next")
    service.tick()
    assert service.store.head()["id"] == first["id"]
    assert service.store.head()["state"] == "sending"
    assert service.snapshot()["queued"] == 3
    assert [row["operation"] for row in browser.requests] == ["login", "send"]
    assert browser.requests[-1]["text"] == "【谢谢大家】"
    assert danmaku_weighted_length(browser.requests[-1]["text"]) <= 20


def test_logged_out_short_head_rechecks_only_after_read_interval(unknown_limit_service):
    service, browser, now = unknown_limit_service
    enqueue(service, "谢谢大家", "short")
    service.tick()
    browser.reply(is_login=False, code=0, error="none")
    now[0] += 3.499
    service.tick()
    assert len(browser.requests) == 1
    now[0] += 0.001
    service.tick()
    assert [row["operation"] for row in browser.requests] == ["login", "login"]
    assert not any(row["operation"] == "send" for row in browser.requests)


def test_short_first_utterance_waits_if_browser_is_not_ready(unknown_limit_service):
    service, browser, _ = unknown_limit_service
    enqueue(service, "谢谢大家", "short")
    browser.is_browser_ready = False
    service.tick()
    assert not browser.requests
    assert service.snapshot()["state"] == "waiting_browser"
    assert service.store.head()["state"] == "pending"


@pytest.mark.parametrize("real_cap", [20, 80])
def test_short_and_following_final_before_login_respect_the_real_cap(unknown_limit_service, real_cap):
    service, browser, now = unknown_limit_service
    enqueue(service, "谢谢大家", "short")
    first_id = service.store.head()["id"]
    enqueue(service, "这句独立排队等待发送", "next")
    head = service.store.head()
    assert head["id"] == first_id
    assert head["text"] == "【谢谢大家】"
    assert head["state"] == "pending"
    assert service.snapshot()["queued"] == 2
    assert not browser.requests
    service.tick()
    assert [row["operation"] for row in browser.requests] == ["login"]
    browser.reply(is_login=True, code=0, error="none", max_weighted_length=real_cap)
    service.tick()
    assert service.store.head()["id"] == first_id
    assert browser.requests[-1]["operation"] == "send"
    assert browser.requests[-1]["text"] == "【谢谢大家】"
    assert danmaku_body_length(browser.requests[-1]["text"]) >= 2
    assert danmaku_utf16_length(browser.requests[-1]["text"]) <= 40
    assert danmaku_weighted_length(browser.requests[-1]["text"]) <= min(real_cap, 40)


def test_single_character_first_utterance_probes_login_and_preserves_until_adjacent_next(unknown_limit_service):
    service, browser, now = unknown_limit_service
    enqueue(service, "啊", "single")
    identifier = service.store.head()["id"]
    service.tick()
    assert [row["operation"] for row in browser.requests] == ["login"]
    browser.reply(is_login=True, code=0, error="none", max_weighted_length=40)
    now[0] += 10
    service.tick()
    assert service.store.head()["id"] == identifier
    assert service.store.head()["text"] == "【啊】"
    assert service.snapshot()["state"] == "waiting_merge"
    assert not any(row["operation"] == "send" for row in browser.requests)
    enqueue(service, "这是接着说的话", "next")
    assert service.store.head()["id"] == identifier
    assert service.store.head()["text"] == "【啊 这是接着说的话】"
    service.tick()
    assert browser.requests[-1]["operation"] == "send"
    assert browser.requests[-1]["text"] == "【啊 这是接着说的话】"
