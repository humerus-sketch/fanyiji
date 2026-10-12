"""A short FIFO head permits authentication probes, never undersized POSTs."""

import pytest

from fanyiji.danmaku_service import DanmakuService
from fanyiji.danmaku_protocol import danmaku_utf16_length, danmaku_weighted_length
from test_danmaku_service import FakeBrowser, app, enqueue


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


def test_short_first_utterance_probes_login_then_merges_with_next_final(unknown_limit_service):
    service, browser, now = unknown_limit_service
    enqueue(service, "谢谢大家", "short")
    first = service.store.head()
    assert danmaku_utf16_length(first["text"]) <= 10
    service.tick()
    assert [row["operation"] for row in browser.requests] == ["login"]
    assert browser.requests[0]["text"] == ""
    browser.reply(is_login=True, code=0, error="none", max_weighted_length=80)
    for _ in range(3):
        now[0] += 1
        service.tick()
    assert not any(row["operation"] == "send" for row in browser.requests)
    enqueue(service, "继续欢迎大家来到直播间", "next")
    head = service.store.head()
    assert head["id"] == first["id"]
    assert head["text"].startswith("【谢谢大家 ")
    service.tick()
    assert browser.requests[-1]["operation"] == "send"
    assert 11 <= danmaku_utf16_length(browser.requests[-1]["text"]) <= 40
    assert danmaku_weighted_length(browser.requests[-1]["text"]) <= 80


def test_known_low_account_cap_does_not_release_or_overtake_short_head(unknown_limit_service):
    service, browser, now = unknown_limit_service
    enqueue(service, "谢谢大家", "short")
    first = service.store.head()
    service.tick()
    browser.reply(is_login=True, code=0, error="none", max_weighted_length=20)
    enqueue(service, "后续字幕足够长也不能抢在前句发送", "next")
    for _ in range(5):
        now[0] += 10
        service.tick()
    assert service.store.head()["id"] == first["id"]
    assert service.store.head()["state"] == "holding_short"
    assert service.snapshot()["queued"] == 2
    assert [row["operation"] for row in browser.requests] == ["login"]


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
    assert service.store.head()["state"] == "holding_short"


@pytest.mark.parametrize("real_cap", [20, 80])
def test_short_and_following_final_before_login_respect_the_real_cap(unknown_limit_service, real_cap):
    service, browser, now = unknown_limit_service
    enqueue(service, "谢谢大家", "short")
    first_id = service.store.head()["id"]
    enqueue(service, "后面这句足够长会先与短句合并", "next")
    head = service.store.head()
    assert head["id"] == first_id
    assert head["text"] == "【谢谢大家 后面这句足够长会先与短句合并】"
    assert head["state"] == "pending"
    assert not browser.requests
    service.tick()
    assert [row["operation"] for row in browser.requests] == ["login"]
    browser.reply(is_login=True, code=0, error="none", max_weighted_length=real_cap)
    service.tick()
    assert service.store.head()["id"] == first_id
    if real_cap == 80:
        assert browser.requests[-1]["operation"] == "send"
        assert 11 <= danmaku_utf16_length(browser.requests[-1]["text"]) <= 40
        assert danmaku_weighted_length(browser.requests[-1]["text"]) <= real_cap
    else:
        assert service.store.head()["state"] == "holding_short"
        assert not any(row["operation"] == "send" for row in browser.requests)
        for _ in range(3):
            now[0] += 10
            service.tick()
        assert service.store.head()["id"] == first_id
        assert not any(row["operation"] == "send" for row in browser.requests)
