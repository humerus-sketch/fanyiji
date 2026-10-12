"""Storage failures must not be presented as automatic platform retries."""
from fanyiji.danmaku_service import DanmakuService
from test_danmaku_service import app, FakeBrowser, enqueue


def test_unsaved_failed_candidate_reports_storage_retry_without_resending(app, tmp_path, monkeypatch):
    browser = FakeBrowser()
    service = DanmakuService(browser, store_path=tmp_path / "queue.sqlite3")
    try:
        service.configure(True, "1698225")
        service.timer.stop()
        service.begin_session(1)
        enqueue(service, "这是一条保存失败的完整字幕")
        head = service.store.head()
        service.store.update(head["id"], "verify", reason="platform_f", platform_code=0)
        service.store.activate_f_recovery(head["id"], new_f=True)

        def unavailable(*args, **kwargs):
            raise OSError("storage_unavailable")

        monkeypatch.setattr(service.content_filter, "record_failure", unavailable)
        service._record_failed_content(head, "platform_f")
        snapshot = service.snapshot()
        assert snapshot["state"] == "saving_failed_record"
        assert snapshot["recovery_action"] == "retry_failure_storage"
        assert snapshot["automatic_recovery"] is True
        assert snapshot["f_retry_active"] is False
        assert snapshot["filter_candidate_error"]
        assert service.store.head()["id"] == head["id"]
        service.tick()
        assert not browser.requests
        assert service.store.counts()["queued"] == 1
        assert service.store.counts()["deferred_analysis"] == 0
    finally:
        service.close()
