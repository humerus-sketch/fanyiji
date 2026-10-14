"""Only a fake QProcess and temporary browser data are used."""
import pytest
from unittest.mock import patch
from PySide6.QtCore import QProcess
from test_browser_service import app, service, fake_process, feed_event, written_commands


ROOM, TEXT, OTHER = "1698225", "【首次未知字幕】", "【后续正常字幕】"


def unknown(service, process, error="network_unknown"):
    process.state.return_value = QProcess.ProcessState.Running
    service._ready = True
    service._current_url = "https://live.bilibili.com/" + ROOM
    process.write.side_effect = lambda data: len(data)
    assert service.request_danmaku("send", "unknown", ROOM, TEXT)
    feed_event(service, process, {"kind": "danmaku_result", "request_id": "unknown", "operation": "send",
        "result": {"outcome": "result_unknown", "error": error, "before_count": 0}})


def test_settled_unknown_handoff_fences_same_text_and_allows_following_payload(service, fake_process):
    results = []
    service.danmaku_result.connect(results.append)
    unknown(service, fake_process)
    assert service.retire_danmaku_submission(ROOM, TEXT)
    assert not service._stopping
    before = len(written_commands(fake_process))
    assert service.request_danmaku("send", "same", ROOM, TEXT)
    assert len(written_commands(fake_process)) == before
    assert results[-1]["result"]["outcome"] == "not_sent"
    assert service.request_danmaku("send", "different", ROOM, OTHER)
    assert written_commands(fake_process)[-1]["text"] == OTHER


def test_unknown_payload_is_already_fenced_before_failed_content_can_be_saved(service, fake_process):
    unknown(service, fake_process)
    before = len(written_commands(fake_process))
    assert service.request_danmaku("send", "same-before-save", ROOM, TEXT)
    assert len(written_commands(fake_process)) == before
    assert not service.request_danmaku("send", "next-before-save", ROOM, OTHER)
    assert len(written_commands(fake_process)) == before


@pytest.mark.parametrize("error", ["script_timeout", "script_unavailable", "navigation_changed"])
def test_unsettled_handoff_waits_for_old_helper_exit_before_reopening_same_profile(service, fake_process, error):
    unknown(service, fake_process, error)
    profile = service.profile
    assert service.retire_danmaku_submission(ROOM, TEXT)
    assert service._stopping and not service.is_browser_ready
    assert written_commands(fake_process)[-1] == {"op": "close"}
    fake_process.start.assert_not_called()
    assert not service.request_danmaku("send", "waiting", ROOM, OTHER)
    assert service.profile == profile
    fake_process.state.return_value = QProcess.ProcessState.NotRunning
    service._finished(0, QProcess.ExitStatus.NormalExit)
    fake_process.start.assert_called_once()
    assert service.profile == profile
    assert service._pending_url == "https://live.bilibili.com/" + ROOM
    assert (ROOM, TEXT) in service._retired_submissions


def test_pending_transport_timeout_can_retire_without_a_helper_receipt(service, fake_process):
    fake_process.state.return_value = QProcess.ProcessState.Running
    fake_process.write.side_effect = lambda data: len(data)
    service._ready = True
    service._current_url = "https://live.bilibili.com/" + ROOM
    assert service.request_danmaku("send", "pending", ROOM, TEXT)
    assert service.retire_danmaku_submission(ROOM, TEXT)
    assert service._stopping
    assert (ROOM, TEXT) in service._retired_submissions


def test_retirement_does_not_open_a_previously_closed_browser(service, fake_process):
    assert service.retire_danmaku_submission(ROOM, TEXT) is False
    unknown(service, fake_process, "script_timeout")
    fake_process.state.return_value = QProcess.ProcessState.NotRunning
    assert service.retire_danmaku_submission(ROOM, TEXT)
    fake_process.start.assert_not_called()
    assert not service._stopping


def test_matching_new_presence_releases_only_its_exact_retired_text(service, fake_process):
    unknown(service, fake_process)
    service.retire_danmaku_submission(ROOM, TEXT)
    for identifier, text, count in (("other", OTHER, 1), ("old", TEXT, 0), ("new", TEXT, 1)):
        assert service.request_danmaku("check", identifier, ROOM, text)
        feed_event(service, fake_process, {"kind": "danmaku_result", "request_id": identifier, "operation": "check",
            "result": {"present": count > 0, "occurrence_count": count, "room_matches": True,
                       "chat_ready": True, "error": "none"}})
        if identifier != "new":
            assert (ROOM, TEXT) in service._retired_submissions
    assert (ROOM, TEXT) not in service._retired_submissions
    assert (ROOM, TEXT) not in service._unknown_submissions
