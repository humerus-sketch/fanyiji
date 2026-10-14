"""Selected-source lifecycle tests do not access devices or user data."""
import json
import os
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from fanyiji.audio import AudioChunk, CaptureStatus
from fanyiji.live_audio_protocol import LiveAudioFrame, LiveAudioStatus
from fanyiji.live_source_service import LiveSourceService


class OrdinaryCapture:
    instances = []

    def __init__(self, on_chunk, on_status):
        self.on_chunk, self.on_status = on_chunk, on_status
        self.channels = None
        self.stopped = False
        self.instances.append(self)

    def start(self, channels):
        self.channels = channels

    def stop(self):
        self.stopped = True


class NativeCapture:
    instances = []

    def __init__(self):
        self.started = None
        self.stopped = False
        self.on_stop = None
        self.instances.append(self)

    def start(self, pid, room, generation, on_frame, on_status):
        self.started = (pid, room, generation)
        self.on_frame, self.on_status = on_frame, on_status
        on_status(LiveAudioStatus(pid, room, generation, "running"))

    def stop(self):
        self.stopped = True
        if self.on_stop:
            self.on_stop()

    def frame(self, sequence=0, *, discontinuity=False):
        _, room, generation = self.started
        value = LiveAudioFrame(room, generation, sequence, np.full(1600, .1, np.float32),
                               10 + sequence / 10, 10.1 + sequence / 10, 20 + sequence / 10,
                               sequence * 1600, (sequence + 1) * 1600, discontinuity=discontinuity)
        self.on_frame(value)
        return value


@pytest.fixture
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app
    app.processEvents()


@pytest.fixture
def make_service(tmp_path, qt_app):
    instances = []
    OrdinaryCapture.instances, NativeCapture.instances = [], []
    with patch("fanyiji.live_source_service.AudioCaptureManager", OrdinaryCapture):
        def make(**kwargs):
            service = LiveSourceService(data_dir=tmp_path, capture_factory=NativeCapture, **kwargs)
            instances.append(service)
            return service
        yield make
        for service in instances:
            service.close()


def source(*, room="1698225", generation=1, browser_generation=1, state="ready", process_id=1234,
           source_kind="live", source_url=None):
    value = dict(room=room, generation=generation, browser_generation=browser_generation,
                 state=state, process_id=process_id, source_kind=source_kind)
    if source_url is not None:
        value["source_url"] = source_url
    return value


def connect(service, source_value=None, channels=None):
    service.bind_source(source_value or source())
    service.set_source_enabled(True)
    chunks, statuses = [], []
    capture = service.create_capture(chunks.append, statuses.append)
    capture.start({"system", "mic"} if channels is None else channels)
    return capture, NativeCapture.instances[-1], chunks, statuses


def job(service, **changes):
    value = dict(channel="system", generation=service.snapshot_value()["generation"],
                 source_room=service.snapshot_value()["room"])
    value.update(changes)
    return SimpleNamespace(**value)


def test_constructor_is_read_only_and_never_reads_voice_profiles(make_service, tmp_path):
    (tmp_path / "speaker-profiles.json").write_text("not-json", encoding="utf-8")
    service = make_service()
    assert service.snapshot_value()["state"] == "disabled"
    assert not service.settings_path.exists()
    assert OrdinaryCapture.instances == NativeCapture.instances == []
    assert not hasattr(service, "engine")


def test_only_legacy_source_switch_is_migrated_and_old_files_are_retained(make_service, tmp_path):
    old = tmp_path / "speaker-settings.json"
    content = '{"source_enabled": true, "lock_enabled": true, "enrollment_room": "123"}'
    old.write_text(content, encoding="utf-8")
    service = make_service()
    assert service.snapshot_value()["source_enabled"] is True
    service.set_source_enabled(False)
    assert json.loads(service.settings_path.read_text()) == {"source_enabled": False}
    assert old.read_text() == content


@pytest.mark.parametrize("content", ["{", "[]", '{"source_enabled": 1}', '{"source_enabled":false,"bad":true}'])
def test_corrupt_new_settings_never_fall_back_to_system_device(make_service, tmp_path, content):
    (tmp_path / "live-source-settings.json").write_text(content, encoding="utf-8")
    service = make_service()
    service.bind_source(source())
    capture = service.create_capture(lambda _: None)
    capture.start({"system", "mic"})
    assert OrdinaryCapture.instances[-1].channels == {"mic"}
    assert not NativeCapture.instances
    assert service.snapshot_value()["state"] == "error"
    assert not service.validate_job(job(service))
    service.set_source_enabled(True)
    assert service.snapshot_value()["state"] != "error"


def test_setting_save_failure_keeps_switch_and_current_capture(make_service):
    service = make_service()
    with patch("fanyiji.live_source_service.os.replace", side_effect=OSError("secret")):
        with pytest.raises(OSError, match="已有设置保留"):
            service.set_source_enabled(True)
    assert service.snapshot_value()["source_enabled"] is False
    assert not list(service.data_dir.glob("*.tmp"))


def test_mic_and_selected_system_capture_do_not_capture_other_apps(make_service):
    service = make_service()
    _, native, chunks, statuses = connect(service)
    assert OrdinaryCapture.instances[-1].channels == {"mic"}
    native.frame()
    assert chunks[-1].source_room == "1698225"
    assert service.validate_job(job(service))
    assert service.validate_job(job(service, channel="mic", generation=-1))
    assert statuses[-1].message == "指定直播音源已连接"


def test_reference_video_is_now_a_subtitle_source_with_no_live_room(make_service):
    service = make_service()
    value = source(room="", source_kind="video", source_url="https://www.bilibili.com/video/BV1xx411c7mD")
    _, native, chunks, statuses = connect(service, value)
    assert native.started[1] == ""
    native.frame()
    assert chunks[-1].source_room == ""
    assert service.validate_job(job(service))
    assert not service.validate_job(job(service, source_room="1698225"))
    assert statuses[-1].message == "指定视频音源已连接"
    assert not hasattr(service, "begin_enrollment")


def test_navigation_drops_old_frames_queued_jobs_and_translation_jobs(make_service):
    service = make_service()
    _, native, chunks, _ = connect(service)
    old_job = job(service)
    old_translation = job(service, source_generation=old_job.generation, generation=999)
    assert service.validate_job(old_translation)
    native.on_stop = lambda: native.frame(sequence=1)
    service.bind_source(source(room="21399689", generation=2))
    native.frame(sequence=2)
    assert chunks == []
    assert native.stopped
    assert not service.validate_job(old_job)
    assert not service.validate_job(old_translation)


def test_browser_close_zero_generation_snapshot_invalidates_selected_source(make_service):
    service = make_service()
    _, native, chunks, _ = connect(service)
    old_job = job(service)
    service.bind_source({"state": "unavailable", "room": "", "process_id": 0,
                         "generation": 0, "browser_generation": 1})
    assert native.stopped
    native.frame()
    assert chunks == []
    assert not service.validate_job(old_job)
    assert service.snapshot_value()["state"] == "waiting_browser"


def test_source_generation_staleness_includes_same_room_different_url_and_process(make_service):
    service = make_service()
    connect(service)
    old = job(service)
    service.bind_source(source(generation=2, process_id=2234))
    assert not service.validate_job(old)
    current = service.snapshot_value()
    service.bind_source(source(room="21399689", generation=0, browser_generation=0))
    assert service.snapshot_value() == current


def test_video_url_change_invalidates_old_audio_even_though_room_stays_empty(make_service):
    service = make_service()
    video = source(room="", source_kind="video", source_url="https://www.bilibili.com/video/BV1xx411c7mD")
    connect(service, video)
    old = job(service)
    service.bind_source({**video, "generation": 2, "source_url": "https://www.bilibili.com/video/BV2xx411c7mD"})
    assert not service.validate_job(old)


def test_pcm_discontinuity_advances_job_epoch_but_not_native_target(make_service):
    service = make_service()
    _, native, chunks, _ = connect(service)
    old = job(service)
    native.frame(discontinuity=True)
    assert not service.validate_job(old)
    assert service.validate_job(job(service))
    native.frame(sequence=1)
    assert len(chunks) == 2
    assert chunks[0].generation == chunks[1].generation
    assert chunks[0].start_sample == 0


def test_failed_native_capture_stays_failed_until_explicit_refresh(make_service):
    service = make_service()
    capture, native, chunks, statuses = connect(service)
    native.on_status(LiveAudioStatus(*native.started, "error", "token=private"))
    native.frame()
    assert not chunks
    assert service.snapshot_value()["state"] == "error"
    assert "private" not in service.snapshot_value()["message"]
    capture.start({"system"})
    assert len(NativeCapture.instances) == 1
    service.bind_source(source(generation=2))
    capture.start({"system"})
    assert len(NativeCapture.instances) == 2


def test_source_off_uses_existing_device_capture_and_mic_only_never_opens_native(make_service):
    service = make_service()
    chunks = []
    capture = service.create_capture(chunks.append)
    capture.start({"system", "mic"})
    assert OrdinaryCapture.instances[-1].channels == {"system", "mic"}
    service.set_source_enabled(True)
    capture.start({"mic"})
    assert OrdinaryCapture.instances[-1].channels == {"mic"}
    assert not NativeCapture.instances


def test_source_toggle_fences_old_video_jobs_and_maps_ordinary_device_epochs(make_service):
    service = make_service()
    video = source(room="", source_kind="video", source_url="https://www.bilibili.com/video/BV1xx411c7mD")
    _, native, _, _ = connect(service, video)
    old = job(service)
    service.set_source_enabled(False)
    chunks = []
    capture = service.create_capture(chunks.append)
    capture.start({"system"})
    assert not service.validate_job(old)
    ordinary = OrdinaryCapture.instances[-1]
    ordinary.on_chunk(AudioChunk("system", np.full(1600, .1, np.float32), 0, 1600, 20, 1))
    first = chunks[-1]
    assert service.validate_job(SimpleNamespace(channel="system", source_generation=first.generation, source_room=""))
    ordinary.on_chunk(AudioChunk("system", np.full(1600, .1, np.float32), 0, 1600, 21, 2))
    assert chunks[-1].generation != first.generation
    assert not service.validate_job(SimpleNamespace(channel="system", source_generation=first.generation, source_room=""))


@pytest.mark.parametrize("changes", [{"process_id": True}, {"room": ""}, {"generation": -1},
    {"source_url": "https://live.bilibili.com/999"},
    {"source_kind": "video", "room": "", "source_url": "https://user:password@www.bilibili.com/video/BV1xx411c7mD"}])
def test_untrusted_source_events_are_ignored(make_service, changes):
    service = make_service()
    initial = service.snapshot_value()
    service.bind_source(source(**changes))
    assert service.snapshot_value() == initial


def test_close_rejects_late_job_and_callback(make_service):
    service = make_service()
    _, native, chunks, _ = connect(service)
    old = job(service)
    service.close()
    native.frame()
    assert not chunks
    assert not service.validate_job(old)
    with pytest.raises(RuntimeError):
        service.create_capture(lambda _: None)
