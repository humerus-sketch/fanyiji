"""Selected-browser capture with source identity and queued-job fences.

This service owns no speaker model or voice profile. A selected video is a
valid subtitle source, but carries an empty live room for downstream sending.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import threading
import uuid

from PySide6.QtCore import QObject, Signal

from .audio import AudioCaptureManager, AudioChunk, CaptureStatus, SAMPLE_RATE
from .live_audio_capture import LiveAudioCapture
from .live_audio_protocol import normalize_live_room, normalize_reference_video_url
from .paths import data_root


class LiveSourceService(QObject):
    snapshot = Signal(dict)
    configuration_changed = Signal()

    def __init__(self, parent=None, data_dir=None, *, capture_factory=None):
        super().__init__(parent)
        self.data_dir = Path(data_dir) if data_dir is not None else data_root()
        self.settings_path = self.data_dir / "live-source-settings.json"
        self._lock = threading.RLock()
        self._source_enabled = False
        self._source = {"state": "unavailable", "room": "", "process_id": 0,
                        "generation": 0, "browser_generation": 0,
                        "source_kind": "", "source_url": ""}
        self._generation = 0
        self._adapter = None
        self._native_ready = self._source_failed = self._subtitle_active = False
        self._settings_invalid = self._closed = False
        self._error = ""
        self._capture_factory = capture_factory or LiveAudioCapture
        self._read_settings()

    def _read_settings(self) -> None:
        path = self.settings_path
        legacy = not path.exists()
        if legacy:
            path = self.data_dir / "speaker-settings.json"
        if not path.exists():
            return
        try:
            if path.stat().st_size > 4096:
                raise ValueError("source_settings_size")
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict) or type(value.get("source_enabled")) is not bool:
                raise ValueError("source_settings_format")
            if not legacy and set(value) != {"source_enabled"}:
                raise ValueError("source_settings_format")
            # Only the source switch is migrated. Voice profiles are never read.
            self._source_enabled = value["source_enabled"]
        except (OSError, ValueError, UnicodeError):
            # Corrupt source preferences must not silently capture whole-PC audio.
            self._source_enabled = self._settings_invalid = True
            self._error = "指定音源设置读取失败，请重新保存采集设置"

    def _write_settings(self, enabled: bool) -> None:
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.settings_path.with_name(self.settings_path.name + "." + uuid.uuid4().hex + ".tmp")
        try:
            with temporary.open("x", encoding="utf-8", newline="\n") as stream:
                json.dump({"source_enabled": enabled}, stream)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.settings_path)
        except OSError:
            raise OSError("指定音源设置保存失败，已有设置保留") from None
        finally:
            if temporary.exists():
                temporary.unlink()

    def _source_identity(self) -> tuple:
        source = self._source
        return tuple(source[key] for key in ("state", "source_kind", "source_url", "room",
                                            "process_id", "generation", "browser_generation"))

    def snapshot_value(self) -> dict:
        with self._lock:
            ready = bool(self._source["state"] == "ready" and self._native_ready)
            state = ("error" if self._settings_invalid or self._source_failed or self._error
                     else "disabled" if not self._source_enabled
                     else "ready" if ready else "waiting_browser")
            return {**self._source, "source_enabled": self._source_enabled,
                    "source_ready": ready, "subtitle_active": self._subtitle_active,
                    "generation": self._generation, "state": state, "message": self._error}

    def _publish(self) -> None:
        self.snapshot.emit(self.snapshot_value())

    def _assert_open(self) -> None:
        if self._closed:
            raise RuntimeError("指定音源服务已关闭")

    def set_source_enabled(self, enabled: bool) -> None:
        if type(enabled) is not bool:
            raise ValueError("指定音源开关无效")
        with self._lock:
            self._assert_open()
            if enabled == self._source_enabled and not self._settings_invalid:
                return
            self._write_settings(enabled)
            self._source_enabled = enabled
            self._settings_invalid = self._source_failed = self._native_ready = False
            self._generation += 1
            self._error = ""
            adapter = self._adapter
        if adapter is not None:
            adapter.invalidate()
        self._publish()
        self.configuration_changed.emit()

    def bind_source(self, event: dict) -> None:
        if not isinstance(event, dict) or event.get("state") not in {"ready", "navigating", "unavailable"}:
            return
        state = event["state"]
        generation, browser_generation = event.get("generation", 0), event.get("browser_generation", 0)
        if (type(generation) is not int or generation < 0
                or type(browser_generation) is not int or browser_generation < 0):
            return
        room, process_id, source_kind, source_url = "", 0, "", ""
        if state == "ready":
            source_kind = event.get("source_kind", "live")
            try:
                if source_kind == "video":
                    if event.get("room") != "":
                        return
                    source_url = normalize_reference_video_url(event.get("source_url", ""))
                    if source_url != event.get("source_url"):
                        return
                elif source_kind == "live":
                    room = normalize_live_room(event.get("room", ""))
                    source_url = "https://live.bilibili.com/" + room
                    if "source_url" in event and event["source_url"] != source_url:
                        return
                else:
                    return
            except (TypeError, ValueError):
                return
            process_id = event.get("process_id")
            if type(process_id) is not int or not 0 < process_id <= 0xFFFFFFFF:
                return
        selected = {"state": state, "room": room, "process_id": process_id,
                    "generation": generation, "browser_generation": browser_generation,
                    "source_kind": source_kind, "source_url": source_url}
        with self._lock:
            if self._closed or browser_generation < self._source["browser_generation"]:
                return
            if state == "ready" and browser_generation == self._source["browser_generation"] and generation < self._source["generation"]:
                return
            if selected == self._source:
                return
            self._source = selected
            self._source_failed = self._native_ready = False
            notify = self._source_enabled
            if notify:
                self._generation += 1
            self._error = "" if not self._settings_invalid else self._error
            adapter = self._adapter if notify else None
        if adapter is not None:
            adapter.invalidate()
        self._publish()
        if notify:
            self.configuration_changed.emit()

    def validate_job(self, job) -> bool:
        """Call before and after expensive work to reject navigated-source jobs."""
        with self._lock:
            if self._closed:
                return False
            if getattr(job, "channel", "") != "system":
                return True
            adapter = self._adapter
            generation = getattr(job, "source_generation", getattr(job, "generation", None))
            if not self._source_enabled:
                return bool(generation == self._generation and getattr(job, "source_room", "") == ""
                            and adapter is not None and adapter._valid(adapter._lease))
            return bool(not self._settings_invalid and not self._source_failed
                and self._native_ready and self._source["state"] == "ready"
                and generation == self._generation
                and getattr(job, "source_room", "") == self._source["room"]
                and adapter is not None and adapter._valid(adapter._lease)
                and adapter._source_identity == self._source_identity())

    def create_capture(self, on_chunk, on_status=None):
        self._assert_open()
        return _SelectedCapture(self, on_chunk, on_status)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._generation += 1
            self._native_ready = self._subtitle_active = False
            adapter, self._adapter = self._adapter, None
        if adapter is not None:
            adapter.stop()


class _SelectedCapture:
    """Existing microphone capture plus the selected browser process tree."""
    def __init__(self, service: LiveSourceService, on_chunk, on_status):
        self.service, self.on_chunk, self.on_status = service, on_chunk, on_status
        self._ordinary = self._native = self._lease = self._target = None
        self._counter, self._generation, self._sequence = 0, 0, -1
        self._source_identity = None
        self._source_room = ""
        self._ordinary_system_generation = None

    def start(self, enabled_channels=None) -> None:
        channels = set(enabled_channels) if enabled_channels is not None else {"mic", "system"}
        if not channels <= {"mic", "system"}:
            raise ValueError("enabled_channels must contain only 'mic' and 'system'")
        self.stop()
        service = self.service
        with service._lock:
            service._assert_open()
            previous = service._adapter
        if previous is not None and previous is not self:
            previous.stop()
        with service._lock:
            self._lease = object()
            service._adapter = self
            service._generation += 1
            self._generation = service._generation
            service._native_ready = False
            service._subtitle_active = "system" in channels
            self._counter, self._sequence = 0, -1
            self._ordinary_system_generation = None
            selected, failed, invalid = service._source_enabled, service._source_failed, service._settings_invalid
            source = dict(service._source)
            self._source_identity = service._source_identity()
            self._source_room = source["room"]
            self._target = (source["room"], self._generation, source["process_id"])
            lease = self._lease
        ordinary_channels = channels - {"system"} if selected else channels
        self._ordinary = AudioCaptureManager(lambda chunk: self._ordinary_chunk(lease, chunk),
                                            lambda status: self._ordinary_status(lease, status))
        self._ordinary.start(ordinary_channels)
        if selected and "system" in channels:
            if source["state"] == "ready" and not failed and not invalid:
                try:
                    self._native = service._capture_factory()
                    self._native.start(source["process_id"], source["room"], self._generation,
                                       lambda frame: self._frame(lease, frame),
                                       lambda status: self._status(lease, status))
                except Exception:
                    with service._lock:
                        if self._valid(lease):
                            service._native_ready = False
                            service._error = "指定音源启动失败"
                    self._send_status("error", "指定音源启动失败")
            else:
                self._send_status("error" if failed or invalid else "starting",
                                  "指定音源设置读取失败，请重新保存采集设置" if invalid else
                                  "刷新或重新打开播放来源后重试" if failed else "等待打开指定播放来源")
        service._publish()

    def _valid(self, lease) -> bool:
        service = self.service
        return bool(lease is not None and not service._closed and service._adapter is self
                    and self._lease is lease and self._generation == service._generation
                    and (not service._source_enabled or self._source_identity == service._source_identity()))

    def _ordinary_chunk(self, lease, chunk: AudioChunk) -> None:
        with self.service._lock:
            if self._valid(lease) and not (self.service._source_enabled and chunk.channel == "system"):
                if chunk.channel == "system":
                    # Device capture owns its own epochs. Map them to the same
                    # service epoch so switching from selected-video audio to
                    # whole-PC capture cannot admit an old queued video job.
                    previous = self._ordinary_system_generation
                    if previous is not None and previous != chunk.generation:
                        self.service._generation += 1
                        self._generation = self.service._generation
                    self._ordinary_system_generation = chunk.generation
                    chunk = AudioChunk(chunk.channel, chunk.samples, chunk.start_sample, chunk.end_sample,
                                       chunk.captured_at, self._generation, chunk.sample_rate, "")
                self.on_chunk(chunk)

    def _ordinary_status(self, lease, status: CaptureStatus) -> None:
        with self.service._lock:
            if self._valid(lease) and not (self.service._source_enabled and status.channel == "system") and self.on_status:
                self.on_status(status)

    def _frame(self, lease, frame) -> None:
        service = self.service
        with service._lock:
            if (not self._valid(lease) or not service._source_enabled
                    or (frame.room, frame.generation) != self._target[:2]
                    or frame.sequence <= self._sequence or not service._native_ready):
                return
            if frame.discontinuity:
                service._generation += 1
                self._generation = service._generation
                self._counter = 0
            start = self._counter
            self._counter += frame.samples.size
            self._sequence = frame.sequence
            self.on_chunk(AudioChunk("system", frame.samples, start, self._counter,
                                     frame.audio_end_monotonic, self._generation,
                                     sample_rate=SAMPLE_RATE, source_room=self._source_room))

    def _status(self, lease, status) -> None:
        service = self.service
        invalidated = False
        with service._lock:
            if not self._valid(lease) or (status.room, status.generation, status.process_id) != self._target:
                return
            service._native_ready = status.state in {"ready", "running"}
            if status.state in {"error", "stopped"}:
                code = str(status.code)
                service._error = code if re.fullmatch(r"live_audio_[a-z_]+(?::0x[0-9a-fA-F]{8})?", code) else "指定音源采集失败"
                service._source_failed = True
                service._generation += 1
                self._lease = None
                invalidated = True
            elif service._native_ready:
                service._error = ""
        state = {"ready": "running", "running": "running", "starting": "starting", "error": "error", "stopped": "stopped"}.get(status.state, "error")
        self._send_status(state, "指定视频音源已连接" if state == "running" and self._source_identity[1] == "video"
                          else "指定直播音源已连接" if state == "running" else "指定音源采集失败" if state == "error" else "等待指定播放来源")
        service._publish()
        if invalidated:
            service.configuration_changed.emit()

    def _send_status(self, state, message):
        if self.on_status:
            self.on_status(CaptureStatus("system", state, message, self._generation))

    def invalidate(self) -> None:
        with self.service._lock:
            self._lease = None
            native, self._native = self._native, None
        if native is not None:
            native.stop()

    def stop(self) -> None:
        service = self.service
        with service._lock:
            self._lease = None
            native, self._native = self._native, None
            ordinary, self._ordinary = self._ordinary, None
            if service._adapter is self:
                service._adapter = None
                service._generation += 1
                service._native_ready = service._subtitle_active = False
        if native is not None:
            native.stop()
        if ordinary is not None:
            ordinary.stop()
        service._publish()
