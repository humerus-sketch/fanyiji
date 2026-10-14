import ctypes
import sys
import threading
import time
import uuid

import numpy as np
import pytest

from fanyiji.live_audio_capture import (
    LiveAudioCapture, LiveAudioError, RawLiveAudioPacket, _ActivationCallback,
    _ActivationParams, _Blob, _CALLBACK_KEEPALIVE, _Guid, _PropVariant, _WaveFormat,
    _WindowsProcessLoopback, _check, _method,
)


class FakeBackend:
    def __init__(self, packets=()):
        self.packets = list(packets)
        self.closed = threading.Event()
        self.read_started = threading.Event()

    def read(self, stop):
        self.read_started.set()
        if self.packets:
            return self.packets.pop(0)
        stop.wait(.005)
        return None

    def close(self):
        self.closed.set()


def until(condition, timeout=3):
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        time.sleep(.005)
    assert condition()


def packet(duration=.1, rate=16000, at=100, amplitude=.5, discontinuity=False):
    return RawLiveAudioPacket(np.full(round(duration * rate), amplitude, dtype=np.float32),
                              rate, at, at + duration + .01, discontinuity)


def test_creation_does_not_open_native_audio():
    calls = []
    LiveAudioCapture(backend_factory=lambda pid: calls.append(pid))
    assert calls == []


def test_unsupported_platform_fails_and_stops_without_device_fallback():
    calls, states = [], []
    capture = LiveAudioCapture(backend_factory=lambda pid: calls.append(pid), supported=lambda: False)
    capture.start(123, "1698225", 7, lambda frame: pytest.fail("unexpected PCM"), states.append)
    until(lambda: any(item.state == "stopped" for item in states))
    assert calls == []
    assert [item.state for item in states] == ["starting", "error", "stopped"]
    assert states[1].code == "live_audio_platform_unsupported"
    capture.stop()


def test_native_initialization_is_background_and_scoped_to_target_pid():
    entered, release = threading.Event(), threading.Event()
    calls, frames, states = [], [], []
    backend = FakeBackend([packet()])
    def factory(pid):
        calls.append(pid)
        entered.set()
        release.wait(3)
        return backend
    capture = LiveAudioCapture(backend_factory=factory, supported=lambda: True)
    started = time.monotonic()
    capture.start(123, "1698225", 7, frames.append, states.append)
    assert time.monotonic() - started < .1
    assert entered.wait(1)
    assert [item.state for item in states] == ["starting"]
    release.set()
    until(lambda: bool(frames))
    assert calls == [123]
    assert [item.state for item in states[:3]] == ["starting", "ready", "running"]
    assert {(item.process_id, item.room, item.generation) for item in states} == {(123, "1698225", 7)}
    assert frames[0].room == "1698225" and frames[0].generation == 7
    capture.stop()
    assert backend.closed.is_set()


@pytest.mark.parametrize("rate", [16000, 44100, 48000])
def test_pcm_resampled_to_mono_16k_with_contiguous_source_time(rate):
    frames, states = [], []
    backend = FakeBackend([packet(rate=rate, at=100 + i * .1) for i in range(12)])
    capture = LiveAudioCapture(backend_factory=lambda pid: backend, supported=lambda: True)
    capture.start(123, "1698225", 1, frames.append, states.append)
    until(lambda: backend.read_started.is_set() and len(frames) >= 5)
    until(lambda: not backend.packets)
    capture.stop()
    assert all(frame.sample_rate == 16000 and frame.samples.dtype == np.float32
               and frame.samples.ndim == 1 for frame in frames)
    assert .95 * 16000 <= sum(frame.samples.size for frame in frames) <= 1.2 * 16000
    assert np.mean(np.concatenate([frame.samples for frame in frames])[100:]) == pytest.approx(.5, abs=.001)
    assert frames[0].audio_start_monotonic == pytest.approx(100)
    for before, after in zip(frames, frames[1:]):
        assert before.end_sample == after.start_sample
        assert before.sequence + 1 == after.sequence
        assert before.audio_end_monotonic == pytest.approx(after.audio_start_monotonic)


def test_room_switch_drops_late_packet_from_old_generation():
    late_started, release_late = threading.Event(), threading.Event()
    class LateBackend(FakeBackend):
        def read(self, stop):
            late_started.set()
            release_late.wait(5)  # Deliberately ignores cancellation to exercise epoch fence.
            return packet()
    old, new = LateBackend(), FakeBackend([packet(at=200)])
    frames, states = [], []
    capture = LiveAudioCapture(backend_factory=lambda pid: old if pid == 123 else new,
                               supported=lambda: True)
    capture.start(123, "1698225", 1, frames.append, states.append)
    assert late_started.wait(1)
    capture.start(456, "21399689", 2, frames.append, states.append)
    until(lambda: bool(frames))
    release_late.set()
    assert old.closed.wait(1)
    capture.stop()
    assert frames and all(frame.room == "21399689" and frame.generation == 2 for frame in frames)


def test_stop_from_frame_callback_does_not_join_itself_or_emit_more_pcm():
    frames, states = [], []
    backend = FakeBackend([packet(), packet(at=101)])
    capture = LiveAudioCapture(backend_factory=lambda pid: backend, supported=lambda: True)
    def receive(frame):
        frames.append(frame)
        capture.stop()
    capture.start(123, "1698225", 1, receive, states.append)
    assert backend.closed.wait(1)
    assert len(frames) == 1
    assert states[-1].state == "stopped"


def test_activation_failure_keeps_hresult_for_diagnosis():
    states = []
    def factory(pid):
        raise LiveAudioError("live_audio_activate", -2147483634)  # E_ILLEGAL_METHOD_CALL
    capture = LiveAudioCapture(backend_factory=factory, supported=lambda: True)
    capture.start(123, "1698225", 1, lambda _: None, states.append)
    until(lambda: states[-1].state == "stopped")
    assert states[1].code == "live_audio_activate:0x8000000e"
    capture.stop()


def test_bad_packet_stops_source_and_releases_backend():
    states = []
    backend = FakeBackend([RawLiveAudioPacket(np.array([np.nan], dtype=np.float32), 16000, 1, 2)])
    capture = LiveAudioCapture(backend_factory=lambda pid: backend, supported=lambda: True)
    capture.start(123, "1698225", 1, lambda _: pytest.fail("invalid PCM leaked"), states.append)
    assert backend.closed.wait(1)
    assert [row.state for row in states] == ["starting", "ready", "running", "error", "stopped"]
    assert states[-2].code == "live_audio_invalid_packet"
    capture.stop()


def test_discontinuity_restarts_resampler_but_preserves_sample_sequence():
    frames, states = [], []
    backend = FakeBackend([packet(at=100), packet(at=200, discontinuity=True)])
    capture = LiveAudioCapture(backend_factory=lambda pid: backend, supported=lambda: True)
    capture.start(123, "1698225", 1, frames.append, states.append)
    until(lambda: len(frames) == 2)
    capture.stop()
    assert frames[1].discontinuity is True
    assert frames[1].audio_start_monotonic == pytest.approx(200)
    assert frames[1].start_sample == frames[0].end_sample


@pytest.mark.skipif(sys.platform != "win32", reason="Windows COM ABI")
def test_native_activation_structure_layout_matches_windows_sdk():
    assert ctypes.sizeof(_Guid) == 16
    assert ctypes.sizeof(_ActivationParams) == 12
    assert ctypes.sizeof(_WaveFormat) == 18
    assert _PropVariant.value.offset == 8
    if ctypes.sizeof(ctypes.c_void_p) == 8:
        assert ctypes.sizeof(_PropVariant) == 24
        assert _Blob.data.offset == 8
    params = _ActivationParams(1, 123, 0)
    assert params.activation_type == 1 and params.target_process_id == 123
    assert params.process_loopback_mode == 0


@pytest.mark.skipif(sys.platform != "win32", reason="Windows COM ABI")
def test_completion_handler_supports_agile_interface_and_native_refcount():
    callback = _ActivationCallback()
    identifier = _Guid.from_string("94ea2b94-e9cc-49e0-c0ff-ee64ca8f5b90")
    output = ctypes.c_void_p()
    try:
        assert id(callback) in _CALLBACK_KEEPALIVE
        assert callback._query(None, ctypes.pointer(identifier), ctypes.pointer(output)) == 0
        assert output.value == callback.pointer.value
        assert callback.references == 2
        assert callback._release(None) == 1
        unknown = _Guid.from_string(str(uuid.uuid4()))
        assert callback._query(None, ctypes.pointer(unknown), ctypes.pointer(output)) == -2147467262
        assert output.value is None
    finally:
        assert callback._release(None) == 0
    assert id(callback) not in _CALLBACK_KEEPALIVE


@pytest.mark.skipif(sys.platform != "win32", reason="Windows COM ABI")
def test_completion_handler_callback_lives_until_final_native_release():
    callback = _ActivationCallback()
    callback._add_ref(None)  # Windows owns a reference while activation is pending.
    assert callback._release(None) == 1  # Simulate the worker timing out.
    assert id(callback) in _CALLBACK_KEEPALIVE
    assert callback._release(None) == 0
    assert id(callback) not in _CALLBACK_KEEPALIVE


@pytest.mark.skipif(sys.platform != "win32", reason="Windows process-loopback API")
def test_native_process_activation_and_format_without_start_or_recording():
    """Exercise actual COM/PCM ABI on our own silent process; never start capture."""
    from fanyiji.live_audio_protocol import process_loopback_supported
    if not process_loopback_supported():
        pytest.skip("Windows process loopback requires build 20348 or later")
    import os
    native = _WindowsProcessLoopback.__new__(_WindowsProcessLoopback)
    native.audio, native.capture = ctypes.c_void_p(), ctypes.c_void_p()
    native.event = native.process = None
    native.started = native.initialized = False
    native._ole = ctypes.WinDLL("ole32")
    native._kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    native._media = ctypes.WinDLL("mmdevapi")
    native._configure_functions()
    try:
        initialized = native._ole.CoInitializeEx(None, 0)
        if initialized < 0 and initialized & 0xFFFFFFFF == 0x80010106:
            pytest.skip("This test runner already uses an STA COM apartment")
        _check(initialized, "test_com_initialize")
        native.initialized = True
        native._activate(os.getpid())
        assert native.audio
        form = _WaveFormat(1, 2, 44100, 176400, 4, 16, 0)
        _check(_method(native.audio, 3, ctypes.c_long, ctypes.c_uint32, ctypes.c_uint32,
                       ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(_WaveFormat),
                       ctypes.c_void_p)(native.audio, 0, 0x00020000 | 0x00040000 | 0x80000000,
                                       0, 0, ctypes.byref(form), None), "test_format_initialize")
        identifier = _Guid.from_string("c8adbd64-e71e-48a0-a4de-185c395cd317")
        _check(_method(native.audio, 14, ctypes.c_long, ctypes.POINTER(_Guid),
                       ctypes.POINTER(ctypes.c_void_p))(
                           native.audio, ctypes.byref(identifier), ctypes.byref(native.capture)),
               "test_capture_interface")
        assert native.capture
        native.event = native._kernel.CreateEventW(None, False, False, None)
        assert native.event
        _check(_method(native.audio, 13, ctypes.c_long, ctypes.c_void_p)(native.audio, native.event),
               "test_set_event")
        assert native.started is False
    finally:
        native.close()
    assert not native.audio and not native.capture
