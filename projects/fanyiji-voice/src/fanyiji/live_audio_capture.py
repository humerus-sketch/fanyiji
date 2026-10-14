"""WASAPI process-tree capture, never falling back to device-wide loopback.

Native COM initialization and capture run in a worker. The target is the native
WebView2 browser process, so its audio-rendering children are included automatically.
No browser scripts, cookies, endpoint enumeration, or disk recordings are used.
"""
from __future__ import annotations

import ctypes
import math
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .live_audio_protocol import (
    SAMPLE_RATE, LiveAudioFrame, LiveAudioStatus, LiveAudioTarget, process_loopback_supported,
)


class LiveAudioError(RuntimeError):
    def __init__(self, code: str, hresult: int | None = None):
        self.code, self.hresult = code, hresult
        super().__init__(code if hresult is None else f"{code}: 0x{hresult & 0xFFFFFFFF:08x}")


@dataclass(frozen=True)
class RawLiveAudioPacket:
    samples: np.ndarray
    sample_rate: int
    audio_start_monotonic: float
    captured_at: float
    discontinuity: bool = False


class _Guid(ctypes.Structure):
    _fields_ = [("data1", ctypes.c_uint32), ("data2", ctypes.c_uint16),
                ("data3", ctypes.c_uint16), ("data4", ctypes.c_ubyte * 8)]

    @classmethod
    def from_string(cls, value: str):
        return cls.from_buffer_copy(uuid.UUID(value).bytes_le)


class _ActivationParams(ctypes.Structure):
    _fields_ = [("activation_type", ctypes.c_uint32), ("target_process_id", ctypes.c_uint32),
                ("process_loopback_mode", ctypes.c_uint32)]


class _Blob(ctypes.Structure):
    _fields_ = [("size", ctypes.c_uint32), ("data", ctypes.c_void_p)]


class _PropValue(ctypes.Union):
    # PROPVARIANT has a 16-byte value union on x64; BLOB itself has pointer alignment.
    _fields_ = [("blob", _Blob), ("storage", ctypes.c_ubyte * 16)]


class _PropVariant(ctypes.Structure):
    _fields_ = [("vt", ctypes.c_uint16), ("reserved1", ctypes.c_uint16),
                ("reserved2", ctypes.c_uint16), ("reserved3", ctypes.c_uint16),
                ("value", _PropValue)]


class _WaveFormat(ctypes.Structure):
    _pack_ = 2
    _fields_ = [("format_tag", ctypes.c_uint16), ("channels", ctypes.c_uint16),
                ("sample_rate", ctypes.c_uint32), ("bytes_per_second", ctypes.c_uint32),
                ("block_align", ctypes.c_uint16), ("bits_per_sample", ctypes.c_uint16),
                ("extra_size", ctypes.c_uint16)]


def _method(pointer, index, result, *arguments):
    if not pointer:
        raise LiveAudioError("live_audio_missing_interface")
    table = ctypes.cast(pointer, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    return ctypes.WINFUNCTYPE(result, ctypes.c_void_p, *arguments)(table[index])


def _check(result: int, operation: str) -> None:
    if result < 0:
        raise LiveAudioError(operation, result)


def _release(pointer) -> None:
    if pointer:
        _method(pointer, 2, ctypes.c_uint32)(pointer)


_CALLBACK_KEEPALIVE: dict[int, object] = {}
_CALLBACK_LOCK = threading.Lock()


class _ActivationCallback:
    """IUnknown / agile completion handler whose native lifetime survives timeouts."""
    def __init__(self):
        self.event = threading.Event()
        self.interface = ctypes.c_void_p()
        self.result = -2147467259
        self.references = 1
        self.lock = threading.Lock()
        query = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p,
                                  ctypes.POINTER(_Guid), ctypes.POINTER(ctypes.c_void_p))
        count = ctypes.WINFUNCTYPE(ctypes.c_uint32, ctypes.c_void_p)
        complete = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p)
        self.callbacks = (query(self._query), count(self._add_ref),
                          count(self._release), complete(self._complete))
        self.table = (ctypes.c_void_p * 4)(*[ctypes.cast(fn, ctypes.c_void_p).value
                                          for fn in self.callbacks])
        self.instance = ctypes.c_void_p(ctypes.addressof(self.table))
        self.pointer = ctypes.cast(ctypes.pointer(self.instance), ctypes.c_void_p)
        self.accepted = {uuid.UUID(value).bytes_le for value in (
            "00000000-0000-0000-c000-000000000046",  # IUnknown
            "41d949ab-9862-444a-80f6-c261334da5eb",  # Completion handler
            "94ea2b94-e9cc-49e0-c0ff-ee64ca8f5b90",  # IAgileObject
        )}
        with _CALLBACK_LOCK:
            _CALLBACK_KEEPALIVE[id(self)] = self

    def _query(self, _this, identifier, output):
        if not output:
            return -2147467261  # E_POINTER
        output[0] = None
        if identifier and ctypes.string_at(identifier, 16) in self.accepted:
            output[0] = self.pointer.value
            self._add_ref(None)
            return 0
        return -2147467262  # E_NOINTERFACE

    def _add_ref(self, _this):
        with self.lock:
            self.references += 1
            return self.references

    def _release(self, _this):
        with self.lock:
            self.references -= 1
            count = self.references
        if count == 0:
            # Timed-out activations may still return an interface; never leak it.
            _release(self.interface)
            self.interface = ctypes.c_void_p()
            with _CALLBACK_LOCK:
                _CALLBACK_KEEPALIVE.pop(id(self), None)
        return count

    def _complete(self, _this, operation):
        try:
            activated = ctypes.c_long()
            interface = ctypes.c_void_p()
            result = _method(operation, 3, ctypes.c_long, ctypes.POINTER(ctypes.c_long),
                             ctypes.POINTER(ctypes.c_void_p))(
                                 operation, ctypes.byref(activated), ctypes.byref(interface))
            self.result = result if result < 0 else activated.value
            self.interface = interface
        except Exception:
            self.result = -2147467259
        finally:
            self.event.set()
        return 0

    def take_interface(self):
        value, self.interface = self.interface, ctypes.c_void_p()
        return value


class _WindowsProcessLoopback:
    """Owns all native interfaces on one MTA capture thread."""
    SOURCE_RATE = 44100

    def __init__(self, process_id: int):
        if not process_loopback_supported():
            raise LiveAudioError("live_audio_platform_unsupported")
        self.audio = ctypes.c_void_p()
        self.capture = ctypes.c_void_p()
        self.event = None
        self.process = None
        self.started = False
        self.initialized = False
        self._ole = ctypes.WinDLL("ole32")
        self._kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self._media = ctypes.WinDLL("mmdevapi")
        self._configure_functions()
        try:
            result = self._ole.CoInitializeEx(None, 0)
            _check(result, "live_audio_com_initialize")
            self.initialized = True
            self.process = self._kernel.OpenProcess(0x00100000, False, process_id)
            if not self.process:
                raise LiveAudioError("live_audio_process_unavailable")
            self._activate(process_id)
            # This explicit format is the same PCM format as Microsoft's sample.
            # GetMixFormat is not implemented by every virtual process endpoint.
            form = _WaveFormat(1, 2, self.SOURCE_RATE, self.SOURCE_RATE * 4, 4, 16, 0)
            _check(_method(self.audio, 3, ctypes.c_long, ctypes.c_uint32, ctypes.c_uint32,
                           ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(_WaveFormat),
                           ctypes.c_void_p)(self.audio, 0, 0x00020000 | 0x00040000 | 0x80000000,
                                           0, 0, ctypes.byref(form), None), "live_audio_initialize")
            iid = _Guid.from_string("c8adbd64-e71e-48a0-a4de-185c395cd317")
            _check(_method(self.audio, 14, ctypes.c_long, ctypes.POINTER(_Guid),
                           ctypes.POINTER(ctypes.c_void_p))(
                               self.audio, ctypes.byref(iid), ctypes.byref(self.capture)),
                   "live_audio_capture_service")
            self.event = self._kernel.CreateEventW(None, False, False, None)
            if not self.event:
                raise LiveAudioError("live_audio_event_unavailable")
            _check(_method(self.audio, 13, ctypes.c_long, ctypes.c_void_p)(self.audio, self.event),
                   "live_audio_set_event")
            _check(_method(self.audio, 10, ctypes.c_long)(self.audio), "live_audio_start")
            self.started = True
            self._qpc_offset = time.monotonic() - time.perf_counter()
        except BaseException:
            self.close()
            raise

    def _configure_functions(self):
        self._ole.CoInitializeEx.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        self._ole.CoInitializeEx.restype = ctypes.c_long
        self._ole.CoUninitialize.argtypes, self._ole.CoUninitialize.restype = [], None
        self._kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
        self._kernel.OpenProcess.restype = ctypes.c_void_p
        self._kernel.CreateEventW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_wchar_p]
        self._kernel.CreateEventW.restype = ctypes.c_void_p
        self._kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        self._kernel.WaitForSingleObject.restype = ctypes.c_uint32
        self._kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        self._kernel.CloseHandle.restype = ctypes.c_int
        self._media.ActivateAudioInterfaceAsync.argtypes = [
            ctypes.c_wchar_p, ctypes.POINTER(_Guid), ctypes.POINTER(_PropVariant),
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        self._media.ActivateAudioInterfaceAsync.restype = ctypes.c_long

    def _activate(self, process_id):
        callback = _ActivationCallback()
        operation = ctypes.c_void_p()
        params = _ActivationParams(1, process_id, 0)  # INCLUDE_TARGET_PROCESS_TREE only.
        variant = _PropVariant()
        variant.vt = 65  # VT_BLOB
        variant.value.blob = _Blob(ctypes.sizeof(params), ctypes.addressof(params))
        # Async activation may outlive the worker timeout; native parameters and
        # their pointer must remain alive with the COM completion handler.
        callback.activation_storage = (params, variant)
        iid = _Guid.from_string("1cb9ad4c-dbfa-4c32-b178-c2f568a703b2")
        try:
            _check(self._media.ActivateAudioInterfaceAsync(
                "VAD\\Process_Loopback", ctypes.byref(iid), ctypes.byref(variant),
                callback.pointer, ctypes.byref(operation)), "live_audio_activate")
            if not callback.event.wait(15):
                raise LiveAudioError("live_audio_activate_timeout")
            _check(callback.result, "live_audio_activation_result")
            interface = callback.take_interface()
            if not interface:
                raise LiveAudioError("live_audio_missing_interface")
            try:
                _check(_method(interface, 0, ctypes.c_long, ctypes.POINTER(_Guid),
                               ctypes.POINTER(ctypes.c_void_p))(
                                   interface, ctypes.byref(iid), ctypes.byref(self.audio)),
                       "live_audio_client_interface")
            finally:
                _release(interface)
        finally:
            _release(operation)
            callback._release(None)

    def read(self, stop: threading.Event) -> RawLiveAudioPacket | None:
        while not stop.is_set():
            process_state = self._kernel.WaitForSingleObject(self.process, 0)
            if process_state == 0:
                raise LiveAudioError("live_audio_process_exited")
            if process_state == 0xFFFFFFFF:
                raise LiveAudioError("live_audio_process_unavailable")
            frames = ctypes.c_uint32()
            _check(_method(self.capture, 5, ctypes.c_long, ctypes.POINTER(ctypes.c_uint32))(
                self.capture, ctypes.byref(frames)), "live_audio_packet_size")
            if not frames.value:
                result = self._kernel.WaitForSingleObject(self.event, 20)
                if result == 0xFFFFFFFF:
                    raise LiveAudioError("live_audio_event_failed")
                continue
            data, flags = ctypes.c_void_p(), ctypes.c_uint32()
            position, qpc = ctypes.c_uint64(), ctypes.c_uint64()
            _check(_method(self.capture, 3, ctypes.c_long, ctypes.POINTER(ctypes.c_void_p),
                           ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
                           ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64))(
                               self.capture, ctypes.byref(data), ctypes.byref(frames),
                               ctypes.byref(flags), ctypes.byref(position), ctypes.byref(qpc)),
                   "live_audio_get_buffer")
            try:
                if flags.value & 2:
                    samples = np.zeros(frames.value, dtype=np.float32)
                elif data and frames.value:
                    pcm = np.frombuffer(ctypes.string_at(data, frames.value * 4), dtype="<i2")
                    samples = pcm.reshape(-1, 2).astype(np.float32).mean(axis=1) / 32768.0
                else:
                    raise LiveAudioError("live_audio_invalid_buffer")
                captured_at = time.monotonic()
                duration = frames.value / self.SOURCE_RATE
                audio_start = (qpc.value / 10_000_000 + self._qpc_offset
                               if qpc.value and not flags.value & 4 else captured_at - duration)
                return RawLiveAudioPacket(samples, self.SOURCE_RATE, audio_start, captured_at,
                                          bool(flags.value & 1))
            finally:
                _check(_method(self.capture, 4, ctypes.c_long, ctypes.c_uint32)(
                    self.capture, frames.value), "live_audio_release_buffer")
        return None

    def close(self):
        if self.started and self.audio:
            _method(self.audio, 11, ctypes.c_long)(self.audio)
        self.started = False
        _release(self.capture)
        self.capture = ctypes.c_void_p()
        _release(self.audio)
        self.audio = ctypes.c_void_p()
        for handle in (self.event, self.process):
            if handle:
                self._kernel.CloseHandle(handle)
        self.event = self.process = None
        if self.initialized:
            self._ole.CoUninitialize()
            self.initialized = False


class LiveAudioCapture:
    def __init__(self, *, backend_factory=None, supported=None, clock=None):
        self._backend_factory = backend_factory or _WindowsProcessLoopback
        self._supported = supported or process_loopback_supported
        self._clock = clock or time.monotonic
        self._lock = threading.RLock()
        self._epoch = 0
        self._worker = None
        self._stop = None
        self._target = None
        self._on_status = None

    def start(self, process_id: int, room: str, generation: int,
              on_frame: Callable[[LiveAudioFrame], None],
              on_status: Callable[[LiveAudioStatus], None]) -> None:
        target = LiveAudioTarget(process_id, room, generation)
        if not callable(on_frame) or not callable(on_status):
            raise ValueError("invalid_live_callback")
        self.stop()
        with self._lock:
            self._epoch += 1
            epoch = self._epoch
            self._stop = stop = threading.Event()
            self._target, self._on_status = target, on_status
            worker = threading.Thread(target=self._run,
                args=(target, epoch, stop, on_frame, on_status), name="live-room-audio", daemon=True)
            self._worker = worker
            on_status(LiveAudioStatus(target.process_id, target.room, target.generation, "starting"))
            worker.start()

    def stop(self) -> None:
        with self._lock:
            self._epoch += 1
            stop, worker = self._stop, self._worker
            target, on_status = self._target, self._on_status
            self._stop = self._worker = self._target = self._on_status = None
            if stop:
                stop.set()
        if worker and worker is not threading.current_thread():
            worker.join(timeout=2)
        if target and on_status:
            on_status(LiveAudioStatus(target.process_id, target.room, target.generation, "stopped"))

    def _deliver(self, epoch, stop, callback, value):
        # The identity check and delivery share a lock with stop(). Once stop()
        # returns, neither a late COM callback nor a drained packet can leak PCM.
        with self._lock:
            if epoch == self._epoch and not stop.is_set():
                callback(value)

    def _run(self, target, epoch, stop, on_frame, on_status):
        backend = None
        def status(state, code=""):
            self._deliver(epoch, stop, on_status, LiveAudioStatus(
                target.process_id, target.room, target.generation, state, code))
        try:
            if not self._supported():
                raise LiveAudioError("live_audio_platform_unsupported")
            backend = self._backend_factory(target.process_id)
            if stop.is_set():
                return
            import soxr
            status("ready")
            resampler, rate, anchor, emitted, sequence = None, None, None, 0, 0
            pending_discontinuity = False
            status("running")
            while not stop.is_set():
                packet = backend.read(stop)
                if packet is None:
                    continue
                if (not isinstance(packet, RawLiveAudioPacket) or packet.sample_rate <= 0
                        or packet.samples.ndim != 1 or not np.isfinite(packet.samples).all()
                        or not all(math.isfinite(value) for value in (
                            packet.audio_start_monotonic, packet.captured_at))):
                    raise LiveAudioError("live_audio_invalid_packet")
                if not packet.samples.size:
                    continue
                if resampler is None or packet.sample_rate != rate or packet.discontinuity:
                    pending_discontinuity = resampler is not None or packet.discontinuity
                    rate = packet.sample_rate
                    resampler = soxr.ResampleStream(rate, SAMPLE_RATE, 1, dtype="float32", quality="HQ")
                    anchor = packet.audio_start_monotonic - emitted / SAMPLE_RATE
                converted = resampler.resample_chunk(np.asarray(packet.samples, dtype=np.float32))
                if not converted.size:
                    continue
                converted = np.clip(converted, -1.0, 1.0).astype(np.float32, copy=False)
                end = emitted + converted.size
                frame = LiveAudioFrame(target.room, target.generation, sequence, converted,
                    anchor + emitted / SAMPLE_RATE, anchor + end / SAMPLE_RATE,
                    packet.captured_at, emitted, end, discontinuity=pending_discontinuity)
                self._deliver(epoch, stop, on_frame, frame)
                emitted, sequence, pending_discontinuity = end, sequence + 1, False
        except LiveAudioError as error:
            code = error.code if error.hresult is None else f"{error.code}:0x{error.hresult & 0xFFFFFFFF:08x}"
            status("error", code)
        except Exception:
            status("error", "live_audio_capture_failed")
        finally:
            if backend is not None:
                try:
                    backend.close()
                except Exception:
                    pass
            status("stopped")
            with self._lock:
                if epoch == self._epoch:
                    self._stop = self._worker = self._target = self._on_status = None
