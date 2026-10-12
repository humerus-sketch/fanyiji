"""Windows WASAPI capture, with no device access before ``start()``.

PortAudio caches its device list. A default-endpoint change therefore closes
both streams and recreates the shared context; each source gets a new generation.
The callback only queues PCM. Conversion and application callbacks run outside
the PortAudio callback, on one independent worker per enabled source.
"""
from __future__ import annotations

import ctypes
import logging
import queue
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Literal

import numpy as np

LOGGER = logging.getLogger(__name__)
Channel = Literal["mic", "system"]
CaptureState = Literal["starting", "running", "reconnecting", "error", "stopped"]
SAMPLE_RATE = 16000


@dataclass(frozen=True)
class AudioChunk:
    channel: str
    samples: np.ndarray
    start_sample: int
    end_sample: int
    captured_at: float
    generation: int
    sample_rate: int = SAMPLE_RATE
    source_room: str = ""


@dataclass(frozen=True)
class CaptureStatus:
    channel: str
    state: CaptureState
    message: str
    generation: int


@dataclass(frozen=True)
class AudioCaptureSettings:
    block_seconds: float = 0.032
    default_device_poll_seconds: float = 1.0
    reconnect_seconds: float = 1.0
    raw_queue_size: int = 64

    def __post_init__(self) -> None:
        if not 0.01 <= self.block_seconds <= 0.2:
            raise ValueError("block_seconds must be between 0.01 and 0.2")
        if self.default_device_poll_seconds <= 0 or self.reconnect_seconds <= 0:
            raise ValueError("Device poll and reconnect intervals must be positive")
        if self.raw_queue_size < 2:
            raise ValueError("raw_queue_size must be at least 2")


@dataclass
class _CaptureSource:
    channel: str
    generation: int
    rate: int
    channels: int
    resampler: Any
    raw_queue: queue.Queue[tuple[bytes, float]]
    discontinuity: threading.Event
    stop: threading.Event
    failed: threading.Event
    stream: Any
    worker: threading.Thread | None = None


def pcm_to_mono(raw: bytes, channels: int) -> np.ndarray:
    """Convert native float32 interleaved PCM; never infer the device rate."""
    if channels < 1:
        raise ValueError("The capture device has no input channels")
    values = np.frombuffer(raw, dtype=np.float32)
    if values.size % channels:
        raise ValueError("Incomplete interleaved audio frame")
    mono = values.reshape(-1, channels).mean(axis=1, dtype=np.float32)
    np.nan_to_num(mono, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(mono, -1.0, 1.0).astype(np.float32, copy=False)


class _Guid(ctypes.Structure):
    _fields_ = [("data1", ctypes.c_uint32), ("data2", ctypes.c_uint16),
                ("data3", ctypes.c_uint16), ("data4", ctypes.c_ubyte * 8)]

    @classmethod
    def from_string(cls, value: str) -> _Guid:
        return cls.from_buffer_copy(uuid.UUID(value).bytes_le)


def _com_method(pointer: ctypes.c_void_p, index: int, result: Any, *args: Any) -> Any:
    table = ctypes.cast(pointer, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    return ctypes.WINFUNCTYPE(result, ctypes.c_void_p, *args)(table[index])


class _DefaultEndpointProbe:
    """Read multimedia endpoint IDs through MMDevice; no audio is opened here."""

    def __init__(self) -> None:
        if sys.platform != "win32":
            raise RuntimeError("WASAPI capture is available only on Windows")
        # WinDLL leaves HRESULT checking to us and does not misinterpret the
        # void return values of CoTaskMemFree / CoUninitialize as HRESULTs.
        self._ole = ctypes.WinDLL("ole32")
        self._ole.CoInitializeEx.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        self._ole.CoInitializeEx.restype = ctypes.c_long
        self._ole.CoCreateInstance.argtypes = [ctypes.POINTER(_Guid), ctypes.c_void_p,
                                              ctypes.c_uint32, ctypes.POINTER(_Guid),
                                              ctypes.POINTER(ctypes.c_void_p)]
        self._ole.CoCreateInstance.restype = ctypes.c_long
        self._ole.CoTaskMemFree.argtypes = [ctypes.c_void_p]
        self._ole.CoTaskMemFree.restype = None
        self._ole.CoUninitialize.argtypes = []
        self._ole.CoUninitialize.restype = None
        # Use the same apartment model as PortAudio's WASAPI initialization.
        result = self._ole.CoInitializeEx(None, 2)
        self._uninitialize = result in (0, 1)
        if result < 0 and (result & 0xFFFFFFFF) != 0x80010106:
            raise OSError(f"CoInitializeEx failed: 0x{result & 0xFFFFFFFF:08x}")
        self._enumerator = ctypes.c_void_p()
        clsid = _Guid.from_string("bcde0395-e52f-467c-8e3d-c4579291692e")
        iid = _Guid.from_string("a95664d2-9614-4f35-a746-de8db63617e6")
        try:
            result = self._ole.CoCreateInstance(ctypes.byref(clsid), None, 1,
                                                ctypes.byref(iid), ctypes.byref(self._enumerator))
            if result < 0:
                raise OSError(f"MMDeviceEnumerator failed: 0x{result & 0xFFFFFFFF:08x}")
        except Exception:
            self.close()
            raise

    def read(self, channels: set[str]) -> dict[str, str | None]:
        return {channel: self._endpoint_id(1 if channel == "mic" else 0)
                for channel in sorted(channels)}

    def _endpoint_id(self, flow: int) -> str | None:
        device = ctypes.c_void_p()
        # IMMDeviceEnumerator::GetDefaultAudioEndpoint(eCapture/eRender, eMultimedia).
        result = _com_method(self._enumerator, 4, ctypes.c_long, ctypes.c_int,
                             ctypes.c_int, ctypes.POINTER(ctypes.c_void_p))(
                                 self._enumerator, flow, 1, ctypes.byref(device))
        if result < 0:
            if (result & 0xFFFFFFFF) == 0x80070490:  # E_NOTFOUND: no default device.
                return None
            raise OSError(f"Default endpoint query failed: 0x{result & 0xFFFFFFFF:08x}")
        device_id = ctypes.c_void_p()
        try:
            result = _com_method(device, 5, ctypes.c_long, ctypes.POINTER(ctypes.c_void_p))(
                device, ctypes.byref(device_id))
            if result < 0:
                raise OSError(f"Endpoint ID query failed: 0x{result & 0xFFFFFFFF:08x}")
            return ctypes.wstring_at(device_id)
        finally:
            if device_id.value:
                self._ole.CoTaskMemFree(device_id)
            _com_method(device, 2, ctypes.c_ulong)(device)

    def close(self) -> None:
        if getattr(self, "_enumerator", None) and self._enumerator.value:
            _com_method(self._enumerator, 2, ctypes.c_ulong)(self._enumerator)
            self._enumerator = ctypes.c_void_p()
        if getattr(self, "_uninitialize", False):
            self._ole.CoUninitialize()
            self._uninitialize = False


class AudioCaptureManager:
    """Application callbacks must enqueue work promptly, without ASR or MT.

    ``captured_at`` is the estimated monotonic time of the chunk's last output
    sample, including SoXR's pending output delay. Sample counters are local to
    each channel and generation. ``start(set())`` never imports audio backends.
    """

    def __init__(self, on_chunk: Callable[[AudioChunk], None],
                 on_status: Callable[[CaptureStatus], None] | None = None,
                 settings: AudioCaptureSettings | None = None,
                 *, _probe_factory: Callable[[], Any] | None = None,
                 _backend_loader: Callable[[], tuple[Any, Any]] | None = None) -> None:
        self._on_chunk = on_chunk
        self._on_status = on_status
        self.settings = settings or AudioCaptureSettings()
        self._probe_factory = _probe_factory or _DefaultEndpointProbe
        self._backend_loader = _backend_loader or self._load_backends
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._channels: set[str] = set()
        self._generation = 0
        self._source_generations: dict[str, int] = {}
        self._lifecycle = threading.RLock()

    @staticmethod
    def _load_backends() -> tuple[Any, Any]:
        if sys.platform != "win32":
            raise RuntimeError("音频采集仅支持 Windows WASAPI")
        import pyaudiowpatch
        import soxr
        return pyaudiowpatch, soxr

    def start(self, enabled_channels: set[str] | None = None) -> None:
        channels = set(enabled_channels) if enabled_channels is not None else {"mic", "system"}
        if not channels <= {"mic", "system"}:
            raise ValueError("enabled_channels must contain only 'mic' and 'system'")
        with self._lifecycle:
            if self._thread is not None and self._thread.is_alive():
                if channels == self._channels:
                    return
                self.stop()
                if self._thread is not None and self._thread.is_alive():
                    raise RuntimeError("上一轮音频采集尚未停止")
            self._channels = channels
            if not channels:
                return
            self._stop = threading.Event()
            self._thread = threading.Thread(target=self._supervise, name="wasapi-supervisor", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        with self._lifecycle:
            self._stop.set()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5.0)
        with self._lifecycle:
            if thread is self._thread and (thread is None or not thread.is_alive()):
                self._thread = None

    def _status(self, channel: str, state: CaptureState, message: str, generation: int) -> None:
        if self._on_status is None:
            return
        try:
            self._on_status(CaptureStatus(channel, state, message, generation))
        except Exception:
            LOGGER.exception("Capture status callback failed")

    def _supervise(self) -> None:
        probe = None
        try:
            pyaudio, soxr = self._backend_loader()
            probe = self._probe_factory()
            while not self._stop.is_set():
                context = None
                sources: dict[str, _CaptureSource] = {}
                retry_after = {channel: 0.0 for channel in self._channels}
                blocked: set[str] = set()
                try:
                    baseline = probe.read(self._channels)
                    context = pyaudio.PyAudio()
                    next_probe = time.monotonic() + self.settings.default_device_poll_seconds
                    while not self._stop.is_set():
                        now = time.monotonic()
                        if now >= next_probe:
                            current = probe.read(self._channels)
                            next_probe = now + self.settings.default_device_poll_seconds
                            if current != baseline:
                                for channel in sorted(self._channels):
                                    self._status(channel, "reconnecting",
                                        "默认设备已变化，正在重连已启用的音频通道…",
                                        self._source_generations.get(channel, 0))
                                break
                        for channel in sorted(self._channels):
                            source = sources.get(channel)
                            if source is not None:
                                if not source.failed.is_set():
                                    try:
                                        if not source.stream.is_active():
                                            self._status(channel, "error", "本路音频设备连接已中断", source.generation)
                                            source.failed.set()
                                    except Exception as exc:
                                        self._status(channel, "error", f"本路设备状态异常：{exc}", source.generation)
                                        source.failed.set()
                                if source.failed.is_set():
                                    closed = self._close_source(source)
                                    if not closed:
                                        blocked.add(channel)
                                    del sources[channel]
                                    retry_after[channel] = now + self.settings.reconnect_seconds
                                    if closed:
                                        self._status(channel, "reconnecting", "本路采集正在重连，另一通道继续运行…", source.generation)
                            elif channel not in blocked and now >= retry_after[channel]:
                                self._generation += 1
                                generation = self._generation
                                self._source_generations[channel] = generation
                                self._status(channel, "starting", "正在连接默认音频设备…", generation)
                                try:
                                    # Keep every PortAudio/COM call on its initialization thread.
                                    sources[channel] = self._open_source(channel, generation, context, pyaudio, soxr)
                                except Exception as exc:
                                    self._status(channel, "error", f"本路采集失败：{exc}", generation)
                                    retry_after[channel] = now + self.settings.reconnect_seconds
                        self._stop.wait(0.05)
                except Exception as exc:
                    for channel in sorted(self._channels):
                        self._status(channel, "error", f"音频设备上下文不可用：{exc}",
                                     self._source_generations.get(channel, 0))
                finally:
                    for source in sources.values():
                        if not self._close_source(source):
                            # Do not start a new generation while an old callback
                            # can still mutate the same source's segmenter.
                            self._stop.set()
                    if context is not None:
                        context.terminate()
                if not self._stop.is_set():
                    for channel in sorted(self._channels):
                        self._status(channel, "reconnecting",
                            "正在重建默认设备连接（已启用的通道会一起重连）…",
                            self._source_generations.get(channel, 0))
                    self._stop.wait(self.settings.reconnect_seconds)
        except Exception as exc:
            for channel in sorted(self._channels):
                self._status(channel, "error", f"无法启动音频采集：{exc}",
                             self._source_generations.get(channel, 0))
        finally:
            if probe is not None:
                probe.close()
            for channel in sorted(self._channels):
                self._status(channel, "stopped", "音频采集已停止",
                             self._source_generations.get(channel, 0))

    def _open_source(self, channel: str, generation: int, context: Any,
                     pyaudio: Any, soxr: Any) -> _CaptureSource:
        if channel == "system":
            device = context.get_default_wasapi_loopback()
        else:
            host = context.get_host_api_info_by_type(pyaudio.paWASAPI)
            index = int(host["defaultInputDevice"])
            if index < 0:
                raise RuntimeError("未找到默认麦克风")
            device = context.get_device_info_by_index(index)
            if device.get("isLoopbackDevice", False):
                raise RuntimeError("默认麦克风是回环设备，请选择实体麦克风")
        rate = int(device["defaultSampleRate"])
        channels = int(device["maxInputChannels"])
        if rate <= 0 or channels <= 0:
            raise RuntimeError("默认设备没有有效采集格式")
        resampler = None if rate == SAMPLE_RATE else soxr.ResampleStream(
            rate, SAMPLE_RATE, 1, dtype="float32", quality="MQ")
        source = _CaptureSource(channel, generation, rate, channels, resampler,
            queue.Queue(self.settings.raw_queue_size), threading.Event(), threading.Event(),
            threading.Event(), None)

        def callback(raw: bytes, frame_count: int, timing: dict, flags: int) -> tuple[None, int]:
            if self._stop.is_set() or source.stop.is_set():
                return None, pyaudio.paComplete
            if flags:
                source.discontinuity.set()
            # PortAudio stream time and ADC time share a clock. Translate the
            # last captured frame to monotonic; never use a wall-clock timestamp.
            now = time.monotonic()
            adc_end = float(timing.get("input_buffer_adc_time", 0.0)) + frame_count / rate
            current = float(timing.get("current_time", 0.0))
            lag = current - adc_end
            captured_at = now - lag if current > 0 and adc_end > 0 and 0 <= lag <= 5 else now
            try:
                source.raw_queue.put_nowait((raw, captured_at))
            except queue.Full:
                source.discontinuity.set()
            return None, pyaudio.paContinue

        try:
            source.stream = context.open(format=pyaudio.paFloat32, channels=channels, rate=rate,
                input=True, input_device_index=int(device["index"]),
                frames_per_buffer=max(1, round(rate * self.settings.block_seconds)),
                stream_callback=callback, start=False)
            source.worker = threading.Thread(target=self._capture, args=(source,),
                name=f"wasapi-{channel}", daemon=True)
            source.worker.start()
            if not self._stop.is_set():
                source.stream.start_stream()
                self._status(channel, "running", f"已连接：{device['name']}", generation)
            return source
        except Exception:
            self._close_source(source)
            raise

    def _close_source(self, source: _CaptureSource) -> bool:
        source.stop.set()
        if source.stream is not None:
            try:
                if source.stream.is_active():
                    source.stream.stop_stream()
            except Exception:
                LOGGER.warning("Failed to stop WASAPI %s stream", source.channel)
            try:
                source.stream.close()
            except Exception:
                LOGGER.warning("Failed to close WASAPI %s stream", source.channel)
        if source.worker is not None and source.worker is not threading.current_thread():
            source.worker.join(timeout=1.0)
            if source.worker.is_alive():
                self._status(source.channel, "error", "本路音频处理未及时退出，请暂停后重启软件。", source.generation)
                return False
        return True

    def _capture(self, source: _CaptureSource) -> None:
        counter = 0
        try:
            while not self._stop.is_set() and not source.stop.is_set():
                if source.discontinuity.is_set():
                    raise RuntimeError("采集缓冲区溢出或数据中断，正在重连以重置分句")
                try:
                    raw, captured_at = source.raw_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                samples = pcm_to_mono(raw, source.channels)
                if source.resampler is not None:
                    samples = source.resampler.resample_chunk(samples, last=False)
                    captured_at -= float(source.resampler.delay()) / SAMPLE_RATE
                if not samples.size or self._stop.is_set() or source.stop.is_set():
                    continue
                samples = np.ascontiguousarray(samples, dtype=np.float32)
                end = counter + samples.size
                self._on_chunk(AudioChunk(source.channel, samples, counter, end, captured_at, source.generation))
                counter = end
        except Exception as exc:
            if not self._stop.is_set() and not source.stop.is_set():
                self._status(source.channel, "error", f"本路采集失败：{exc}", source.generation)
                source.failed.set()
