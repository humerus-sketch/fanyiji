"""Bounded speech enhancement/source separation on recognition workers.

Neither backend identifies a speaker. Native offline calls start with fresh
state for every utterance so different sources never share denoiser history.
Failures preserve the original waveform and its sample clock.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
import time
from typing import Any, Callable

import numpy as np

from .audio_processing_models import (PROCESSING_MODES, processing_model_entry,
                                      processing_model_path, verify_processing_model)

SAMPLE_RATE = 16000
MAX_PROCESSING_SECONDS = 30
WET_RATIO = np.float32(0.8)
DRY_RATIO = np.float32(0.2)
GTCRN_CONTEXT_SAMPLES = 512
UVR_SAMPLE_RATE = 44100


@dataclass(frozen=True)
class ProcessingResult:
    samples: np.ndarray
    metadata: dict[str, Any]


class _InvalidOutput(ValueError):
    pass


def _create_backend(mode: str, model: Path, num_threads: int):
    import sherpa_onnx

    if mode == "gtcrn":
        config = sherpa_onnx.OfflineSpeechDenoiserConfig(
            model=sherpa_onnx.OfflineSpeechDenoiserModelConfig(
                gtcrn=sherpa_onnx.OfflineSpeechDenoiserGtcrnModelConfig(model=str(model)),
                num_threads=num_threads, debug=False, provider="cpu"))
        if not config.validate():
            raise ValueError("invalid_native_config")
        return sherpa_onnx.OfflineSpeechDenoiser(config)
    config = sherpa_onnx.OfflineSourceSeparationConfig(
        model=sherpa_onnx.OfflineSourceSeparationModelConfig(
            uvr=sherpa_onnx.OfflineSourceSeparationUvrModelConfig(model=str(model)),
            num_threads=num_threads, debug=False, provider="cpu"))
    if not config.validate():
        raise ValueError("invalid_native_config")
    return sherpa_onnx.OfflineSourceSeparation(config)


def _resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    import soxr
    return np.asarray(soxr.resample(samples, source_rate, target_rate, quality="HQ"),
                      dtype=np.float32)


class AudioProcessor:
    def __init__(self, models_dir: Path, mode: str = "gtcrn", num_threads: int = 1, *,
                 backend_factory: Callable = _create_backend,
                 verifier: Callable = verify_processing_model,
                 resampler: Callable = _resample):
        if mode not in PROCESSING_MODES:
            raise ValueError("invalid_audio_processing_mode")
        if type(num_threads) is not int or not 1 <= num_threads <= 8:
            raise ValueError("invalid_audio_processing_threads")
        self.models_dir, self.mode, self.num_threads = Path(models_dir), mode, num_threads
        self._factory, self._verifier, self._resampler = backend_factory, verifier, resampler
        self._lock = threading.RLock()
        self._native = None
        self._closed = False
        self._attempted = False
        self._last_identity = None
        self._state = "off" if mode == "off" else "unloaded"
        self._reason = "disabled" if mode == "off" else "not_loaded"
        self._error_type = ""

    @property
    def backend_info(self) -> dict[str, Any]:
        with self._lock:
            return {"backend": "sherpa_onnx", "mode": self.mode,
                    "model_id": "" if self.mode == "off" else processing_model_entry(self.mode).id,
                    "state": self._state, "ready": self._state in ("ready", "off"),
                    "reason": self._reason, "error_type": self._error_type,
                    "provider": "cpu", "num_threads": self.num_threads,
                    "sample_rate": SAMPLE_RATE, "wet_ratio": float(WET_RATIO)}

    def _file_identity(self):
        try:
            stat = processing_model_path(self.models_dir, self.mode).stat()
            return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns
        except OSError:
            return None

    def _load(self) -> None:
        if self._closed or self.mode == "off" or self._native is not None:
            return
        identity = self._file_identity()
        # Recheck a previously missing/invalid model only when the file changes.
        if self._attempted and identity == self._last_identity:
            return
        self._attempted, self._last_identity = True, identity
        try:
            verification = self._verifier(self.models_dir, self.mode)
            if not verification.ready:
                self._state = "missing" if verification.state in ("missing", "partial") else "invalid"
                self._reason = "model_missing" if self._state == "missing" else "model_invalid"
                self._error_type = ""
                return
            self._native = self._factory(self.mode, processing_model_path(self.models_dir, self.mode),
                                         self.num_threads)
            if self._native is None:
                raise ValueError("empty_backend")
            self._state, self._reason, self._error_type = "ready", "ready", ""
        except Exception as exc:
            self._native = None
            self._state, self._reason, self._error_type = "error", "load_error", type(exc).__name__

    def preload(self) -> dict[str, Any]:
        with self._lock:
            self._load()
            return self.backend_info

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._native = None
            self._state, self._reason, self._error_type = "closed", "closed", ""

    def _result(self, samples: np.ndarray, action: str, reason: str, started: float,
                *, error_type: str = "") -> ProcessingResult:
        info = self.backend_info
        metadata = {"mode": self.mode, "action": action, "reason": reason,
                    "model_id": info["model_id"], "backend_info": info,
                    "elapsed_ms": max(0.0, (time.perf_counter() - started) * 1000),
                    "input_samples": len(samples), "output_samples": len(samples),
                    "sample_rate": SAMPLE_RATE, "error_type": error_type}
        return ProcessingResult(np.ascontiguousarray(samples, dtype=np.float32).copy(), metadata)

    @staticmethod
    def _finite_audio(value, *, channels: bool = False) -> np.ndarray:
        data = np.asarray(value, dtype=np.float32)
        if channels:
            if data.ndim != 2 or data.shape[0] not in (1, 2) or not data.shape[1]:
                raise _InvalidOutput("invalid_stem_shape")
        elif data.ndim != 1 or not data.size:
            raise _InvalidOutput("invalid_audio_shape")
        if not np.isfinite(data).all():
            raise _InvalidOutput("nonfinite_audio")
        return data

    def _enhance(self, source: np.ndarray) -> np.ndarray:
        # The native STFT may discard the last partial 256-sample frame.
        # Real zero context protects the edges; crop by the original position,
        # never time-stretch or shift the utterance to force its length to fit.
        left = GTCRN_CONTEXT_SAMPLES
        padded = np.pad(source, (left, GTCRN_CONTEXT_SAMPLES))
        output = self._native(np.ascontiguousarray(padded), SAMPLE_RATE)
        if type(output.sample_rate) is not int or output.sample_rate != SAMPLE_RATE:
            raise _InvalidOutput("unexpected_sample_rate")
        enhanced = self._finite_audio(output.samples)
        if not left + len(source) <= len(enhanced) <= len(padded) + 256:
            raise _InvalidOutput("unexpected_audio_length")
        return enhanced[left:left + len(source)]

    def _separate(self, source: np.ndarray) -> np.ndarray:
        converted = self._finite_audio(self._resampler(source, SAMPLE_RATE, UVR_SAMPLE_RATE))
        expected = round(len(source) * UVR_SAMPLE_RATE / SAMPLE_RATE)
        if abs(len(converted) - expected) > 2:
            raise _InvalidOutput("invalid_input_resampling")
        # sherpa's official UVR example and native implementation both define
        # stem 0 as vocals and stem 1 as non-vocals. Channels are first.
        stereo = np.ascontiguousarray(np.repeat(converted[np.newaxis, :], 2, axis=0))
        output = self._native.process(samples=stereo, sample_rate=UVR_SAMPLE_RATE)
        if type(output.sample_rate) is not int or output.sample_rate != UVR_SAMPLE_RATE:
            raise _InvalidOutput("unexpected_sample_rate")
        if len(output.stems) != 2:
            raise _InvalidOutput("unexpected_stems")
        vocals = self._finite_audio(output.stems[0].data, channels=True)
        if abs(vocals.shape[1] - len(converted)) > 2:
            raise _InvalidOutput("unexpected_stem_length")
        mono = np.mean(vocals, axis=0, dtype=np.float32)
        restored = self._finite_audio(self._resampler(mono, UVR_SAMPLE_RATE, SAMPLE_RATE))
        if abs(len(restored) - len(source)) > 2:
            raise _InvalidOutput("unexpected_audio_length")
        if len(restored) < len(source):
            # At most two rounding samples: preserve the original tail.
            restored = np.concatenate((restored, source[len(restored):]))
        return restored[:len(source)]

    def process(self, samples: np.ndarray) -> ProcessingResult:
        started = time.perf_counter()
        source = np.asarray(samples, dtype=np.float32)
        if source.ndim != 1:
            raise ValueError("audio_processing_requires_mono")
        source = np.ascontiguousarray(source).copy()
        if not np.isfinite(source).all():
            source = np.nan_to_num(source, nan=0.0, posinf=0.0, neginf=0.0)
            return self._result(source, "bypass", "invalid_input", started)
        with self._lock:
            if self._closed:
                return self._result(source, "bypass", "closed", started)
            if self.mode == "off":
                return self._result(source, "off", "disabled", started)
            if not source.size:
                return self._result(source, "bypass", "empty_audio", started)
            if source.size > MAX_PROCESSING_SECONDS * SAMPLE_RATE:
                return self._result(source, "bypass", "audio_too_long", started)
            self._load()
            if self._native is None:
                return self._result(source, "bypass", self._reason, started,
                                    error_type=self._error_type)
            try:
                processed = self._enhance(source) if self.mode == "gtcrn" else self._separate(source)
                mixed = WET_RATIO * processed + DRY_RATIO * source
                if mixed.shape != source.shape or not np.isfinite(mixed).all():
                    raise _InvalidOutput("invalid_mixed_audio")
                mixed = np.clip(mixed, -1.0, 1.0)
                return self._result(mixed, "enhanced" if self.mode == "gtcrn" else "separated",
                                    "processed", started)
            except Exception as exc:
                return self._result(source, "bypass", "invalid_output" if isinstance(exc, _InvalidOutput)
                                    else "processing_error", started, error_type=type(exc).__name__)
