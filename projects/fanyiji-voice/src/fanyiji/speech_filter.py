"""Conservative local speech gate before ASR, using all AudioSet class scores.

This classifier masks clearly non-spoken regions; it does not separate a
speaker from singing that overlaps the same audio samples.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import threading
import time

import numpy as np

from .engines import ModelAssetsError


SAMPLE_RATE = 16000
MODEL_FILENAME = "model.int8.onnx"
LABELS_FILENAME = "class_labels_indices.csv"
MODEL_SIZE_BYTES = 27038066
MODEL_SHA256 = "69304b8a1b96bbe6b7d16c24079f0732c65faf3568a14cb82a8238c709afe76c"
LABELS_SIZE_BYTES = 14675
LABELS_SHA256 = "cdd1049833c4b86127c2773ac0d14a2754b6a6d0d1798002ed5c66e699708429"
SPEECH_IDS = (0, 1, 2, 3, 4, 5, 7, 15)
SINGING_IDS = tuple(range(27, 38)) + (254, 255)
MUSIC_IDS = (137, 267, 270, 272)
CLASS_COUNT = 527


@dataclass(frozen=True)
class FilterDecision:
    samples: np.ndarray
    metadata: dict

    @property
    def skip(self) -> bool:
        return self.metadata["action"] == "skip"


class SpeechOnlyFilter:
    """A CPU-only, serialized filter; run it on a worker, not audio callbacks."""

    def __init__(self, models_dir: Path, *, num_threads: int = 1, tagger=None):
        if not isinstance(num_threads, int) or isinstance(num_threads, bool) or not 1 <= num_threads <= 4:
            raise ValueError("Speech filter num_threads 必须在 1 到 4 之间")
        base = Path(models_dir).resolve()
        candidates = (base, base / "speech_filter", base / "zipformer-small-int8",
                      base / "speech_filter" / "zipformer-small-int8")
        self.models_dir = next((path for path in candidates if (path / MODEL_FILENAME).is_file()),
                               base if base.name in ("speech_filter", "zipformer-small-int8") else base / "speech_filter")
        self.num_threads = num_threads
        self._tagger = tagger
        self._injected = tagger is not None
        self._closed = False
        self._load_seconds = 0.0
        self._lock = threading.RLock()
        self.thresholds = {
            "window_seconds": 2.0, "hop_seconds": 1.0, "short_preserve_seconds": 1.5,
            "singing_min": 0.45, "singing_speech_max": 0.20, "singing_margin_min": 0.25,
            "music_min": 0.80, "music_speech_max": 0.08, "music_margin_min": 0.72,
            "speech_boundary_padding_seconds": 0.20, "minimum_mask_seconds": 0.25,
        }

    @property
    def backend_info(self) -> dict:
        with self._lock:
            return {"backend": "sherpa_onnx", "model_kind": "zipformer_audio_tagging",
                    "model_id": "zipformer-small-int8", "device": "cpu", "num_threads": self.num_threads,
                    "sample_rate": SAMPLE_RATE, "class_count": CLASS_COUNT,
                    "model_loaded": self._tagger is not None, "injected_tagger": self._injected,
                    "asset_sha256": {MODEL_FILENAME: MODEL_SHA256, LABELS_FILENAME: LABELS_SHA256},
                    "thresholds": dict(self.thresholds), "score_groups": {
                        "speech": list(SPEECH_IDS), "singing": list(SINGING_IDS), "music": list(MUSIC_IDS)},
                    "policy": "Only strong non-speech evidence is masked; overlapping keep windows preserve speech",
                    "limitation": "Overlapping speech and singing are not separated; scores are not measured accuracy"}

    def preload(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("Speech filter 已关闭")
            if self._tagger is not None:
                return
            started = time.monotonic()
            for filename, size, sha256 in ((MODEL_FILENAME, MODEL_SIZE_BYTES, MODEL_SHA256),
                                           (LABELS_FILENAME, LABELS_SIZE_BYTES, LABELS_SHA256)):
                path = self.models_dir / filename
                if not path.is_file() or path.stat().st_size != size:
                    raise ModelAssetsError(f"只翻译说话需要本地声音分类模型：{path}")
                with path.open("rb") as source:
                    if hashlib.file_digest(source, "sha256").hexdigest() != sha256:
                        raise ModelAssetsError(f"声音分类模型校验失败：{filename}")
            import sherpa_onnx

            config = sherpa_onnx.AudioTaggingConfig(
                model=sherpa_onnx.AudioTaggingModelConfig(
                    zipformer=sherpa_onnx.OfflineZipformerAudioTaggingModelConfig(
                        model=str(self.models_dir / MODEL_FILENAME)),
                    num_threads=self.num_threads, provider="cpu", debug=False),
                labels=str(self.models_dir / LABELS_FILENAME), top_k=CLASS_COUNT)
            if not config.validate():
                raise ModelAssetsError("声音分类模型配置无效")
            self._tagger = sherpa_onnx.AudioTagging(config)
            self._load_seconds = time.monotonic() - started

    def _window(self, samples: np.ndarray) -> dict:
        started = time.monotonic()
        stream = self._tagger.create_stream()
        stream.accept_waveform(sample_rate=SAMPLE_RATE, waveform=samples.copy())
        events = self._tagger.compute(stream, top_k=CLASS_COUNT)
        by_index = {}
        for event in events:
            index, score = int(event.index), float(event.prob)
            if index in by_index or not np.isfinite(score) or not 0 <= score <= 1:
                raise RuntimeError("声音分类模型返回了无效类别分数")
            by_index[index] = score
        if len(events) != CLASS_COUNT or set(by_index) != set(range(CLASS_COUNT)):
            raise RuntimeError("声音分类模型未返回完整类别分数")
        scores = {name: max(by_index[index] for index in indices)
                  for name, indices in (("speech", SPEECH_IDS), ("singing", SINGING_IDS), ("music", MUSIC_IDS))}
        speech, singing, music = scores["speech"], scores["singing"], scores["music"]
        thresholds = self.thresholds
        if (singing >= thresholds["singing_min"] and speech <= thresholds["singing_speech_max"]
                and singing - speech >= thresholds["singing_margin_min"]):
            decision, reason = "reject", "singing_without_clear_speech"
        elif (music >= thresholds["music_min"] and speech <= thresholds["music_speech_max"]
              and music - speech >= thresholds["music_margin_min"]):
            decision, reason = "reject", "music_without_clear_speech"
        else:
            decision, reason = "keep", "speech_or_uncertain"
        return {"decision": decision, "reason": reason, "scores": scores,
                "speech_minus_singing": speech - singing, "speech_minus_music": speech - music,
                "selected_class_scores": {str(index): by_index[index] for index in
                                          sorted(set(SPEECH_IDS + SINGING_IDS + MUSIC_IDS))},
                "processing_seconds": time.monotonic() - started}

    def process(self, samples: np.ndarray) -> FilterDecision:
        if not isinstance(samples, np.ndarray) or samples.dtype != np.float32 or samples.ndim != 1:
            raise ValueError("Speech filter 需要一维 float32 numpy.ndarray")
        if not np.isfinite(samples).all():
            raise ValueError("Speech filter 音频不得包含 NaN 或 Infinity")
        started = time.monotonic()
        output = np.ascontiguousarray(samples).copy()
        metadata = {"action": "keep", "reason": "speech_or_uncertain", "windows": [],
                    "thresholds": dict(self.thresholds), "input_duration_seconds": output.size / SAMPLE_RATE,
                    "masked_duration_seconds": 0.0, "retained_fraction": 1.0,
                    "tagging_seconds": 0.0, "load_seconds": self._load_seconds,
                    "same_length": True, "overlap_source_separation_performed": False}
        with self._lock:
            if self._closed:
                raise RuntimeError("Speech filter 已关闭")
            if not output.size:
                metadata.update(action="skip", reason="empty_audio", retained_fraction=0.0)
            elif output.size / SAMPLE_RATE < self.thresholds["short_preserve_seconds"]:
                # A real 0.616 s greeting scored Speech=.146. Do not gate it from
                # weak short-window evidence or silently claim classification.
                metadata["reason"] = "short_audio_preserved_without_classification"
            else:
                self.preload()
                window_size = round(self.thresholds["window_seconds"] * SAMPLE_RATE)
                hop_size = round(self.thresholds["hop_seconds"] * SAMPLE_RATE)
                last_start = max(0, output.size - window_size)
                starts = list(range(0, last_start + 1, hop_size))
                if starts[-1] != last_start:
                    starts.append(last_start)
                rejected = np.zeros(output.size, dtype=bool)
                preserved = np.zeros(output.size, dtype=bool)
                padding = round(self.thresholds["speech_boundary_padding_seconds"] * SAMPLE_RATE)
                for begin in starts:
                    end = min(output.size, begin + window_size)
                    window = self._window(output[begin:end])
                    window.update(start_seconds=begin / SAMPLE_RATE, end_seconds=end / SAMPLE_RATE)
                    metadata["windows"].append(window)
                    if window["decision"] == "reject":
                        rejected[begin:end] = True
                    else:
                        preserved[max(0, begin - padding):min(output.size, end + padding)] = True
                # A keep window wins every overlap. This prevents an isolated
                # misclassification from removing a spoken word inside BGM.
                masked = rejected & ~preserved
                changes = np.diff(np.pad(masked.astype(np.int8), (1, 1)))
                for begin, end in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
                    if end - begin < round(self.thresholds["minimum_mask_seconds"] * SAMPLE_RATE):
                        masked[begin:end] = False
                output[masked] = 0
                masked_count = int(masked.sum())
                metadata["masked_duration_seconds"] = masked_count / SAMPLE_RATE
                metadata["retained_fraction"] = 1 - masked_count / output.size
                metadata["load_seconds"] = self._load_seconds
                metadata["tagging_seconds"] = sum(window["processing_seconds"] for window in metadata["windows"])
                if masked_count == output.size:
                    metadata.update(action="skip", reason="all_windows_strong_non_speech")
                elif masked_count:
                    metadata.update(action="masked", reason="strong_non_speech_regions_masked")
        metadata["processing_seconds"] = time.monotonic() - started
        return FilterDecision(output, metadata)

    def close(self) -> None:
        with self._lock:
            self._tagger = None
            self._closed = True
