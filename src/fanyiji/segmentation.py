"""Independent Silero VAD segmenters for 16 kHz mono source audio.

No neural model is loaded on import. Inject a VAD-compatible object to exercise
the sample clocks, generation resets and boundaries without audio hardware.
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np

from .audio import AudioChunk, SAMPLE_RATE


@dataclass(frozen=True)
class SegmentationConfig:
    threshold: float = 0.5
    min_silence_seconds: float = 1.1
    min_speech_seconds: float = 0.25
    max_speech_seconds: float = 10.0
    pre_speech_seconds: float = 0.3
    window_size: int = 512
    drafts_enabled: bool = False
    draft_interval_seconds: float = 1.0
    max_cut_hold_seconds: float = 1.8
    max_cut_tail_seconds: float = 0.65
    preferred_speech_seconds: float = 5.5
    preferred_pause_seconds: float = 0.45

    def __post_init__(self) -> None:
        if not 0 < self.threshold < 1:
            raise ValueError("threshold must be between 0 and 1")
        if min(self.min_silence_seconds, self.min_speech_seconds,
               self.max_speech_seconds, self.draft_interval_seconds) <= 0:
            raise ValueError("Segmentation durations must be positive")
        if self.pre_speech_seconds < 0 or self.min_speech_seconds >= self.max_speech_seconds:
            raise ValueError("Invalid speech duration or pre-speech padding")
        if self.window_size != 512:
            raise ValueError("Silero VAD uses 512-sample windows at 16 kHz")
        if (not math.isfinite(self.max_cut_hold_seconds)
                or not 0 <= self.max_cut_hold_seconds <= 3
                or not math.isfinite(self.max_cut_tail_seconds)
                or not 0 < self.max_cut_tail_seconds <= 1):
            raise ValueError("Hard-cut buffering must be bounded to 3 s with tails at most 1 s")
        if (not math.isfinite(self.preferred_speech_seconds)
                or self.preferred_speech_seconds < 0
                or not math.isfinite(self.preferred_pause_seconds)
                or not 0 < self.preferred_pause_seconds <= 3):
            raise ValueError("Preferred speech target and pause duration must be valid and bounded")


@dataclass(frozen=True)
class SegmentEvent:
    channel: str
    kind: Literal["draft", "final", "reset"]
    utterance_id: str
    revision: int
    samples: np.ndarray
    start_sample: int
    end_sample: int
    reason: str
    generation: int
    emitted_at: float


@dataclass(frozen=True)
class SegmentationProgress:
    """Immutable capture-worker snapshot; consumers never access native VAD."""

    generation: int | None
    processed_sample: int
    earliest_future_start_sample: int
    speech_active: bool
    held_start_sample: int | None


class VadLike(Protocol):
    def accept_waveform(self, samples: np.ndarray) -> None: ...
    def is_speech_detected(self) -> bool: ...
    def empty(self) -> bool: ...
    @property
    def front(self) -> Any: ...
    def pop(self) -> None: ...
    def reset(self) -> None: ...
    def flush(self) -> None: ...


def create_silero_vad(model_path: Path, config: SegmentationConfig) -> VadLike:
    if not model_path.is_file():
        raise FileNotFoundError(f"本地 VAD 模型不存在：{model_path}")
    import sherpa_onnx

    model = sherpa_onnx.VadModelConfig()
    model.sample_rate = SAMPLE_RATE
    model.num_threads = 1
    model.provider = "cpu"
    model.silero_vad.model = str(model_path)
    model.silero_vad.threshold = config.threshold
    model.silero_vad.min_silence_duration = config.min_silence_seconds
    model.silero_vad.min_speech_duration = config.min_speech_seconds
    # sherpa shortens its silence threshold to 0.1 s above this native hint.
    # Keep the hint beyond our own boundary so it cannot split a thinking pause
    # before the application finalizes continuous speech.
    model.silero_vad.max_speech_duration = (config.max_speech_seconds
        + config.min_silence_seconds + config.pre_speech_seconds + 1.0)
    model.silero_vad.window_size = config.window_size
    return sherpa_onnx.VoiceActivityDetector(model, buffer_size_in_seconds=
        config.max_speech_seconds + config.min_silence_seconds + config.pre_speech_seconds + 5)


class SpeechSegmenter:
    """Feed from a single source's worker; this object is intentionally not shared."""

    def __init__(self, channel: str, config: SegmentationConfig | None = None,
                 vad_model_path: Path = Path("models/vad/silero_vad.onnx"),
                 vad: VadLike | None = None, pause_vad: VadLike | None = None) -> None:
        self.channel = channel
        self.config = config or SegmentationConfig()
        self._vad_model_path = vad_model_path
        self._vad = vad if vad is not None else create_silero_vad(vad_model_path, self.config)
        self._pause_vad = None
        if (0 < self.config.preferred_speech_seconds < self.config.max_speech_seconds
                and self.config.preferred_pause_seconds < self.config.min_silence_seconds):
            # A separate native detector reports shorter natural pauses without
            # weakening the primary detector's short-answer/thinking-pause rule.
            # Injected test detectors must explicitly inject this one as well.
            if pause_vad is not None:
                self._pause_vad = pause_vad
            elif vad is None:
                self._pause_vad = create_silero_vad(vad_model_path, replace(self.config,
                    min_silence_seconds=self.config.preferred_pause_seconds))
        self._generation: int | None = None
        self._sequence = 0
        self._clear()

    def _clear(self) -> None:
        self._origin = 0
        self._expected: int | None = None
        self._processed = 0
        self._pending = np.empty(0, dtype=np.float32)
        self._history: deque[tuple[int, np.ndarray]] = deque()
        self._history_samples = 0
        self._last_final_end = 0
        self._utterance: str | None = None
        self._revision = 0
        self._last_draft_end = 0
        self._speech_detected_at: int | None = None
        self._held_hard_final: SegmentEvent | None = None
        self._held_hard_deadline = 0
        self._soft_resume_end: int | None = None

    def continuation_snapshot(self) -> SegmentationProgress:
        """Call on the feed owner after enqueueing all returned final segments.

        A not-yet-confirmed onset can precede the processed window by the
        configured minimum speech duration and available pre-speech prefix.
        Round up and allow three extra windows before excluding that onset:
        native Silero v5 reads 576 samples while shifting by 512, and its
        onset can include two native windows plus the internal buffered tail.
        Active speech or a held complete segment must be checked separately.
        """
        window = self.config.window_size
        onset_windows = math.ceil((self.config.min_speech_seconds
            + self.config.pre_speech_seconds) * SAMPLE_RATE / window) + 3
        processed = self._origin + self._processed
        return SegmentationProgress(self._generation, processed,
            max(self._last_final_end, processed - onset_windows * window),
            bool(self._vad.is_speech_detected()),
            self._held_hard_final.start_sample if self._held_hard_final is not None else None)

    def reset(self, reason: str = "reset") -> list[SegmentEvent]:
        old_generation = self._generation
        end = self._expected or 0
        held = self._release_hard_final()
        self._vad.reset()
        if self._pause_vad is not None:
            self._pause_vad.reset()
        self._clear()
        self._generation = None
        if old_generation is None:
            return held
        # Completed audio precedes the discontinuity marker; unfinished native
        # audio is still discarded, and can never cross capture generations.
        return held + [SegmentEvent(self.channel, "reset", "", 0, np.empty(0, dtype=np.float32),
                                    end, end, reason, old_generation, time.monotonic())]

    def reconfigure(self, config: SegmentationConfig) -> None:
        """Switch endpointing modes without publishing the preceding mode's PCM.

        Native VAD parameters are fixed at construction. Keep the segment
        sequence while replacing the detectors so a capture generation cannot
        reuse an utterance id after a settings change.
        """
        if config == self.config:
            return
        vad = create_silero_vad(self._vad_model_path, config)
        pause_vad = None
        if (0 < config.preferred_speech_seconds < config.max_speech_seconds
                and config.preferred_pause_seconds < config.min_silence_seconds):
            pause_vad = create_silero_vad(self._vad_model_path,
                replace(config, min_silence_seconds=config.preferred_pause_seconds))
        # reset may release a held final; intentionally discard it on a mode
        # change rather than submit an old segment to the translation path.
        self.reset("segmentation_mode_changed")
        self.config = config
        self._vad, self._pause_vad = vad, pause_vad

    def feed(self, chunk: AudioChunk) -> list[SegmentEvent]:
        if chunk.channel != self.channel:
            raise ValueError("A segmenter must receive only its own audio channel")
        if chunk.sample_rate != SAMPLE_RATE:
            raise ValueError("SpeechSegmenter expects 16 kHz audio")
        samples = np.asarray(chunk.samples, dtype=np.float32)
        if samples.ndim != 1 or chunk.end_sample - chunk.start_sample != samples.size:
            raise ValueError("Audio chunk must be mono with matching sample counters")
        if chunk.start_sample < 0 or chunk.end_sample < chunk.start_sample:
            raise ValueError("Invalid audio sample counters")
        if not np.isfinite(samples).all():
            raise ValueError("Audio chunk contains non-finite samples")
        events: list[SegmentEvent] = []
        if self._generation is not None:
            if self._generation != chunk.generation:
                events.extend(self.reset("capture_restarted"))
            elif self._expected != chunk.start_sample:
                events.extend(self.reset("audio_gap"))
        if self._generation is None:
            self._generation = chunk.generation
            self._origin = chunk.start_sample
            self._last_final_end = chunk.start_sample
        self._expected = chunk.end_sample
        if not samples.size:
            return events
        owned = samples.copy()
        self._history.append((chunk.start_sample, owned))
        self._history_samples += owned.size
        self._pending = np.concatenate((self._pending, owned))
        window = self.config.window_size
        while self._pending.size >= window:
            block, self._pending = self._pending[:window], self._pending[window:]
            self._vad.accept_waveform(block)
            if self._pause_vad is not None:
                self._pause_vad.accept_waveform(block)
            self._processed += window
            finals = self._collect_finals()
            finals.extend(self._finish_preferred_pause())
            finals.extend(self._limit_continuous_speech())
            events.extend(self._buffer_hard_finals(finals, self._origin + self._processed))
            if self.config.drafts_enabled:
                events.extend(self._draft())
        self._trim_history()
        return events

    def flush(self) -> list[SegmentEvent]:
        """Finalize the current source tail; callers may instead reset on cancel."""
        if self._generation is None:
            return []
        if self._pending.size:
            # Native VAD keeps a sub-window tail in an internal buffer that
            # flush() does not process. Pad once, then clip events to the real
            # source end so the last <= 31 ms is retained without fake audio.
            block = np.pad(self._pending, (0, self.config.window_size - self._pending.size))
            self._vad.accept_waveform(block)
            self._processed += block.size
            self._pending = np.empty(0, dtype=np.float32)
        self._vad.flush()
        events = self._buffer_hard_finals(self._collect_finals("flush"),
                                         self._expected or 0, release_all=True)
        # A public tail flush may be followed by more feed calls. Restart its
        # clock at the real end rather than the temporarily padded native end.
        self._vad.reset()
        if self._pause_vad is not None:
            self._pause_vad.reset()
        self._origin = self._expected or 0
        self._processed = 0
        self._utterance = None
        self._speech_detected_at = None
        self._soft_resume_end = None
        return events

    def _release_hard_final(self) -> list[SegmentEvent]:
        held = self._held_hard_final
        self._held_hard_final = None
        self._held_hard_deadline = 0
        return [replace(held, emitted_at=time.monotonic())] if held is not None else []

    def _buffer_hard_finals(self, finals: list[SegmentEvent], boundary: int,
                            *, release_all: bool = False) -> list[SegmentEvent]:
        """Hold a hard cut once, on the audio clock, for a tiny adjacent tail.

        Only PCM already copied by the native detector is joined. A long tail,
        a real gap, another hard cut, or expiry releases the first final rather
        than extending its deadline or withholding ongoing speech indefinitely.
        """
        events: list[SegmentEvent] = []
        if self._held_hard_final is not None and boundary > self._held_hard_deadline:
            events.extend(self._release_hard_final())
        tail_limit = round(self.config.max_cut_tail_seconds * SAMPLE_RATE)
        for final in finals:
            held = self._held_hard_final
            if held is not None:
                if (final.reason in ("silence", "flush")
                        and final.channel == held.channel
                        and final.generation == held.generation
                        and final.start_sample == held.end_sample
                        and final.samples.size <= tail_limit
                        and boundary <= self._held_hard_deadline):
                    events.append(SegmentEvent(held.channel, "final", held.utterance_id,
                        held.revision + 1, np.concatenate((held.samples, final.samples)),
                        held.start_sample, final.end_sample, "max_duration_tail_join",
                        held.generation, time.monotonic()))
                    self._held_hard_final = None
                    self._held_hard_deadline = 0
                    continue
                events.extend(self._release_hard_final())
            if final.reason == "max_duration" and self.config.max_cut_hold_seconds:
                self._held_hard_final = final
                # Feed advances in fixed VAD windows; expiry permits only that
                # single-window rounding, never another renewed waiting period.
                hold = math.ceil(self.config.max_cut_hold_seconds * SAMPLE_RATE
                                 / self.config.window_size) * self.config.window_size
                self._held_hard_deadline = boundary + hold
            else:
                events.append(final)
        if release_all or (self._held_hard_final is not None
                           and boundary >= self._held_hard_deadline):
            events.extend(self._release_hard_final())
        return events

    def _limit_continuous_speech(self) -> list[SegmentEvent]:
        # sherpa's max_speech_duration only raises its threshold; persistent
        # high-scoring speech still needs an application-enforced boundary.
        boundary = self._origin + self._processed
        if not self._vad.is_speech_detected():
            self._speech_detected_at = None
            return []
        if self._speech_detected_at is None:
            self._speech_detected_at = boundary
        maximum = round(self.config.max_speech_seconds * SAMPLE_RATE)
        if boundary - self._speech_detected_at < maximum:
            return []
        return self._cut_speech("max_duration", boundary)

    def _finish_preferred_pause(self) -> list[SegmentEvent]:
        if self._pause_vad is None:
            return []
        candidates: list[int] = []
        while not self._pause_vad.empty():
            segment = self._pause_vad.front
            # Copy scalar geometry before pop invalidates the native front.
            end = self._origin + int(segment.start) + len(segment.samples)
            self._pause_vad.pop()
            candidates.append(end)
        if not candidates or self._speech_detected_at is None or not self._vad.is_speech_detected():
            return []
        target = round(self.config.preferred_speech_seconds * SAMPLE_RATE)
        eligible = [end for end in candidates
                    if end > self._last_final_end and end - self._speech_detected_at >= target]
        if not eligible:
            return []
        boundary = self._origin + self._processed
        # End on the detector's completed voice span, not the later source
        # frame which may already contain the next sentence's first syllable.
        cut = min(max(eligible), boundary)
        events = self._cut_speech("natural_pause", boundary, end_sample=cut)
        if events:
            self._soft_resume_end = events[-1].end_sample
        return events

    def _cut_speech(self, reason: str, boundary: int,
                    *, end_sample: int | None = None) -> list[SegmentEvent]:
        self._vad.flush()
        events = self._collect_finals(reason, end_sample=end_sample)
        # A naked flush retains the native silence counter: when speech stops
        # near a cut, its computed end can precede the new segment's start.
        # Reset, then warm detection with a small source prefix. Absolute sample
        # clocks exclude this already-emitted prefix from subsequent finals;
        # neither audio output duplication nor text-based deduplication is used.
        window = self.config.window_size
        warm_seconds = max(self.config.pre_speech_seconds,
                           self.config.min_speech_seconds + self.config.min_silence_seconds
                           + 2 * window / SAMPLE_RATE)
        warm_length = int(np.ceil(warm_seconds * SAMPLE_RATE / window)) * window
        warm = self._slice(max(self._origin, boundary - warm_length), boundary)
        warm_windows = warm.size // window
        warm = warm[-warm_windows * window:] if warm_windows else np.empty(0, dtype=np.float32)
        self._vad.reset()
        if self._pause_vad is not None:
            self._pause_vad.reset()
        self._origin = boundary - warm.size
        self._processed = 0
        for offset in range(0, warm.size, window):
            self._vad.accept_waveform(warm[offset:offset + window])
            if self._pause_vad is not None:
                self._pause_vad.accept_waveform(warm[offset:offset + window])
            self._processed += window
        events.extend(self._collect_finals())
        if self._pause_vad is not None:
            while not self._pause_vad.empty():
                self._pause_vad.pop()  # A warmed already-emitted pause is not a new boundary.
        self._speech_detected_at = boundary if self._vad.is_speech_detected() else None
        return events

    def _new_utterance(self) -> str:
        self._sequence += 1
        self._revision = 0
        return f"g{self._generation}-u{self._sequence}"

    def _collect_finals(self, reason: str | None = None,
                        *, end_sample: int | None = None) -> list[SegmentEvent]:
        events: list[SegmentEvent] = []
        pad = round(self.config.pre_speech_seconds * SAMPLE_RATE)
        while not self._vad.empty():
            # front is invalidated by ANY detector method: copy before pop().
            segment = self._vad.front
            raw_start = self._origin + int(segment.start)
            raw = np.asarray(segment.samples, dtype=np.float32).copy()
            self._vad.pop()
            if not raw.size:
                continue
            raw_end = min(raw_start + raw.size, self._expected or 0)
            if end_sample is not None:
                raw_end = min(raw_end, end_sample)
            speech_start = max(self._last_final_end, raw_start, self._origin)
            if speech_start >= raw_end:
                continue
            # Native raw preserves the complete utterance even if our bounded
            # history has evicted its beginning. History supplies only an
            # available contiguous prefix, including reset detection latency.
            prefix_start = max(self._last_final_end, raw_start - pad, self._origin)
            if (self._soft_resume_end == self._last_final_end
                    and 0 <= speech_start - self._last_final_end <= round(
                        (self.config.min_silence_seconds + self.config.pre_speech_seconds) * SAMPLE_RATE)):
                # Keep the captured pause/next onset that arrived while the
                # shorter VAD was confirming the previous natural boundary.
                prefix_start = max(self._last_final_end, self._origin)
            prefix = self._slice(prefix_start, speech_start)
            speech = raw[speech_start - raw_start:raw_end - raw_start]
            audio = np.concatenate((prefix, speech)).astype(np.float32, copy=False)
            start = speech_start - prefix.size
            if not audio.size:
                continue
            utterance = self._utterance or self._new_utterance()
            self._revision += 1
            # Length alone cannot tell a natural end from an application cut.
            ended_by = reason or "silence"
            events.append(SegmentEvent(self.channel, "final", utterance, self._revision,
                audio, start, raw_end, ended_by, self._generation or 0, time.monotonic()))
            self._last_final_end = raw_end
            self._utterance = None
            self._revision = 0
            self._last_draft_end = raw_end
            self._soft_resume_end = None
        return events

    def _draft(self) -> list[SegmentEvent]:
        if not self._vad.is_speech_detected():
            return []
        current = getattr(self._vad, "current_segment", None)
        if current is None:
            return []
        raw_start = self._origin + int(current.start)
        end = min(self._origin + self._processed, self._expected or 0)
        if end - self._last_draft_end < round(self.config.draft_interval_seconds * SAMPLE_RATE):
            return []
        start = max(raw_start - round(self.config.pre_speech_seconds * SAMPLE_RATE),
                    self._last_final_end, self._origin)
        start = max(start, end - round(self.config.max_speech_seconds * SAMPLE_RATE))
        audio = self._slice(start, end)
        if not audio.size:
            return []
        if self._utterance is None:
            self._utterance = self._new_utterance()
        self._revision += 1
        self._last_draft_end = end
        return [SegmentEvent(self.channel, "draft", self._utterance, self._revision,
                             audio, start, end, "partial", self._generation or 0, time.monotonic())]

    def _slice(self, start: int, end: int) -> np.ndarray:
        pieces = [audio[max(0, start - offset):min(audio.size, end - offset)]
                  for offset, audio in self._history if offset < end and offset + audio.size > start]
        return np.concatenate(pieces).astype(np.float32, copy=False) if pieces else np.empty(0, dtype=np.float32)

    def _trim_history(self) -> None:
        limit = round((self.config.max_speech_seconds + self.config.min_silence_seconds
                       + self.config.pre_speech_seconds + 5) * SAMPLE_RATE)
        while self._history and self._history_samples - self._history[0][1].size > limit:
            self._history_samples -= self._history.popleft()[1].size
