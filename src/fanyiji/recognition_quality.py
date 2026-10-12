"""Bounded ASR rechecks without changing sample rate or assuming consensus is truth.

WSOLA edits overlapping waveform frames at their original sample rate. Its
similarity search preserves local periods rather than lowering pitch through
resampling. Algorithm reference (not a claim of ASR accuracy improvement):
https://www.isca-archive.org/eurospeech_1993/roelands93_eurospeech.html
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
import time
from typing import Callable, Any

import numpy as np

from .engines import SAMPLE_RATE
from .language_policy import AsrResult, normalize_detected_language, validate_asr_language
from .asr_protocol import require_subtitle_text


_REPEATED = re.compile(r"([^\s，。！？、,.!?]{1,8})\1{4,}")
_TEXT_SYMBOLS = re.compile(r"[\s，。！？、,.!?；;：:「」『』（）()\[\]…ー]")
_GENERATION_LIMIT = re.compile(r"Qwen.*(?:token.*(?:上限|limit)|(?:上限|limit).*token)", re.IGNORECASE)
_LATIN_WORDS = re.compile(r"[A-Za-z\u00c0-\u024f]+(?:['’\-][A-Za-z\u00c0-\u024f]+)*")
_KANA = re.compile(r"[\u3040-\u30ff]")


class RecognitionCancelled(RuntimeError):
    """Cancellation is checked around inference; native decode is not interrupted."""


@dataclass(frozen=True)
class QualityRecognition:
    text: str
    metadata: dict[str, Any]


def _check_cancel(cancelled: Callable[[], bool] | None) -> None:
    if cancelled is not None and cancelled():
        raise RecognitionCancelled("识别复核已取消")


def _validate(samples: np.ndarray) -> None:
    if not isinstance(samples, np.ndarray):
        raise TypeError("samples 必须是 numpy float32 数组")
    if samples.ndim != 1 or samples.dtype != np.float32 or not np.isfinite(samples).all():
        raise ValueError("samples 必须是有限值的一维 16 kHz float32 数组")


def time_stretch_wsola(samples: np.ndarray, speed: float = .92, *,
                       cancelled: Callable[[], bool] | None = None) -> np.ndarray:
    """Return mild pitch-preserving slowdown; 0.92 means output length / 0.92.

    The 40 ms frames keep their original samples. Normalized correlation selects
    joins within 10 ms of the intended analysis position; overlapping frames use
    complementary linear fades. Ends are kept exactly. Very short audio has no
    reliable overlapping frame, so callers should preserve it without slowdown.
    """
    _validate(samples)
    if isinstance(speed, bool) or not math.isfinite(speed) or not .85 <= speed <= 1.0:
        raise ValueError("WSOLA speed 必须在 0.85 到 1.0 之间")
    _check_cancel(cancelled)
    if not samples.size or speed == 1.0:
        return samples.copy()
    frame = SAMPLE_RATE * 40 // 1000
    hop = frame // 2
    search = SAMPLE_RATE * 10 // 1000
    if samples.size < frame * 2:
        raise ValueError("WSOLA 至少需要 80 ms 音频；短音频应保留原速")
    target_length = round(samples.size / speed)
    positions = list(range(0, target_length - frame + 1, hop))
    if positions[-1] != target_length - frame:
        # Replace the short final hop: the final untouched half-frame must not
        # be blended with a nearly coincident preceding analysis window.
        positions.pop()
        positions.append(target_length - frame)
    output = np.zeros(target_length, dtype=np.float64)
    weights = np.zeros(target_length, dtype=np.float64)
    previous_start = previous_position = 0
    source = np.ascontiguousarray(samples)
    for index, position in enumerate(positions):
        _check_cancel(cancelled)
        if index == 0:
            start = 0
        elif index == len(positions) - 1:
            start = source.size - frame
        else:
            delta = position - previous_position
            overlap = frame - delta
            intended = round(position * speed)
            low = max(previous_start + 1, intended - search)
            high = min(source.size - frame, intended + search)
            if low > high:
                start = min(max(intended, previous_start), source.size - frame)
            else:
                reference = source[previous_start + delta:previous_start + delta + overlap].astype(np.float64)
                reference -= reference.mean()
                reference_energy = float(np.dot(reference, reference))
                if reference_energy < 1e-12:
                    start = min(max(intended, low), high)
                else:
                    region = source[low:high + overlap].astype(np.float64)
                    dot = np.correlate(region, reference, mode="valid")
                    sums = np.r_[0., np.cumsum(region)]
                    squares = np.r_[0., np.cumsum(region * region)]
                    means = sums[overlap:] - sums[:-overlap]
                    energy = squares[overlap:] - squares[:-overlap] - means * means / overlap
                    scores = dot / np.sqrt(reference_energy * np.maximum(energy, 1e-12))
                    # Favor the intended position for equally good periodic joins.
                    offsets = np.arange(low, high + 1)
                    scores -= np.abs(offsets - intended) * 1e-7
                    start = low + int(np.argmax(scores))
        fade = np.ones(frame, dtype=np.float64)
        if index:
            overlap = frame - (position - previous_position)
            fade[:overlap] = np.linspace(0., 1., overlap, endpoint=False)
        if index + 1 < len(positions):
            overlap = frame - (positions[index + 1] - position)
            fade[-overlap:] = np.linspace(1., 0., overlap, endpoint=False)
        output[position:position + frame] += source[start:start + frame] * fade
        weights[position:position + frame] += fade
        previous_start, previous_position = start, position
    np.divide(output, weights, out=output, where=weights > 0)
    return output.astype(np.float32)


def _density_parts(text: str) -> tuple[int, int]:
    # Latin spelling length is not comparable to a Han/Kana character count.
    # Keep word spaces when counting, before compacting other script characters.
    return len(_TEXT_SYMBOLS.sub("", _LATIN_WORDS.sub("", text))), len(_LATIN_WORDS.findall(text))


def _suspicions(text: str, seconds: float, language: str, rms: float) -> list[str]:
    compact = _TEXT_SYMBOLS.sub("", text)
    reasons = []
    if not compact:
        if seconds >= .4 and rms >= 1e-4:
            reasons.append("empty_with_audio")
        return reasons
    if "\ufffd" in text:
        reasons.append("unicode_replacement")
    if len(compact) >= 16 and _REPEATED.search(compact):
        reasons.append("repetitive_output")
    # The sum is a conservative text anomaly trigger, not a speaking-rate
    # estimate or language detector. Latin words use their own budget so normal
    # English and long game names do not trigger a CJK character threshold.
    non_latin, words = _density_parts(text)
    japanese_script = language == "ja" or (language in {"auto", "en"} and bool(_KANA.search(text)))
    long_density = non_latin / (12.0 if japanese_script else 9.0) + words / 8.0
    if seconds >= 1.5 and long_density / seconds > 1.0:
        reasons.append("high_text_density")
    # Hard-cut tails can contain only a few milliseconds of speech while the
    # decoder emits a whole sentence. Higher short-window thresholds and a
    # minimum length preserve brief answers instead of treating them as errors.
    short_density = non_latin / (32.0 if japanese_script else 24.0) + words / 16.0
    if 0 < seconds < 1.5 and (non_latin >= 8 or words >= 8) and short_density / seconds > 1.0:
        reasons.append("short_audio_text_density")
    if seconds >= 4.0 and len(compact) <= 2 and rms >= 1e-4:
        reasons.append("low_text_coverage")
    return reasons


def _normalized(text: str) -> str:
    return _TEXT_SYMBOLS.sub("", text).casefold()


def _limit_error(exc: Exception) -> bool:
    return bool(getattr(exc, "generation_limited", False) or _GENERATION_LIMIT.search(str(exc)))


def _split_boundaries(samples: np.ndarray, count: int) -> list[int]:
    """Partition exact original samples near low-energy joins, with no overlap."""
    size = samples.size
    minimum = round(.4 * SAMPLE_RATE)
    radius = round(.2 * SAMPLE_RATE)
    window = round(.02 * SAMPLE_RATE)
    boundaries = [0]
    power = np.r_[0., np.cumsum(samples.astype(np.float64) ** 2)]
    for part in range(1, count):
        intended = round(size * part / count)
        low = max(boundaries[-1] + minimum, intended - radius)
        high = min(size - (count - part) * minimum, intended + radius)
        centers = np.arange(low, high + 1, 80, dtype=np.int64)
        left = np.maximum(0, centers - window // 2)
        right = np.minimum(size, centers + window // 2)
        energy = (power[right] - power[left]) / np.maximum(1, right - left)
        # Quiet joins win; a tiny distance penalty makes silence deterministic.
        score = energy + np.abs(centers - intended) / max(1, size) * 1e-9
        boundaries.append(int(centers[int(np.argmin(score))]))
    return boundaries + [size]


class QualityRecognizer:
    """Wrap an injected recognizer; maximum 2 normal or 4 capped decode calls.

    Only output anomalies trigger a 0.92 speed recheck. Agreement is recorded
    as agreement, never confidence or evidence that the transcript is correct.
    Disagreement retains the original except empty/undecodable original text.
    Generation-limit recovery uses up to three shorter, non-overlapping parts;
    it never recursively retries a failed part or performs a speed retry too.
    """
    def __init__(self, recognizer: Any, *, cancel_event: Any = None,
                 speed: float = .92, clock: Callable[[], float] = time.perf_counter):
        if isinstance(speed, bool) or not math.isfinite(speed) or not .85 <= speed < 1.0:
            raise ValueError("复核速度必须在 0.85 到 1.0 之间")
        self.recognizer = recognizer
        self.cancel_event = cancel_event
        self.speed = speed
        self.clock = clock

    def _cancelled(self) -> bool:
        return self.cancel_event is not None and self.cancel_event.is_set()

    def _decode(self, samples: np.ndarray, language: str, metadata: dict,
                *, speed: float, purpose: str, start: int = 0, end: int | None = None) -> AsrResult:
        _check_cancel(self._cancelled)
        began = self.clock()
        attempt = {"purpose": purpose, "speed": speed, "sample_rate": SAMPLE_RATE,
                   "input_samples": int(samples.size), "input_seconds": samples.size / SAMPLE_RATE,
                   "source_start_sample": start, "source_end_sample": end,
                   "text": None, "error": None, "language_policy": language,
                   "detected_language": None, "detection_source": "unknown"}
        metadata["attempts"].append(attempt)
        try:
            # A Mock may manufacture arbitrary attributes on demand. Only use
            # the rich API when the actual recognizer type defines that method.
            detailed = getattr(type(self.recognizer), "recognize_detailed", None)
            if callable(detailed):
                result = self.recognizer.recognize_detailed(samples.copy(), language)
                if not isinstance(result, AsrResult):
                    raise TypeError("recognizer.recognize_detailed 必须返回 AsrResult")
                if not isinstance(result.text, str):
                    raise TypeError("AsrResult.text 必须是 str")
            else:
                text = self.recognizer.recognize(samples.copy(), language)
                if not isinstance(text, str):
                    raise TypeError("recognizer.recognize 必须返回 str")
                result = AsrResult(text)
            attempt["text"] = result.text
            require_subtitle_text(result.text)
            attempt["detected_language"] = normalize_detected_language(result.detected_language)
            attempt["detection_source"] = result.detection_source if attempt["detected_language"] else "unknown"
            _check_cancel(self._cancelled)
            return result
        except Exception as exc:
            attempt["error"] = {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            attempt["elapsed_seconds"] = max(0., self.clock() - began)

    def recognize(self, samples: np.ndarray, language: str) -> QualityRecognition:
        _validate(samples)
        language = validate_asr_language(language)
        _check_cancel(self._cancelled)
        began = self.clock()
        seconds = samples.size / SAMPLE_RATE
        rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2))) if samples.size else 0.
        metadata = {"policy": "bounded_quality_recheck_v2", "source_seconds": seconds,
                    "source_sample_count": int(samples.size), "source_rms": rms,
                    "original_text": None, "review_text": None, "selected": "original",
                    "suspicions": [], "attempts": [], "status": "original",
                    "uncertain": False, "disagreement": False,
                    "text_density_characters_per_second": None,
                    "text_density_non_latin_characters_per_second": None,
                    "text_density_words_per_second": None,
                    "text_density_policy": "script_and_word_count",
                    "text_density_is_speaking_rate": False,
                    "language_policy": language,
                    "detected_language": None, "detection_source": "unknown",
                    "speed_adjustment": None, "elapsed_seconds": None}
        try:
            if not samples.size:
                metadata["original_text"] = ""
                metadata["status"] = "empty_input"
                return QualityRecognition("", metadata)
            try:
                original_result = self._decode(samples, language, metadata, speed=1.,
                                               purpose="original", end=samples.size)
            except RecognitionCancelled:
                raise
            except Exception as exc:
                if not _limit_error(exc):
                    raise
                metadata["suspicions"] = ["generation_limit"]
                return self._recover_limit(samples, language, metadata)
            original = original_result.text
            self._select_detection(original_result, metadata)
            metadata["original_text"] = original
            metadata["text_density_characters_per_second"] = len(_normalized(original)) / seconds
            non_latin, words = _density_parts(original)
            metadata["text_density_non_latin_characters_per_second"] = non_latin / seconds
            metadata["text_density_words_per_second"] = words / seconds
            reasons = _suspicions(original, seconds, metadata["detected_language"] or language, rms)
            metadata["suspicions"] = reasons
            if not reasons:
                return QualityRecognition(original, metadata)
            metadata["uncertain"] = True
            if seconds < .4:
                metadata["status"] = "short_suspicious_original_preserved"
                return QualityRecognition(original, metadata)
            stretch_started = self.clock()
            slower = time_stretch_wsola(samples, self.speed, cancelled=self._cancelled)
            metadata["speed_adjustment"] = {"method": "WSOLA", "speed": self.speed,
                    "sample_rate_unchanged": True, "output_sample_count": int(slower.size),
                    "elapsed_seconds": max(0., self.clock() - stretch_started)}
            try:
                review_result = self._decode(slower, language, metadata, speed=self.speed, purpose="speed_review",
                                            end=samples.size)
            except RecognitionCancelled:
                raise
            except Exception:
                metadata["status"] = "review_failed_original_preserved"
                return QualityRecognition(original, metadata)
            review = review_result.text
            metadata["review_text"] = review
            review_language = normalize_detected_language(review_result.detected_language) or language
            metadata["review_suspicions"] = _suspicions(review, seconds, review_language, rms)
            agrees = _normalized(original) == _normalized(review)
            metadata["disagreement"] = not agrees
            if agrees:
                metadata["status"] = "review_agrees_unverified"
                return QualityRecognition(original, metadata)
            if (not _normalized(original) or "unicode_replacement" in reasons) and review.strip() and not metadata["review_suspicions"]:
                metadata["selected"] = "speed_review"
                metadata["status"] = "review_recovered_unverified"
                self._select_detection(review_result, metadata)
                return QualityRecognition(review, metadata)
            metadata["status"] = "review_disagrees_original_preserved"
            return QualityRecognition(original, metadata)
        finally:
            metadata["elapsed_seconds"] = max(0., self.clock() - began)

    @staticmethod
    def _select_detection(result: AsrResult, metadata: dict) -> None:
        detected = normalize_detected_language(result.detected_language)
        metadata["detected_language"] = detected
        metadata["detection_source"] = result.detection_source if detected else "unknown"

    def _recover_limit(self, samples: np.ndarray, language: str, metadata: dict) -> QualityRecognition:
        metadata["uncertain"] = True
        count = max(2, math.ceil(samples.size / (4.5 * SAMPLE_RATE)))
        if count > 3 or samples.size < 2 * round(.4 * SAMPLE_RATE):
            metadata["status"] = "generation_limit_recovery_outside_budget"
            metadata["selected"] = "none"
            return QualityRecognition("", metadata)
        boundaries = _split_boundaries(samples, count)
        metadata["split_boundaries_samples"] = boundaries
        metadata["split_overlap_samples"] = 0
        pieces = []
        results = []
        for start, end in zip(boundaries, boundaries[1:]):
            try:
                result = self._decode(samples[start:end], language, metadata, speed=1.,
                                      purpose="generation_limit_part", start=start, end=end)
            except RecognitionCancelled:
                raise
            except Exception:
                metadata["status"] = "generation_limit_part_failed"
                metadata["selected"] = "none"
                return QualityRecognition("", metadata)
            piece = result.text
            pieces.append(piece)
            results.append(result)
            if not piece.strip():
                metadata["status"] = "generation_limit_part_empty"
                metadata["selected"] = "none"
                return QualityRecognition("", metadata)
            if _suspicions(piece, (end - start) / SAMPLE_RATE, normalize_detected_language(result.detected_language) or language,
                           float(np.sqrt(np.mean(samples[start:end].astype(np.float64) ** 2)))):
                metadata["status"] = "generation_limit_part_suspicious"
                metadata["selected"] = "none"
                return QualityRecognition("", metadata)
        metadata["review_text"] = " ".join(piece.strip() for piece in pieces)
        metadata["selected"] = "split_recovery"
        metadata["status"] = "generation_limit_recovered_unverified"
        detected = {normalize_detected_language(result.detected_language) for result in results}
        if len(detected) == 1 and None not in detected:
            metadata["detected_language"] = detected.pop()
            sources = {result.detection_source for result in results}
            metadata["detection_source"] = sources.pop() if len(sources) == 1 else "multiple_results"
        elif None not in detected:
            metadata["detection_source"] = "mixed_parts"
        return QualityRecognition(metadata["review_text"], metadata)
