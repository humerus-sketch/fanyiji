"""Text-free latency measurements ending at an accepted subtitle's Qt paint."""
from __future__ import annotations

from collections import OrderedDict, deque
import math
import statistics
from typing import Any


# These clocks are all time.monotonic() in the application process. Audio clocks
# are reconstructed from the captured sample positions, never from wall time.
_STAGES = {
    "audio_seconds": ("audio_started_at", "audio_ended_at"),
    "segmentation_seconds": ("audio_ended_at", "segment_ready_at"),
    "asr_queue_seconds": ("segment_ready_at", "recognition_dequeued_at"),
    "speech_filter_seconds": ("filter_started_at", "filter_finished_at"),
    "audio_processing_seconds": ("audio_processing_started_at", "audio_processing_finished_at"),
    "asr_seconds": ("asr_started_at", "asr_finished_at"),
    "assembly_seconds": ("asr_finished_at", "translation_queued_at"),
    "translation_queue_seconds": ("translation_queued_at", "translation_started_at"),
    "translation_seconds": ("translation_started_at", "translation_finished_at"),
    "ui_dispatch_seconds": ("translation_finished_at", "ui_received_at"),
    "ui_update_seconds": ("ui_received_at", "ui_updated_at"),
    "ui_paint_seconds": ("ui_updated_at", "ui_painted_at"),
}
_CLOCK_ORDER = (
    "audio_started_at", "audio_ended_at", "segment_ready_at",
    "recognition_dequeued_at", "filter_started_at", "filter_finished_at",
    "asr_started_at", "asr_finished_at", "translation_queued_at",
    "translation_started_at", "translation_finished_at", "ui_received_at",
    "ui_updated_at", "ui_painted_at",
)
_RPC_KEYS = frozenset({
    "auth_seconds", "thread_seconds", "turn_start_seconds", "receive_seconds",
    "unsubscribe_seconds", "total_seconds", "setup_seconds", "lock_wait_seconds",
    "first_output_seconds", "output_completion_seconds", "thread_reused", "thread_turn",
    "thread_history_bytes", "thread_reset",
})
_OPTIONAL_CLOCKS = ("audio_processing_started_at", "audio_processing_finished_at")


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def _duration(clocks: dict[str, float], start: str, end: str) -> float | None:
    if start not in clocks or end not in clocks:
        return None
    duration = clocks[end] - clocks[start]
    return round(duration, 6) if duration >= 0 else None


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[max(0, math.ceil(len(ordered) * fraction) - 1)], 6)


class LatencyTracker:
    """Bounded session statistics; no source text, translations or credentials."""

    def __init__(self, window: int = 200):
        if type(window) is not int or window < 1:
            raise ValueError("Latency window must be a positive integer")
        self.window = window
        self.session: int | None = None
        self._records: deque[dict] = deque(maxlen=window)
        self._seen: OrderedDict[tuple, None] = OrderedDict()
        self._drafts: OrderedDict[tuple, float] = OrderedDict()

    def begin_session(self, epoch: int) -> None:
        self.session = epoch
        self._records.clear()
        self._seen.clear()
        self._drafts.clear()

    def on_presented(self, event: dict) -> dict | None:
        if self.session is None or event.get("session") != self.session:
            return None
        stage = event.get("translation_stage", event.get("stage", "final"))
        if event.get("failed") or (event.get("provisional") and stage == "final"):
            return None
        if stage not in ("draft", "final") or not event.get("segment_id"):
            return None
        source_key = (event.get("channel"), event["segment_id"],
                      event.get("revision", 1), event.get("context_generation"))
        key = (*source_key, stage)
        if key in self._seen:
            return None
        timing = event.get("timing")
        timing = timing if isinstance(timing, dict) else {}
        clocks = {name: value for name in (*_CLOCK_ORDER, *_OPTIONAL_CLOCKS)
                  if (value := _number(event.get(name, timing.get(name)))) is not None}
        missing = [name for name in _CLOCK_ORDER if name not in clocks]
        source_count = timing.get("source_segment_count", 1)
        source_count = source_count if type(source_count) is int and source_count >= 1 else 1
        if source_count == 1:
            chains = (_CLOCK_ORDER,)
        else:
            # A revision may join audio captured while an older fragment was
            # already being recognized. Its ASR envelope overlaps that audio;
            # calling the whole envelope inference time would overstate work.
            chains = (("audio_started_at", "audio_ended_at", "segment_ready_at"),
                      ("recognition_dequeued_at", "filter_started_at", "filter_finished_at",
                       "asr_started_at", "asr_finished_at"),
                      ("asr_finished_at", "translation_queued_at", "translation_started_at",
                       "translation_finished_at", "ui_received_at", "ui_updated_at", "ui_painted_at"))
        ordered = all(all(a <= b for a, b in zip(values, values[1:]))
                      for chain in chains
                      if (values := [clocks[name] for name in chain if name in clocks]))
        if any(key in clocks for key in _OPTIONAL_CLOCKS):
            processing_chain = [clocks.get(key) for key in (
                "filter_finished_at", *_OPTIONAL_CLOCKS, "asr_started_at")]
            ordered = ordered and all(value is not None for value in processing_chain)
            if ordered and source_count == 1:
                ordered = all(a <= b for a, b in zip(processing_chain, processing_chain[1:]))
        metrics = {name: event.get(name) for name in (
            "session", "channel", "segment_id", "revision", "context_generation",
            "translation_provider", "translation_model",
        )}
        metrics.update(stage=stage, measurement="audio_samples_to_qt_paint",
            source_segment_count=source_count,
            breakdown_mode="sequential" if source_count == 1 else "overlapping_joined_audio",
            valid=not missing and ordered, missing_fields=missing,
            clock_order_valid=ordered, timing=clocks,
            end_to_end_seconds=_duration(clocks, "audio_started_at", "ui_painted_at"),
            post_audio_seconds=_duration(clocks, "audio_ended_at", "ui_painted_at"),
            recognition_to_display_seconds=_duration(clocks, "asr_started_at", "ui_painted_at"),
            translation_envelope_seconds=_duration(clocks, "translation_started_at", "translation_finished_at"),
            breakdown={name: _duration(clocks, *bounds) for name, bounds in _STAGES.items()})
        if not any(key in clocks for key in _OPTIONAL_CLOCKS):
            metrics["breakdown"]["audio_processing_seconds"] = 0.0
        dispatch = (_duration(clocks, "recognition_dequeued_at", "filter_started_at"),)
        if all(key in clocks for key in _OPTIONAL_CLOCKS):
            dispatch += (_duration(clocks, "filter_finished_at", "audio_processing_started_at"),
                         _duration(clocks, "audio_processing_finished_at", "asr_started_at"))
        else:
            dispatch += (_duration(clocks, "filter_finished_at", "asr_started_at"),)
        metrics["breakdown"]["recognition_dispatch_seconds"] = (
            round(sum(dispatch), 6) if all(value is not None for value in dispatch) else None)
        for name in ("segmentation_seconds", "asr_queue_seconds", "speech_filter_seconds", "audio_processing_seconds",
                     "asr_seconds", "translation_seconds", "recognition_dispatch_seconds"):
            if (value := _number(timing.get(name))) is not None and value >= 0:
                metrics["breakdown"][name] = value
        if (joint := _number(timing.get("joint_asr_seconds"))) is not None and joint >= 0:
            if metrics["breakdown"]["asr_seconds"] is not None:
                metrics["breakdown"]["asr_seconds"] = round(metrics["breakdown"]["asr_seconds"] + joint, 6)
        confirmation_wait = _number(event.get("source_confirmation_wait_seconds", 0))
        if confirmation_wait is not None and confirmation_wait >= 0:
            paint_wait = metrics["breakdown"]["ui_paint_seconds"]
            if paint_wait is not None and confirmation_wait <= paint_wait:
                metrics["breakdown"]["source_confirmation_seconds"] = confirmation_wait
                metrics["breakdown"]["ui_paint_seconds"] = round(paint_wait - confirmation_wait, 6)
                if (confirmed_at := _number(event.get("source_confirmed_at"))) is not None:
                    metrics["source_confirmed_at"] = confirmed_at
            else:
                metrics["valid"] = False
        else:
            metrics["valid"] = False
        for name in ("joint_asr_seconds", "assembly_processing_seconds"):
            if (value := _number(timing.get(name))) is not None and value >= 0:
                metrics[name] = value
        rpc = event.get("rpc_metrics")
        if isinstance(rpc, dict):
            metrics["rpc_metrics"] = {name: value for name in _RPC_KEYS
                                      if (value := _number(rpc.get(name))) is not None and value >= 0}
        # Partial timings may be inspected, but never mixed into trustworthy P95.
        if metrics["valid"]:
            self._seen[key] = None
            if stage == "draft":
                self._drafts.setdefault(source_key, metrics["end_to_end_seconds"])
            metrics["draft_end_to_end_seconds"] = self._drafts.get(source_key)
            self._records.append(metrics)
            for bounded in (self._seen, self._drafts):
                while len(bounded) > self.window * 4:
                    bounded.popitem(last=False)
        else:
            metrics["draft_end_to_end_seconds"] = None
        return metrics

    def summary(self) -> dict:
        finals = [record for record in self._records if record["stage"] == "final"]
        drafts = [record for record in self._records if record["stage"] == "draft"]
        values = [record["end_to_end_seconds"] for record in finals]
        after_end = [record["post_audio_seconds"] for record in finals]
        draft_values = [record["end_to_end_seconds"] for record in drafts]
        draft_after_end = [record["post_audio_seconds"] for record in drafts]
        result = {"session": self.session, "window": self.window, "sample_count": len(self._records),
                  "final_count": len(finals), "draft_count": len(drafts),
                  "final_p50": round(statistics.median(values), 6) if values else None,
                  "final_p95": _percentile(values, .95),
                  "after_end_p50": round(statistics.median(after_end), 6) if after_end else None,
                  "after_end_p95": _percentile(after_end, .95),
                  "draft_p50": round(statistics.median(draft_values), 6) if draft_values else None,
                  "draft_p95": _percentile(draft_values, .95),
                  "draft_after_end_p50": round(statistics.median(draft_after_end), 6) if draft_after_end else None,
                  "draft_after_end_p95": _percentile(draft_after_end, .95), "breakdown": {}}
        for name in (*_STAGES, "recognition_dispatch_seconds", "source_confirmation_seconds"):
            data = [record["breakdown"][name] for record in finals
                    if record["breakdown"].get(name) is not None]
            result["breakdown"][name] = {"p50": round(statistics.median(data), 6) if data else None,
                                         "p95": _percentile(data, .95)}
        return result
