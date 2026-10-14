"""Only actual, current subtitle presentations enter recorder latency statistics."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from fanyiji.config import AppConfig
from fanyiji.diagnostics import SessionRecorder


def source(recorder, identifier="a"):
    recorder.on_segment(SimpleNamespace(channel="system", segment_id=identifier,
        samples=np.zeros(16000, np.float32), direction=("ja", "zh"),
        timing={"audio_started_at": 100, "audio_ended_at": 101}), True)


def original(identifier="a", *, revision=1, generation=2, provisional=False):
    return {"session": 4, "channel": "system", "kind": "original", "segment_id": identifier,
        "text": "今日は晴れです。", "revision": revision, "context_generation": generation,
        "source_segment_ids": [identifier], "provisional": provisional}


def translation(identifier="a", *, revision=1, generation=2, stage="final", provisional=False):
    names = ("audio_started_at", "audio_ended_at", "segment_ready_at", "recognition_dequeued_at",
        "filter_started_at", "filter_finished_at", "asr_started_at", "asr_finished_at",
        "translation_queued_at", "translation_started_at", "translation_finished_at")
    return {"session": 4, "channel": "system", "kind": "translation", "segment_id": identifier,
        "text": "今天晴天。", "revision": revision, "context_generation": generation,
        "translation_stage": stage, "translation_provider": "local" if stage == "draft" else "codex",
        "translation_model": "fixture", "timing": dict(zip(names, range(100, 111))),
        "latency": 10, "source_segment_ids": [identifier], "provisional": provisional}


def presentation(event, *, paint=113):
    return {**event, "kind": "subtitle_presented", "timing": deepcopy(event["timing"]),
        "ui_received_at": 111, "ui_updated_at": 112, "ui_painted_at": paint}


def read(recorder):
    recorder.snapshot()
    return json.loads((recorder.directory / "results.json").read_text(encoding="utf-8"))


@pytest.fixture
def recorder(tmp_path):
    result = SessionRecorder(tmp_path / "recording", AppConfig())
    yield result
    result.finish()


def test_final_response_does_not_count_until_ui_paint_and_then_merges_complete_measurement(recorder):
    source(recorder)
    recorder.event(original())
    event = translation()
    recorder.event(event)
    pre_paint = read(recorder)
    assert pre_paint["performance_summary"]["final_count"] == 0
    assert "presentation_metrics" not in pre_paint["samples"][0]
    recorder.event(presentation(event))
    report = read(recorder)
    sample = report["samples"][0]
    assert sample["presentation_metrics"]["valid"]
    assert sample["end_to_end_display_seconds"] == 13
    assert sample["post_audio_display_seconds"] == 12
    assert sample["recognition_to_display_seconds"] == 7
    assert sample["pipeline_latency_seconds"] == 10
    assert report["performance_summary"]["final_count"] == 1
    assert report["performance_summary"]["final_p50"] == 13
    assert "private" not in json.dumps(sample["presentation_metrics"])


def test_duplicate_paints_and_repeated_snapshots_do_not_multiply_statistics(recorder):
    source(recorder)
    recorder.event(original())
    event = translation()
    recorder.event(event)
    recorder.event(presentation(event))
    recorder.event(presentation(event, paint=114))
    first, second = read(recorder), read(recorder)
    assert first["performance_summary"] == second["performance_summary"]
    assert first["samples"] == second["samples"]
    assert second["performance_summary"]["final_count"] == 1
    assert second["performance_summary"]["final_p95"] == 13


@pytest.mark.parametrize("boundary", ["revision", "context"])
def test_late_presentation_for_old_source_cannot_enter_samples_or_summary(recorder, boundary):
    source(recorder)
    recorder.event(original(revision=2 if boundary == "revision" else 1,
        generation=3 if boundary == "context" else 2))
    recorder.event(presentation(translation()))
    report = read(recorder)
    assert "presentation_metrics" not in report["samples"][0]
    assert report["performance_summary"]["final_count"] == 0


def test_failed_missing_or_provisional_clock_does_not_claim_display_success(recorder):
    source(recorder)
    recorder.event(original())
    bad_events = [presentation({**translation(), "failed": True}),
        presentation(translation(provisional=True)), presentation(translation())]
    bad_events[-1]["timing"].pop("audio_started_at")
    for event in bad_events:
        recorder.event(event)
    report = read(recorder)
    assert report["performance_summary"]["final_count"] == 0
    assert "end_to_end_display_seconds" not in report["samples"][0]
    assert report["samples"][0]["presentation_stages"]["final"]["valid"] is False


def test_source_confirmation_then_true_final_paint_keeps_original_translation_timing(recorder):
    source(recorder)
    recorder.event(original(provisional=True))
    event = translation(provisional=True)
    recorder.event(event)
    recorder.event(presentation(event))
    assert read(recorder)["performance_summary"]["final_count"] == 0
    recorder.event({"session": 4, "channel": "system", "kind": "subtitle_final",
        "segment_id": "a", "revision": 1})
    confirmed = presentation({**event, "provisional": False}, paint=120)
    confirmed["ui_updated_at"] = 119
    recorder.event(confirmed)
    report = read(recorder)
    assert report["samples"][0]["translation_provisional"] is False
    assert report["samples"][0]["presentation_metrics"]["timing"]["translation_finished_at"] == 110
    assert report["performance_summary"]["final_p50"] == 20


def test_late_local_candidate_after_codex_final_does_not_create_a_display_measurement(recorder):
    source(recorder)
    recorder.event(original())
    final = translation()
    recorder.event(final)
    recorder.event(presentation(final))
    draft = {**translation(stage="draft"), "kind": "translation_candidate"}
    recorder.event(draft)
    report = read(recorder)
    sample = report["samples"][0]
    assert sample["translation_stage"] == "final"
    assert sample["presentation_metrics"]["stage"] == "final"
    assert list(sample["presentation_stages"]) == ["final"]
    assert report["performance_summary"]["final_count"] == 1
    assert report["performance_summary"]["draft_count"] == 0


def test_joined_source_uses_cumulative_asr_work_without_multiplying_rpc_request(recorder):
    for identifier in ("a", "b"):
        source(recorder, identifier)
    joined = original(revision=2)
    joined["source_segment_ids"] = ["a", "b"]
    recorder.event(joined)
    event = translation(revision=2)
    event.update(source_segment_ids=["a", "b"], batch_id="batch-1", batch_size=2,
        processing_seconds=4, rpc_metrics={"total_seconds": 3.8, "receive_seconds": 3})
    event["timing"].update(source_segment_count=2, audio_ended_at=108,
        segment_ready_at=109, asr_seconds=1.5, joint_asr_seconds=.25,
        segmentation_seconds=.6, asr_queue_seconds=.2, speech_filter_seconds=.1,
        assembly_processing_seconds=.02)
    recorder.event(event)
    recorder.event(presentation(event))
    recorder.event({**event, "segment_id": "b"})
    report = read(recorder)
    sample = report["samples"][0]
    assert sample["presentation_metrics"]["breakdown_mode"] == "overlapping_joined_audio"
    assert sample["presentation_metrics"]["breakdown"]["asr_seconds"] == 1.75
    assert sample["end_to_end_display_seconds"] == 13
    assert len(report["translation_requests"]) == 1
    assert report["translation_requests"][0]["processing_seconds"] == 4


def test_fallback_statistics_separate_actual_translation_work_and_wait_envelope(recorder):
    source(recorder)
    recorder.event(original())
    event = translation()
    # Timeline is ordered, but one failed-batch source waits while another
    # source is retried: request work=3s, first-attempt-to-finish envelope=4s.
    event["timing"].update(translation_started_at=109, translation_finished_at=113,
        translation_seconds=3)
    presented = presentation(event, paint=116)
    presented.update(ui_received_at=114, ui_updated_at=115)
    recorder.event(event)
    recorder.event(presented)
    report = read(recorder)
    measurement = report["samples"][0]["presentation_metrics"]
    assert measurement["translation_envelope_seconds"] == 4
    assert measurement["breakdown"]["translation_seconds"] == 3
    assert report["performance_summary"]["breakdown"]["translation_seconds"]["p50"] == 3
