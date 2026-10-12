from copy import deepcopy
import json

import pytest

from fanyiji.latency import LatencyTracker


def presented(stage="final", offset=0):
    names = ("audio_started_at", "audio_ended_at", "segment_ready_at", "recognition_dequeued_at",
             "filter_started_at", "filter_finished_at", "asr_started_at", "asr_finished_at",
             "translation_queued_at", "translation_started_at", "translation_finished_at",
             "ui_received_at", "ui_updated_at", "ui_painted_at")
    return {"session": 4, "channel": "system", "segment_id": "u1", "revision": 1,
            "context_generation": 2, "translation_stage": stage, "translation_provider": "codex",
            "translation_model": "test", "timing": dict(zip(names, [offset + i for i in range(14)]))}


def tracker():
    result = LatencyTracker()
    result.begin_session(4)
    return result


def test_full_timeline_has_audio_start_end_and_recognizer_start_totals():
    value = tracker().on_presented(presented())
    assert value["valid"]
    assert value["end_to_end_seconds"] == 13
    assert value["post_audio_seconds"] == 12
    assert value["recognition_to_display_seconds"] == 7
    assert sum(value["breakdown"].values()) == 13


def test_draft_and_final_are_independent_and_duplicate_repaint_not_counted():
    item = tracker()
    draft = presented("draft")
    final = presented()
    final["timing"]["translation_finished_at"] += .5
    assert item.on_presented(draft)["draft_end_to_end_seconds"] == 13
    assert item.on_presented(final)["draft_end_to_end_seconds"] == 13
    assert item.on_presented(final) is None
    assert item.summary()["sample_count"] == 2
    assert item.summary()["final_count"] == 1


def test_source_confirmation_wait_is_separate_from_ui_work():
    event = presented()
    event["timing"]["ui_painted_at"] = 25
    event.update(source_confirmed_at=24, source_confirmation_wait_seconds=12)
    value = tracker().on_presented(event)
    assert value["valid"]
    assert value["end_to_end_seconds"] == 25
    assert value["breakdown"]["source_confirmation_seconds"] == 12
    assert value["breakdown"]["ui_paint_seconds"] == 1


def test_first_draft_display_can_be_measured_with_unconfirmed_source():
    item = tracker()
    assert item.on_presented({**presented("draft"), "provisional": True})["valid"]
    assert item.summary()["draft_count"] == 1
    assert item.summary()["draft_p50"] == 13
    assert item.summary()["draft_p95"] == 13
    assert item.summary()["draft_after_end_p50"] == 12
    assert item.summary()["final_p50"] is None


@pytest.mark.parametrize("override", [
    {"session": 3}, {"failed": True}, {"provisional": True}, {"translation_stage": "unknown"},
])
def test_stale_failed_or_unconfirmed_results_are_not_counted(override):
    item = tracker()
    assert item.on_presented({**presented(), **override}) is None
    assert item.summary()["sample_count"] == 0


def test_invalid_or_missing_clock_does_not_fabricate_total_or_count_sample():
    item = tracker()
    event = presented()
    event["timing"]["audio_started_at"] = float("nan")
    value = item.on_presented(event)
    assert not value["valid"]
    assert value["end_to_end_seconds"] is None
    assert "audio_started_at" in value["missing_fields"]
    assert item.summary()["final_count"] == 0
    event = presented()
    event["timing"]["asr_started_at"] = 900
    assert item.on_presented(event)["clock_order_valid"] is False
    assert item.summary()["sample_count"] == 0


def test_source_revision_and_generation_do_not_borrow_old_draft_time():
    item = tracker()
    item.on_presented(presented("draft"))
    final = {**presented(), "revision": 2}
    assert item.on_presented(final)["draft_end_to_end_seconds"] is None
    final = {**presented(), "context_generation": 3}
    assert item.on_presented(final)["draft_end_to_end_seconds"] is None


def test_joined_audio_uses_measured_asr_work_instead_of_overlapping_envelope():
    event = presented()
    event["timing"].update(source_segment_count=2, audio_ended_at=8, segment_ready_at=9,
                            asr_seconds=1.5, joint_asr_seconds=.25, segmentation_seconds=.6,
                            asr_queue_seconds=.2)
    value = tracker().on_presented(event)
    assert value["valid"]
    assert value["end_to_end_seconds"] == 13
    assert value["breakdown_mode"] == "overlapping_joined_audio"
    assert value["breakdown"]["asr_seconds"] == 1.75
    assert value["breakdown"]["segmentation_seconds"] == .6


def test_logging_is_text_free_and_rpc_metrics_are_whitelisted():
    event = {**presented(), "text": "private words", "original": "private source",
             "rpc_metrics": {"receive_seconds": 3, "token": "private token", "account": "private account"}}
    value = tracker().on_presented(event)
    assert value["rpc_metrics"] == {"receive_seconds": 3.0}
    assert "private" not in json.dumps(value)
    assert event["text"] == "private words"


def test_session_reset_and_bounded_statistics_with_nearest_rank_p95():
    item = LatencyTracker(window=2)
    item.begin_session(4)
    for index in range(3):
        event = {**presented(offset=index), "segment_id": str(index)}
        item.on_presented(event)
    assert item.summary()["final_count"] == 2
    assert item.summary()["final_p95"] == 13
    item.begin_session(5)
    assert item.summary()["final_count"] == 0
    assert item.on_presented(presented()) is None


def test_audio_processing_duration_is_not_counted_as_dispatch_again():
    event = presented()
    event["timing"].update(audio_processing_started_at=5.1,
        audio_processing_finished_at=5.8, audio_processing_seconds=.7)
    value = tracker().on_presented(event)
    assert value["valid"]
    assert value["breakdown"]["audio_processing_seconds"] == .7
    assert value["breakdown"]["recognition_dispatch_seconds"] == 1.3
    assert sum(value["breakdown"].values()) == 13


def test_partial_processing_clocks_do_not_enter_latency_statistics():
    event = presented()
    event["timing"]["audio_processing_started_at"] = 5.1
    value = tracker().on_presented(event)
    assert not value["valid"]


def test_joined_processing_sums_real_work_without_reprocessing_envelope():
    event = presented()
    event["timing"].update(source_segment_count=2,
        audio_processing_started_at=5.1, audio_processing_finished_at=7.5,
        audio_processing_seconds=.3, recognition_dispatch_seconds=.15)
    value = tracker().on_presented(event)
    assert value["valid"]
    assert value["breakdown"]["audio_processing_seconds"] == .3
    assert value["breakdown"]["recognition_dispatch_seconds"] == .15
