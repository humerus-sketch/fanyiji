"""Parallel branch provenance and timing without devices or model requests."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from fanyiji.config import AppConfig, SourceConfig
from fanyiji.diagnostics import SessionRecorder
from fanyiji.specialized_model import SpecializedModelService
from fanyiji.specialized_store import SpecializedStore


def original(identifier="a", **extra):
    return {"kind": "original", "session": 7, "channel": "system", "segment_id": identifier,
            "revision": 1, "context_generation": 3, "text": "明日も雨です。", **extra}


def result(stage, text="明天也下雨。", identifier="a", kind="translation", **extra):
    provider = "local" if stage == "draft" else "codex"
    return {"kind": kind, "session": 7, "channel": "system", "segment_id": identifier,
            "revision": 1, "context_generation": 3, "text": text, "translation_stage": stage,
            "translation_provider": provider, "translation_model": "hy-local" if provider == "local" else "gpt-actual",
            "latency": .7 if stage == "draft" else 2.1, "processing_seconds": .4 if stage == "draft" else 1.8,
            "queue_wait_seconds": .1, **extra}


@pytest.fixture
def corpus(tmp_path):
    store = SpecializedStore(tmp_path / "corpus")
    yield store
    store.close()


def record(store, event):
    store.record_event("hybrid-session", event, direction="ja_to_zh", provider="hybrid", model="not-a-model")


@pytest.mark.parametrize("final_first", [False, True])
def test_hybrid_candidates_keep_true_origins_and_manual_training_boundary(corpus, final_first):
    record(corpus, original())
    draft = result("draft", "明天还下雨。", kind="translation_candidate" if final_first else "translation")
    final = result("final")
    for event in (final, draft) if final_first else (draft, final):
        record(corpus, event)
    sample = corpus.list_samples()[0]
    assert not sample["failed"]
    assert sample["metadata"]["translation_display"] == {"stage": "final", "provider": "codex", "model": "gpt-actual"}
    assert {(candidate["origin"], candidate["model"]) for candidate in sample["candidates"]} == {
        ("local", "hy-local"), ("codex", "gpt-actual")}
    assert not any(candidate["training_allowed"] for candidate in sample["candidates"])
    for candidate in sample["candidates"]:
        corpus.review_candidate(sample["id"], candidate["id"], source_verified=True, rights_confirmed=True)
    checked = corpus.get_sample(sample["id"])
    assert all(not candidate["training_allowed"] for candidate in checked["candidates"] if candidate["origin"] == "codex")
    assert corpus.summary()["trainable"] == 1
    assert corpus.summary()["deployment_locked"] is True


@pytest.mark.parametrize("failure_first", [False, True])
def test_codex_failure_does_not_disqualify_a_valid_local_candidate(corpus, failure_first):
    record(corpus, original())
    failed = result("final", "", kind="translation_status", failed=True, overall_failed=False, error="TimeoutError")
    draft = result("draft", final_failed=True)
    for event in (failed, draft) if failure_first else (draft, failed):
        record(corpus, event)
    sample = corpus.list_samples()[0]
    assert not sample["failed"]
    assert len(sample["candidates"]) == 1
    assert sample["candidates"][0]["origin"] == "local"
    assert sample["metadata"]["translation_branches"]["codex"]["state"] == "failed"
    assert sample["metadata"]["translation_display"]["stage"] == "draft"


def test_both_failed_branches_are_not_training_labels(corpus):
    record(corpus, original())
    record(corpus, result("draft", "", kind="translation_status", failed=True, overall_failed=False))
    record(corpus, result("final", "", kind="translation_status", failed=True, overall_failed=True))
    record(corpus, result("final", "[翻译失败]", failed=True, overall_failed=True))
    sample = corpus.list_samples()[0]
    assert sample["failed"]
    assert sample["candidates"] == []
    assert corpus.pending_comparisons(["other-local"]) == []


def test_late_failure_status_cannot_clear_an_overall_failure(corpus):
    record(corpus, original())
    record(corpus, result("final", "[翻译失败]", failed=True, overall_failed=True))
    record(corpus, result("draft", "", kind="translation_status", failed=True, overall_failed=False))
    assert corpus.list_samples()[0]["failed"]


def test_late_failure_never_reclassifies_a_successful_final_branch(corpus):
    record(corpus, original())
    record(corpus, result("final"))
    record(corpus, result("draft", "", kind="translation_status", failed=True, overall_failed=False))
    record(corpus, result("final", "", kind="translation_status", failed=True, overall_failed=True))
    sample = corpus.list_samples()[0]
    assert not sample["failed"]
    assert sample["metadata"]["translation_branches"]["codex"]["state"] == "success"
    assert sample["metadata"]["translation_display"]["stage"] == "final"


def test_late_candidate_keeps_final_context_and_source_confirmation_separate(corpus):
    record(corpus, original(provisional=True))
    record(corpus, result("final", context="终稿原文上下文", provisional=False))
    record(corpus, result("draft", kind="translation_candidate", context="迟到原文上下文", provisional=True))
    sample = corpus.list_samples()[0]
    assert sample["provisional"] and not sample["finalized"]
    assert sample["context"] == "终稿原文上下文"
    record(corpus, {**original(), "kind": "subtitle_final"})
    record(corpus, original(provisional=True))
    confirmed = corpus.get_sample(sample["id"])
    assert confirmed["finalized"] and not confirmed["provisional"]
    assert confirmed["metadata"]["translation_display"]["stage"] == "final"
    assert len(confirmed["candidates"]) == 2


def test_context_generation_and_revision_cannot_cross_attach_candidates(corpus):
    record(corpus, original())
    record(corpus, result("draft", context_generation=4))
    record(corpus, result("final", revision=2))
    assert corpus.list_samples()[0]["candidates"] == []


def test_hybrid_data_is_readable_after_reopening_without_schema_migration(tmp_path):
    directory = tmp_path / "corpus"
    store = SpecializedStore(directory)
    before = [(row[1], row[2]) for row in store._db.execute("PRAGMA table_info(samples)")]
    record(store, original())
    record(store, result("draft"))
    store.close()
    reopened = SpecializedStore(directory)
    try:
        assert [(row[1], row[2]) for row in reopened._db.execute("PRAGMA table_info(samples)")] == before
        assert reopened.list_samples()[0]["metadata"]["translation_branches"]["local"]["model"] == "hy-local"
    finally:
        reopened.close()


def test_service_uses_actual_models_for_both_branches_and_rejects_unlabelled_hybrid(tmp_path):
    store = Mock()
    service = SpecializedModelService(tmp_path / "specialized", tmp_path / "models", store=store)
    try:
        config = AppConfig(microphone=SourceConfig(enabled=False), translation_provider="hybrid", codex_model="configured-model")
        service.begin_session(7, config)
        service.on_event({"kind": "models_ready", "session": 7, "models": {"translation": {
            "mode": "hybrid", "model_id": "actual-cloud", "draft_model": {"model_id": "actual-local"}}}})
        service.on_event(original())
        service.on_event(result("draft"))
        service.on_event(result("final"))
        unlabelled = original(kind="translation")
        service.on_event(unlabelled)
        inferred = result("draft", kind="translation_candidate")
        inferred.pop("translation_provider")
        inferred.pop("translation_model")
        service.on_event(inferred)
        service.on_event(result("final", kind="translation_status", failed=True, session=6))
        service.flush()
        assert store.record_event.call_count == 4
        calls = store.record_event.call_args_list
        assert [(call.kwargs["provider"], call.kwargs["model"]) for call in calls[1:]] == [
            ("local", "hy-local"), ("codex", "gpt-actual"), ("local", "actual-local")]
    finally:
        service.close()


def recorder(tmp_path):
    recording = SessionRecorder(tmp_path / "recording", AppConfig(microphone=SourceConfig(enabled=False), translation_provider="hybrid"))
    for identifier in ("a", "b"):
        recording.on_segment(SimpleNamespace(channel="system", segment_id=identifier,
            samples=np.zeros(1600, np.float32), direction=("ja", "zh")), True)
        recording.event(original(identifier))
    return recording


def finish(recording):
    recording.snapshot()
    recording.snapshot()
    recording.finish()
    return json.loads((recording.directory / "results.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("final_first", [False, True])
def test_diagnostics_separate_draft_final_and_never_replace_final_with_late_candidate(tmp_path, final_first):
    recording = recorder(tmp_path)
    draft = result("draft", "明天还下雨。", kind="translation_candidate" if final_first else "translation")
    final = result("final")
    for event in (final, draft) if final_first else (draft, final):
        recording.event(event)
    sample = finish(recording)["samples"][0]
    assert sample["display_translation"] == sample["production_translation"] == "明天也下雨。"
    assert sample["translation_provider"] == "codex" and sample["translation_model"] == "gpt-actual"
    assert sample["translation_stage"] == "final"
    assert len(sample["translation_candidates"]) == 2
    assert len(sample["translation_results"]) == 2
    assert sample["draft_latency_seconds"] == .7 and sample["final_latency_seconds"] == 2.1
    assert sample["first_display_latency_seconds"] == (2.1 if final_first else .7)
    assert ("draft_display_latency_seconds" in sample) is not final_first


def test_diagnostics_count_actual_requests_once_per_provider_not_per_output_or_snapshot(tmp_path):
    recording = recorder(tmp_path)
    for stage in ("draft", "final"):
        request = result(stage, kind="translation_request", batch_id="same-id", batch_size=2, segment_ids=["a", "b"])
        request.pop("segment_id")
        recording.event(request)
        for identifier in ("a", "b"):
            recording.event(result(stage, identifier=identifier, batch_id="same-id", batch_size=2))
    requests = finish(recording)["translation_requests"]
    assert len(requests) == 2
    assert {request["translation_provider"] for request in requests} == {"local", "codex"}
    assert all(request["segment_ids"] == ["a", "b"] for request in requests)
    assert sum(request["processing_seconds"] for request in requests) == pytest.approx(2.2)


def test_diagnostics_retains_local_draft_and_cloud_failure_after_late_draft(tmp_path):
    recording = recorder(tmp_path)
    recording.event(result("final", "", kind="translation_status", failed=True, overall_failed=False, error="TimeoutError"))
    recording.event(result("draft", final_failed=True))
    sample = finish(recording)["samples"][0]
    assert sample["display_translation_valid"]
    assert sample["translation_stage"] == "draft"
    assert sample["translation_final_failed"] and not sample["translation_overall_failed"]
    assert sample["translation_stages"]["final"]["error"] == "TimeoutError"
    assert len(sample["translation_candidates"]) == 1


@pytest.mark.parametrize("new_revision,new_generation", [(2, 3), (1, 4)])
def test_diagnostic_replay_is_idempotent_and_never_carries_old_source_timings(tmp_path, new_revision, new_generation):
    recording = recorder(tmp_path)
    recording.event(result("draft", "旧暂译", latency=.7))
    recording.event(result("final", "", kind="translation_status", failed=True, error="old-failure"))
    recording.event(original(text="変更した予定です。", revision=new_revision, context_generation=new_generation))
    recording.event(result("final", "这是调整后的安排。", revision=new_revision, context_generation=new_generation, latency=2.1))
    try:
        snapshots = []
        for _ in range(2):
            recording.snapshot()
            snapshots.append(json.loads((recording.directory / "results.json").read_text(encoding="utf-8"))["samples"][0])
        assert snapshots[0] == snapshots[1]
        sample = snapshots[0]
        assert sample["display_translation"] == "这是调整后的安排。"
        assert sample["first_display_latency_seconds"] == sample["final_latency_seconds"] == 2.1
        assert "draft_latency_seconds" not in sample and "draft_display_latency_seconds" not in sample
        assert list(sample["translation_stages"]) == ["final"]
        assert not sample["translation_final_failed"]
        # History remains available for comparison, tagged with its own source.
        assert len(sample["translation_candidates"]) == 2
        assert len(sample["translation_results"]) == 3
    finally:
        recording.finish()


def test_diagnostics_new_source_pending_has_no_previous_revision_draft_metrics(tmp_path):
    recording = recorder(tmp_path)
    recording.event(result("draft", "旧暂译"))
    recording.snapshot()
    recording.event(original(text="新しい予定です。", revision=2))
    try:
        recording.snapshot()
        sample = json.loads((recording.directory / "results.json").read_text(encoding="utf-8"))["samples"][0]
        assert sample["display_original_revision"] == 2
        assert not sample["display_translation_valid"]
        assert sample["translation_stages"] == {}
        assert "translation_stage" not in sample and "draft_latency_seconds" not in sample
        assert sample["production_translation"] == "旧暂译"
        assert sample["production_translation_revision"] == 1
    finally:
        recording.finish()


def test_hybrid_late_raw_final_is_retained_without_replacing_newer_joined_display(tmp_path):
    recording = recorder(tmp_path)
    recording.event(result("draft", "旧暂译", source_segment_ids=["a"]))
    recording.event(original(text="森の妖精です。", revision=2, source_segment_ids=["a", "b"]))
    recording.event(result("final", "森林精灵。", revision=2, source_segment_ids=["a", "b"], latency=2.1))
    recording.event(result("final", "森林的。", source_segment_ids=["a"], latency=3.2))
    recording.event(result("draft", "迟到旧暂译", kind="translation_candidate", source_segment_ids=["a"], latency=3.3))
    sample = finish(recording)["samples"][0]
    assert sample["display_translation"] == "森林精灵。"
    assert sample["translation_revision"] == 2
    assert sample["production_translation"] == "森林的。"
    assert sample["production_translation_revision"] == 1
    assert sample["production_translation_stage"] == "final"
    assert sample["production_translation_provider"] == "codex"
    assert sample["production_translation_model"] == "gpt-actual"
    assert sample["first_display_latency_seconds"] == sample["final_latency_seconds"] == 2.1
    assert "draft_latency_seconds" not in sample
