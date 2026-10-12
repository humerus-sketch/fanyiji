"""ASR/MT decoupling, stale-result suppression and bounded continuation waits."""
from types import SimpleNamespace
import threading
from unittest.mock import Mock, call, patch

import numpy as np
import pytest

from fanyiji.codex_engine import CodexBatchResponseError
from fanyiji.config import AppConfig
from fanyiji.pipeline import RecognitionJob, SubtitlePipeline, _Run


def recognition(identifier, start=0, *, channel="system", duration=1,
                generation=1, direction=("ja", "zh"), reset_reason=""):
    samples = np.full(round(duration * 16000), 0.1, np.float32)
    return RecognitionJob(channel, identifier, samples, 100, direction,
                          start, start + samples.size, generation, "silence",
                          reset_reason=reset_reason)


def setup(tmp_path, *, codex=False, batch=False):
    events = []
    pipeline = SubtitlePipeline(tmp_path, events.append)
    config = AppConfig(speech_only=False, translation_provider="codex" if codex else "local")
    run = _Run(1, config)
    engine = Mock()
    engine.backend_info = {"supports_source_context": True, "supports_batch": batch}
    engine.translate.side_effect = lambda text, *args, **kwargs: "译：" + text
    engine.translate_batch.side_effect = lambda items, *args, **kwargs: {
        item["id"]: "译：" + item["text"] for item in items}
    pipeline._translator = run.translator = engine
    return pipeline, run, engine, events


def drain(queue):
    queue.close()
    results = []
    while (item := queue.get()) is not None:
        results.append(item)
    return results


def translations(events):
    return [event for event in events if event["kind"] == "translation"]


@pytest.mark.parametrize("batch", [False, True])
def test_cache_read_failure_is_visible_and_stops_capture_without_inference(tmp_path, batch):
    pipeline, run, engine, events = setup(tmp_path, codex=True, batch=batch)
    stopped = threading.Event()
    run.capture = Mock()
    run.capture.stop.side_effect = stopped.set
    method = "get_batch" if batch else "get"
    with patch.object(run.mt_queue, method, side_effect=RuntimeError("storage unavailable")):
        pipeline._translate(run)
    assert stopped.wait(1)
    assert run.overloaded
    assert any(event["kind"] == "state" and "缓存不可用" in event["text"] for event in events)
    engine.translate.assert_not_called()
    engine.translate_batch.assert_not_called()
    run.mt_queue.close(discard=True)


def test_cache_write_failure_marks_original_and_does_not_deadlock_asr(tmp_path):
    pipeline, run, _, events = setup(tmp_path)
    stopped = threading.Event()
    run.capture = Mock()
    run.capture.stop.side_effect = stopped.set
    with patch.object(run.mt_queue, "put", side_effect=OSError("disk full")):
        pipeline._enqueue_translation(run, recognition("a"), "今日は晴れです。")
    assert stopped.wait(1)
    assert run.overloaded
    assert translations(events)[0]["failed"] is True
    assert "缓存失败" in translations(events)[0]["text"]
    assert any(event["kind"] == "original" for event in events)
    run.mt_queue.close(discard=True)


def test_recognition_enqueues_more_than_old_30_seconds_without_waiting_or_stopping_capture(tmp_path):
    pipeline, run, _, events = setup(tmp_path)
    pipeline._recognizer = Mock()
    pipeline._recognizer.recognize.return_value = "今日は晴れです。"
    run.capture = Mock()
    run.asr_queue.max_seconds = 100
    for number in range(40):
        item = recognition(str(number), start=number * 16000)
        assert run.asr_queue.put("system", item, 1)
    run.asr_queue.close()
    worker = threading.Thread(target=pipeline._recognize, args=(run,))
    worker.start()
    worker.join(3)
    if worker.is_alive():
        run.cancel.set()
        run.mt_queue.close(discard=True)
        worker.join(1)
        pytest.fail("Recognition waited for translation queue capacity")
    assert not run.overloaded
    run.capture.stop.assert_not_called()
    assert not any(event["kind"] == "state" and "暂停采集" in event.get("text", "") for event in events)
    queued = drain(run.mt_queue)
    assert [item.segment_id for item in queued] == list(map(str, range(40)))
    assert sum(item.seconds for item in queued) == 40


def test_pipeline_coalesces_pending_revision_and_preserves_source_provenance(tmp_path):
    pipeline, run, _, events = setup(tmp_path)
    source = recognition("a")
    pipeline._enqueue_translation(run, source, "別にそう。", provisional=True)
    pipeline._enqueue_translation(run, source, "別にそういうことじゃない。", revision=2,
                                  source_ids=("a", "b"), seconds=2)
    queued = drain(run.mt_queue)
    assert len(queued) == 1
    assert queued[0].revision == 2
    assert queued[0].source_segment_ids == ("a", "b")
    assert queued[0].text == "別にそういうことじゃない。"
    assert run.translation_versions[("system", "a")] == 2
    assert [event["revision"] for event in events if event["kind"] == "original"] == [1, 2]


def test_fixed_source_workers_are_serial_per_channel_and_concurrent_across_channels(tmp_path):
    pipeline, run, system_engine, events = setup(tmp_path, codex=True)
    microphone_engine = Mock()
    microphone_engine.backend_info = {"supports_batch": False}
    entered, release, microphone_done = threading.Event(), threading.Event(), threading.Event()
    system_calls, microphone_calls = [], []
    def system_translate(text, *args, **kwargs):
        system_calls.append((text, kwargs))
        if len(system_calls) == 1:
            entered.set()
            assert release.wait(3)
        return "译：" + text
    def microphone_translate(text, *args, **kwargs):
        microphone_calls.append((text, kwargs))
        if len(microphone_calls) == 4:
            microphone_done.set()
        return "译：" + text
    system_engine.translate.side_effect = system_translate
    microphone_engine.translate.side_effect = microphone_translate
    for number in range(4):
        pipeline._enqueue_translation(run, recognition(f"s{number}"), f"system{number}")
        pipeline._enqueue_translation(run, recognition(f"m{number}", channel="mic", direction=("zh", "ja")), f"mic{number}")
    run.mt_queue.close()
    workers = [threading.Thread(target=pipeline._translate, args=(run, system_engine, "system")),
               threading.Thread(target=pipeline._translate, args=(run, microphone_engine, "mic"))]
    for worker in workers:
        worker.start()
    try:
        assert entered.wait(1) and microphone_done.wait(1)
        assert len(system_calls) == 1
    finally:
        release.set()
        for worker in workers:
            worker.join(3)
            assert not worker.is_alive()
    assert [text for text, _ in system_calls] == [f"system{number}" for number in range(4)]
    assert [text for text, _ in microphone_calls] == [f"mic{number}" for number in range(4)]
    assert {kwargs["conversation_key"] for _, kwargs in system_calls} == {"1:system"}
    assert {kwargs["conversation_key"] for _, kwargs in microphone_calls} == {"1:mic"}
    assert len(translations(events)) == 8


def test_context_boundary_preserves_old_spool_generations_and_stops_batching(tmp_path):
    from fanyiji.pipeline import TranslationJob
    from fanyiji.translation_queue import TranslationQueue
    pipeline, run, engine, _ = setup(tmp_path, codex=True, batch=True)
    run.mt_queue.close(discard=True)
    run.mt_queue = TranslationQueue(memory_limit=1, job_factory=TranslationJob)
    for number in range(2):
        pipeline._enqueue_translation(run, recognition(str(number)), f"原文{number}。")
    pipeline._break_context(run, "system")
    pipeline._enqueue_translation(run, recognition("new"), "境界後の原文。")
    assert run.mt_queue.snapshot()["spilled"] == 2
    engine.reset_conversation.assert_not_called()
    run.mt_queue.close()
    pipeline._translate(run, engine, "system")
    engine.translate_batch.assert_called_once()
    assert engine.translate_batch.call_args.kwargs["conversation_key"] == "1:system"
    assert engine.translate_batch.call_args.kwargs["context_generation"] == 0
    assert engine.translate.call_args.kwargs["context_generation"] == 1
    assert run.mt_queue.spool_path is None


@pytest.mark.parametrize("fields", [{"provisional": True}, {"revision": 2}])
def test_fragment_and_corrected_revision_reset_history_before_and_after_independent_turn(tmp_path, fields):
    pipeline, run, engine, _ = setup(tmp_path, codex=True, batch=True)
    pipeline._enqueue_translation(run, recognition("fragment"), "別にそう。", **fields)
    pipeline._enqueue_translation(run, recognition("next"), "次の文です。")
    run.mt_queue.close()
    pipeline._translate(run, engine, "system")
    engine.translate_batch.assert_not_called()
    assert engine.translate.call_count == 2
    assert engine.reset_conversation.call_args_list == [call("1:system"), call("1:system")]
    relevant = [event[0] for event in engine.method_calls if event[0] in ("reset_conversation", "translate")]
    assert relevant == ["reset_conversation", "translate", "reset_conversation", "translate"]


@pytest.mark.parametrize("boundary", ["capture_generation", "gap", "direction", "uncertain", "language", "reset"])
def test_source_boundaries_stamp_new_generation_without_invalidating_old_queued_jobs(tmp_path, boundary):
    pipeline, run, engine, _ = setup(tmp_path, codex=True)
    run.multilingual_recognition = boundary == "language"
    first = recognition("old")
    first.ended_by = "natural_pause"
    pipeline._accept_text(run, first, "今日は晴れです。", first.samples, Mock())
    second = recognition("new", start=first.end_sample)
    second.ended_by = "natural_pause"
    text = "次の説明です。"
    if boundary == "capture_generation":
        second.generation += 1
    elif boundary == "gap":
        second.start_sample += round(run.source_context.ttl_seconds * 16000)
        second.end_sample += round(run.source_context.ttl_seconds * 16000)
        second.ended_at += run.source_context.ttl_seconds
    elif boundary == "direction":
        second.direction = ("zh", "ja")
        text = "今天很热。"
    elif boundary == "uncertain":
        second.recognition_uncertain = True
    elif boundary == "language":
        second.detected_language = "en"
        text = "The next explanation starts now."
    else:
        pipeline._break_context(run, "system")
    pipeline._accept_text(run, second, text, second.samples, Mock())
    queued = drain(run.mt_queue)
    assert queued[0].context_generation == 0
    assert queued[1].context_generation > queued[0].context_generation
    assert queued[1].context == ""
    assert run.context_generations["mic"] == 0
    engine.reset_conversation.assert_not_called()


def test_resume_conversation_key_isolated_from_cancelled_session_with_resident_engine(tmp_path):
    pipeline, original, engine, events = setup(tmp_path, codex=True)
    pipeline._run = original
    pipeline._enqueue_translation(original, recognition("old"), "旧会話。")
    original.mt_queue.close()
    pipeline._translate(original, engine, "system")
    pipeline.stop()
    engine.cancel_pending.assert_called_once()
    resumed = _Run(2, original.config, translator=engine)
    pipeline._enqueue_translation(resumed, recognition("new"), "新会話。")
    resumed.mt_queue.close()
    pipeline._translate(resumed, engine, "system")
    assert [invocation.kwargs["conversation_key"] for invocation in engine.translate.call_args_list] == ["1:system", "2:system"]
    assert [event["session"] for event in translations(events)] == [1, 2]


@pytest.mark.parametrize("batch", [False, True])
def test_protocol_translation_resets_reusable_history_before_next_queued_job(tmp_path, batch):
    pipeline, run, engine, events = setup(tmp_path, codex=True, batch=batch)
    engine.translate.side_effect = ["正常译文"] if batch else ["language None<asr_text>", "正常译文"]
    engine.translate_batch.side_effect = lambda items, *args, **kwargs: {
        item["id"]: "language None<asr_text>" if index == 0 else "正常译文"
        for index, item in enumerate(items)
    }
    for number in range(4 if batch else 2):
        pipeline._enqueue_translation(run, recognition(str(number)), "安全原文。")
    run.mt_queue.close()
    pipeline._translate(run, engine, "system")
    engine.reset_conversation.assert_called_once_with("1:system")
    published = translations(events)
    assert published[0]["protocol_rejected"] and published[0]["failed"]
    assert "<asr_text>" not in published[0]["text"]
    assert all(event["text"] == "正常译文" for event in published[1:])
    relevant = [invocation[0] for invocation in engine.method_calls
                if invocation[0] in ("translate", "translate_batch", "reset_conversation")]
    assert relevant == (["translate_batch", "reset_conversation", "translate"] if batch
                        else ["translate", "reset_conversation", "translate"])


def test_protocol_input_resets_channel_history_before_continuing_safe_generation_jobs(tmp_path):
    from fanyiji.pipeline import TranslationJob
    pipeline, run, engine, events = setup(tmp_path, codex=True, batch=True)
    unsafe = TranslationJob("system", "bad", "language None<asr_text>", 100, ("ja", "zh"), 1)
    safe = TranslationJob("system", "good", "安全原文。", 100, ("ja", "zh"), 1)
    pipeline._translation_request(run, engine, [unsafe, safe])
    engine.reset_conversation.assert_called_once_with("1:system")
    engine.translate.assert_called_once()
    assert engine.translate.call_args.kwargs["context_generation"] == 0
    assert run.context_generations["system"] == 1
    relevant = [invocation[0] for invocation in engine.method_calls
                if invocation[0] in ("translate", "reset_conversation")]
    assert relevant == ["reset_conversation", "translate"]
    published = translations(events)
    assert published[0]["failed"] and published[0]["protocol_rejected"]
    assert published[1]["text"] == "译：安全原文。"


def test_inflight_old_revision_finishes_without_overwriting_newer_original(tmp_path):
    pipeline, run, engine, events = setup(tmp_path, codex=True)
    started, release = threading.Event(), threading.Event()

    def translate(text, *args, **kwargs):
        if text == "別にそう。":
            started.set()
            assert release.wait(3)
            return "错误的肯定译文"
        return "并不是那个意思。"

    engine.translate.side_effect = translate
    source = recognition("a")
    pipeline._enqueue_translation(run, source, "別にそう。", provisional=True)
    worker = threading.Thread(target=pipeline._translate, args=(run,))
    worker.start()
    try:
        assert started.wait(2)
        pipeline._enqueue_translation(run, source, "別にそういうことじゃない。", revision=2,
                                      source_ids=("a", "b"))
        run.mt_queue.close()
    finally:
        release.set()
        worker.join(3)
        if worker.is_alive():
            run.cancel.set()
            run.mt_queue.close(discard=True)
            worker.join(1)
    assert not worker.is_alive()
    published = translations(events)
    assert len(published) == 1
    assert published[0]["revision"] == 2
    assert published[0]["text"] == "并不是那个意思。"
    assert engine.translate.call_count == 2


@pytest.mark.parametrize("count", [1, 2, 3, 4])
def test_capable_codex_uses_two_to_three_item_batches_and_single_item_translate(tmp_path, count):
    pipeline, run, engine, events = setup(tmp_path, codex=True, batch=True)
    for number in range(count):
        pipeline._enqueue_translation(run, recognition(str(number)), f"原文{number}。")
    run.mt_queue.close()
    pipeline._translate(run)
    published = translations(events)
    assert [event["segment_id"] for event in published] == list(map(str, range(count)))
    assert [event["text"] for event in published] == [f"译：原文{number}。" for number in range(count)]
    assert engine.translate_batch.call_count == (1 if count >= 2 else 0)
    assert engine.translate.call_count == (1 if count in (1, 4) else 0)
    for call in engine.translate_batch.call_args_list:
        assert 2 <= len(call.args[0]) <= 3
        assert call.args[1:3] == ("ja", "zh")
        assert call.kwargs["cancellation_event"] is run.cancel


def test_pipeline_batch_never_crosses_direction_boundary(tmp_path):
    pipeline, run, engine, events = setup(tmp_path, codex=True, batch=True)
    for number, direction in enumerate((("ja", "zh"), ("zh", "ja"), ("ja", "zh"))):
        pipeline._enqueue_translation(run, recognition(str(number), direction=direction), f"文本{number}")
    run.mt_queue.close()
    pipeline._translate(run)
    engine.translate_batch.assert_not_called()
    assert engine.translate.call_count == 3
    assert [event["segment_id"] for event in translations(events)] == ["0", "1", "2"]


def test_malformed_batch_falls_back_once_per_sentence_without_duplicates(tmp_path):
    pipeline, run, engine, events = setup(tmp_path, codex=True, batch=True)
    engine.translate_batch.side_effect = CodexBatchResponseError("missing id")
    for number in range(3):
        pipeline._enqueue_translation(run, recognition(str(number)), f"原文{number}。")
    run.mt_queue.close()
    pipeline._translate(run)
    engine.translate_batch.assert_called_once()
    assert [call.args[0] for call in engine.translate.call_args_list] == ["原文0。", "原文1。", "原文2。"]
    assert len(translations(events)) == 3
    assert all(event["text"].startswith("译：") for event in translations(events))


@pytest.mark.parametrize("error", [TimeoutError("network timed out"),
                                    RuntimeError("ChatGPT login missing"),
                                    PermissionError("subscription quota unavailable")])
def test_network_or_auth_batch_failure_does_not_automatically_retry_single_sentences(tmp_path, error):
    pipeline, run, engine, events = setup(tmp_path, codex=True, batch=True)
    engine.translate_batch.side_effect = error
    for number in range(2):
        pipeline._enqueue_translation(run, recognition(str(number)), f"原文{number}。")
    run.mt_queue.close()
    pipeline._translate(run)
    engine.translate_batch.assert_called_once()
    engine.translate.assert_not_called()
    assert [event["segment_id"] for event in translations(events)] == ["0", "1"]
    assert all("失败" in event["text"] for event in translations(events))


def test_language_rejection_after_engine_correction_does_not_expand_batch_retries(tmp_path):
    from fanyiji.translation_output import ChineseOutputError
    pipeline, run, engine, events = setup(tmp_path, codex=True, batch=True)
    engine.translate_batch.side_effect = ChineseOutputError("foreign after correction")
    for number in range(3):
        pipeline._enqueue_translation(run, recognition(str(number)), f"原文{number}。")
    run.mt_queue.close()
    pipeline._translate(run)
    engine.translate_batch.assert_called_once()
    engine.translate.assert_not_called()
    assert len(translations(events)) == 3
    assert all(event["failed"] for event in translations(events))


def test_cancellation_during_batch_fallback_stops_before_next_sentence(tmp_path):
    pipeline, run, engine, events = setup(tmp_path, codex=True, batch=True)
    engine.translate_batch.side_effect = CodexBatchResponseError("missing id")

    def cancel(*args, **kwargs):
        run.cancel.set()
        return "cancelled late result"

    engine.translate.side_effect = cancel
    for number in range(3):
        pipeline._enqueue_translation(run, recognition(str(number)), f"原文{number}。")
    run.mt_queue.close()
    pipeline._translate(run)
    assert engine.translate.call_count == 1
    assert translations(events) == []
    run.mt_queue.close(discard=True)


def begin_pending(pipeline, run):
    first = recognition("a")
    with patch("fanyiji.pipeline.time.monotonic", return_value=10):
        pipeline._accept_text(run, first, "別にそう。", first.samples, Mock())
    assert run.pending_text["system"].expires_at == 25
    return first


def test_queued_contiguous_audio_protects_pending_phrase_past_processing_ttl(tmp_path):
    pipeline, run, _, events = setup(tmp_path)
    first = begin_pending(pipeline, run)
    following = recognition("b", start=first.end_sample)
    run.asr_queue.put("system", following, 1)
    with patch("fanyiji.pipeline.time.monotonic", return_value=26):
        pipeline._flush_pending(run)
    assert "system" in run.pending_text
    assert not run.pending_text["system"].enqueued
    assert not run.mt_queue._queues["system"]
    assert not any(event["kind"] == "subtitle_final" for event in events)
    assert run.asr_queue.get(timeout=0) is following
    quality = Mock()
    quality.recognize.return_value = SimpleNamespace(text="別にそういうことじゃない。", metadata={})
    with patch("fanyiji.pipeline.time.monotonic", return_value=30):
        pipeline._accept_text(run, following, "いうことじゃない。", following.samples, quality)
    queued = drain(run.mt_queue)
    assert len(queued) == 1 and queued[0].revision == 2
    assert queued[0].source_segment_ids == ("a", "b")
    assert queued[0].text == "別にそういうことじゃない。"
    run.asr_queue.close(discard=True)


def test_queued_continuation_avoids_provisional_cloud_request_before_expiry(tmp_path):
    pipeline, run, _, events = setup(tmp_path)
    first = begin_pending(pipeline, run)
    following = recognition("b", start=first.end_sample)
    run.asr_queue.put("system", following, 1)
    for now in (11.3, 13, 24):
        with patch("fanyiji.pipeline.time.monotonic", return_value=now):
            pipeline._flush_pending(run)
    assert not run.pending_text["system"].enqueued
    assert not run.mt_queue._queues["system"]
    assert not any(event["kind"] == "subtitle_final" for event in events)
    assert run.asr_queue.get(timeout=0) is following
    quality = Mock()
    quality.recognize.return_value = SimpleNamespace(text="別にそういうことじゃない。", metadata={})
    with patch("fanyiji.pipeline.time.monotonic", return_value=24):
        pipeline._accept_text(run, following, "いうことじゃない。", following.samples, quality)
    queued = drain(run.mt_queue)
    assert len(queued) == 1 and queued[0].revision == 2
    assert queued[0].source_segment_ids == ("a", "b")
    run.asr_queue.close(discard=True)


@pytest.mark.parametrize("reason", ["none", "generation", "audio_gap", "direction", "reset", "other_channel"])
def test_unrelated_queued_audio_does_not_delay_provisional_translation(tmp_path, reason):
    pipeline, run, _, _ = setup(tmp_path)
    first = begin_pending(pipeline, run)
    fields = {"start": first.end_sample}
    if reason == "generation":
        fields["generation"] = first.generation + 1
    elif reason == "audio_gap":
        fields["start"] += 2 * 16000
    elif reason == "direction":
        fields["direction"] = ("zh", "ja")
    elif reason == "reset":
        fields["reset_reason"] = "capture_restart"
    elif reason == "other_channel":
        fields["channel"] = "mic"
    if reason != "none":
        following = recognition("b", **fields)
        run.asr_queue.put(following.channel, following, 1)
    with patch("fanyiji.pipeline.time.monotonic", return_value=11.3):
        pipeline._flush_pending(run)
    assert run.pending_text["system"].enqueued
    queued = drain(run.mt_queue)
    assert len(queued) == 1 and queued[0].revision == 1
    run.asr_queue.close(discard=True)


@pytest.mark.parametrize("reason", ["reset", "generation", "audio_gap", "audio_span", "extra_ttl", "other_channel", "force"])
def test_continuation_protection_has_reset_audio_and_wall_time_limits(tmp_path, reason):
    pipeline, run, _, events = setup(tmp_path)
    first = begin_pending(pipeline, run)
    fields = {"start": first.end_sample}
    if reason == "reset":
        fields["reset_reason"] = "capture_restart"
        fields["duration"] = 0
    elif reason == "generation":
        fields["generation"] = first.generation + 1
    elif reason == "audio_gap":
        fields["start"] = first.end_sample + 2 * 16000
    elif reason == "audio_span":
        fields["duration"] = 22
    elif reason == "other_channel":
        fields["channel"] = "mic"
    following = recognition("b", **fields)
    run.asr_queue.put(following.channel, following, following.samples.size / 16000)
    now = 55 if reason == "extra_ttl" else 26
    with patch("fanyiji.pipeline.time.monotonic", return_value=now):
        pipeline._flush_pending(run, force=reason == "force")
    assert "system" not in run.pending_text
    queued = drain(run.mt_queue)
    assert len(queued) == 1 and queued[0].revision == 1
    assert queued[0].source_segment_ids == ("a",)
    assert not any(event["kind"] == "sentence_join" for event in events)
    run.asr_queue.close(discard=True)


def test_protected_phrase_still_expires_after_additional_30_seconds(tmp_path):
    pipeline, run, _, events = setup(tmp_path)
    first = begin_pending(pipeline, run)
    following = recognition("b", start=first.end_sample)
    run.asr_queue.put("system", following, 1)
    with patch("fanyiji.pipeline.time.monotonic", return_value=26):
        pipeline._flush_pending(run)
    assert "system" in run.pending_text
    with patch("fanyiji.pipeline.time.monotonic", return_value=55):
        pipeline._flush_pending(run)
    assert "system" not in run.pending_text
    queued = drain(run.mt_queue)
    assert len(queued) == 1 and queued[0].provisional is False
    assert queued[0].text == "別にそう。"
    assert not any(event["kind"] == "subtitle_final" for event in events)
    run.asr_queue.close(discard=True)
