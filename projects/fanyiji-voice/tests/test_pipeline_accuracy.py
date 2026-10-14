"""Phrase assembly, provenance and cancellation boundaries without inference."""
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from fanyiji.config import AppConfig
from fanyiji.pipeline import FairQueue, RecognitionJob, SubtitlePipeline, TranslationJob, _Run


def job(identifier, start, seconds=1, *, channel="system", generation=1, direction=("ja", "zh")):
    samples = np.full(round(seconds * 16000), .1, np.float32)
    return RecognitionJob(channel, identifier, samples, 100, direction, start,
                          start + samples.size, generation, "silence")


def setup(tmp_path):
    events = []
    return SubtitlePipeline(tmp_path, events.append), _Run(1, AppConfig()), events


def drain(queue):
    queue.close()
    items = []
    while (item := queue.get()) is not None:
        items.append(item)
    return items


def test_natural_pause_clauses_translate_separately_with_previous_source_context(tmp_path):
    pipeline, run, events = setup(tmp_path)
    first, second = job("a", 0, 6), job("b", 96000, 5)
    first.ended_by = second.ended_by = "natural_pause"
    quality = Mock()
    pipeline._accept_text(run, first, "最初に説明しますが。", first.samples, quality)
    assert not run.pending_text
    pipeline._accept_text(run, second, "続いて次の話題です。", second.samples, quality)
    queued = drain(run.mt_queue)
    assert [entry.segment_id for entry in queued] == ["a", "b"]
    assert all(entry.revision == 1 and not entry.provisional for entry in queued)
    assert queued[1].context == "最初に説明しますが。"
    quality.recognize.assert_not_called()
    assert not any(event["kind"] == "sentence_join" for event in events)


def test_natural_pause_finishes_prior_short_pending_fragment_without_rejoining(tmp_path):
    pipeline, run, events = setup(tmp_path)
    first, second = job("a", 0), job("b", 16000, 6)
    second.ended_by = "natural_pause"
    quality = Mock()
    pipeline._accept_text(run, first, "森の。", first.samples, quality)
    pipeline._accept_text(run, second, "妖精みたいな名前ですが。", second.samples, quality)
    queued = drain(run.mt_queue)
    assert [entry.text for entry in queued] == ["森の。", "妖精みたいな名前ですが。"]
    assert not run.pending_text
    quality.recognize.assert_not_called()
    assert not any(event["kind"] == "sentence_join" for event in events)


def test_timeout_distinguishes_open_idle_queue_from_closed_queue():
    queue = FairQueue()
    with pytest.raises(TimeoutError):
        queue.get(timeout=0)
    queue.close()
    assert queue.get(timeout=0) is None


def test_short_contiguous_phrase_rechecks_whole_pcm_and_updates_one_row(tmp_path):
    pipeline, run, events = setup(tmp_path)
    first, second = job("a", 0, 2), job("b", 32000, 1)
    quality = Mock()
    quality.recognize.return_value = SimpleNamespace(text="森の妖精みたいな名前です。", metadata={})
    pipeline._accept_text(run, first, "森の。", first.samples, quality)
    assert not run.mt_queue._queues["system"]
    pipeline._accept_text(run, second, "妖精みたいな名前です。", second.samples, quality)
    translations = drain(run.mt_queue)
    assert len(translations) == 1
    assert translations[0].segment_id == "a" and translations[0].revision == 2
    assert translations[0].source_segment_ids == ("a", "b")
    assert translations[0].context == ""
    np.testing.assert_array_equal(quality.recognize.call_args.args[0],
                                  np.concatenate((first.samples, second.samples)))
    assert [event["segment_id"] for event in events if event["kind"] == "original"] == ["a", "a"]


@pytest.mark.parametrize("change", ["generation", "gap", "direction", "question", "reply"])
def test_unrelated_or_discontinuous_text_is_never_re_recognized_or_merged(tmp_path, change):
    pipeline, run, _ = setup(tmp_path)
    first, second = job("a", 0, 2), job("b", 32000)
    text = "森の。"
    next_text = "妖精です。"
    if change == "generation":
        second.generation = 2
    elif change == "gap":
        second.start_sample += 48000
        second.end_sample += 48000
    elif change == "direction":
        second.direction = ("zh", "ja")
        next_text = "天气很好。"
    elif change == "question":
        text = "明日は行くの？"
    else:
        next_text = "いいえ。"
    quality = Mock()
    pipeline._accept_text(run, first, text, first.samples, quality)
    pipeline._accept_text(run, second, next_text, second.samples, quality)
    pipeline._flush_pending(run, force=True)
    translations = drain(run.mt_queue)
    assert [item.segment_id for item in translations] == ["a", "b"]
    assert all(len(item.source_segment_ids) == 1 for item in translations)
    quality.recognize.assert_not_called()


def test_channel_context_and_pending_are_independent(tmp_path):
    pipeline, run, _ = setup(tmp_path)
    quality = Mock()
    pipeline._accept_text(run, job("a", 0), "森の。", np.zeros(16000, np.float32), quality)
    microphone = job("mic", 0, channel="mic", direction=("zh", "ja"))
    pipeline._accept_text(run, microphone, "今天很热。", microphone.samples, quality)
    pipeline._flush_pending(run, force=True)
    translations = drain(run.mt_queue)
    assert all(item.context == "" for item in translations)
    assert all(len(item.source_segment_ids) == 1 for item in translations)
    quality.recognize.assert_not_called()


def test_pending_timeout_translates_once_then_confirms_without_second_request(tmp_path):
    pipeline, run, events = setup(tmp_path)
    first = job("a", 0)
    with patch("fanyiji.pipeline.time.monotonic", return_value=10):
        pipeline._accept_text(run, first, "森の。", first.samples, Mock())
    with patch("fanyiji.pipeline.time.monotonic", return_value=11.3):
        pipeline._flush_pending(run)
        pipeline._flush_pending(run)
    with patch("fanyiji.pipeline.time.monotonic", return_value=26):
        pipeline._flush_pending(run)
    translations = drain(run.mt_queue)
    assert len(translations) == 1 and translations[0].provisional
    assert len([event for event in events if event["kind"] == "subtitle_final"]) == 1
    assert not run.pending_text


def test_translation_context_is_copied_before_current_text_is_remembered(tmp_path):
    pipeline, run, _ = setup(tmp_path)
    first, second = job("a", 0), job("b", 16000)
    pipeline._accept_text(run, first, "田中さんが来ました。", first.samples, Mock())
    pipeline._accept_text(run, second, "彼はまだ帰りません。", second.samples, Mock())
    translations = drain(run.mt_queue)
    assert translations[0].context == ""
    assert translations[1].context == "田中さんが来ました。"
    assert "帰りません" not in translations[1].context


def test_translation_worker_passes_context_only_to_capable_engine(tmp_path):
    for capable in (False, True):
        pipeline, run, events = setup(tmp_path)
        translator = pipeline._translator = Mock()
        translator.backend_info = {"supports_source_context": capable}
        translator.translate.return_value = "他还没回去。"
        translated = TranslationJob("system", "a", "彼はまだ帰りません。", 100, ("ja", "zh"),
                                    1, context="田中さんが来ました。", revision=2)
        run.mt_queue.put("system", translated, 1)
        run.mt_queue.close()
        pipeline._translate(run)
        assert translator.translate.call_args.kwargs == ({"context": translated.context} if capable else {})
        translated_event = next(event for event in events if event["kind"] == "translation")
        assert translated_event["revision"] == 2


def test_cancelled_phrase_assembly_cannot_publish_or_decode(tmp_path):
    pipeline, run, events = setup(tmp_path)
    run.cancel.set()
    first = job("a", 0)
    pipeline._accept_text(run, first, "こんにちは。", first.samples, Mock())
    assert events == []
    assert not run.mt_queue._queues["system"]


def test_missing_source_segment_finalizes_pending_and_clears_context(tmp_path):
    pipeline, run, _ = setup(tmp_path)
    first = job("a", 0)
    pipeline._accept_text(run, first, "森の。", first.samples, Mock())
    pipeline._break_context(run, "system")
    assert not run.pending_text
    assert run.source_context.get("system", ("ja", "zh")) == ""
    assert drain(run.mt_queue)[0].text == "森の。"


def test_gap_without_pcm_does_not_invent_silence_for_joint_recognition(tmp_path):
    pipeline, run, events = setup(tmp_path)
    first, second = job("a", 0), job("b", 20000)
    quality = Mock()
    pipeline._accept_text(run, first, "森の。", first.samples, quality)
    pipeline._accept_text(run, second, "妖精です。", second.samples, quality)
    quality.recognize.assert_not_called()
    joined = next(event for event in events if event["kind"] == "sentence_join")
    assert joined["gap_samples"] == 4000 and joined["gap_filled"] is False
    assert joined["joint_recognition"]["reason"] == "audio_gap_not_recorded"


def test_uncertain_joint_recheck_does_not_replace_source_text(tmp_path):
    pipeline, run, _ = setup(tmp_path)
    first, second = job("a", 0), job("b", 16000)
    quality = Mock()
    quality.recognize.return_value = SimpleNamespace(text="異なる結果。", metadata={"uncertain": True})
    pipeline._accept_text(run, first, "森の。", first.samples, quality)
    pipeline._accept_text(run, second, "妖精です。", second.samples, quality)
    assert drain(run.mt_queue)[0].text == "森の妖精です。"


def test_text_only_continuation_removes_asr_sentence_boundary_and_preserves_raw_sources(tmp_path):
    pipeline, run, events = setup(tmp_path)
    first, second = job("a", 0), job("b", 20000)
    second.recognition_uncertain = True
    quality = Mock()
    pipeline._accept_text(run, first, "彼女は今日は。", first.samples, quality)
    pipeline._accept_text(run, second, "歌わないと言いました。", second.samples, quality)
    translated = drain(run.mt_queue)[0]
    assert translated.text == "彼女は今日は歌わないと言いました。"
    assert translated.source_segment_ids == ("a", "b")
    quality.recognize.assert_not_called()
    joined = next(event for event in events if event["kind"] == "sentence_join")
    assert joined["raw_source_texts"] == ("彼女は今日は。", "歌わないと言いました。")
    assert joined["boundary_normalized"] is True
    assert next(event for event in reversed(events) if event["kind"] == "original")["uncertain"] is True


@pytest.mark.parametrize("middle", ["", RuntimeError("native failed")])
def test_actual_worker_breaks_phrase_and_context_on_empty_or_failed_recognition(tmp_path, middle):
    pipeline, run, events = setup(tmp_path)
    run.config = AppConfig(speech_only=False)
    pipeline._recognizer = Mock()
    pipeline._recognizer.recognize.side_effect = ["森の。", middle, "妖精です。"]
    for identifier, start in (("a", 0), ("missing", 16000), ("b", 32000)):
        item = job(identifier, start)
        if identifier == "missing":
            item.samples[:] = 0
        run.asr_queue.put("system", item, 1)
    run.asr_queue.close()
    pipeline._recognize(run)
    translations = drain(run.mt_queue)
    assert [item.segment_id for item in translations] == ["a", "b"]
    assert translations[1].context == ""
    assert not any(event["kind"] == "sentence_join" for event in events)
    assert pipeline._recognizer.recognize.call_count == 3


@pytest.mark.parametrize("joint_result", [
    SimpleNamespace(text="異なる結果。", metadata={"uncertain": True}),
    RuntimeError("joint decode failed"),
])
def test_joint_uncertainty_is_visible_and_cannot_become_next_source_context(tmp_path, joint_result):
    pipeline, run, events = setup(tmp_path)
    first, second, third = job("a", 0), job("b", 16000), job("c", 32000)
    quality = Mock()
    if isinstance(joint_result, Exception):
        quality.recognize.side_effect = joint_result
    else:
        quality.recognize.return_value = joint_result
    pipeline._accept_text(run, first, "森の。", first.samples, quality)
    pipeline._accept_text(run, second, "妖精です。", second.samples, quality)
    pipeline._accept_text(run, third, "今日は晴れです。", third.samples, quality)
    translated = drain(run.mt_queue)
    assert translated[0].text == "森の妖精です。"
    assert translated[1].context == ""
    joined_original = next(event for event in events if event["kind"] == "original" and event["revision"] == 2)
    assert joined_original["uncertain"] is True
    assert next(event for event in events if event["kind"] == "sentence_join")["uncertain"] is True


@pytest.mark.parametrize("first_text", ["彼は陽性です。", "森の。"])
def test_uncertain_source_does_not_enter_background_even_after_pending_confirmation(tmp_path, first_text):
    pipeline, run, _ = setup(tmp_path)
    first, second = job("a", 0), job("b", 16000)
    first.recognition_uncertain = True
    pipeline._accept_text(run, first, first_text, first.samples, Mock())
    pipeline._flush_pending(run, force=True)
    pipeline._accept_text(run, second, "それは誤解です。", second.samples, Mock())
    translated = drain(run.mt_queue)
    assert translated[1].context == ""


@pytest.mark.parametrize("pending", [False, True])
def test_backlog_cannot_refresh_an_old_source_context_by_processing_time(tmp_path, pending):
    pipeline, run, _ = setup(tmp_path)
    first, second = job("a", 0), job("b", 480000)
    second.ended_at = first.ended_at + 30
    with patch("fanyiji.pipeline.time.monotonic", return_value=1000):
        pipeline._accept_text(run, first, "森の。" if pending else "田中さんが来ました。", first.samples, Mock())
        pipeline._flush_pending(run, force=True)
        pipeline._accept_text(run, second, "彼はもう帰りました。", second.samples, Mock())
    translated = drain(run.mt_queue)
    assert translated[1].context == ""
