"""Independent local/Codex branches, stage arbitration and resident ownership."""
from contextlib import ExitStack, contextmanager
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from fanyiji.config import AppConfig, SourceConfig
from fanyiji.pipeline import RecognitionJob, SubtitlePipeline, _Run


def config(**changes):
    return AppConfig(speech_only=False, translation_provider="hybrid", **changes)


def engine(provider, **information):
    return SimpleNamespace(backend_info={"backend": "codex_cli" if provider == "codex" else "llama_cpp",
        "model_id": provider + "-actual", "supports_source_context": True, **information},
        supports_mixed_sources=True, preload=Mock(), close=Mock(), cancel_pending=Mock(),
        reset_conversation=Mock(), translate=Mock(return_value=("本地" if provider == "local" else "在线") + "译文"),
        translate_batch=Mock(side_effect=lambda items, *args, **kwargs:
            {item["id"]: "在线译文" for item in items}))


def setup(root):
    events = []
    pipeline = SubtitlePipeline(root, events.append)
    run = pipeline._run = _Run(11, config())
    local, codex = engine("local"), engine("codex")
    pipeline._draft_translator = run.draft_translator = local
    pipeline._translator = run.translator = codex
    pipeline._translators = [codex]
    run.translators = (codex,)
    return pipeline, run, local, codex, events


def source(identifier="a", channel="system", *, generation=0):
    return RecognitionJob(channel, identifier, np.zeros(16000, np.float32), time.monotonic(),
        ("ja", "zh") if channel == "system" else ("zh", "ja"), context_generation=generation)


def enqueue(pipeline, run, **kwargs):
    pipeline._enqueue_translation(run, source(), "今日は晴れです。", **kwargs)
    return run.mt_queue.get(), run.draft_queue.get()


def publish(pipeline, run, translator, job, stage, text="译文", **metadata):
    pipeline._publish_translation(run, job, text,
        **pipeline._translation_identity(run, translator, stage), **metadata)


def visible(events):
    return [event for event in events if event["kind"] == "translation"]


def test_same_source_context_reaches_both_branches_without_local_translation_input(tmp_path):
    pipeline, run, local, codex, events = setup(tmp_path)
    run.source_context.remember("system", ("ja", "zh"), "昨日は雨だった。", now=time.monotonic())
    final, draft = enqueue(pipeline, run)
    assert final.text == draft.text == "今日は晴れです。"
    assert final.context == draft.context and "昨日は雨" in final.context
    assert final.context_generation == draft.context_generation == events[0]["context_generation"]
    pipeline._translation_request(run, local, [draft], stage="draft")
    pipeline._translation_request(run, codex, [final])
    assert local.translate.call_args.args == codex.translate.call_args.args
    assert local.translate.call_args.kwargs["context"] == codex.translate.call_args.kwargs["context"]
    assert "本地译文" not in str(codex.translate.call_args)
    assert [event["translation_stage"] for event in visible(events)] == ["draft", "final"]
    requests = [event for event in events if event["kind"] == "translation_request"]
    assert [(event["translation_provider"], event["translation_model"]) for event in requests] == [
        ("local", "local-actual"), ("codex", "codex-actual")]


def test_codex_first_only_renders_final_and_records_late_local_candidate(tmp_path):
    pipeline, run, local, codex, events = setup(tmp_path)
    final, draft = enqueue(pipeline, run)
    publish(pipeline, run, codex, final, "final", "终稿")
    publish(pipeline, run, local, draft, "draft", "暂译")
    assert [event["text"] for event in visible(events)] == ["终稿"]
    candidate = next(event for event in events if event["kind"] == "translation_candidate")
    assert candidate["text"] == "暂译" and candidate["translation_provider"] == "local"


@pytest.mark.parametrize("final_first", [False, True])
def test_codex_failure_preserves_local_draft_in_both_return_orders(tmp_path, final_first):
    pipeline, run, local, codex, events = setup(tmp_path)
    final, draft = enqueue(pipeline, run)
    steps = [(codex, final, "final", "", {"failed": True, "error": "network"}),
             (local, draft, "draft", "有效暂译", {})]
    for translator, job, stage, text, metadata in steps if final_first else reversed(steps):
        publish(pipeline, run, translator, job, stage, text, **metadata)
    assert [event["text"] for event in visible(events)] == ["有效暂译"]
    status = next(event for event in events if event["kind"] == "translation_status")
    assert status["overall_failed"] is False and status["final_failed"] is True
    assert status["has_translation"] is (not final_first)
    if final_first:
        assert visible(events)[0]["final_failed"] is True


@pytest.mark.parametrize("local_first", [False, True])
def test_local_failure_cannot_damage_successful_codex_final(tmp_path, local_first):
    pipeline, run, local, codex, events = setup(tmp_path)
    final, draft = enqueue(pipeline, run)
    steps = [(local, draft, "draft", "", {"failed": True}), (codex, final, "final", "终稿", {})]
    for translator, job, stage, text, metadata in steps if local_first else reversed(steps):
        publish(pipeline, run, translator, job, stage, text, **metadata)
    assert [event["text"] for event in visible(events)] == ["终稿"]
    assert all(not event["overall_failed"] for event in events if event["kind"] == "translation_status")


@pytest.mark.parametrize("first_stage", ["draft", "final"])
def test_both_failures_finish_pending_with_one_failure_placeholder(tmp_path, first_stage):
    pipeline, run, local, codex, events = setup(tmp_path)
    final, draft = enqueue(pipeline, run)
    by_stage = {"draft": (local, draft), "final": (codex, final)}
    for stage in (first_stage, "draft" if first_stage == "final" else "final"):
        translator, job = by_stage[stage]
        publish(pipeline, run, translator, job, stage, "", failed=True, error="synthetic")
    statuses = [event for event in events if event["kind"] == "translation_status"]
    assert [event["overall_failed"] for event in statuses] == [False, True]
    assert len(visible(events)) == 1 and visible(events)[0]["overall_failed"] is True
    assert visible(events)[0]["translation_stage"] == "final"


@pytest.mark.parametrize("stage", ["draft", "final"])
def test_old_revision_does_not_render_or_mark_new_revision_failed(tmp_path, stage):
    pipeline, run, local, codex, events = setup(tmp_path)
    previous, _ = enqueue(pipeline, run)
    current, _ = enqueue(pipeline, run, revision=2)
    translator = local if stage == "draft" else codex
    publish(pipeline, run, translator, previous, stage, "旧译文")
    publish(pipeline, run, translator, previous, stage, "", failed=True)
    publish(pipeline, run, codex, current, "final", "新终稿")
    assert [event["text"] for event in visible(events)] == ["新终稿"]
    assert not any(event["kind"] == "translation_status" for event in events)


def test_same_revision_replaced_source_generation_rejects_old_draft_and_failure(tmp_path):
    pipeline, run, local, codex, events = setup(tmp_path)
    previous, _ = enqueue(pipeline, run)
    pipeline._enqueue_translation(run, source(generation=2), "明日は曇りです。")
    current = run.mt_queue.get()
    publish(pipeline, run, local, previous, "draft", "旧暂译")
    publish(pipeline, run, codex, previous, "final", "", failed=True)
    publish(pipeline, run, codex, current, "final", "新终稿")
    assert [event["text"] for event in visible(events)] == ["新终稿"]
    assert not any(event["kind"] == "translation_status" for event in events)


def test_new_context_generation_on_other_source_does_not_expire_accepted_old_source(tmp_path):
    pipeline, run, local, codex, events = setup(tmp_path)
    previous, _ = enqueue(pipeline, run)
    pipeline._clear_source_context(run, "system")
    pipeline._enqueue_translation(run, source("b", generation=1), "おはようございます。")
    publish(pipeline, run, codex, previous, "final", "旧片段仍保留")
    assert visible(events)[0]["text"] == "旧片段仍保留"


@pytest.mark.parametrize("stage", ["draft", "final"])
def test_protocol_pollution_is_branch_failure_without_erasing_other_translation(tmp_path, stage):
    pipeline, run, local, codex, events = setup(tmp_path)
    final, draft = enqueue(pipeline, run)
    translator, job = (local, draft) if stage == "draft" else (codex, final)
    publish(pipeline, run, translator, job, stage, "语言 None<asr_text>")
    publish(pipeline, run, codex if stage == "draft" else local, final if stage == "draft" else draft,
        "final" if stage == "draft" else "draft", "有效译文")
    assert [event["text"] for event in visible(events)] == ["有效译文"]
    status = next(event for event in events if event["kind"] == "translation_status")
    assert status["protocol_rejected"] is True and not status["overall_failed"]


@pytest.mark.parametrize("stage", ["draft", "final"])
def test_foreign_output_cannot_render_or_erase_other_chinese_branch(tmp_path, stage):
    pipeline, run, local, codex, events = setup(tmp_path)
    final, draft = enqueue(pipeline, run)
    translator, job = (local, draft) if stage == "draft" else (codex, final)
    publish(pipeline, run, translator, job, stage, "レレン 谢谢你的留言")
    publish(pipeline, run, codex if stage == "draft" else local, final if stage == "draft" else draft,
        "final" if stage == "draft" else "draft", "蕾莲 谢谢你的留言")
    assert [event["text"] for event in visible(events)] == ["蕾莲 谢谢你的留言"]
    status = next(event for event in events if event["kind"] == "translation_status")
    assert status["language_rejected"] is True and not status["overall_failed"]


def test_real_workers_enter_both_independent_engines_before_either_finishes(tmp_path):
    pipeline, run, local, codex, events = setup(tmp_path)
    entered_local, entered_codex, release = threading.Event(), threading.Event(), threading.Event()
    def translate(label, entered):
        def call(*args, **kwargs):
            entered.set()
            assert release.wait(3)
            return label
        return call
    local.translate.side_effect = translate("暂译", entered_local)
    codex.translate.side_effect = translate("终稿", entered_codex)
    pipeline._enqueue_translation(run, source(), "今日は晴れです。")
    run.mt_queue.close()
    run.draft_queue.close()
    workers = [threading.Thread(target=pipeline._translate, args=(run, codex, "system")),
               threading.Thread(target=pipeline._translate_drafts, args=(run, local))]
    for worker in workers:
        worker.start()
    try:
        assert entered_local.wait(1) and entered_codex.wait(1)
    finally:
        release.set()
        for worker in workers:
            worker.join(2)
            assert not worker.is_alive()
    assert any(event["text"] == "终稿" for event in visible(events))


def test_one_local_worker_fairly_serves_both_channels_while_codex_batches_separately(tmp_path):
    pipeline, run, local, codex, events = setup(tmp_path)
    codex.backend_info["supports_batch"] = True
    calls = []
    local.translate.side_effect = lambda text, *args, **kwargs: calls.append(text) or "暂译"
    for channel in ("mic", "system"):
        for number in range(3):
            pipeline._enqueue_translation(run, source(f"{channel}{number}", channel), f"{channel}{number}")
    run.mt_queue.close()
    run.draft_queue.close()
    pipeline._translate_drafts(run, local)
    assert calls == ["mic0", "system0", "mic1", "system1", "mic2", "system2"]
    pipeline._translate(run, codex, "system")
    pipeline._translate(run, codex, "mic")
    assert local.translate_batch.call_count == 0 and codex.translate_batch.call_count == 2
    assert len(visible(events)) == 12
    assert sum(event["translation_stage"] == "final" for event in visible(events)) == 6


def test_stop_cancels_inflight_local_without_closing_idle_resident_model(tmp_path):
    pipeline, run, local, codex, events = setup(tmp_path)
    entered, release = threading.Event(), threading.Event()
    def translate(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        return "取消后的晚到暂译"
    local.translate.side_effect = translate
    local.cancel_pending.side_effect = release.set
    final, draft = enqueue(pipeline, run)
    worker = threading.Thread(target=pipeline._translation_request, args=(run, local, [draft]),
        kwargs={"stage": "draft"})
    run.threads.append(worker)
    worker.start()
    assert entered.wait(1)
    pipeline.stop()
    assert not worker.is_alive()
    local.cancel_pending.assert_called_once()
    local.close.assert_not_called()
    assert not visible(events)
    codex.cancel_pending.assert_called_once()
    assert run.mt_queue.get() is None and run.draft_queue.get() is None
    idle = pipeline._run = _Run(12, config())
    idle.draft_translator = local
    idle.translators = (codex,)
    pipeline.stop()
    local.cancel_pending.assert_called_once()
    pipeline.shutdown()
    local.close.assert_called_once()
    codex.close.assert_called_once()


def test_actual_local_model_file_is_identity_fallback(tmp_path):
    pipeline, run, local, _, _ = setup(tmp_path)
    local.backend_info.pop("model_id")
    local.backend_info["model_file"] = "HY-MT2-7B-Q6_K.gguf"
    assert pipeline._translation_identity(run, local, "draft")["translation_model"] == "HY-MT2-7B-Q6_K.gguf"


def test_cancellation_aware_local_engine_receives_the_run_event(tmp_path):
    pipeline, run, local, _, _ = setup(tmp_path)
    local.supports_cancellation = True
    _, draft = enqueue(pipeline, run)
    pipeline._translation_request(run, local, [draft], stage="draft")
    assert local.translate.call_args.kwargs["cancellation_event"] is run.cancel
    assert run.draft_active == 0


def test_asr_provisional_meaning_is_preserved_for_both_translation_stages(tmp_path):
    pipeline, run, local, codex, events = setup(tmp_path)
    final, draft = enqueue(pipeline, run, provisional=True)
    publish(pipeline, run, local, draft, "draft", "暂译")
    publish(pipeline, run, codex, final, "final", "终稿")
    assert all(event["provisional"] is True for event in visible(events))


def test_both_queues_coalesce_source_revision_before_inference(tmp_path):
    pipeline, run, _, _, _ = setup(tmp_path)
    pipeline._enqueue_translation(run, source(), "途中なので", provisional=True)
    pipeline._enqueue_translation(run, source(), "途中なので待って。", revision=2, source_ids=("a", "b"))
    run.mt_queue.close()
    run.draft_queue.close()
    for queue in (run.mt_queue, run.draft_queue):
        job = queue.get()
        assert job.revision == 2 and job.text == "途中なので待って。"
        assert job.source_segment_ids == ("a", "b") and queue.get() is None


@contextmanager
def initialized(root, *, local_failure=False, codex_failure=False, dual=True, local_mixed=True):
    for relative in ("asr/model.int8.onnx", "asr/tokens.txt", "vad/silero_vad.onnx"):
        file = root / relative
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(b"test asset")
    local, primary, secondary = engine("local"), engine("codex"), engine("codex")
    local.supports_mixed_sources = local_mixed
    if local_failure:
        local.preload.side_effect = RuntimeError("local failed")
    if codex_failure:
        primary.preload.side_effect = RuntimeError("codex failed")
    events = []
    pipeline = SubtitlePipeline(root, events.append)
    selected = config(microphone=SourceConfig(enabled=dual))
    asr = SimpleNamespace(preload=Mock(), close=Mock(), backend_info={"model_id": "asr-test"})
    with ExitStack() as stack:
        stack.enter_context(patch("fanyiji.engines.RecognizerEngine", return_value=asr))
        local_factory = stack.enter_context(patch("fanyiji.engines.create_translation_engine", return_value=local))
        codex_factory = stack.enter_context(patch("fanyiji.codex_engine.CodexTranslationEngine",
            side_effect=[primary, secondary]))
        capture = stack.enter_context(patch("fanyiji.audio.AudioCaptureManager"))
        stack.enter_context(patch("fanyiji.segmentation.SpeechSegmenter"))
        stack.enter_context(patch.object(pipeline, "_recognize"))
        final_workers = stack.enter_context(patch.object(pipeline, "_translate"))
        draft_workers = stack.enter_context(patch.object(pipeline, "_translate_drafts"))
        try:
            run = pipeline._run = _Run(1, selected)
            pipeline._initialize(run)
            for worker in run.threads:
                worker.join(2)
            yield SimpleNamespace(pipeline=pipeline, run=run, local=local, primary=primary, secondary=secondary,
                local_factory=local_factory, codex_factory=codex_factory, capture=capture,
                final_workers=final_workers, draft_workers=draft_workers, events=events, asr=asr)
        finally:
            pipeline.shutdown()


def test_initialize_has_one_local_model_and_two_independent_codex_workers_reused_on_resume(tmp_path):
    with initialized(tmp_path) as harness:
        assert harness.run.translators == (harness.primary, harness.secondary)
        assert harness.run.draft_translator is harness.local
        assert harness.final_workers.call_count == 2 and harness.draft_workers.call_count == 1
        ready = next(event for event in harness.events if event["kind"] == "models_ready")
        info = ready["models"]["translation"]
        assert info["mode"] == "hybrid" and info["worker_count"] == 2 and info["draft_worker_count"] == 1
        assert info["draft_model"]["model_id"] == "local-actual"
        harness.pipeline.stop()
        resumed = harness.pipeline._run = _Run(2, harness.run.config)
        harness.pipeline._initialize(resumed)
        for worker in resumed.threads:
            worker.join(2)
        assert resumed.draft_translator is harness.local
        assert harness.local_factory.call_count == 1 and harness.codex_factory.call_count == 2
        harness.local.close.assert_not_called()
        harness.local.cancel_pending.assert_not_called()


@pytest.mark.parametrize(("local_failure", "codex_failure"), [(True, False), (False, True), (True, True)])
def test_initialization_failure_preserves_the_other_branch_or_stops_if_both_fail(tmp_path, local_failure, codex_failure):
    with initialized(tmp_path, local_failure=local_failure, codex_failure=codex_failure) as harness:
        if local_failure and codex_failure:
            harness.capture.assert_not_called()
            assert any(event["kind"] == "models_failed" for event in harness.events)
            assert not any(event["kind"] == "models_ready" for event in harness.events)
        else:
            ready = next(event for event in harness.events if event["kind"] == "models_ready")
            info = ready["models"]["translation"]
            assert info["available"] is (not codex_failure)
            assert info["draft_worker_count"] == (not local_failure)
            harness.capture.return_value.start.assert_called_once()
            harness.pipeline._enqueue_translation(harness.run, source(), "今日は晴れです。")
            status = next(event for event in harness.events if event["kind"] == "translation_status")
            assert status["error"] == "engine_unavailable" and not status["overall_failed"]
            assert status["translation_provider"] == ("local" if local_failure else "codex")
        harness.pipeline.shutdown()
        harness.local.close.assert_called_once()
        harness.primary.close.assert_called_once()


def test_queue_status_distinguishes_final_and_local_backlog(tmp_path):
    pipeline, run, _, _, events = setup(tmp_path)
    pipeline._enqueue_translation(run, source(), "今日は晴れです。")
    queue = next(event for event in events if event["kind"] == "translation_queue")
    assert queue["pending"] == 1 and queue["draft_queue"]["pending"] == 1
    run.mt_queue.get()
    run.last_queue_report = 0
    pipeline._report_translation_queue(run)
    assert events[-1]["pending"] == 0 and events[-1]["draft_queue"]["pending"] == 1


def test_primary_failure_on_resume_closes_previous_secondary_without_losing_local_model(tmp_path):
    with initialized(tmp_path) as harness:
        harness.pipeline.stop()
        harness.primary.preload.side_effect = RuntimeError("account unavailable on resume")
        resumed = harness.pipeline._run = _Run(2, harness.run.config)
        harness.pipeline._initialize(resumed)
        for worker in resumed.threads:
            worker.join(2)
        assert resumed.translators == () and resumed.draft_translator is harness.local
        harness.secondary.close.assert_called_once()
        harness.local.close.assert_not_called()
        assert any(event["kind"] == "models_ready" and event["session"] == 2 for event in harness.events)


def test_second_factory_failure_still_starts_local_and_primary_workers(tmp_path):
    with initialized(tmp_path, dual=False) as harness:
        harness.pipeline.stop()
        harness.codex_factory.side_effect = RuntimeError("could not launch second helper")
        resumed = harness.pipeline._run = _Run(2, config())
        harness.pipeline._initialize(resumed)
        for worker in resumed.threads:
            worker.join(2)
        assert resumed.translators == (harness.primary,) and resumed.draft_translator is harness.local
        assert any(event["kind"] == "models_ready" and event["session"] == 2 for event in harness.events)


def test_failed_fixed_source_local_engine_does_not_disable_successful_codex_multilingual_asr(tmp_path):
    with initialized(tmp_path, local_failure=True, local_mixed=False) as harness:
        assert harness.run.multilingual_recognition is True
        harness.asr.preload.assert_called_once_with(("auto",))


def test_stop_cancels_published_local_preload_and_releases_model_loader_lock_for_resume(tmp_path):
    for relative in ("asr/model.int8.onnx", "asr/tokens.txt", "vad/silero_vad.onnx"):
        file = tmp_path / relative
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(b"test asset")
    local, codex = engine("local"), engine("codex")
    local.supports_cancellation = True
    entered, released = threading.Event(), threading.Event()
    def preload(*, cancellation_event):
        entered.set()
        assert released.wait(3)
        assert cancellation_event.is_set()
        raise RuntimeError("cancelled loading")
    local.preload.side_effect = preload
    local.cancel_pending.side_effect = released.set
    events = []
    pipeline = SubtitlePipeline(tmp_path, events.append)
    selected = config(microphone=SourceConfig(enabled=False))
    asr = SimpleNamespace(preload=Mock(), close=Mock(), backend_info={"model_id": "asr-test"})
    with patch("fanyiji.engines.RecognizerEngine", return_value=asr), patch(
        "fanyiji.engines.create_translation_engine", return_value=local), patch(
        "fanyiji.codex_engine.CodexTranslationEngine", return_value=codex), patch(
        "fanyiji.audio.AudioCaptureManager") as capture, patch(
        "fanyiji.segmentation.SpeechSegmenter"), patch.object(pipeline, "_recognize"), patch.object(
        pipeline, "_translate"), patch.object(pipeline, "_translate_drafts"):
        try:
            pipeline.start(selected, 1)
            run = pipeline._run
            assert entered.wait(1)
            assert run.draft_active == 1 and run.draft_translator is local
            pipeline.stop()
            local.cancel_pending.assert_called_once()
            assert run.draft_active == 0
            assert not any(worker.is_alive() for worker in run.threads)
            assert pipeline._model_lock.acquire(timeout=0.2)
            pipeline._model_lock.release()
            capture.assert_not_called()
            local.close.assert_not_called()
            local.preload.side_effect = None
            resumed = pipeline._run = _Run(2, selected)
            pipeline._initialize(resumed)
            for worker in resumed.threads:
                worker.join(2)
            local.preload.assert_called_with(cancellation_event=resumed.cancel)
            capture.return_value.start.assert_called_once()
        finally:
            released.set()
            pipeline.shutdown()
