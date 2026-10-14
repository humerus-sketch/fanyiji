"""Domestic account-CLI routing without starting a CLI, network or real model."""
from contextlib import ExitStack, contextmanager
from dataclasses import replace
import sys
import threading
import time
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, call, patch

import numpy as np
import pytest

from fanyiji.config import AppConfig
from fanyiji.pipeline import RecognitionJob, SubtitlePipeline, TranslationJob, _Run


CLI_PROVIDERS = ("minimax", "minimax_code", "kimi", "codebuddy", "trae", "qwen")


def test_final_identity_updates_only_after_client_confirms_actual_model(tmp_path):
    translator = engine(model="")
    with harness(tmp_path, domestic=translator) as h:
        run = initialize(h, config(domestic_cli_model="selected-but-not-yet-observed"))
        assert h.pipeline._translation_identity(run, translator, "final")["translation_model"] == "unknown"
        def translated(*args, **kwargs):
            translator.backend_info["model_id"] = "actual-observed-model"
            return "已经翻译完成"
        translator.translate.side_effect = translated
        h.pipeline._translation_request(run, translator, [job()])
        request = next(event for event in h.events if event["kind"] == "translation_request")
        backend = next(event for event in h.events if event["kind"] == "translation_backend")
        assert request["translation_model"] == "actual-observed-model"
        assert backend["information"]["model_id"] == "actual-observed-model"


def test_unreported_model_is_never_replaced_by_saved_cli_selection(tmp_path):
    translator = engine(model="", cli="kimi")
    with harness(tmp_path, domestic=translator) as h:
        run = initialize(h, config(domestic_cli_provider="kimi", domestic_cli_model="selected-only"))
        h.pipeline._translation_request(run, translator, [job()])
        request = next(event for event in h.events if event["kind"] == "translation_request")
        assert request["translation_model"] == "unknown"


def engine(model="actual-domestic-model", cli="minimax", *, local=False, codex=False):
    information = {"backend": "fake_local" if local else "codex_cli" if codex else "domestic_cli",
                   "model_id": model, "supports_source_context": True,
                   "supports_batch": codex}
    if not local and not codex:
        information.update(provider="domestic_cli", cli_provider=cli)
    return SimpleNamespace(backend_info=information, supports_mixed_sources=True,
                           supports_cancellation=not codex, preload=Mock(), close=Mock(),
                           cancel_pending=Mock(), translate=Mock(return_value="已经翻译完成"),
                           translate_batch=Mock())


def config(*, hybrid=False, **options):
    return AppConfig(translation_provider="hybrid" if hybrid else "domestic_cli",
                     hybrid_final_provider="domestic_cli" if hybrid else "codex",
                     speech_only=False, audio_processing="off", **options)


@contextmanager
def harness(root, *, domestic=None, local=None, codex_engines=None):
    for name in ("asr/model.int8.onnx", "asr/tokens.txt", "vad/silero_vad.onnx"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic local asset")
    module = ModuleType("fanyiji.domestic_cli_engine")
    module.DomesticCliTranslationEngine = Mock(return_value=domestic or engine())
    events = []
    asr = SimpleNamespace(preload=Mock(), close=Mock(), backend_info={"model_id": "fake-asr"})
    with ExitStack() as stack:
        stack.enter_context(patch.dict(sys.modules, {module.__name__: module}))
        stack.enter_context(patch("fanyiji.engines.RecognizerEngine", return_value=asr))
        local_factory = stack.enter_context(patch("fanyiji.engines.create_translation_engine",
                                                return_value=local or engine("local-draft", local=True)))
        codex_factory = stack.enter_context(patch("fanyiji.codex_engine.CodexTranslationEngine",
                                                side_effect=codex_engines))
        web_factory = stack.enter_context(patch("fanyiji.web_translation_engine.GoogleWebTranslationEngine"))
        capture = stack.enter_context(patch("fanyiji.audio.AudioCaptureManager"))
        stack.enter_context(patch("fanyiji.segmentation.SpeechSegmenter"))
        pipeline = SubtitlePipeline(root, events.append)
        stack.enter_context(patch.object(pipeline, "_recognize"))
        final_worker = stack.enter_context(patch.object(pipeline, "_translate"))
        draft_worker = stack.enter_context(patch.object(pipeline, "_translate_drafts"))
        try:
            yield SimpleNamespace(pipeline=pipeline, factory=module.DomesticCliTranslationEngine,
                                  local_factory=local_factory, codex_factory=codex_factory,
                                  web_factory=web_factory, capture=capture, asr=asr,
                                  final_worker=final_worker, draft_worker=draft_worker, events=events)
        finally:
            pipeline.shutdown()


def initialize(h, settings=None, session=1):
    run = h.pipeline._run = _Run(session, settings or config())
    h.pipeline._initialize(run)
    for worker in run.threads:
        worker.join(2)
        assert not worker.is_alive()
    return run


def ready(h, session=1):
    return next(event["models"] for event in h.events
                if event["kind"] == "models_ready" and event["session"] == session)


def job(identifier="s1", *, channel="system", context="", revision=1):
    return TranslationJob(channel, identifier, "こんにちは。", time.monotonic(),
                          ("ja", "zh") if channel == "system" else ("zh", "ja"),
                          1, context=context, revision=revision)


@pytest.mark.parametrize("cli", CLI_PROVIDERS)
def test_domestic_routing_uses_one_fair_account_cli_without_codex_or_local_loading(tmp_path, cli):
    translator = engine(cli=cli)
    with harness(tmp_path, domestic=translator) as h:
        run = initialize(h, config(domestic_cli_provider=cli, domestic_cli_model="requested"))
        h.factory.assert_called_once_with(provider=cli, model="requested")
        h.codex_factory.assert_not_called()
        h.local_factory.assert_not_called()
        h.web_factory.assert_not_called()
        translator.preload.assert_called_once_with(cancellation_event=run.cancel)
        assert run.translators == (translator,) and run.translation_channels == (None,)
        assert run.multilingual_recognition
        h.asr.preload.assert_called_once_with(("auto",))
        h.final_worker.assert_called_once_with(run, translator, None)
        h.draft_worker.assert_not_called()
        information = ready(h)["translation"]
        assert information["provider"] == "domestic_cli" and information["cli_provider"] == cli
        assert information["worker_count"] == 1 and information["batch_limit"] == 1
        assert information["scheduling"] == "fair_serial_audio_channels"
        h.capture.return_value.start.assert_called_once_with(enabled_channels=run.config.enabled_channels)
        assert any(event["kind"] == "state" and "国内 CLI 账号翻译" in event["text"] for event in h.events)


def test_domestic_hybrid_runs_independent_local_draft_and_domestic_final_without_codex(tmp_path):
    final, draft = engine(), engine("local-draft", local=True)
    with harness(tmp_path, domestic=final, local=draft) as h:
        run = initialize(h, config(hybrid=True))
        h.codex_factory.assert_not_called()
        h.factory.assert_called_once_with(provider="minimax", model="")
        h.local_factory.assert_called_once()
        final.preload.assert_called_once_with(cancellation_event=run.cancel)
        draft.preload.assert_called_once_with(cancellation_event=run.cancel)
        assert run.translators == (final,) and run.draft_translator is draft
        h.final_worker.assert_called_once_with(run, final, None)
        h.draft_worker.assert_called_once_with(run, draft)
        information = ready(h)["translation"]
        assert information["mode"] == "hybrid" and information["draft_model"]["model_id"] == "local-draft"
        assert information["worker_count"] == information["draft_worker_count"] == 1
        assert any(event["kind"] == "state" and "本地暂译 / 国内 CLI 终稿" in event["text"] for event in h.events)


def test_domestic_resume_reuses_engine_when_only_codex_controls_change(tmp_path):
    with harness(tmp_path) as h:
        original = initialize(h)
        h.pipeline.stop()
        resumed = initialize(h, replace(original.config, codex_model="irrelevant", codex_effort="high", codex_speed="fast"), 2)
        assert resumed.translator is original.translator
        h.factory.assert_called_once()
        assert original.translator.preload.call_count == 2
        original.translator.cancel_pending.assert_called_once()
        original.translator.close.assert_not_called()


@pytest.mark.parametrize("changes", [{"domestic_cli_provider": "kimi"}, {"domestic_cli_model": "new-model"}])
def test_domestic_provider_or_model_change_closes_previous_engine_before_replacement(tmp_path, changes):
    previous, replacement = engine(), engine("replacement", cli=changes.get("domestic_cli_provider", "minimax"))
    with harness(tmp_path) as h:
        h.factory.side_effect = [previous, replacement]
        original = initialize(h)
        h.pipeline.stop()
        resumed = initialize(h, replace(original.config, **changes), 2)
        previous.close.assert_called_once()
        assert resumed.translator is replacement
        assert h.factory.call_args_list == [call(provider="minimax", model=""),
            call(provider=resumed.config.domestic_cli_provider, model=resumed.config.domestic_cli_model)]
        assert any(event["kind"] == "translation_unloaded" and event["session"] == 2 for event in h.events)


def test_switching_domestic_hybrid_to_codex_hybrid_preserves_true_codex_pool_and_closes_domestic(tmp_path):
    domestic, local_a, local_b = engine(), engine("draft-a", local=True), engine("draft-b", local=True)
    primary, secondary = engine("actual-codex", codex=True), engine("actual-codex", codex=True)
    with harness(tmp_path, domestic=domestic, codex_engines=[primary, secondary]) as h:
        h.local_factory.side_effect = [local_a, local_b]
        original = initialize(h, config(hybrid=True))
        h.pipeline.stop()
        resumed = initialize(h, replace(original.config, hybrid_final_provider="codex"), 2)
        domestic.close.assert_called_once()
        local_a.close.assert_called_once()
        assert resumed.translators == (primary, secondary) and resumed.draft_translator is local_b
        assert resumed.translation_channels == ("mic", "system")
        information = ready(h, 2)["translation"]
        assert information["worker_count"] == 2 and information["batch_limit"] == 3
        assert information["scheduling"] == "ordered_workers_bound_to_audio_channels"
        assert h.pipeline._translation_identity(resumed, primary, "final")["translation_provider"] == "codex"
        assert "translation_cli_provider" not in h.pipeline._translation_identity(resumed, primary, "final")


def test_switching_local_web_codex_domestic_hybrid_does_not_reuse_another_provider_engine(tmp_path):
    local_a, draft = engine("local-a", local=True), engine("draft", local=True)
    web = engine("google_web")
    web.supports_mixed_sources = False
    codex_a, codex_b = engine("codex", codex=True), engine("codex", codex=True)
    with harness(tmp_path, codex_engines=[codex_a, codex_b]) as h:
        h.local_factory.side_effect = [local_a, draft]
        h.web_factory.return_value = web
        local_run = initialize(h, replace(config(), translation_provider="local"))
        h.pipeline.stop()
        web_run = initialize(h, replace(config(), translation_provider="google_web"), 2)
        h.pipeline.stop()
        codex_run = initialize(h, replace(config(), translation_provider="codex"), 3)
        h.pipeline.stop()
        domestic_run = initialize(h, config(hybrid=True), 4)
        assert local_run.translator is local_a and web_run.translator is web
        assert codex_run.translators == (codex_a, codex_b)
        assert domestic_run.translator is h.factory.return_value and domestic_run.draft_translator is draft
        for old in (local_a, web, codex_a, codex_b):
            old.close.assert_called_once()


@pytest.mark.parametrize("hybrid", [False, True])
def test_domestic_preload_failure_does_not_fall_back_to_another_remote_engine(tmp_path, hybrid):
    translator = engine()
    translator.preload.side_effect = RuntimeError("synthetic missing account CLI")
    with harness(tmp_path, domestic=translator) as h:
        run = initialize(h, config(hybrid=hybrid))
        h.codex_factory.assert_not_called()
        h.web_factory.assert_not_called()
        if hybrid:
            information = ready(h)["translation"]
            assert not information["available"] and information["provider"] == "domestic_cli"
            assert information["cli_provider"] == "minimax"
            assert information["worker_count"] == 0 and information["draft_worker_count"] == 1
            assert run.translators == () and run.draft_translator is h.local_factory.return_value
            assert any(event["kind"] == "notice" and "国内 CLI 终稿暂不可用" in event["text"] for event in h.events)
        else:
            h.local_factory.assert_not_called()
            h.capture.assert_not_called()
            assert any(event["kind"] == "models_failed" for event in h.events)


def test_domestic_hybrid_local_failure_keeps_domestic_final_and_names_correct_service(tmp_path):
    with harness(tmp_path) as h:
        h.local_factory.side_effect = RuntimeError("synthetic unavailable local model")
        run = initialize(h, config(hybrid=True))
        assert run.translator is h.factory.return_value and run.draft_translator is None
        assert ready(h)["translation"]["draft_worker_count"] == 0
        assert any(event["kind"] == "notice" and "等待 国内 CLI 终稿" in event["text"] for event in h.events)
        h.codex_factory.assert_not_called()


@pytest.mark.parametrize("hybrid", [False, True])
def test_domestic_requests_are_single_fair_jobs_with_independent_source_context_and_no_batch(tmp_path, hybrid):
    events = []
    pipeline = SubtitlePipeline(tmp_path, events.append)
    run = _Run(4, config(hybrid=hybrid))
    translator = engine(cli="minimax")
    # Even an erroneous capability announcement must not enter Codex's batch
    # protocol or pass its conversation-specific keyword arguments.
    translator.backend_info["supports_batch"] = True
    translator.translate.side_effect = lambda text, source, target, **options: "中文终稿" if target == "zh" else "日本語の訳"
    for channel in ("system", "mic"):
        for number in range(2):
            item = job(f"{channel}-{number}", channel=channel, context=f"{channel} source history")
            run.mt_queue.put(channel, item)
    run.mt_queue.close()
    pipeline._translate(run, translator)
    translator.translate_batch.assert_not_called()
    calls = translator.translate.call_args_list
    assert [request.args[1:] for request in calls] == [("zh", "ja"), ("ja", "zh"), ("zh", "ja"), ("ja", "zh")]
    assert [request.kwargs["context"] for request in calls] == ["mic source history", "system source history"] * 2
    assert all(request.kwargs == {"context": request.kwargs["context"], "cancellation_event": run.cancel} for request in calls)
    results = [event for event in events if event["kind"] == "translation"]
    assert [event["segment_id"] for event in results] == ["mic-0", "system-0", "mic-1", "system-1"]
    assert all(event["translation_provider"] == "domestic_cli" and event["translation_cli_provider"] == "minimax" for event in results)
    assert all(event["translation_stage"] == "final" for event in results)


def test_domestic_request_identity_uses_actual_model_and_keeps_draft_local(tmp_path):
    pipeline = SubtitlePipeline(tmp_path, Mock())
    run = _Run(1, config(hybrid=True, domestic_cli_provider="kimi", domestic_cli_model="requested"))
    translator = engine("actual-model", cli="kimi")
    assert pipeline._translation_identity(run, translator, "final") == {
        "translation_stage": "final", "translation_provider": "domestic_cli",
        "translation_model": "actual-model", "translation_cli_provider": "kimi"}
    assert pipeline._translation_identity(run, engine("local-draft", local=True), "draft") == {
        "translation_stage": "draft", "translation_provider": "local", "translation_model": "local-draft"}


@pytest.mark.parametrize("output,flag", [("こんにちは", "language_rejected"), ("None<asr_text>", "protocol_rejected")])
def test_domestic_output_still_obeys_strict_chinese_and_protocol_guards(tmp_path, output, flag):
    events = []
    pipeline = SubtitlePipeline(tmp_path, events.append)
    run = _Run(1, config())
    translator = engine()
    translator.translate.return_value = output
    pipeline._translation_request(run, translator, [job()])
    result = next(event for event in events if event["kind"] == "translation")
    assert result["failed"] and result[flag]
    assert result["translation_provider"] == "domestic_cli"


def test_domestic_final_late_replaces_draft_and_late_draft_cannot_overwrite_final(tmp_path):
    events = []
    pipeline = SubtitlePipeline(tmp_path, events.append)
    run = _Run(1, config(hybrid=True))
    draft, final = engine("draft", local=True), engine()
    draft.translate.return_value, final.translate.return_value = "本地暂译", "国内最终译文"
    item = job()
    pipeline._translation_request(run, draft, [item], stage="draft")
    pipeline._translation_request(run, final, [item])
    pipeline._translation_request(run, draft, [item], stage="draft")
    results = [event for event in events if event["kind"] in ("translation", "translation_candidate")]
    assert [(event["kind"], event["text"], event["translation_provider"]) for event in results] == [
        ("translation", "本地暂译", "local"), ("translation", "国内最终译文", "domestic_cli"),
        ("translation_candidate", "本地暂译", "local")]


def test_domestic_final_failure_preserves_local_draft_and_continues_next_job(tmp_path):
    events = []
    pipeline = SubtitlePipeline(tmp_path, events.append)
    run = _Run(1, config(hybrid=True))
    draft, final = engine("draft", local=True), engine()
    draft.translate.return_value = "本地暂译"
    final.translate.side_effect = [TimeoutError("synthetic unavailable account"), "下一句终稿"]
    item = job()
    pipeline._translation_request(run, draft, [item], stage="draft")
    pipeline._translation_request(run, final, [item])
    pipeline._translation_request(run, final, [job("s2")])
    results = [event for event in events if event["kind"] == "translation"]
    assert [event["text"] for event in results] == ["本地暂译", "下一句终稿"]
    status = next(event for event in events if event["kind"] == "translation_status")
    assert status["final_failed"] and status["has_translation"] and not status["overall_failed"]
    assert any(event["kind"] == "notice" and "国内 CLI 翻译暂不可用" in event["text"] for event in events)


@pytest.mark.parametrize("hybrid", [False, True])
def test_stop_at_domestic_preload_boundary_cannot_start_cli_for_canceled_run(tmp_path, hybrid):
    translator, launches = engine(), []
    with harness(tmp_path, domestic=translator) as h:
        def preload(*, cancellation_event=None):
            h.pipeline.stop()
            if cancellation_event is not None and cancellation_event.is_set():
                raise RuntimeError("synthetic canceled preload")
            launches.append(True)
        translator.preload.side_effect = preload
        run = initialize(h, config(hybrid=hybrid))
        assert run.cancel.is_set() and not launches
        translator.cancel_pending.assert_called_once()
        h.capture.assert_not_called()
        assert not any(event["kind"] == "models_ready" for event in h.events)


@pytest.mark.parametrize("hybrid", [False, True])
def test_pause_cancels_domestic_request_and_discards_late_result_before_provider_switch(tmp_path, hybrid):
    entered, released = threading.Event(), threading.Event()
    events = []
    pipeline = SubtitlePipeline(tmp_path, events.append)
    translator = engine()
    original = pipeline._run = _Run(1, config(hybrid=hybrid), translator=translator, translators=(translator,))
    def translate(*_args, cancellation_event=None, **_options):
        assert cancellation_event is original.cancel
        entered.set()
        assert released.wait(3)
        return "旧国内会话迟到译文"
    translator.translate.side_effect = translate
    translator.cancel_pending.side_effect = released.set
    worker = threading.Thread(target=pipeline._translation_request, args=(original, translator, [job()]))
    original.threads.append(worker)
    worker.start()
    try:
        assert entered.wait(2)
        pipeline.stop()
        pipeline._run = _Run(2, replace(config(), translation_provider="local"))
        worker.join(2)
        assert not worker.is_alive()
        translator.cancel_pending.assert_called_once()
        assert not events
    finally:
        released.set()
        worker.join(2)
        pipeline.shutdown()


def test_domestic_pending_revision_coalesces_without_changing_original_segment_identity(tmp_path):
    events = []
    pipeline = SubtitlePipeline(tmp_path, events.append)
    run = _Run(1, config())
    source = RecognitionJob("system", "original-1", np.ones(16000, np.float32), time.monotonic(), ("ja", "zh"), generation=1)
    pipeline._enqueue_translation(run, source, "別にそう。", provisional=True)
    pipeline._enqueue_translation(run, source, "別にそういうことではない。", revision=2,
                                  source_ids=("original-1", "original-2"), seconds=2)
    run.mt_queue.close()
    translator = engine()
    pipeline._translate(run, translator)
    translator.translate.assert_called_once()
    assert translator.translate.call_args.args[0] == "別にそういうことではない。"
    result = next(event for event in events if event["kind"] == "translation")
    assert result["segment_id"] == "original-1" and result["revision"] == 2
    assert result["source_segment_ids"] == ("original-1", "original-2")
    assert result["translation_cli_provider"] == "minimax"


def test_domestic_job_from_previous_source_is_not_sent_to_cli(tmp_path):
    events = []
    pipeline = SubtitlePipeline(tmp_path, events.append, source_validator=lambda _item: False)
    run = _Run(1, config())
    translator = engine()
    pipeline._translation_request(run, translator, [job()])
    translator.translate.assert_not_called()
    assert not events
