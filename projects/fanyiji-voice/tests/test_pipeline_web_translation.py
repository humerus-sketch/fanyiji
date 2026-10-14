"""Web provider routing and lifecycle, isolated from browsers/network/models."""
from contextlib import ExitStack, contextmanager
from dataclasses import replace
import sys
import threading
import time
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from fanyiji.config import AppConfig
from fanyiji.pipeline import RecognitionJob, SubtitlePipeline, TranslationJob, _Run


PROVIDERS = ("google_web", "youdao_web")


def engine(provider, text="已经翻译完成"):
    return SimpleNamespace(
        backend_info={"backend": provider, "model_id": provider, "supports_batch": False,
                      "supports_source_context": False},
        supports_mixed_sources=False, supports_cancellation=True,
        preload=Mock(), close=Mock(), cancel_pending=Mock(), show_window=Mock(),
        translate=Mock(return_value=text), translate_batch=Mock(),
    )


@contextmanager
def harness(root, google=None, youdao=None):
    for name in ("asr/model.int8.onnx", "asr/tokens.txt", "vad/silero_vad.onnx"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic local asset")
    events = []
    asr = SimpleNamespace(preload=Mock(), close=Mock(), backend_info={"model_id": "fake-asr"})
    module = ModuleType("fanyiji.web_translation_engine")
    module.GoogleWebTranslationEngine = Mock(return_value=google or engine("google_web"))
    module.YoudaoWebTranslationEngine = Mock(return_value=youdao or engine("youdao_web"))
    with ExitStack() as stack:
        stack.enter_context(patch.dict(sys.modules, {module.__name__: module}))
        stack.enter_context(patch("fanyiji.engines.RecognizerEngine", return_value=asr))
        local = stack.enter_context(patch("fanyiji.engines.create_translation_engine"))
        codex = stack.enter_context(patch("fanyiji.codex_engine.CodexTranslationEngine"))
        capture = stack.enter_context(patch("fanyiji.audio.AudioCaptureManager"))
        stack.enter_context(patch("fanyiji.segmentation.SpeechSegmenter"))
        pipeline = SubtitlePipeline(root, events.append)
        stack.enter_context(patch.object(pipeline, "_recognize"))
        translate = stack.enter_context(patch.object(pipeline, "_translate"))
        try:
            yield SimpleNamespace(pipeline=pipeline, asr=asr, module=module, local=local,
                                  codex=codex, capture=capture, events=events, translate=translate)
        finally:
            pipeline.shutdown()


def initialize(h, provider, session=1, config=None):
    config = config or AppConfig(speech_only=False, audio_processing="off", translation_provider=provider)
    run = h.pipeline._run = _Run(session, config)
    h.pipeline._initialize(run)
    for worker in run.threads:
        worker.join(2)
        assert not worker.is_alive()
    return run


def job(identifier="s1", channel="system", text="こんにちは。", direction=None, **options):
    return TranslationJob(channel, identifier, text, time.monotonic(),
                          direction or (("ja", "zh") if channel == "system" else ("zh", "ja")),
                          1, **options)


@pytest.mark.parametrize("provider", PROVIDERS)
def test_web_source_prepares_single_fair_worker_and_fixed_asr_without_other_translators(tmp_path, provider):
    with harness(tmp_path) as h:
        run = initialize(h, provider)
        factory = (h.module.GoogleWebTranslationEngine if provider == "google_web"
                   else h.module.YoudaoWebTranslationEngine)
        translator = factory.return_value
        factory.assert_called_once_with()
        h.local.assert_not_called()
        h.codex.assert_not_called()
        translator.preload.assert_called_once_with(cancellation_event=run.cancel)
        assert run.translators == (translator,) and run.translation_channels == (None,)
        assert not run.multilingual_recognition
        h.asr.preload.assert_called_once_with(("ja", "zh"))
        h.translate.assert_called_once_with(run, translator, None)
        ready = next(event for event in h.events if event["kind"] == "models_ready")
        assert ready["models"]["translation"]["model_id"] == provider
        assert ready["models"]["translation"]["scheduling"] == "fair_serial_audio_channels"
        assert ready["models"]["translation"]["batch_limit"] == 1
        states = [event["text"] for event in h.events if event["kind"] == "state"]
        assert any(("Google" if provider == "google_web" else "有道") in text for text in states)


@pytest.mark.parametrize("provider", PROVIDERS)
def test_web_resume_reuses_same_host_ignoring_codex_model_controls(tmp_path, provider):
    with harness(tmp_path) as h:
        old = initialize(h, provider)
        h.pipeline.stop()
        current = initialize(h, provider, 2, replace(old.config, codex_model="changed", codex_speed="fast"))
        assert current.translator is old.translator
        old.translator.cancel_pending.assert_called_once()
        old.translator.close.assert_not_called()
        assert old.translator.preload.call_count == 2
        h.local.assert_not_called()
        h.codex.assert_not_called()


def test_local_google_youdao_codex_local_switch_closes_previous_engine_and_never_relabels(tmp_path):
    with harness(tmp_path) as h:
        local_a, local_b = engine("local-a"), engine("local-b")
        local_a.supports_cancellation = local_b.supports_cancellation = False
        h.local.side_effect = [local_a, local_b]
        codex_primary, codex_secondary = engine("codex"), engine("codex")
        h.codex.side_effect = [codex_primary, codex_secondary]
        previous = []
        for session, provider in enumerate(("local", "google_web", "youdao_web", "codex", "local"), 1):
            run = initialize(h, provider, session)
            assert h.pipeline._translation_identity(run, run.translator, "final")["translation_provider"] == provider
            for translator in previous:
                translator.close.assert_called_once()
            previous.extend(run.translators)
            h.pipeline.stop()
        assert h.local.call_count == 2 and h.codex.call_count == 2
        h.module.GoogleWebTranslationEngine.assert_called_once()
        h.module.YoudaoWebTranslationEngine.assert_called_once()
        h.pipeline.shutdown()
        for translator in previous:
            translator.close.assert_called_once()


@pytest.mark.parametrize("provider", PROVIDERS)
def test_web_preload_failure_is_visible_and_does_not_fall_back_to_codex_or_local(tmp_path, provider):
    broken = engine(provider)
    broken.preload.side_effect = RuntimeError("synthetic page unavailable")
    with harness(tmp_path, **{"google" if provider == "google_web" else "youdao": broken}) as h:
        initialize(h, provider)
        h.local.assert_not_called()
        h.codex.assert_not_called()
        h.capture.assert_not_called()
        assert any(event["kind"] == "models_failed" for event in h.events)
        assert any(event["kind"] == "error" for event in h.events)


@pytest.mark.parametrize("provider", PROVIDERS)
def test_web_jobs_are_single_fair_requests_with_directions_cancellation_and_original_ids(tmp_path, provider):
    events = []
    pipeline = SubtitlePipeline(tmp_path, events.append)
    run = _Run(7, AppConfig(translation_provider=provider))
    translator = engine(provider)
    translator.translate.side_effect = lambda text, source, target, **options: "中文终稿" if target == "zh" else "日本語の訳"
    for channel in ("system", "mic"):
        for number in range(2):
            item = job(f"{channel}-{number}", channel)
            run.mt_queue.put(channel, item)
    run.mt_queue.close()
    pipeline._translate(run, translator)
    translator.translate_batch.assert_not_called()
    assert [request.args[1:] for request in translator.translate.call_args_list] == [
        ("zh", "ja"), ("ja", "zh"), ("zh", "ja"), ("ja", "zh")]
    assert all(request.kwargs == {"cancellation_event": run.cancel}
               for request in translator.translate.call_args_list)
    translations = [event for event in events if event["kind"] == "translation"]
    assert [event["segment_id"] for event in translations] == ["mic-0", "system-0", "mic-1", "system-1"]
    assert all(event["translation_provider"] == provider for event in translations)
    assert all(event["translation_stage"] == "final" for event in translations)
    assert not any(event.get("translation_provider") == "codex" for event in events)


@pytest.mark.parametrize("provider", PROVIDERS)
def test_web_still_rejects_foreign_characters_and_asr_protocol_in_output(tmp_path, provider):
    events = []
    pipeline = SubtitlePipeline(tmp_path, events.append)
    run = _Run(1, AppConfig(translation_provider=provider))
    translator = engine(provider, "こんにちは")
    pipeline._translation_request(run, translator, [job()])
    translation = next(event for event in events if event["kind"] == "translation")
    assert translation["failed"] and translation["language_rejected"]
    assert translation["translation_provider"] == provider
    events.clear()
    translator.translate.return_value = "None<asr_text>"
    pipeline._translation_request(run, translator, [job()])
    translation = next(event for event in events if event["kind"] == "translation")
    assert translation["failed"] and translation["protocol_rejected"]


@pytest.mark.parametrize("provider", PROVIDERS)
def test_cancel_web_inflight_request_suppresses_late_output_and_does_not_touch_new_session(tmp_path, provider):
    entered, released = threading.Event(), threading.Event()
    events = []
    pipeline = SubtitlePipeline(tmp_path, events.append)
    translator = engine(provider)
    def translate(*args, cancellation_event=None):
        assert cancellation_event is original.cancel
        entered.set()
        assert released.wait(3)
        return "旧会话迟到终稿"
    translator.translate.side_effect = translate
    translator.cancel_pending.side_effect = released.set
    original = pipeline._run = _Run(1, AppConfig(translation_provider=provider),
                                    translator=translator, translators=(translator,))
    worker = threading.Thread(target=pipeline._translation_request, args=(original, translator, [job()]))
    original.threads.append(worker)
    worker.start()
    try:
        assert entered.wait(2)
        pipeline.stop()
        pipeline._run = _Run(2, AppConfig(translation_provider="local"))
        worker.join(2)
        assert not worker.is_alive()
        translator.cancel_pending.assert_called_once()
        assert not events
    finally:
        released.set()
        worker.join(2)
        pipeline.shutdown()


@pytest.mark.parametrize("provider", PROVIDERS)
def test_pause_cancels_web_host_while_preload_is_in_flight_without_blocking_ui(tmp_path, provider):
    entered, released = threading.Event(), threading.Event()
    translator = engine(provider)
    def preload(*, cancellation_event=None):
        entered.set()
        assert released.wait(3)
    translator.preload.side_effect = preload
    translator.cancel_pending.side_effect = released.set
    with harness(tmp_path, **{"google" if provider == "google_web" else "youdao": translator}) as h:
        try:
            h.pipeline.start(AppConfig(translation_provider=provider, speech_only=False, audio_processing="off"), 1)
            run = h.pipeline._run
            assert entered.wait(2)
            began = time.monotonic()
            h.pipeline.stop()
            assert time.monotonic() - began < 1.5
            translator.cancel_pending.assert_called_once()
            for worker in run.threads:
                worker.join(2)
                assert not worker.is_alive()
            h.capture.assert_not_called()
            assert not any(event["kind"] == "models_ready" for event in h.events)
        finally:
            released.set()


@pytest.mark.parametrize("provider", PROVIDERS)
def test_web_source_revision_coalescing_preserves_segmentation_identity(tmp_path, provider):
    events = []
    pipeline = SubtitlePipeline(tmp_path, events.append)
    run = _Run(1, AppConfig(translation_provider=provider))
    source = RecognitionJob("system", "original-1", np.ones(16000, np.float32), time.monotonic(),
                            ("ja", "zh"), generation=1)
    pipeline._enqueue_translation(run, source, "別にそう。", provisional=True)
    pipeline._enqueue_translation(run, source, "別にそういうことではない。", revision=2,
                                  source_ids=("original-1", "original-2"), seconds=2)
    run.mt_queue.close()
    translator = engine(provider)
    pipeline._translate(run, translator)
    translator.translate.assert_called_once()
    assert translator.translate.call_args.args[0] == "別にそういうことではない。"
    translation = next(event for event in events if event["kind"] == "translation")
    assert translation["segment_id"] == "original-1" and translation["revision"] == 2
    assert translation["source_segment_ids"] == ("original-1", "original-2")


@pytest.mark.parametrize("provider", PROVIDERS)
def test_web_network_error_fails_current_subtitle_with_provider_notice_without_retrying_elsewhere(tmp_path, provider):
    events = []
    pipeline = SubtitlePipeline(tmp_path, events.append)
    run = _Run(1, AppConfig(translation_provider=provider))
    translator = engine(provider)
    translator.translate.side_effect = TimeoutError("synthetic timeout")
    pipeline._translation_request(run, translator, [job()])
    translator.translate.assert_called_once()
    failed = next(event for event in events if event["kind"] == "translation")
    assert failed["failed"] and failed["error"] == "TimeoutError"
    assert failed["translation_provider"] == provider
    assert any(event["kind"] == "notice" and ("Google" if provider == "google_web" else "有道")
               in event["text"] for event in events)


@pytest.mark.parametrize("provider", PROVIDERS)
def test_translation_window_open_uses_selected_cached_host_asynchronously_without_new_translator(tmp_path, provider):
    entered, released = threading.Event(), threading.Event()
    caller_thread = threading.get_ident()
    caller_ids = []
    with harness(tmp_path) as h:
        run = initialize(h, provider)
        def show(*, cancellation_event=None):
            assert cancellation_event is run.cancel
            caller_ids.append(threading.get_ident())
            entered.set()
            assert released.wait(3)
        run.translator.show_window.side_effect = show
        try:
            began = time.monotonic()
            assert h.pipeline.show_translation_window() is True
            assert time.monotonic() - began < 0.5
            assert entered.wait(2)
            assert caller_ids == [next(worker.ident for worker in run.threads
                                       if worker.name == "translation-browser-show")]
            assert caller_ids[0] != caller_thread
            assert h.pipeline.show_translation_window() is True
            run.translator.show_window.assert_called_once_with(cancellation_event=run.cancel)
            h.local.assert_not_called()
            h.codex.assert_not_called()
            factory = (h.module.GoogleWebTranslationEngine if provider == "google_web"
                       else h.module.YoudaoWebTranslationEngine)
            factory.assert_called_once()
        finally:
            released.set()
            for worker in run.threads:
                worker.join(2)
                assert not worker.is_alive()


@pytest.mark.parametrize("provider", ("local", "codex", "hybrid", *PROVIDERS))
def test_translation_window_open_before_provider_ready_returns_false_without_starting_host(tmp_path, provider):
    pipeline = SubtitlePipeline(tmp_path, Mock())
    assert pipeline.show_translation_window() is False
    pipeline._run = _Run(1, AppConfig(translation_provider=provider))
    with patch("fanyiji.pipeline.threading.Thread") as worker:
        assert pipeline.show_translation_window() is False
        worker.assert_not_called()


@pytest.mark.parametrize("provider", PROVIDERS)
def test_translation_window_error_is_reported_using_notice_and_does_not_disable_translation(tmp_path, provider):
    with harness(tmp_path) as h:
        run = initialize(h, provider)
        run.translator.show_window.side_effect = RuntimeError("synthetic unavailable browser")
        assert h.pipeline.show_translation_window() is True
        for worker in run.threads:
            worker.join(2)
            assert not worker.is_alive()
        assert any(event["kind"] == "notice" and "无法打开翻译器" in event["text"] for event in h.events)
        assert not run.cancel.is_set()
        run.translator.cancel_pending.assert_not_called()
        assert h.pipeline._translation_window_request is None


@pytest.mark.parametrize("provider", PROVIDERS)
def test_canceled_delayed_show_request_does_not_open_old_or_new_provider(tmp_path, provider):
    with harness(tmp_path) as h:
        run = initialize(h, provider)
        with patch("fanyiji.pipeline.threading.Thread") as worker:
            assert h.pipeline.show_translation_window() is True
            invocation = worker.call_args
            h.pipeline.stop()
        # Fake workers are deliberately never started; remove them before the
        # harness exercises the real shutdown lifecycle.
        run.threads.clear()
        replacement = initialize(h, "youdao_web" if provider == "google_web" else "google_web", 2)
        invocation.kwargs["target"](*invocation.kwargs["args"])
        run.translator.show_window.assert_not_called()
        replacement.translator.show_window.assert_not_called()
        assert not any(event["kind"] == "notice" and "无法打开翻译器" in event["text"] for event in h.events)


@pytest.mark.parametrize("provider", PROVIDERS)
def test_stop_at_web_preload_boundary_cannot_start_a_helper_for_canceled_run(tmp_path, provider):
    translator = engine(provider)
    launched = []
    with harness(tmp_path, **{"google" if provider == "google_web" else "youdao": translator}) as h:
        def preload(*, cancellation_event=None):
            # Stop after pipeline's first cancel check, before the engine has
            # captured its generation or tried to create a browser process.
            h.pipeline.stop()
            if cancellation_event is not None and cancellation_event.is_set():
                raise RuntimeError("synthetic canceled preload")
            launched.append(True)
        translator.preload.side_effect = preload
        run = initialize(h, provider)
        assert run.cancel.is_set()
        translator.preload.assert_called_once_with(cancellation_event=run.cancel)
        translator.cancel_pending.assert_called_once()
        assert not launched
        h.capture.assert_not_called()
        assert not any(event["kind"] == "models_ready" for event in h.events)
