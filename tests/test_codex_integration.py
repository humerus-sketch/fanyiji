"""Translation-provider selection, cancellation and offline asset boundaries."""
from contextlib import ExitStack
import threading
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fanyiji.app import configured_models_available
from fanyiji.config import AppConfig, SourceConfig
from fanyiji.pipeline import SubtitlePipeline, TranslationJob, _Run


def recognition_assets(root):
    for name in ("asr/model.int8.onnx", "asr/tokens.txt", "vad/silero_vad.onnx"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic local asset")


def engine(kind):
    return SimpleNamespace(
        backend_info={"backend": kind, "model_id": "account-model", "device": "cloud" if kind == "codex_cli" else "cpu"},
        preload=Mock(), close=Mock(), cancel_pending=Mock(),
    )


def isolated_pipeline(root, events, codex, local):
    stack = ExitStack()
    asr = SimpleNamespace(preload=Mock(), backend_info={"device": "cpu"})
    stack.enter_context(patch("fanyiji.engines.RecognizerEngine", return_value=asr))
    codex_factory = stack.enter_context(patch("fanyiji.codex_engine.CodexTranslationEngine", return_value=codex))
    local_factory = stack.enter_context(patch("fanyiji.engines.create_translation_engine", return_value=local))
    capture_factory = stack.enter_context(patch("fanyiji.audio.AudioCaptureManager"))
    stack.enter_context(patch("fanyiji.segmentation.SpeechSegmenter"))
    pipeline = SubtitlePipeline(root, events.append)
    stack.enter_context(patch.object(pipeline, "_recognize"))
    stack.enter_context(patch.object(pipeline, "_translate"))
    return stack, pipeline, codex_factory, local_factory, capture_factory, asr


def initialize(pipeline, config, session):
    run = pipeline._run = _Run(session, config)
    pipeline._initialize(run)
    return run


def test_codex_asset_gate_requires_recognition_but_no_local_translation(tmp_path):
    recognition_assets(tmp_path)
    # An invalid local MT profile must not block the separately selected CLI.
    (tmp_path / "translation").mkdir()
    profile = tmp_path / "translation/backend.json"
    profile.write_text("malformed old MT profile", encoding="utf-8")
    assert configured_models_available(tmp_path, speech_only=False, translation_provider="codex")
    assert not configured_models_available(tmp_path, speech_only=False)
    assert not configured_models_available(tmp_path, speech_only=True, translation_provider="codex")
    (tmp_path / "vad/silero_vad.onnx").write_bytes(b"")
    assert not configured_models_available(tmp_path, speech_only=False, translation_provider="codex")


def test_codex_start_uses_cloud_metadata_without_loading_local_mt(tmp_path):
    recognition_assets(tmp_path)
    events = []
    codex, local = engine("codex_cli"), engine("ctranslate2")
    stack, pipeline, codex_factory, local_factory, capture, _ = isolated_pipeline(tmp_path, events, codex, local)
    with stack:
        try:
            initialize(pipeline, AppConfig(microphone=SourceConfig(enabled=False), speech_only=False,
                translation_provider="codex", codex_model="chosen-model"), 1)
            codex_factory.assert_called_once_with(model="chosen-model", effort="", speed="standard")
            local_factory.assert_not_called()
            codex.preload.assert_called_once()
            ready = next(event for event in events if event["kind"] == "models_ready")
            assert ready["models"]["translation"] == {
                **codex.backend_info, "worker_count": 1, "batch_limit": 1,
                "scheduling": "ordered_workers_bound_to_audio_channels", "worker_channels": ["system"]}
            assert events[-1]["text"] == "正在听 · 本地识别 / Codex 订阅翻译"
            capture.return_value.start.assert_called_once_with(enabled_channels={"system"})
        finally:
            pipeline.shutdown()


def test_provider_change_closes_previous_translator_and_reuses_local_asr(tmp_path):
    recognition_assets(tmp_path)
    events = []
    codex, local = engine("codex_cli"), engine("ctranslate2")
    stack, pipeline, codex_factory, local_factory, asr_capture, asr = isolated_pipeline(tmp_path, events, codex, local)
    with stack:
        try:
            initialize(pipeline, AppConfig(microphone=SourceConfig(enabled=False), speech_only=False), 1)
            pipeline.stop()
            initialize(pipeline, AppConfig(microphone=SourceConfig(enabled=False), speech_only=False,
                translation_provider="codex"), 2)
            local.close.assert_called_once()
            assert pipeline._translator is codex
            assert pipeline._recognizer is asr
            pipeline.stop()
            codex.cancel_pending.assert_called_once()
            initialize(pipeline, AppConfig(microphone=SourceConfig(enabled=False), speech_only=False), 3)
            codex.close.assert_called_once()
            assert pipeline._translator is local
            assert local_factory.call_count == 2
            assert codex_factory.call_count == 1
            assert asr_capture.return_value.start.call_count == 3
        finally:
            pipeline.shutdown()


def test_codex_model_change_replaces_engine_but_resume_reuses_same_selection(tmp_path):
    recognition_assets(tmp_path)
    events = []
    codex, local = engine("codex_cli"), engine("ctranslate2")
    stack, pipeline, factory, _, _, _ = isolated_pipeline(tmp_path, events, codex, local)
    with stack:
        try:
            config = AppConfig(microphone=SourceConfig(enabled=False), speech_only=False,
                translation_provider="codex", codex_model="model-a")
            initialize(pipeline, config, 1)
            pipeline.stop()
            initialize(pipeline, config, 2)
            assert factory.call_count == 1
            pipeline.stop()
            initialize(pipeline, AppConfig(microphone=SourceConfig(enabled=False), speech_only=False,
                translation_provider="codex", codex_model="model-b"), 3)
            assert factory.call_count == 2
            factory.assert_called_with(model="model-b", effort="", speed="standard")
            codex.close.assert_called_once()
        finally:
            pipeline.shutdown()


def test_subscription_preload_failure_never_starts_capture_or_uses_local_mt(tmp_path, caplog):
    recognition_assets(tmp_path)
    events = []
    codex, local = engine("codex_cli"), engine("ctranslate2")
    codex.preload.side_effect = RuntimeError("需要 ChatGPT 订阅登录")
    stack, pipeline, _, local_factory, capture, _ = isolated_pipeline(tmp_path, events, codex, local)
    with stack:
        try:
            initialize(pipeline, AppConfig(microphone=SourceConfig(enabled=False), speech_only=False,
                translation_provider="codex"), 1)
            assert not any(event["kind"] == "models_ready" for event in events)
            assert any(event["kind"] == "models_failed" for event in events)
            capture.assert_not_called()
            local_factory.assert_not_called()
            assert "Model initialization failed: RuntimeError: 需要 ChatGPT 订阅登录" in caplog.text
        finally:
            pipeline.shutdown()


def test_canceled_dequeued_job_never_calls_replacement_cloud_engine(tmp_path):
    events = []
    pipeline = SubtitlePipeline(tmp_path, events.append)
    old, replacement = Mock(), Mock()
    pipeline._translator = old
    run = _Run(1, AppConfig(microphone=SourceConfig(enabled=False), translation_provider="codex"), translator=old)
    run.mt_queue.put("system", TranslationJob("system", "old", "旧片段", 0, ("ja", "zh"), 1), 1)
    entered, release = threading.Event(), threading.Event()
    original_get = run.mt_queue.get

    def blocked_get(*args, **kwargs):
        item = original_get(*args, **kwargs)
        entered.set()
        assert release.wait(3)
        return item

    run.mt_queue.get = blocked_get
    worker = threading.Thread(target=pipeline._translate, args=(run,))
    worker.start()
    try:
        assert entered.wait(2)
        run.cancel.set()
        pipeline._translator = replacement
        release.set()
        worker.join(2)
        assert not worker.is_alive()
        old.translate.assert_not_called()
        replacement.translate.assert_not_called()
        assert events == []
    finally:
        release.set()
        worker.join(2)
