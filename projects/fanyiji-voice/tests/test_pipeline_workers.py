"""Independent translation worker ownership and cancellation, without CLI/audio."""
from contextlib import ExitStack, contextmanager
from dataclasses import replace
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import pytest

from fanyiji.config import AppConfig, SourceConfig
from fanyiji.pipeline import SubtitlePipeline, _Run


def _engine(model_id="actual-model"):
    return SimpleNamespace(
        backend_info={"backend": "codex_cli", "model_id": model_id, "supports_batch": True},
        preload=Mock(), close=Mock(), cancel_pending=Mock(),
    )


def _config(**changes):
    return AppConfig(speech_only=False, translation_provider="codex", **changes)


@contextmanager
def _isolated_pipeline(root, engines=None, factory_side_effect=None):
    for name in ("asr/model.int8.onnx", "asr/tokens.txt", "vad/silero_vad.onnx"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic local asset")
    events = []
    asr = SimpleNamespace(preload=Mock(), close=Mock(), backend_info={"model_id": "fake-asr"})
    with ExitStack() as stack:
        stack.enter_context(patch("fanyiji.engines.RecognizerEngine", return_value=asr))
        factory = stack.enter_context(patch("fanyiji.codex_engine.CodexTranslationEngine",
            side_effect=factory_side_effect if factory_side_effect is not None else engines))
        capture = stack.enter_context(patch("fanyiji.audio.AudioCaptureManager"))
        stack.enter_context(patch("fanyiji.segmentation.SpeechSegmenter"))
        pipeline = SubtitlePipeline(root, events.append)
        recognize = stack.enter_context(patch.object(pipeline, "_recognize"))
        translate = stack.enter_context(patch.object(pipeline, "_translate"))
        try:
            yield SimpleNamespace(pipeline=pipeline, factory=factory, capture=capture,
                asr=asr, events=events, recognize=recognize, translate=translate)
        finally:
            pipeline.shutdown()


def _initialize(harness, config, session):
    run = harness.pipeline._run = _Run(session, config)
    harness.pipeline._initialize(run)
    _join(run)
    return run


def _join(run):
    for worker in run.threads:
        worker.join(timeout=2)
        assert not worker.is_alive()


def _ready(harness, session):
    return next(event for event in harness.events
                if event["kind"] == "models_ready" and event["session"] == session)


def test_pool_start_creates_second_engine_using_actual_model_and_binds_distinct_workers(tmp_path):
    primary, secondary = _engine(), _engine()
    with _isolated_pipeline(tmp_path, [primary, secondary]) as harness:
        config = _config(codex_model="requested-model", codex_effort="low")
        run = _initialize(harness, config, 1)
        assert harness.factory.call_args_list == [
            call(model="requested-model", effort="low", speed="standard"),
            call(model="actual-model", effort="low", speed="standard"),
        ]
        assert run.translator is primary
        assert len(run.translators) == 2
        assert run.translators[0] is primary and run.translators[1] is secondary
        assert harness.translate.call_count == 2
        assert {id(invocation.args[1]) for invocation in harness.translate.call_args_list} == {id(primary), id(secondary)}
        assert all(invocation.args[0] is run for invocation in harness.translate.call_args_list)
        assert harness.translate.call_args_list == [call(run, primary, "mic"), call(run, secondary, "system")]
        info = _ready(harness, 1)["models"]["translation"]
        assert info["model_id"] == "actual-model"
        assert info["worker_count"] == 2 and info["batch_limit"] == 3
        assert info["worker_channels"] == ["mic", "system"]
        primary.preload.assert_called_once()
        secondary.preload.assert_called_once()
        harness.capture.return_value.start.assert_called_once_with(enabled_channels=config.enabled_channels)


@pytest.mark.parametrize("channel", ["mic", "system"])
def test_single_source_creates_only_one_channel_bound_cli(tmp_path, channel):
    primary = _engine()
    with _isolated_pipeline(tmp_path, [primary]) as harness:
        config = _config(microphone=SourceConfig(enabled=channel == "mic"),
                         playback=SourceConfig(enabled=channel == "system", direction="ja_to_zh"))
        run = _initialize(harness, config, 1)
        assert harness.factory.call_count == 1
        assert run.translators == (primary,)
        assert harness.translate.call_args_list == [call(run, primary, channel)]
        assert _ready(harness, 1)["models"]["translation"]["worker_channels"] == [channel]


def test_dual_to_single_source_closes_unused_second_cli_on_resume(tmp_path):
    primary, secondary = _engine(), _engine()
    with _isolated_pipeline(tmp_path, [primary, secondary]) as harness:
        _initialize(harness, _config(), 1)
        harness.pipeline.stop()
        config = _config(microphone=SourceConfig(enabled=False))
        resumed = _initialize(harness, config, 2)
        assert resumed.translators == (primary,)
        assert resumed.translation_channels == ("system",)
        assert harness.factory.call_count == 2
        secondary.close.assert_called_once()
        primary.close.assert_not_called()
        assert harness.translate.call_args_list[-1] == call(resumed, primary, "system")


def test_pause_cancels_both_workers_and_resume_reuses_pool_until_shutdown(tmp_path):
    primary, secondary = _engine(), _engine()
    with _isolated_pipeline(tmp_path, [primary, secondary]) as harness:
        config = _config()
        original = _initialize(harness, config, 1)
        harness.pipeline.stop()
        assert original.cancel.is_set()
        primary.cancel_pending.assert_called_once()
        secondary.cancel_pending.assert_called_once()
        primary.close.assert_not_called()
        secondary.close.assert_not_called()
        resumed = _initialize(harness, config, 2)
        assert resumed.translators[0] is primary and resumed.translators[1] is secondary
        assert harness.factory.call_count == 2
        assert secondary.preload.call_count == 2
        assert _ready(harness, 2)["models"]["translation"]["worker_count"] == 2
        assert harness.translate.call_count == 4
        harness.pipeline.shutdown()
        harness.pipeline.shutdown()
        assert resumed.cancel.is_set()
        assert primary.cancel_pending.call_count == secondary.cancel_pending.call_count == 2
        primary.close.assert_called_once()
        secondary.close.assert_called_once()
        harness.asr.close.assert_called_once()


def test_resume_replaces_secondary_when_account_default_model_changes(tmp_path):
    primary = _engine("model-a")
    previous, replacement = _engine("model-a"), _engine("model-b")
    for engine in (primary, previous, replacement):
        engine.backend_info["reasoning_effort"] = "low"
    default_models = iter(("model-a", "model-b"))

    def preload():
        primary.backend_info["model_id"] = next(default_models)

    primary.preload.side_effect = preload
    with _isolated_pipeline(tmp_path, [primary, previous, replacement]) as harness:
        config = _config()
        _initialize(harness, config, 1)
        harness.pipeline.stop()
        resumed = _initialize(harness, config, 2)
        previous.close.assert_called_once()
        primary.close.assert_not_called()
        assert resumed.translators[0] is primary and resumed.translators[1] is replacement
        assert {engine.backend_info["model_id"] for engine in resumed.translators} == {"model-b"}
        assert harness.factory.call_args_list == [
            call(model="", effort="", speed="standard"),
            call(model="model-a", effort="low", speed="standard"),
            call(model="model-b", effort="low", speed="standard"),
        ]
        info = _ready(harness, 2)["models"]["translation"]
        assert info["model_id"] == "model-b" and info["worker_count"] == 2
        harness.pipeline.shutdown()
        for engine in (primary, previous, replacement):
            engine.close.assert_called_once()


def test_resume_rebuilds_secondary_with_primary_actual_account_default_effort(tmp_path):
    primary, previous, replacement = _engine(), _engine(), _engine()
    primary.backend_info["reasoning_effort"] = "low"
    previous.backend_info["reasoning_effort"] = "low"
    replacement.backend_info["reasoning_effort"] = "medium"
    default_efforts = iter(("low", "medium"))

    def preload():
        primary.backend_info["reasoning_effort"] = next(default_efforts)

    primary.preload.side_effect = preload
    with _isolated_pipeline(tmp_path, [primary, previous, replacement]) as harness:
        config = _config()
        _initialize(harness, config, 1)
        harness.pipeline.stop()
        resumed = _initialize(harness, config, 2)
        previous.close.assert_called_once()
        primary.close.assert_not_called()
        assert resumed.translators[0] is primary and resumed.translators[1] is replacement
        assert {engine.backend_info["reasoning_effort"] for engine in resumed.translators} == {"medium"}
        assert harness.factory.call_args_list == [
            call(model="", effort="", speed="standard"),
            call(model="actual-model", effort="low", speed="standard"),
            call(model="actual-model", effort="medium", speed="standard"),
        ]
        info = _ready(harness, 2)["models"]["translation"]
        assert info["reasoning_effort"] == "medium" and info["worker_count"] == 2
        harness.pipeline.shutdown()
        for engine in (primary, previous, replacement):
            engine.close.assert_called_once()


@pytest.mark.parametrize("changes", [
    {"codex_model": "replacement-model"}, {"codex_effort": "medium"}, {"codex_speed": "fast"},
])
def test_changed_model_controls_close_entire_previous_pool_and_rebuild(tmp_path, changes):
    previous = [_engine("old-model"), _engine("old-model")]
    replacement = [_engine("new-model"), _engine("new-model")]
    with _isolated_pipeline(tmp_path, previous + replacement) as harness:
        config = _config(codex_model="selected-model", codex_effort="low")
        _initialize(harness, config, 1)
        harness.pipeline.stop()
        changed = replace(config, **changes)
        run = _initialize(harness, changed, 2)
        assert harness.factory.call_count == 4
        for old in previous:
            old.close.assert_called_once()
        assert run.translators[0] is replacement[0] and run.translators[1] is replacement[1]
        assert harness.factory.call_args_list[-2:] == [
            call(model=changed.codex_model, effort=changed.codex_effort, speed=changed.codex_speed),
            call(model="new-model", effort=changed.codex_effort, speed=changed.codex_speed),
        ]
        assert _ready(harness, 2)["models"]["translation"]["model_id"] == "new-model"
        assert _ready(harness, 2)["models"]["translation"]["worker_count"] == 2
        harness.pipeline.shutdown()
        for engine in previous + replacement:
            engine.close.assert_called_once()


def test_second_preload_failure_continues_same_primary_model_with_one_worker(tmp_path):
    primary, secondary = _engine(), _engine()
    secondary.preload.side_effect = RuntimeError("second helper unavailable")
    with _isolated_pipeline(tmp_path, [primary, secondary]) as harness:
        run = _initialize(harness, _config(), 1)
        assert len(run.translators) == 1 and run.translators[0] is primary
        assert harness.translate.call_args_list == [call(run, primary, None)]
        ready = _ready(harness, 1)["models"]["translation"]
        assert ready["model_id"] == primary.backend_info["model_id"]
        assert ready["worker_count"] == 1 and ready["batch_limit"] == 3
        assert any(event["kind"] == "notice" and "第二路" in event["text"] for event in harness.events)
        assert not any(event["kind"] in ("models_failed", "error") for event in harness.events)
        harness.capture.return_value.start.assert_called_once()
        secondary.close.assert_called_once()
        primary.close.assert_not_called()
        harness.pipeline.shutdown()
        primary.close.assert_called_once()
        secondary.close.assert_called_once()


def test_shutdown_during_second_preload_cancels_and_closes_both_without_capture(tmp_path):
    entered, released = threading.Event(), threading.Event()
    primary, secondary = _engine(), _engine()

    def preload():
        entered.set()
        assert released.wait(3)

    secondary.preload.side_effect = preload
    secondary.cancel_pending.side_effect = released.set
    secondary.close.side_effect = released.set
    with _isolated_pipeline(tmp_path, [primary, secondary]) as harness:
        try:
            harness.pipeline.start(_config(), 1)
            run = harness.pipeline._run
            assert entered.wait(2)
            began = time.monotonic()
            harness.pipeline.shutdown()
            assert time.monotonic() - began < 1.5
            _join(run)
            assert run.cancel.is_set()
            primary.cancel_pending.assert_called_once()
            secondary.cancel_pending.assert_called_once()
            primary.close.assert_called_once()
            secondary.close.assert_called_once()
            harness.capture.assert_not_called()
            harness.translate.assert_not_called()
            assert not any(event["kind"] == "models_ready" for event in harness.events)
        finally:
            released.set()


def test_shutdown_during_second_factory_return_closes_unpublished_extra(tmp_path):
    entered, released = threading.Event(), threading.Event()
    primary, secondary = _engine(), _engine()
    calls = []

    def factory(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return primary
        entered.set()
        assert released.wait(3)
        return secondary

    with _isolated_pipeline(tmp_path, factory_side_effect=factory) as harness:
        try:
            harness.pipeline.start(_config(), 1)
            run = harness.pipeline._run
            assert entered.wait(2)
            began = time.monotonic()
            harness.pipeline.shutdown()
            assert time.monotonic() - began < 1.5
            released.set()
            _join(run)
            assert run.cancel.is_set()
            assert calls[1]["model"] == "actual-model"
            primary.cancel_pending.assert_called_once()
            primary.close.assert_called_once()
            secondary.close.assert_called_once()
            secondary.preload.assert_not_called()
            harness.capture.assert_not_called()
            harness.translate.assert_not_called()
            assert not any(event["kind"] == "models_ready" for event in harness.events)
        finally:
            released.set()
