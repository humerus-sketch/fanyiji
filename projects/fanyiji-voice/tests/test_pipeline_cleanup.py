"""Failure paths must release every owned worker; all devices/models are fake."""
from unittest.mock import Mock, patch

import pytest

from fanyiji.config import AppConfig
from fanyiji.pipeline import SubtitlePipeline, _Run


@pytest.mark.parametrize("phase", ["constructor", "start"])
def test_initialization_failure_closes_queues_and_partially_started_capture(tmp_path, phase):
    for relative in ("asr/model.int8.onnx", "asr/tokens.txt", "vad/silero_vad.onnx"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    events = []
    pipeline = SubtitlePipeline(tmp_path, events.append)
    run = pipeline._run = _Run(1, AppConfig(speech_only=False))
    recognizer = Mock(backend_info={})
    translator = Mock(backend_info={})
    capture = Mock()
    if phase == "start":
        capture.start.side_effect = RuntimeError("capture startup failed")
    try:
        with patch("fanyiji.engines.RecognizerEngine", return_value=recognizer), \
                patch("fanyiji.engines.create_translation_engine", return_value=translator), \
                patch("fanyiji.segmentation.SpeechSegmenter"), \
                patch("fanyiji.audio.AudioCaptureManager", return_value=capture,
                      side_effect=RuntimeError("capture construction failed") if phase == "constructor" else None):
            pipeline._initialize(run)
        assert any(event["kind"] == "error" for event in events)
        assert run.cancel.is_set()
        assert pipeline._run is None
        assert run.asr_queue.get(timeout=0.1) is None
        assert run.mt_queue.get(timeout=0.1) is None
        assert run.draft_queue.get(timeout=0.1) is None
        assert not any(worker.is_alive() for worker in run.threads)
        if phase == "start":
            capture.stop.assert_called_once()
    finally:
        pipeline.shutdown()


def test_stop_continues_after_one_cli_cancellation_fails(tmp_path):
    pipeline = SubtitlePipeline(tmp_path, Mock())
    run = pipeline._run = _Run(1, AppConfig(translation_provider="hybrid"))
    first, second, draft = Mock(), Mock(), Mock()
    first.cancel_pending.side_effect = OSError("fixture helper already gone")
    run.translators = (first, second)
    run.draft_translator, run.draft_active = draft, 1
    run.capture = Mock()
    pipeline.stop()
    assert run.cancel.is_set()
    second.cancel_pending.assert_called_once()
    draft.cancel_pending.assert_called_once()
    run.capture.stop.assert_called_once()


def test_old_initialization_failure_does_not_cancel_new_run_or_its_cli(tmp_path):
    pipeline = SubtitlePipeline(tmp_path, Mock())
    old = _Run(1, AppConfig(translation_provider="codex"))
    current = pipeline._run = _Run(2, AppConfig(translation_provider="codex"))
    current_cli = Mock()
    pipeline._translators = [current_cli]
    pipeline._stop_run(old)
    assert old.cancel.is_set()
    assert pipeline._run is current
    assert not current.cancel.is_set()
    current_cli.cancel_pending.assert_not_called()


@pytest.mark.parametrize("failure", ["translator", "filter", "processor"])
def test_shutdown_closes_all_resources_even_when_one_release_fails(tmp_path, failure):
    pipeline = SubtitlePipeline(tmp_path, Mock())
    primary, secondary, draft, recognizer, speech_filter, gtcrn, uvr = [Mock() for _ in range(7)]
    pipeline._translator, pipeline._translators = primary, [primary, secondary]
    pipeline._draft_translator = draft
    pipeline._recognizer, pipeline._speech_filter = recognizer, speech_filter
    pipeline._audio_processors = {"gtcrn": gtcrn, "uvr": uvr}
    {"translator": primary, "filter": speech_filter, "processor": gtcrn}[failure].close.side_effect = OSError("fixture close failed")
    pipeline.shutdown()
    pipeline.shutdown()
    for resource in (primary, secondary, draft, recognizer, speech_filter, gtcrn, uvr):
        resource.close.assert_called_once()
    assert not pipeline._audio_processors
