"""Model readiness comes from successful native preload stubs, not saved profiles."""
import threading
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fanyiji.config import AppConfig
from fanyiji.pipeline import SubtitlePipeline


def assets(root):
    for name in ("asr/model.int8.onnx", "asr/tokens.txt", "vad/silero_vad.onnx"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"stub")


def test_models_ready_contains_actual_backend_metadata_only_after_both_preloads(tmp_path):
    assets(tmp_path)
    entered, release, running = threading.Event(), threading.Event(), threading.Event()
    events = []
    def event(value):
        events.append(value)
        if value.get("text") == "正在听 · 完全本地":
            running.set()
    asr_info = {"model_id": "sensevoice-int8", "model_kind": "sense_voice", "device": "cpu"}
    mt_info = {"model_kind": "m2m100", "device": "cpu"}
    def preload():
        entered.set()
        assert release.wait(3)
    recognizer = SimpleNamespace(preload=Mock(), backend_info=asr_info)
    translator = SimpleNamespace(preload=Mock(side_effect=preload), backend_info=mt_info)
    capture = Mock()
    pipeline = SubtitlePipeline(tmp_path, event)
    with patch("fanyiji.engines.RecognizerEngine", return_value=recognizer), patch(
        "fanyiji.engines.create_translation_engine", return_value=translator
    ), patch("fanyiji.segmentation.SpeechSegmenter"), patch(
        "fanyiji.audio.AudioCaptureManager", return_value=capture
    ):
        try:
            pipeline.start(AppConfig(speech_only=False), 7)
            assert entered.wait(2)
            assert any(event["kind"] == "models_loading" for event in events)
            assert not any(event["kind"] == "models_ready" for event in events)
            release.set()
            assert running.wait(2)
            ready = [event for event in events if event["kind"] == "models_ready"]
            assert len(ready) == 1
            assert ready[0]["session"] == 7
            assert ready[0]["models"]["asr"] == asr_info
            assert ready[0]["models"]["translation"] == mt_info
            assert ready[0]["models"]["speech_filter"] == {"enabled": False}
            assert events.index(ready[0]) < next(i for i, event in enumerate(events) if event.get("text") == "正在听 · 完全本地")
        finally:
            release.set()
            pipeline.shutdown()


def test_failed_preload_emits_failure_and_never_publishes_selected_model(tmp_path):
    assets(tmp_path)
    failed = threading.Event()
    events = []
    def event(value):
        events.append(value)
        if value["kind"] == "error":
            failed.set()
    recognizer = SimpleNamespace(preload=Mock(), backend_info={"device": "cpu"})
    translator = SimpleNamespace(preload=Mock(side_effect=RuntimeError("synthetic load failure")),
                                 backend_info={"model_file": "selected.gguf", "device": "cuda"})
    pipeline = SubtitlePipeline(tmp_path, event)
    with patch("fanyiji.engines.RecognizerEngine", return_value=recognizer), patch(
        "fanyiji.engines.create_translation_engine", return_value=translator
    ), patch("fanyiji.audio.AudioCaptureManager") as capture:
        try:
            pipeline.start(AppConfig(speech_only=False), 8)
            assert failed.wait(2)
            assert any(value["kind"] == "models_failed" for value in events)
            assert not any(value["kind"] == "models_ready" for value in events)
            capture.assert_not_called()
        finally:
            pipeline.shutdown()
