from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest
import wave

import numpy as np

from fanyiji.config import AppConfig, SourceConfig
from fanyiji.diagnostics import SessionRecorder
from fanyiji.speech_filter import FilterDecision


class ActualPipelineRecordingTests(unittest.TestCase):
    def test_segments_events_and_disabled_source_counts(self):
        with tempfile.TemporaryDirectory() as parent:
            directory = Path(parent) / "recording"
            config = AppConfig(microphone=SourceConfig(enabled=False))
            recorder = SessionRecorder(directory, config)
            pcm = np.full(1600, .25, dtype=np.float32)
            recorder.on_chunk(SimpleNamespace(channel="system", samples=pcm))
            recorder.on_chunk(SimpleNamespace(channel="mic", samples=pcm))
            job = SimpleNamespace(channel="system", segment_id="1:system:test", samples=pcm, direction=("ja", "zh"))
            # Recognition may finish before the audio callback stores its segment.
            recorder.event({"kind": "original", "segment_id": job.segment_id, "text": "明日"})
            recorder.on_segment(job, True)
            recorder.event({"kind": "translation", "segment_id": job.segment_id, "text": "明天", "latency": .5})
            recorder.finish()
            report = json.loads((directory / "results.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "application_closed")
            self.assertEqual(report["raw_audio_seconds"], {"system": .1})
            self.assertFalse(report["recording_uploaded"])
            self.assertEqual(report["samples"][0]["production_translation"], "明天")
            with wave.open(report["samples"][0]["audio_path"], "rb") as stream:
                self.assertEqual((stream.getnchannels(), stream.getframerate(), stream.getnframes()), (1, 16000, 1600))
            recorder.event({"kind": "notice", "text": "late"})
            self.assertFalse(recorder.active)

    def test_recording_never_overwrites_existing_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                SessionRecorder(Path(directory), AppConfig())

    def test_filter_callback_race_preserves_raw_and_reference_uses_masked_audio(self):
        for filter_first in (True, False):
            with tempfile.TemporaryDirectory() as directory:
                recorder = SessionRecorder(Path(directory) / "recording", AppConfig())
                raw = np.full(32000, .25, dtype=np.float32)
                job = SimpleNamespace(channel="system", segment_id="1:system:masked", samples=raw, direction=("ja", "zh"))
                masked = raw.copy()
                masked[:16000] = 0
                decision = FilterDecision(masked, {"action": "masked", "masked_duration_seconds": 1})
                operations = (lambda: recorder.on_filtered(job, decision), lambda: recorder.on_segment(job, True))
                for operation in operations if filter_first else reversed(operations):
                    operation()
                recorder.finish()
                report = json.loads((recorder.directory / "results.json").read_text(encoding="utf-8"))
                sample = report["samples"][0]
                self.assertNotEqual(sample["audio_path"], sample["audio_original_path"])
                with wave.open(sample["audio_path"], "rb") as stream:
                    actual = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2")
                    self.assertTrue(np.all(actual[:16000] == 0))
                    self.assertTrue(np.all(actual[16000:] > 0))
                with wave.open(sample["audio_original_path"], "rb") as stream:
                    self.assertTrue(np.all(np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2") > 0))
                self.assertEqual(sample["speech_filter"]["action"], "masked")

    def test_fail_preserves_error_type_and_closes_stream(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = SessionRecorder(Path(directory) / "recording", AppConfig())
            recorder.fail(OSError("private data must not be logged"))
            report = json.loads((recorder.directory / "results.json").read_text(encoding="utf-8"))
            self.assertEqual(report["errors"], ["OSError"])
            self.assertEqual(report["status"], "failed")
            self.assertTrue(recorder.stream.closed)

    def test_ready_metadata_is_saved_immediately_and_latest_queue_state_is_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = SessionRecorder(Path(directory) / "recording", AppConfig())
            models = {"translation": {"model_id": "gpt-6.1-sol", "worker_count": 2, "batch_limit": 3},
                      "asr": {"model_id": "asr", "accuracy_processing": {"pending_wait_seconds": 1.2}}}
            recorder.event({"kind": "models_ready", "models": models})
            self.assertEqual(recorder.model_info["translation"]["worker_count"], 2)
            models["translation"]["worker_count"] = 99
            models["asr"]["accuracy_processing"]["pending_wait_seconds"] = 99
            state = {"kind": "translation_queue", "pending": 3, "oldest_wait_seconds": 4.5}
            recorder.event(state)
            recorder.event({"kind": "translation_queue", "pending": 1, "oldest_wait_seconds": .5})
            state["pending"] = 99
            recorder.finish()
            report = json.loads((recorder.directory / "results.json").read_text(encoding="utf-8"))
            self.assertEqual(report["models"]["translation"]["worker_count"], 2)
            self.assertEqual(report["models"]["asr"]["accuracy_processing"]["pending_wait_seconds"], 1.2)
            self.assertEqual(report["translation_queue"]["pending"], 1)
            self.assertEqual(report["translation_queue"]["oldest_wait_seconds"], .5)

    def test_stage_timings_and_batch_requests_are_not_multiplied_by_subtitle_count(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = SessionRecorder(Path(directory) / "recording", AppConfig())
            for identifier in ("a", "b", "c"):
                recorder.on_segment(SimpleNamespace(channel="system", segment_id=identifier,
                    samples=np.zeros(1600, np.float32), direction=("ja", "zh")), True)
            recorder.event({"kind": "recognition", "segment_id": "a", "text": "原文", "quality": {},
                            "processing_seconds": 2.3, "speech_filter_seconds": .2, "queue_wait_seconds": .4})
            for identifier in ("a", "b"):
                recorder.event({"kind": "translation", "segment_id": identifier, "text": "译文", "latency": 8,
                    "processing_seconds": 5, "queue_wait_seconds": 1.1, "batch_id": "batch-1", "batch_size": 2,
                    "rpc_metrics": {"total_seconds": 4.8, "receive_seconds": 4}, "fallback": False})
            recorder.event({"kind": "translation", "segment_id": "c", "text": "单句", "batch_id": "batch-2",
                "batch_size": 1, "processing_seconds": 3, "queue_wait_seconds": .2,
                "rpc_metrics": {"total_seconds": 2.9}, "fallback": True})
            recorder.snapshot()
            recorder.snapshot()
            recorder.finish()
            report = json.loads((recorder.directory / "results.json").read_text(encoding="utf-8"))
            sample = report["samples"][0]
            self.assertEqual(sample["asr_processing_seconds"], 2.3)
            self.assertEqual(sample["speech_filter_processing_seconds"], .2)
            self.assertEqual(sample["asr_queue_wait_seconds"], .4)
            self.assertEqual(sample["translation_processing_seconds"], 5)
            self.assertEqual(sample["translation_queue_wait_seconds"], 1.1)
            self.assertEqual(sample["translation_batch_id"], "batch-1")
            self.assertEqual(sample["translation_batch_size"], 2)
            self.assertEqual(sample["translation_rpc_metrics"]["receive_seconds"], 4)
            self.assertFalse(sample["translation_fallback"])
            requests = report["translation_requests"]
            self.assertEqual(len(requests), 2)
            self.assertEqual(requests[0]["segment_ids"], ["a", "b"])
            self.assertEqual(sum(request["processing_seconds"] for request in requests), 8)
            self.assertAlmostEqual(sum(request["total_seconds"] for request in requests), 7.7)
            self.assertTrue(requests[1]["fallback"])

    def test_failed_request_without_a_saved_segment_is_still_counted_once(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = SessionRecorder(Path(directory) / "recording", AppConfig())
            event = {"kind": "translation_request", "batch_id": "failed-1", "batch_size": 2,
                     "processing_seconds": 6, "rpc_metrics": {"total_seconds": 5.9}, "fallback": True}
            recorder.event(event)
            recorder.event(event)
            recorder.finish()
            report = json.loads((recorder.directory / "results.json").read_text(encoding="utf-8"))
            self.assertEqual(report["samples"], [])
            self.assertEqual(len(report["translation_requests"]), 1)
            self.assertEqual(report["translation_requests"][0]["total_seconds"], 5.9)

    def test_legacy_events_keep_existing_fields_without_inventing_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = SessionRecorder(Path(directory) / "recording", AppConfig())
            recorder.on_segment(SimpleNamespace(channel="system", segment_id="legacy",
                samples=np.zeros(1600, np.float32), direction=("ja", "zh")), True)
            recorder.event({"kind": "recognition", "segment_id": "legacy", "text": "明日", "quality": {}})
            recorder.event({"kind": "translation", "segment_id": "legacy", "text": "明天", "latency": .5})
            recorder.finish()
            report = json.loads((recorder.directory / "results.json").read_text(encoding="utf-8"))
            self.assertEqual(report["samples"][0]["production_text"], "明日")
            self.assertEqual(report["samples"][0]["production_translation"], "明天")
            self.assertEqual(report["samples"][0]["pipeline_latency_seconds"], .5)
            self.assertIsNone(report["samples"][0]["asr_processing_seconds"])
            self.assertIsNone(report["samples"][0]["translation_batch_id"])
            self.assertEqual(report["translation_requests"], [])
