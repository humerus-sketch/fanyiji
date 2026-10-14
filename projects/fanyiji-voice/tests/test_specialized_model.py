"""Corpus orchestration tests without audio devices, cloud calls or real weights."""
from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from fanyiji.config import AppConfig, SourceConfig
from fanyiji.pipeline import RecognitionJob
from fanyiji.specialized_model import SpecializedModelService


class SpecializedModelServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = Mock()
        self.store.summary.return_value = {"samples": 0, "deployment_locked": False}
        self.store.pending_comparisons.return_value = []
        self.service = SpecializedModelService(self.root, self.root / "models", store=self.store)
        self.config = AppConfig(microphone=SourceConfig(enabled=False), translation_provider="codex",
                                codex_model="gpt-6.1-sol")
        self.service.begin_session(3, self.config)

    def tearDown(self):
        self.service.close()
        self.temp.cleanup()

    def test_only_current_enabled_channel_is_collected_and_actual_model_is_recorded(self):
        self.service.on_event({"kind": "models_ready", "session": 3,
                              "models": {"translation": {"model_id": "actual-model"}}})
        self.service.on_event({"kind": "original", "session": 2, "channel": "system", "text": "old"})
        self.service.on_event({"kind": "original", "session": 3, "channel": "mic", "text": "disabled"})
        current = {"kind": "original", "session": 3, "channel": "system", "text": "current",
                   "revision": 2}
        self.service.on_event(current)
        current["text"] = "modified later"
        self.service.flush()
        self.store.record_event.assert_called_once()
        args, kwargs = self.store.record_event.call_args
        self.assertEqual(args[1]["text"], "current")
        self.assertEqual(kwargs, {"direction": "ja_to_zh", "provider": "codex", "model": "actual-model"})

    def test_session_identifiers_do_not_collide_across_runs_and_queued_direction_is_frozen(self):
        self.service.on_event({"kind": "original", "session": 3, "channel": "system", "text": "first"})
        self.service.begin_session(4, AppConfig(playback=SourceConfig(direction="zh_to_ja")))
        self.service.on_event({"kind": "original", "session": 4, "channel": "system", "text": "second"})
        self.service.flush()
        first, second = self.store.record_event.call_args_list
        self.assertNotEqual(first.args[0], second.args[0])
        self.assertEqual(first.kwargs["direction"], "ja_to_zh")
        self.assertEqual(second.kwargs["direction"], "zh_to_ja")

    def test_audio_is_copied_and_old_or_rejected_audio_not_saved(self):
        samples = np.ones(32, dtype=np.float32)
        job = RecognitionJob("system", "3:system:1", samples, 0, ("ja", "zh"))
        self.service.on_segment(job, True)
        self.service.on_segment(RecognitionJob("system", "2:system:2", samples, 0, ("ja", "zh")), True)
        self.service.on_segment(job, False)
        samples[:] = 0
        self.service.flush()
        self.store.save_audio.assert_called_once()
        recorded = self.store.save_audio.call_args.args[1]
        self.assertTrue(np.all(recorded.samples == 1))

    def test_disabled_collection_persists_and_never_unlocks_deployment(self):
        self.service.set_enabled(False)
        self.service.on_event({"kind": "translation", "session": 3, "channel": "system", "text": "x"})
        self.service.flush()
        self.store.record_event.assert_not_called()
        self.assertIs(self.service.summary()["deployment_locked"], True)
        self.service.close()
        self.service = SpecializedModelService(self.root, self.root / "models", store=self.store)
        self.assertFalse(self.service.summary()["enabled"])

    def test_failed_background_write_stops_collection_without_propagating_to_subtitles(self):
        self.store.record_event.side_effect = OSError("disk failure")
        self.service.on_event({"kind": "original", "session": 3, "channel": "system", "text": "x"})
        self.service.flush()
        self.assertFalse(self.service.summary()["enabled"])
        self.assertIn("写入失败", self.service.summary()["error"])

    def test_teacher_comparison_serializes_load_close_and_does_not_change_profile(self):
        timeline = []
        teachers = [SimpleNamespace(id=mid, name=mid, model_kind="hy",
                                     files=[SimpleNamespace(relative_path=f"translation/{mid}.gguf")])
                    for mid in ("first", "second")]
        self.service.available_teachers = lambda: teachers
        self.store.pending_comparisons.return_value = [{"id": 42, "direction": "ja_to_zh", "source": "source",
                                                        "context": "previous", "pending_model_ids": ["first", "second"]}]
        profile = self.root / "models" / "translation" / "backend.json"
        profile.parent.mkdir(parents=True)
        profile.write_text('{"model_file":"active.gguf"}')

        class Engine:
            def __init__(_self, _root, *, config):
                _self.mid = config["model_file"]
                timeline.append(("init", _self.mid))
            def preload(_self):
                timeline.append(("preload", _self.mid))
            def translate(_self, text, source, target, *, context):
                self.assertEqual((text, source, target, context), ("source", "ja", "zh", "previous"))
                timeline.append(("translate", _self.mid))
                return "target"
            def close(_self):
                timeline.append(("close", _self.mid))

        self.service._engine_factory = Engine
        self.service._request_compare = Mock()
        self.assertTrue(self.service.compare_local())
        self.assertFalse(self.service.compare_local())
        self.service.run_comparison()
        self.assertEqual(timeline, [(op, mid + ".gguf") for mid in ("first", "second")
                                    for op in ("init", "preload", "translate", "close")])
        self.assertEqual(self.store.add_candidate.call_count, 2)
        self.assertEqual(profile.read_text(), '{"model_file":"active.gguf"}')
        self.assertFalse(self.service.busy)
        self.assertTrue(self.service.summary()["deployment_locked"])

    def test_close_cancels_owned_teacher_and_drains_all_accepted_writes(self):
        engine = Mock()
        self.service._engine = engine
        self.service.on_event({"kind": "original", "session": 3, "channel": "system", "text": "x"})
        self.service.close()
        engine.close.assert_called_once()
        self.store.record_event.assert_called_once()
        self.store.close.assert_called_once()
        self.assertFalse(self.service._worker.is_alive())


if __name__ == "__main__":
    unittest.main()
