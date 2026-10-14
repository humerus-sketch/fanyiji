"""The dedicated student stays locked and Codex labels cannot be approved."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from fanyiji.config import AppConfig
from fanyiji.specialized_dialog import SpecializedModelDialog
from fanyiji.ui import MainWindow


class FakeService:
    def __init__(self, root):
        self.root = Path(root)
        self.enabled = True
        self.busy = False
        self.calls = []
        self.samples = [
            {"id": "s1", "segment_id": "u1", "revision": 1, "channel": "playback", "direction": "ja_to_zh",
             "source": "今日は金曜日です。", "provisional": False, "uncertain": False, "failed": False,
             "audio_paths": [], "candidates": [
                 {"id": "c1", "origin": "codex", "model": "gpt-account-model", "text": "今天是星期五。",
                  "reviewed": False, "training_allowed": False},
                 {"id": "l1", "origin": "local", "model": "Hy-MT2 7B", "text": "今天是星期五。",
                  "reviewed": False, "training_allowed": False},
             ]},
            {"id": "s2", "segment_id": "u2", "revision": 1, "channel": "playback", "direction": "ja_to_zh",
             "source": "明日は土曜日です。", "provisional": False, "uncertain": False, "failed": False,
             "audio_paths": [], "candidates": [
                 {"id": "l2", "origin": "local", "model": "Hy-MT2 1.8B", "text": "明天是星期六。",
                  "reviewed": False, "training_allowed": False},
             ]},
        ]

    def summary(self):
        return {"enabled": self.enabled, "busy": self.busy, "status": "数据积累", "phase": "collecting",
                "samples": len(self.samples), "audio_files": 0, "local_candidates": 2, "codex_candidates": 1,
                "reviewed": 0, "trainable": 0, "deployment_locked": False}

    def list_samples(self, limit=100):
        return self.samples[:limit]

    def set_enabled(self, enabled):
        self.enabled = enabled
        self.calls.append(("enabled", enabled))

    def compare_local(self):
        self.calls.append(("compare",))
        self.busy = True
        return True

    def cancel_comparison(self):
        self.calls.append(("cancel",))

    def review_candidate(self, sample_id, candidate_id, **kwargs):
        self.calls.append(("review", sample_id, candidate_id, kwargs))

    def export_dataset(self):
        self.calls.append(("export",))
        raise ValueError("可训练样本不足；Codex 候选不进入训练集")

    def prepare_training(self):
        self.calls.append(("prepare",))
        return {"plan": str(self.root / "training-plan.json"), "trained": False}


class SpecializedDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = TemporaryDirectory()
        self.service = FakeService(self.temp.name)
        self.dialog = SpecializedModelDialog(self.service)
        self.dialog.show()
        self.app.processEvents()

    def tearDown(self):
        self.dialog.close()
        self.dialog.deleteLater()
        self.app.processEvents()
        self.temp.cleanup()

    def test_main_window_has_independent_entry_and_emits_request(self):
        window = MainWindow(AppConfig())
        requests = []
        window.specialized_models_requested.connect(lambda: requests.append(True))
        self.assertEqual(window.specialized_button.text(), "专用模型")
        self.assertTrue(window.specialized_button.isEnabled())
        window.specialized_button.click()
        self.assertEqual(requests, [True])
        window.close()
        window.deleteLater()

    def test_manifest_cannot_unlock_student_and_no_activation_call_exists(self):
        (self.service.root / "deployment.json").write_text('{"approved":true}', encoding="utf-8")
        self.dialog.refresh()
        self.assertFalse(self.dialog.use_button.isEnabled())
        self.dialog.use_button.click()
        self.assertEqual(self.service.calls, [])

    def test_codex_candidate_cannot_be_approved_even_direct_handler_call(self):
        self.assertEqual(self.dialog.candidates_combo.currentData()["origin"], "codex")
        self.assertIn("隔离", self.dialog.candidate_label.text())
        self.assertFalse(self.dialog.source_verified.isEnabled())
        self.dialog.source_verified.setChecked(True)
        self.dialog.rights_confirmed.setChecked(True)
        self.dialog._review()
        self.assertFalse(self.dialog.review_button.isEnabled())
        self.assertEqual(self.service.calls, [])

    def test_local_approval_requires_both_independent_checks(self):
        self.dialog.candidates_combo.setCurrentIndex(1)
        self.assertFalse(self.dialog.review_button.isEnabled())
        self.dialog.source_verified.setChecked(True)
        self.assertFalse(self.dialog.review_button.isEnabled())
        self.dialog.rights_confirmed.setChecked(True)
        self.assertTrue(self.dialog.review_button.isEnabled())
        self.dialog.review_button.click()
        self.assertEqual(self.service.calls, [("review", "s1", "l1", {"source_verified": True, "rights_confirmed": True})])

    def test_reviewed_candidate_can_be_reselected_after_another_teacher_was_selected(self):
        candidate = self.service.samples[0]["candidates"][1]
        candidate["reviewed"] = True
        candidate["training_allowed"] = False
        self.dialog.refresh()
        self.dialog.candidates_combo.setCurrentIndex(1)
        self.assertTrue(self.dialog.source_verified.isEnabled())
        self.dialog.source_verified.setChecked(True)
        self.dialog.rights_confirmed.setChecked(True)
        self.assertTrue(self.dialog.review_button.isEnabled())
        candidate["training_allowed"] = True
        self.dialog.refresh()
        self.assertFalse(self.dialog.review_button.isEnabled())

    def test_provisional_or_uncertain_source_cannot_be_approved(self):
        for key in ("provisional", "uncertain", "failed"):
            with self.subTest(key=key):
                self.service.samples[0][key] = True
                self.dialog.refresh()
                self.dialog.candidates_combo.setCurrentIndex(1)
                self.dialog.source_verified.setChecked(True)
                self.dialog.rights_confirmed.setChecked(True)
                self.assertFalse(self.dialog.review_button.isEnabled())
                self.service.samples[0][key] = False

    def test_collection_toggle_changes_only_service_and_preserves_display(self):
        self.dialog.collection_enabled.setChecked(False)
        self.assertEqual(self.service.calls, [("enabled", False)])
        self.assertFalse(self.dialog.collection_enabled.isChecked())
        self.assertEqual(self.dialog.samples_list.count(), 2)
        self.dialog.collection_enabled.setChecked(True)
        self.assertEqual(self.service.calls[-1], ("enabled", True))

    def test_refresh_keeps_selected_sample_and_model_when_new_samples_arrive(self):
        self.dialog.samples_list.setCurrentRow(1)
        sample = dict(self.service.samples[0], id="s3", segment_id="u3", candidates=[])
        self.service.samples.insert(0, sample)
        self.dialog.refresh()
        self.assertEqual(self.dialog._selected_sample()["id"], "s2")
        self.assertEqual(self.dialog._selected_candidate()["id"], "l2")

    def test_live_recording_keeps_checks_until_selected_original_or_candidate_changes(self):
        self.dialog.candidates_combo.setCurrentIndex(1)
        self.dialog.source_verified.setChecked(True)
        self.dialog.rights_confirmed.setChecked(True)
        self.service.samples.insert(0, dict(self.service.samples[1], id="new"))
        self.dialog.refresh()
        self.assertTrue(self.dialog.source_verified.isChecked())
        self.assertTrue(self.dialog.rights_confirmed.isChecked())
        self.assertTrue(self.dialog.review_button.isEnabled())
        selected = next(sample for sample in self.service.samples if sample["id"] == "s1")
        selected["source"] = "明日は金曜日です。"
        self.dialog.refresh()
        self.assertFalse(self.dialog.source_verified.isChecked())
        self.assertFalse(self.dialog.rights_confirmed.isChecked())

    def test_compare_runs_service_and_locks_parallel_actions(self):
        self.dialog.compare_button.click()
        self.assertEqual(self.service.calls, [("compare",)])
        self.assertFalse(self.dialog.compare_button.isEnabled())
        self.assertFalse(self.dialog.export_button.isEnabled())
        self.assertFalse(self.dialog.prepare_button.isEnabled())

    def test_cancel_only_visible_during_comparison_and_requested_once(self):
        self.assertTrue(self.dialog.cancel_button.isHidden())
        self.dialog.compare_button.click()
        self.assertTrue(self.dialog.cancel_button.isVisible())
        self.assertTrue(self.dialog.cancel_button.isEnabled())
        self.dialog.cancel_button.click()
        self.dialog._cancel_compare()
        self.assertEqual(self.service.calls, [("compare",), ("cancel",)])
        self.assertFalse(self.dialog.cancel_button.isEnabled())
        self.assertIn("等待本地模型释放资源", self.dialog.result_label.text())
        self.service.busy = False
        self.dialog.refresh()
        self.assertTrue(self.dialog.cancel_button.isHidden())
        self.assertIn("已取消", self.dialog.result_label.text())

    def test_export_insufficiency_is_visible_and_never_reports_training_success(self):
        self.dialog.export_button.click()
        self.assertEqual(self.service.calls, [("export",)])
        self.assertIn("可训练样本不足", self.dialog.result_label.text())
        self.assertIn("尚未导出", self.dialog.result_label.text())
        self.assertFalse(self.dialog.use_button.isEnabled())

    def test_prepare_only_produces_plan(self):
        self.dialog.prepare_button.click()
        self.assertEqual(self.service.calls, [("prepare",)])
        self.assertIn("尚未训练", self.dialog.result_label.text())
        self.assertFalse(self.dialog.use_button.isEnabled())

    def test_real_training_preparation_result_is_concise_and_honest(self):
        result = {"directory": str(self.service.root / "training" / "plan"),
                  "recipe_path": "recipe", "guide_path": "guide", "preflight_path": "preflight",
                  "training_tool_path": "tool", "training_executed": False, "deployment_locked": True,
                  "phase": "prepared_not_trained", "blockers": ["缺少训练权重", "有效样本不足", "缺少训练依赖"]}
        text = self.dialog._result_text(result)
        self.assertIn("训练尚未执行", text)
        self.assertIn("有效样本不足", text)
        self.assertIn("另有 1 项", text)
        self.assertNotIn("preflight_path", text)

    def test_open_directory_only_opens_feature_root(self):
        with patch("fanyiji.specialized_dialog.QDesktopServices.openUrl", return_value=True) as opened:
            self.dialog.open_directory_button.click()
            self.assertEqual(Path(opened.call_args.args[0].toLocalFile()), self.service.root.resolve())

    def test_audio_outside_feature_root_cannot_be_opened(self):
        self.service.samples[0]["audio_paths"] = [str(Path(__file__).resolve())]
        self.dialog.refresh()
        self.assertFalse(self.dialog.audio_button.isEnabled())
        with patch("fanyiji.specialized_dialog.QDesktopServices.openUrl") as opened:
            self.dialog._open_audio()
            opened.assert_not_called()

    def test_audio_in_feature_root_can_be_opened_without_network(self):
        audio = self.service.root / "audio.wav"
        audio.write_bytes(b"example")
        self.service.samples[0]["audio_paths"] = [str(audio)]
        self.dialog.refresh()
        self.assertTrue(self.dialog.audio_button.isEnabled())
        with patch("fanyiji.specialized_dialog.QDesktopServices.openUrl", return_value=True) as opened:
            self.dialog.audio_button.click()
            self.assertEqual(Path(opened.call_args.args[0].toLocalFile()), audio.resolve())

    def test_joined_sentence_can_select_and_open_each_audio_in_source_order(self):
        first, second = self.service.root / "first.wav", self.service.root / "second.wav"
        first.write_bytes(b"first")
        second.write_bytes(b"second")
        self.service.samples[0]["audio_paths"] = [str(first), str(second)]
        self.dialog.refresh()
        self.assertEqual(self.dialog.audio_selector.count(), 2)
        self.assertEqual(self.dialog.audio_selector.itemText(0), "音频片段 1 / 2")
        with patch("fanyiji.specialized_dialog.QDesktopServices.openUrl", return_value=True) as opened:
            self.dialog.audio_button.click()
            self.assertEqual(Path(opened.call_args.args[0].toLocalFile()), first.resolve())
            self.dialog.audio_selector.setCurrentIndex(1)
            self.dialog.audio_button.click()
            self.assertEqual(Path(opened.call_args.args[0].toLocalFile()), second.resolve())
        self.service.samples.insert(0, dict(self.service.samples[1], id="new"))
        self.dialog.refresh()
        self.assertEqual(self.dialog.audio_selector.currentData(), str(second.resolve()))

    def test_selecting_new_sample_never_plays_previous_samples_audio(self):
        audio = self.service.root / "first.wav"
        audio.write_bytes(b"first")
        self.service.samples[0]["audio_paths"] = [str(audio)]
        self.dialog.refresh()
        self.dialog.samples_list.setCurrentRow(1)
        self.assertEqual(self.dialog.audio_selector.count(), 0)
        self.assertFalse(self.dialog.audio_button.isEnabled())

    def test_dialog_close_stops_polling_without_destroying_service(self):
        self.dialog.reject()
        self.assertFalse(self.dialog.refresh_timer.isActive())
        self.assertEqual(self.service.calls, [])


if __name__ == "__main__":
    unittest.main()
