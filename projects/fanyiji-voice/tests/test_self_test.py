"""Self-test acceptance reporting without inference models or audio devices."""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from fanyiji.self_test import run_self_test


class SelfTestReportTests(unittest.TestCase):
    def _run(self, directory: Path, *, asr_text: str | None = None, samples: bool = True):
        directory.mkdir(parents=True, exist_ok=True)
        if samples:
            (directory / "self_test").mkdir()
            for language in ("zh", "ja"):
                with wave.open(str(directory / "self_test" / f"{language}.wav"), "wb") as stream:
                    stream.setnchannels(1)
                    stream.setsampwidth(2)
                    stream.setframerate(16_000)
                    stream.writeframes(b"\0\0" * 800)
        recognizer = Mock()
        recognizer.recognize.side_effect = lambda samples, language: (
            asr_text if asr_text is not None else {"zh": "中文原文。", "ja": "日本語の原文。"}[language]
        )
        translator = Mock()
        translator.translate.side_effect = lambda text, source, target: f"{target}:{text}"
        backend = Mock()
        backend.get_portaudio_version_text.return_value = "test backend"
        resampler = Mock()
        resampler.resample_chunk.return_value = np.zeros(1_600, dtype=np.float32)
        soxr = Mock()
        soxr.ResampleStream.return_value = resampler
        output = directory / "report.json"
        with patch("fanyiji.engines.RecognizerEngine", return_value=recognizer), patch(
            "fanyiji.engines.TranslationEngine", return_value=translator
        ) as translation_class, patch(
            "fanyiji.self_test._memory_snapshot", return_value={"available": False}
        ), patch("fanyiji.audio.AudioCaptureManager._load_backends", return_value=(backend, soxr)), patch(
            "fanyiji.segmentation.SpeechSegmenter"
        ), contextlib.redirect_stdout(io.StringIO()):
            translation_class.REQUIRED_FILES = ("model.bin",)
            result = run_self_test(directory, output)
        return result, json.loads(output.read_text(encoding="utf-8")), recognizer, translator

    def test_text_cases_and_both_audio_chains_are_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            result, report, recognizer, translator = self._run(Path(directory) / "models")
        self.assertEqual(result, 0)
        self.assertEqual(report["status"], "passed")
        self.assertFalse(report["audio_capture"])
        self.assertFalse(report["network_required"])
        self.assertEqual(len(report["results"]), 6)
        self.assertEqual(len(report["audio_results"]), 2)
        self.assertEqual(translator.translate.call_count, 8)
        recognizer.preload.assert_called_once_with(("zh", "ja"))
        for entry, source, target in zip(report["audio_results"], ("zh", "ja"), ("ja", "zh")):
            self.assertEqual((entry["source"], entry["target"]), (source, target))
            self.assertEqual(entry["sample_rate_hz"], 16_000)
            self.assertEqual(entry["duration_seconds"], 0.05)
            self.assertEqual(len(entry["sample_sha256"]), 64)
            translator.translate.assert_any_call(entry["source_text"], source, target)
            self.assertIn("asr_seconds", entry)
            self.assertIn("translation_seconds", entry)
            self.assertIn("processing_seconds", entry)
        self.assertIn("asr_load_seconds", report["timings"])
        self.assertIn("translation_load_seconds", report["timings"])

    def test_missing_bundled_audio_writes_failed_report(self):
        with tempfile.TemporaryDirectory() as directory:
            result, report, _, _ = self._run(Path(directory) / "models", samples=False)
        self.assertEqual(result, 1)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["error"]["stage"], "audio_zh_reading")
        self.assertEqual(report["error"]["type"], "FileNotFoundError")
        self.assertIn("不会联网下载", report["error"]["message"])
        self.assertEqual(len(report["results"]), 6)

    def test_empty_asr_cannot_be_mistaken_for_passed_translation(self):
        with tempfile.TemporaryDirectory() as directory:
            result, report, _, translator = self._run(Path(directory) / "models", asr_text="")
        self.assertEqual(result, 1)
        self.assertEqual(report["error"]["stage"], "audio_zh_recognition")
        self.assertEqual(report["audio_results"][0]["source_text"], "")
        self.assertEqual(translator.translate.call_count, 6)


if __name__ == "__main__":
    unittest.main()
