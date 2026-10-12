import json
from types import SimpleNamespace

import numpy as np

from fanyiji.config import AppConfig
from fanyiji.diagnostics import SessionRecorder


def test_late_translation_keeps_raw_output_but_cannot_override_display_revision(tmp_path):
    recorder = SessionRecorder(tmp_path / "capture", AppConfig())
    for identifier in ("a", "b"):
        recorder.on_segment(SimpleNamespace(channel="system", segment_id=identifier,
            samples=np.zeros(1600, np.float32), direction=("ja", "zh"),
            start_sample=0, end_sample=1600, generation=1, ended_by="silence"), True)
    recorder.event({"kind": "recognition", "segment_id": "a", "text": "森の。", "quality": {}})
    recorder.event({"kind": "original", "segment_id": "a", "text": "森の妖精。",
                    "revision": 2, "source_segment_ids": ["a", "b"]})
    recorder.event({"kind": "translation", "segment_id": "a", "text": "旧译文",
                    "revision": 1, "source_segment_ids": ["a"]})
    recorder.snapshot()
    first = recorder.samples["a"]
    assert first["production_text"] == "森の。"
    assert first["display_original"] == "森の妖精。"
    assert "display_translation" not in first
    recorder.event({"kind": "translation", "segment_id": "a", "text": "森林精灵。",
                    "revision": 2, "source_segment_ids": ["a", "b"]})
    recorder.finish()
    report = json.loads((recorder.directory / "results.json").read_text(encoding="utf-8"))
    first, second = report["samples"]
    assert first["display_translation"] == "森林精灵。"
    assert first["production_translation"] == "旧译文"
    assert first["translated_source_segment_ids"] == ["a", "b"]
    assert second["translation_group_id"] == "a"
    assert first["ended_by"] == "silence"


def test_confirmation_before_queued_translation_stays_final_in_saved_report(tmp_path):
    recorder = SessionRecorder(tmp_path / "capture", AppConfig())
    recorder.on_segment(SimpleNamespace(channel="system", segment_id="a",
        samples=np.zeros(1600, np.float32), direction=("ja", "zh")), True)
    recorder.event({"kind": "original", "segment_id": "a", "text": "森の。", "revision": 1})
    recorder.event({"kind": "subtitle_final", "segment_id": "a", "revision": 1})
    recorder.event({"kind": "translation", "segment_id": "a", "text": "森林的。",
                    "revision": 1, "provisional": True})
    recorder.finish()
    assert recorder.samples["a"]["translation_provisional"] is False
    assert recorder.samples["a"]["display_translation_valid"] is True


def test_normalized_phrase_keeps_raw_recognition_and_assembly_provenance(tmp_path):
    recorder = SessionRecorder(tmp_path / "capture", AppConfig())
    recorder.on_segment(SimpleNamespace(channel="system", segment_id="a",
        samples=np.zeros(1600, np.float32), direction=("ja", "zh")), True)
    recorder.event({"kind": "recognition", "segment_id": "a", "text": "森の。", "quality": {}})
    recorder.event({"kind": "sentence_join", "segment_id": "a",
        "text": "森の妖精です。", "raw_source_texts": ("森の。", "妖精です。"), "uncertain": True,
        "assembly_text": "森の妖精です。", "boundary_normalized": True,
        "joint_recognition": {"performed": False}, "gap_samples": 4000,
        "gap_filled": False, "source_segment_ids": ("a", "b")})
    recorder.event({"kind": "original", "segment_id": "a", "text": "森の妖精です。",
        "revision": 2, "source_segment_ids": ("a", "b"), "uncertain": True})
    recorder.finish()
    report = json.loads((recorder.directory / "results.json").read_text(encoding="utf-8"))
    sample = report["samples"][0]
    assert sample["production_text"] == "森の。"
    assert sample["sentence_assembly"]["raw_source_texts"] == ["森の。", "妖精です。"]
    assert sample["sentence_assembly"]["assembly_text"] == "森の妖精です。"
    assert sample["sentence_assembly"]["boundary_normalized"] is True
    assert sample["sentence_assembly"]["gap_filled"] is False
    assert sample["sentence_assembly"]["uncertain"] is True
    assert sample["display_original_uncertain"] is True
