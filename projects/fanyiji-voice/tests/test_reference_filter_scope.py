import importlib.util
import json
from pathlib import Path
from unittest.mock import patch


def test_all_filtered_segments_are_not_claimed_as_production_asr_comparison(tmp_path):
    spec = importlib.util.spec_from_file_location("reference_scope_tool", Path(__file__).resolve().parents[1] / "tools/reference_whisper.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    capture = tmp_path / "results.json"
    capture.write_text(json.dumps({"samples": [{"id": "1:system:music", "source_lang": "ja",
                                                "speech_filter": {"action": "skip"}}]}), encoding="utf-8")
    output = tmp_path / "reference.json"
    with patch.object(tool, "verify_models", side_effect=AssertionError("No model inference is needed")):
        tool.compare(capture, tmp_path / "no-models", output, 1, "turbo")
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "completed"
    assert report["samples"] == []
    assert report["production_filter_skipped_ids"] == ["1:system:music"]
    assert report["total_processing_seconds"] == 0
