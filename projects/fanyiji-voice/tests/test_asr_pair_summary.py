"""Keep source-latency comparisons fair when one model joins two segments."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from summarize_asr_pair import distribution, source_finals


def test_joined_final_retains_both_original_waits_without_inventing_requests():
    report = {"capture_started_monotonic": 100,
              "segments": [{"segment_id": "a", "start_sample": 0, "end_sample": 16000},
                           {"segment_id": "b", "start_sample": 32000, "end_sample": 48000}],
              "originals": [{"channel": "system", "segment_id": "a", "revision": 2,
                             "context_generation": 0, "source_segment_ids": ["a", "b"]}],
              "latency_records": [{"valid": True, "stage": "final", "channel": "system",
                                   "segment_id": "a", "revision": 2, "context_generation": 0,
                                   "timing": {"ui_painted_at": 110}}]}
    rows = source_finals(report)
    assert len(rows) == 2 and len(report["latency_records"]) == 1
    assert [(r["start_to_final"], r["end_to_final"]) for r in rows] == [(10, 9), (8, 7)]


def test_missing_paint_is_not_a_fast_source_success():
    report = {"capture_started_monotonic": 100,
              "segments": [{"segment_id": "a", "start_sample": 0, "end_sample": 16000}],
              "originals": [], "latency_records": []}
    assert source_finals(report) == []
    assert distribution([]) == {"count": 0, "p50": None, "p95": None, "max": None}


def test_small_sample_p95_is_nearest_rank_and_retains_slow_tail():
    assert distribution([1, 2, 3, 20]) == {"count": 4, "p50": 2.5, "p95": 20, "max": 20}
