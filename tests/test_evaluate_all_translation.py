"""Narrow tooling checks only; no model inference or network connections."""
import importlib.util
import hashlib
import json
from pathlib import Path
import socket
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("evaluate_all_translation", ROOT / "tools/evaluate_all_translation.py")
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


@pytest.fixture
def frozen_corpus(tmp_path):
    """Tooling tests must run without private, Git-ignored evaluation data."""
    path = tmp_path / "corpus.json"
    samples = [
        {"id": f"fixture-{source}-{index}", "source_lang": source,
         "target_lang": target, "source_text": "synthetic protocol fixture",
         "human_expected": ["fixture fact"]}
        for source, target in (("zh", "ja"), ("ja", "zh")) for index in range(30)
    ]
    path.write_text(json.dumps({"samples": samples, "audio_ground_truth": False,
                               "automatic_accuracy_score": False}), encoding="utf-8")
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(".sha256").write_text(checksum, encoding="ascii")
    return path


def test_frozen_corpus_has_balanced_fact_controls_and_no_audio_ground_truth(frozen_corpus):
    value = TOOL.read_corpus(frozen_corpus)
    assert len(value["samples"]) == 60
    assert not value["audio_ground_truth"] and not value["automatic_accuracy_score"]
    assert TOOL.CONTROL_PROVENANCE["authored_by"] == "assistant"
    assert not TOOL.CONTROL_PROVENANCE["user_human_translation_ground_truth"]


def test_modified_corpus_is_rejected_even_if_it_still_has_sixty_items(tmp_path, frozen_corpus):
    corpus = tmp_path / "corpus.json"
    corpus.write_bytes(frozen_corpus.read_bytes())
    corpus.with_suffix(".sha256").write_text("0" * 64, encoding="ascii")
    with pytest.raises(ValueError, match="checksum"):
        TOOL.read_corpus(corpus)


def test_python_remote_connection_is_blocked_without_contacting_remote_host():
    with TOOL.loopback_only() as attempts, socket.socket() as connection:
        with pytest.raises(RuntimeError, match="owned local"):
            connection.connect(("example.com", 443))
        assert attempts == ["non_loopback_connection_blocked"]


def test_production_request_observer_passes_body_through_and_redacts_alias(tmp_path):
    response = {"choices": [{"finish_reason": "stop"}], "usage": {"completion_tokens": 3}}
    request = Mock(return_value=response)
    engine = SimpleNamespace(_start_runtime=Mock(), _request=request)
    report = {"model_id": "hy7"}
    TOOL.audit_gguf(engine, report, tmp_path)
    payload = {"model": "private-owned-alias", "temperature": 0, "top_k": 0,
               "top_p": 1, "messages": [{"role": "user", "content": "test"}]}
    assert engine._request("POST", "/v1/chat/completions", payload) == response
    request.assert_called_once_with("POST", "/v1/chat/completions", payload)
    assert payload["model"] == "private-owned-alias"
    assert report["last_actual_request"]["model"] == "<owned-alias>"
    assert report["last_response_metadata"]["finish_reason"] == "stop"


def test_blind_export_hides_models_timings_and_preserves_prior_review(tmp_path, frozen_corpus):
    corpus = TOOL.read_corpus(frozen_corpus)
    for model in ("hy7", "qwen17"):
        TOOL.save(tmp_path / f"{model}.json", {
            "status": "completed", "model_id": model, "input_sha256": TOOL.digest(frozen_corpus),
            "samples": [{**x, "translation": "synthetic fixture", "seconds": 123} for x in corpus["samples"]],
        })
    TOOL.export_blind(tmp_path, frozen_corpus)
    review = json.loads((tmp_path / "mt-blind-review.json").read_text(encoding="utf-8"))
    assert len(review["samples"]) == 120
    assert all("model_id" not in item and "seconds" not in item for item in review["samples"])
    assert all(item["overall"] is None for item in review["samples"])
    before = (tmp_path / "mt-blind-review.json").read_bytes()
    with pytest.raises(FileExistsError, match="Preserving"):
        TOOL.export_blind(tmp_path, frozen_corpus)
    assert (tmp_path / "mt-blind-review.json").read_bytes() == before
