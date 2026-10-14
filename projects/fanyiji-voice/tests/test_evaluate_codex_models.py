"""Benchmark provenance, isolation and interruption checks; never contact cloud."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("evaluate_codex_models", ROOT / "tools/evaluate_codex_models.py")
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


@pytest.fixture
def corpus(tmp_path):
    path = tmp_path / "input.json"
    path.write_text(json.dumps({"samples": [
        {"id": "ja-1", "source_lang": "ja", "target_lang": "zh", "source_text": "原文一",
         "source_context": "source-only context", "reference_translation": "DO NOT SEND GOLD",
         "expected_assertions": ["DO NOT SEND ASSERTION"]},
        {"id": "zh-1", "source_lang": "zh", "target_lang": "ja", "source_text": "原文二"},
        {"id": "ja-2", "source_lang": "ja", "target_lang": "zh", "source_text": "原文三"},
    ]}), encoding="utf-8")
    return path


class FakeEngine:
    def __init__(self, model, effort, speed, events, failure=None, info_updates=None):
        self.model = model
        self.effort = effort
        self.speed = speed
        self.events = events
        self.failure = failure
        self.info_updates = info_updates or {}
        self.last_request_metrics = {"receive_seconds": 1.25, "auth_seconds": .05}
        self.calls = 0
        events.append(("create", model))

    def preload(self):
        self.events.append(("preload", self.model))

    @property
    def backend_info(self):
        return {"backend": "codex_cli", "model_id": self.model, "auth_mode": "chatgpt",
                "subscription_only": True, "reasoning_effort": self.effort, "speed": self.speed,
                "account_email": "private@example.invalid", "helper_pid": 123,
                "available_models": [{"model": self.model, "displayName": self.model,
                                      "account_secret": "private-token",
                                      "supportedReasoningEfforts": [{"reasoningEffort": "low"}]}],
                **self.info_updates}

    def translate(self, text, source, target, *, context):
        self.calls += 1
        self.events.append(("translate", self.model, text, source, target, context))
        if self.failure:
            value = self.failure(self.calls)
            if value:
                raise value
        return "translated fixture"

    def close(self):
        self.events.append(("close", self.model))


def factory(events, **extras):
    return lambda **kwargs: FakeEngine(events=events, **extras, **kwargs)


def test_plan_does_not_create_output_or_engine(corpus, tmp_path):
    output = tmp_path / "run"
    def forbidden(**kwargs):
        pytest.fail("Planning must never create an inference engine")
    assert TOOL.run(corpus, ["model-a"], output, engine_factory=forbidden) == 0
    assert not output.exists()


def test_serial_isolation_same_order_no_gold_prompt_and_sanitized_metadata(corpus, tmp_path):
    events = []
    output = tmp_path / "run"
    assert TOOL.run(corpus, ["model-a", "model-b"], output, execute=True, engine_factory=factory(events)) == 0
    assert events.index(("close", "model-a")) < events.index(("create", "model-b"))
    calls = {model: [event for event in events if event[0] == "translate" and event[1] == model]
             for model in ("model-a", "model-b")}
    assert [event[2] for event in calls["model-a"]] == [event[2] for event in calls["model-b"]]
    assert len(calls["model-a"]) == 4  # one unscored warmup, three scored cases
    assert "DO NOT SEND" not in repr(events)
    assert any(event[-1] == "source-only context" for event in calls["model-a"])
    for path in output.rglob("*.json*"):
        if path.name == "corpus.json":
            continue
        text = path.read_text(encoding="utf-8")
        assert "private@example.invalid" not in text
        assert "private-token" not in text
        assert "helper_pid" not in text
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert all(row["successful"] == 3 and len(row["warmup"]) == 1 for row in summary["models"])
    assert all(row["accuracy_score"] is None for row in summary["models"])


def test_resume_never_reruns_completed_samples_or_warmup(corpus, tmp_path):
    output = tmp_path / "run"
    TOOL.run(corpus, ["model-a"], output, execute=True, engine_factory=factory([]))
    events = []
    assert TOOL.run(corpus, ["model-a"], output, execute=True, engine_factory=factory(events)) == 0
    assert not events
    assert len(TOOL.read_rows(output / "round-01/model-a/results.jsonl")) == 3


def test_two_consecutive_transport_failures_stop_without_retry_and_close(corpus, tmp_path):
    output = tmp_path / "run"
    events = []
    failed = lambda number: ConnectionError("sensitive remote payload") if number > 1 else None
    assert TOOL.run(corpus, ["model-a", "model-b"], output, execute=True,
                    engine_factory=factory(events, failure=failed)) == 1
    rows = TOOL.read_rows(output / "round-01/model-a/results.jsonl")
    assert len(rows) == 2 and all(row["error_kind"] == "transport_or_remote" for row in rows)
    assert ("close", "model-a") in events and ("create", "model-b") not in events
    assert "sensitive remote payload" not in (output / "round-01/model-a/results.jsonl").read_text()
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["models"][0]["status"] == "partial"
    assert summary["models"][0]["median_seconds"] is None


def test_protocol_failures_are_not_transport_and_completed_error_not_retried(corpus, tmp_path):
    output = tmp_path / "run"
    events = []
    failed = lambda number: RuntimeError("Codex 返回了空译文") if number > 1 else None
    assert TOOL.run(corpus, ["model-a", "model-b"], output, execute=True,
                    engine_factory=factory(events, failure=failed)) == 1
    assert len(TOOL.read_rows(output / "round-01/model-a/results.jsonl")) == 3
    assert ("create", "model-b") in events
    assert all(row["error_kind"] == "response_protocol"
               for row in TOOL.read_rows(output / "round-01/model-a/results.jsonl"))
    later = []
    assert TOOL.run(corpus, ["model-a", "model-b"], output, execute=True, engine_factory=factory(later)) == 1
    assert not later


@pytest.mark.parametrize("updates", [{"auth_mode": "api_key"}, {"model_id": "wrong-model"},
                                     {"reasoning_effort": "high"}, {"speed": "fast"},
                                     {"available_models": []}])
def test_actual_backend_mismatch_blocks_inference(corpus, tmp_path, updates):
    events = []
    assert TOOL.run(corpus, ["model-a", "model-b"], tmp_path / "run", execute=True,
                    engine_factory=factory(events, info_updates=updates)) == 1
    assert not any(event[0] == "translate" for event in events)
    assert ("close", "model-a") in events
    assert ("create", "model-b") not in events


def test_partial_resume_only_missing_cases(corpus, tmp_path):
    output = tmp_path / "run"
    TOOL.run(corpus, ["model-a"], output, execute=True, engine_factory=factory([]))
    path = output / "round-01/model-a/results.jsonl"
    rows = TOOL.read_rows(path)
    path.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    events = []
    assert TOOL.run(corpus, ["model-a"], output, execute=True, engine_factory=factory(events)) == 0
    calls = [event for event in events if event[0] == "translate"]
    assert len(calls) == 2 and all(event[2] != TOOL.WARMUP["source_text"] for event in calls)
    assert len(TOOL.read_rows(path)) == 3


def test_manifest_and_corpus_hash_change_rejected(corpus, tmp_path):
    output = tmp_path / "run"
    TOOL.run(corpus, ["model-a"], output, execute=True, engine_factory=factory([]))
    with pytest.raises(ValueError, match="manifest differs"):
        TOOL.run(corpus, ["model-b"], output, execute=True, engine_factory=factory([]))
    corpus.with_suffix(".sha256").write_text("0" * 64, encoding="ascii")
    with pytest.raises(ValueError, match="checksum mismatch"):
        TOOL.read_corpus(corpus)


def test_second_round_reverse_models_keeps_same_case_order(corpus, tmp_path):
    events = []
    assert TOOL.run(corpus, ["model-a", "model-b"], tmp_path / "run", rounds=2, execute=True,
                    engine_factory=factory(events)) == 0
    assert [event[1] for event in events if event[0] == "create"] == ["model-a", "model-b", "model-b", "model-a"]


def test_blind_export_hides_model_timing_and_protects_review(corpus, tmp_path):
    output = tmp_path / "run"
    TOOL.run(corpus, ["model-a", "model-b"], output, execute=True, blind_export=True, engine_factory=factory([]))
    path = output / "blind-review.json"
    text = path.read_text(encoding="utf-8")
    assert "model-a" not in text and "model-b" not in text
    assert "seconds" not in text and "rpc_metrics" not in text
    assert len(json.loads(text)["samples"]) == 6
    samples, _ = TOOL.read_corpus(corpus)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    with pytest.raises(FileExistsError, match="Preserving"):
        TOOL.export_blind(output, samples, manifest)
    assert path.read_text(encoding="utf-8") == text


def test_latency_statistics_exclude_errors_but_total_counts_them():
    rows = [{"status": "success", "seconds": value} for value in range(1, 11)]
    rows.append({"status": "error", "seconds": 100})
    report = TOOL.latency_summary(rows)
    assert report["successful"] == 10 and report["failed"] == 1
    assert report["median_seconds"] == 5.5 and report["p90_seconds_nearest_rank"] == 9
    assert report["success_seconds_total"] == 55
    assert report["attempt_seconds_total_including_failures"] == 155


def test_truncated_jsonl_rejected_without_modification(tmp_path):
    path = tmp_path / "results.jsonl"
    path.write_text('{"id": "unfinished"', encoding="utf-8")
    original = path.read_bytes()
    with pytest.raises(ValueError, match="Incomplete result"):
        TOOL.read_rows(path)
    assert path.read_bytes() == original


@pytest.mark.parametrize("changes", [{"id": ""}, {"source_text": ""}, {"source_lang": "en"},
                                    {"source_context": 123}])
def test_invalid_corpus_rejected(corpus, changes):
    value = json.loads(corpus.read_text(encoding="utf-8"))
    value["samples"][0].update(changes)
    corpus.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError):
        TOOL.read_corpus(corpus)


@pytest.mark.parametrize("exc,kind", [(RuntimeError("Codex 订阅额度已用完"), "quota"),
                                     (RuntimeError("Codex 登录已失效"), "authentication"),
                                     (RuntimeError("Codex 返回的实际模型与所选模型不同"), "configuration"),
                                     (RuntimeError("Codex 返回的实际思考强度与所选配置不同"), "configuration")])
def test_error_categories(exc, kind):
    assert TOOL.error_kind(exc) == kind


def test_generic_remote_failure_is_not_mislabeled_as_confirmed_quota():
    failure = RuntimeError("Codex 翻译失败，请检查网络、账号额度或 CLI 版本")
    assert TOOL.error_kind(failure) == "transport_or_remote"


def test_generic_rpc_failure_does_not_claim_confirmed_authentication_failure():
    failure = RuntimeError("Codex 请求失败（turn/start），请检查 CLI 版本、登录和账户权限")
    assert TOOL.error_kind(failure) == "transport_or_remote"
