import json
import time
from types import SimpleNamespace

import pytest

from fanyiji import codex_engine
from fanyiji.codex_engine import CodexTranslationEngine, _Session
from test_codex_engine import server


def test_request_exposes_first_output_and_completion_as_parts_of_receive(server):
    engine = CodexTranslationEngine()
    engine.translate("明日です。", "ja", "zh")
    metrics = engine.last_request_metrics
    assert 0 <= metrics["first_output_seconds"] <= metrics["receive_seconds"]
    assert 0 <= metrics["output_completion_seconds"] <= metrics["receive_seconds"]
    assert metrics["first_output_seconds"] + metrics["output_completion_seconds"] <= metrics["receive_seconds"]
    assert "明日" not in json.dumps(metrics)
    engine.close()


def test_other_turn_and_empty_delta_do_not_mark_first_output(monkeypatch):
    engine = CodexTranslationEngine()
    session = _Session(SimpleNamespace(), None, None)
    payload = json.dumps({"translation": "明天"})
    for thread, turn, delta in (("old", "t", "old"), ("s", "old", "old"), ("s", "t", ""), ("s", "t", payload)):
        session.events.put({"method": "item/agentMessage/delta", "params": {
            "threadId": thread, "turnId": turn, "itemId": "i", "delta": delta}})
    session.events.put({"method": "turn/completed", "params": {"threadId": "s",
        "turn": {"id": "t", "status": "completed"}}})
    clock = iter([10., 12., 13.])
    monkeypatch.setattr(codex_engine.time, "perf_counter", lambda: next(clock))
    assert engine._receive_payload(session, "s", "t", time.monotonic() + 1) == {"translation": "明天"}
    assert session.response_first_output_seconds == 2
    assert session.response_output_completion_seconds == 1


def test_timeout_before_output_does_not_invent_first_output():
    engine = CodexTranslationEngine()
    session = _Session(SimpleNamespace(), None, None)
    with pytest.raises(TimeoutError):
        engine._receive_payload(session, "s", "t", time.monotonic() - 1)
    assert session.response_first_output_seconds is None
    assert session.response_output_completion_seconds is None
