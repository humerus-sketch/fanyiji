"""Conversation lifecycle tests use mocked owned transports, never live accounts."""
import json
import threading
import time

import pytest

from fanyiji import codex_engine
from fanyiji.codex_engine import CodexTranslationEngine
from test_codex_engine import batch_items, methods, server


def request_data(process, index=-1):
    return json.loads(methods(process, "turn/start")[index]["input"][0]["text"])


def wait_for(predicate):
    deadline = time.monotonic() + 2
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.002)
    assert predicate()


def test_same_key_reuses_thread_auth_is_checked_each_turn_and_bytes_are_counted_once(server):
    processes, _ = server
    engine = CodexTranslationEngine()
    engine.translate("明日会議です。", "ja", "zh", conversation_key="loopback")
    first_metrics = engine.last_request_metrics
    engine.translate("午後3時です。", "ja", "zh", context="明日会議です。", conversation_key="loopback")
    process = processes[0]
    turns = methods(process, "turn/start")
    assert turns[0]["threadId"] == turns[1]["threadId"]
    assert len(methods(process, "thread/start")) == 2  # empty preload plus one conversation
    assert len(methods(process, "thread/unsubscribe")) == 1
    assert process.account_read_count == 3  # startup plus every translating turn
    assert first_metrics["thread_reused"] == 0 and first_metrics["thread_turn"] == 1
    metrics = engine.last_request_metrics
    assert metrics["thread_reused"] == 1 and metrics["thread_turn"] == 2
    assert metrics["thread_seconds"] == metrics["unsubscribe_seconds"] == 0
    system_bytes = len(methods(process, "thread/start")[-1]["baseInstructions"].encode("utf-8"))
    input_bytes = sum(len(turn["input"][0]["text"].encode("utf-8")) for turn in turns)
    output_bytes = 2 * len(json.dumps({"translation": process.translation}).encode("utf-8"))
    assert metrics["thread_history_bytes"] == system_bytes + input_bytes + output_bytes
    info = json.dumps(engine.backend_info, ensure_ascii=False)
    assert "loopback" not in info and "明日会議です。" not in info and "private@example.invalid" not in info
    assert engine.backend_info["conversation_policy"]["budget_unit"] == "utf8_text_bytes"
    engine.close()
    assert not engine._conversations and process.returncode == 0


def test_single_batch_single_use_one_stable_prompt_and_per_turn_schema(server):
    processes, _ = server
    engine = CodexTranslationEngine()
    engine.translate("明日の会議です。", "ja", "zh", conversation_key="system")
    process = processes[0]
    process.raw_override = json.dumps({"translations": [
        {"id": "numbers", "translation": "数字"}, {"id": "negation", "translation": "否定"}]})
    assert engine.translate_batch(batch_items(), "ja", "zh", conversation_key="system") == {
        "numbers": "数字", "negation": "否定"}
    process.raw_override = None
    engine.translate("その後です。", "ja", "zh", conversation_key="system")
    turns = methods(process, "turn/start")
    assert len({turn["threadId"] for turn in turns}) == 1
    assert turns[0]["outputSchema"] == turns[2]["outputSchema"] == codex_engine._OUTPUT_SCHEMA
    assert turns[1]["outputSchema"]["properties"]["translations"]["items"]["properties"]["id"]["enum"] == [
        "numbers", "negation"]
    assert len(methods(process, "thread/start")) == 2
    assert engine.last_request_metrics["thread_turn"] == 3
    engine.close()


def test_dynamic_terms_change_request_data_without_changing_system_or_reusing_reference_terms(server, monkeypatch):
    processes, _ = server
    calls = []

    def hints(source, target, texts):
        texts = tuple(texts)
        calls.append((source, target, texts))
        return "conditional first" if "一" in texts[0] else "conditional second"

    monkeypatch.setattr(codex_engine, "livestream_term_hints", hints)
    engine = CodexTranslationEngine()
    engine.translate("一。", "ja", "zh", context="history only", conversation_key="stream")
    engine.translate("二。", "ja", "zh", context="different history", conversation_key="stream")
    process = processes[0]
    assert calls == [("ja", "zh", ("一。",)), ("ja", "zh", ("二。",))]
    assert request_data(process, 0)["term_hints"] == "conditional first"
    assert request_data(process, 1)["term_hints"] == "conditional second"
    instructions = methods(process, "thread/start")[-1]["baseInstructions"]
    assert "conditional first" not in instructions and "conditional second" not in instructions
    assert "they must never override its meaning" in instructions
    assert methods(process, "turn/start")[0]["threadId"] == methods(process, "turn/start")[1]["threadId"]
    engine.close()


def test_twentieth_turn_is_allowed_twenty_first_rotates_and_carries_current_reference(server):
    processes, _ = server
    engine = CodexTranslationEngine()
    for index in range(20):
        engine.translate(f"文{index}", "ja", "zh", conversation_key="system")
    process = processes[0]
    first_thread = methods(process, "turn/start")[0]["threadId"]
    assert {turn["threadId"] for turn in methods(process, "turn/start")} == {first_thread}
    assert engine.last_request_metrics["thread_turn"] == 20
    engine.translate("文20", "ja", "zh", context="renewed reference", conversation_key="system")
    assert methods(process, "turn/start")[-1]["threadId"] != first_thread
    assert methods(process, "thread/unsubscribe")[-1] == {"threadId": first_thread}
    assert request_data(process)["reference_context"] == "renewed reference"
    assert engine.last_request_metrics["thread_reset"] == 1
    assert engine.last_request_metrics["thread_turn"] == 1
    engine.close()


def test_age_budget_rotates_at_120_seconds(server, monkeypatch):
    processes, _ = server
    engine = CodexTranslationEngine()
    engine.translate("最初", "ja", "zh")
    monotonic = time.monotonic
    monkeypatch.setattr(codex_engine.time, "monotonic", lambda: monotonic() + 120)
    engine.translate("次", "ja", "zh")
    turns = methods(processes[0], "turn/start")
    assert turns[0]["threadId"] != turns[1]["threadId"]
    assert engine.last_request_metrics["thread_reset"] == 1
    engine.close()


def test_utf8_byte_budget_rotates_before_next_input_would_exceed_it(server):
    processes, _ = server
    engine = CodexTranslationEngine()
    engine.translate("文" * 1000, "ja", "zh")
    first_budget = engine.last_request_metrics["thread_history_bytes"]
    process = processes[0]
    next_data = dict(request_data(process), current_text="字" * 1000)
    next_size = len(json.dumps(next_data, ensure_ascii=False).encode("utf-8"))
    engine.MAX_CONVERSATION_BYTES = int(first_budget + next_size - 1)
    engine.translate("字" * 1000, "ja", "zh")
    turns = methods(process, "turn/start")
    assert turns[0]["threadId"] != turns[1]["threadId"]
    assert engine.last_request_metrics["thread_reset"] == 1
    assert engine.last_request_metrics["thread_history_bytes"] <= engine.MAX_CONVERSATION_BYTES
    engine.close()


def test_oversize_one_request_preserves_all_original_and_forces_next_rotation(server):
    processes, _ = server
    engine = CodexTranslationEngine()
    source = "文" * 4096
    engine.translate(source, "ja", "zh", context="valid reference")
    process = processes[0]
    assert request_data(process)["current_text"] == source
    assert request_data(process)["reference_context"] == "valid reference"
    assert engine.last_request_metrics["thread_history_bytes"] > engine.MAX_CONVERSATION_BYTES
    engine.translate("次", "ja", "zh")
    turns = methods(process, "turn/start")
    assert turns[0]["threadId"] != turns[1]["threadId"]
    assert engine.last_request_metrics["thread_reset"] == 1
    engine.close()


def test_actual_output_bytes_are_included_and_force_rotation_after_completion(server):
    processes, settings = server
    settings["translation"] = "译" * 1000
    engine = CodexTranslationEngine()
    engine.translate("一", "ja", "zh")
    assert engine.last_request_metrics["thread_history_bytes"] < engine.MAX_CONVERSATION_BYTES
    engine.translate("二", "ja", "zh")
    assert engine.last_request_metrics["thread_reused"] == 1
    assert engine.last_request_metrics["thread_history_bytes"] > engine.MAX_CONVERSATION_BYTES
    engine.translate("三", "ja", "zh")
    turns = methods(processes[0], "turn/start")
    assert turns[0]["threadId"] == turns[1]["threadId"] != turns[2]["threadId"]
    assert engine.last_request_metrics["thread_reset"] == 1
    engine.close()


@pytest.mark.parametrize("change", ["source", "target", "generation", "model", "effort", "speed"])
def test_signature_change_rotates_instead_of_inheriting_history(server, change):
    processes, _ = server
    engine = CodexTranslationEngine()
    engine.translate("最初", "ja", "zh", conversation_key="lane", context_generation=3)
    source, target, generation = "ja", "zh", 3
    if change == "source":
        source = "mixed"
    elif change == "target":
        source, target = "zh", "ja"
    elif change == "generation":
        generation = 4
    elif change == "model":
        engine._model = "other-model"
    elif change == "effort":
        engine._reasoning_effort = "low"
    else:
        engine.requested_speed = "fast"
    engine.translate("次", source, target, conversation_key="lane", context_generation=generation)
    process = processes[0]
    turns = methods(process, "turn/start")
    assert turns[0]["threadId"] != turns[1]["threadId"]
    assert engine.last_request_metrics["thread_reset"] == 1
    if change == "model":
        assert methods(process, "thread/start")[-1]["model"] == "other-model"
    if change == "effort":
        assert turns[-1]["effort"] == "low"
    if change == "speed":
        assert turns[-1]["serviceTier"] == "fast"
    engine.close()


def test_four_keys_are_isolated_and_least_recently_used_thread_is_unsubscribed(server):
    processes, _ = server
    engine = CodexTranslationEngine()
    threads = {}
    for key in ("mic", "system", "third", "fourth"):
        engine.translate("文", "ja", "zh", conversation_key=key)
        threads[key] = methods(processes[0], "turn/start")[-1]["threadId"]
    assert len(set(threads.values())) == 4
    engine.translate("更新", "ja", "zh", conversation_key="mic")
    assert methods(processes[0], "turn/start")[-1]["threadId"] == threads["mic"]
    engine.translate("新規", "ja", "zh", conversation_key="fifth")
    assert len(engine._conversations) == 4 and "system" not in engine._conversations
    assert methods(processes[0], "thread/unsubscribe")[-1] == {"threadId": threads["system"]}
    assert engine.last_request_metrics["thread_reset"] == 1
    engine.translate("保持", "ja", "zh", conversation_key="mic")
    assert methods(processes[0], "turn/start")[-1]["threadId"] == threads["mic"]
    engine.close()


def test_reset_is_local_only_and_does_not_invalidate_another_channel(server):
    processes, _ = server
    engine = CodexTranslationEngine()
    engine.translate("一", "ja", "zh", conversation_key="mic")
    engine.translate("二", "ja", "zh", conversation_key="system")
    process = processes[0]
    first_turns = methods(process, "turn/start")
    count = len(process.requests)
    engine.reset_conversation("mic")
    engine.reset_conversation("unknown")
    assert len(process.requests) == count
    engine.translate("三", "ja", "zh", conversation_key="system")
    engine.translate("四", "ja", "zh", context="fresh reference", conversation_key="mic")
    turns = methods(process, "turn/start")
    assert turns[2]["threadId"] == first_turns[1]["threadId"]
    assert turns[3]["threadId"] != first_turns[0]["threadId"]
    assert request_data(process)["reference_context"] == "fresh reference"
    engine.close()


def test_reset_during_thread_creation_returns_immediately_and_next_turn_rebuilds(server, monkeypatch):
    processes, _ = server
    engine = CodexTranslationEngine()
    engine.preload()
    entered, released = threading.Event(), threading.Event()
    rpc = engine._rpc

    def blocked(session, method, params, deadline):
        if method == "thread/start":
            entered.set()
            assert released.wait(2)
        return rpc(session, method, params, deadline)

    monkeypatch.setattr(engine, "_rpc", blocked)
    results = []
    worker = threading.Thread(target=lambda: results.append(engine.translate(
        "文", "ja", "zh", conversation_key="system")))
    worker.start()
    assert entered.wait(2)
    began = time.perf_counter()
    engine.reset_conversation("system")
    assert time.perf_counter() - began < 0.1
    released.set()
    worker.join(2)
    assert results == ["明天开会。"] and not worker.is_alive()
    engine.translate("次", "ja", "zh", conversation_key="system")
    turns = methods(processes[0], "turn/start")
    assert turns[0]["threadId"] != turns[1]["threadId"]
    engine.close()


def test_reset_and_close_do_not_wait_for_an_unanswered_turn_start_rpc(server):
    processes, _ = server
    engine = CodexTranslationEngine()
    engine.translate("一", "ja", "zh")
    process = processes[0]
    received = threading.Event()
    original = process.receive

    def receive(request):
        if request.get("method") == "turn/start":
            process.requests.append(request)
            received.set()
            return  # Network response deliberately never arrives.
        original(request)

    process.receive = receive
    failures = []

    def work():
        try:
            engine.translate("二", "ja", "zh")
        except RuntimeError as error:
            failures.append(str(error))

    worker = threading.Thread(target=work)
    worker.start()
    assert received.wait(2)
    began = time.perf_counter()
    engine.reset_conversation()
    assert time.perf_counter() - began < 0.1
    began = time.perf_counter()
    engine.close()
    assert time.perf_counter() - began < 0.5
    worker.join(2)
    assert not worker.is_alive() and failures and "停止" in failures[0]
    assert process.returncode == 0 and not engine._conversations
    assert len(methods(process, "turn/start")) == 2


@pytest.mark.parametrize("identity", [{"email": "changed@example.invalid"}, {"id": "account-next"}])
def test_other_chatgpt_identity_is_rejected_before_turn_and_never_exported(server, identity):
    processes, settings = server
    if "id" in identity:
        settings["account"] = {"type": "chatgpt", "id": "account-original"}
    engine = CodexTranslationEngine()
    engine.translate("一", "ja", "zh")
    process = processes[0]
    process.account = {"type": "chatgpt", "planType": "plus", **identity}
    with pytest.raises(RuntimeError, match="登录账号已切换") as error:
        engine.translate("二", "ja", "zh")
    assert len(methods(process, "turn/start")) == 1
    assert process.returncode == 0 and not engine._conversations
    assert engine._account_fingerprint is None
    metadata = str(error.value) + json.dumps(engine.backend_info)
    assert all(value not in metadata for value in identity.values())
    engine.translate("三", "ja", "zh")
    assert len(processes) == 2 and len(methods(processes[1], "turn/start")) == 1
    engine.close()


def test_missing_stable_account_identity_conservatively_uses_new_thread(server):
    processes, settings = server
    settings["account"] = {"type": "chatgpt", "planType": "plus"}
    engine = CodexTranslationEngine()
    engine.translate("一", "ja", "zh")
    engine.translate("二", "ja", "zh")
    turns = methods(processes[0], "turn/start")
    assert turns[0]["threadId"] != turns[1]["threadId"]
    assert engine.last_request_metrics["thread_reused"] == 0
    engine.close()


def test_email_capitalization_does_not_report_an_account_switch(server):
    processes, _ = server
    engine = CodexTranslationEngine()
    engine.translate("一", "ja", "zh")
    processes[0].account["email"] = " PRIVATE@EXAMPLE.INVALID "
    engine.translate("二", "ja", "zh")
    assert engine.last_request_metrics["thread_reused"] == 1
    engine.close()


@pytest.mark.parametrize("failure", ["timeout", "rpc", "invalid_json", "extra_fields", "cancel"])
def test_failed_reused_request_discards_all_channel_histories_without_retry(server, failure):
    processes, _ = server
    engine = CodexTranslationEngine()
    engine.translate("一", "ja", "zh", conversation_key="mic")
    engine.translate("二", "ja", "zh", conversation_key="system")
    process = processes[0]
    options = {}
    if failure == "timeout":
        process.complete_turn = False
        engine.REQUEST_TIMEOUT = 0.02
    elif failure == "rpc":
        process.rpc_error_method = "turn/start"
    elif failure == "invalid_json":
        process.raw_override = "private invalid output"
    elif failure == "extra_fields":
        process.raw_override = '{"translation":"result","extra":"private"}'
    else:
        event = threading.Event()
        event.set()
        options["cancellation_event"] = event
    with pytest.raises((RuntimeError, TimeoutError)):
        engine.translate("三", "ja", "zh", conversation_key="mic", **options)
    assert len(methods(process, "turn/start")) == (2 if failure == "cancel" else 3)
    assert process.returncode == 0 and not engine._conversations
    assert engine._session is None
    engine.translate("四", "ja", "zh", conversation_key="system")
    assert len(processes) == 2 and engine.last_request_metrics["thread_turn"] == 1
    engine.close()


def test_old_turn_events_cannot_block_tools_or_override_new_translation_on_same_thread(server):
    processes, _ = server
    engine = CodexTranslationEngine()
    engine.translate("一", "ja", "zh")
    process = processes[0]
    thread = methods(process, "turn/start")[0]["threadId"]
    process.emit({"method": "item/started", "params": {
        "threadId": thread, "turnId": "turn-1", "item": {"type": "commandExecution"}}})
    process.emit({"method": "item/completed", "params": {
        "threadId": thread, "turnId": "turn-1", "item": {
            "id": "agent-1", "type": "agentMessage", "text": '{"translation":"STALE"}'}}})
    process.emit({"method": "turn/completed", "params": {
        "threadId": thread, "turn": {"id": "turn-1", "status": "failed"}}})
    assert engine.translate("二", "ja", "zh") == "明天开会。"
    assert engine.last_request_metrics["thread_reused"] == 1
    assert process.returncode is None
    engine.close()


@pytest.mark.parametrize("method", ["item/started", "error"])
def test_current_turn_tools_and_error_events_stop_owned_session_without_retry(server, method):
    processes, _ = server
    engine = CodexTranslationEngine()
    engine.translate("一", "ja", "zh")
    process = processes[0]
    process.complete_turn = False
    process.started.clear()
    failures = []

    def work():
        try:
            engine.translate("二", "ja", "zh")
        except RuntimeError as error:
            failures.append(str(error))

    worker = threading.Thread(target=work)
    worker.start()
    assert process.started.wait(2)
    params = {"threadId": methods(process, "turn/start")[-1]["threadId"], "turnId": "turn-2"}
    if method == "item/started":
        params["item"] = {"type": "commandExecution"}
    else:
        params["error"] = {"message": "private internal text"}
    process.emit({"method": method, "params": params})
    worker.join(2)
    assert not worker.is_alive() and failures
    assert "private internal text" not in failures[0]
    assert process.returncode == 0 and not engine._conversations
    assert len(methods(process, "turn/start")) == 2


def test_thread_closed_between_turns_forces_next_request_to_create_new_thread(server):
    processes, _ = server
    engine = CodexTranslationEngine()
    engine.translate("一", "ja", "zh")
    process = processes[0]
    thread = methods(process, "turn/start")[0]["threadId"]
    process.emit({"method": "thread/closed", "params": {"threadId": thread}})
    wait_for(lambda: engine._conversations["default"].invalidated)
    engine.translate("二", "ja", "zh")
    assert methods(process, "turn/start")[-1]["threadId"] != thread
    assert len(methods(process, "turn/start")) == 2
    engine.close()


def test_thread_closed_while_turn_active_stops_unknown_request_without_retry(server):
    processes, _ = server
    engine = CodexTranslationEngine()
    engine.translate("一", "ja", "zh")
    process = processes[0]
    process.complete_turn = False
    process.started.clear()
    failures = []

    def work():
        try:
            engine.translate("二", "ja", "zh")
        except RuntimeError as error:
            failures.append(str(error))

    worker = threading.Thread(target=work)
    worker.start()
    assert process.started.wait(2)
    process.emit({"method": "thread/closed", "params": {
        "threadId": methods(process, "turn/start")[-1]["threadId"]}})
    worker.join(2)
    assert not worker.is_alive() and failures and "未自动重发" in failures[0]
    assert len(methods(process, "turn/start")) == 2 and process.returncode == 0
    assert not engine._conversations


def test_helper_exit_invalidates_cached_histories_and_later_request_restarts(server):
    processes, _ = server
    engine = CodexTranslationEngine()
    engine.translate("一", "ja", "zh")
    processes[0].terminate()
    wait_for(lambda: not engine._conversations)
    assert engine.backend_info["auth_type"] == "not_verified"
    engine.translate("二", "ja", "zh")
    assert len(processes) == 2 and engine.last_request_metrics["thread_turn"] == 1
    engine.close()


def test_passthrough_does_not_reuse_or_mutate_existing_conversations(server):
    processes, _ = server
    engine = CodexTranslationEngine()
    engine.translate("一", "ja", "zh")
    process = processes[0]
    before = len(process.requests), engine.last_request_metrics
    assert engine.translate("已经是中文", "zh", "zh") == "已经是中文"
    assert engine.translate_batch(batch_items(), "ja", "ja") == {
        row["id"]: row["text"] for row in batch_items()}
    assert (len(process.requests), engine.last_request_metrics) == before
    engine.translate("二", "ja", "zh")
    assert engine.last_request_metrics["thread_reused"] == 1
    engine.close()


@pytest.mark.parametrize("key,generation", [("", 0), (None, 0), ("key", -1), ("key", True), ("key", "0")])
def test_invalid_conversation_options_do_not_start_owned_helper(server, key, generation):
    processes, _ = server
    engine = CodexTranslationEngine()
    with pytest.raises(ValueError):
        engine.translate("文", "ja", "zh", conversation_key=key, context_generation=generation)
    with pytest.raises(ValueError):
        engine.translate_batch(batch_items(), "ja", "zh", conversation_key=key, context_generation=generation)
    assert not processes
