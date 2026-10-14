"""Cross-layer conversation isolation using mocks, never real model requests."""
from dataclasses import replace
from contextlib import nullcontext
import json
from pathlib import Path
import threading
import time

import numpy as np
import pytest

from fanyiji.config import AppConfig
from fanyiji import codex_engine
from fanyiji.codex_engine import CodexTranslationEngine
from fanyiji.pipeline import RecognitionJob, SubtitlePipeline, TranslationJob, _Run
from fanyiji.translation_queue import TranslationQueue
from fanyiji.translation_output import is_chinese_output
from test_codex_engine import FakeJob, FakeProcess, methods


class RecordingTranslator:
    supports_mixed_sources = True
    backend_info = {"supports_batch": True, "supports_source_context": True}
    last_request_metrics = {}

    def __init__(self):
        self.calls = []

    def reset_conversation(self, key):
        self.calls.append(("reset", key))

    def translate(self, text, source, target, **kwargs):
        self.calls.append(("single", text, source, target, kwargs))
        return "测试译文"

    def translate_batch(self, items, source, target, **kwargs):
        self.calls.append(("batch", items, source, target, kwargs))
        return {item["id"]: "测试译文" for item in items}


def environment(tmp_path):
    events = []
    pipeline = SubtitlePipeline(tmp_path, events.append)
    run = _Run(session=42, config=AppConfig(translation_provider="codex", speech_only=False))
    pipeline._run = run
    return pipeline, run, events


def translation_job(segment, *, channel="system", generation=0, **changes):
    job = TranslationJob(channel, segment, "配信は続きます。", time.monotonic(), ("ja", "zh"), 1.0,
                         context_generation=generation)
    return replace(job, **changes)


def test_request_uses_job_generation_snapshot_after_later_asr_boundary(tmp_path):
    pipeline, run, _ = environment(tmp_path)
    recognition = RecognitionJob("system", "first", np.zeros(16000, dtype=np.float32),
                                 time.monotonic(), ("ja", "zh"))
    pipeline._enqueue_translation(run, recognition, "配信は続きます。")
    queued = run.mt_queue.get(channel="system")
    assert queued.context_generation == 0
    pipeline._clear_source_context(run, "system")
    assert run.context_generations["system"] == 1
    translator = RecordingTranslator()
    pipeline._translation_request(run, translator, [queued])
    assert translator.calls[0][-1]["conversation_key"] == "42:system"
    assert translator.calls[0][-1]["context_generation"] == 0
    run.mt_queue.close(discard=True)


def test_uncertain_asr_changes_generation_before_enqueuing_translation(tmp_path):
    pipeline, run, _ = environment(tmp_path)
    job = RecognitionJob("system", "uncertain", np.zeros(16000, dtype=np.float32),
                          time.monotonic(), ("ja", "zh"), recognition_uncertain=True)
    pipeline._enqueue_translation(run, job, "配信は続きます。")
    queued = run.mt_queue.get(channel="system")
    assert queued.context_generation > 0
    assert queued.context_generation == run.context_generations["system"]
    assert run.context_generations["mic"] == 0
    run.mt_queue.close(discard=True)


@pytest.mark.parametrize("changes", [{"provisional": True}, {"revision": 2}])
def test_provisional_and_revision_are_reset_before_and_after_request(tmp_path, changes):
    pipeline, run, events = environment(tmp_path)
    translator = RecordingTranslator()
    job = translation_job("fragment", generation=3, **changes)
    pipeline._translation_request(run, translator, [job])
    assert [call[0] for call in translator.calls] == ["reset", "single", "reset"]
    assert translator.calls[0] == translator.calls[-1] == ("reset", "42:system")
    assert translator.calls[1][-1]["context_generation"] == 3
    assert any(event["kind"] == "translation" and not event["failed"] for event in events)
    run.mt_queue.close(discard=True)


def test_failed_provisional_request_still_resets_conversation_after_failure(tmp_path):
    pipeline, run, events = environment(tmp_path)
    translator = RecordingTranslator()

    def fail(text, source, target, **kwargs):
        translator.calls.append(("single", text, source, target, kwargs))
        raise RuntimeError("controlled mock failure")

    translator.translate = fail
    pipeline._translation_request(run, translator, [translation_job("fragment", provisional=True)])
    assert [call[0] for call in translator.calls] == ["reset", "single", "reset"]
    assert any(event["kind"] == "translation" and event["failed"] for event in events)
    run.mt_queue.close(discard=True)


def test_spilled_queue_keeps_generation_boundaries_and_isolates_revision(tmp_path):
    pipeline, run, _ = environment(tmp_path)
    run.mt_queue = TranslationQueue(memory_limit=1, job_factory=TranslationJob)
    jobs = [translation_job("first", generation=0), translation_job("second", generation=0),
            translation_job("third", generation=1), translation_job("revised", generation=1, revision=2),
            translation_job("last", generation=1)]
    for job in jobs:
        run.mt_queue.put("system", job, 1.0)
    assert run.mt_queue.spool_path is not None
    run.mt_queue.close()
    translator = RecordingTranslator()
    pipeline._translate(run, translator, channel="system")
    requests = [call for call in translator.calls if call[0] != "reset"]
    assert [call[0] for call in requests] == ["batch", "single", "single", "single"]
    assert [item["id"] for item in requests[0][1]] == ["first", "second"]
    assert [call[-1]["context_generation"] for call in requests] == [0, 1, 1, 1]
    assert sum(call[0] == "reset" for call in translator.calls) == 2
    assert run.mt_queue.spool_path is None


def test_fixed_channel_workers_allow_other_channel_without_reordering_same_channel(tmp_path):
    pipeline, run, _ = environment(tmp_path)
    entered, release, microphone_done = threading.Event(), threading.Event(), threading.Event()
    system = RecordingTranslator()
    microphone = RecordingTranslator()
    original_system = system.translate
    original_microphone = microphone.translate

    def slow_system(text, source, target, **kwargs):
        entered.set()
        assert release.wait(3), "test must release the first system request"
        return original_system(text, source, target, **kwargs)

    def fast_microphone(text, source, target, **kwargs):
        result = original_microphone(text, source, target, **kwargs)
        microphone_done.set()
        return result

    system.translate = slow_system
    microphone.translate = fast_microphone
    first = translation_job("first", text="第一句话。")
    run.mt_queue.put("system", first, 1.0)
    system_worker = threading.Thread(target=pipeline._translate, args=(run, system, "system"))
    microphone_worker = threading.Thread(target=pipeline._translate, args=(run, microphone, "mic"))
    system_worker.start()
    try:
        assert entered.wait(2)
        run.mt_queue.put("system", translation_job("second", text="第二句话。"), 1.0)
        run.mt_queue.put("mic", translation_job("mic", channel="mic", text="麦克风的话。"), 1.0)
        microphone_worker.start()
        assert microphone_done.wait(2)
        assert system.calls == []  # The blocked owner still holds the first request.
        assert microphone.calls[0][1] == "麦克风的话。"
        assert microphone.calls[0][-1]["conversation_key"] == "42:mic"
    finally:
        release.set()
        run.mt_queue.close()
        system_worker.join(timeout=3)
        if microphone_worker.ident is not None:
            microphone_worker.join(timeout=3)
    assert not system_worker.is_alive() and not microphone_worker.is_alive()
    assert [call[1] for call in system.calls] == ["第一句话。", "第二句话。"]
    assert all(call[-1]["conversation_key"] == "42:system" for call in system.calls)


def test_single_fallback_worker_keeps_independent_channel_keys(tmp_path):
    pipeline, run, _ = environment(tmp_path)
    translator = RecordingTranslator()
    run.mt_queue.put("system", translation_job("system"), 1.0)
    run.mt_queue.put("mic", translation_job("mic", channel="mic"), 1.0)
    run.mt_queue.close()
    pipeline._translate(run, translator, channel=None)
    assert {call[-1]["conversation_key"] for call in translator.calls} == {"42:system", "42:mic"}
    assert len(translator.calls) == 2


class ConversationProcess(FakeProcess):
    """Real reader/RPC flow, with structured results chosen from the turn input."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inject_previous_turn_events = False
        self.close_active_thread = False
        self.previous_turn = None

    def receive(self, request):
        if request.get("method") != "turn/start":
            return super().receive(request)
        self.requests.append(request)
        params = request["params"]
        thread = params["threadId"]
        self.turn_number += 1
        turn = f"turn-{self.turn_number}"
        self.started.set()
        if self.inject_previous_turn_events and self.previous_turn is not None:
            previous_thread, previous_turn = self.previous_turn
            stale_item = {"id": "stale", "type": "agentMessage", "text": '{"translation":"错误的旧译文"}'}
            self.emit({"method": "item/started", "params": {
                "threadId": previous_thread, "turnId": previous_turn,
                "item": {"id": "stale-tool", "type": "commandExecution"}}})
            self.emit({"method": "item/completed", "params": {
                "threadId": previous_thread, "turnId": previous_turn, "item": stale_item}})
            self.emit({"method": "turn/completed", "params": {
                "threadId": previous_thread, "turn": {"id": previous_turn, "status": "completed",
                                                        "items": [stale_item]}}})
        if self.close_active_thread:
            self.emit({"method": "thread/closed", "params": {"threadId": thread}})
        elif self.complete_turn:
            data = json.loads(params["input"][0]["text"])
            # This fixture tests conversation routing, not translation quality.
            # Echo only Chinese source; foreign source gets a valid Chinese stub.
            output = lambda text: "译文：" + text if is_chinese_output(text) else "测试译文"
            payload = ({"translations": [{"id": item["id"], "translation": output(item["current_text"])}
                                         for item in data["items"]]}
                       if "items" in data else {"translation": output(data["current_text"])})
            raw = self.raw_override if self.raw_override is not None else json.dumps(payload, ensure_ascii=False)
            item = {"id": f"answer-{turn}", "type": "agentMessage", "text": raw, "phase": "final_answer"}
            self.emit({"method": "item/completed", "params": {
                "threadId": thread, "turnId": turn, "item": item}})
            self.emit({"method": "turn/completed", "params": {
                "threadId": thread, "turn": {"id": turn, "status": "completed", "items": [item]}}})
        self.previous_turn = (thread, turn)
        self.emit({"id": request["id"], "result": {"turn": {"id": turn, "status": "inProgress", "items": []}}})


@pytest.fixture
def conversation_server(monkeypatch):
    processes = []

    def spawn(command, **kwargs):
        process = ConversationProcess(command, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(codex_engine, "find_codex_cli", lambda: Path("C:/mock/codex.exe"))
    monkeypatch.setattr(codex_engine.subprocess, "Popen", spawn)
    monkeypatch.setattr(codex_engine, "_WindowsJob", FakeJob)
    monkeypatch.setattr(codex_engine, "_external_dll_search", nullcontext)
    yield processes
    for process in processes:
        process.terminate()


def test_engine_single_batch_single_reuses_one_thread_and_changes_schema(conversation_server):
    engine = CodexTranslationEngine()
    try:
        assert engine.translate("第一句。", "ja", "zh", conversation_key="system") == "译文：第一句。"
        items = [{"id": "a", "text": "第二句。"}, {"id": "b", "text": "第三句。"}]
        assert engine.translate_batch(items, "ja", "zh", conversation_key="system") == {
            "a": "译文：第二句。", "b": "译文：第三句。"}
        assert engine.translate("第四句。", "ja", "zh", conversation_key="system") == "译文：第四句。"
        process = conversation_server[0]
        turns = methods(process, "turn/start")
        assert len({turn["threadId"] for turn in turns}) == 1
        assert [set(turn["outputSchema"]["properties"]) for turn in turns] == [
            {"translation"}, {"translations"}, {"translation"}]
        assert len(methods(process, "thread/start")) == 2  # Validation plus one actual conversation.
        assert len(methods(process, "thread/unsubscribe")) == 1  # Validation only.
        assert engine.last_request_metrics["thread_reused"] == 1
    finally:
        engine.close()


def test_engine_dynamic_terms_are_turn_data_and_do_not_create_new_thread(conversation_server):
    engine = CodexTranslationEngine()
    try:
        engine.translate("ナイス！", "ja", "zh", conversation_key="system")
        engine.translate("スパチャありがとう。", "ja", "zh", conversation_key="system")
        process = conversation_server[0]
        turns = methods(process, "turn/start")
        assert turns[0]["threadId"] == turns[1]["threadId"]
        first, second = [json.loads(turn["input"][0]["text"]) for turn in turns]
        assert first["term_hints"] != second["term_hints"]
        assert "ナイス" in json.dumps(first["term_hints"], ensure_ascii=False)
        assert "付费留言" in json.dumps(second["term_hints"], ensure_ascii=False)
        instructions = methods(process, "thread/start")[-1]["baseInstructions"]
        assert "ナイス：" not in instructions and "スパチャ：" not in instructions
    finally:
        engine.close()


def test_engine_keys_generation_and_explicit_reset_keep_histories_separate(conversation_server):
    engine = CodexTranslationEngine()
    try:
        engine.translate("电脑第一句。", "ja", "zh", conversation_key="system", context_generation=0)
        engine.translate("麦克风第一句。", "ja", "zh", conversation_key="mic", context_generation=0)
        engine.translate("电脑第二句。", "ja", "zh", conversation_key="system", context_generation=0)
        engine.translate("电脑换代。", "ja", "zh", conversation_key="system", context_generation=1)
        engine.reset_conversation("system")
        engine.translate("电脑重置。", "ja", "zh", conversation_key="system", context_generation=1)
        engine.translate("麦克风第二句。", "ja", "zh", conversation_key="mic", context_generation=0)
        threads = [turn["threadId"] for turn in methods(conversation_server[0], "turn/start")]
        assert threads[0] == threads[2]
        assert threads[1] == threads[5]
        assert len({threads[0], threads[1], threads[3], threads[4]}) == 4
    finally:
        engine.close()


def test_old_turn_events_cannot_trigger_tool_error_or_replace_current_answer(conversation_server):
    engine = CodexTranslationEngine()
    try:
        engine.translate("上一句。", "ja", "zh")
        process = conversation_server[0]
        process.inject_previous_turn_events = True
        assert engine.translate("当前句。", "ja", "zh") == "译文：当前句。"
        assert len(conversation_server) == 1
        turns = methods(process, "turn/start")
        assert turns[0]["threadId"] == turns[1]["threadId"]
    finally:
        engine.close()


def test_server_closed_notification_rebuilds_before_next_turn(conversation_server):
    engine = CodexTranslationEngine()
    try:
        engine.translate("上一句。", "ja", "zh")
        process = conversation_server[0]
        first_thread = methods(process, "turn/start")[0]["threadId"]
        process.emit({"method": "thread/closed", "params": {"threadId": first_thread}})
        deadline = time.monotonic() + 2
        while first_thread not in engine._session.closed_threads and time.monotonic() < deadline:
            time.sleep(0.005)
        assert first_thread in engine._session.closed_threads
        assert engine.translate("下一句。", "ja", "zh") == "译文：下一句。"
        assert methods(process, "turn/start")[1]["threadId"] != first_thread
    finally:
        engine.close()


def test_active_server_close_fails_once_and_never_resends_current_text(conversation_server):
    engine = CodexTranslationEngine()
    try:
        engine.preload()
        process = conversation_server[0]
        process.close_active_thread = True
        with pytest.raises(RuntimeError, match="会话已关闭"):
            engine.translate("不能重发这一句。", "ja", "zh")
        assert len(methods(process, "turn/start")) == 1
        assert process.returncode is not None
        assert engine.translate("新的下一句。", "ja", "zh") == "译文：新的下一句。"
        assert len(conversation_server) == 2
        assert len(methods(conversation_server[1], "turn/start")) == 1
    finally:
        engine.close()


def test_pipeline_invalid_batch_fallback_does_not_keep_failed_thread_history(tmp_path, conversation_server):
    pipeline, run, events = environment(tmp_path)
    engine = CodexTranslationEngine()
    try:
        engine.preload()
        conversation_server[0].raw_override = '{"translation":"错误的单句格式"}'
        jobs = [translation_job("first"), translation_job("second")]
        pipeline._translation_request(run, engine, jobs)
        assert len(conversation_server) == 2
        assert len(methods(conversation_server[0], "turn/start")) == 1
        assert len(methods(conversation_server[1], "turn/start")) == 2
        translated = [event for event in events if event["kind"] == "translation"]
        assert len(translated) == 2 and all(not event["failed"] for event in translated)
        assert all(event["fallback"] for event in translated)
    finally:
        engine.close()
        run.mt_queue.close(discard=True)
