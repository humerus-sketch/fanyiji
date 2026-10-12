"""Backend contracts and owned process lifetime; no inference-quality claims."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from unittest.mock import patch

import pytest

from fanyiji.engines import ModelAssetsError, TranslationEngine, create_translation_engine
from fanyiji.gguf_engine import GgufTranslationEngine, _WindowsJob
from fanyiji.livestream_prompt import PROFILE_ID, hy_translation_prompt, livestream_rules


def make_engine(tmp_path, **config):
    directory = tmp_path / "models" / "translation"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "test.gguf").write_bytes(b"not-a-real-model")
    return GgufTranslationEngine(directory, config={"model_file": "test.gguf", **config})


def test_legacy_factory_and_explicit_gguf_selection(tmp_path):
    for name in TranslationEngine.REQUIRED_FILES:
        (tmp_path / name).write_bytes(b"placeholder")
    assert isinstance(create_translation_engine(tmp_path), TranslationEngine)
    (tmp_path / "backend.json").write_text('{"backend":"ctranslate2"}', encoding="utf-8")
    assert isinstance(create_translation_engine(tmp_path), TranslationEngine)
    (tmp_path / "test.gguf").write_bytes(b"placeholder")
    (tmp_path / "backend.json").write_text(json.dumps({
        "backend": "llama_cpp", "model_file": "test.gguf", "model_kind": "hy",
        "architecture": "hy_v3", "gpu_layers": None,
    }), encoding="utf-8")
    engine = create_translation_engine(tmp_path)
    assert isinstance(engine, GgufTranslationEngine)
    assert engine.backend_info["device"] == "not_loaded"
    engine.close()


@pytest.mark.parametrize("field,value", [
    ("model_file", "../escape.gguf"), ("model_file", "C:/outside/model.gguf"),
    ("gpu_layers", -1), ("gpu_layers", True), ("runtime_preference", "remote"), ("model_kind", "unvalidated"),
])
def test_backend_rejects_invalid_paths_and_runtime_settings(tmp_path, field, value):
    with pytest.raises(ModelAssetsError):
        make_engine(tmp_path, **{field: value})


def test_runtime_dir_cannot_escape_models_runtime_and_override_is_explicit(tmp_path, monkeypatch):
    engine = make_engine(tmp_path, runtime_dir="../escape")
    monkeypatch.delenv("FANYIJI_RUNTIME", raising=False)
    monkeypatch.delenv("LOCAL_SUBTITLES_RUNTIME", raising=False)
    with pytest.raises(ModelAssetsError):
        engine._runtime_roots()
    runtime = tmp_path / "external-runtime"
    monkeypatch.setenv("FANYIJI_RUNTIME", str(runtime))
    assert engine._runtime_roots() == [runtime.resolve()]
    engine.close()


def test_renamed_runtime_override_wins_and_legacy_alias_remains_compatible(tmp_path, monkeypatch):
    engine = make_engine(tmp_path)
    legacy = tmp_path / "legacy-runtime"
    (legacy / "cpu").mkdir(parents=True)
    (legacy / "cpu" / "llama-server.exe").write_bytes(b"placeholder")
    monkeypatch.delenv("FANYIJI_RUNTIME", raising=False)
    monkeypatch.setenv("LOCAL_SUBTITLES_RUNTIME", str(legacy))
    assert engine._candidates() == [("cpu", legacy.resolve() / "cpu" / "llama-server.exe")]
    missing = tmp_path / "missing-new-runtime"
    monkeypatch.setenv("FANYIJI_RUNTIME", str(missing))
    assert engine._runtime_roots() == [missing.resolve()]
    with pytest.raises(ModelAssetsError, match="runtime"):
        engine._candidates()
    engine.close()


@pytest.mark.parametrize("model_kind", ["hy", "qwen"])
def test_model_uses_verified_user_prompt_and_fixed_parameters_without_chat_history(tmp_path, model_kind):
    engine = make_engine(tmp_path, model_kind=model_kind)
    requests = []

    def request(method, path, body):
        if path == "/apply-template":
            return {"prompt": body["messages"][0]["content"]}
        if path == "/tokenize":
            return {"tokens": list(range(len(body["content"])))}
        assert path == "/v1/chat/completions"
        requests.append(body)
        return {"choices": [{"finish_reason": "stop", "message": {"content": "契约译文"}}]}

    with patch.object(engine, "_ensure_loaded"), patch.object(engine, "_request", side_effect=request):
        assert engine.translate("コメントが流れた。", "ja", "zh") == "契约译文"
        assert engine.translate("明天开会。", "zh", "ja") == "契约译文"
    assert [body["max_tokens"] for body in requests] == [512, 512]
    assert engine.backend_info["model_kind"] == model_kind
    assert engine.backend_info["translation_profile"] == PROFILE_ID
    for body in requests:
        assert len(body["messages"]) == 1 and body["messages"][0]["role"] == "user"
        assert body["temperature"] == 0 and body["seed"] == 42 and body["repeat_penalty"] == 1.05
        assert body["top_p"] == 1 and body["top_k"] == 0
        assert body["reasoning_effort"] == "none" and body["chat_template_kwargs"] == {"enable_thinking": False}
    if model_kind == "hy":
        assert requests[0]["messages"][0]["content"] == (
            "将以下文本翻译为简体中文，注意只需要输出翻译后的结果，不要额外解释：\n"
            + livestream_rules("ja", "zh", ("コメントが流れた。",)) + "\n\nコメントが流れた。")
        assert requests[1]["messages"][0]["content"] == (
            "将以下文本翻译为日语，注意只需要输出翻译后的结果，不要额外解释：\n\n明天开会。")
    else:
        assert requests[0]["messages"][0]["content"] == (
            "将以下日语文本翻译为简体中文。只输出译文，保留人名、数字与否定关系，不要补充原文没有的内容。"
            "\n" + livestream_rules("ja", "zh", ("コメントが流れた。",)) + "\n\nコメントが流れた。\n/no_think")
        assert requests[1]["messages"][0]["content"] == (
            "将以下中文文本翻译为日语。只输出译文，保留人名、数字与否定关系，不要补充原文没有的内容。"
            "\n\n明天开会。\n/no_think")
    engine.close()


@pytest.mark.parametrize("model_kind", ["hy", "qwen"])
@pytest.mark.parametrize("target,target_name", [("zh", "简体中文"), ("ja", "日语")])
@pytest.mark.parametrize("context", ["", "同一通道前文，仅用于理解。"])
def test_mixed_source_prompt_keeps_fixed_target_for_each_template_and_context(
    tmp_path, model_kind, target, target_name, context
):
    engine = make_engine(tmp_path, model_kind=model_kind)
    requests = []
    text = "ここで stop，别开3号门，已经是中文也要保留原意。"

    def request(method, path, body):
        if path == "/tokenize":
            return {"tokens": list(range(100))}
        if path == "/apply-template":
            return {"prompt": body["messages"][0]["content"]}
        assert path == "/v1/chat/completions"
        requests.append(body)
        return {"choices": [{"finish_reason": "stop", "message": {"content": "契约译文"}}]}

    with patch.object(engine, "_ensure_loaded") as load, patch.object(engine, "_request", side_effect=request):
        assert engine.translate(text, "mixed", target, context=context) == "契约译文"
    load.assert_called_once()
    assert len(requests) == 1
    prompt = requests[0]["messages"][0]["content"]
    assert "日语、中文和英语" in prompt
    assert f"所有当前原文都统一译为{target_name}" in prompt
    assert "不得随原文语言改变翻译目标" in prompt
    assert "已经是目标语言的部分保留原意" in prompt
    assert "不得将中文误当日语解释" in prompt
    assert "保持数字、否定、语气和不确定性" in prompt
    assert text in prompt
    profile_rules = livestream_rules("mixed", target, (text,))
    if target == "zh":
        assert profile_rules and profile_rules in prompt
    else:
        assert not profile_rules
    assert prompt.endswith("/no_think") is (model_kind == "qwen")
    if context:
        assert "背景信息" in prompt and "不要重复翻译背景信息" in prompt
        assert prompt.index(context) < prompt.index("〖待翻译文本〗\n" + text)
    assert engine.supports_mixed_sources is True
    assert engine.backend_info["supports_mixed_sources"] is True
    engine.close()


@pytest.mark.parametrize("model_kind", ["hy", "qwen"])
def test_english_source_is_accepted_without_mislabelling_as_japanese(tmp_path, model_kind):
    engine = make_engine(tmp_path, model_kind=model_kind)
    response = {"choices": [{"finish_reason": "stop", "message": {"content": "不要打开3号门"}}]}
    with patch.object(engine, "_ensure_loaded"), patch.object(engine, "_request", return_value=response) as request:
        engine.translate("Do not open door number 3.", "en", "zh")
    prompt = request.call_args.args[2]["messages"][0]["content"]
    assert "简体中文" in prompt
    assert "以下日语文本" not in prompt
    if model_kind == "qwen":
        assert "以下英语文本" in prompt
    assert sum(call.args[1] == "/v1/chat/completions" for call in request.call_args_list) == 1
    assert not livestream_rules("en", "zh", ("Do not open door number 3.",))
    engine.close()


@pytest.mark.parametrize("model_kind", ["hy", "qwen"])
@pytest.mark.parametrize("source", ["ja", "mixed"])
@pytest.mark.parametrize("context", ["", "以前の文は雑談だけです。"])
def test_livestream_hints_use_only_current_source_and_preserve_user_template(
    tmp_path, model_kind, source, context
):
    engine = make_engine(tmp_path, model_kind=model_kind)
    text = "スパチャありがとう、切り抜きはまだです。"
    prompt = engine._translation_prompt(text, source, "zh", context)
    rules = livestream_rules(source, "zh", (text,))
    assert rules and rules in prompt
    if model_kind == "hy" or source == "mixed" or context:
        assert prompt == hy_translation_prompt(text, source, "zh", context) + (
            "\n/no_think" if model_kind == "qwen" else ""
        )
    assert prompt.endswith(text + ("\n/no_think" if model_kind == "qwen" else ""))
    if context:
        assert prompt.index(context) < prompt.index("〖待翻译文本〗\n" + text)
        assert "不要重复翻译背景信息" in prompt
        assert "不得从背景添加、替换或推断" in prompt
    # A previous reference alone must not select terms for the current source.
    plain = engine._translation_prompt("そうですね。", source, "zh", text)
    plain_rules = livestream_rules(source, "zh", ("そうですね。",))
    assert plain_rules in plain
    assert rules != plain_rules
    engine.close()


@pytest.mark.parametrize("model_kind", ["hy", "qwen"])
def test_livestream_profile_does_not_change_reverse_translation_with_reference(tmp_path, model_kind):
    engine = make_engine(tmp_path, model_kind=model_kind)
    prompt = engine._translation_prompt("谢谢留言。", "zh", "ja", "前文只作为背景。")
    assert not livestream_rules("zh", "ja", ("谢谢留言。",))
    assert "〖背景信息" in prompt and "翻译为日语" in prompt
    assert prompt.endswith("谢谢留言。" + ("\n/no_think" if model_kind == "qwen" else ""))
    engine.close()


@pytest.mark.parametrize("source,target", [("auto", "zh"), ("mixed", "en"), ("en", "mixed")])
def test_invalid_source_or_target_does_not_load_local_runtime(tmp_path, source, target):
    engine = make_engine(tmp_path)
    with patch.object(engine, "_ensure_loaded") as load:
        with pytest.raises(ValueError):
            engine.translate("text", source, target)
    load.assert_not_called()
    engine.close()


def test_exhausted_output_limit_is_reported_instead_of_accepting_partial_translation(tmp_path):
    engine = make_engine(tmp_path)
    response = {"choices": [{"finish_reason": "length", "message": {"content": "partial"}}]}
    with patch.object(engine, "_ensure_loaded"), patch.object(engine, "_request", return_value=response), patch.object(
        engine, "_stop_runtime"
    ) as stop:
        with pytest.raises(RuntimeError, match="输出上限"):
            engine.translate("text", "ja", "zh")
        stop.assert_called_once()
    engine.close()


class FakeProcess:
    pid = 123

    def __init__(self, returncode=None):
        self.returncode = returncode

    def poll(self):
        return self.returncode

    def kill(self):
        self.returncode = -9

    def wait(self, timeout):
        if self.returncode is None:
            raise subprocess.TimeoutExpired("test-owned-helper", timeout)
        return self.returncode


class FakeJob:
    def __init__(self):
        self.process = None

    def assign(self, process):
        self.process = process

    def close(self):
        if self.process is not None:
            self.process.returncode = -1


def runtime_tree(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime"
    for device in ("cpu", "cuda"):
        (runtime / device).mkdir(parents=True)
        (runtime / device / "llama-server.exe").write_bytes(b"not-an-executable")
    monkeypatch.setenv("FANYIJI_RUNTIME", str(runtime))
    return runtime


def test_auto_cuda_start_failure_falls_back_to_cpu_without_download_options(tmp_path, monkeypatch):
    runtime_tree(tmp_path, monkeypatch)
    engine = make_engine(tmp_path, gpu_layers=None)
    processes = [FakeProcess(1), FakeProcess()]

    def response(method, path, **kwargs):
        return {"status": "ok"} if path == "/health" else {"data": [{"id": engine._alias}]}

    with patch("fanyiji.gguf_engine._WindowsJob", FakeJob), patch(
        "fanyiji.gguf_engine.subprocess.Popen", side_effect=processes
    ) as popen, patch.object(engine, "_request", side_effect=response):
        engine.preload()
        engine.preload()
        assert popen.call_count == 2
        assert engine.backend_info["device"] == "cpu"
        cuda, cpu = [call.args[0] for call in popen.call_args_list]
        assert "--fit" in cuda and cuda[cuda.index("--fit-target") + 1] == "1024"
        assert "--gpu-layers" not in cuda
        assert cpu[cpu.index("--gpu-layers") + 1] == "0"
        for call in popen.call_args_list:
            command, environment = call.args[0], call.kwargs["env"]
            assert "--offline" in command and "--no-webui" in command
            assert command[command.index("--ctx-size") + 1] == "2048"
            assert not any(arg in command for arg in ("-hf", "--hf-repo", "--model-url", "--rpc"))
            assert "--api-key" not in command
            assert "LLAMA_API_KEY" in environment and "HTTP_PROXY" not in environment
            assert "poppler" not in environment["PATH"].lower()
        engine.close()
        assert processes[1].poll() is not None


def test_close_during_preload_prevents_restart_and_does_not_wait_for_startup_timeout(tmp_path, monkeypatch):
    runtime_tree(tmp_path, monkeypatch)
    engine = make_engine(tmp_path, runtime_preference="cpu")
    requested = threading.Event()
    process = FakeProcess()

    def loading(*args, **kwargs):
        requested.set()
        raise RuntimeError("HTTP 503 loading")

    with patch("fanyiji.gguf_engine._WindowsJob", FakeJob), patch(
        "fanyiji.gguf_engine.subprocess.Popen", return_value=process
    ) as popen, patch.object(engine, "_request", side_effect=loading), ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(engine.preload)
        assert requested.wait(2)
        started = time.monotonic()
        engine.close()
        assert time.monotonic() - started < 1
        with pytest.raises(RuntimeError, match="已关闭"):
            future.result(timeout=2)
        with pytest.raises(RuntimeError, match="已关闭"):
            engine.preload()
        assert popen.call_count == 1 and process.poll() is not None


def test_close_during_translation_does_not_wait_for_translation_lock(tmp_path):
    engine = make_engine(tmp_path)
    entered, released = threading.Event(), threading.Event()

    class Connection:
        def close(self):
            released.set()

    def blocked_request(*args, **kwargs):
        engine._connection = Connection()
        entered.set()
        assert released.wait(2)
        raise OSError("owned helper closed")

    with patch.object(engine, "_ensure_loaded"), patch.object(engine, "_request", side_effect=blocked_request), ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(engine.translate, "text", "ja", "zh")
        assert entered.wait(2)
        engine.close()
        with pytest.raises(OSError):
            future.result(timeout=2)
        assert released.is_set()


def test_cancel_pending_interrupts_inference_and_engine_can_translate_again(tmp_path):
    engine = make_engine(tmp_path)
    entered, released = threading.Event(), threading.Event()

    class Connection:
        def close(self):
            released.set()

    def blocked_request(*args, **kwargs):
        engine._connection = Connection()
        entered.set()
        assert released.wait(2)
        raise OSError("cancelled owned helper")

    with patch.object(engine, "_ensure_loaded"), patch.object(engine, "_request", side_effect=blocked_request), ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(engine.translate, "Do not open it.", "en", "zh")
        assert entered.wait(2)
        engine.cancel_pending()
        with pytest.raises(OSError):
            future.result(timeout=2)
    assert not engine._closed
    response = {"choices": [{"finish_reason": "stop", "message": {"content": "別の文です。"}}]}
    with patch.object(engine, "_ensure_loaded"), patch.object(engine, "_request", return_value=response):
        assert engine.translate("另一个句子。", "zh", "ja") == "別の文です。"
    engine.close()


def test_cancel_during_startup_does_not_retry_cpu_and_next_preload_can_resume(tmp_path, monkeypatch):
    runtime_tree(tmp_path, monkeypatch)
    engine = make_engine(tmp_path)
    entered = threading.Event()
    first, second = FakeProcess(), FakeProcess()

    def loading(*args, **kwargs):
        entered.set()
        raise RuntimeError("HTTP 503 loading")

    with patch("fanyiji.gguf_engine._WindowsJob", FakeJob), patch(
        "fanyiji.gguf_engine.subprocess.Popen", side_effect=[first, second]
    ) as popen:
        with patch.object(engine, "_request", side_effect=loading), ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(engine.preload)
            assert entered.wait(2)
            engine.cancel_pending()
            with pytest.raises(RuntimeError, match="已取消"):
                future.result(timeout=2)
        assert popen.call_count == 1 and first.poll() is not None
        assert not engine._closed
        with patch.object(engine, "_request", side_effect=[
            {"status": "ok"}, {"data": [{"id": engine._alias}]},
        ]):
            engine.preload()
        assert popen.call_count == 2 and engine.backend_info["device"] == "cuda"
        engine.close()
        assert second.poll() is not None


def test_run_cancelled_while_waiting_for_shared_local_engine_does_not_reload(tmp_path):
    engine = make_engine(tmp_path)
    cancelled = threading.Event()
    engine._translation_lock.acquire()
    try:
        with patch.object(engine, "_ensure_loaded") as load, ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(engine.translate, "Do not open it.", "en", "zh", cancellation_event=cancelled)
            cancelled.set()
            engine._translation_lock.release()
            with pytest.raises(RuntimeError, match="已取消"):
                future.result(timeout=2)
            load.assert_not_called()
        assert engine._active_cancellation_event is None and not engine._closed
        assert engine.backend_info["supports_cancellation"] is True
    finally:
        if engine._translation_lock.locked():
            engine._translation_lock.release()
        engine.close()


def test_run_cancelled_before_startup_cannot_launch_native_helper(tmp_path, monkeypatch):
    runtime_tree(tmp_path, monkeypatch)
    engine = make_engine(tmp_path)
    cancelled = threading.Event()
    cancelled.set()
    with patch("fanyiji.gguf_engine.subprocess.Popen") as popen:
        with pytest.raises(RuntimeError, match="已取消"):
            engine.translate("Do not open it.", "en", "zh", cancellation_event=cancelled)
        popen.assert_not_called()
    engine.close()


def test_cancelled_preload_does_not_create_helper_and_keeps_engine_reusable(tmp_path):
    engine = make_engine(tmp_path)
    cancelled = threading.Event()
    cancelled.set()
    with patch.object(engine, "_ensure_loaded") as load:
        with pytest.raises(RuntimeError, match="已取消"):
            engine.preload(cancellation_event=cancelled)
        load.assert_not_called()
        engine.preload()
        load.assert_called_once_with()
    assert engine._active_cancellation_event is None and not engine._closed
    engine.close()


def test_unexpected_connection_close_error_does_not_leave_owned_helper_alive(tmp_path):
    engine = make_engine(tmp_path)
    process, job = FakeProcess(), FakeJob()
    job.assign(process)

    class Connection:
        def close(self):
            raise RuntimeError("unexpected close failure")

    engine._process, engine._job, engine._connection = process, job, Connection()
    engine.close()
    assert process.poll() is not None


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object lifecycle")
def test_real_windows_job_close_terminates_only_its_owned_idle_python_child():
    # This is a real process-lifetime check, not an audio or model inference test.
    process = subprocess.Popen([sys.executable, "-B", "-c", "import time; time.sleep(60)"],
                               stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               creationflags=subprocess.CREATE_NO_WINDOW)
    job = _WindowsJob()
    try:
        job.assign(process)
        assert process.poll() is None
        job.close()
        assert process.wait(timeout=5) is not None
        job.close()
    finally:
        job.close()
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
