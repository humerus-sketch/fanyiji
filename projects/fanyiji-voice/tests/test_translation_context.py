"""Reference text is bounded and never becomes another translation target."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from fanyiji.engines import TranslationEngine
from fanyiji.gguf_engine import GgufTranslationEngine
from fanyiji.livestream_prompt import PROFILE_ID, livestream_rules


def gguf_engine(tmp_path, model_kind="hy"):
    (tmp_path / "test.gguf").write_bytes(b"test-only-placeholder")
    return GgufTranslationEngine(tmp_path, config={"model_file": "test.gguf", "model_kind": model_kind})


def completion_result():
    return {"choices": [{"finish_reason": "stop", "message": {"content": "当前译文"}}]}


@pytest.mark.parametrize("model_kind", ["hy", "qwen"])
def test_reference_prompt_separates_current_source_and_discloses_capability(tmp_path, model_kind):
    engine = gguf_engine(tmp_path, model_kind)
    calls = []

    def request(method, path, body):
        calls.append((path, body))
        if path == "/tokenize":
            return {"tokens": list(range(len(body["content"])))}
        if path == "/apply-template":
            return {"prompt": body["messages"][0]["content"]}
        return completion_result()

    with patch.object(engine, "_ensure_loaded"), patch.object(engine, "_request", side_effect=request):
        assert engine.translate("彼女は歌わない。", "ja", "zh", context="ミコさんは配信者です。") == "当前译文"
    prompt = calls[-1][1]["messages"][0]["content"]
    assert "〖背景信息：同一通道前文原文，仅用于理解，不翻译〗\nミコさんは配信者です。" in prompt
    assert "〖待翻译文本〗\n彼女は歌わない。" in prompt
    assert "只输出当前待翻译文本的译文" in prompt and "不得从背景添加、替换或推断" in prompt
    assert prompt.endswith("\n/no_think") is (model_kind == "qwen")
    assert engine.backend_info["supports_source_context"] is True
    assert engine.backend_info["translation_profile"] == PROFILE_ID
    assert livestream_rules("ja", "zh", ("彼女は歌わない。",)) in prompt
    assert calls[-1][1]["max_tokens"] == 512
    engine.close()


def test_oversize_reference_is_trimmed_but_current_source_is_kept_complete(tmp_path):
    engine = gguf_engine(tmp_path)
    prompts = []

    def request(method, path, body):
        if path == "/apply-template":
            return {"prompt": body["messages"][0]["content"]}
        if path == "/tokenize":
            return {"tokens": [1] * (len(body["content"]) * 3)}
        prompts.append(body["messages"][0]["content"])
        return completion_result()

    current = "300元を返して、200元は残してください。"
    with patch.object(engine, "_ensure_loaded"), patch.object(engine, "_request", side_effect=request):
        engine.translate(current, "ja", "zh", context="前" * 10000)
    reference = prompts[0].split("不翻译〗\n", 1)[1].split("\n\n", 1)[0]
    assert len(reference) * 3 <= 256
    assert prompts[0].endswith(current)
    engine.close()


def test_total_budget_drops_old_reference_without_truncating_current_source(tmp_path):
    engine = gguf_engine(tmp_path)
    prompts = []

    def request(method, path, body):
        if path == "/apply-template":
            return {"prompt": body["messages"][0]["content"]}
        if path == "/tokenize":
            content = body["content"]
            # Simulate a near-budget current utterance, with 4 tokens per
            # reference character and instruction overhead only in the prompt.
            count = 1000 + content.count("前") * 4 if "待翻译文本" in content else len(content)
            return {"tokens": [1] * count}
        prompts.append(body["messages"][0]["content"])
        return completion_result()

    with patch.object(engine, "_ensure_loaded"), patch.object(engine, "_request", side_effect=request):
        engine.translate("完整当前原文，数字是2077，不是277。", "ja", "zh", context="前" * 200)
    assert prompts[0].count("前") < 200
    assert prompts[0].endswith("完整当前原文，数字是2077，不是277。")
    engine.close()


def test_missing_local_tokenizer_uses_bounded_fallback_and_still_translates(tmp_path):
    engine = gguf_engine(tmp_path)
    prompts = []

    def request(method, path, body):
        if path in ("/tokenize", "/apply-template"):
            raise RuntimeError("本地 GGUF helper 返回 HTTP 404")
        prompts.append(body["messages"][0]["content"])
        return completion_result()

    with patch.object(engine, "_ensure_loaded"), patch.object(engine, "_request", side_effect=request):
        assert engine.translate("いいえ。", "ja", "zh", context="前" * 10000) == "当前译文"
    assert len(prompts[0].encode("utf-8")) + 128 <= 2048 - 512
    assert prompts[0].endswith("いいえ。")
    engine.close()


def test_overlong_current_source_is_rejected_instead_of_silently_truncated(tmp_path):
    engine = gguf_engine(tmp_path)
    completions = []

    def request(method, path, body):
        if path in ("/tokenize", "/apply-template"):
            raise RuntimeError("本地 GGUF helper 返回 HTTP 404")
        completions.append(body)
        return completion_result()

    with patch.object(engine, "_ensure_loaded"), patch.object(engine, "_request", side_effect=request):
        with pytest.raises(ValueError, match="当前待翻译文本"):
            engine.translate("長" * 1000, "ja", "zh", context="前文")
    assert completions == []
    engine.close()


def test_tokenizer_transport_failure_is_propagated_and_owned_helper_stopped(tmp_path):
    engine = gguf_engine(tmp_path)
    with patch.object(engine, "_ensure_loaded"), patch.object(engine, "_request", side_effect=TimeoutError), patch.object(
        engine, "_stop_runtime"
    ) as stop:
        with pytest.raises(TimeoutError):
            engine.translate("現在の文。", "ja", "zh", context="前文")
        stop.assert_called_once()
    engine.close()


def test_m2m100_reference_is_ignored_without_translating_previous_segments(tmp_path):
    for name in TranslationEngine.REQUIRED_FILES:
        (tmp_path / name).write_bytes(b"test-only-placeholder")
    engine = TranslationEngine(tmp_path)
    encoded = []

    class Tokenizer:
        def encode(self, text, language):
            encoded.append((text, language))
            return ["current-token"]

        def decode(self, tokens):
            return "当前译文"

    engine._tokenizer = Tokenizer()
    engine._translator = SimpleNamespace(translate_batch=lambda *args, **kwargs: [SimpleNamespace(hypotheses=[["译文"]])])
    assert engine.translate("現在の文。", "ja", "zh", context="前文は翻訳しない。") == "当前译文"
    assert encoded == [("現在の文。", "ja")]
    assert engine.backend_info["supports_source_context"] is False


@pytest.mark.parametrize("context", ["", " \n "])
def test_empty_reference_still_budgets_new_livestream_rules_and_complete_current_source(tmp_path, context):
    engine = gguf_engine(tmp_path)
    calls = []

    def request(method, path, body):
        calls.append((path, body))
        if path == "/apply-template":
            return {"prompt": body["messages"][0]["content"]}
        if path == "/tokenize":
            return {"tokens": [1] * len(body["content"])}
        return completion_result()

    with patch.object(engine, "_ensure_loaded"), patch.object(engine, "_request", side_effect=request):
        engine.translate("現在の全文。", "ja", "zh", context=context)
    assert [path for path, body in calls] == ["/apply-template", "/tokenize", "/v1/chat/completions"]
    prompt = calls[-1][1]["messages"][0]["content"]
    assert livestream_rules("ja", "zh", ("現在の全文。",)) in prompt
    assert prompt.endswith("現在の全文。")
    assert "〖背景信息" not in prompt
    assert calls[1][1]["content"] == prompt
    engine.close()


def test_whitespace_reference_keeps_reverse_prompt_and_adds_no_tokenizer_requests(tmp_path):
    engine = gguf_engine(tmp_path)
    with patch.object(engine, "_ensure_loaded"), patch.object(engine, "_request", return_value=completion_result()) as request:
        engine.translate("原文", "zh", "ja", context=" \n ")
    assert request.call_count == 1
    assert request.call_args.args[1] == "/v1/chat/completions"
    assert request.call_args.args[2]["messages"][0]["content"] == (
        "将以下文本翻译为日语，注意只需要输出翻译后的结果，不要额外解释：\n\n原文")
    engine.close()


@pytest.mark.parametrize("model_kind", ["hy", "qwen"])
@pytest.mark.parametrize("source", ["ja", "mixed"])
@pytest.mark.parametrize("extra_token", [0, 1])
def test_livestream_rules_without_reference_obey_complete_template_budget(
    tmp_path, model_kind, source, extra_token
):
    engine = gguf_engine(tmp_path, model_kind)
    completions = []
    tokenized = []
    current = "メンバーにありがとう、数字は2077です。"
    budget = engine.CONTEXT_WINDOW_TOKENS - engine.MAX_OUTPUT_TOKENS

    def request(method, path, body):
        if path == "/apply-template":
            assert len(body["messages"]) == 1 and body["messages"][0]["role"] == "user"
            return {"prompt": "template-prefix\n" + body["messages"][0]["content"]}
        if path == "/tokenize":
            tokenized.append(body["content"])
            return {"tokens": [1] * (budget - engine.TEMPLATE_TOKEN_RESERVE + extra_token)}
        completions.append(body)
        return completion_result()

    with patch.object(engine, "_ensure_loaded"), patch.object(engine, "_request", side_effect=request):
        if extra_token:
            with pytest.raises(ValueError, match="当前待翻译文本"):
                engine.translate(current, source, "zh")
            assert not completions
        else:
            assert engine.translate(current, source, "zh") == "当前译文"
            prompt = completions[0]["messages"][0]["content"]
            assert prompt.endswith(current + ("\n/no_think" if model_kind == "qwen" else ""))
            assert completions[0]["temperature"] == 0 and completions[0]["seed"] == 42
    assert tokenized and tokenized[0].startswith("template-prefix\n")
    assert current in tokenized[0]
    assert livestream_rules(source, "zh", (current,)) in tokenized[0]
    engine.close()


@pytest.mark.parametrize("model_kind", ["hy", "qwen"])
@pytest.mark.parametrize("source", ["ja", "mixed"])
def test_livestream_profile_discards_reference_before_any_current_text(tmp_path, model_kind, source):
    engine = gguf_engine(tmp_path, model_kind)
    completions = []
    current = "300元を返して、200元は残してください。"

    def request(method, path, body):
        if path == "/apply-template":
            return {"prompt": body["messages"][0]["content"]}
        if path == "/tokenize":
            content = body["content"]
            count = 1500 if "〖背景信息" in content else 1000 if current in content else 1
            return {"tokens": [1] * count}
        completions.append(body)
        return completion_result()

    with patch.object(engine, "_ensure_loaded"), patch.object(engine, "_request", side_effect=request):
        assert engine.translate(current, source, "zh", context="前" * 10000) == "当前译文"
    prompt = completions[0]["messages"][0]["content"]
    assert "〖背景信息" not in prompt
    assert prompt.endswith(current + ("\n/no_think" if model_kind == "qwen" else ""))
    assert livestream_rules(source, "zh", (current,)) in prompt
    engine.close()


@pytest.mark.parametrize("source", ["ja", "mixed"])
def test_missing_tokenizer_rejects_overlong_livestream_source_without_reference(tmp_path, source):
    engine = gguf_engine(tmp_path)
    completions = []

    def request(method, path, body):
        if path in ("/tokenize", "/apply-template"):
            raise RuntimeError("本地 GGUF helper 返回 HTTP 404")
        completions.append(body)
        return completion_result()

    with patch.object(engine, "_ensure_loaded"), patch.object(engine, "_request", side_effect=request):
        with pytest.raises(ValueError, match="当前待翻译文本"):
            engine.translate("長" * 1000, source, "zh")
    assert not completions
    engine.close()


@pytest.mark.parametrize("model_kind", ["hy", "qwen"])
@pytest.mark.parametrize("source", ["ja", "mixed"])
def test_tokenizer_fallback_keeps_short_livestream_source_and_bounded_reference(tmp_path, model_kind, source):
    engine = gguf_engine(tmp_path, model_kind)
    completions = []
    current = "スパチャありがとう、切り抜きはまだです。"

    def request(method, path, body):
        if path in ("/tokenize", "/apply-template"):
            raise RuntimeError("本地 GGUF helper 返回 HTTP 404")
        completions.append(body)
        return completion_result()

    with patch.object(engine, "_ensure_loaded"), patch.object(engine, "_request", side_effect=request):
        assert engine.translate(current, source, "zh", context="前" * 10000) == "当前译文"
    prompt = completions[0]["messages"][0]["content"]
    assert len(prompt.encode("utf-8")) + engine.TEMPLATE_TOKEN_RESERVE <= (
        engine.CONTEXT_WINDOW_TOKENS - engine.MAX_OUTPUT_TOKENS
    )
    if "不翻译〗\n" in prompt:
        reference = prompt.split("不翻译〗\n", 1)[1].split("\n\n", 1)[0]
        assert len(reference.encode("utf-8")) <= 256
    assert prompt.endswith(current + ("\n/no_think" if model_kind == "qwen" else ""))
    assert livestream_rules(source, "zh", (current,)) in prompt
    engine.close()


@pytest.mark.parametrize("model_kind", ["hy", "qwen"])
@pytest.mark.parametrize("source", ["ja", "mixed"])
@pytest.mark.parametrize("budget_stage", ["reference", "whole_prompt"])
def test_reference_budget_drops_complete_old_segment_without_leaving_its_tail(
    tmp_path, model_kind, source, budget_stage
):
    engine = gguf_engine(tmp_path, model_kind)
    old_segment = "OLD_BEGIN_" + "旧" * 90 + "_OLD_END"
    latest_segment = "NEW_COMPLETE_SEGMENT_否定を残す。"
    current = "現在の全文、数字は2077です。"
    completions = []

    def request(method, path, body):
        if path == "/apply-template":
            return {"prompt": body["messages"][0]["content"]}
        if path == "/tokenize":
            content = body["content"]
            whole_prompt = current in content
            over_budget = "OLD_BEGIN_" in content and (
                whole_prompt if budget_stage == "whole_prompt" else not whole_prompt
            )
            count = 1800 if whole_prompt and over_budget else 300 if over_budget else 100
            return {"tokens": [1] * count}
        completions.append(body)
        return completion_result()

    with patch.object(engine, "_ensure_loaded"), patch.object(engine, "_request", side_effect=request):
        assert engine.translate(current, source, "zh", context=old_segment + "\n" + latest_segment) == "当前译文"
    prompt = completions[0]["messages"][0]["content"]
    reference = prompt.split("不翻译〗\n", 1)[1].split("\n\n", 1)[0]
    assert reference == latest_segment
    assert "OLD_BEGIN_" not in prompt and "_OLD_END" not in prompt and "旧" not in reference
    assert prompt.endswith(current + ("\n/no_think" if model_kind == "qwen" else ""))
    engine.close()


@pytest.mark.parametrize("budget_stage", ["reference", "whole_prompt"])
def test_newest_complete_reference_is_tried_before_shortening_a_long_single_segment(tmp_path, budget_stage):
    engine = gguf_engine(tmp_path)
    old_segment = "OLD_BEGIN_短い旧文。"
    latest_segment = "NEW_COMPLETE_" + "新" * 150 + "_LATEST_END"
    current = "現在の全文。"
    completions = []

    def request(method, path, body):
        if path == "/apply-template":
            return {"prompt": body["messages"][0]["content"]}
        if path == "/tokenize":
            content = body["content"]
            whole_prompt = current in content
            over_budget = "OLD_BEGIN_" in content and (
                whole_prompt if budget_stage == "whole_prompt" else not whole_prompt
            )
            count = 1800 if whole_prompt and over_budget else 300 if over_budget else 100
            return {"tokens": [1] * count}
        completions.append(body)
        return completion_result()

    with patch.object(engine, "_ensure_loaded"), patch.object(engine, "_request", side_effect=request):
        assert engine.translate(current, "ja", "zh", context=old_segment + "\n" + latest_segment) == "当前译文"
    prompt = completions[0]["messages"][0]["content"]
    reference = prompt.split("不翻译〗\n", 1)[1].split("\n\n", 1)[0]
    assert reference == latest_segment
    assert "OLD_BEGIN_" not in prompt
    assert prompt.endswith(current)
    engine.close()
