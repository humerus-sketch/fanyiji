"""Synthetic waveforms and recognizer stubs; no models, microphones or network."""
import json
import hashlib
import threading
from types import SimpleNamespace

import numpy as np
import pytest

import fanyiji.asr_engine as module
import fanyiji.engines as legacy
from fanyiji.asr_engine import create_recognizer_engine as create_production_recognizer_engine
from fanyiji.engines import ModelAssetsError
from fanyiji.asr_protocol import AsrProtocolError


def create_recognizer_engine(directory, num_threads=4):
    """Exercise archived adapters directly; production selection is tested separately."""
    profile = directory / "backend.json"
    config = json.loads(profile.read_text(encoding="utf-8")) if profile.is_file() else {}
    if config.get("backend", "sense_voice") == "sense_voice":
        return create_production_recognizer_engine(directory, num_threads=num_threads)
    return module.ProfileRecognizerEngine(directory, num_threads, config)


@pytest.mark.parametrize("backend", ["whisper", "qwen3_asr", "external"])
def test_production_factory_rejects_other_asr_before_loading(tmp_path, monkeypatch, backend):
    (tmp_path / "backend.json").write_text(json.dumps({"backend": backend}), encoding="utf-8")
    dependencies = []
    monkeypatch.setattr(module, "_dependency", lambda name: dependencies.append(name))
    with pytest.raises(ModelAssetsError, match="仅支持 SenseVoice"):
        create_production_recognizer_engine(tmp_path)
    assert dependencies == []


def test_production_sensevoice_default_uses_four_cpu_threads(tmp_path, monkeypatch):
    assets(tmp_path, "sense_voice")
    calls = stub(monkeypatch)
    engine = create_production_recognizer_engine(tmp_path)
    engine.preload(("auto",))
    assert calls[0][0] == "from_sense_voice"
    assert calls[0][1]["num_threads"] == 4
    engine.close()


def test_sensevoice_without_backend_profile_releases_sessions_on_close(tmp_path, monkeypatch):
    import weakref
    assets(tmp_path, "sense_voice")
    stub(monkeypatch)
    engine = legacy.RecognizerEngine(tmp_path)
    engine.preload(("auto",))
    native = weakref.ref(engine._recognizers["auto"])
    assert native() is not None
    engine.close()
    engine.close()
    assert engine._recognizers == {}
    assert native() is None
    with pytest.raises(RuntimeError, match="已关闭"):
        engine.preload(("auto",))
    with pytest.raises(RuntimeError, match="已关闭"):
        engine.recognize(np.zeros(16000, np.float32), "auto")


def assets(root, kind="whisper"):
    if kind == "sense_voice":
        files = ("model.int8.onnx", "tokens.txt")
        profile = {"backend": kind}
    elif kind == "whisper":
        profile = {"backend": kind, "encoder": "whisper/encoder.int8.onnx",
                   "decoder": "whisper/decoder.int8.onnx", "tokens": "whisper/tokens.txt"}
        files = tuple(value for key, value in profile.items() if key != "backend")
    else:
        profile = {"backend": kind, "conv_frontend": "qwen/conv_frontend.onnx",
                   "encoder": "qwen/encoder.int8.onnx", "decoder": "qwen/decoder.int8.onnx",
                   "tokenizer": "qwen/tokenizer"}
        files = tuple(value for key, value in profile.items() if key not in ("backend", "tokenizer"))
        files += tuple("qwen/tokenizer/" + name for name in ("vocab.json", "merges.txt", "tokenizer_config.json"))
    for name in files:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"c3R1Yg== 0\n" if kind == "whisper" and name == profile["tokens"] else b"synthetic asset")
    (root / "backend.json").write_text(json.dumps(profile), encoding="utf-8")
    return profile, files


def stub(monkeypatch, *, text="测试结果", tokens=(), decode=None, whisper_markers=True, options=None,
         detected_language=None):
    calls = []
    class Recognizer:
        def __init__(self, kind):
            self.kind = kind
        def create_stream(self):
            def accept(rate, samples):
                assert rate == 16000 and samples.dtype == np.float32 and samples.ndim == 1
            def set_option(name, value):
                if options is not None:
                    options.append((name, value))
            output = "__WBYTE_" + text.encode("utf-8").hex() + "__" if self.kind == "from_whisper" and whisper_markers else text
            return SimpleNamespace(accept_waveform=accept, set_option=set_option,
                                   result=SimpleNamespace(text=output, tokens=tokens, lang=detected_language))
        def decode_stream(self, stream):
            if decode:
                decode(stream)
    def build(kind, kwargs):
        calls.append((kind, kwargs))
        return Recognizer(kind)
    offline = SimpleNamespace(**{name: (lambda name=name, **kw: build(name, kw))
                                 for name in ("from_whisper", "from_qwen3_asr", "from_sense_voice")})
    native = SimpleNamespace(OfflineRecognizer=offline)
    def dependency(name):
        return np if name == "numpy" else native
    monkeypatch.setattr(module, "_dependency", dependency)
    monkeypatch.setattr(legacy, "_dependency", dependency)
    monkeypatch.setattr(module, "WHISPER_TOKEN_SHA256", hashlib.sha256(b"c3R1Yg== 0\n").hexdigest())
    return calls


@pytest.mark.parametrize("raw,body", [
    ("language None<asr_text>", ""), ("<asr_text>", ""),
    ("language None<asr_text>はい。", "はい。"),
    (" LANGUAGE None <asr_text>We need ammo.", "We need ammo."),
    ("language Korean<asr_text>안녕하세요", "안녕하세요"),
])
def test_qwen_unknown_or_empty_language_protocol_is_not_a_subtitle(raw, body, tmp_path, monkeypatch):
    assets(tmp_path, "qwen3_asr")
    stub(monkeypatch, text=raw)
    engine = create_recognizer_engine(tmp_path)
    result = engine.recognize_detailed(np.ones(16, dtype=np.float32), "auto")
    assert result.text == body
    assert result.detected_language is None and result.detection_source == "unknown"
    engine.close()


@pytest.mark.parametrize("raw", [
    "language Unknown<asr_text>hello", "language Japanese<asr_text>text <asr_text>",
    "这是 <asr_text> 这个标记", "language None", "&lt;asr_text&gt;",
])
def test_qwen_residual_protocol_is_rejected_instead_of_erasing_body(raw, tmp_path, monkeypatch):
    assets(tmp_path, "qwen3_asr")
    stub(monkeypatch, text=raw)
    engine = create_recognizer_engine(tmp_path)
    with pytest.raises(AsrProtocolError):
        engine.recognize(np.ones(16, dtype=np.float32), "auto")
    engine.close()


def test_missing_profile_keeps_sensevoice_contract_and_assets(tmp_path, monkeypatch):
    for name in ("model.int8.onnx", "tokens.txt"):
        (tmp_path / name).write_bytes(b"stub")
    calls = stub(monkeypatch)
    engine = create_recognizer_engine(tmp_path)
    assert engine.asset_files == ("model.int8.onnx", "tokens.txt")
    assert engine.backend_info["model_id"] == "sensevoice-int8"
    engine.preload(("zh", "ja"))
    assert [kwargs["language"] for _, kwargs in calls] == ["zh", "ja"]
    assert engine.recognize(np.ones(16, dtype=np.float32), "ja") == "测试结果"
    engine.close()
    with pytest.raises(RuntimeError, match="已关闭"):
        engine.preload(("zh",))


def test_named_sensevoice_uses_nested_import_assets_and_reports_display_name(tmp_path, monkeypatch):
    model_id = "local-" + "1" * 32
    relative = "imported/" + model_id + "/"
    for filename in ("model.int8.onnx", "tokens.txt"):
        path = tmp_path / relative / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic")
    profile = {"backend": "sense_voice", "model_id": model_id, "display_name": "专用识别",
               "model": relative + "model.int8.onnx", "tokens": relative + "tokens.txt"}
    (tmp_path / "backend.json").write_text(json.dumps(profile), encoding="utf-8")
    calls = stub(monkeypatch)
    engine = create_recognizer_engine(tmp_path)
    try:
        engine.preload(("auto",))
        assert engine.backend_info["model_id"] == model_id
        assert engine.backend_info["display_name"] == "专用识别"
        assert calls[0][1]["model"] == str(tmp_path / profile["model"])
        assert calls[0][1]["tokens"] == str(tmp_path / profile["tokens"])
        assert not (tmp_path / "model.int8.onnx").exists()
    finally:
        engine.close()


def test_whisper_lazily_loads_each_language_with_passed_thread_budget(tmp_path, monkeypatch):
    _, files = assets(tmp_path)
    calls = stub(monkeypatch)
    engine = create_recognizer_engine(tmp_path, num_threads=4)
    assert engine.asset_files == files and calls == []
    engine.preload(("ja",))
    engine.recognize(np.ones(16, dtype=np.float32), "zh")
    engine.recognize(np.ones(16, dtype=np.float32), "ja")
    assert [kw["language"] for _, kw in calls] == ["ja", "zh"]
    assert all(name == "from_whisper" and kw["task"] == "transcribe"
               and kw["provider"] == "cpu" and kw["num_threads"] == 4 for name, kw in calls)
    assert engine.backend_info["recognizer_instances"] == 2
    engine.close()


def test_qwen_shares_model_uses_per_stream_language_and_fixed_generation_cap(tmp_path, monkeypatch):
    assets(tmp_path, "qwen3_asr")
    options = []
    calls = stub(monkeypatch, options=options)
    engine = create_recognizer_engine(tmp_path, num_threads=4)
    engine.preload(("zh", "ja"))
    engine.recognize(np.ones(16, dtype=np.float32), "ja")
    engine.recognize(np.ones(16, dtype=np.float32), "zh")
    assert len(calls) == 1 and calls[0][0] == "from_qwen3_asr"
    kwargs = calls[0][1]
    assert "language" not in kwargs and kwargs["num_threads"] == 4
    assert kwargs["max_total_len"] == 512 and kwargs["max_new_tokens"] == 128
    assert options == [("language", "Japanese"), ("language", "Chinese")]
    assert engine.backend_info["language_selection"] == "auto or per-stream source language (Chinese/Japanese/English)"
    engine.close()


@pytest.mark.parametrize(("raw", "clean"), [
    ("language Japanese<asr_text>まあまあ", "まあまあ"),
    ("language Chinese<asr_text>测试", "测试"),
    ("language English<asr_text>Hello", "Hello"),
    ("<asr_text>こんにちは", "こんにちは"),
    ("正文 language Japanese 保留", "正文 language Japanese 保留"),
    ("language learning 保留", "language learning 保留"),
])
def test_qwen_only_removes_leading_known_protocol(raw, clean, tmp_path, monkeypatch):
    assets(tmp_path, "qwen3_asr")
    stub(monkeypatch, text=raw)
    engine = create_recognizer_engine(tmp_path)
    assert engine.recognize(np.ones(16, dtype=np.float32), "ja") == clean
    engine.close()


@pytest.mark.parametrize("token_count", [112, 125, 128])
def test_qwen_near_token_limit_rejects_cleaned_tokens_even_with_short_text(tmp_path, monkeypatch, token_count):
    assets(tmp_path, "qwen3_asr")
    stub(monkeypatch, text="短句", tokens=tuple(range(token_count)))
    engine = create_recognizer_engine(tmp_path)
    with pytest.raises(RuntimeError, match="128 token"):
        engine.recognize(np.ones(16, dtype=np.float32), "ja")
    engine.close()


def test_qwen_111_cleaned_tokens_allowed_and_guard_is_public(tmp_path, monkeypatch):
    assets(tmp_path, "qwen3_asr")
    stub(monkeypatch, text="完整短句", tokens=tuple(range(111)))
    engine = create_recognizer_engine(tmp_path)
    assert engine.recognize(np.ones(16, dtype=np.float32), "ja") == "完整短句"
    assert engine.backend_info["generation"]["cleaned_token_guard"] == 112
    engine.close()


def test_close_waits_for_active_native_call_and_prevents_reload(tmp_path, monkeypatch):
    assets(tmp_path)
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()
    def decode(stream):
        entered.set()
        assert release.wait(2)
    stub(monkeypatch, decode=decode)
    engine = create_recognizer_engine(tmp_path)
    call = threading.Thread(target=engine.recognize, args=(np.ones(16, dtype=np.float32), "ja"))
    shutdown = threading.Thread(target=lambda: (engine.close(), closed.set()))
    try:
        call.start()
        assert entered.wait(1)
        shutdown.start()
        assert not closed.wait(0.05)
        release.set()
        call.join(2)
        shutdown.join(2)
        assert closed.is_set() and not call.is_alive() and not shutdown.is_alive()
        with pytest.raises(RuntimeError, match="已关闭"):
            engine.preload(("ja",))
    finally:
        release.set()
        call.join(2)
        if shutdown.ident:
            shutdown.join(2)


@pytest.mark.parametrize("change", [
    {"encoder": "../../outside.onnx"}, {"encoder": "C:/outside.onnx"},
    {"encoder": "whisper/model.onnx:alternate_stream"},
    {"backend": "external"}, {"provider": "cuda"},
])
def test_profile_rejects_unknown_backend_fields_and_escaping_paths(tmp_path, change):
    profile, _ = assets(tmp_path)
    profile.update(change)
    (tmp_path / "backend.json").write_text(json.dumps(profile), encoding="utf-8")
    with pytest.raises(ModelAssetsError):
        create_recognizer_engine(tmp_path)


def test_qwen_missing_tokenizer_file_fails_before_native_load(tmp_path):
    assets(tmp_path, "qwen3_asr")
    (tmp_path / "qwen/tokenizer/vocab.json").write_bytes(b"")
    with pytest.raises(ModelAssetsError):
        create_recognizer_engine(tmp_path)


def test_whisper_constructor_uses_derived_table_and_original_assets_remain(tmp_path, monkeypatch):
    profile, files = assets(tmp_path)
    original = (tmp_path / profile["tokens"]).read_bytes()
    calls = stub(monkeypatch)
    engine = create_recognizer_engine(tmp_path)
    engine.preload(("ja", "zh"))
    derived = [kwargs["tokens"] for _, kwargs in calls]
    assert derived[0] == derived[1] and derived[0] != str(tmp_path / profile["tokens"])
    assert engine.asset_files == files and (tmp_path / profile["tokens"]).read_bytes() == original
    assert engine.backend_info["output_recovery"]["original_token_ids_preserved"]
    engine.close()


def test_whisper_preserves_visible_incomplete_utf8_and_records_count(tmp_path, monkeypatch):
    assets(tmp_path)
    stub(monkeypatch, text="__WBYTE_e6b1__", whisper_markers=False)
    engine = create_recognizer_engine(tmp_path)
    assert engine.recognize(np.ones(16, dtype=np.float32), "ja") == "\ufffd"
    info = engine.backend_info["output_recovery"]
    assert info["last_replacement_characters"] == 1 and info["total_replacement_characters"] == 1
    engine.close()


def test_whisper_bad_output_is_explicit_recognition_failure(tmp_path, monkeypatch):
    assets(tmp_path)
    stub(monkeypatch, text="__WBYTE_e6b1____WBYTE_9a__extra", whisper_markers=False)
    engine = create_recognizer_engine(tmp_path)
    with pytest.raises(RuntimeError, match="输出标记无法解析，识别失败"):
        engine.recognize(np.ones(16, dtype=np.float32), "ja")
    engine.close()


def test_qwen_unknown_source_language_rejected_before_constructor(tmp_path, monkeypatch):
    assets(tmp_path, "qwen3_asr")
    calls = stub(monkeypatch)
    engine = create_recognizer_engine(tmp_path)
    with pytest.raises(ValueError):
        engine.recognize(np.ones(16, dtype=np.float32), "ko")
    assert not calls
    engine.close()


def test_qwen_auto_omits_language_hint_and_shares_fixed_english_model(tmp_path, monkeypatch):
    assets(tmp_path, "qwen3_asr")
    options = []
    calls = stub(monkeypatch, text="Please wait", options=options)
    engine = create_recognizer_engine(tmp_path)
    engine.preload(("auto", "en", "ja", "zh"))
    automatic = engine.recognize_detailed(np.ones(16, dtype=np.float32), "auto")
    assert automatic.text == "Please wait" and automatic.detected_language is None
    assert automatic.detection_source == "unknown" and options == []
    english = engine.recognize_detailed(np.ones(16, dtype=np.float32), "en")
    assert english.detected_language is None  # A forced hint is not a detection.
    assert options == [("language", "English")] and len(calls) == 1
    engine.recognize(np.ones(16, dtype=np.float32), "auto")
    assert options == [("language", "English")]
    engine.close()


@pytest.mark.parametrize(("kind", "native_languages"), [
    ("whisper", ["", "en", "ja", "zh"]),
    ("sense_voice", ["auto", "en", "ja", "zh"]),
])
def test_auto_english_and_legacy_language_caches_have_native_values(kind, native_languages, tmp_path, monkeypatch):
    assets(tmp_path, kind)
    calls = stub(monkeypatch, detected_language="English")
    engine = create_recognizer_engine(tmp_path)
    engine.preload(("auto", "en", "ja", "zh"))
    for language in ("auto", "en", "ja", "zh"):
        result = engine.recognize_detailed(np.ones(16, dtype=np.float32), language)
        assert result.detected_language == "en" and result.detection_source == "native_result"
    assert [kw["language"] for _, kw in calls] == native_languages
    assert len(calls) == 4
    engine.close()


@pytest.mark.parametrize(("raw", "native", "language", "source"), [
    ("language Japanese<asr_text>テスト", None, "ja", "qwen_protocol"),
    ("language Chinese<asr_text>测试", None, "zh", "qwen_protocol"),
    ("language English<asr_text>test", None, "en", "qwen_protocol"),
    ("language English<asr_text>test", "Japanese", "ja", "native_result"),
    ("<asr_text>test", None, None, "unknown"),
    ("language learning test", None, None, "unknown"),
    ("ordinary language Japanese words", None, None, "unknown"),
    ("test", "Korean", None, "unknown"),
])
def test_qwen_detailed_detection_uses_only_native_or_known_leading_header(raw, native, language, source,
                                                                      tmp_path, monkeypatch):
    assets(tmp_path, "qwen3_asr")
    stub(monkeypatch, text=raw, detected_language=native)
    engine = create_recognizer_engine(tmp_path)
    result = engine.recognize_detailed(np.ones(16, dtype=np.float32), "auto")
    assert result.detected_language == language and result.detection_source == source
    assert "confidence" not in result.__dict__
    engine.close()


@pytest.mark.parametrize("kind", ["whisper", "sense_voice", "qwen3_asr"])
def test_empty_rich_result_neither_loads_nor_infers_language(kind, tmp_path, monkeypatch):
    assets(tmp_path, kind)
    calls = stub(monkeypatch)
    engine = create_recognizer_engine(tmp_path)
    result = engine.recognize_detailed(np.zeros(0, dtype=np.float32), "auto")
    assert result.text == "" and result.detected_language is None and result.detection_source == "unknown"
    assert not calls
    engine.close()
