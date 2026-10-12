"""Optional fixed local ASR profiles, retaining the SenseVoice calling contract."""
from __future__ import annotations

import json
import re
from pathlib import Path
import threading

from .engines import ModelAssetsError, RecognizerEngine, _dependency, _require_files, sanitize_sensevoice
from .language_policy import AsrResult, normalize_detected_language, validate_asr_language
from .asr_protocol import require_subtitle_text
from .whisper_tokens import METHOD, WHISPER_TOKEN_SHA256, prepare_whisper_tokens, restore_whisper_bytes


_QWEN_LANGUAGES = (
    "Chinese", "English", "Cantonese", "Arabic", "German", "French", "Spanish", "Portuguese",
    "Indonesian", "Italian", "Korean", "Russian", "Thai", "Vietnamese", "Japanese", "Turkish",
    "Hindi", "Malay", "Dutch", "Swedish", "Danish", "Finnish", "Polish", "Czech", "Filipino",
    "Persian", "Greek", "Romanian", "Hungarian", "Macedonian", "None",
)
_QWEN_PREFIX = re.compile(
    r"^(?:language\s+(?P<language>" + "|".join(_QWEN_LANGUAGES) + r")\s*)?<asr_text>",
    re.IGNORECASE,
)
_QWEN_OUTPUT_GUARD = 112


def _relative(root: Path, value, name: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ModelAssetsError(f"ASR {name} 必须是本地相对路径")
    if any(part in (".", "..") or ":" in part or part.endswith((" ", ".")) for part in Path(value).parts):
        raise ModelAssetsError(f"ASR {name} 相对路径无效")
    result = (root / value).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ModelAssetsError(f"ASR {name} 不得越出识别模型目录")
    return result


class ProfileRecognizerEngine(RecognizerEngine):
    """CPU Whisper per language, or one Qwen3-ASR model with per-stream hints."""

    def __init__(self, model_dir: Path, num_threads: int, config: dict):
        if not 1 <= num_threads <= 4:
            raise ValueError("ASR num_threads 必须在 1 到 4 之间")
        self.model_dir = Path(model_dir).resolve()
        self.num_threads = num_threads
        self.kind = config.get("backend", "sense_voice")
        self.model_id = config.get("model_id", "sensevoice-int8" if self.kind == "sense_voice" else self.kind)
        self.display_name = config.get("display_name")
        if self.display_name is not None and (not isinstance(self.display_name, str) or not 1 <= len(self.display_name) <= 80):
            raise ModelAssetsError("模型显示名称必须是 1 到 80 个字符")
        allowed = {"backend", "model_id", "display_name"}
        if self.kind == "whisper":
            allowed |= {"encoder", "decoder", "tokens"}
            fields = ("encoder", "decoder", "tokens")
        elif self.kind == "qwen3_asr":
            allowed |= {"conv_frontend", "encoder", "decoder", "tokenizer"}
            fields = ("conv_frontend", "encoder", "decoder", "tokenizer")
        elif self.kind == "sense_voice":
            allowed |= {"model", "tokens"}
            config = {**config, "model": config.get("model", "model.int8.onnx"),
                      "tokens": config.get("tokens", "tokens.txt")}
            fields = ("model", "tokens")
        else:
            raise ModelAssetsError("ASR backend 必须是 sense_voice/whisper/qwen3_asr")
        if set(config) - allowed:
            raise ModelAssetsError("ASR backend.json 包含未支持的配置字段")
        self._paths = {field: _relative(self.model_dir, config.get(field), field) for field in fields}
        if self.kind == "sense_voice":
            self._sensevoice_model, self._sensevoice_tokens = self._paths["model"], self._paths["tokens"]
            assets = tuple(path.relative_to(self.model_dir).as_posix() for path in self._paths.values())
        else:
            assets = tuple(path.relative_to(self.model_dir).as_posix() for field, path in self._paths.items() if field != "tokenizer")
            if self.kind == "qwen3_asr":
                assets += tuple(_relative(self.model_dir, str(self._paths["tokenizer"].relative_to(self.model_dir) / name),
                                         "tokenizer").relative_to(self.model_dir).as_posix()
                                for name in ("vocab.json", "merges.txt", "tokenizer_config.json"))
        self.asset_files = assets
        _require_files(self.model_dir, assets)
        self._recognizers = {}
        self._lock = threading.RLock()
        self._closed = False
        self._whisper_tokens = None
        self._last_utf8_replacements = 0
        self._utf8_replacements_total = 0

    @property
    def backend_info(self) -> dict:
        information = {"backend": "sherpa_onnx", "model_kind": self.kind, "model_id": self.model_id,
                "device": "cpu", "num_threads": self.num_threads, "sample_rate": 16000,
                "language_selection": "auto or per-stream source language (Chinese/Japanese/English)" if self.kind == "qwen3_asr" else "auto/zh/ja/en per recognizer",
                "recognizer_instances": len(self._recognizers),
                "generation": {"max_total_len": 512, "max_new_tokens": 128,
                               "temperature": 0.000001, "top_p": 0.8, "seed": 42,
                               "cleaned_token_guard": _QWEN_OUTPUT_GUARD,
                               "guard_reason": "native may strip up to 16 prefix tokens; near-cap output may be incomplete"}
                              if self.kind == "qwen3_asr" else None}
        if self.display_name:
            information["display_name"] = self.display_name
        if self.kind == "whisper":
            table = self._whisper_tokens
            information["output_recovery"] = {
                "method": METHOD, "original_token_ids_preserved": True,
                "original_model_files_modified": False,
                "source_token_sha256": table.source_sha256 if table else WHISPER_TOKEN_SHA256,
                "derived_token_sha256": table.derived_sha256 if table else None,
                "utf8_errors": "replace", "last_replacement_characters": self._last_utf8_replacements,
                "total_replacement_characters": self._utf8_replacements_total,
            }
        return information

    def _recognizer(self, language: str):
        language = validate_asr_language(language)
        if self._closed:
            raise RuntimeError("本地识别引擎已关闭")
        if self.kind == "sense_voice":
            return super()._recognizer(language)
        key = "shared" if self.kind == "qwen3_asr" else language
        if key not in self._recognizers:
            _require_files(self.model_dir, self.asset_files)
            sherpa = _dependency("sherpa_onnx")
            common = {"num_threads": self.num_threads, "provider": "cpu", "debug": False,
                      "decoding_method": "greedy_search"}
            try:
                if self.kind == "whisper":
                    self._whisper_tokens = prepare_whisper_tokens(
                        self._paths["tokens"], expected_sha256=WHISPER_TOKEN_SHA256)
                    paths = {field: str(path) for field, path in self._paths.items()}
                    paths["tokens"] = str(self._whisper_tokens.path)
                    recognizer = sherpa.OfflineRecognizer.from_whisper(
                        **paths,
                        language="" if language == "auto" else language, task="transcribe", **common)
                else:
                    # Its constructor has no language argument. One shared
                    # model reads the source language from each new stream.
                    recognizer = sherpa.OfflineRecognizer.from_qwen3_asr(
                        **{field: str(path) for field, path in self._paths.items()},
                        sample_rate=16000, feature_dim=128, max_total_len=512, max_new_tokens=128,
                        temperature=0.000001, top_p=0.8, seed=42, **common)
            except Exception as exc:
                raise ModelAssetsError(f"无法加载本地 {self.kind} ASR 模型：{exc}") from exc
            self._recognizers[key] = recognizer
        return self._recognizers[key]

    def recognize(self, samples, language: str) -> str:
        return self.recognize_detailed(samples, language).text

    def recognize_detailed(self, samples, language: str) -> AsrResult:
        if self.kind == "sense_voice":
            return super().recognize_detailed(samples, language)
        language = validate_asr_language(language)
        numpy = _dependency("numpy")
        if not isinstance(samples, numpy.ndarray):
            raise TypeError("samples 必须是 numpy float32 数组")
        if samples.ndim != 1 or samples.dtype != numpy.float32 or not numpy.isfinite(samples).all():
            raise ValueError("samples 必须是有限值的一维 16 kHz float32 数组")
        if not samples.size:
            return AsrResult("")
        with self._lock:
            recognizer = self._recognizer(language)
            stream = recognizer.create_stream()
            if self.kind == "qwen3_asr" and language != "auto":
                # In sherpa-onnx 1.13.8 an omitted option leaves the language
                # prefix empty. Sending the literal "auto" would force it.
                stream.set_option("language", {"zh": "Chinese", "ja": "Japanese", "en": "English"}[language])
            stream.accept_waveform(16000, numpy.ascontiguousarray(samples))
            recognizer.decode_stream(stream)
            if self.kind == "whisper":
                try:
                    text = restore_whisper_bytes(stream.result.text)
                except ValueError as exc:
                    raise RuntimeError(str(exc)) from exc
                self._last_utf8_replacements = text.count("\ufffd")
                self._utf8_replacements_total += self._last_utf8_replacements
                detected = normalize_detected_language(getattr(stream.result, "lang", None))
                return AsrResult(sanitize_sensevoice(text), detected,
                                 "native_result" if detected else "unknown")
            # Native output removes protocol prefix IDs. A real 128-token
            # truncated result contained only 125 exposed tokens, so retain a
            # conservative margin rather than claiming an exact cap detector.
            if len(stream.result.tokens) >= _QWEN_OUTPUT_GUARD:
                raise RuntimeError("Qwen识别接近128 token生成上限，可能不完整，该段未翻译；请缩短语音片段")
            # Some exported Qwen models retain their protocol header. Strip
            # only that leading header; ordinary text containing it is kept.
            raw = stream.result.text.lstrip("\ufeff \t\r\n")
            header = _QWEN_PREFIX.match(raw)
            detected = normalize_detected_language(getattr(stream.result, "lang", None))
            source = "native_result" if detected else "unknown"
            if header is not None and (header.group("language") or "").casefold() == "none":
                # Qwen's official parser preserves any body, with no language
                # claim. A header-only None result is empty, never a subtitle.
                detected, source = None, "unknown"
            if detected is None and header is not None:
                detected = normalize_detected_language(header.group("language"))
                source = "qwen_protocol" if detected else "unknown"
            clean = _QWEN_PREFIX.sub("", raw, count=1)
            require_subtitle_text(clean)
            return AsrResult(sanitize_sensevoice(clean),
                             detected, source)

    def close(self) -> None:
        # Native CPU inference is not safely interruptible. Serialized teardown
        # waits for the current call before releasing these ONNX sessions.
        with self._lock:
            self._closed = True
            self._recognizers.clear()


def create_recognizer_engine(asr_dir: str | Path, num_threads: int = 4):
    """Production ASR uses SenseVoice; legacy adapters remain for archived evaluations."""
    directory = Path(asr_dir).resolve()
    profile = directory / "backend.json"
    if not profile.is_file():
        config = {"backend": "sense_voice"}
    else:
        try:
            if not profile.resolve().is_relative_to(directory):
                raise ModelAssetsError("ASR backend.json 不得越出识别模型目录")
            if profile.stat().st_size > 65536:
                raise ModelAssetsError("ASR backend.json 过大")
            config = json.loads(profile.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as exc:
            raise ModelAssetsError("无法读取本地识别后端配置") from exc
        if not isinstance(config, dict):
            raise ModelAssetsError("ASR backend.json 必须是 JSON 对象")
    if config.get("backend", "sense_voice") != "sense_voice":
        raise ModelAssetsError("当前版本仅支持 SenseVoice 识别，请在模型管理中启用 SenseVoice。")
    return ProfileRecognizerEngine(directory, num_threads, config)
