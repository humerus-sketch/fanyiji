"""Validated, atomic settings persistence; no audio or model initialization."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

DIRECTIONS = ("zh_to_ja", "ja_to_zh")
TRANSLATION_PROVIDERS = ("local", "codex", "hybrid")
CODEX_MODEL_MAX_LENGTH = 128
CODEX_EFFORTS = ("", "low", "medium", "high", "xhigh", "max")
CODEX_SPEEDS = ("standard", "fast")
UI_LANGUAGES = ("zh-CN", "ja", "en")
AUDIO_PROCESSING_MODES = ("off", "gtcrn", "uvr")


@dataclass(frozen=True)
class SourceConfig:
    enabled: bool = True
    direction: str = "zh_to_ja"

    @property
    def source_lang(self) -> str:
        return "zh" if self.direction == "zh_to_ja" else "ja"

    @property
    def target_lang(self) -> str:
        return "ja" if self.direction == "zh_to_ja" else "zh"


@dataclass(frozen=True)
class AppConfig:
    microphone: SourceConfig = field(default_factory=SourceConfig)
    playback: SourceConfig = field(default_factory=lambda: SourceConfig(direction="ja_to_zh"))
    schema_version: int = 1
    setup_complete: bool = True
    speech_only: bool = True
    translation_provider: str = "local"
    codex_model: str = ""
    codex_effort: str = ""
    codex_speed: str = "standard"
    multilingual_asr: bool = True
    ui_language: str = "zh-CN"
    audio_processing: str = "gtcrn"

    def for_channel(self, channel: str) -> SourceConfig:
        if channel in ("mic", "microphone"):
            return self.microphone
        if channel in ("system", "playback"):
            return self.playback
        raise ValueError(f"Unknown audio source: {channel}")

    @property
    def enabled_channels(self) -> set[str]:
        return {channel for channel in ("mic", "system") if self.for_channel(channel).enabled}


class ConfigError(ValueError):
    pass


def validate_translation_settings(provider: object, model: object, effort: object = "", speed: object = "standard") -> None:
    if not isinstance(provider, str) or provider not in TRANSLATION_PROVIDERS:
        raise ConfigError("翻译源无效，仅支持本地模型、Codex CLI 或混合翻译")
    if (not isinstance(model, str) or len(model) > CODEX_MODEL_MAX_LENGTH
            or any(ord(char) < 32 or char in "\x7f\x85\u2028\u2029" for char in model)):
        raise ConfigError("Codex 模型名称无效：最多 128 个字符，不能包含换行或控制字符")
    if not isinstance(effort, str) or effort not in CODEX_EFFORTS:
        raise ConfigError("Codex 推理强度无效")
    if not isinstance(speed, str) or speed not in CODEX_SPEEDS:
        raise ConfigError("Codex 速度模式无效")


def validate_ui_language(language: object) -> None:
    if not isinstance(language, str) or language not in UI_LANGUAGES:
        raise ConfigError("界面语言无效")


def validate_audio_processing(mode: object) -> None:
    if not isinstance(mode, str) or mode not in AUDIO_PROCESSING_MODES:
        raise ConfigError("语音处理模式无效")


def _source_from_json(raw: object, name: str) -> SourceConfig:
    if not isinstance(raw, dict):
        raise ConfigError(f"{name}设置缺失")
    enabled, direction = raw.get("enabled"), raw.get("direction")
    if type(enabled) is not bool or direction not in DIRECTIONS:
        raise ConfigError(f"{name}的开关或翻译方向无效")
    return SourceConfig(enabled=enabled, direction=direction)


def load_config(path: Path) -> AppConfig | None:
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("schema_version") != 1:
            raise ConfigError("设置文件版本无效")
        if raw.get("setup_complete") is not True:
            return None
        speech_only = raw.get("speech_only", True)
        if type(speech_only) is not bool:
            raise ConfigError("讲话筛选开关无效")
        multilingual_asr = raw.get("multilingual_asr", True)
        if type(multilingual_asr) is not bool:
            raise ConfigError("自动多语识别开关无效")
        ui_language = raw.get("ui_language", "zh-CN")
        validate_ui_language(ui_language)
        audio_processing = raw.get("audio_processing", "gtcrn")
        validate_audio_processing(audio_processing)
        translation_provider = raw.get("translation_provider", "local")
        codex_model = raw.get("codex_model", "")
        codex_effort = raw.get("codex_effort", "")
        codex_speed = raw.get("codex_speed", "standard")
        validate_translation_settings(translation_provider, codex_model, codex_effort, codex_speed)
        # Read only current fields so retired feature settings do not invalidate subtitles.
        return AppConfig(
            microphone=_source_from_json(raw.get("microphone"), "麦克风"),
            playback=_source_from_json(raw.get("playback"), "播放声音"),
            speech_only=speech_only,
            translation_provider=translation_provider,
            codex_model=codex_model,
            codex_effort=codex_effort,
            codex_speed=codex_speed,
            multilingual_asr=multilingual_asr,
            ui_language=ui_language,
            audio_processing=audio_processing,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ConfigError("无法读取设置文件，请重新设置") from exc


def save_config(path: Path, config: AppConfig) -> None:
    # Validate before replacing a known-good configuration.
    _source_from_json(asdict(config.microphone), "麦克风")
    _source_from_json(asdict(config.playback), "播放声音")
    if type(config.speech_only) is not bool:
        raise ConfigError("讲话筛选开关无效")
    if type(config.multilingual_asr) is not bool:
        raise ConfigError("自动多语识别开关无效")
    validate_ui_language(config.ui_language)
    validate_audio_processing(config.audio_processing)
    validate_translation_settings(config.translation_provider, config.codex_model, config.codex_effort, config.codex_speed)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(asdict(config), stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
