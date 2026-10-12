import json
from dataclasses import replace

import pytest

from fanyiji.config import AppConfig, ConfigError, SourceConfig, load_config, save_config


@pytest.mark.parametrize("mode", ["off", "gtcrn", "uvr"])
def test_audio_processing_round_trip_and_legacy_default(tmp_path, mode):
    path = tmp_path / "settings.json"
    config = replace(AppConfig(), audio_processing=mode)
    save_config(path, config)
    assert load_config(path) == config
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw.pop("audio_processing")
    path.write_text(json.dumps(raw), encoding="utf-8")
    assert load_config(path).audio_processing == "gtcrn"


@pytest.mark.parametrize("mode", [None, [], True, "speaker", "GTCRN", "unknown"])
def test_invalid_audio_processing_cannot_replace_saved_config(tmp_path, mode):
    path = tmp_path / "settings.json"
    save_config(path, AppConfig())
    before = path.read_bytes()
    with pytest.raises(ConfigError):
        save_config(path, replace(AppConfig(), audio_processing=mode))
    assert path.read_bytes() == before
    raw = json.loads(before)
    raw["audio_processing"] = mode
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


def test_legacy_settings_enable_multilingual_and_preserve_explicit_opt_out(tmp_path):
    path = tmp_path / "settings.json"
    save_config(path, AppConfig())
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw.pop("multilingual_asr")
    path.write_text(json.dumps(raw), encoding="utf-8")
    assert load_config(path).multilingual_asr is True
    config = replace(load_config(path), multilingual_asr=False)
    save_config(path, config)
    assert load_config(path) == config


@pytest.mark.parametrize("value", [1, "true", None, []])
def test_invalid_multilingual_flag_cannot_replace_saved_settings(tmp_path, value):
    path = tmp_path / "settings.json"
    save_config(path, AppConfig())
    before = path.read_bytes()
    with pytest.raises(ConfigError):
        save_config(path, replace(AppConfig(), multilingual_asr=value))
    assert path.read_bytes() == before
    raw = json.loads(before)
    raw["multilingual_asr"] = value
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


def test_first_run_and_saved_independent_sources(tmp_path):
    path = tmp_path / "中文 路径" / "settings.json"
    assert load_config(path) is None
    config = AppConfig(SourceConfig(False, "ja_to_zh"), SourceConfig(True, "ja_to_zh"))
    save_config(path, config)
    assert load_config(path) == config
    assert load_config(path).enabled_channels == {"system"}


def test_both_disabled_is_valid(tmp_path):
    path = tmp_path / "settings.json"
    config = AppConfig(SourceConfig(False, "zh_to_ja"), SourceConfig(False, "ja_to_zh"))
    save_config(path, config)
    assert load_config(path).enabled_channels == set()


@pytest.mark.parametrize("raw", ["broken json", "[]", '{"schema_version": 2}',
    '{"schema_version":1,"setup_complete":true,"microphone":{"enabled":"false","direction":"zh_to_ja"},"playback":{}}'])
def test_invalid_configuration_cannot_start_capture(tmp_path, raw):
    path = tmp_path / "settings.json"
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


def test_incomplete_setup_is_still_first_run(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"schema_version":1,"setup_complete":False}), encoding="utf-8")
    assert load_config(path) is None


def test_legacy_settings_enable_speech_filter_and_preserve_opt_out(tmp_path):
    path = tmp_path / "settings.json"
    raw = {"schema_version": 1, "setup_complete": True,
           "microphone": {"enabled": False, "direction": "zh_to_ja"},
           "playback": {"enabled": True, "direction": "ja_to_zh"}}
    path.write_text(json.dumps(raw), encoding="utf-8")
    assert load_config(path).speech_only is True
    raw["speech_only"] = False
    path.write_text(json.dumps(raw), encoding="utf-8")
    config = load_config(path)
    save_config(path, config)
    assert load_config(path).speech_only is False


def test_legacy_settings_keep_local_translation_without_new_fields(tmp_path):
    path = tmp_path / "settings.json"
    raw = {"schema_version": 1, "setup_complete": True,
           "microphone": {"enabled": False, "direction": "zh_to_ja"},
           "playback": {"enabled": True, "direction": "ja_to_zh"}}
    path.write_text(json.dumps(raw), encoding="utf-8")
    loaded = load_config(path)
    assert loaded.translation_provider == "local"
    assert loaded.codex_model == ""
    assert loaded.codex_effort == "" and loaded.codex_speed == "standard"
    assert loaded.enabled_channels == {"system"}


@pytest.mark.parametrize("model", ["", "gpt-account-model"])
def test_codex_settings_round_trip_and_preserve_model_when_returning_to_local(tmp_path, model):
    path = tmp_path / "settings.json"
    config = AppConfig(microphone=SourceConfig(False, "ja_to_zh"),
                       translation_provider="codex", codex_model=model, speech_only=False)
    save_config(path, config)
    assert load_config(path) == config
    local = replace(config, translation_provider="local")
    save_config(path, local)
    assert load_config(path) == local


@pytest.mark.parametrize("provider,model", [
    ("api", ""), (None, ""), ([], ""), ("codex\n", ""),
    ("codex", None), ("codex", 123), ("codex", "a" * 129),
    ("codex", "gpt-model\nother"), ("codex", "gpt-model\r"),
    ("codex", "gpt-model\x00"), ("codex", "gpt-model\u2028other"),
])
def test_invalid_translation_settings_rejected_before_overwriting(tmp_path, provider, model):
    path = tmp_path / "settings.json"
    save_config(path, AppConfig())
    before = path.read_bytes()
    invalid = replace(AppConfig(), translation_provider=provider, codex_model=model)
    with pytest.raises(ConfigError):
        save_config(path, invalid)
    assert path.read_bytes() == before
    raw = json.loads(before)
    raw.update(translation_provider=provider, codex_model=model)
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


@pytest.mark.parametrize("value", [1, "true", None])
def test_invalid_speech_filter_flag_rejected_without_replacing_settings(tmp_path, value):
    path = tmp_path / "settings.json"
    save_config(path, AppConfig())
    before = path.read_bytes()
    with pytest.raises(ConfigError):
        save_config(path, replace(AppConfig(), speech_only=value))
    assert path.read_bytes() == before


def test_codex_model_controls_survive_switch_to_offline_and_restart(tmp_path):
    path = tmp_path / "settings.json"
    config = AppConfig(translation_provider="codex", codex_model="gpt-6.1-sol",
                       codex_effort="low", codex_speed="fast")
    save_config(path, config)
    assert load_config(path) == config
    save_config(path, replace(config, translation_provider="local"))
    assert load_config(path) == replace(config, translation_provider="local")


@pytest.mark.parametrize("effort,speed", [("ultra", "standard"), (None, "standard"),
                                        ("low", "ultrafast"), ("", None)])
def test_invalid_codex_controls_cannot_overwrite_last_configuration(tmp_path, effort, speed):
    path = tmp_path / "settings.json"
    save_config(path, AppConfig())
    before = path.read_bytes()
    with pytest.raises(ConfigError):
        save_config(path, AppConfig(codex_effort=effort, codex_speed=speed))
    assert path.read_bytes() == before


@pytest.mark.parametrize("retired_fields", [
    {
        "danmaku_enabled": True,
        "danmaku_room": "1698225",
        "danmaku_source": "system",
        "danmaku_dedicated_browser": True,
        "danmaku_accounts": [{"id": "default", "name": "旧账号", "enabled": True}],
        "danmaku_rotation_enabled": True,
        "danmaku_active_account": "default",
        "danmaku_retry_cooldown_seconds": 1.0,
    },
    {
        "danmaku_enabled": "invalid",
        "danmaku_room": None,
        "danmaku_source": [],
        "danmaku_dedicated_browser": 1,
        "danmaku_accounts": {"invalid": "old account data"},
        "danmaku_rotation_enabled": None,
        "danmaku_active_account": "../invalid",
        "danmaku_retry_cooldown_seconds": "invalid",
        "danmaku_unknown_future_field": {"preserved_on_disk": True},
    },
])
def test_retired_feature_fields_do_not_discard_subtitle_settings(tmp_path, retired_fields):
    path = tmp_path / "settings.json"
    config = AppConfig(
        microphone=SourceConfig(False, "ja_to_zh"),
        playback=SourceConfig(True, "zh_to_ja"),
        speech_only=False,
        translation_provider="codex",
        codex_model="gpt-account-model",
        codex_effort="low",
        codex_speed="fast",
        multilingual_asr=False,
        ui_language="ja",
    )
    save_config(path, config)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw.update(retired_fields)
    path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    before = path.read_bytes()

    loaded = load_config(path)
    assert loaded == config
    assert path.read_bytes() == before
    assert not any(name.startswith("danmaku_") for name in vars(loaded))

    save_config(path, loaded)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert not any(name.startswith("danmaku_") for name in saved)
    assert load_config(path) == config


def test_atomic_replace_failure_keeps_original_subtitle_settings(tmp_path, monkeypatch):
    path = tmp_path / "settings.json"
    saved = AppConfig()
    save_config(path, saved)
    before = path.read_bytes()

    def fail_replace(*_args):
        raise OSError("synthetic replace failure")

    monkeypatch.setattr("fanyiji.config.os.replace", fail_replace)
    with pytest.raises(OSError):
        save_config(path, replace(saved, ui_language="en", speech_only=False))
    assert path.read_bytes() == before
    assert load_config(path) == saved
