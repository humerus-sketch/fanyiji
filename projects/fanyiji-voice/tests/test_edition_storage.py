"""Voice preferences stay isolated while external model weights may be shared."""
import builtins
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

from fanyiji import paths
from fanyiji.identity import APP_ID


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    for name in ("DATA", "MODELS", "RUNTIME"):
        for prefix in ("FANYIJI_VOICE_", "FANYIJI_BROWSER_", "FANYIJI_", "LOCAL_SUBTITLES_"):
            monkeypatch.delenv(prefix + name, raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.delattr(sys, "frozen", raising=False)


def load_entrypoint():
    spec = importlib.util.spec_from_file_location("voice_test_entrypoint", Path(__file__).parents[1] / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fresh_data_path_does_not_create_storage(tmp_path):
    assert APP_ID == "FanyiJiVoice"
    assert paths.data_root() == tmp_path / APP_ID
    assert not paths.data_root().exists()


@pytest.mark.parametrize("marker", ["settings.json", "models/weights.gguf", "specialized_model/corpus.sqlite3"])
def test_populated_legacy_storage_never_becomes_voice_storage(tmp_path, marker):
    original = tmp_path / "LocalJaZhSubtitles" / marker
    original.parent.mkdir(parents=True)
    original.write_bytes(b"original-user-data")
    assert paths.data_root() == tmp_path / APP_ID
    assert original.read_bytes() == b"original-user-data"
    assert not paths.data_root().exists()


@pytest.mark.parametrize("name,resolver", [("DATA", paths.data_root), ("MODELS", paths.models_root)])
def test_explicit_paths_follow_edition_generic_legacy_precedence(tmp_path, monkeypatch, name, resolver):
    legacy, generic, voice = (tmp_path / part for part in ("legacy", "generic", "voice"))
    monkeypatch.setenv("LOCAL_SUBTITLES_" + name, str(legacy))
    assert resolver() == legacy.resolve()
    monkeypatch.setenv("FANYIJI_" + name, str(generic))
    assert resolver() == generic.resolve()
    monkeypatch.setenv("FANYIJI_VOICE_" + name, str(voice))
    monkeypatch.setenv("FANYIJI_BROWSER_" + name, str(tmp_path / "browser"))
    assert resolver() == voice.resolve()
    assert not voice.exists()


def test_frozen_models_use_shared_external_default(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    (tmp_path / "LocalJaZhSubtitles/models").mkdir(parents=True)
    assert paths.models_root() == tmp_path / "FanyiJiModels/models"
    assert not paths.models_root().exists()


def test_migrated_reference_reuses_existing_weights_without_copy(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    weights = tmp_path / "existing/models"
    weights.mkdir(parents=True)
    (weights / "weights.gguf").write_bytes(b"unchanged-model")
    data = paths.data_root()
    data.mkdir()
    (data / "models-location.json").write_text(json.dumps({"directory": str(weights)}), encoding="utf-8")
    assert paths.models_root() == weights
    assert (weights / "weights.gguf").read_bytes() == b"unchanged-model"
    assert not (data / "models").exists()


@pytest.mark.parametrize("value", ["bad json", "[]", "{}", '{"directory":"relative/models"}',
                                 '{"directory":"C:/nonexistent-voice-test-models"}'])
def test_invalid_model_reference_keeps_external_default(tmp_path, monkeypatch, value):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    data = paths.data_root()
    data.mkdir()
    (data / "models-location.json").write_text(value, encoding="utf-8")
    assert paths.models_root() == tmp_path / "FanyiJiModels/models"


def test_development_models_only_use_an_existing_project_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "app_root", lambda: tmp_path / "source")
    assert paths.models_root() == tmp_path / "FanyiJiModels/models"
    (tmp_path / "source/models").mkdir(parents=True)
    assert paths.models_root() == tmp_path / "source/models"


def test_frozen_app_root_uses_executable_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "program/翻译姬·语音版.exe"))
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path / "unpacked"), raising=False)
    assert paths.app_root() == (tmp_path / "program").resolve()


@pytest.mark.parametrize("scenario", ["fresh", "legacy", "both", "legacy_override", "generic_override", "voice_override"])
def test_startup_fallback_matches_isolated_data_contract(tmp_path, monkeypatch, scenario):
    if scenario in ("legacy", "both"):
        (tmp_path / "LocalJaZhSubtitles/models").mkdir(parents=True)
    if scenario == "both":
        (tmp_path / APP_ID).mkdir()
    if scenario in ("legacy_override", "generic_override", "voice_override"):
        monkeypatch.setenv("LOCAL_SUBTITLES_DATA", str(tmp_path / "legacy-explicit"))
    if scenario in ("generic_override", "voice_override"):
        monkeypatch.setenv("FANYIJI_DATA", str(tmp_path / "generic-explicit"))
    if scenario == "voice_override":
        monkeypatch.setenv("FANYIJI_VOICE_DATA", str(tmp_path / "voice-explicit"))
    expected = paths.data_root()
    entrypoint = load_entrypoint()
    original_import = builtins.__import__

    def broken_package(name, *args, **kwargs):
        if name == "fanyiji.paths":
            raise ImportError("controlled package import failure")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_package)
    assert entrypoint._startup_data_root() == expected


@pytest.mark.parametrize("flag,module_name", [("--browser-helper", "fanyiji.browser_helper"),
                                             ("--web-translation-helper", "fanyiji.web_translation_helper")])
def test_native_helpers_bypass_main_and_preserve_arguments(monkeypatch, flag, module_name):
    entrypoint, received = load_entrypoint(), []
    helper = ModuleType(module_name)
    helper.main = lambda args: received.append(args) or 0
    monkeypatch.setitem(sys.modules, module_name, helper)
    monkeypatch.setattr(sys, "argv", ["run.py", flag, "--smoke-test"])
    assert entrypoint.launch() == 0
    assert received == [["--smoke-test"]]

