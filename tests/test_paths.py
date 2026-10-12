"""Identity migration reuses user data without copying models or credentials."""
import builtins
import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import pytest

from fanyiji import paths


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    for name in ("DATA", "MODELS", "RUNTIME"):
        for prefix in ("FANYIJI_", "LOCAL_SUBTITLES_"):
            monkeypatch.delenv(prefix + name, raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.delattr(sys, "frozen", raising=False)


def load_entrypoint():
    spec = importlib.util.spec_from_file_location("fanyiji_test_entrypoint", Path(__file__).parents[1] / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fresh_install_uses_new_name_without_creating_storage(tmp_path):
    assert paths.data_root() == tmp_path / "FanyiJi"
    assert not (tmp_path / "FanyiJi").exists()
    assert not (tmp_path / "LocalJaZhSubtitles").exists()


@pytest.mark.parametrize("marker", ["settings.json", "models/weights.gguf", "specialized_model/corpus.sqlite3"])
def test_existing_legacy_data_is_reused_in_place(tmp_path, marker):
    legacy = tmp_path / "LocalJaZhSubtitles"
    file = legacy / marker
    file.parent.mkdir(parents=True)
    file.write_bytes(b"user-data-remains-here")
    assert paths.data_root() == legacy
    assert file.read_bytes() == b"user-data-remains-here"
    assert not (tmp_path / "FanyiJi").exists()


def test_empty_legacy_directory_does_not_determine_new_identity(tmp_path):
    (tmp_path / "LocalJaZhSubtitles").mkdir()
    assert paths.data_root() == tmp_path / "FanyiJi"


def test_new_directory_has_stable_precedence_over_populated_legacy(tmp_path):
    legacy = tmp_path / "LocalJaZhSubtitles"
    legacy.mkdir()
    (legacy / "settings.json").write_text("old settings", encoding="utf-8")
    current = tmp_path / "FanyiJi"
    current.mkdir()
    assert paths.data_root() == current
    (legacy / "models").mkdir()
    assert paths.data_root() == current
    assert (legacy / "settings.json").read_text(encoding="utf-8") == "old settings"


@pytest.mark.parametrize("name,resolver", [("DATA", paths.data_root), ("MODELS", paths.models_root)])
def test_new_override_wins_without_falling_back(tmp_path, monkeypatch, name, resolver):
    old = tmp_path / "existing-override"
    old.mkdir()
    new = tmp_path / "not-created-override"
    monkeypatch.setenv("LOCAL_SUBTITLES_" + name, str(old))
    assert resolver() == old.resolve()
    monkeypatch.setenv("FANYIJI_" + name, str(new))
    assert resolver() == new.resolve()
    assert not new.exists()


def test_frozen_models_follow_selected_data_storage_and_override(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    legacy = tmp_path / "LocalJaZhSubtitles"
    (legacy / "models").mkdir(parents=True)
    assert paths.models_root() == legacy / "models"
    (tmp_path / "FanyiJi").mkdir()
    assert paths.models_root() == tmp_path / "FanyiJi" / "models"
    override = tmp_path / "external-models"
    monkeypatch.setenv("LOCAL_SUBTITLES_MODELS", str(override))
    assert paths.models_root() == override.resolve()


def test_frozen_app_root_is_executable_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "program" / "翻译姬.exe"))
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path / "unpacked"), raising=False)
    assert paths.app_root() == (tmp_path / "program").resolve()


def test_development_models_remain_in_project(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "app_root", lambda: tmp_path / "source")
    assert paths.models_root() == tmp_path / "source" / "models"


def test_startup_failure_uses_application_data_resolver(tmp_path, monkeypatch):
    entrypoint = load_entrypoint()
    target = tmp_path / "resolved-data"
    monkeypatch.setattr(paths, "data_root", lambda: target)
    app = ModuleType("fanyiji.app")

    def fail():
        raise RuntimeError("controlled startup failure")

    app.main = fail
    monkeypatch.setitem(sys.modules, "fanyiji.app", app)
    monkeypatch.setattr(sys, "argv", ["run.py", "--smoke-test"])
    assert entrypoint.launch() == 1
    assert "controlled startup failure" in (target / "startup-error.log").read_text(encoding="utf-8")
    assert not (tmp_path / "FanyiJi").exists()


@pytest.mark.parametrize("scenario", ["fresh", "legacy", "both", "legacy_override", "new_override"])
def test_entrypoint_fallback_matches_normal_path_selection(tmp_path, monkeypatch, scenario):
    legacy = tmp_path / "LocalJaZhSubtitles"
    if scenario in ("legacy", "both"):
        (legacy / "models").mkdir(parents=True)
    if scenario == "both":
        (tmp_path / "FanyiJi").mkdir()
    if scenario in ("legacy_override", "new_override"):
        monkeypatch.setenv("LOCAL_SUBTITLES_DATA", str(tmp_path / "old-explicit"))
    if scenario == "new_override":
        monkeypatch.setenv("FANYIJI_DATA", str(tmp_path / "new-explicit"))
    expected = paths.data_root()
    entrypoint = load_entrypoint()
    original_import = builtins.__import__

    def broken_package(name, *args, **kwargs):
        if name == "fanyiji.paths":
            raise ImportError("simulated package startup failure")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_package)
    assert entrypoint._startup_data_root() == expected
