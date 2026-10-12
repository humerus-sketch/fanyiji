"""Windowed entry point with a readable startup failure report."""
import os
import sys
import traceback
from pathlib import Path


def _startup_data_root() -> Path:
    try:
        from fanyiji.paths import data_root
        return data_root()
    except Exception:
        # Startup logging must also work when the package itself cannot import.
        value = os.environ.get("FANYIJI_DATA") or os.environ.get("LOCAL_SUBTITLES_DATA")
        if value:
            return Path(value).resolve()
        base = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / ".local")))
        current, legacy = base / "FanyiJi", base / "LocalJaZhSubtitles"
        if current.exists() or current.is_symlink():
            return current
        if (legacy / "settings.json").is_file() or any(
            (legacy / name).is_dir() for name in ("models", "specialized_model")
        ):
            return legacy
        return current


def launch() -> int:
    try:
        if "--browser-helper" in sys.argv:
            from fanyiji.browser_helper import main as browser_main
            return browser_main([item for item in sys.argv[1:] if item != "--browser-helper"])
        from fanyiji.app import main
        return main()
    except Exception:
        message = traceback.format_exc()
        directory = _startup_data_root()
        try:
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "startup-error.log").write_text(message, encoding="utf-8")
        except OSError:
            pass
        if sys.stderr is not None:
            print(message, file=sys.stderr)
        if not {"--self-test", "--smoke-test"}.intersection(sys.argv):
            try:
                from PySide6.QtWidgets import QApplication, QMessageBox
                app = QApplication.instance() or QApplication(sys.argv[:1])
                QMessageBox.critical(None, "翻译姬 — 无法启动", "启动失败。诊断信息：\n" + str(directory / "startup-error.log"))
            except Exception:
                pass
        return 1

if __name__ == "__main__":
    raise SystemExit(launch())
