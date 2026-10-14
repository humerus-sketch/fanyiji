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
        value = (os.environ.get("FANYIJI_VOICE_DATA") or os.environ.get("FANYIJI_DATA")
                 or os.environ.get("LOCAL_SUBTITLES_DATA"))
        if value:
            return Path(value).resolve()
        base = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / ".local")))
        return base / "FanyiJiVoice"


def launch() -> int:
    try:
        if "--web-translation-helper" in sys.argv:
            from fanyiji.web_translation_helper import main as web_translation_main
            return web_translation_main([item for item in sys.argv[1:] if item != "--web-translation-helper"])
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
                QMessageBox.critical(None, "翻译姬·语音版 — 无法启动", "启动失败。诊断信息：\n" + str(directory / "startup-error.log"))
            except Exception:
                pass
        return 1

if __name__ == "__main__":
    raise SystemExit(launch())
