"""Cancellable account-model discovery; never starts a generation or audio capture."""
from __future__ import annotations

import threading

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QDialog


class CodexModelDiscovery(QObject):
    models_ready = Signal(object)
    failed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._lock = threading.Lock()
        self._closed = False
        self._busy = False
        self._engine = None

    @property
    def is_closed(self) -> bool:
        with self._lock:
            return self._closed

    def request(self) -> None:
        with self._lock:
            if self._closed or self._busy:
                return
            self._busy = True
        threading.Thread(target=self._discover, name="codex-model-list", daemon=True).start()

    def _discover(self) -> None:
        engine = None
        models = None
        error = ""
        try:
            from .codex_engine import CodexTranslationEngine
            engine = CodexTranslationEngine()
            with self._lock:
                if self._closed:
                    return
                self._engine = engine
            engine.preload()
            models = engine.backend_info["available_models"]
        except Exception as exc:
            error = str(exc)
        finally:
            try:
                if engine is not None:
                    engine.close()
            except Exception:
                models = None
                error = error or "模型查询清理失败，请重试"
            finally:
                with self._lock:
                    self._engine = None
                    self._busy = False
            # Cleanup finishes before enabling another refresh in the UI.
            if not self.is_closed:
                try:
                    if error:
                        self.failed.emit(error)
                    elif models is not None:
                        self.models_ready.emit(models)
                except RuntimeError:
                    # The parent dialog may already have been destroyed.
                    pass

    def close(self) -> None:
        with self._lock:
            self._closed = True
            engine = self._engine
        if engine is not None:
            def stop():
                try:
                    engine.close()
                except Exception:
                    # A closed dialog cannot display cleanup errors; the worker
                    # performs its own final cleanup without reopening the UI.
                    pass
            threading.Thread(target=stop, name="codex-model-list-stop", daemon=True).start()


def wire_codex_picker(dialog) -> None:
    """Use the same nonblocking picker for first-run and later settings dialogs."""
    if not isinstance(dialog, QDialog):
        return
    discovery = CodexModelDiscovery(dialog)
    dialog.codex_discovery = discovery

    def refresh():
        if discovery.is_closed:
            return
        dialog.set_codex_models_loading(True)
        discovery.request()

    def ready(models):
        if not discovery.is_closed:
            dialog.set_codex_models(models)

    def failed(message):
        if not discovery.is_closed:
            dialog.set_codex_models([], error=message)

    dialog.codex_models_requested.connect(refresh)
    discovery.models_ready.connect(ready)
    discovery.failed.connect(failed)
    dialog.finished.connect(lambda _result: discovery.close())

    def source_changed():
        if dialog.translation_provider.currentData() in ("codex", "hybrid") and not dialog._codex_models:
            refresh()

    dialog.translation_provider.currentIndexChanged.connect(source_changed)
    source_changed()
