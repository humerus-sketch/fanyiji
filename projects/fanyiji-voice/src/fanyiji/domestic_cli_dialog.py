"""Explicit account checks and test translation; credentials remain in official CLIs."""
from __future__ import annotations

import threading

from PySide6.QtCore import QObject, QThread, Qt, QUrl, Signal, Slot
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QApplication, QComboBox, QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout

from .domestic_cli_catalog import BLOCKED_PROVIDERS, CLI_NAMES, CLI_PROVIDERS
from .i18n import tr


# A closed dialog cannot destroy a still-running QThread. Keep its worker alive
# until the vendor command observes cancellation and the thread finishes.
_RUNNING_WORKERS: dict[QThread, QObject] = {}
_WORKER_LIFECYCLES: dict[QThread, QObject] = {}


def _shutdown_workers() -> None:
    # aboutToQuit runs before QApplication's event loop stops. Worker completion
    # must not depend on another delivery to that stopped GUI event loop.
    import time
    pending = list(_RUNNING_WORKERS.items())
    for _, worker in pending:
        worker.cancellation_event.set()
    deadline = time.monotonic() + 8
    for thread, _ in pending:
        remaining = max(0, int((deadline - time.monotonic()) * 1000))
        thread.wait(remaining)


def _label(text: str = "") -> QLabel:
    label = QLabel(tr(text))
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setWordWrap(True)
    return label


class _AccountWorker(QObject):
    result = Signal(object)
    finished = Signal()

    def __init__(self, service, provider, model, action, text, generation, cancellation_event):
        super().__init__()
        self.service, self.provider, self.model, self.action = service, provider, model, action
        self.text, self.generation, self.cancellation_event = text, generation, cancellation_event

    @Slot()
    def run(self):
        result = None
        try:
            if not self.cancellation_event.is_set():
                if self.action == "test":
                    result = self.service.check_translation(self.provider, self.model, self.text,
                                                           cancellation_event=self.cancellation_event)
                else:
                    if self.action == "login":
                        self.service.login(self.provider, cancellation_event=self.cancellation_event)
                    if not self.cancellation_event.is_set():
                        result = self.service.probe(self.provider, cancellation_event=self.cancellation_event)
        except Exception as error:
            result = {"status": "error", "error": type(error).__name__}
            from .domestic_cli_engine import DomesticCliError
            if isinstance(error, DomesticCliError):
                result["message"] = str(error)
        finally:
            if result is not None and not self.cancellation_event.is_set():
                self.result.emit({"generation": self.generation, "provider": self.provider,
                                  "action": self.action, "result": result})
            self.finished.emit()


class _WorkerLifecycle(QObject):
    def __init__(self, thread: QThread):
        super().__init__(QApplication.instance())
        self.thread = thread

    @Slot()
    def release(self):
        # A plain Python lambda attached to QThread.finished can run on the
        # worker thread, before its native teardown completes. Marshal cleanup
        # to this GUI-owned QObject and join before dropping Python ownership.
        if not self.thread.wait(5000):
            return
        _RUNNING_WORKERS.pop(self.thread, None)
        _WORKER_LIFECYCLES.pop(self.thread, None)
        self.thread.deleteLater()
        self.deleteLater()


class DomesticCliManagerDialog(QDialog):
    def __init__(self, parent=None, *, provider="minimax", model="", service=None):
        super().__init__(parent)
        if service is None:
            from .domestic_cli_service import DomesticCliService
            service = DomesticCliService()
        self.service = service
        application = QApplication.instance()
        if application is not None and not application.property("fanyiji_domestic_shutdown_connected"):
            application.aboutToQuit.connect(_shutdown_workers)
            application.setProperty("fanyiji_domestic_shutdown_connected", True)
        self._thread = self._worker = None
        self._cancel = threading.Event()
        self._generation = 0
        self._closing = False
        self.last_probe = None
        self.result_choice = None
        self.setWindowTitle(tr("国内 CLI 账号与模型"))
        self.resize(660, 450)
        layout = QVBoxLayout(self)
        layout.addWidget(_label("登录由官方 CLI 或官方网页完成，程序不保存密码，不提供 API Key 配置。检查和试译仅在点击按钮后执行。"))
        row = QHBoxLayout()
        row.addWidget(_label("国内 CLI 服务"))
        self.provider = QComboBox()
        for identifier in CLI_PROVIDERS:
            self.provider.addItem(tr(CLI_NAMES[identifier]), identifier)
            if identifier in BLOCKED_PROVIDERS:
                item = self.provider.model().item(self.provider.count() - 1)
                item.setEnabled(False)
                item.setToolTip(tr(BLOCKED_PROVIDERS[identifier]))
        self.provider.setCurrentIndex(max(0, self.provider.findData(provider)))
        row.addWidget(self.provider, 1)
        layout.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(_label("模型（留空使用账号默认）"))
        self.model = QComboBox()
        self.model.setEditable(True)
        self.model.lineEdit().setMaxLength(128)
        self._set_models([], model)
        row.addWidget(self.model, 1)
        layout.addLayout(row)
        self.status = _label()
        layout.addWidget(self.status)
        actions = QHBoxLayout()
        self.check_button = QPushButton(tr("检查账号状态"))
        self.login_button = QPushButton(tr("官方账号登录"))
        self.download_button = QPushButton(tr("官方下载与说明"))
        self.check_button.clicked.connect(lambda: self._start("probe"))
        self.login_button.clicked.connect(lambda: self._start("login"))
        self.download_button.clicked.connect(self._open_official)
        for button in (self.check_button, self.login_button, self.download_button):
            actions.addWidget(button)
        layout.addLayout(actions)
        layout.addWidget(_label("日语试译原文"))
        self.test_text = QLineEdit("今日はゲームの音量を少し下げます。")
        self.test_text.setMaxLength(2000)
        layout.addWidget(self.test_text)
        self.test_button = QPushButton(tr("试译（发送上方文字）"))
        self.test_button.clicked.connect(lambda: self._start("test"))
        layout.addWidget(self.test_button)
        self.test_result = _label()
        self.test_result.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.test_result)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        self.use_button = buttons.addButton(tr("使用选中设置"), QDialogButtonBox.ButtonRole.AcceptRole)
        self.use_button.setEnabled(False)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.provider.currentIndexChanged.connect(self._provider_changed)
        self._provider_changed()

    def _selected_model(self) -> str:
        index = self.model.currentIndex()
        if index >= 0 and self.model.currentText() == self.model.itemText(index):
            return self.model.itemData(index) or ""
        return self.model.currentText().strip()

    def _set_models(self, models, selected):
        self.model.clear()
        self.model.addItem(tr("账号默认"), "")
        if isinstance(models, list):
            for model in models:
                if not isinstance(model, dict):
                    continue
                identifier = model.get("id")
                if not isinstance(identifier, str) or not identifier or len(identifier) > 128:
                    continue
                name = model.get("name")
                self.model.addItem(name if isinstance(name, str) and name else identifier, identifier)
        index = self.model.findData(selected)
        if selected and index < 0:
            self.model.addItem(selected, selected)
            index = self.model.count() - 1
        self.model.setCurrentIndex(max(0, index))

    def _provider_changed(self):
        self.last_probe = None
        self.test_result.clear()
        self.status.setText(tr(BLOCKED_PROVIDERS[self.provider.currentData()]) if self.provider.currentData() in BLOCKED_PROVIDERS
                            else tr("尚未检查账号状态；启动时按实际 CLI 登录状态核验。"))
        self._set_busy(False)

    def _set_busy(self, busy):
        blocked = self.provider.currentData() in BLOCKED_PROVIDERS
        self.provider.setEnabled(not busy)
        self.model.setEnabled(not busy and not blocked)
        for button in (self.check_button, self.login_button, self.test_button):
            button.setEnabled(not busy and not blocked)
        self.download_button.setEnabled(not busy)
        capabilities = self.last_probe.get("capabilities", {}) if isinstance(self.last_probe, dict) else {}
        ready = bool(isinstance(self.last_probe, dict) and self.last_probe.get("status") == "ready"
                     and isinstance(capabilities, dict) and capabilities.get("account_auth") is True
                     and capabilities.get("tool_free") is True)
        self.use_button.setEnabled(not busy and not blocked and ready)

    def _start(self, action):
        if self._closing or self._thread is not None or self.provider.currentData() in BLOCKED_PROVIDERS:
            return
        text = self.test_text.text().strip()
        if action == "test" and not text:
            return
        self._generation += 1
        self._cancel = threading.Event()
        thread = QThread()
        lifecycle = _WorkerLifecycle(thread)
        worker = _AccountWorker(self.service, self.provider.currentData(), self._selected_model(), action, text,
                                self._generation, self._cancel)
        self._thread, self._worker = thread, worker
        _RUNNING_WORKERS[thread] = worker
        _WORKER_LIFECYCLES[thread] = lifecycle
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.result.connect(self._on_result)
        worker.finished.connect(thread.quit, Qt.ConnectionType.DirectConnection)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(self._finished)
        thread.finished.connect(lifecycle.release, Qt.ConnectionType.QueuedConnection)
        self._set_busy(True)
        self.status.setText(tr("正在执行官方 CLI 操作…"))
        if action == "login" and self.provider.currentData() == "codebuddy":
            self.status.setText(tr("在官方 CLI 窗口完成账号登录后关闭该窗口，程序将重新检查登录状态。"))
        thread.start()

    @Slot(object)
    def _on_result(self, payload):
        if self._closing or self._cancel.is_set() or payload.get("generation") != self._generation or payload.get("provider") != self.provider.currentData():
            return
        result = payload.get("result")
        if not isinstance(result, dict):
            self.status.setText(tr("CLI 未返回有效检查结果"))
            return
        if result.get("error"):
            self.status.setText(tr("CLI 操作失败：{0}", str(result.get("message", result["error"]))[:2000]))
            return
        if payload["action"] == "test":
            self.test_result.setText(tr("试译结果：{0}\n耗时：{1} 秒 · 模型：{2}",
                str(result.get("translation", ""))[:4000], str(result.get("seconds", ""))[:32], str(result.get("model", ""))[:128]))
            self.status.setText(tr("试译完成；实际字幕启动仍会核验账号和模型。"))
        else:
            self.last_probe = result
            self.status.setText(tr("检查结果：{0}", str(result.get("message", result.get("status", "")))[:2000]))
            self._set_models(result.get("models", []), self._selected_model())

    def _finished(self):
        self._thread = self._worker = None
        if not self._closing:
            self._set_busy(False)

    def _open_official(self):
        try:
            QDesktopServices.openUrl(QUrl(self.service.official_url(self.provider.currentData())))
        except Exception:
            self.status.setText(tr("无法打开官方说明页面"))

    def accept(self):
        if not self.use_button.isEnabled():
            return
        self.result_choice = (self.provider.currentData(), self._selected_model())
        super().accept()

    def reject(self):
        self._closing = True
        self._cancel.set()
        super().reject()

    def closeEvent(self, event):
        self._closing = True
        self._cancel.set()
        super().closeEvent(event)


def wire_domestic_cli_manager(settings_dialog, service=None):
    if not hasattr(settings_dialog, "domestic_cli_requested"):
        return
    def open_manager(provider, model):
        dialog = DomesticCliManagerDialog(settings_dialog, provider=provider, model=model, service=service)
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.result_choice is not None:
            settings_dialog.set_domestic_cli_choice(*dialog.result_choice, status=dialog.last_probe)
    settings_dialog.domestic_cli_requested.connect(open_manager)
