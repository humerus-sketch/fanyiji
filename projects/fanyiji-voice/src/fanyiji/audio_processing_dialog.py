"""Selected audio source and offline speech enhancement controls."""
from __future__ import annotations

import threading

from PySide6.QtCore import QObject, QThread, Qt, QUrl, Signal, Slot
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                              QHBoxLayout, QLabel, QLineEdit, QProgressBar,
                              QPushButton, QVBoxLayout, QWidget)

from .browser_service import room_url
from .i18n import get_language, tr
from .identity import APP_NAME
from .live_audio_protocol import normalize_reference_video_url
from .paths import models_root


def _label(text=""):
    result = QLabel(tr(text))
    result.setTextFormat(Qt.TextFormat.PlainText)
    result.setWordWrap(True)
    return result


class _ProcessingModelWorker(QObject):
    progress = Signal(int)
    result = Signal(object)
    finished = Signal()

    def __init__(self, directory, mode, action, cancel):
        super().__init__()
        self.directory, self.mode, self.action, self.cancel = directory, mode, action, cancel

    def _progress(self, _phase, done, total):
        if isinstance(done, (int, float)) and isinstance(total, (int, float)) and total > 0:
            self.progress.emit(max(0, min(100, int(done * 100 / total))))

    @Slot()
    def run(self):
        try:
            from .audio_processing_models import download_processing_model, verify_processing_model
            if self.action == "download":
                download_processing_model(self.directory, self.mode, cancel=self.cancel, progress=self._progress)
            verified = verify_processing_model(self.directory, self.mode, cancel=self.cancel)
            self.result.emit({"mode": self.mode, "ready": verified.ready, "error": ""})
        except Exception as error:
            self.result.emit({"mode": self.mode, "ready": False, "error": type(error).__name__,
                              "cancelled": self.cancel.is_set()})
        finally:
            self.finished.emit()


class AudioProcessingDialog(QDialog):
    def __init__(self, service, parent=None, *, browser_service=None,
                 mode_getter=None, apply_mode=None, models_dir=None, model_only=False):
        super().__init__(parent)
        self._model_only = model_only is True
        self.service = service
        self.browser_service = None if self._model_only else browser_service
        self.mode_getter = (lambda: "gtcrn") if self._model_only else mode_getter or (lambda: "gtcrn")
        self.apply_mode = None if self._model_only else apply_mode
        self.models_dir = models_root() if models_dir is None else models_dir
        self._thread = self._worker = None
        self._cancel = threading.Event()
        self._close_requested = False
        self._verified = {}
        self.setWindowTitle(APP_NAME + " · " + tr("管理音频处理模型" if self._model_only else "音源与语音处理"))
        self.setMinimumWidth(620)
        layout = QVBoxLayout(self)
        self.source_controls = QWidget()
        source_layout = QVBoxLayout(self.source_controls)
        source_layout.setContentsMargins(0, 0, 0, 0)
        source_layout.addWidget(_label("指定来源只隔离其他软件的声音；主播、游戏对白和音乐仍属于同一份混音。"))
        self.source_status = _label()
        source_layout.addWidget(self.source_status)
        self.source_enabled = QCheckBox(tr("仅采集内置浏览器当前播放来源"))
        self.source_enabled.toggled.connect(self._set_source_enabled)
        source_layout.addWidget(self.source_enabled)
        room_row = QHBoxLayout()
        room_row.addWidget(_label("直播间"))
        self.room = QLineEdit(getattr(self.browser_service, "room", ""))
        self.room.setPlaceholderText(tr("直播间号或 HTTPS 直播间地址"))
        room_row.addWidget(self.room, 1)
        self.room_button = QPushButton(tr("打开直播间"))
        self.room_button.clicked.connect(self._open_room)
        self.room_button.setEnabled(self.browser_service is not None)
        room_row.addWidget(self.room_button)
        source_layout.addLayout(room_row)
        video_row = QHBoxLayout()
        video_row.addWidget(_label("B站视频链接"))
        self.video_url = QLineEdit()
        self.video_url.setPlaceholderText(tr("粘贴 B站 HTTPS 普通视频地址"))
        video_row.addWidget(self.video_url, 1)
        self.video_button = QPushButton(tr("打开视频"))
        self.video_button.clicked.connect(self._open_video)
        self.video_button.setEnabled(self.browser_service is not None)
        video_row.addWidget(self.video_button)
        source_layout.addLayout(video_row)
        source_layout.addWidget(_label("普通视频也可生成字幕，来源不会被当作直播间。启动程序不会自动打开浏览器。"))
        self.source_controls.setVisible(not self._model_only)
        self.source_controls.setEnabled(not self._model_only)
        layout.addWidget(self.source_controls)
        mode_row = QHBoxLayout()
        mode_row.addWidget(_label("语音处理模式"))
        self.mode = QComboBox()
        for mode, name in (("off", "关闭语音处理"), ("gtcrn", "GTCRN · 实时语音增强（推荐）"),
                           ("uvr", "UVR · 人声与伴奏分离（较慢）")):
            if not self._model_only or mode == "gtcrn":
                self.mode.addItem(tr(name), mode)
        self.mode.setCurrentIndex(max(0, self.mode.findData(self.mode_getter())))
        self.mode.currentIndexChanged.connect(self._mode_changed)
        mode_row.addWidget(self.mode, 1)
        self.save_button = QPushButton(tr("应用语音处理模式"))
        self.save_button.clicked.connect(self._apply_mode)
        self.save_button.setEnabled(callable(self.apply_mode))
        self.save_button.setVisible(not self._model_only)
        mode_row.addWidget(self.save_button)
        layout.addLayout(mode_row)
        self.active_mode = _label()
        self.active_mode.setVisible(not self._model_only)
        layout.addWidget(self.active_mode)
        self.model_status = _label()
        layout.addWidget(self.model_status)
        model_row = QHBoxLayout()
        self.download_button = QPushButton(tr("下载处理模型"))
        self.download_button.clicked.connect(lambda: self._start_task("download"))
        model_row.addWidget(self.download_button)
        self.verify_button = QPushButton(tr("校验本地处理模型"))
        self.verify_button.clicked.connect(lambda: self._start_task("verify"))
        model_row.addWidget(self.verify_button)
        self.model_link_button = QPushButton(tr("官方模型链接"))
        self.model_link_button.clicked.connect(self._open_model_link)
        model_row.addWidget(self.model_link_button)
        layout.addLayout(model_row)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.hide()
        layout.addWidget(self.progress)
        if self._model_only:
            layout.addWidget(_label("GTCRN 主要减少噪声和部分背景声，不能锁定主播、分离游戏角色讲话或保证去除唱歌。Chrome 识别处理后声音的链路仍属实验，以实际状态为准。处理模型单独下载，不包含在程序包中。"))
        else:
            layout.addWidget(_label("模型单独下载到本机，不包含在程序包中。缺失或处理失败时回退原音，并显示状态。"))
            layout.addWidget(_label("语音增强主要减少噪声；伴奏分离可能改善背景音乐，但仍会保留唱歌和游戏角色人声，不能只提取主播。"))
        self.status = _label()
        layout.addWidget(self.status)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self.buttons.button(QDialogButtonBox.StandardButton.Close).setText(tr("关闭"))
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        if not self._model_only:
            self.service.snapshot.connect(self.on_snapshot)
            self.on_snapshot(self.service.snapshot_value())
        self._mode_changed()

    @property
    def is_busy(self):
        return self._thread is not None

    def showEvent(self, event):
        if not self.is_busy:
            self._close_requested = False
            self.mode.setCurrentIndex(max(0, self.mode.findData(self.mode_getter())))
            self._mode_changed()
        super().showEvent(event)

    def on_snapshot(self, value):
        if self._model_only:
            return
        if not isinstance(value, dict):
            self.source_status.setText(tr("指定音源状态不可用"))
            return
        self.source_enabled.blockSignals(True)
        self.source_enabled.setChecked(value.get("source_enabled") is True)
        self.source_enabled.blockSignals(False)
        selected = value.get("source_url", "")
        if not value.get("source_enabled"):
            status = tr("电脑声音采集")
        elif value.get("source_ready"):
            status = tr("指定视频音源已连接" if value.get("source_kind") == "video" else "指定直播音源已连接")
        else:
            status = tr("等待指定播放来源")
        self.source_status.setText(status + ("\n" + selected if isinstance(selected, str) and selected.startswith("https://") else ""))
        if value.get("state") == "error":
            self.status.setText(tr("指定音源异常，请重新保存采集设置或刷新播放来源"))

    def _set_source_enabled(self, enabled):
        if self._model_only:
            return
        try:
            self.service.set_source_enabled(enabled)
            self.status.setText(tr("采集设置已保存"))
        except (ValueError, OSError, RuntimeError):
            self.status.setText(tr("采集设置保存失败，已有设置保留"))
            self.on_snapshot(self.service.snapshot_value())

    def _open_room(self):
        if self._model_only:
            return
        try:
            url = room_url(self.room.text())
            self.browser_service.open(url, get_language())
        except (ValueError, OSError, RuntimeError):
            self.status.setText(tr("直播间格式无效或浏览器无法打开"))

    def _open_video(self):
        if self._model_only:
            return
        try:
            url = normalize_reference_video_url(self.video_url.text())
            self.browser_service.open_reference_video(url, get_language())
        except (ValueError, OSError, RuntimeError):
            self.status.setText(tr("视频地址无效或浏览器无法打开"))

    def _mode_changed(self, *_):
        mode = self.mode.currentData()
        self.active_mode.setText(tr("当前保存模式：{0}", self.mode.itemText(self.mode.findData(self.mode_getter()))))
        ready = self._verified.get(mode)
        self.model_status.setText(tr("处理已关闭" if mode == "off" else "本地处理模型校验通过" if ready is True
                                    else "本地处理模型缺失或校验未通过" if ready is False else "点击校验以检查本地处理模型"))
        busy = self.is_busy
        self.mode.setEnabled(not busy)
        self.save_button.setEnabled(not busy and callable(self.apply_mode))
        self.download_button.setEnabled(not busy and mode != "off")
        self.verify_button.setEnabled(not busy and mode != "off")
        self.model_link_button.setEnabled(mode != "off")

    def _apply_mode(self):
        if self._model_only or not callable(self.apply_mode):
            return
        try:
            if self.apply_mode(self.mode.currentData()) is False:
                self.status.setText(tr("语音处理模式保存失败，已有设置保留"))
            else:
                self.status.setText(tr("语音处理模式已保存，字幕重新加载后生效"))
                self._mode_changed()
        except (ValueError, OSError, RuntimeError):
            self.status.setText(tr("语音处理模式保存失败，已有设置保留"))

    def _open_model_link(self):
        if self.mode.currentData() == "off":
            return
        from .audio_processing_models import processing_model_entry
        entry = processing_model_entry(self.mode.currentData())
        if entry is not None:
            url = entry.info_url
            if isinstance(url, str) and url.startswith("https://"):
                QDesktopServices.openUrl(QUrl(url))

    def _start_task(self, action):
        if self.is_busy or self.mode.currentData() == "off" or self._close_requested:
            return
        self._cancel.clear()
        thread = QThread(self)
        worker = _ProcessingModelWorker(self.models_dir, self.mode.currentData(), action, self._cancel)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self.progress.setValue)
        worker.result.connect(self._on_result)
        worker.finished.connect(worker.deleteLater)
        worker.finished.connect(thread.quit, Qt.ConnectionType.DirectConnection)
        thread.finished.connect(self._task_finished)
        thread.finished.connect(thread.deleteLater)
        self._thread, self._worker = thread, worker
        self.progress.setValue(0)
        self.progress.show()
        self.status.setText(tr("正在下载处理模型…" if action == "download" else "正在校验本地处理模型…"))
        self._mode_changed()
        thread.start()

    def _on_result(self, result):
        self._verified[result["mode"]] = result.get("ready") is True
        self.status.setText(tr("操作已取消" if result.get("cancelled") else "本地处理模型校验通过" if result.get("ready")
                               else "处理模型操作失败，请检查本地文件或网络")
                           + (" · " + result["error"] if result.get("error") and not result.get("cancelled") else ""))

    def _task_finished(self):
        self._thread = self._worker = None
        self.progress.hide()
        self._mode_changed()
        if self._close_requested:
            super().reject()

    def reject(self):
        if self.is_busy:
            self._close_requested = True
            self._cancel.set()
            self.status.setText(tr("正在取消处理模型操作…"))
        else:
            super().reject()

    def closeEvent(self, event):
        if self.is_busy:
            event.ignore()
            self.reject()
        else:
            super().closeEvent(event)
