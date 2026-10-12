"""Explicit model selection/download UI; all disk hashing and HTTP run off the GUI thread."""
from __future__ import annotations

from pathlib import Path
import json
import logging
import threading
import time

from PySide6.QtCore import QObject, QThread, QUrl, Signal, Slot
from PySide6.QtGui import QCloseEvent, QDesktopServices, QShowEvent
from PySide6.QtWidgets import (
    QApplication, QButtonGroup, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QMenu, QMessageBox,
    QProgressBar, QPushButton, QRadioButton, QScrollArea, QTabWidget, QVBoxLayout, QWidget,
)
from PySide6.QtCore import Qt

from .models_download import (
    DownloadCancelled, DownloadError, ModelCatalog, ModelDownloader, ModelEntry, VerificationResult,
)
from .asr_catalog import AsrCatalog, profile_for_entry, vad_dependencies
from .model_import import ModelImporter, ImportInspection, ImportOption
from .model_registry import ModelRegistry, validate_display_name
from .i18n import tr
from .ui import loaded_model_display, loaded_models_summary

LOGGER = logging.getLogger(__name__)


def _label(text: str) -> QLabel:
    result = QLabel(tr(text))
    result.setTextFormat(Qt.TextFormat.PlainText)
    result.setWordWrap(True)
    return result


def _size(size: int) -> str:
    return f"{size / (1024 ** 3):.2f} GiB" if size >= 1024 ** 3 else f"{size / (1024 ** 2):.1f} MiB"


def _status_text(text: str) -> str:
    if text.startswith("校验失败："):
        return tr("校验失败：{0}", text[len("校验失败："):])
    return tr(text)


class ModelImportDialog(QDialog):
    """Second step after background detection, with an explicit final choice."""

    def __init__(self, inspection: ImportInspection, source: Path, parent=None, *, existing_names=()):
        super().__init__(parent)
        self.inspection = inspection
        self._existing_names = {name.casefold() for name in existing_names}
        self.setWindowTitle(tr("导入本地模型"))
        self.resize(620, 340)
        layout = QVBoxLayout(self)
        layout.addWidget(_label(tr("模型来源：{0}", str(source))))
        layout.addWidget(_label("请选择用途与兼容模板，并填写显示名称。导入后需要手动启用。"))
        self.template_combo = QComboBox()
        self.template_combo.setObjectName("importTemplateCombo")
        for option in inspection.options:
            role = tr("识别模型" if option.role == "asr" else "翻译模型")
            self.template_combo.addItem(tr("{0} · {1}", role, option.template.name), option)
        layout.addWidget(self.template_combo)
        self.verification_label = _label("")
        layout.addWidget(self.verification_label)
        layout.addWidget(_label("模型显示名称（1–80 字符）"))
        self.name_edit = QLineEdit()
        self.name_edit.setObjectName("importDisplayName")
        self.name_edit.setMaxLength(80)
        layout.addWidget(self.name_edit)
        self.error_label = _label("")
        layout.addWidget(self.error_label)
        buttons = QDialogButtonBox()
        self.confirm_button = buttons.addButton(tr("校验并导入"), QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.addButton(tr("取消"), QDialogButtonBox.ButtonRole.RejectRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.template_combo.currentIndexChanged.connect(self._template_changed)
        self.name_edit.textChanged.connect(self._validate)
        self._template_changed()

    @property
    def selected_option(self) -> ImportOption:
        return self.template_combo.currentData()

    @property
    def display_name(self) -> str:
        return validate_display_name(self.name_edit.text())

    def _template_changed(self, *_):
        option = self.selected_option
        name = option.suggested_name
        number = 2
        while name.casefold() in self._existing_names:
            suffix = " (" + str(number) + ")"
            name = option.suggested_name[:80 - len(suffix)] + suffix
            number += 1
        self.name_edit.setText(name)
        self.verification_label.setText(tr(
            "已识别预设模型版本；使用目录固定 SHA-256 校验，原模型包保留。"
            if option.template.verification == "pinned_catalog_sha256" else
            "自定义 GGUF：结构与本地 SHA-256 校验；官方来源、运行兼容性与翻译效果未经验证。"))
        self._validate()

    def _validate(self, *_):
        try:
            if self.display_name.casefold() in self._existing_names:
                raise DownloadError("本地模型名称或 ID 已存在，请使用其他名称。")
            self.error_label.clear()
            self.confirm_button.setEnabled(True)
        except DownloadError as exc:
            self.error_label.setText(tr(str(exc)))
            self.confirm_button.setEnabled(False)


class _Worker(QObject):
    progress = Signal(str, str, int)
    result = Signal(object)
    finished = Signal()

    def __init__(self, downloader: ModelDownloader, catalog: ModelCatalog,
                 action: str, model_id: str | None, cancel: threading.Event,
                 restart: bool = False, *, asr_catalog: AsrCatalog | None = None,
                 asr_model_id: str | None = None, entries: tuple[ModelEntry, ...] | None = None,
                 speech_only: bool = True, translation_required: bool = True,
                 importer: ModelImporter | None = None, import_source: Path | None = None,
                 registry: ModelRegistry | None = None, import_option: ImportOption | None = None,
                 display_name: str | None = None):
        super().__init__()
        self.downloader, self.catalog = downloader, catalog
        self.action, self.model_id = action, model_id
        self.cancel, self.restart = cancel, restart
        self.asr_catalog, self.asr_model_id = asr_catalog, asr_model_id
        self.entries = entries if entries is not None else catalog.entries
        self.speech_only = speech_only
        self.translation_required = translation_required
        self.importer, self.import_source = importer, import_source
        self.registry, self.import_option, self.display_name = registry, import_option, display_name

    def _progress(self, model_id: str):
        last = ["", -1, 0.0]
        def update(phase: str, done: int, total: int) -> None:
            # Signal(int) is 32-bit: send a percentage, never a GGUF byte count.
            percent = min(100, int(done * 100 / total))
            now = time.monotonic()
            if (phase != last[0] or done == total
                    or (percent != last[1] and now - last[2] >= 0.05)):
                self.progress.emit(model_id, phase, percent)
                last[:] = [phase, percent, now]
        return update

    @Slot()
    def run(self) -> None:
        states: dict[str, VerificationResult] = {}
        try:
            if self.action in ("verify", "verify_full"):
                for entry in self.entries:
                    self.progress.emit(entry.id, "verifying", 0)
                    if self.action == "verify_full":
                        states[entry.id] = self.downloader.verify_entry(
                            entry, self._progress(entry.id), self.cancel, force=True)
                    else:
                        states[entry.id] = self.downloader.verify_entry(
                            entry, self._progress(entry.id), self.cancel)
                    if (states[entry.id].ready and entry.verification == "pinned_catalog_sha256"
                            and self.importer is not None and self.importer.has_provenance(entry)):
                        states[entry.id] = VerificationResult(True, "ready", "本地导入 · 目录固定 SHA-256 校验通过")
            elif self.action == "inspect":
                if self.importer is None or self.import_source is None or self.registry is None:
                    raise DownloadError("未选择要导入的本地模型包。")
                templates = (*self.registry.catalog.entries, *self.registry.asr_catalog.entries)
                inspection = self.importer.inspect_source(self.import_source, templates,
                                                         self._progress("new-import"), self.cancel)
                self.result.emit({"states": states, "inspection": inspection, "source": self.import_source})
                return
            elif self.action == "import_named":
                if self.importer is None or self.import_source is None or self.import_option is None or self.registry is None:
                    raise DownloadError("未选择要导入的本地模型包。")
                entry = self.importer.import_named(self.import_option, self.import_source, self.display_name,
                                                  self.registry, self._progress("new-import"), self.cancel)
                state = self.downloader.verify_entry(entry, self._progress("new-import"), self.cancel)
                if not state.ready:
                    raise DownloadError(state.detail)
                states[entry.id] = (VerificationResult(True, "ready", "本地导入 · 目录固定 SHA-256 校验通过")
                                    if entry.verification == "pinned_catalog_sha256" else state)
                self.model_id = entry.id
            elif self.action == "download":
                entry = next(entry for entry in self.entries if entry.id == self.model_id)
                self.downloader.download_entry(entry, self._progress(entry.id),
                                               self.cancel, restart=self.restart)
                states[entry.id] = VerificationResult(True, "ready", "已下载 · SHA-256 校验通过")
            elif self.action == "import":
                entry = next(entry for entry in self.entries if entry.id == self.model_id)
                if self.importer is None or self.import_source is None:
                    raise DownloadError("未选择要导入的本地模型包。")
                self.importer.import_entry(entry, self.import_source, self._progress(entry.id), self.cancel)
                state = self.downloader.verify_entry(entry, self._progress(entry.id), self.cancel)
                if not state.ready:
                    raise DownloadError(state.detail)
                states[entry.id] = VerificationResult(True, "ready", "本地导入 · 目录固定 SHA-256 校验通过")
            elif self.action == "activate":
                if self.asr_catalog is None:
                    if not self.translation_required:
                        raise DownloadError("Codex 翻译模式需要选择本地识别模型")
                    self.downloader.activate_model(self.catalog, self.model_id,
                                                   self._progress(self.model_id), self.cancel)
                else:
                    self.downloader.activate_models(self.catalog, self.model_id, self.asr_catalog,
                                                    self.asr_model_id, self._progress(self.model_id if self.translation_required else self.asr_model_id), self.cancel,
                                                    speech_only=self.speech_only,
                                                    translation_required=self.translation_required)
            result = {"states": states, "activated": self.model_id if self.action == "activate" and self.translation_required else None,
                      "activated_asr": self.asr_model_id if self.action == "activate" else None}
            if self.action in ("import", "import_named"):
                result["imported"] = self.model_id
            self.result.emit(result)
        except Exception as exc:
            if self.cancel.is_set() and not isinstance(exc, DownloadError):
                exc = DownloadCancelled("已取消；未完成的 .part 文件保留，可继续下载")
            self.result.emit({"states": states, "error": str(exc),
                              "cancelled": isinstance(exc, DownloadCancelled)})
        finally:
            self.finished.emit()


class ModelManagerDialog(QDialog):
    model_selected = Signal(str)

    @property
    def is_busy(self) -> bool:
        """Before QApplication.quit, cancel/reject and await dialog.finished if busy."""
        return self._thread is not None

    def __init__(self, models_root: Path, parent=None, initial_model_id: str | None = None,
                 *, catalog: ModelCatalog | None = None, downloader: ModelDownloader | None = None,
                 asr_catalog: AsrCatalog | None = None, initial_asr_model_id: str | None = None,
                 active_models: dict | None = None, speech_only: bool = True,
                 translation_required: bool = True, importer: ModelImporter | None = None):
        super().__init__(parent)
        self.models_root = Path(models_root).resolve()
        self.speech_only = speech_only
        self.translation_required = translation_required
        self._base_catalog = catalog or ModelCatalog.default()
        # Custom legacy catalogs keep their original one-tab API. Normal app
        # construction exposes both independent selections.
        self._base_asr_catalog = asr_catalog if asr_catalog is not None else (AsrCatalog.default() if catalog is None or not translation_required else None)
        self.registry = ModelRegistry(self.models_root, catalog=self._base_catalog, asr_catalog=self._base_asr_catalog)
        if self._base_asr_catalog is not None:
            self._base_asr_catalog = self.registry.asr_catalog
        self._registry_error = None
        try:
            self.catalog, registered_asr = self.registry.catalogs()
            self.asr_catalog = registered_asr if self._base_asr_catalog is not None else None
        except (DownloadError, OSError) as exc:
            self.catalog, self.asr_catalog = self._base_catalog, self._base_asr_catalog
            self._registry_error = str(exc)
        self._required_entries = vad_dependencies(self.catalog, speech_only=speech_only) if self.asr_catalog is not None else self.catalog.required_entries
        self._translation_entries = (*self._required_entries, *(entry for entry in self.catalog.entries if not entry.required))
        self._entries = (*self._translation_entries, *self.asr_catalog.entries) if self.asr_catalog is not None else self.catalog.entries
        self.downloader = downloader or ModelDownloader(self.models_root)
        self.importer = importer or ModelImporter(self.models_root)
        self.active_models = active_models or {}
        self.selected_model_id: str | None = None
        self.selected_asr_model_id: str | None = None
        self._states: dict[str, VerificationResult] = {}
        self._thread: QThread | None = None
        self._worker: _Worker | None = None
        self._cancel = threading.Event()
        self._close_requested = False
        self._verified_once = False
        self._activated_pending: str | None = None
        self._activated_asr_pending: str | None = None
        self._active_id: str | None = None
        self._active_action: str | None = None
        self._inspection_pending = None
        self._catalogs_pending = False
        self._building_pages = False
        application = QApplication.instance()
        if application is not None:
            application.aboutToQuit.connect(self._shutdown)
        self.setWindowTitle(tr("离线模型管理"))
        self.resize(820, 720)
        layout = QVBoxLayout(self)
        self.instructions_label = _label(
            "模型独立存放在下方目录。联网下载一次，校验后可断网使用。下载不会自动切换当前模型；"
            "选中识别、翻译模型和基础分句都就绪后，点击“使用选中模型”启用。未改动的文件复用校验记录；"
            "切换保留其他已下载模型。" if translation_required else
            "当前翻译源：Codex CLI（联网，使用 ChatGPT 订阅额度）。识别固定使用 SenseVoice + Silero VAD；"
            "识别、基础分句和已开启的声音筛选依赖就绪后即可启用，无需下载本地翻译权重。"
            "本地翻译模型仍可查看和下载，已保存的翻译模型设置会保留。")
        layout.addWidget(self.instructions_label)
        self.import_instructions_label = _label(
            "点击顶部“导入本地模型”，选择文件、ZIP 或文件夹；离线识别兼容模板后填写名称。"
            "识别仅支持 SenseVoice，翻译模型导入保留。导入不联网、不覆盖已有模型，也不会自动启用。未知 ONNX / CT2 暂不支持。")
        layout.addWidget(self.import_instructions_label)
        self.import_button = QPushButton(tr("导入本地模型"))
        self.import_button.setObjectName("unifiedModelImport")
        import_menu = QMenu(self.import_button)
        import_menu.addAction(tr("选择 ZIP / 模型文件")).triggered.connect(lambda: self._choose_source(folder=False))
        import_menu.addAction(tr("选择模型文件夹")).triggered.connect(lambda: self._choose_source(folder=True))
        self.import_button.setMenu(import_menu)
        layout.addWidget(self.import_button)
        self.path_label = _label(str(self.models_root))
        self.path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.path_label)
        self.current_models_label = _label(tr("当前模型：{0}", loaded_models_summary(self.active_models)))
        layout.addWidget(self.current_models_label)
        self.group = QButtonGroup(self)
        self.asr_group = QButtonGroup(self)
        self.radios: dict[str, QRadioButton] = {}
        self.asr_radios: dict[str, QRadioButton] = {}
        self.status_labels: dict[str, QLabel] = {}
        self.download_buttons: dict[str, QPushButton] = {}
        self.restart_buttons: dict[str, QPushButton] = {}
        self.import_buttons: dict[str, QPushButton] = {}
        self.active_labels: dict[str, QLabel] = {}
        self.tabs: QTabWidget | None = None
        self.pages_widget = QWidget()
        self.pages_layout = QVBoxLayout(self.pages_widget)
        self.pages_layout.setContentsMargins(0, 0, 0, 0)
        self._build_pages()
        layout.addWidget(self.pages_widget, 1)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)
        self.message = _label("打开时检查本地文件，未变化则复用校验记录，不自动联网下载。")
        if self._registry_error:
            self.message.setText(tr(self._registry_error))
        layout.addWidget(self.message)
        bottom = QHBoxLayout()
        self.verify_button = QPushButton(tr("重新校验本地文件"))
        self.verify_button.clicked.connect(lambda: self._start_task("verify_full"))
        self.cancel_button = QPushButton(tr("取消当前任务"))
        self.cancel_button.clicked.connect(self._cancel_task)
        self.use_button = QPushButton(tr("使用选中模型" if translation_required else "使用选中识别模型"))
        self.use_button.clicked.connect(self._activate)
        close = QPushButton(tr("关闭"))
        close.clicked.connect(self.reject)
        for button in (self.verify_button, self.cancel_button, self.use_button, close):
            bottom.addWidget(button)
        layout.addLayout(bottom)
        preferred = initial_model_id or self._current_model_id() or self.catalog.default_model_id
        if preferred not in self.radios:
            preferred = self.catalog.default_model_id
        if self.asr_catalog is not None:
            asr_preferred = initial_asr_model_id or self._current_asr_model_id() or self.asr_catalog.default_model_id
            if asr_preferred not in self.asr_radios:
                asr_preferred = self.asr_catalog.default_model_id
            self.asr_radios[asr_preferred].setChecked(True)
        self.radios[preferred].setChecked(True)
        self._refresh_controls()

    def _build_pages(self) -> None:
        if self.asr_catalog is not None:
            self.tabs = QTabWidget()
            self.tabs.addTab(self._model_page(self._translation_entries, False), tr("翻译模型"))
            self.tabs.addTab(self._model_page(self.asr_catalog.entries, True), tr("识别模型"))
            if not self.translation_required:
                self.tabs.setCurrentIndex(1)
            self.pages_layout.addWidget(self.tabs)
        else:
            self.pages_layout.addWidget(self._model_page(self._entries, False))

    def _reload_catalogs(self) -> None:
        chosen, chosen_asr = self._chosen_id(), self._chosen_asr_id()
        tab = self.tabs.currentIndex() if self.tabs else 0
        self.catalog, registered_asr = self.registry.catalogs()
        self.asr_catalog = registered_asr if self._base_asr_catalog is not None else None
        self._required_entries = vad_dependencies(self.catalog, speech_only=self.speech_only) if self.asr_catalog is not None else self.catalog.required_entries
        self._translation_entries = (*self._required_entries, *(entry for entry in self.catalog.entries if not entry.required))
        self._entries = (*self._translation_entries, *self.asr_catalog.entries) if self.asr_catalog is not None else self.catalog.entries
        self._building_pages = True
        try:
            for group in (self.group, self.asr_group):
                for button in group.buttons():
                    group.removeButton(button)
            while self.pages_layout.count():
                item = self.pages_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            for mapping in (self.radios, self.asr_radios, self.status_labels, self.download_buttons,
                            self.restart_buttons, self.import_buttons, self.active_labels):
                mapping.clear()
            self.tabs = None
            self._build_pages()
            self.radios[chosen if chosen in self.radios else self.catalog.default_model_id].setChecked(True)
            if self.asr_catalog is not None:
                self.asr_radios[chosen_asr if chosen_asr in self.asr_radios else self.asr_catalog.default_model_id].setChecked(True)
            if self.tabs:
                self.tabs.setCurrentIndex(tab)
            for mid, state in self._states.items():
                if mid in self.status_labels:
                    self.status_labels[mid].setText(_status_text(state.detail))
        finally:
            self._building_pages = False

    def _model_page(self, entries: tuple[ModelEntry, ...], asr: bool) -> QScrollArea:
        content, rows = QWidget(), QVBoxLayout()
        content.setLayout(rows)
        for entry in entries:
            box = QGroupBox(entry.name)
            box_layout = QVBoxLayout(box)
            role = "asr" if asr else "translation"
            active_info = self.active_models.get(role)
            if not asr and isinstance(active_info, dict) and active_info.get("mode") == "hybrid":
                active_info = active_info.get("draft_model")
            _, device, active_id = loaded_model_display(
                active_info, role, self.asr_catalog if asr else self.catalog)
            if active_id == entry.id:
                active = _label(tr("当前已加载 · {0}", device))
                active.setStyleSheet("font-weight: 600;")
                box_layout.addWidget(active)
                self.active_labels[entry.id] = active
            if not entry.required:
                radio = QRadioButton(tr("选择此识别模型" if asr else "选择此翻译模型"))
                (self.asr_group if asr else self.group).addButton(radio)
                (self.asr_radios if asr else self.radios)[entry.id] = radio
                radio.toggled.connect(self._refresh_controls)
                box_layout.addWidget(radio)
            box_layout.addWidget(_label(
                tr("权重：{0} · 推荐指数：{1}", _size(entry.size), tr(entry.recommendation)) + "\n"
                + tr(entry.description) + "\n" + tr(entry.min_memory_description)
                + ("\n" + tr("推荐依据：{0}", tr(entry.recommendation_basis)) if entry.recommendation_basis else "")))
            status = _label("尚未校验")
            self.status_labels[entry.id] = status
            box_layout.addWidget(status)
            buttons = QHBoxLayout()
            download = QPushButton(tr("下载 / 续传"))
            download.clicked.connect(lambda checked=False, mid=entry.id: self._start_task("download", mid))
            self.download_buttons[entry.id] = download
            buttons.addWidget(download)
            restart = QPushButton(tr("重新下载"))
            restart.clicked.connect(lambda checked=False, mid=entry.id: self._start_task("download", mid, restart=True))
            self.restart_buttons[entry.id] = restart
            buttons.addWidget(restart)
            if entry.info_url:
                info = QPushButton(tr("官方资料"))
                info.clicked.connect(lambda checked=False, url=entry.info_url: QDesktopServices.openUrl(QUrl(url)))
                buttons.addWidget(info)
            if all(file.url is not None for file in entry.files):
                direct = QPushButton(tr("下载直链"))
                menu = QMenu(direct)
                for file in entry.files:
                    action = menu.addAction(Path(file.relative_path).name)
                    action.triggered.connect(lambda checked=False, url=file.url: QDesktopServices.openUrl(QUrl(url)))
                direct.setMenu(menu)
                buttons.addWidget(direct)
            manual = QPushButton(tr("离线放置说明"))
            manual.clicked.connect(lambda checked=False, model=entry: self._manual_instructions(model))
            buttons.addWidget(manual)
            box_layout.addLayout(buttons)
            rows.addWidget(box)
        rows.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        return scroll

    def _current_asr_model_id(self) -> str | None:
        try:
            profile = self.models_root / "asr/backend.json"
            if not profile.resolve().is_relative_to(self.models_root) or profile.stat().st_size > 65536:
                return None
            raw = json.loads(profile.read_text(encoding="utf-8-sig"))
            if not isinstance(raw, dict):
                return None
            for entry in self.asr_catalog.entries:
                expected = profile_for_entry(entry)
                if raw.get("model_id") not in (None, entry.id):
                    continue
                # Canonical and independently imported SenseVoice use the same
                # backend; their actual model paths distinguish the selection.
                if (raw.get("backend") == expected["backend"]
                        and all(raw.get(key, default) == expected.get(key, default)
                                for key, default in (("model", "model.int8.onnx"), ("tokens", "tokens.txt")))):
                    return entry.id
        except (OSError, ValueError):
            pass
        return None

    def _current_model_id(self) -> str | None:
        """Read only the tiny active profile, never hash a weight on the UI thread."""
        try:
            directory = (self.models_root / "translation").resolve()
            profile = directory / "backend.json"
            if not directory.is_relative_to(self.models_root) or profile.stat().st_size > 65536:
                return None
            raw = json.loads(profile.read_text(encoding="utf-8-sig"))
            if not isinstance(raw, dict) or raw.get("backend") != "llama_cpp":
                return None
            value = raw.get("model_file")
            if not isinstance(value, str) or not value or Path(value).is_absolute():
                return None
            path = (directory / value).resolve()
            if not path.is_relative_to(directory):
                return None
            return next((entry.id for entry in self.catalog.entries if not entry.required
                         and len(entry.files) == 1
                         and self.downloader.path_for(entry.files[0]).resolve() == path), None)
        except (OSError, ValueError):
            return None

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        if not self._verified_once:
            self._verified_once = True
            self._start_task("verify")

    def _chosen_id(self) -> str | None:
        return next((mid for mid, radio in self.radios.items() if radio.isChecked()), None)

    def _chosen_asr_id(self) -> str | None:
        return next((mid for mid, radio in self.asr_radios.items() if radio.isChecked()), None)

    def _refresh_controls(self, *args) -> None:
        if self._building_pages:
            return
        busy = self._thread is not None
        chosen = self._chosen_id()
        ready_ids = [entry.id for entry in self._required_entries]
        if self.translation_required:
            ready_ids += [chosen] if chosen else [None]
        if self.asr_catalog is not None:
            chosen_asr = self._chosen_asr_id()
            ready_ids += [chosen_asr] if chosen_asr else [None]
        ready = all(self._states.get(mid, VerificationResult(False, "missing", "")).ready
                    for mid in ready_ids)
        self.use_button.setEnabled(not busy and ready)
        self.verify_button.setEnabled(not busy)
        self.cancel_button.setEnabled(busy and not self._cancel.is_set())
        self.import_button.setEnabled(not busy and self._registry_error is None)
        for mid, button in self.download_buttons.items():
            ready_entry = self._states.get(mid, VerificationResult(False, "missing", "")).ready
            entry = next(entry for entry in self._entries if entry.id == mid)
            downloadable = all(file.url is not None for file in entry.files)
            button.setEnabled(not busy and not ready_entry and downloadable)
            self.restart_buttons[mid].setEnabled(not busy and not ready_entry and downloadable)
        for radio in self.radios.values():
            radio.setEnabled(not busy and self.translation_required)
        for radio in self.asr_radios.values():
            radio.setEnabled(not busy)

    def _choose_import(self, entry: ModelEntry, *, folder: bool) -> None:
        if self.is_busy or self._close_requested:
            return
        title = tr("导入 {0}", entry.name)
        if folder:
            source = QFileDialog.getExistingDirectory(self, title)
        else:
            source, _ = QFileDialog.getOpenFileName(
                self, title, "", tr("ZIP / 模型文件 (*.zip *.gguf *.onnx *.txt *.csv);;所有文件 (*)"))
        if source:
            self._start_task("import", entry.id, import_source=Path(source))

    def _choose_source(self, *, folder: bool) -> None:
        if self.is_busy or self._close_requested:
            return
        if folder:
            source = QFileDialog.getExistingDirectory(self, tr("选择模型文件夹"))
        else:
            source, _ = QFileDialog.getOpenFileName(
                self, tr("选择 ZIP / 模型文件"), "",
                tr("ZIP / 模型文件 (*.zip *.gguf *.onnx);;所有文件 (*)"))
        if source:
            self._start_task("inspect", import_source=Path(source))

    def _confirm_import(self, inspection: ImportInspection, source: Path) -> None:
        dialog = ModelImportDialog(inspection, source, self,
                                   existing_names=(entry.name for entry in self.registry.entries()))
        try:
            if dialog.exec() == QDialog.DialogCode.Accepted and not self._close_requested:
                self._start_task("import_named", import_source=source,
                                 import_option=dialog.selected_option, display_name=dialog.display_name)
        finally:
            dialog.deleteLater()

    def _start_task(self, action: str, model_id: str | None = None, *, restart: bool = False,
                    import_source: Path | None = None, import_option: ImportOption | None = None,
                    display_name: str | None = None) -> None:
        if self._thread is not None or self._close_requested:
            return
        self._cancel = threading.Event()
        self._active_id = self._chosen_asr_id() if action == "activate" and not self.translation_required else model_id
        self._active_action = action
        self._activated_pending = self._activated_asr_pending = None
        if action in ("verify", "verify_full"):
            self._states.clear()
        message = ("正在离线检查模型包与兼容模板…" if action == "inspect" else
                   "正在导入本地模型… 原模型包保留，导入不会自动切换。" if action in ("import", "import_named") else
                   "正在下载；可取消并保留进度。" if action == "download" else "正在校验本地文件…")
        self.message.setText(tr(message))
        self.progress_bar.setValue(0)
        thread = QThread(self)
        worker = _Worker(self.downloader, self.catalog, action, model_id, self._cancel, restart,
                         asr_catalog=self.asr_catalog, asr_model_id=self._chosen_asr_id(), entries=self._entries,
                         speech_only=self.speech_only, translation_required=self.translation_required,
                         importer=self.importer, import_source=import_source, registry=self.registry,
                         import_option=import_option, display_name=display_name)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_progress)
        worker.result.connect(self._on_result)
        worker.finished.connect(worker.deleteLater)
        worker.finished.connect(thread.quit, Qt.ConnectionType.DirectConnection)
        thread.finished.connect(self._task_finished)
        thread.finished.connect(thread.deleteLater)
        self._thread, self._worker = thread, worker
        self._refresh_controls()
        thread.start()

    @Slot(str, str, int)
    def _on_progress(self, model_id: str, phase: str, percent: int) -> None:
        text = (tr("检查中… {0}%", percent) if phase == "inspecting" else
                tr("校验中…") if phase == "verifying" else tr("复制中… {0}%", percent) if phase == "importing"
                else tr("保存导入模型… {0}%", percent) if phase == "publishing" else tr("下载中… {0}%", percent))
        if model_id in self.status_labels:
            self.status_labels[model_id].setText(text)
        else:
            self.message.setText(text)
        self.progress_bar.setValue(percent)

    @Slot(object)
    def _on_result(self, result: dict) -> None:
        self._states.update(result["states"])
        if result.get("inspection"):
            self._inspection_pending = (result["inspection"], result["source"])
            self.message.setText(tr("检查完成，请确认用途、模板与显示名称。"))
            return
        if self._active_action == "import_named":
            self._catalogs_pending = True
        if result.get("error"):
            if self._active_id:
                previous = self._states.get(self._active_id)
                state = (previous if self._active_action in ("import", "import_named") and previous is not None and previous.ready else
                         VerificationResult(False, "partial", "导入已取消，原模型包保留" if self._active_action in ("import", "import_named", "inspect") else "已取消，可续传")
                         if result.get("cancelled") else VerificationResult(False, "error", "任务失败，可手动重试"))
                self._states[self._active_id] = state
            self.message.setText(_status_text(result["error"]))
        else:
            self.message.setText(tr(("导入完成。原模型包保留；请选择模型后点击“使用选中模型”，不会自动切换。"
                                  if self.translation_required else
                                  "导入完成。原模型包保留，当前翻译源仍为 Codex CLI；模型不会自动切换。")
                                 if result.get("imported") else
                                 "操作完成。下载不会自动启用；请选择模型后点击“使用选中模型”。"
                                 if self.translation_required else
                                 "操作完成。当前翻译源仍为 Codex CLI；可点击“使用选中识别模型”切换本地识别。"))
        self._activated_pending = result.get("activated")
        self._activated_asr_pending = result.get("activated_asr")
        for mid, state in self._states.items():
            if mid in self.status_labels:
                self.status_labels[mid].setText(_status_text(state.detail))
        if result.get("error"):
            for mid, label in self.status_labels.items():
                if mid not in self._states:
                    label.setText(tr("校验未完成；可重新校验"))

    @Slot()
    def _task_finished(self) -> None:
        self._thread, self._worker = None, None
        if self._catalogs_pending:
            self._catalogs_pending = False
            try:
                self._reload_catalogs()
            except (DownloadError, OSError) as exc:
                self._registry_error = str(exc)
                self.message.setText(tr(str(exc)))
        self._active_id = None
        self._active_action = None
        self._refresh_controls()
        if self._activated_pending or self._activated_asr_pending:
            self.selected_model_id = self._activated_pending
            self.selected_asr_model_id = self._activated_asr_pending
            self.model_selected.emit(self.selected_model_id or self.selected_asr_model_id)
            super().accept()
        elif self._close_requested:
            super().reject()
        elif self._inspection_pending is not None:
            inspection, source = self._inspection_pending
            self._inspection_pending = None
            if not self._cancel.is_set():
                try:
                    self._confirm_import(inspection, source)
                except (DownloadError, OSError) as exc:
                    self._registry_error = str(exc)
                    self.message.setText(tr(str(exc)))
                    self._refresh_controls()

    def _cancel_task(self) -> None:
        self._cancel.set()
        self.downloader.cancel_active_download()
        self.message.setText(tr("正在取消导入… 原模型包与已有模型保留。" if self._active_action in ("import", "import_named", "inspect") else
                                "正在取消… 当前网络读取结束后退出；.part 保留，可续传。"))
        self._refresh_controls()

    @Slot()
    def _shutdown(self) -> None:
        """Final process cleanup only; ordinary dialog closes remain asynchronous."""
        thread = self._thread
        if thread is not None:
            self._cancel.set()
            self.downloader.cancel_active_download()
            thread.quit()
            # An in-progress connect cannot be interrupted through HTTPResponse.
            # Keep this bounded above the default socket timeout, without using
            # QThread.terminate(), which could corrupt a partial/model file.
            if not thread.wait(int((self.downloader.timeout + 5) * 1000)):
                # DNS and multiple redirects do not share a total socket timeout.
                # The controller should defer quit asynchronously via is_busy /
                # dialog.finished. Once aboutToQuit runs, raising cannot stop Qt
                # shutdown: retain the thread until it actually exits instead.
                LOGGER.warning("退出等待模型网络连接结束；请在 application.quit 前异步关闭模型管理窗口")
                thread.wait()

    def _activate(self) -> None:
        chosen = self._chosen_id() if self.translation_required else None
        if self.use_button.isEnabled():
            self._start_task("activate", chosen)

    def _manual_instructions(self, entry: ModelEntry) -> None:
        lines = [tr("自定义模型没有下载直链；以下 SHA-256 仅用于确认导入后文件未变化。"
                    if entry.verification == "local_sha256" else
                    "可在其他电脑从“下载直链”下载，再复制到下列完整路径："), ""]
        for file in entry.files:
            lines.extend((str(self.downloader.path_for(file)),
                          tr("大小：{0} B", f"{file.size:,}"), f"SHA-256：{file.sha256}", ""))
        lines.append(tr("保留模型官方许可。放置后点击“重新校验本地文件”；文件名必须完全一致。"))
        message = QMessageBox(self)
        message.setWindowTitle(tr("离线手动放置模型"))
        message.setTextFormat(Qt.TextFormat.PlainText)
        message.setText("\n".join(lines))
        message.exec()

    def reject(self) -> None:
        if self._thread is not None:
            self._close_requested = True
            self._cancel_task()
            return
        super().reject()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._thread is not None:
            self._close_requested = True
            self._cancel_task()
            event.ignore()
        else:
            super().closeEvent(event)
