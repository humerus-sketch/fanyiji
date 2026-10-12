"""Qt Widgets interface; this module does not start audio or persist settings."""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass, replace
import math
from pathlib import Path
import time
from typing import Hashable

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QCloseEvent, QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .config import AppConfig, SourceConfig
from .i18n import LANGUAGE_NAMES, get_language, recovery_action_text, tr
from .identity import APP_NAME


_DIRECTION_LABELS = {
    "zh_to_ja": "中文 → 日文",
    "ja_to_zh": "日文 → 中文",
}
_SOURCE_ALIASES = {
    "microphone": "microphone",
    "mic": "microphone",
    "playback": "playback",
    "system": "playback",
}
_FONT_FAMILY = "Microsoft YaHei UI"
_LATENCY_TOOLTIP = "全程含本段音频时长；句尾等待从音频结束计时，均到原文确认后终稿标签完成 Qt 绘制。识别开始到终稿及 ASR、翻译排队、请求和 UI 各阶段耗时见下方或 LATENCY 日志。"
_LATENCY_BREAKDOWN_LABELS = (
    ("segmentation_seconds", "语音分段"), ("asr_queue_seconds", "识别排队"),
    ("speech_filter_seconds", "讲话筛选"), ("audio_processing_seconds", "语音处理"),
    ("asr_seconds", "识别模型"),
    ("assembly_seconds", "半句拼接"), ("translation_queue_seconds", "翻译排队"),
    ("translation_seconds", "翻译请求"), ("source_confirmation_seconds", "原文确认"),
    ("ui_dispatch_seconds", "界面分发"), ("ui_update_seconds", "界面更新"),
    ("ui_paint_seconds", "界面绘制"),
)
_CODEX_EFFORT_LABELS = {
    "low": "低（low）",
    "medium": "中（medium）",
    "high": "高（high）",
    "xhigh": "很高（xhigh）",
    "max": "最高（max）",
}


def _codex_model_label(model: dict) -> str:
    slug = model["model"]
    display_name = model.get("displayName")
    if not isinstance(display_name, str) or not display_name.strip():
        display_name = slug
    candidate = ""
    if slug == "gpt-6.1-sol":
        candidate = tr("综合候选")
    elif "luna" in slug:
        candidate = tr("速度候选")
    elif "astra" in slug:
        candidate = tr("质量对照")
    return f"{display_name} · {candidate}" if candidate else display_name


def loaded_model_display(info: dict | None, role: str, catalog=None) -> tuple[str, str, str | None]:
    """Name only a successfully loaded backend, never a saved selection."""
    if (not isinstance(info, dict) or not info or info.get("device") == "not_loaded"
            or info.get("available") is False):
        return tr("未加载"), "", None
    if role == "translation" and (info.get("backend") == "codex_cli" or info.get("model_kind") == "codex"):
        model = info.get("model_id") or info.get("model")
        name = f"Codex CLI · {model}" if isinstance(model, str) and model.strip() else tr("Codex CLI · 模型待确认")
        auth = tr(" · ChatGPT 订阅") if info.get("auth_mode") == "chatgpt" else ""
        return name, tr('云端{0}', auth), None
    if catalog is None:
        if role == "asr":
            from .asr_catalog import AsrCatalog
            catalog = AsrCatalog.default()
        else:
            from .models_download import ModelCatalog
            catalog = ModelCatalog.default()
    model_id = info.get("model_id")
    local_id = isinstance(model_id, str) and model_id.startswith("local-")
    matched = next((entry for entry in catalog.entries if entry.id == model_id), None)
    if role == "asr":
        name = {"sense_voice": "SenseVoice INT8", "whisper": "Whisper", "qwen3_asr": "Qwen3-ASR"}.get(
            info.get("model_kind"), tr("本地识别模型"))
    else:
        filename = Path(str(info.get("model_file", ""))).name
        if matched is None and not local_id:
            matched = next((entry for entry in catalog.entries if not entry.required and len(entry.files) == 1
                            and Path(entry.files[0].relative_path).name == filename), None)
        name = "M2M100 418M · INT8" if info.get("model_kind") == "m2m100" else (filename or tr("本地翻译模型"))
    if matched is not None:
        name = matched.name
    display_name = info.get("display_name")
    if (isinstance(display_name, str) and 0 < len(display_name.strip()) <= 80
            and not any(ord(character) < 32 or ord(character) == 127 for character in display_name)):
        name = display_name.strip()
    device = {"cpu": "CPU", "cuda": "CUDA"}.get(info.get("device"), tr("设备未知"))
    identifier = model_id if local_id else (matched.id if matched is not None else None)
    return name, device, identifier


def loaded_models_summary(models: dict | None) -> str:
    values = []
    for role, label in (("asr", tr("识别")), ("translation", tr("翻译"))):
        info = (models or {}).get(role)
        name, device, _ = loaded_model_display(info, role)
        if role == "translation" and isinstance(info, dict) and info.get("mode") == "hybrid":
            draft_name, draft_device, _ = loaded_model_display(info.get("draft_model"), role)
            values.append(tr("暂译：{0}", draft_name) + (f"（{draft_device}）" if draft_device else ""))
            values.append(tr("终稿：{0}", name) + (f"（{device}）" if device else ""))
        else:
            values.append(f"{tr(label)}：{name}" + (f"（{device}）" if device else ""))
        if role == "asr" and isinstance(info, dict) and info.get("device") != "not_loaded":
            if info.get("recognition_mode") == "auto":
                values.append(tr("识别语言：自动多语"))
            elif info.get("recognition_mode") == "fixed":
                values.append(tr("识别语言：按通道方向"))
        if (role == "translation" and isinstance(info, dict)
                and (info.get("backend") == "codex_cli" or info.get("model_kind") == "codex")):
            effort = info.get("reasoning_effort", info.get("effort"))
            effort_label = tr(_CODEX_EFFORT_LABELS.get(effort, effort)) if isinstance(effort, str) and effort else tr("待确认")
            speed_value = info.get("speed")
            speed = {"standard": tr("标准"), "fast": "Fast"}.get(speed_value, tr("待确认")) if isinstance(speed_value, str) else tr("待确认")
            values.append(tr('推理：{0} · 速度：{1}', effort_label, speed))
    filter_info = (models or {}).get("speech_filter")
    if isinstance(filter_info, dict):
        if filter_info.get("enabled") is False:
            values.append(tr("讲话筛选：关闭"))
        elif filter_info.get("enabled") is True and filter_info.get("model_loaded") is True:
            values.append(tr("讲话筛选：已启用"))
    processing = (models or {}).get("audio_processing")
    if isinstance(processing, dict):
        mode = processing.get("mode", "off")
        label = {"off": tr("关闭"), "gtcrn": tr("GTCRN 语音增强"),
                 "uvr": tr("UVR 伴奏分离")}.get(mode, tr("待确认"))
        if mode != "off" and (processing.get("model_loaded") is not True or processing.get("action") == "bypass"):
            label += " · " + tr("原音回退")
        values.append(tr("语音处理：{0}", label))
    return "  ·  ".join(values)


def _plain_label(text: str = "", *, wrap: bool = False) -> QLabel:
    label = QLabel(text)
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setWordWrap(wrap)
    return label


class _PresentedLabel(QLabel):
    """A result is presented only after its visible label has been painted."""

    presented = Signal(dict)

    def __init__(self, text: str = "") -> None:
        super().__init__(text)
        self.setTextFormat(Qt.TextFormat.PlainText)
        self.setWordWrap(True)
        self.presentation: dict | None = None
        self.presentation_allowed = None
        self._last_presented = None

    def clear_presentation(self) -> None:
        self.presentation = None

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        pending = self.presentation
        if (pending is None or not self.isVisible() or self.window().isMinimized()
                or self.visibleRegion().intersected(event.region()).isEmpty()):
            return
        if self.presentation_allowed is not None and not self.presentation_allowed():
            return
        identity = tuple(pending.get(key) for key in (
            "session", "channel", "segment_id", "revision", "context_generation", "stage"))
        self.presentation = None
        if identity == self._last_presented:
            return
        self._last_presented = identity
        self.presented.emit({**pending, "ui_painted_at": time.monotonic()})


def _direction_combo(direction: str) -> QComboBox:
    combo = QComboBox()
    for value, label in _DIRECTION_LABELS.items():
        combo.addItem(tr(label), value)
    index = combo.findData(direction)
    combo.setCurrentIndex(index if index >= 0 else 0)
    combo.setToolTip(tr("自动多语识别开启时，箭头右侧决定译文语言；左侧作为通道主要语言参考。"))
    return combo


class SettingsDialog(QDialog):
    """Collect source settings without accessing storage or audio devices."""

    codex_models_requested = Signal()
    browser_requested = Signal()
    audio_processing_requested = Signal()

    def __init__(
        self,
        config: AppConfig | None = None,
        first_run: bool = False,
        parent: QWidget | None = None,
        *,
        codex_models: list[dict] | None = None,
    ) -> None:
        super().__init__(parent)
        self._base_config = config
        self._result_config: AppConfig | None = None
        self._first_run = first_run
        self._codex_models: dict[str, dict] = {}
        self.setWindowTitle(APP_NAME + " · " + (tr("首次设置") if first_run else tr("字幕设置")))
        self.setFont(QFont(_FONT_FAMILY, 10))
        self.setMinimumWidth(560)
        self.setStyleSheet(
            "QWidget { color: #182337; background: transparent; }"
            "QDialog { background: #ffffff; }"
            "QPushButton { background: #eef3fa; border: 1px solid #c4cfde;"
            " border-radius: 5px; padding: 6px 12px; }"
            "QPushButton:hover { background: #dce9f8; }"
        )

        microphone = (
            config.microphone
            if config is not None
            else SourceConfig(enabled=True, direction="zh_to_ja")
        )
        playback = (
            config.playback
            if config is not None
            else SourceConfig(enabled=True, direction="ja_to_zh")
        )

        outer = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        body = QWidget()
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)
        layout = QVBoxLayout(body)
        layout.setContentsMargins(22, 20, 22, 20)
        layout.setSpacing(16)
        title = _plain_label(tr("选择声音来源和翻译方向"))
        title.setFont(QFont(_FONT_FAMILY, 14, QFont.Weight.Bold))
        layout.addWidget(title)
        explanation = (
            tr("保存设置后才会开始采集声音。两路可以分别开启，也可以分别选择翻译方向。")
            if first_run
            else tr("两路可以分别开启，也可以分别选择翻译方向。保存后应用新设置。")
        )
        layout.addWidget(_plain_label(explanation, wrap=True))

        language_row = QHBoxLayout()
        language_row.addWidget(_plain_label(tr("界面语言（重新启动后生效）")))
        self.ui_language = QComboBox()
        self.ui_language.setObjectName("uiLanguageCombo")
        for locale, name in LANGUAGE_NAMES.items():
            self.ui_language.addItem(name, locale)
        language_index = self.ui_language.findData(getattr(config, "ui_language", get_language()))
        self.ui_language.setCurrentIndex(max(0, language_index))
        self.ui_language.setToolTip(tr("界面语言与识别语言、翻译方向相互独立。保存后重新启动软件生效。"))
        language_row.addWidget(self.ui_language, 1)
        layout.addLayout(language_row)

        provider_row = QHBoxLayout()
        provider_row.addWidget(_plain_label(tr("翻译源")))
        self.translation_provider = QComboBox()
        self.translation_provider.addItem(tr("本地模型（离线）"), "local")
        self.translation_provider.addItem(tr("Codex CLI（联网，使用 ChatGPT 订阅）"), "codex")
        self.translation_provider.addItem(tr("混合翻译（本地暂译 + Codex 终稿）"), "hybrid")
        index = self.translation_provider.findData(getattr(config, "translation_provider", "local"))
        self.translation_provider.setCurrentIndex(index if index >= 0 else 0)
        provider_row.addWidget(self.translation_provider, 1)
        layout.addLayout(provider_row)

        self.hybrid_notice = _plain_label(tr(
            "本地模型与 Codex 同时翻译同一份原文。本地结果先显示为暂译，Codex 返回后替换为终稿。"
            "请在模型管理中选择已下载的本地翻译模型；Codex 不可用时保留暂译并标记未确认。"), wrap=True)
        self.hybrid_notice.setStyleSheet("color: #5e6c82;")
        layout.addWidget(self.hybrid_notice)

        self.codex_options = QWidget()
        codex_layout = QVBoxLayout(self.codex_options)
        codex_layout.setContentsMargins(0, 0, 0, 0)
        model_row = QHBoxLayout()
        model_row.addWidget(_plain_label(tr("Codex 模型")))
        self.codex_model = QComboBox()
        self.codex_model.setToolTip(tr("从当前账户的模型列表选择；主页显示实际生效的模型。"))
        self.codex_model.addItem(tr("账户默认（模型目录推荐）"), "")
        saved_model = getattr(config, "codex_model", "")
        if saved_model:
            self.codex_model.addItem(tr('{0} · 暂未验证', saved_model), saved_model)
            self.codex_model.setCurrentIndex(1)
        model_row.addWidget(self.codex_model, 1)
        self.codex_models_refresh = QPushButton(tr("刷新模型"))
        self.codex_models_refresh.clicked.connect(self.codex_models_requested.emit)
        model_row.addWidget(self.codex_models_refresh)
        codex_layout.addLayout(model_row)
        self.codex_models_status = _plain_label(tr("请刷新账户模型列表，再选择可用模型。"), wrap=True)
        self.codex_models_status.setStyleSheet("color: #5e6c82;")
        codex_layout.addWidget(self.codex_models_status)
        tuning_row = QHBoxLayout()
        tuning_row.addWidget(_plain_label(tr("推理强度")))
        self.codex_effort = QComboBox()
        self.codex_effort.addItem(tr("账户默认"), "")
        saved_effort = getattr(config, "codex_effort", "")
        if saved_effort:
            self.codex_effort.addItem(tr('{0} · 暂未验证', tr(_CODEX_EFFORT_LABELS.get(saved_effort, saved_effort))), saved_effort)
            self.codex_effort.setCurrentIndex(1)
        self.codex_effort.setToolTip(tr("选项依据当前账户返回的模型能力。低推理可作为速度对照，翻译准确率需要实测。"))
        tuning_row.addWidget(self.codex_effort, 1)
        tuning_row.addWidget(_plain_label(tr("速度")))
        self.codex_speed = QComboBox()
        self.codex_speed.addItem(tr("标准"), "standard")
        self.codex_speed.addItem(tr("Fast（额度消耗更快）"), "fast")
        speed_index = self.codex_speed.findData(getattr(config, "codex_speed", "standard"))
        self.codex_speed.setCurrentIndex(max(0, speed_index))
        self.codex_speed.setToolTip(tr("Fast 需要账户和模型支持，订阅额度消耗通常为标准模式的 2.5 倍；不支持时会报告错误。"))
        tuning_row.addWidget(self.codex_speed, 1)
        codex_layout.addLayout(tuning_row)
        self.codex_model.currentIndexChanged.connect(self._update_codex_efforts)
        if codex_models is not None:
            self.set_codex_models(codex_models)
        self.codex_notice = _plain_label(
            tr("声音先在本机识别，再将原文与最近上下文发送给 Codex 翻译。"
            "需要联网并登录 ChatGPT，持续翻译会消耗 Codex 订阅额度。"
            "Fast 需要账户及模型支持，通常消耗约 2.5 倍订阅额度。"), wrap=True)
        self.codex_notice.setStyleSheet("color: #5e6c82;")
        self.codex_notice.setMinimumHeight(self.codex_notice.fontMetrics().lineSpacing() * 3 + 6)
        codex_layout.addWidget(self.codex_notice)
        layout.addWidget(self.codex_options)
        self.translation_provider.currentIndexChanged.connect(self._update_translation_provider)
        self._update_translation_provider()

        self.microphone_enabled = QCheckBox(tr("麦克风"))
        self.microphone_enabled.setChecked(microphone.enabled)
        self.microphone_direction = _direction_combo(microphone.direction)
        self.playback_enabled = QCheckBox(tr("电脑声音"))
        self.playback_enabled.setChecked(playback.enabled)
        self.playback_direction = _direction_combo(playback.direction)
        self._add_source_row(
            layout,
            self.microphone_enabled,
            self.microphone_direction,
            tr("来自本地话筒"),
        )
        self._add_source_row(
            layout,
            self.playback_enabled,
            self.playback_direction,
            tr("来自电脑正在播放的声音"),
        )
        self.multilingual_asr = QCheckBox(tr("自动多语识别（日语、中文、英语；译文语言按方向设置）"))
        self.multilingual_asr.setChecked(getattr(config, "multilingual_asr", True))
        self.multilingual_asr.setToolTip(
            tr("识别不再强制单一语言，允许主播和游戏切换语言。Codex CLI 和本地 GGUF 翻译支持此模式；"
            "旧 M2M100 翻译保持固定语言并提示。不能分离同时重叠的人声。"))
        layout.addWidget(self.multilingual_asr)
        self.speech_only = QCheckBox(tr("只翻译讲话（过滤音乐/唱歌片段）"))
        self.speech_only.setToolTip(tr("跳过高置信度的音乐、唱歌片段；短句和不确定片段优先保留。无法分离同时重叠的讲话与歌声。"))
        self.speech_only.setChecked(getattr(config, "speech_only", True))
        layout.addWidget(self.speech_only)
        note = _plain_label(tr("建议使用耳机，避免电脑声音再次进入麦克风。"), wrap=True)
        note.setStyleSheet("color: #5e6c82;")
        layout.addWidget(note)

        if not first_run:
            self.audio_processing_button = QPushButton(tr("音源与语音处理"))
            self.audio_processing_button.clicked.connect(self.audio_processing_requested.emit)
            layout.addWidget(self.audio_processing_button)
            self.browser_settings_button = QPushButton(tr("直播弹幕设置"))
            self.browser_settings_button.clicked.connect(self.browser_requested.emit)
            layout.addWidget(self.browser_settings_button)

        self.resize(660, 650)

        buttons = QDialogButtonBox()
        save_button = buttons.addButton(
            tr("保存并开始") if first_run else tr("保存设置"),
            QDialogButtonBox.ButtonRole.AcceptRole,
        )
        save_button.setDefault(True)
        save_button.setMinimumHeight(34)
        buttons.addButton(
            tr("退出") if first_run else tr("取消"),
            QDialogButtonBox.ButtonRole.RejectRole,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

    def _update_translation_provider(self) -> None:
        provider = self.translation_provider.currentData()
        self.codex_options.setVisible(provider in ("codex", "hybrid"))
        self.hybrid_notice.setVisible(provider == "hybrid")

    def set_codex_models_loading(self, loading: bool = True) -> None:
        self.codex_models_refresh.setEnabled(not loading)
        if loading:
            self.codex_models_status.setText(tr("正在读取已登录账户的可用模型…"))

    def set_codex_models(self, models: list[dict], error: str = "") -> None:
        selected = self.codex_model.currentData() or ""
        self._codex_models = {
            entry["model"]: entry for entry in models
            if isinstance(entry, dict) and isinstance(entry.get("model"), str) and entry["model"]
        }
        self.codex_model.blockSignals(True)
        self.codex_model.clear()
        self.codex_model.addItem(tr("账户默认（模型目录推荐）"), "")
        for slug, entry in self._codex_models.items():
            self.codex_model.addItem(_codex_model_label(entry), slug)
            self.codex_model.setItemData(self.codex_model.count() - 1, slug, Qt.ItemDataRole.ToolTipRole)
        if selected and selected not in self._codex_models:
            self.codex_model.addItem(tr('{0} · 暂未验证', selected), selected)
        self.codex_model.setCurrentIndex(max(0, self.codex_model.findData(selected)))
        self.codex_model.blockSignals(False)
        self._populate_codex_efforts(preserve_unverified=True)
        self.set_codex_models_loading(False)
        if error:
            self.codex_models_status.setText(tr('读取模型失败：{0}。已保存的选择仍保留。', error))
        elif self._codex_models:
            self.codex_models_status.setText(
                tr('已读取 {0} 个账户模型。综合、速度、质量标记是评测候选，实际表现需对比。', len(self._codex_models)))
        else:
            self.codex_models_status.setText(tr("账户未返回可用模型，请确认 Codex 登录后重试；可保留账户默认。"))

    def _update_codex_efforts(self) -> None:
        self._populate_codex_efforts(preserve_unverified=False)

    def _populate_codex_efforts(self, *, preserve_unverified: bool) -> None:
        selected = self.codex_effort.currentData() or ""
        entry = self._codex_models.get(self.codex_model.currentData())
        self.codex_effort.clear()
        self.codex_effort.addItem(tr("账户默认"), "")
        efforts = entry.get("supportedReasoningEfforts") if entry else None
        if isinstance(efforts, list):
            for value in efforts:
                effort = value.get("reasoningEffort") if isinstance(value, dict) else value
                if isinstance(effort, str) and effort in _CODEX_EFFORT_LABELS and self.codex_effort.findData(effort) < 0:
                    self.codex_effort.addItem(tr(_CODEX_EFFORT_LABELS[effort]), effort)
        index = self.codex_effort.findData(selected)
        if selected and index < 0 and (preserve_unverified or (entry is None and self.codex_model.currentData())):
            self.codex_effort.addItem(tr('{0} · 暂未验证', tr(_CODEX_EFFORT_LABELS.get(selected, selected))), selected)
            index = self.codex_effort.count() - 1
        self.codex_effort.setCurrentIndex(max(0, index))

    @staticmethod
    def _add_source_row(
        layout: QVBoxLayout,
        enabled: QCheckBox,
        direction: QComboBox,
        description: str,
    ) -> None:
        frame = QFrame()
        frame.setStyleSheet(
            "QFrame { background: #f3f6fa; border-radius: 8px; }"
            "QComboBox { background: white; padding: 5px; }"
        )
        row = QVBoxLayout(frame)
        row.setContentsMargins(14, 12, 14, 12)
        upper = QHBoxLayout()
        upper.addWidget(enabled)
        upper.addStretch()
        upper.addWidget(direction)
        row.addLayout(upper)
        description_label = _plain_label(description)
        description_label.setStyleSheet("color: #5e6c82;")
        row.addWidget(description_label)
        direction.setEnabled(enabled.isChecked())
        enabled.toggled.connect(direction.setEnabled)
        layout.addWidget(frame)

    @property
    def result_config(self) -> AppConfig | None:
        return self._result_config

    def accept(self) -> None:
        microphone = SourceConfig(
            enabled=self.microphone_enabled.isChecked(),
            direction=self.microphone_direction.currentData(),
        )
        playback = SourceConfig(
            enabled=self.playback_enabled.isChecked(),
            direction=self.playback_direction.currentData(),
        )
        if self._base_config is None:
            self._result_config = AppConfig(
                microphone=microphone,
                playback=playback,
                schema_version=1,
                setup_complete=True,
                speech_only=self.speech_only.isChecked(),
                multilingual_asr=self.multilingual_asr.isChecked(),
                ui_language=self.ui_language.currentData(),
                audio_processing=getattr(self._base_config, "audio_processing", "gtcrn"),
                translation_provider=self.translation_provider.currentData(),
                codex_model=self.codex_model.currentData() or "",
                codex_effort=self.codex_effort.currentData() or "",
                codex_speed=self.codex_speed.currentData(),
            )
        else:
            self._result_config = replace(
                self._base_config,
                microphone=microphone,
                playback=playback,
                schema_version=1,
                setup_complete=True,
                speech_only=self.speech_only.isChecked(),
                multilingual_asr=self.multilingual_asr.isChecked(),
                ui_language=self.ui_language.currentData(),
                translation_provider=self.translation_provider.currentData(),
                codex_model=self.codex_model.currentData() or "",
                codex_effort=self.codex_effort.currentData() or "",
                codex_speed=self.codex_speed.currentData(),
            )
        super().accept()

    def reject(self) -> None:
        self._result_config = None
        super().reject()


@dataclass
class _SubtitleRow:
    widget: QWidget
    original: QLabel
    translation: QLabel
    completed: bool = False
    cancelled: bool = False
    revision: int = 0
    confirmed_revision: int = 0
    translated_revision: int = 0
    stage_label: QLabel | None = None
    translation_stage: str = ""
    translation_provider: str = ""
    translation_model: str = ""
    translation_failed: bool = False
    context_generation: int | None = None
    final_failed: bool = False
    local_failed: bool = False


class _SourcePanel(QFrame):
    """Keep completed history bounded while preserving every pending segment."""

    subtitle_presented = Signal(dict)

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.enabled = True
        self._font_size = 16
        self._rows: OrderedDict[Hashable, _SubtitleRow] = OrderedDict()
        self._completion_order: deque[Hashable] = deque()
        self.setObjectName("sourcePanel")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(7)

        header = QHBoxLayout()
        self.title_label = _plain_label(title)
        self.title_label.setFont(QFont(_FONT_FAMILY, 12, QFont.Weight.Bold))
        self.direction_label = _plain_label()
        self.direction_label.setStyleSheet("color: #92c9fb;")
        header.addWidget(self.title_label)
        header.addWidget(self.direction_label)
        header.addStretch()
        self.level = QProgressBar()
        self.level.setRange(0, 100)
        self.level.setValue(0)
        self.level.setTextVisible(False)
        self.level.setFixedSize(85, 8)
        self.level.setToolTip(tr("输入音量"))
        header.addWidget(self.level)
        layout.addLayout(header)

        self.status_label = _plain_label(tr("准备中"))
        self.status_label.setStyleSheet("color: #b4bfd0;")
        layout.addWidget(self.status_label)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setMinimumHeight(60)
        self.content = QWidget()
        self.rows_layout = QVBoxLayout(self.content)
        self.rows_layout.setContentsMargins(0, 3, 8, 3)
        self.rows_layout.setSpacing(12)
        self.placeholder = _plain_label(tr("等待声音…"))
        self.placeholder.setStyleSheet("color: #8e9db3;")
        self.rows_layout.addWidget(self.placeholder)
        self.rows_layout.addStretch()
        self.scroll.setWidget(self.content)
        layout.addWidget(self.scroll, 1)

    def update_source(self, config: SourceConfig) -> None:
        self.enabled = config.enabled
        self.direction_label.setText(tr(_DIRECTION_LABELS.get(config.direction, "方向未设置")))
        self.clear()
        self.status_label.setText(tr("准备中") if self.enabled else tr("已关闭"))
        self.placeholder.setText(tr("等待声音…") if self.enabled else tr("此路声音已关闭"))
        self.level.setValue(0)

    def clear(self) -> None:
        for row in self._rows.values():
            row.translation.clear_presentation()
            self.rows_layout.removeWidget(row.widget)
            row.widget.deleteLater()
        self._rows.clear()
        self._completion_order.clear()
        self.placeholder.show()

    def set_original(self, text: str, segment_id: Hashable, *, revision: int = 1,
                     provisional: bool = False, uncertain: bool = False,
                     context_generation: int | None = None) -> None:
        if not self.enabled:
            return
        row = self._rows.get(segment_id)
        if row is not None and (row.cancelled or revision < row.revision):
            return
        if (row is not None and revision == row.revision and context_generation is not None
                and row.context_generation is not None and context_generation < row.context_generation):
            return
        if row is None:
            row_widget = QWidget()
            row_layout = QVBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(3)
            original = _plain_label(wrap=True)
            original.setStyleSheet("color: #aebdd1;")
            translation = _PresentedLabel(tr("翻译中…"))
            translation.presented.connect(self.subtitle_presented.emit)
            translation.setStyleSheet("color: #94a4ba;")
            for label in (original, translation):
                label.setTextInteractionFlags(
                    Qt.TextInteractionFlag.TextSelectableByMouse
                    | Qt.TextInteractionFlag.TextSelectableByKeyboard
                )
                label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
                row_layout.addWidget(label)
            stage_label = _plain_label()
            stage_label.setStyleSheet("color: #e4b96a;")
            stage_label.hide()
            row_layout.addWidget(stage_label)
            row = _SubtitleRow(row_widget, original, translation, stage_label=stage_label)
            self._rows[segment_id] = row
            self.rows_layout.insertWidget(self.rows_layout.count() - 1, row_widget)
            self._apply_row_font(row)
            self._prune_completed()
        if (revision > row.revision or (context_generation is not None
                and row.context_generation is not None and context_generation > row.context_generation)):
            row.translation.clear_presentation()
            if row.completed:
                row.completed = False
                self._completion_order.remove(segment_id)
            row.translation.setText(tr("等待后半句…") if provisional else tr("翻译中…"))
            row.translation_stage = ""
            row.translation_provider = row.translation_model = ""
            row.translation_failed = False
            row.translated_revision = 0
            row.confirmed_revision = 0
            row.final_failed = row.local_failed = False
            row.stage_label.hide()
        row.revision = revision
        if not provisional:
            row.confirmed_revision = max(row.confirmed_revision, revision)
        if context_generation is not None:
            row.context_generation = context_generation
        row.original.setText(text)
        row.original.setStyleSheet("color: #e4b96a;" if uncertain else "color: #aebdd1;")
        row.original.setToolTip(tr("识别复核尚未确认此段原文，请核对。") if uncertain else "")
        if not provisional and row.translated_revision == revision:
            self.finalize(segment_id, revision)
        self.placeholder.hide()
        self._scroll_to_bottom()

    def set_translation(self, text: str, segment_id: Hashable, *, revision: int = 1,
                        provisional: bool = False, translation_stage: str = "",
                        translation_provider: str = "", translation_model: str = "",
                        context_generation: int | None = None, final_failed: bool = False,
                        failed: bool = False, overall_failed: bool = False,
                        presentation: dict | None = None) -> None:
        # A stale result after a settings/session reset must not create a new row.
        row = self._rows.get(segment_id)
        if row is None or row.cancelled or not self.enabled or revision != row.revision:
            return
        if context_generation is not None and row.context_generation is not None and context_generation != row.context_generation:
            return
        if translation_stage == "draft" and row.translation_stage == "final":
            return
        if final_failed:
            row.final_failed = True
        elif translation_stage == "final" and not failed:
            row.final_failed = False
        row.translated_revision = revision
        row.translation_stage = translation_stage or "final"
        row.translation_provider, row.translation_model = translation_provider, translation_model
        row.translation_failed = failed or overall_failed
        if row.translation_stage == "final" and not provisional:
            row.confirmed_revision = max(row.confirmed_revision, revision)
        row.translation.clear_presentation()
        row.translation.setText(text if text.strip() else tr("（未返回译文）"))
        row.translation.setStyleSheet("color: #f4f7fc;")
        if translation_stage:
            self._update_stage_label(row, translation_provider, translation_model)
        if (failed or overall_failed) and row.stage_label is not None:
            row.stage_label.setText(tr("本地与 Codex 翻译均未完成") if overall_failed else tr("翻译未完成"))
            row.stage_label.setStyleSheet("color: #e4b96a;")
            row.stage_label.show()
        if presentation is not None and text.strip() and not failed and not overall_failed:
            row.translation.presentation = {
                **presentation, "revision": revision, "context_generation": row.context_generation,
                "stage": row.translation_stage, "translation_stage": row.translation_stage,
                "provisional": provisional and row.confirmed_revision < revision,
                "failed": False, "ui_updated_at": time.monotonic()}
            # Source confirmation and stale-result checks still apply at paint time.
            row.translation.presentation_allowed = lambda: (
                self.enabled and self._rows.get(segment_id) is row and not row.cancelled
                and row.revision == revision and row.translation_stage == (translation_stage or "final")
                and (row.translation_stage == "draft" or row.confirmed_revision >= revision))
        if translation_stage == "draft":
            row.translation.setStyleSheet("color: #c4cddb;")
            if row.final_failed and row.confirmed_revision >= revision:
                self._complete_row(segment_id, row)
            self._scroll_to_bottom()
            return
        if not provisional or row.confirmed_revision >= revision:
            row.confirmed_revision = max(row.confirmed_revision, revision)
        else:
            row.translation.setStyleSheet("color: #c4cddb;")
            return
        self._complete_row(segment_id, row)
        self._scroll_to_bottom()

    def _complete_row(self, segment_id: Hashable, row: _SubtitleRow) -> None:
        if not row.completed:
            self._completion_order.append(segment_id)
        row.completed = True
        self._prune_completed()

    def _update_stage_label(self, row: _SubtitleRow, provider: str = "", model: str = "") -> None:
        if row.stage_label is None:
            return
        if row.translation_stage == "draft":
            text = tr("本地暂译 · Codex 未确认") if row.final_failed else tr("本地暂译 · 等待 Codex 终稿")
            row.stage_label.setStyleSheet("color: #e4b96a;")
        elif row.translation_stage == "final":
            provider = row.translation_provider or provider
            model = row.translation_model or model
            if row.confirmed_revision < row.revision:
                text = tr("Codex 译文 · 等待后半句确认") if provider == "codex" else tr("译文 · 等待后半句确认")
                row.stage_label.setStyleSheet("color: #e4b96a;")
            else:
                text = tr("Codex 终稿") if provider == "codex" else tr("终稿")
                row.stage_label.setStyleSheet("color: #92c9fb;")
        else:
            text = tr("Codex 未确认 · 等待本地暂译") if row.final_failed else tr("本地暂译不可用 · 等待 Codex 终稿")
            row.stage_label.setStyleSheet("color: #e4b96a;")
        row.stage_label.setText(text)
        row.stage_label.setToolTip(model)
        row.stage_label.show()

    def set_translation_status(self, segment_id: Hashable, *, revision: int = 1,
                               translation_stage: str = "", translation_provider: str = "",
                               translation_model: str = "", context_generation: int | None = None,
                               failed: bool = False, overall_failed: bool = False, error: str = "") -> None:
        row = self._rows.get(segment_id)
        if row is None or row.cancelled or not self.enabled or revision != row.revision:
            return
        if context_generation is not None and row.context_generation is not None and context_generation != row.context_generation:
            return
        if not failed or row.translation_stage == "final":
            return
        if translation_stage == "final":
            row.final_failed = True
        elif translation_stage == "draft":
            row.local_failed = True
        self._update_stage_label(row, translation_provider, translation_model)
        if row.stage_label is not None and error:
            row.stage_label.setToolTip(error)
        if overall_failed:
            row.translation_failed = True
            row.translation.clear_presentation()
            row.translation.setText(tr("（翻译未完成）"))
            row.stage_label.setText(tr("本地与 Codex 翻译均未完成"))
            if row.confirmed_revision >= revision:
                self._complete_row(segment_id, row)
        elif row.translation_stage == "draft" and row.final_failed and row.confirmed_revision >= revision:
            self._complete_row(segment_id, row)

    def finalize(self, segment_id: Hashable, revision: int) -> None:
        row = self._rows.get(segment_id)
        if row is None or row.cancelled or revision != row.revision:
            return
        row.confirmed_revision = max(row.confirmed_revision, revision)
        if row.translation_stage == "final" and not row.translation_failed:
            self._update_stage_label(row)
        if row.translation_stage == "final" and row.translation.presentation is not None:
            pending = row.translation.presentation
            if pending.get("provisional"):
                confirmed_at = time.monotonic()
                pending["source_confirmed_at"] = confirmed_at
                pending["source_confirmation_wait_seconds"] = max(0.0, confirmed_at - pending["ui_updated_at"])
                pending["provisional"] = False
            row.translation.update()
        # Confirmation can arrive before or after its queued translation.
        if row.translated_revision >= revision and (row.translation_stage != "draft" or row.final_failed):
            if not row.completed:
                self._completion_order.append(segment_id)
            row.completed = True
            row.translation.setStyleSheet("color: #c4cddb;" if row.translation_stage == "draft" else "color: #f4f7fc;")
            self._prune_completed()

    def cancel_pending(self) -> None:
        for key, row in self._rows.items():
            row.translation.clear_presentation()
            if row.completed:
                continue
            row.translation.setText(tr("已暂停，此片段已取消"))
            row.translation.setStyleSheet("color: #ffd08d;")
            row.completed = True
            row.cancelled = True
            self._completion_order.append(key)
        # Keep existing completed captions visible at pause. History is trimmed
        # when a subsequent session produces a new row or translation.
        self.level.setValue(0)

    def _prune_completed(self) -> None:
        # A late translation remains in the last five completed results instead
        # of being immediately removed because its audio was captured earlier.
        while len(self._completion_order) > 5:
            key = self._completion_order.popleft()
            row = self._rows.pop(key)
            row.translation.clear_presentation()
            self.rows_layout.removeWidget(row.widget)
            row.widget.deleteLater()

    def _scroll_to_bottom(self) -> None:
        # The range updates after layout, so following rangeChanged handles wraps.
        bar = self.scroll.verticalScrollBar()
        bar.setValue(bar.maximum())

    def set_font_size(self, size: int) -> None:
        self._font_size = size
        for row in self._rows.values():
            self._apply_row_font(row)

    def _apply_row_font(self, row: _SubtitleRow) -> None:
        row.original.setFont(QFont(_FONT_FAMILY, max(11, round(self._font_size * 0.58))))
        row.translation.setFont(QFont(_FONT_FAMILY, self._font_size))


class MainWindow(QMainWindow):
    pause_toggled = Signal(bool)
    settings_requested = Signal()
    browser_requested = Signal()
    audio_processing_requested = Signal()
    models_requested = Signal()
    specialized_models_requested = Signal()
    subtitle_presented = Signal(dict)
    closing = Signal()

    def __init__(
        self,
        config: AppConfig | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._paused = False
        self._closing_emitted = False
        self._notices: deque[str] = deque(maxlen=2)
        self._presentation_session: int | None = None
        self._latest_final_latency: dict | None = None
        self._latest_draft_latency: dict | None = None
        self._danmaku_status_label = None
        self.setWindowTitle(APP_NAME)
        self.setFont(QFont(_FONT_FAMILY, 10))
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        self.setMinimumSize(610, 530)
        self.resize(round(self.minimumHeight() * 21 / 9), self.minimumHeight())
        self.setStyleSheet(
            "QMainWindow, QWidget { background: #121b29; color: #e8eef7; }"
            "QFrame#sourcePanel { background: #1c293c; border-radius: 9px; }"
            "QFrame#sourcePanel QWidget { background: transparent; }"
            "QPushButton { background: #293c56; border: 1px solid #405779;"
            " border-radius: 5px; padding: 6px 12px; }"
            "QPushButton:hover { background: #365071; }"
            "QPushButton:disabled { color: #8492a7; }"
            "QSpinBox { background: #24354c; padding: 3px; }"
            "QProgressBar { background: #32445e; border: none; border-radius: 3px; }"
            "QProgressBar::chunk { background: #67c6a8; border-radius: 3px; }"
            "QScrollBar:vertical { width: 8px; background: transparent; }"
            "QScrollBar::handle:vertical { background: #526781; min-height: 22px; }"
        )

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(9)

        header = QHBoxLayout()
        title = _plain_label(APP_NAME)
        title.setFont(QFont(_FONT_FAMILY, 14, QFont.Weight.Bold))
        title.setToolTip(tr("拖动窗口标题栏可以移动位置"))
        header.addWidget(title)
        header.addStretch()
        self.session_state = _plain_label(tr("准备中"))
        self.session_state.setStyleSheet("color: #b6c8de;")
        header.addWidget(self.session_state)
        layout.addLayout(header)

        controls = QHBoxLayout()
        self.pause_button = QPushButton(tr("暂停"))
        self.pause_button.setToolTip(tr("暂停或继续两路声音处理"))
        self.pause_button.clicked.connect(self._toggle_pause)
        controls.addWidget(self.pause_button)
        settings_button = QPushButton(tr("更改设置"))
        settings_button.clicked.connect(lambda: self.settings_requested.emit())
        controls.addWidget(settings_button)
        models_button = QPushButton(tr("模型管理"))
        models_button.clicked.connect(lambda: self.models_requested.emit())
        controls.addWidget(models_button)
        browser_button = QPushButton(tr("直播浏览器"))
        browser_button.clicked.connect(lambda: self.browser_requested.emit())
        controls.addWidget(browser_button)
        self.audio_processing_button = QPushButton(tr("音源与语音处理"))
        self.audio_processing_button.clicked.connect(self.audio_processing_requested.emit)
        controls.addWidget(self.audio_processing_button)
        self.topmost_checkbox = QCheckBox(tr("置顶"))
        self.topmost_checkbox.setChecked(True)
        self.topmost_checkbox.toggled.connect(self._set_topmost)
        controls.addWidget(self.topmost_checkbox)
        controls.addStretch()
        controls.addWidget(_plain_label(tr("译文字号")))
        self.font_size = QSpinBox()
        self.font_size.setRange(16, 48)
        self.font_size.setValue(16)
        self.font_size.valueChanged.connect(self._set_font_size)
        controls.addWidget(self.font_size)
        controls.addWidget(_plain_label(tr("透明度")))
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(55, 100)
        self.opacity_slider.setFixedWidth(75)
        self.opacity_slider.setToolTip(tr("调整整个字幕窗口的透明度"))
        self.opacity_slider.valueChanged.connect(lambda value: self.setWindowOpacity(value / 100))
        self.opacity_slider.setValue(66)
        controls.addWidget(self.opacity_slider)
        exit_button = QPushButton(tr("退出"))
        exit_button.clicked.connect(self.close)
        controls.addWidget(exit_button)
        layout.addLayout(controls)

        feature_controls = QHBoxLayout()
        self.specialized_button = QPushButton(tr("专用模型"))
        self.specialized_button.setToolTip(tr("查看专用模型的本地数据、模型对照和训练准备；当前暂不开放使用"))
        self.specialized_button.clicked.connect(lambda: self.specialized_models_requested.emit())
        feature_controls.addWidget(self.specialized_button)
        specialized_status = _plain_label(tr("专用模型：独立数据与训练准备 · 暂未开放使用"))
        specialized_status.setStyleSheet("color: #aebdd1;")
        feature_controls.addWidget(specialized_status)
        feature_controls.addStretch()
        layout.addLayout(feature_controls)

        self.models_label = _plain_label(tr("模型：未加载"), wrap=True)
        self.models_label.setFont(QFont(_FONT_FAMILY, 9))
        self.models_label.setStyleSheet("color: #aebdd1;")
        layout.addWidget(self.models_label)

        self.latency_label = _plain_label(wrap=True)
        self.latency_label.setFont(QFont(_FONT_FAMILY, 9))
        self.latency_label.setStyleSheet("color: #92c9fb;")
        self.latency_label.setToolTip(tr(_LATENCY_TOOLTIP))
        self.latency_label.hide()
        layout.addWidget(self.latency_label)


        self._source_panels = {
            "microphone": _SourcePanel(tr("麦克风")),
            "playback": _SourcePanel(tr("电脑声音")),
        }
        for panel in self._source_panels.values():
            panel.subtitle_presented.connect(self._subtitle_presented)
            panel.scroll.verticalScrollBar().rangeChanged.connect(
                lambda _minimum, maximum, bar=panel.scroll.verticalScrollBar(): bar.setValue(maximum)
            )
            layout.addWidget(panel, 1)

        self.idle_label = _plain_label(tr("待机中\n通过“更改设置”开启声音来源。"), wrap=True)
        self.idle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.idle_label.setFont(QFont(_FONT_FAMILY, 12))
        self.idle_label.setStyleSheet("color: #b4bfd0;")
        self.idle_label.hide()
        layout.addWidget(self.idle_label, 1)

        self.notice_label = _plain_label(wrap=True)
        self.notice_label.setStyleSheet("color: #ffd08d;")
        self.notice_label.setMaximumHeight(65)
        self.notice_label.hide()
        layout.addWidget(self.notice_label)

        if config is not None:
            self.update_config(config)
        else:
            for panel in self._source_panels.values():
                panel.enabled = False
                panel.hide()
            self.idle_label.setText(tr("请先保存首次设置。"))
            self.idle_label.show()

    def _panel(self, source_id: str) -> _SourcePanel | None:
        source = _SOURCE_ALIASES.get(source_id)
        return self._source_panels.get(source) if source is not None else None

    def _toggle_pause(self) -> None:
        self._paused = not self._paused
        self.pause_button.setText(tr("继续") if self._paused else tr("暂停"))
        self.pause_toggled.emit(self._paused)

    def _set_topmost(self, enabled: bool) -> None:
        visible = self.isVisible()
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, enabled)
        if visible:
            self.show()

    def _set_font_size(self, size: int) -> None:
        for panel in self._source_panels.values():
            panel.set_font_size(size)

    def set_session_state(self, state: str) -> None:
        labels = {
            "starting": tr("准备中"),
            "loading": tr("正在准备识别和翻译…"),
            "running": tr("正在听"),
            "paused": tr("已暂停"),
            "stopped": tr("已停止"),
            "error": tr("处理遇到问题"),
        }
        self.session_state.setText(tr(labels.get(state, state)))
        if state in ("paused", "running", "stopped"):
            self._paused = state != "running"
            self.pause_button.setText(tr("继续") if self._paused else tr("暂停"))

    def set_model_state(self, state: str, models: dict | None = None) -> None:
        summary = loaded_models_summary(models) if models else tr("模型：未加载")
        prefix = {"loading": tr("模型加载中…"), "switching": tr("模型切换中…"),
                  "failed": tr("模型加载失败"), "ready": "", "not_loaded": ""}.get(state, state)
        self.models_label.setText((prefix + "  ·  " if prefix else "") + summary)

    def set_source_status(self, source_id: str, text: str) -> None:
        panel = self._panel(source_id)
        if panel is not None:
            panel.status_label.setText(tr(text) if panel.enabled else tr("已关闭"))

    def set_level(self, source_id: str, level: float) -> None:
        panel = self._panel(source_id)
        if panel is None:
            return
        value = float(level)
        if not math.isfinite(value) or not panel.enabled:
            value = 0
        panel.level.setValue(round(max(0.0, min(1.0, value)) * 100))

    def set_original(self, source_id: str, text: str, segment_id: Hashable, *,
                     revision: int = 1, provisional: bool = False, uncertain: bool = False,
                     context_generation: int | None = None) -> None:
        panel = self._panel(source_id)
        if panel is not None:
            panel.set_original(text, segment_id, revision=revision, provisional=provisional, uncertain=uncertain,
                               context_generation=context_generation)

    def set_translation(self, source_id: str, text: str, segment_id: Hashable, *,
                        revision: int = 1, provisional: bool = False, translation_stage: str = "",
                        translation_provider: str = "", translation_model: str = "",
                        context_generation: int | None = None, final_failed: bool = False,
                        failed: bool = False, overall_failed: bool = False,
                        presentation: dict | None = None) -> None:
        panel = self._panel(source_id)
        if panel is not None:
            if (presentation is not None and self._presentation_session is not None
                    and presentation.get("session") != self._presentation_session):
                return
            panel.set_translation(text, segment_id, revision=revision, provisional=provisional,
                                  translation_stage=translation_stage, translation_provider=translation_provider,
                                  translation_model=translation_model, context_generation=context_generation,
                                  final_failed=final_failed, failed=failed, overall_failed=overall_failed,
                                  presentation=presentation)

    def set_presentation_session(self, session: int) -> None:
        self._presentation_session = session
        self._latest_final_latency = self._latest_draft_latency = None
        self.latency_label.hide()
        for panel in self._source_panels.values():
            for row in panel._rows.values():
                row.translation.clear_presentation()

    def _subtitle_presented(self, event: dict) -> None:
        if self._presentation_session is not None and event.get("session") != self._presentation_session:
            return
        self.subtitle_presented.emit(event)

    def set_latency(self, metrics: dict) -> None:
        if not metrics.get("valid"):
            return
        stage = metrics.get("stage", metrics.get("translation_stage"))
        if stage == "final":
            self._latest_final_latency = dict(metrics)
        elif stage == "draft":
            self._latest_draft_latency = dict(metrics)
        values = []
        if self._latest_final_latency is not None:
            final = self._latest_final_latency
            values.append(tr("最近终稿：全程 {0:.2f} 秒 · 句尾等待 {1:.2f} 秒",
                             final["end_to_end_seconds"], final["post_audio_seconds"]))
        if self._latest_draft_latency is not None:
            values.append(tr("最近暂译：全程 {0:.2f} 秒", self._latest_draft_latency["end_to_end_seconds"]))
        if values:
            self.latency_label.setText("  |  ".join(values))
            detail = [tr(_LATENCY_TOOLTIP)]
            if self._latest_final_latency is not None:
                final = self._latest_final_latency
                recognition = final.get("recognition_to_display_seconds")
                if isinstance(recognition, (int, float)) and math.isfinite(recognition):
                    detail.append(tr("识别开始到终稿：{0:.2f} 秒", recognition))
                for key, label in _LATENCY_BREAKDOWN_LABELS:
                    duration = final.get("breakdown", {}).get(key)
                    if isinstance(duration, (int, float)) and math.isfinite(duration):
                        detail.append(tr("{0}：{1:.2f} 秒", tr(label), duration))
            self.latency_label.setToolTip("\n".join(detail))
            self.latency_label.show()

    def set_translation_status(self, source_id: str, segment_id: Hashable, **options) -> None:
        panel = self._panel(source_id)
        if panel is not None:
            panel.set_translation_status(segment_id, **options)

    def set_danmaku_status(self, value: dict) -> None:
        if self._danmaku_status_label is None:
            self._danmaku_status_label = _plain_label("")
            self._danmaku_status_label.setStyleSheet("color: #a9c7e5;")
            self.centralWidget().layout().insertWidget(3, self._danmaku_status_label)
        state = str(value.get("state", "disabled"))
        labels = {"disabled": "已关闭", "idle": "等待终稿", "ready": "等待终稿",
                  "empty": "等待终稿", "waiting": "等待已登录的目标直播页",
                  "waiting_merge": "短句等待下一句合并", "checking_login": "正在核实账号登录状态",
                  "sending": "正在发送", "checking": "正在核验", "verifying": "正在核验",
                  "waiting_browser": "等待浏览器连接", "waiting_connection": "等待浏览器连接",
                  "waiting_login": "等待登录", "login_required": "等待登录",
                  "queued_room_mismatch": "待发消息属于其他直播间，请核对设置",
                  "storage_error": "队列保存失败，请检查本机存储",
                  "filter_error": "过滤名单异常，待发消息保留",
                  "awaiting_verification": "接口回执已接收，尚未确认页面收到",
                  "auto_verifying": "发送结果未知，正在自动只读核验",
                  "auto_rechecking": "连续发送失败，正在自动复检账号和页面",
                  "auto_retry": "正在自动恢复当前条，保留队列顺序",
                  "ai_analyzing": "正在请求 Codex 分析失败正文",
                  "ai_repair_ready": "修改后正文已通过本地检查，等待独立测试验证",
                  "ai_retrying": "正在按发送间隔重试测试正文",
                  "waiting_validation": "修改后正文等待独立测试房间验证，尚未生成规则",
                  "validating": "正在独立测试房间验证修改后正文",
                  "validation_failed": "测试正文未通过，保留记录且不生成规则",
                  "waiting_validation_browser": "等待手动打开已登录的验证直播间",
                  "validation_storage_error": "验证记录暂存失败 自动重试保存",
                  "validation_passed": "独立测试房间验证通过，正式房间效果未确认",
                  "validation_idle": "等待需要验证的修改后正文",
                  "validation_configuration_changed": "待核验请求属于原测试配置，请恢复对应房间设置",
                  "failure_transferred": "失败已移入独立处理，后续队列继续发送",
                  "saving_failed_record": "失败记录保存异常，正在重试写入",
                  "sent_page_visible": "直播页面已收到弹幕",
                  "platform_rejected": "平台拒绝，消息保留等待核实",
                  "platform_f": "平台返回 f，按发送间隔自动重试当前条",
                  "waiting_f_retry": "平台返回 f，等待自动重试当前条",
                  "retrying_f": "正在自动重试当前条，保留队列顺序",
                  "result_unknown": "发送结果未知，消息保留等待核实",
                  "cleared": "已清理未提交消息", "queue_cleared": "已清理未提交消息",
                  "cooldown": "发送间隔", "blocked": "待核实，队列保留"}
        message = tr("弹幕推送：{0} · 待发 {1} · 页面可见 {2}",
            tr(labels.get(state, state)), value.get("queued", 0), value.get("page_visible", 0))
        f_retries = value.get("f_retries", 0)
        if type(f_retries) is int and f_retries > 0:
            message += tr(" · f 自动重试累计 {0} 次", f_retries)
        rule_count, candidate_count = value.get("filter_rule_count"), value.get("filter_candidate_count")
        if type(rule_count) is int and rule_count >= 0 and type(candidate_count) is int and candidate_count >= 0:
            message += tr(" · 过滤规则 {0} · 未处理失败 {1}", rule_count, candidate_count)
        resolved_count = value.get("filter_resolved_count")
        if type(resolved_count) is int and resolved_count > 0:
            message += tr(" · 已处理历史 {0}", resolved_count)
        deferred = value.get("deferred_analysis")
        if type(deferred) is int and deferred >= 0:
            message += tr(" · 累计移入独立处理 {0} 段", deferred)
        if value.get("recovery_action"):
            message += tr(" · 处置 {0}", recovery_action_text(value["recovery_action"]))
        if value.get("filter_error") and state != "filter_error":
            message += tr(" · 过滤名单异常，待发消息保留")
        if value.get("filter_candidate_error"):
            message += tr(" · 失败候选保存失败，请检查本地文件")
        remaining = value.get("cooldown_remaining", value.get("wait_seconds", 0))
        if (value.get("f_retry_active", False) or state in {
                "platform_f", "waiting_f_retry", "retrying_f", "auto_rechecking", "auto_retry", "ai_retrying"}) and type(remaining) in {
                    int, float} and remaining > 0:
            message += tr(" · 还需等待 {0:.1f} 秒", remaining)
        self._danmaku_status_label.setText(message)
        self._danmaku_status_label.setVisible(value.get("enabled", state != "disabled") or bool(value.get("queued")) or state in {
            "ai_analyzing", "ai_repair_ready", "ai_retrying", "waiting_validation", "validating", "validation_failed",
            "waiting_validation_browser", "validation_storage_error", "validation_passed", "validation_configuration_changed",
            "failure_transferred", "saving_failed_record"})

    def finalize_subtitle(self, source_id: str, segment_id: Hashable, revision: int) -> None:
        panel = self._panel(source_id)
        if panel is not None:
            panel.finalize(segment_id, revision)

    def update_config(self, config: AppConfig) -> None:
        sources = {"microphone": config.microphone, "playback": config.playback}
        for source_id, source in sources.items():
            panel = self._source_panels[source_id]
            panel.update_source(source)
            panel.setVisible(source.enabled)
        self.idle_label.setText(tr("待机中\n通过“更改设置”开启声音来源。"))
        self.idle_label.setVisible(not any(source.enabled for source in sources.values()))

    def cancel_pending(self) -> None:
        for panel in self._source_panels.values():
            panel.cancel_pending()

    def append_notice(self, text: str) -> None:
        if not text.strip():
            return
        self._notices.append(tr(text.strip()))
        self.notice_label.setText("\n".join(self._notices))
        self.notice_label.show()

    def closeEvent(self, event: QCloseEvent) -> None:
        guard = getattr(self, "close_guard", None)
        if guard is not None and not guard():
            event.ignore()
            return
        if not self._closing_emitted:
            self._closing_emitted = True
            self.closing.emit()
        event.accept()
