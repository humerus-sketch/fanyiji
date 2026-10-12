"""Inspect the isolated training corpus; no student deployment controls exist here."""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, QUrl
from .i18n import tr
from PySide6.QtGui import QDesktopServices, QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)


def _label(text: str = "") -> QLabel:
    label = QLabel(text)
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setWordWrap(True)
    return label


class SpecializedModelDialog(QDialog):
    """The service owns provenance, storage and GPU coordination, not this view."""

    def __init__(self, service, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.service = service
        self._samples: list[dict] = []
        self._signature = ""
        self._review_context: tuple | None = None
        self._summary: dict = {}
        self._cancellation_requested = False
        self.setObjectName("specializedModelDialog")
        self.setWindowTitle(tr("专用本地模型 · 数据与训练准备"))
        self.setFont(QFont("Microsoft YaHei UI", 10))
        self.resize(960, 800)
        self.setMinimumSize(780, 660)
        self.setStyleSheet(
            "QWidget { color: #182337; background: transparent; }"
            "QDialog#specializedModelDialog { background: #ffffff; color: #182337; }"
            "QPushButton { background: #eef3fa; border: 1px solid #c4cfde;"
            " border-radius: 5px; padding: 6px 10px; }"
            "QPushButton:disabled { color: #8794a5; background: #f1f3f6; }"
            "QListWidget, QPlainTextEdit, QComboBox { background: #ffffff; }"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(10)

        deployment = QHBoxLayout()
        self.phase_label = _label(tr("专用模型尚未开放使用"))
        self.phase_label.setStyleSheet("font-weight: bold; color: #3b526e;")
        deployment.addWidget(self.phase_label, 1)
        self.use_button = QPushButton(tr("使用专用模型（未开放）"))
        self.use_button.setEnabled(False)
        self.use_button.setToolTip(tr("需要完成独立评测并另行开放；现有模型和训练文件不会自动启用此按钮"))
        deployment.addWidget(self.use_button)
        layout.addLayout(deployment)
        layout.addWidget(_label(
            tr("推荐基底：Hy-MT2 1.8B；先对照本地模型，再准备 QLoRA 微调。"
            "训练环境和原始训练权重需要单独准备，已有 GGUF 用于翻译对照。"
            "本页不会切换当前正在使用的翻译模型。")))

        self.summary_label = _label()
        layout.addWidget(self.summary_label)
        self.collection_enabled = QCheckBox(tr("积累专用模型数据（本地单独存放）"))
        self.collection_enabled.toggled.connect(self._set_collection_enabled)
        layout.addWidget(self.collection_enabled)
        self.path_label = _label(tr('数据目录：{0}', Path(service.root)))
        self.path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.path_label)

        actions = QHBoxLayout()
        self.compare_button = QPushButton(tr("对照本地模型（自动暂停字幕）"))
        self.compare_button.setToolTip(tr("按顺序加载已下载的本地翻译模型，避免与语音识别争抢显存；结束后恢复原字幕状态"))
        self.compare_button.clicked.connect(self._compare)
        actions.addWidget(self.compare_button)
        self.cancel_button = QPushButton(tr("取消对照"))
        self.cancel_button.clicked.connect(self._cancel_compare)
        actions.addWidget(self.cancel_button)
        self.export_button = QPushButton(tr("导出训练集"))
        self.export_button.clicked.connect(self._export)
        actions.addWidget(self.export_button)
        self.prepare_button = QPushButton(tr("准备训练方案"))
        self.prepare_button.clicked.connect(self._prepare)
        self.prepare_button.setVisible(callable(getattr(service, "prepare_training", None)))
        actions.addWidget(self.prepare_button)
        self.open_directory_button = QPushButton(tr("打开数据目录"))
        self.open_directory_button.clicked.connect(self._open_directory)
        actions.addWidget(self.open_directory_button)
        layout.addLayout(actions)

        layout.addWidget(_label(
            tr("Codex 纠错仅作隔离对照，不自动进入训练集。"
            "本地候选需要核对完整原话、译文和素材使用权；断句、听错和未完成字幕不能当作正确答案。")))
        self.result_label = _label()
        self.result_label.setStyleSheet("color: #785713;")
        self.result_label.setMaximumHeight(96)
        layout.addWidget(self.result_label)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.samples_list = QListWidget()
        self.samples_list.setMinimumWidth(230)
        self.samples_list.currentRowChanged.connect(self._sample_changed)
        splitter.addWidget(self.samples_list)
        detail = QWidget()
        detail_layout = QVBoxLayout(detail)
        detail_layout.setContentsMargins(10, 0, 0, 0)
        self.sample_label = _label(tr("选择样本查看原文和候选译文"))
        detail_layout.addWidget(self.sample_label)
        detail_layout.addWidget(_label(tr("识别原文（需独立听音核对）")))
        self.source_text = QPlainTextEdit()
        self.source_text.setReadOnly(True)
        self.source_text.setMaximumHeight(120)
        detail_layout.addWidget(self.source_text)
        audio_controls = QHBoxLayout()
        self.audio_selector = QComboBox()
        self.audio_selector.setToolTip(tr("合并语句可能对应多个录音片段，请逐段听音核对完整原文"))
        self.audio_selector.currentIndexChanged.connect(self._audio_changed)
        audio_controls.addWidget(self.audio_selector, 1)
        self.audio_button = QPushButton(tr("播放样本音频"))
        self.audio_button.clicked.connect(self._open_audio)
        audio_controls.addWidget(self.audio_button)
        detail_layout.addLayout(audio_controls)
        self.candidates_combo = QComboBox()
        self.candidates_combo.currentIndexChanged.connect(self._candidate_changed)
        detail_layout.addWidget(self.candidates_combo)
        self.target_text = QPlainTextEdit()
        self.target_text.setReadOnly(True)
        detail_layout.addWidget(self.target_text, 1)
        self.candidate_label = _label()
        detail_layout.addWidget(self.candidate_label)
        self.source_verified = QCheckBox(tr("已独立听音，确认原文完整且准确"))
        self.source_verified.toggled.connect(self._update_review_button)
        detail_layout.addWidget(self.source_verified)
        self.rights_confirmed = QCheckBox(tr("已确认原始素材及本地译文允许用于训练"))
        self.rights_confirmed.toggled.connect(self._update_review_button)
        detail_layout.addWidget(self.rights_confirmed)
        self.review_button = QPushButton(tr("确认本地译文准确"))
        self.review_button.clicked.connect(self._review)
        detail_layout.addWidget(self.review_button)
        splitter.addWidget(detail)
        splitter.setSizes([300, 600])
        layout.addWidget(splitter, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.button(QDialogButtonBox.StandardButton.Close).setText(tr("关闭"))
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.refresh_timer = QTimer(self)
        self.refresh_timer.setInterval(1000)
        self.refresh_timer.timeout.connect(self.refresh)
        self.refresh()
        self.refresh_timer.start()

    def refresh(self) -> None:
        try:
            summary = self.service.summary()
            samples = self.service.list_samples(limit=100)
        except Exception as exc:
            self.result_label.setText(tr('读取专用模型数据失败：{0}', exc))
            return
        self._summary = summary
        counts = summary.get("counts", summary)
        self.summary_label.setText(
            tr('样本 {0} · 音频 {1} · 本地候选 {2} · Codex 隔离候选 {3} · 已核对 {4} · 可训练 {5}', counts.get('samples', 0), counts.get('audio_files', 0), counts.get('local_candidates', 0), counts.get('codex_candidates', 0), counts.get('reviewed', 0), counts.get('trainable', 0)))
        self.collection_enabled.blockSignals(True)
        self.collection_enabled.setChecked(bool(summary.get("enabled", False)))
        self.collection_enabled.blockSignals(False)
        busy = bool(summary.get("busy", False))
        if self._cancellation_requested and not busy:
            self._cancellation_requested = False
            self.result_label.setText(tr("对照已取消；原字幕状态将自动恢复。"))
        self.compare_button.setEnabled(not busy)
        self.cancel_button.setVisible(busy)
        self.cancel_button.setEnabled(busy and not self._cancellation_requested)
        self.export_button.setEnabled(not busy)
        self.prepare_button.setEnabled(not busy)
        status = tr(summary.get("status") or summary.get("phase") or "数据积累")
        self.phase_label.setText(tr('{0} · 暂未开放使用', status))
        # Deployment is deliberately unconditional: a manifest cannot unlock it.
        self.use_button.setEnabled(False)
        if summary.get("error"):
            self.result_label.setText(tr(str(summary["error"])))

        signature = json.dumps(samples, sort_keys=True, ensure_ascii=False, default=str)
        if signature != self._signature:
            previous = self._selected_sample()
            selected_id = previous.get("id") if previous else None
            previous_candidate = self._selected_candidate()
            candidate_id = previous_candidate.get("id") if previous_candidate else None
            self._signature = signature
            self._samples = samples
            self.samples_list.blockSignals(True)
            self.samples_list.clear()
            selected_row = -1
            for index, sample in enumerate(samples):
                source = str(sample.get("source", "")).replace("\n", " ")
                direction = {"ja_to_zh": tr("日→中"), "zh_to_ja": tr("中→日")}.get(sample.get("direction"), "")
                channel = {"playback": tr("电脑"), "system": tr("电脑"), "microphone": tr("话筒"), "mic": tr("话筒")}.get(
                    sample.get("channel"), str(sample.get("channel", "")))
                self.samples_list.addItem(f"{channel} {direction} · {source[:64]}")
                if sample.get("id") == selected_id:
                    selected_row = index
            if selected_row < 0 and samples:
                selected_row = 0
            self.samples_list.setCurrentRow(selected_row)
            self.samples_list.blockSignals(False)
            self._render_sample(candidate_id=candidate_id)
        self._update_review_button()

    def _selected_sample(self) -> dict | None:
        row = self.samples_list.currentRow()
        return self._samples[row] if 0 <= row < len(self._samples) else None

    def _selected_candidate(self) -> dict | None:
        candidate = self.candidates_combo.currentData()
        return candidate if isinstance(candidate, dict) else None

    def _sample_changed(self, _row: int) -> None:
        self._render_sample()

    def _render_sample(self, *, candidate_id=None) -> None:
        sample = self._selected_sample()
        self.candidates_combo.blockSignals(True)
        self.candidates_combo.clear()
        self.source_text.setPlainText(str(sample.get("source", "")) if sample else "")
        if sample:
            marks = [mark for key, mark in (("provisional", tr("未完成")), ("uncertain", tr("识别存疑")),
                                            ("failed", tr("翻译失败"))) if sample.get(key)]
            self.sample_label.setText(
                tr('样本 {0} · 版本 {1}', sample.get('segment_id', sample.get('id', '')), sample.get('revision', 1)) + (" · " + "、".join(marks) if marks else ""))
            for candidate in sample.get("candidates", []):
                origin = candidate.get("origin")
                label = tr("Codex（隔离，仅对照）") if origin == "codex" else tr("本地候选") if origin == "local" else tr("未知来源（隔离）")
                reviewed = tr(" · 已核对") if candidate.get("reviewed") else ""
                self.candidates_combo.addItem(f"{label} · {candidate.get('model', tr('模型未知'))}{reviewed}", candidate)
            if candidate_id is not None:
                for index in range(self.candidates_combo.count()):
                    candidate = self.candidates_combo.itemData(index)
                    if candidate.get("id") == candidate_id:
                        self.candidates_combo.setCurrentIndex(index)
                        break
        else:
            self.sample_label.setText(tr("尚无样本；启用数据积累后会记录实际识别结果"))
        self.candidates_combo.blockSignals(False)
        previous_audio = self.audio_selector.currentData()
        audio_paths = self._sample_audio_paths()
        self.audio_selector.blockSignals(True)
        self.audio_selector.clear()
        for index, path in enumerate(audio_paths):
            self.audio_selector.addItem(tr('音频片段 {0} / {1}', index + 1, len(audio_paths)), str(path))
        selected_audio = self.audio_selector.findData(previous_audio)
        if selected_audio >= 0:
            self.audio_selector.setCurrentIndex(selected_audio)
        self.audio_selector.blockSignals(False)
        self.audio_selector.setEnabled(bool(audio_paths))
        self._audio_changed()
        self._candidate_changed()

    def _candidate_changed(self, _index: int = -1) -> None:
        sample = self._selected_sample()
        candidate = self._selected_candidate()
        self.target_text.setPlainText(str(candidate.get("text", "")) if candidate else "")
        review_context = (
            sample.get("id"), sample.get("revision"), sample.get("source"),
            sample.get("provisional"), sample.get("uncertain"), sample.get("failed"),
            candidate.get("id"), candidate.get("origin"), candidate.get("text"),
            candidate.get("reviewed"), candidate.get("training_allowed"),
        ) if sample and candidate else None
        # Incoming recordings must not clear checks on an unchanged selected pair.
        if review_context != self._review_context:
            self.source_verified.setChecked(False)
            self.rights_confirmed.setChecked(False)
        self._review_context = review_context
        if candidate is None:
            text = tr("尚无候选译文；可启动本地模型对照。")
        elif candidate.get("origin") == "codex":
            text = tr("Codex 纠错保留原始来源，隔离存放，仅供对照；当前不能核准或导出用于训练。")
        elif candidate.get("origin") != "local":
            text = tr("候选来源未验证，当前不能核准用于训练。")
        elif candidate.get("reviewed"):
            text = tr("已完成本地候选人工核对。") + (
                tr("符合当前训练导出条件。") if candidate.get("training_allowed") else tr("当前未选为训练译文，可重新核对后确认该候选。"))
        else:
            text = tr("本地模型结果尚未核对；请独立听音并确认本地译文准确。")
        self.candidate_label.setText(text)
        self._update_review_button()

    def _can_review(self) -> bool:
        sample = self._selected_sample()
        candidate = self._selected_candidate()
        return bool(sample and candidate and candidate.get("origin") == "local"
                    and str(candidate.get("text", "")).strip()
                    and not (candidate.get("reviewed") and candidate.get("training_allowed"))
                    and not any(sample.get(key) for key in ("provisional", "uncertain", "failed"))
                    and not self._summary.get("busy", False))

    def _update_review_button(self) -> None:
        allowed = self._can_review()
        self.source_verified.setEnabled(allowed)
        self.rights_confirmed.setEnabled(allowed)
        self.review_button.setEnabled(allowed and self.source_verified.isChecked() and self.rights_confirmed.isChecked())

    def _set_collection_enabled(self, enabled: bool) -> None:
        try:
            self.service.set_enabled(enabled)
            self.result_label.setText(tr("已开启本地数据积累。") if enabled else tr("已暂停数据积累；字幕声音采集保持当前设置。"))
        except Exception as exc:
            self.result_label.setText(tr('更改数据积累设置失败：{0}', exc))
        self.refresh()

    def _compare(self) -> None:
        try:
            started = self.service.compare_local()
            self.result_label.setText(tr("本地模型对照已开始，完成后恢复原字幕状态。") if started else tr("本地模型对照未开始，请检查状态。"))
        except Exception as exc:
            self.result_label.setText(tr('无法开始本地对照：{0}', exc))
        self.refresh()

    def _cancel_compare(self) -> None:
        if not self._summary.get("busy", False) or self._cancellation_requested:
            return
        try:
            self.service.cancel_comparison()
            self._cancellation_requested = True
            self.result_label.setText(tr("正在取消对照，等待本地模型释放资源后恢复字幕。"))
        except Exception as exc:
            self.result_label.setText(tr('取消对照未完成：{0}', exc))
        self.refresh()

    def _review(self) -> None:
        if not self.review_button.isEnabled() or not self._can_review():
            return
        sample, candidate = self._selected_sample(), self._selected_candidate()
        try:
            self.service.review_candidate(sample["id"], candidate["id"],
                                          source_verified=self.source_verified.isChecked(),
                                          rights_confirmed=self.rights_confirmed.isChecked())
            self.result_label.setText(tr("已记录本地译文核对结果；训练导出仍会检查完整性和来源。"))
        except Exception as exc:
            self.result_label.setText(tr('核对结果未保存：{0}', exc))
        self.refresh()

    def _export(self) -> None:
        try:
            result = self.service.export_dataset()
            self.result_label.setText(tr("训练集已导出：") + self._result_text(result))
        except Exception as exc:
            self.result_label.setText(tr('训练集尚未导出：{0}', exc))
        self.refresh()

    def _prepare(self) -> None:
        prepare = getattr(self.service, "prepare_training", None)
        if not callable(prepare):
            return
        try:
            result = prepare()
            self.result_label.setText(tr("训练准备结果：") + self._result_text(result))
        except Exception as exc:
            self.result_label.setText(tr('训练准备尚未完成：{0}', exc))
        self.refresh()

    @staticmethod
    def _result_text(result) -> str:
        if isinstance(result, dict):
            parts = []
            if "train_rows" in result:
                parts.append(tr('训练 {0} 条', result['train_rows']))
            if "validation_rows" in result:
                parts.append(tr('验证 {0} 条', result['validation_rows']))
            if "training_executed" in result:
                parts.append(tr("训练已执行") if result["training_executed"] else tr("训练尚未执行"))
            elif "trained" in result:
                parts.append(tr("已有训练产物") if result["trained"] else tr("尚未训练"))
            directory = result.get("directory") or result.get("plan") or result.get("manifest")
            if directory:
                parts.append(tr('目录：{0}', directory))
            blockers = result.get("blockers")
            if isinstance(blockers, list) and blockers:
                shown = "；".join(str(item)[:90] for item in blockers[:2])
                suffix = tr('；另有 {0} 项，详见训练说明', len(blockers) - 2) if len(blockers) > 2 else ""
                parts.append(tr("待准备：") + shown + suffix)
            return " · ".join(parts) if parts else " · ".join(
                f"{key}：{str(value)[:150]}" for key, value in list(result.items())[:3])
        return str(result)

    def _open_directory(self) -> None:
        root = Path(self.service.root).resolve()
        if root.is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(root)))
        else:
            self.result_label.setText(tr("数据目录尚未创建，开始积累数据后再打开。"))

    def _sample_audio_paths(self) -> list[Path]:
        sample = self._selected_sample()
        root = Path(self.service.root).resolve()
        paths = []
        for value in sample.get("audio_paths", []) if sample else []:
            path = Path(value)
            if not path.is_absolute():
                path = root / path
            path = path.resolve()
            if (path.is_relative_to(root) and path.is_file()
                    and path.suffix.lower() in {".wav", ".flac", ".mp3", ".ogg"}):
                paths.append(path)
        return paths

    def _audio_path(self) -> Path | None:
        selected = self.audio_selector.currentData()
        for path in self._sample_audio_paths():
            if str(path) == selected:
                return path
        return None

    def _audio_changed(self, _index: int = -1) -> None:
        self.audio_button.setEnabled(self._audio_path() is not None)

    def _open_audio(self) -> None:
        path = self._audio_path()
        if path is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def done(self, result: int) -> None:
        self.refresh_timer.stop()
        super().done(result)
