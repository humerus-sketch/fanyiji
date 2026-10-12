"""Manage local text rules and automatic failure recovery history."""
from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog,
    QDialogButtonBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QPlainTextEdit, QPushButton, QTableWidget, QTableWidgetItem, QTabWidget,
    QVBoxLayout, QWidget)

from .danmaku_filter import DanmakuFilterError
from .i18n import recovery_action_text, recovery_category_text, tr


def _label(text=""):
    label = QLabel(text)
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setWordWrap(True)
    return label


def _table(headers):
    table = QTableWidget(0, len(headers))
    table.setHorizontalHeaderLabels([tr(header) for header in headers])
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setWordWrap(False)
    table.horizontalHeader().setStretchLastSection(True)
    return table


class DanmakuFilterDialog(QDialog):
    def __init__(self, content_filter, room="", parent=None):
        super().__init__(parent)
        self.content_filter = content_filter
        self.room = room
        self._rules = []
        self._candidates = []
        self._rule_id = None
        self._candidate_id = None
        self.setObjectName("danmakuFilterDialog")
        self.setWindowTitle(tr("本地过滤词管理"))
        self.setFont(QFont("Microsoft YaHei UI", 10))
        self.resize(940, 760)
        self.setMinimumSize(720, 640)
        self.setStyleSheet(
            "QWidget { color: #182337; background: transparent; }"
            "QDialog#danmakuFilterDialog { background: #ffffff; }"
            "QPushButton { background: #eef3fa; border: 1px solid #c4cfde;"
            " border-radius: 5px; padding: 6px 10px; }"
            "QPushButton:disabled { color: #8794a5; background: #f1f3f6; }"
            "QTableWidget, QPlainTextEdit, QLineEdit, QComboBox { background: #ffffff; }")
        layout = QVBoxLayout(self)
        layout.addWidget(_label(tr("仅修改发送弹幕的文字，不修改字幕。手工规则由你确认；自动恢复使用 Codex 生成修改后正文，"
            "独立测试成功后才比对原句保存局部规则。")))
        layout.addWidget(_label(tr("所有已保存规则统一适用于全部直播间；失败记录中的房间号仅表示来源和验证记录。")))
        layout.addWidget(_label(tr("实测确认后可选择“从弹幕删除”，仅删除弹幕正文中的原词，字幕终稿保持原文。")))
        layout.addWidget(_label(tr("未验证的测试正文只保存在失败记录中，不创建过滤规则，也不归档；测试成功后生成的规则可编辑或停用。")))
        layout.addWidget(_label(tr("AI局部修正后的页面可见结果仅作为经验规则，不代表该词已被平台证实禁用；可编辑或停用。")))
        layout.addWidget(_label(tr("失败消息移入独立处理记录，后续队列继续发送；记录保留并自动更新处理状态。返回 f 不代表句中存在禁词，"
            "词库候选不会自动生效；AI经验规则也不代表已确认禁词。")))
        self.controls = QWidget(self)
        controls = QVBoxLayout(self.controls)
        controls.setContentsMargins(0, 0, 0, 0)
        toolbar = QHBoxLayout()
        self.enabled = QCheckBox(tr("启用本地过滤名单"))
        toolbar.addWidget(self.enabled)
        self.save_enabled_button = QPushButton(tr("保存名单开关"))
        self.save_enabled_button.clicked.connect(self.save_enabled)
        toolbar.addWidget(self.save_enabled_button)
        self.refresh_button = QPushButton(tr("刷新列表"))
        self.refresh_button.clicked.connect(self.refresh)
        toolbar.addWidget(self.refresh_button)
        toolbar.addStretch()
        controls.addLayout(toolbar)
        self.tabs = QTabWidget()
        self.rules_table = _table(("启用", "原词", "处理结果", "适用直播间", "更新时间（UTC+8）", "规则来源"))
        self.rules_table.itemSelectionChanged.connect(self.select_rule)
        self.tabs.addTab(self.rules_table, tr("已保存词规则"))
        candidates = QWidget()
        candidate_layout = QVBoxLayout(candidates)
        candidate_layout.setContentsMargins(0, 0, 0, 0)
        self.show_history = QCheckBox(tr("显示已处理历史"))
        self.show_history.toggled.connect(self.refresh_candidates)
        candidate_layout.addWidget(self.show_history)
        self.candidate_status = _label()
        candidate_layout.addWidget(self.candidate_status)
        self.candidates_table = _table(("失败原句", "直播间", "失败次数", "最近失败（UTC+8）", "失败原因", "检测结果", "处理状态", "处置说明"))
        self.candidates_table.itemSelectionChanged.connect(self.select_candidate)
        candidate_layout.addWidget(self.candidates_table)
        self.candidate_text = QPlainTextEdit()
        self.candidate_text.setReadOnly(True)
        self.candidate_text.setMaximumHeight(90)
        self.candidate_text.setPlaceholderText(tr("选择失败句查看完整内容；此处不是已启用规则"))
        candidate_layout.addWidget(self.candidate_text)
        self.candidate_analysis = QPlainTextEdit()
        self.candidate_analysis.setReadOnly(True)
        self.candidate_analysis.setMaximumHeight(150)
        self.candidate_analysis.setPlaceholderText(tr("选择失败句查看自动诊断、处理进展和本地检测结果"))
        candidate_layout.addWidget(self.candidate_analysis)
        self.candidate_button = QPushButton(tr("从候选填写规则"))
        self.candidate_button.setEnabled(False)
        self.candidate_button.clicked.connect(self.prepare_candidate)
        candidate_layout.addWidget(self.candidate_button)
        self.tabs.addTab(candidates, tr("失败处理记录"))
        dictionary = QWidget()
        dictionary_layout = QVBoxLayout(dictionary)
        self.dictionary_status = _label()
        dictionary_layout.addWidget(self.dictionary_status)
        dictionary_layout.addWidget(_label(tr("社区参考词库仅检测提示，命中不代表 B 站禁用；不会自动替换。"
            "游戏用词容易误报，可将确认正常的词加入白名单。")))
        self.dictionary_search = QLineEdit()
        self.dictionary_search.setPlaceholderText(tr("搜索参考词、分类或来源"))
        self.dictionary_search.textChanged.connect(self.filter_dictionary)
        dictionary_layout.addWidget(self.dictionary_search)
        self.dictionary_table = _table(("参考词", "分类", "词库来源", "适用直播间"))
        dictionary_layout.addWidget(self.dictionary_table, 1)
        self.dictionary_sources = QPlainTextEdit()
        self.dictionary_sources.setReadOnly(True)
        self.dictionary_sources.setMaximumHeight(90)
        dictionary_layout.addWidget(self.dictionary_sources)
        dictionary_layout.addWidget(_label(tr("检测白名单（每行一个完整词，保存后重新检测失败句）")))
        self.allowlist = QPlainTextEdit()
        self.allowlist.setMaximumHeight(70)
        dictionary_layout.addWidget(self.allowlist)
        dictionary_buttons = QHBoxLayout()
        self.reload_dictionary_button = QPushButton(tr("重新加载词库并检测"))
        self.reload_dictionary_button.clicked.connect(self.reload_dictionary)
        dictionary_buttons.addWidget(self.reload_dictionary_button)
        self.save_allowlist_button = QPushButton(tr("保存检测白名单"))
        self.save_allowlist_button.clicked.connect(self.save_allowlist)
        dictionary_buttons.addWidget(self.save_allowlist_button)
        dictionary_layout.addLayout(dictionary_buttons)
        self.tabs.addTab(dictionary, tr("参考词库与检测"))
        controls.addWidget(self.tabs, 1)
        self.editor_note = _label(tr("新增规则：填写原词，选择中性替换或从弹幕删除，保存后生效。"))
        controls.addWidget(self.editor_note)
        form = QFormLayout()
        self.term = QLineEdit()
        form.addRow(tr("原词"), self.term)
        self.rule_action = QComboBox()
        self.rule_action.addItem(tr("中性替换"), "replace")
        self.rule_action.addItem(tr("从弹幕删除"), "remove")
        self.rule_action.currentIndexChanged.connect(self._action_changed)
        form.addRow(tr("处理方式"), self.rule_action)
        self.replacement = QLineEdit()
        self.replacement.setPlaceholderText(tr("留空使用“已过滤”"))
        form.addRow(tr("中性替换词"), self.replacement)
        form.addRow(tr("适用范围"), _label(tr("全部直播间")))
        self.rule_enabled = QCheckBox(tr("启用此规则"))
        self.rule_enabled.setChecked(True)
        form.addRow(self.rule_enabled)
        controls.addLayout(form)
        editor_buttons = QHBoxLayout()
        self.new_button = QPushButton(tr("新增规则"))
        self.new_button.clicked.connect(self.new_rule)
        editor_buttons.addWidget(self.new_button)
        self.save_button = QPushButton(tr("保存规则"))
        self.save_button.clicked.connect(self.save_rule)
        editor_buttons.addWidget(self.save_button)
        self.remove_button = QPushButton(tr("删除所选规则"))
        self.remove_button.setEnabled(False)
        self.remove_button.clicked.connect(self.remove_rule)
        editor_buttons.addWidget(self.remove_button)
        controls.addLayout(editor_buttons)
        layout.addWidget(self.controls, 1)
        self.status = _label()
        self.status.setStyleSheet("color: #785713;")
        layout.addWidget(self.status)
        path = getattr(content_filter, "path", "")
        self.path_label = _label(tr("本地名单文件：{0}", path))
        self.path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.path_label.setVisible(bool(path))
        layout.addWidget(self.path_label)
        layout.addWidget(_label(tr("名单和失败候选保存在本机用户数据目录，不进入分发包。"
            "发送中或结果未知的消息不会因编辑规则而改写；已确认失败的消息重试前应用最新规则。")))
        close = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        close.rejected.connect(self.reject)
        layout.addWidget(close)
        self.new_rule()
        self.refresh()
        self.tabs.currentChanged.connect(self._refresh_visible_candidates)
        self.recovery_timer = QTimer(self)
        self.recovery_timer.setInterval(2000)
        self.recovery_timer.timeout.connect(self._refresh_visible_candidates)
        self.recovery_timer.start()

    def _action_changed(self):
        self.replacement.setEnabled(self.rule_action.currentData() == "replace")

    def _error(self, error):
        message = str(error)
        known = {
            "过滤词不能为空": "过滤词不能为空",
            "过滤词不能超过80个字符（按 UTF-16 计数）": "过滤词不能超过80个字符（按 UTF-16 计数）",
            "替换内容不能超过80个字符（按 UTF-16 计数）": "替换内容不能超过80个字符（按 UTF-16 计数）",
            "过滤词不能包含控制字符或隐藏字符": "过滤词不能包含控制字符或隐藏字符",
            "替换内容不能包含控制字符或隐藏字符": "替换内容不能包含控制字符或隐藏字符",
            "该直播间范围已存在相同过滤词，请编辑原规则": "已存在相同过滤词，请编辑原规则",
            "该直播间范围已存在相同过滤词": "已存在相同过滤词，请编辑原规则",
            "已存在相同过滤词，请编辑原规则": "已存在相同过滤词，请编辑原规则",
            "确认的过滤词必须来自所选失败内容": "确认的过滤词必须来自所选失败内容",
            "直播间必须是数字房间号；留空表示全部直播间": "请填写数字直播间号，或选择全部直播间。",
        }
        if message in known:
            result = tr(known[message])
        elif message.startswith("无法保存本地过滤名单"):
            result = tr("无法保存本地过滤名单，原规则未修改。")
        else:
            result = tr("过滤规则操作失败，请检查内容和本地名单文件。")
        self.status.setText(result)

    def refresh(self):
        if self.content_filter is None:
            self.controls.setEnabled(False)
            self.status.setText(tr("本地过滤名单暂不可用，请检查词库文件。"))
            return False
        try:
            rules = self.content_filter.list_rules()
            self.content_filter.analyze_candidates()
            candidates = self.content_filter.list_candidates()
        except (DanmakuFilterError, OSError) as error:
            self._error(error)
            self.controls.setEnabled(False)
            return False
        self.controls.setEnabled(True)
        self.enabled.setChecked(self.content_filter.enabled)
        self._render_rules(rules)
        self._render_candidates(candidates)
        self.refresh_dictionary()
        self.status.setText(tr("已加载 {0} 条规则、{1} 条失败处理记录。", len(rules), len(candidates)))
        return True

    def _render_rules(self, rules, *, preserve_selection=False):
        row = self.rules_table.currentRow()
        selected_id = self._rules[row]["id"] if preserve_selection and 0 <= row < len(self._rules) else None
        self._rules = rules
        self.rules_table.blockSignals(True)
        self.rules_table.setRowCount(len(rules))
        for index, rule in enumerate(rules):
            result = tr("从弹幕删除") if rule.get("action", "replace") == "remove" else rule["replacement"]
            values = (tr("已启用" if rule["enabled"] else "已停用"), rule["term"],
                result, tr("全部直播间"), rule["updated_at"],
                self.rule_origin_text(rule))
            for column, value in enumerate(values):
                self.rules_table.setItem(index, column, QTableWidgetItem(str(value)))
        self.rules_table.clearSelection()
        self.rules_table.setCurrentCell(-1, -1)
        selected_row = next((index for index, rule in enumerate(rules) if rule["id"] == selected_id), None)
        if selected_row is not None:
            self.rules_table.selectRow(selected_row)
        self.rules_table.blockSignals(False)

    @staticmethod
    def rule_origin_text(rule):
        if rule.get("origin") == "ai_analysis" and rule.get("validation"):
            return tr("专用测试房间验证通过（正式房间原因未证实）")
        if rule.get("origin") == "ai_analysis" and rule.get("enabled") is False:
            return tr("AI建议（尚未验证，已停用，原因未证实）")
        origins = {"ai_analysis": "AI建议（尚未实测，原因未证实）",
                   "ai_success_observed": "AI修正后可见（经验规则，原因未证实）",
                   "ai_test_validated": "AI建议已通过测试房间验证（正式房间原因未证实）"}
        return tr(origins.get(rule.get("origin"), "手工确认"))

    def _refresh_visible_candidates(self, *_):
        if not self.isVisible():
            return
        if self.tabs.currentIndex() == 1:
            self.refresh_candidates()
        elif self.tabs.currentIndex() == 0:
            self.refresh_rule_records()

    def refresh_rule_records(self):
        if self.content_filter is None:
            return False
        try:
            rules = self.content_filter.list_rules()
        except (DanmakuFilterError, OSError) as error:
            self._error(error)
            return False
        if rules != self._rules:
            self._render_rules(rules, preserve_selection=True)
        return True

    def refresh_candidates(self, *_):
        if self.content_filter is None:
            return False
        try:
            candidates = self.content_filter.list_candidates()
        except (DanmakuFilterError, OSError) as error:
            self._error(error)
            return False
        self._render_candidates(candidates)
        return True

    def _render_candidates(self, candidates):
        row = self.candidates_table.currentRow()
        selected_id = self._candidates[row]["id"] if 0 <= row < len(self._candidates) else None
        scroll_position = self.candidates_table.verticalScrollBar().value()
        resolved_count = sum(candidate.get("status") == "resolved" for candidate in candidates)
        self._candidates = [candidate for candidate in candidates
                            if self.show_history.isChecked() or candidate.get("status") != "resolved"]
        self.candidate_status.setText(tr("未处理 {0} 条 · 已处理历史 {1} 条", len(candidates) - resolved_count, resolved_count))
        self.candidates_table.blockSignals(True)
        self.candidates_table.setRowCount(len(self._candidates))
        reasons = {"platform_f": "平台返回 f（原因未确定）",
                   "platform_rejected": "平台拒绝（原因待确认）",
                   "result_unknown": "发送结果未知"}
        for index, candidate in enumerate(self._candidates):
            analysis = candidate.get("analysis", {})
            checked = {"checked": "已检测", "unavailable": "未配置词库", "error": "检测失败"}
            values = (candidate["text"], candidate["room"] or tr("全部直播间"), candidate["count"],
                candidate["last_time"], tr(reasons.get(candidate["reason"], "发送失败（原因待确认）")),
                tr(checked.get(analysis.get("status"), "尚未检测")),
                self.recovery_status_text(candidate), self.disposition_text(candidate))
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setToolTip(str(value))
                self.candidates_table.setItem(index, column, item)
        self.candidates_table.clearSelection()
        self.candidates_table.setCurrentCell(-1, -1)
        selected_row = next((index for index, candidate in enumerate(self._candidates)
                             if candidate["id"] == selected_id), None)
        if selected_row is not None:
            self.candidates_table.selectRow(selected_row)
        self.candidates_table.blockSignals(False)
        self.candidates_table.verticalScrollBar().setValue(scroll_position)
        if selected_row is not None:
            self.select_candidate()
        else:
            self.candidate_button.setEnabled(False)
            self.candidate_text.clear()
            self.candidate_analysis.clear()

    @staticmethod
    def recovery_status_text(candidate):
        if candidate.get("status") == "waiting_validation":
            validations = {"waiting_room": "等待配置独立测试房间", "ready": "等待打开验证直播间",
                           "testing": "正在验证修改后正文", "failed": "验证未通过，等待继续测试"}
            return tr(validations.get((candidate.get("validation") or {}).get("state"), "等待独立测试房间验证"))
        if candidate.get("status") == "action_required" and (candidate.get("diagnosis") or {}).get("action") == "review_on_restart":
            return tr("下次启动自动重试分析")
        states = {"active": "待处理", "recovering": "自动处理中", "waiting_confirmation": "等待结果确认",
                  "resolved": "已处理（历史）", "action_required": "需要登录或人工处理"}
        return tr(states.get(candidate.get("status", "active"), "待处理"))

    @staticmethod
    def disposition_text(candidate):
        if candidate.get("status") == "waiting_validation":
            return tr("测试正文等待验证，成功后才比对原句生成局部规则")
        if candidate.get("status") == "resolved":
            resolutions = {"page_observed": "页面出现同文，来源未确认",
                           "corrected_page_observed": "页面出现同文，来源未确认",
                           "ai_no_local_edit": "分析已完成，未提出局部修改",
                           "ai_rules_saved": "分析已完成，已保存局部建议规则（尚未实测）",
                           "test_room_validated": "独立测试房间验证通过，正式房间效果未确认"}
            if candidate.get("resolution") in resolutions:
                return tr(resolutions[candidate["resolution"]])
        diagnosis = candidate.get("diagnosis") or {}
        if diagnosis.get("message"):
            return tr(diagnosis["message"])
        defaults = {"active": "等待自动分析失败正文", "recovering": "正在独立分析和验证失败正文",
                    "waiting_confirmation": "保留消息，自动检查发送结果",
                    "resolved": "处理完成，原始记录保留", "action_required": "保留队列，等待登录或验证"}
        return tr(defaults.get(candidate.get("status", "active"), "等待自动分析失败正文"))

    def new_rule(self):
        self._rule_id = self._candidate_id = None
        self.term.clear()
        self.rule_action.setCurrentIndex(self.rule_action.findData("replace"))
        self._action_changed()
        self.replacement.clear()
        self.rule_enabled.setChecked(True)
        self.rule_enabled.setEnabled(True)
        self.remove_button.setEnabled(False)
        self.editor_note.setText(tr("新增规则：填写原词，选择中性替换或从弹幕删除，保存后生效。"))

    def select_rule(self):
        row = self.rules_table.currentRow()
        if not 0 <= row < len(self._rules):
            return
        rule = self._rules[row]
        self._rule_id, self._candidate_id = rule["id"], None
        self.term.setText(rule["term"])
        self.rule_action.setCurrentIndex(self.rule_action.findData(rule.get("action", "replace")))
        self._action_changed()
        self.replacement.setText(rule["replacement"])
        self.rule_enabled.setChecked(rule["enabled"])
        unvalidated = rule.get("origin") == "ai_analysis" and not rule.get("validation")
        self.rule_enabled.setEnabled(not unvalidated)
        self.remove_button.setEnabled(True)
        self.editor_note.setText(tr("编辑AI建议规则：未验证时保持停用，可修改内容，验证通过后才可启用。" if unvalidated else
            "编辑已保存规则：更改后点击保存规则。"))

    def select_candidate(self):
        row = self.candidates_table.currentRow()
        if 0 <= row < len(self._candidates):
            candidate = self._candidates[row]
            self.candidate_text.setPlainText(candidate["text"])
            self.candidate_analysis.setPlainText(self.candidate_details(candidate))
            self.candidate_button.setEnabled(True)

    @classmethod
    def candidate_details(cls, candidate):
        lines = [tr("处理状态：{0}", cls.recovery_status_text(candidate)),
                 tr("处置说明：{0}", cls.disposition_text(candidate))]
        if candidate.get("queue_id") is not None:
            lines.append(tr("关联队列编号：{0}", candidate["queue_id"]))
        if candidate.get("resolved_at"):
            lines.append(tr("处理时间（UTC+8）：{0}", candidate["resolved_at"]))
        if candidate.get("status") == "resolved":
            lines.append(tr("处理记录归档不代表弹幕已经送达。"))
        diagnosis = candidate.get("diagnosis") or {}
        if diagnosis:
            lines.append(tr("诊断分类：{0}", recovery_category_text(diagnosis.get("category"))))
            lines.append(tr("处理动作：{0}", recovery_action_text(diagnosis.get("action"))))
        validation = candidate.get("validation") or {}
        previous = candidate.get("last_validation") or {}
        if previous:
            if previous.get("revised_text"):
                lines.append(tr("上一测试正文：{0}", previous["revised_text"]))
            if previous.get("test_room"):
                lines.append(tr("上一测试直播间：{0}", previous["test_room"]))
            lines.append(tr("上一测试未通过，记录保留；当前正文须重新独立验证。"))
        if validation:
            if validation.get("test_room"):
                lines.append(tr("独立测试直播间：{0}", validation["test_room"]))
            if validation.get("formal_room"):
                lines.append(tr("正式直播间：{0}", validation["formal_room"]))
            if type(validation.get("attempts")) is int:
                lines.append(tr("测试尝试次数：{0}", validation["attempts"]))
            if validation.get("original_text"):
                lines.append(tr("原始正文：{0}", validation["original_text"]))
            if validation.get("revised_text"):
                lines.append(tr("待验证正文：{0}", validation["revised_text"]))
            if validation.get("message"):
                lines.append(tr("测试处理说明：{0}", tr(validation["message"])))
            lines.append(tr("测试通过只说明测试房间可发，正式房间效果和禁词原因仍未确认。"))
        lines.append(tr("页面出现同文只代表页面观察结果，不能证明由程序自动发送；人工发送也可能产生相同文字。"))
        lines.append("")
        lines.append(cls.analysis_text(candidate.get("analysis", {})))
        return "\n".join(lines)

    @staticmethod
    def analysis_text(analysis):
        messages = {
            "no_wordlist": "未配置本地参考词库，尚未进行词库检测。",
            "dictionary_error": "本地词库不可用，检测失败；字幕与原队列继续保留。",
            "no_hits": "已完成本地检测：未命中参考词或已确认规则。发送失败原因仍未确定。",
            "advisory_hits": "已完成本地检测：发现参考词命中，请结合上下文核对。",
        }
        lines = [tr(messages.get(analysis.get("summary_code"), "尚未检测，请刷新列表。"))]
        if analysis.get("summary_code") == "no_hits" and analysis.get("known_rules"):
            lines[0] = tr("已完成本地检测：未命中参考词，但命中已确认规则。发送失败原因仍未确定。")
        if any(hit.get("category") == "user-confirmed-trigger" for hit in analysis.get("hits", [])):
            lines[0] = tr("已完成本地检测：命中当前直播间的实测触发词。")
        if analysis.get("checked_at"):
            lines.append(tr("检测时间（UTC+8）：{0}", analysis["checked_at"]))
        for hit in analysis.get("hits", []):
            lines.append(tr("参考命中：{0} · 原文：{1} · 分类：{2} · 来源：{3} · 位置：{4}–{5}",
                hit["term"], hit["matched"], hit["category"], hit["source_id"], hit["start"] + 1, hit["end"]))
        for rule in analysis.get("known_rules", []):
            if rule.get("action", "replace") == "remove":
                lines.append(tr("已确认删除规则：{0}（只从弹幕删除）", rule["term"]))
            else:
                lines.append(tr("已确认规则：{0} → {1}", rule["term"], rule["replacement"]))
        if analysis.get("truncated"):
            lines.append(tr("命中过多，仅显示前 100 项。"))
        lines.append(tr("检测结果不等于平台判定；未命中也不能保证发送成功。"))
        return "\n".join(lines)

    def refresh_dictionary(self):
        info = self.content_filter.detection_info()
        if info["status"] == "ready":
            self.dictionary_status.setText(tr("本地参考词库：{0} 个词 · {1} 个来源 · 离线检测",
                info["entry_count"], info["source_count"]))
        elif info["status"] == "no_wordlist":
            self.dictionary_status.setText(tr("未配置本地参考词库，尚未进行词库检测。"))
        else:
            self.dictionary_status.setText(tr("本地词库不可用，检测失败；字幕与原队列继续保留。"))
        entries = self.content_filter.detector_entries()
        self.dictionary_table.setRowCount(len(entries))
        for index, entry in enumerate(entries):
            for column, key in enumerate(("term", "category", "source_id")):
                self.dictionary_table.setItem(index, column, QTableWidgetItem(entry[key]))
            self.dictionary_table.setItem(index, 3, QTableWidgetItem(entry.get("room") or tr("全部直播间")))
        sources = self.content_filter.detector_sources()
        self.dictionary_sources.setPlainText("\n\n".join(
            f"{row['name']} · {row['license']}\n{row['url']}\n{row['revision']}" for row in sources))
        self.allowlist.setPlainText("\n".join(self.content_filter.get_allowlist()))
        self.filter_dictionary()

    def filter_dictionary(self):
        query = self.dictionary_search.text().strip().casefold()
        for row in range(self.dictionary_table.rowCount()):
            values = [self.dictionary_table.item(row, column).text()
                      for column in range(self.dictionary_table.columnCount())]
            self.dictionary_table.setRowHidden(row, bool(query) and not any(
                query in value.casefold() for value in values))

    def reload_dictionary(self):
        if self.content_filter is None:
            return False
        self.content_filter.reload_detector()
        return self.refresh()

    def save_allowlist(self):
        if self.content_filter is None:
            return False
        try:
            terms = [line.strip() for line in self.allowlist.toPlainText().splitlines() if line.strip()]
            self.content_filter.set_allowlist(terms)
        except (DanmakuFilterError, OSError, ValueError):
            self.status.setText(tr("白名单保存失败，原词库未修改。"))
            return False
        self.refresh()
        self.status.setText(tr("检测白名单已保存，失败句已重新检测。"))
        return True

    def prepare_candidate(self):
        row = self.candidates_table.currentRow()
        if not 0 <= row < len(self._candidates):
            return
        candidate = self._candidates[row]
        self.new_rule()
        self._candidate_id = candidate["id"]
        self.editor_note.setText(tr("请填写已确认需要过滤的原词；失败句不会自动成为规则。"))
        self.term.setFocus()

    def save_rule(self):
        if self.content_filter is None:
            return False
        action = self.rule_action.currentData()
        replacement = self.replacement.text() if action == "replace" else ""
        try:
            if self._rule_id:
                self.content_filter.update_rule(self._rule_id, term=self.term.text(),
                    replacement=replacement, room="", enabled=self.rule_enabled.isChecked(), action=action)
            elif self._candidate_id:
                # Promotion requires an explicit term from the selected failed text.
                self.content_filter.promote_candidate(self._candidate_id, self.term.text(),
                    replacement, room="", enabled=self.rule_enabled.isChecked(), action=action)
            else:
                self.content_filter.add_rule(self.term.text(), replacement,
                    room="", enabled=self.rule_enabled.isChecked(), action=action)
        except (DanmakuFilterError, OSError) as error:
            self._error(error)
            return False
        self.new_rule()
        self.refresh()
        self.status.setText(tr("过滤规则已保存，用于后续发送。"))
        return True

    def save_enabled(self):
        if self.content_filter is None:
            return False
        try:
            self.content_filter.set_enabled(self.enabled.isChecked())
        except (DanmakuFilterError, OSError) as error:
            self.enabled.setChecked(self.content_filter.enabled)
            self._error(error)
            return False
        self.status.setText(tr("过滤名单开关已保存。"))
        return True

    def remove_rule(self):
        if self.content_filter is None or not self._rule_id:
            return False
        answer = QMessageBox.question(self, tr("删除所选规则"),
            tr("删除选中的本地过滤规则？后续弹幕将不再使用这条规则。"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if answer != QMessageBox.StandardButton.Yes:
            return False
        try:
            self.content_filter.remove_rule(self._rule_id)
        except (DanmakuFilterError, OSError) as error:
            self._error(error)
            return False
        self.new_rule()
        self.refresh()
        self.status.setText(tr("所选过滤规则已删除。"))
        return True
