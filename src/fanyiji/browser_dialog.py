"""Small control window for the optional, account-isolated live browser."""
from PySide6.QtCore import Qt
from PySide6.QtGui import QDesktopServices
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDoubleSpinBox,
                              QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton,
                              QVBoxLayout, QWidget)

from .browser_service import room_url
from .browser_runtime import runtime_snapshot
from .i18n import get_language, recovery_action_text, tr
from .identity import APP_NAME


class BrowserDialog(QDialog):
    def __init__(self, service, parent=None):
        super().__init__(parent)
        self.service = service
        self._saving_browser_controls = False
        self._danmaku_snapshot = {}
        self.available_version = None
        self.setWindowTitle(APP_NAME + " · " + tr("直播浏览器"))
        self.setMinimumWidth(560)
        layout = QVBoxLayout(self)
        info = QLabel(tr("首次扫码登录一次，后续复用独立登录状态。无需安装字幕弹幕桥扩展。"))
        info.setWordWrap(True)
        layout.addWidget(info)
        room_row = QHBoxLayout()
        room_row.addWidget(QLabel(tr("直播间")))
        self.room = QLineEdit(service.room)
        self.room.setPlaceholderText(tr("直播间号或 HTTPS 直播间地址"))
        room_row.addWidget(self.room)
        layout.addLayout(room_row)
        buttons = QHBoxLayout()
        for text, action in (("登录账号", self.login), ("打开直播间", self.open_room),
                             ("刷新网页", service.refresh), ("关闭浏览器", service.stop_browser)):
            button = QPushButton(tr(text))
            button.clicked.connect(action)
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.audio_processing_button = None
        if callable(getattr(service, "audio_processing_requested", None)):
            self.audio_processing_button = QPushButton(tr("音源与语音处理"))
            self.audio_processing_button.clicked.connect(service.audio_processing_requested)
            layout.addWidget(self.audio_processing_button)
        self.auto_check = QCheckBox(tr("自动检查浏览器内核更新"))
        self.auto_check.setChecked(service.auto_check)
        layout.addWidget(self.auto_check)
        self.push_controls = QWidget(self)
        push_layout = QVBoxLayout(self.push_controls)
        push_layout.setContentsMargins(0, 0, 0, 0)
        self.danmaku_enabled = QCheckBox(tr("同步终稿译文到直播间"))
        self.danmaku_enabled.setChecked(getattr(service, "danmaku_enabled", False))
        push_layout.addWidget(self.danmaku_enabled)
        validation_row = QHBoxLayout()
        validation_row.addWidget(QLabel(tr("AI验证直播间")))
        self.validation_room = QLineEdit(getattr(service, "danmaku_validation_room", ""))
        self.validation_room.setPlaceholderText(tr("填写独立测试房间号，不能与正式直播间相同"))
        validation_row.addWidget(self.validation_room, 1)
        self.validation_open_button = QPushButton(tr("打开验证直播间"))
        self.validation_open_button.clicked.connect(self.open_validation_room)
        validation_row.addWidget(self.validation_open_button)
        push_layout.addLayout(validation_row)
        validation_note = QLabel(tr("修改后正文须在独立测试房间验证通过，才比对原句生成局部规则并归档。测试通过只说明该测试房间可发，不代表正式房间禁词原因已确认。"))
        validation_note.setWordWrap(True)
        push_layout.addWidget(validation_note)
        source_row = QHBoxLayout()
        source_row.addWidget(QLabel(tr("发送来源")))
        self.danmaku_source = QComboBox()
        for source, text in (("system", "电脑声音的译文"), ("mic", "麦克风的译文"),
                             ("all", "两路译文")):
            self.danmaku_source.addItem(tr(text), source)
        index = self.danmaku_source.findData(getattr(service, "danmaku_source", "system"))
        self.danmaku_source.setCurrentIndex(max(index, 0))
        source_row.addWidget(self.danmaku_source, 1)
        source_row.addWidget(QLabel(tr("发送间隔")))
        self.danmaku_interval = QDoubleSpinBox()
        self.danmaku_interval.setRange(3.5, 60.0)
        self.danmaku_interval.setDecimals(1)
        self.danmaku_interval.setSingleStep(0.5)
        self.danmaku_interval.setSuffix(tr(" 秒"))
        self.danmaku_interval.setValue(getattr(service, "danmaku_interval_seconds", 3.5))
        source_row.addWidget(self.danmaku_interval)
        push_layout.addLayout(source_row)
        push_row = QHBoxLayout()
        self.save_button = QPushButton(tr("保存推送设置"))
        self.save_button.clicked.connect(self.save_danmaku_preferences)
        push_row.addWidget(self.save_button)
        self.resume_button = QPushButton(tr("恢复推送"))
        self.resume_button.clicked.connect(service.resume_danmaku)
        self.resume_button.setEnabled(False)
        push_row.addWidget(self.resume_button)
        self.clear_button = QPushButton(tr("清空待发队列"))
        self.clear_button.setEnabled(False)
        self.clear_button.clicked.connect(self.clear_pending)
        push_row.addWidget(self.clear_button)
        self.filter_button = QPushButton(tr("本地过滤词管理"))
        self.filter_button.clicked.connect(service.request_content_filter)
        push_row.addWidget(self.filter_button)
        push_row.addStretch()
        push_layout.addLayout(push_row)
        self.danmaku_status = QLabel(tr("弹幕推送未开启"))
        self.danmaku_status.setTextFormat(Qt.TextFormat.PlainText)
        self.danmaku_status.setWordWrap(True)
        push_layout.addWidget(self.danmaku_status)
        push_note = QLabel(tr("保存后生效。启动后请手动打开正式直播间，不自动启动浏览器；只发送终稿，"
                              "每段【译文】连同括号须为 11 到 40 字符。正文去标点，断句用空格，保留外层【】。"
                              "10 秒内重复内容跳过，发送间隔至少 3.5 秒。"
                              "本轮队列不过期；失败消息保存到独立处理记录，交给 Codex 分析，后续队列继续发送；"
                              "修改后正文须在独立测试房间验证通过才生成规则，结果未知的测试只读核验，不重复提交；"
                              "验证码、登录和禁言仍须在浏览器处理，失败内容保留。"))
        push_note.setWordWrap(True)
        push_layout.addWidget(push_note)
        restart_note = QLabel(tr("每次启动清空上一轮待发队列与运行统计，失败处理记录和过滤规则保留并自动分析。"))
        restart_note.setWordWrap(True)
        push_layout.addWidget(restart_note)
        merge_note = QLabel(tr("含【】少于 11 字符时等待下一句合并；含【】和分隔空格最多 40 字符。"
                               "实际账号加权上限可能低于 40，按当前允许长度拆分。"))
        merge_note.setWordWrap(True)
        push_layout.addWidget(merge_note)
        available = getattr(service, "danmaku_available", True)
        self.push_controls.setVisible(available)
        self.push_controls.setEnabled(available)
        layout.addWidget(self.push_controls)
        self.danmaku_access_note = QLabel(tr("弹幕推送仅本机启用，分发版不开放"))
        self.danmaku_access_note.setVisible(not available)
        self.danmaku_access_note.setWordWrap(True)
        layout.addWidget(self.danmaku_access_note)
        self.versions = QLabel(tr("正在读取浏览器版本…"))
        self.versions.setTextFormat(Qt.TextFormat.PlainText)
        self.versions.setWordWrap(True)
        layout.addWidget(self.versions)
        self.status = QLabel(tr("浏览器尚未打开"))
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        update_row = QHBoxLayout()
        check = QPushButton(tr("检查更新"))
        check.clicked.connect(service.check_updates)
        update_row.addWidget(check)
        download = QPushButton(tr("微软官方内核下载"))
        download.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(
            "https://developer.microsoft.com/microsoft-edge/webview2/#download-section")))
        update_row.addWidget(download)
        update_row.addStretch()
        layout.addLayout(update_row)
        note = QLabel(tr("WebView2 Evergreen 由微软自动下载更新。安装新版后重新打开浏览器生效；"
                        "联网检查失败时不会标记为最新版。"))
        note.setWordWrap(True)
        layout.addWidget(note)
        service.event.connect(self.on_event)
        service.snapshot.connect(self.on_snapshot)
        service.prefs_changed.connect(self.on_preferences)
        service.danmaku_status.connect(self.on_danmaku_status)
        self.on_danmaku_status(getattr(service, "danmaku_snapshot", {}) or {
            "enabled": getattr(service, "danmaku_enabled", False), "state": "waiting"})
        self.on_snapshot(runtime_snapshot(service.active_version, False))
        service.activate_checks()

    def _save(self):
        try:
            # Navigating or signing in must not submit unsaved push controls.
            self._saving_browser_controls = True
            self.service.save_preferences(self.room.text(), self.auto_check.isChecked())
            return True
        except ValueError:
            self.status.setText(tr("直播间格式无效"))
        except OSError:
            self.status.setText(tr("浏览器设置保存失败"))
        finally:
            self._saving_browser_controls = False
        return False

    def save_danmaku_preferences(self):
        if not getattr(self.service, "danmaku_available", True):
            return False
        try:
            self.service.save_preferences(self.room.text(), self.auto_check.isChecked(),
                danmaku_enabled=self.danmaku_enabled.isChecked(),
                danmaku_source=self.danmaku_source.currentData(),
                danmaku_interval_seconds=self.danmaku_interval.value(),
                danmaku_validation_room=self.validation_room.text())
        except ValueError:
            try:
                validation = room_url(self.validation_room.text()).rsplit("/", 1)[-1] if self.validation_room.text().strip() else ""
                formal = room_url(self.room.text()).rsplit("/", 1)[-1] if self.room.text().strip() else ""
            except ValueError:
                validation, formal = "", ""
            self.status.setText(tr("AI验证直播间不能与正式直播间相同" if validation and validation == formal else "直播间格式或推送设置无效"))
            return False
        except OSError:
            self.status.setText(tr("浏览器设置保存失败"))
            return False
        self.status.setText(tr("推送设置已保存"))
        return True

    def on_preferences(self, preferences):
        if self._saving_browser_controls:
            return
        self.room.setText(preferences.get("room", self.service.room))
        self.auto_check.setChecked(preferences.get("auto_check", self.service.auto_check))
        self.danmaku_enabled.setChecked(preferences.get("enabled", preferences.get(
            "danmaku_enabled", getattr(self.service, "danmaku_enabled", False))))
        index = self.danmaku_source.findData(preferences.get("source", preferences.get(
            "danmaku_source", getattr(self.service, "danmaku_source", "system"))))
        self.danmaku_source.setCurrentIndex(max(index, 0))
        self.danmaku_interval.setValue(preferences.get("interval_seconds", preferences.get(
            "danmaku_interval_seconds", getattr(self.service, "danmaku_interval_seconds", 3.5))))
        self.validation_room.setText(preferences.get("validation_room", preferences.get(
            "danmaku_validation_room", getattr(self.service, "danmaku_validation_room", ""))))

    def on_danmaku_status(self, status):
        self._danmaku_snapshot = dict(status)
        enabled = status.get("enabled", getattr(self.service, "danmaku_enabled", False))
        state = status.get("state", "waiting")
        states = {
            "waiting": "等待已登录的目标直播页", "ready": "等待终稿译文", "idle": "等待终稿译文",
            "empty": "等待终稿译文", "checking_login": "正在核实账号登录状态",
            "waiting_merge": "短句等待下一句合并",
            "waiting_browser": "等待内置浏览器连接", "waiting_connection": "等待内置浏览器连接",
            "login_required": "请在内置浏览器登录账号", "queued_room_mismatch": "待发消息属于其他直播间，请核对设置",
            "storage_error": "队列保存失败，请检查本机存储",
            "filter_error": "过滤名单异常，待发消息保留",
            "queued": "按队列等待发送", "sending": "正在提交弹幕", "accepted": "接口回执已接收，尚未确认页面收到",
            "awaiting_verification": "接口回执已接收，尚未确认页面收到",
            "verifying": "正在只读核实页面是否收到弹幕",
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
            "page_visible": "直播页面已收到弹幕", "cooldown": "等待发送间隔",
            "sent_page_visible": "直播页面已收到弹幕",
            "paused": "推送已暂停，队列保留", "rejected": "平台拒绝，消息保留等待核实",
            "platform_rejected": "平台拒绝，消息保留等待核实",
            "platform_f": "平台返回 f，按发送间隔自动重试当前条",
            "waiting_f_retry": "平台返回 f，等待自动重试当前条",
            "retrying_f": "正在自动重试当前条，保留队列顺序",
            "unknown": "发送结果未知，消息保留等待核实", "disabled": "弹幕推送未开启",
            "result_unknown": "发送结果未知，消息保留等待核实",
            "queue_cleared": "待发队列已清理", "cleared": "待发队列已清理",
        }
        recovery_visible = state in {"ai_analyzing", "ai_repair_ready", "ai_retrying",
                                     "waiting_validation", "validating", "validation_failed",
                                     "waiting_validation_browser", "validation_storage_error", "validation_passed",
                                     "validation_configuration_changed", "failure_transferred"}
        description = tr(states.get(state, "等待已登录的目标直播页")) if enabled or recovery_visible else tr("弹幕推送未开启")
        queued = status.get("queued", status.get("pending", 0))
        accepted = status.get("accepted", 0)
        visible = status.get("page_visible", 0)
        message = tr("{0} · 待发 {1} 段 · 接口回执 {2} · 页面收到 {3}",
                     description, queued, accepted, visible)
        remaining = status.get("cooldown_remaining", status.get("wait_seconds", 0))
        if isinstance(remaining, (int, float)) and remaining > 0:
            message += tr(" · 还需等待 {0:.1f} 秒", remaining)
        f_retries = status.get("f_retries", 0)
        if type(f_retries) is int and f_retries > 0:
            message += tr(" · f 自动重试累计 {0} 次", f_retries)
        rule_count, candidate_count = status.get("filter_rule_count"), status.get("filter_candidate_count")
        if type(rule_count) is int and rule_count >= 0 and type(candidate_count) is int and candidate_count >= 0:
            message += tr(" · 过滤规则 {0} · 未处理失败 {1}", rule_count, candidate_count)
        resolved_count = status.get("filter_resolved_count")
        if type(resolved_count) is int and resolved_count > 0:
            message += tr(" · 已处理历史 {0}", resolved_count)
        deferred = status.get("deferred_analysis")
        if type(deferred) is int and deferred >= 0:
            message += tr(" · 累计移入独立处理 {0} 段", deferred)
        if status.get("recovery_action"):
            message += tr(" · 处置 {0}", recovery_action_text(status["recovery_action"]))
        if status.get("filter_error") and state != "filter_error":
            message += tr(" · 过滤名单异常，待发消息保留")
        if status.get("filter_candidate_error"):
            message += tr(" · 失败候选保存失败，请检查本地文件")
        cleared = status.get("cleared", 0)
        if isinstance(cleared, int) and cleared > 0:
            message += tr(" · 已清理 {0} 段未发送消息", cleared)
        code, characters = status.get("last_platform_code"), status.get("head_characters")
        # Show diagnostic counters only; platform message text may contain private data.
        if type(code) is int:
            message += tr(" · 平台码 {0}", code)
        if type(characters) is int and characters >= 0:
            message += tr(" · 当前单条长度 {0}（UTF-16）", characters)
        maximum = status.get("max_weighted_length")
        if type(maximum) is int and maximum > 0:
            message += tr(" · 当前账号加权上限 {0}", maximum)
        self.danmaku_status.setText(message)
        clearable = status.get("clearable", 0)
        self.clear_button.setEnabled(bool(getattr(self.service, "danmaku_available", True)
                                          and type(clearable) is int and clearable > 0))
        automatic_verify = state == "auto_verifying"
        automatic_ai = state in {"ai_analyzing", "ai_repair_ready", "ai_retrying", "validating", "validation_storage_error"}
        transferred = state == "failure_transferred"
        automatic_retry = bool(status.get("f_retry_active", False) or state in {
            "platform_f", "waiting_f_retry", "retrying_f", "auto_rechecking", "auto_retry"})
        uncertain = state in {"unknown", "result_unknown"}
        can_resume = status.get("can_resume", state == "paused" or (
            status.get("blocked", False) and state in {
                "platform_rejected", "rejected", "unknown", "result_unknown"}))
        self.resume_button.setText(tr("自动核验中" if automatic_verify else "自动处理中" if automatic_ai else "自动重试中" if automatic_retry else
                                      "核验待定消息" if uncertain else "恢复推送"))
        self.resume_button.setEnabled(bool(enabled and can_resume and not automatic_retry and not automatic_verify and not automatic_ai and not transferred))
        self.resume_button.setToolTip(tr("正在自动只读核验，不会重复提交结果未知的消息。" if automatic_verify else
            "正在自动分析和处理当前失败消息，队列顺序保持。" if automatic_ai else
            "按发送间隔自动重试当前条，无需手动恢复。" if automatic_retry else
            "只读核验页面是否收到待定消息，不重新发送。" if uncertain else
            "先核实直播页和账号状态。发送结果未知时不会自动重发。"))

    def clear_pending(self):
        if not getattr(self.service, "danmaku_available", True):
            return False
        clearable = self._danmaku_snapshot.get("clearable", 0)
        if type(clearable) is not int or clearable <= 0:
            return False
        retained = self._danmaku_snapshot.get("retained_uncertain", 0)
        if type(retained) is not int or retained < 0:
            retained = 0
        answer = QMessageBox.question(self, tr("清空待发队列"),
            tr("将清理 {0} 段尚未发送的消息。已提交或结果未确认的 {1} 段消息保留，不会撤回已发送弹幕。是否继续？",
               clearable, retained), QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if answer != QMessageBox.StandardButton.Yes:
            return False
        self.service.clear_danmaku_queue()
        return True

    def login(self):
        if self._save():
            self.service.open("https://passport.bilibili.com/login", get_language())

    def open_room(self):
        try:
            url = room_url(self.room.text())
        except ValueError:
            self.status.setText(tr("直播间格式无效"))
            return
        if self._save():
            self.service.open(url, get_language())

    def open_validation_room(self):
        try:
            url = room_url(self.validation_room.text())
            # Only persist the test room; unsaved formal-room and push controls
            # must not be submitted by an explicit test-window action.
            self._saving_browser_controls = True
            self.service.save_preferences(self.service.room, self.service.auto_check,
                danmaku_validation_room=self.validation_room.text())
        except ValueError as error:
            self.status.setText(tr("AI验证直播间不能与正式直播间相同" if str(error) == "same_validation_room" else "AI验证直播间格式无效"))
            return False
        except OSError:
            self.status.setText(tr("浏览器设置保存失败"))
            return False
        finally:
            self._saving_browser_controls = False
        self.validation_room.setText(url.rsplit("/", 1)[-1])
        self.service.open(url, get_language())
        return True

    def on_event(self, event):
        kind = event.get("kind")
        messages = {
            "checking": "正在检查浏览器内核更新…", "starting": "正在启动内置浏览器…",
            "ready": "内置浏览器已打开", "closed": "浏览器已关闭，登录状态保留",
            "update_available": "本机已有新版内核，关闭并重新打开浏览器后生效",
        }
        if kind in messages:
            self.status.setText(tr(messages[kind]))
        elif kind == "error":
            self.status.setText(tr("浏览器启动或连接失败：{0}。请检查 WebView2 Evergreen 是否已安装。",
                                   str(event.get("code", "unknown"))))
        if kind in ("ready", "version"):
            self.available_version = event.get("available_version")
            self.versions.setText(tr("正在运行：{0} · 本机可用：{1}",
                event.get("active_version") or tr("未启动"), event.get("available_version") or tr("未确认")))
        elif kind == "closed":
            self.versions.setText(tr("正在运行：{0} · 本机可用：{1}",
                tr("未启动"), self.available_version or tr("未确认")))

    def on_snapshot(self, snapshot):
        self.versions.setText(tr("正在运行：{0} · 本机安装：{1} · 微软目录：{2}",
            self.service.active_version or tr("未启动"), snapshot.get("installed_version") or tr("未安装"),
            snapshot.get("catalog_version") or tr("未确认")))
        states = {
            "missing": "请先安装 WebView2 Evergreen 内核",
            "unknown": "尚未联网检查最新版",
            "check_unavailable": "联网版本检查未完成，不能确认是否为最新版",
            "new_available": "微软目录已有新版，Evergreen 将按微软更新策略更新",
            "restart_required": "本机已有新版内核，关闭并重新打开浏览器后生效",
            "current": "本机版本不低于微软公开目录版本",
        }
        self.status.setText(tr(states.get(snapshot.get("state"), "尚未联网检查最新版")))
