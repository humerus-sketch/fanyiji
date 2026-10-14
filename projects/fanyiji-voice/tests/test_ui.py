"""UI state tests; no settings writes, model loading, or audio capture."""

import os
from dataclasses import replace
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QComboBox, QDialog, QLabel, QPushButton

from fanyiji.config import AppConfig, SourceConfig
from fanyiji.models_download import ModelCatalog
from fanyiji.identity import APP_NAME
from fanyiji.ui import MainWindow, SettingsDialog, loaded_model_display, loaded_models_summary


CODEX_MODELS = [
    {"model": "gpt-6.1-sol", "displayName": "GPT-6.1 Sol",
     "supportedReasoningEfforts": [{"reasoningEffort": "low"}, {"reasoningEffort": "high"}]},
    {"model": "gpt-6-luna", "displayName": "GPT-6 Luna",
     "supportedReasoningEfforts": [{"reasoningEffort": "low"}, {"reasoningEffort": "medium"}]},
    {"model": "gpt-6-astra", "displayName": "GPT-6 Astra",
     "supportedReasoningEfforts": ["medium", "high", "max"]},
]


class UiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def tearDown(self):
        for widget in self.app.topLevelWidgets():
            widget.close()
            widget.deleteLater()
        self.app.processEvents()

    def test_multilingual_defaults_on_and_saved_choice_preserves_channel_targets(self):
        dialog = SettingsDialog()
        self.assertTrue(dialog.multilingual_asr.isChecked())
        dialog.multilingual_asr.setChecked(False)
        dialog.accept()
        self.assertFalse(dialog.result_config.multilingual_asr)
        self.assertEqual(dialog.result_config.playback.target_lang, "zh")
        restored = SettingsDialog(dialog.result_config)
        self.assertFalse(restored.multilingual_asr.isChecked())
        restored.multilingual_asr.setChecked(True)
        restored.accept()
        self.assertTrue(restored.result_config.multilingual_asr)
        self.assertEqual(restored.result_config.playback.direction, "ja_to_zh")

    def test_product_name_is_used_in_the_window_heading_and_settings(self):
        window = MainWindow(AppConfig())
        self.assertEqual(window.windowTitle(), APP_NAME)
        self.assertIn(APP_NAME, [label.text() for label in window.findChildren(QLabel)])
        self.assertEqual(SettingsDialog(AppConfig()).windowTitle(), APP_NAME + " · 字幕设置")
        self.assertEqual(SettingsDialog(first_run=True).windowTitle(), APP_NAME + " · 首次设置")

    def test_audio_processing_mode_survives_unrelated_settings_edit(self):
        for mode in ("off", "gtcrn", "uvr"):
            dialog = SettingsDialog(replace(AppConfig(), audio_processing=mode))
            dialog.multilingual_asr.setChecked(False)
            dialog.accept()
            self.assertEqual(dialog.result_config.audio_processing, mode)

    def test_audio_processing_summary_distinguishes_raw_fallback(self):
        self.assertIn("GTCRN 语音增强", loaded_models_summary(
            {"audio_processing": {"mode": "gtcrn", "model_loaded": True}}))
        self.assertNotIn("原音回退", loaded_models_summary(
            {"audio_processing": {"mode": "gtcrn", "model_loaded": True}}))
        self.assertIn("原音回退", loaded_models_summary(
            {"audio_processing": {"mode": "uvr", "model_loaded": False}}))

    def test_audio_processing_entry_emits_main_window_signal(self):
        window = MainWindow(AppConfig())
        opened = []
        window.audio_processing_requested.connect(lambda: opened.append(True))
        window.audio_processing_button.click()
        self.assertEqual(opened, [True])

    def test_model_summary_displays_actual_recognition_policy(self):
        info = {"model_kind": "qwen3_asr", "device": "cpu", "recognition_mode": "auto"}
        self.assertIn("识别语言：自动多语", loaded_models_summary({"asr": info}))
        info["recognition_mode"] = "fixed"
        self.assertIn("识别语言：按通道方向", loaded_models_summary({"asr": info}))
        info["device"] = "not_loaded"
        self.assertNotIn("识别语言", loaded_models_summary({"asr": info}))

    def test_imported_model_name_is_shown_only_after_the_backend_is_loaded(self):
        info = {"model_kind": "qwen3_asr", "model_id": "local-test-import", "device": "cpu",
                "display_name": "我的日语直播模型"}
        name, device, model_id = loaded_model_display(info, "asr")
        self.assertEqual((name, device, model_id), ("我的日语直播模型", "CPU", "local-test-import"))
        self.assertIn("我的日语直播模型", loaded_models_summary({"asr": info}))
        info["device"] = "not_loaded"
        self.assertNotIn("我的日语直播模型", loaded_models_summary({"asr": info}))

    def test_codex_name_is_not_overridden_by_local_import_alias(self):
        info = {"backend": "codex_cli", "model_id": "gpt-6-luna", "device": "cloud",
                "display_name": "本地模型名字"}
        name, _device, identifier = loaded_model_display(info, "translation")
        self.assertIn("gpt-6-luna", name)
        self.assertNotIn("本地模型名字", name)
        self.assertIsNone(identifier)

    def test_imported_translation_is_identified_by_id_even_with_the_preset_filename(self):
        catalog = ModelCatalog.default()
        preset = next(entry for entry in catalog.entries if entry.model_kind == "hy")
        local_id = "local-" + "b" * 32
        imported = replace(preset, id=local_id, name="自定义直播模型",
                           files=tuple(replace(file, relative_path="translation/imported/" + local_id + "/" +
                                               file.relative_path.split("/")[-1]) for file in preset.files))
        catalog = replace(catalog, entries=(*catalog.entries, imported))
        info = {"model_kind": "hy", "model_id": local_id, "model_file": imported.files[0].relative_path,
                "display_name": imported.name, "device": "cpu"}
        self.assertEqual(loaded_model_display(info, "translation", catalog),
                         (imported.name, "CPU", local_id))
        # MainWindow uses the base catalog; it must still retain the actual local ID.
        self.assertEqual(loaded_model_display(info, "translation")[2], local_id)


    def test_first_run_cancel_does_not_produce_config(self):
        dialog = SettingsDialog(None, first_run=True)
        self.assertIsNone(dialog.result_config)
        self.assertTrue(dialog.microphone_enabled.isChecked())
        self.assertEqual(dialog.microphone_direction.currentData(), "zh_to_ja")
        self.assertEqual(dialog.playback_direction.currentData(), "ja_to_zh")
        dialog.reject()
        self.assertEqual(dialog.result(), QDialog.DialogCode.Rejected)
        self.assertIsNone(dialog.result_config)

    def test_sources_can_share_direction_and_both_be_disabled(self):
        original = AppConfig()
        dialog = SettingsDialog(original)
        dialog.playback_direction.setCurrentIndex(
            dialog.playback_direction.findData("zh_to_ja")
        )
        dialog.accept()
        self.assertEqual(dialog.result_config.microphone.direction, "zh_to_ja")
        self.assertEqual(dialog.result_config.playback.direction, "zh_to_ja")
        self.assertEqual(original.playback.direction, "ja_to_zh")

        disabled = SettingsDialog(dialog.result_config)
        disabled.microphone_enabled.setChecked(False)
        disabled.playback_enabled.setChecked(False)
        disabled.accept()
        self.assertFalse(disabled.result_config.microphone.enabled)
        self.assertFalse(disabled.result_config.playback.enabled)
        self.assertTrue(disabled.result_config.setup_complete)

    def test_speech_only_defaults_true_and_saved_toggle_preserves_sources(self):
        dialog = SettingsDialog(None, first_run=True)
        self.assertTrue(dialog.speech_only.isChecked())
        original = AppConfig(microphone=SourceConfig(enabled=False), speech_only=False)
        dialog = SettingsDialog(original)
        self.assertFalse(dialog.speech_only.isChecked())
        dialog.speech_only.setChecked(True)
        dialog.accept()
        self.assertTrue(dialog.result_config.speech_only)
        self.assertEqual(dialog.result_config.microphone, original.microphone)
        self.assertEqual(dialog.result_config.playback, original.playback)

    def test_first_run_defaults_local_and_codex_options_follow_provider(self):
        dialog = SettingsDialog(None, first_run=True)
        dialog.show()
        self.app.processEvents()
        self.assertEqual(dialog.translation_provider.currentData(), "local")
        self.assertFalse(dialog.codex_options.isVisible())
        dialog.translation_provider.setCurrentIndex(dialog.translation_provider.findData("codex"))
        self.app.processEvents()
        self.assertTrue(dialog.codex_options.isVisible())
        self.assertEqual(dialog.codex_model.currentData(), "")
        self.assertIn("原文", dialog.codex_notice.text())
        self.assertIn("订阅额度", dialog.codex_notice.text())
        dialog.accept()
        self.assertEqual(dialog.result_config.translation_provider, "codex")
        self.assertEqual(dialog.result_config.codex_model, "")

    def test_saved_codex_settings_restore_and_local_switch_keeps_model_and_sources(self):
        original = AppConfig(microphone=SourceConfig(False, "ja_to_zh"),
                             playback=SourceConfig(True, "zh_to_ja"), speech_only=False,
                             translation_provider="codex", codex_model="gpt-account-model")
        dialog = SettingsDialog(original)
        dialog.show()
        self.app.processEvents()
        self.assertEqual(dialog.translation_provider.currentData(), "codex")
        self.assertTrue(dialog.codex_options.isVisible())
        self.assertEqual(dialog.codex_model.currentData(), "gpt-account-model")
        dialog.translation_provider.setCurrentIndex(dialog.translation_provider.findData("local"))
        self.app.processEvents()
        self.assertFalse(dialog.codex_options.isVisible())
        dialog.accept()
        updated = dialog.result_config
        self.assertEqual(updated.translation_provider, "local")
        self.assertEqual(updated.codex_model, original.codex_model)
        self.assertEqual(updated.microphone, original.microphone)
        self.assertEqual(updated.playback, original.playback)
        self.assertFalse(updated.speech_only)

    def test_codex_catalog_is_non_editable_and_saves_selected_model_effort_speed(self):
        dialog = SettingsDialog(AppConfig(translation_provider="codex"), codex_models=CODEX_MODELS)
        self.assertIsInstance(dialog.codex_model, QComboBox)
        self.assertFalse(dialog.codex_model.isEditable())
        self.assertEqual(dialog.codex_model.currentData(), "")
        self.assertIn("综合候选", dialog.codex_model.itemText(dialog.codex_model.findData("gpt-6.1-sol")))
        dialog.codex_model.setCurrentIndex(dialog.codex_model.findData("gpt-6.1-sol"))
        self.assertEqual([dialog.codex_effort.itemData(index) for index in range(dialog.codex_effort.count())],
                         ["", "low", "high"])
        dialog.codex_effort.setCurrentIndex(dialog.codex_effort.findData("low"))
        dialog.codex_speed.setCurrentIndex(dialog.codex_speed.findData("fast"))
        dialog.accept()
        self.assertEqual(dialog.result_config.codex_model, "gpt-6.1-sol")
        self.assertEqual(dialog.result_config.codex_effort, "low")
        self.assertEqual(dialog.result_config.codex_speed, "fast")
        restored = SettingsDialog(dialog.result_config, codex_models=CODEX_MODELS)
        self.assertEqual(restored.codex_model.currentData(), "gpt-6.1-sol")
        self.assertEqual(restored.codex_effort.currentData(), "low")
        self.assertEqual(restored.codex_speed.currentData(), "fast")

    def test_codex_refresh_preserves_unknown_saved_choices_without_inventing_models(self):
        config = AppConfig(translation_provider="codex", codex_model="gpt-previous-model", codex_effort="xhigh")
        dialog = SettingsDialog(config)
        self.assertEqual(dialog.codex_model.count(), 2)
        self.assertIn("暂未验证", dialog.codex_model.currentText())
        self.assertIn("刷新", dialog.codex_models_status.text())
        dialog.set_codex_models(CODEX_MODELS)
        self.assertEqual(dialog.codex_model.currentData(), config.codex_model)
        self.assertEqual(dialog.codex_effort.currentData(), config.codex_effort)
        self.assertIn("暂未验证", dialog.codex_effort.currentText())
        dialog.set_codex_models([], error="连接失败")
        self.assertEqual(dialog.codex_model.currentData(), config.codex_model)
        self.assertEqual(dialog.codex_effort.currentData(), config.codex_effort)
        self.assertIn("连接失败", dialog.codex_models_status.text())
        dialog.accept()
        self.assertEqual(dialog.result_config.codex_model, config.codex_model)
        self.assertEqual(dialog.result_config.codex_effort, config.codex_effort)

    def test_codex_model_change_resets_unsupported_effort_and_keeps_supported_effort(self):
        dialog = SettingsDialog(AppConfig(codex_model="gpt-6.1-sol", codex_effort="high"),
                                codex_models=CODEX_MODELS)
        dialog.codex_model.setCurrentIndex(dialog.codex_model.findData("gpt-6-astra"))
        self.assertEqual(dialog.codex_effort.currentData(), "high")
        dialog.codex_model.setCurrentIndex(dialog.codex_model.findData("gpt-6-luna"))
        self.assertEqual(dialog.codex_effort.currentData(), "")
        self.assertEqual(dialog.codex_effort.findData("high"), -1)
        dialog.codex_effort.setCurrentIndex(dialog.codex_effort.findData("medium"))
        dialog.codex_model.setCurrentIndex(dialog.codex_model.findData(""))
        self.assertEqual(dialog.codex_effort.currentData(), "")
        dialog.accept()
        self.assertEqual(dialog.result_config.codex_effort, "")

    def test_codex_refresh_requests_controller_and_loading_blocks_duplicate_request(self):
        dialog = SettingsDialog(AppConfig(translation_provider="codex"))
        requests = []
        dialog.codex_models_requested.connect(lambda: requests.append(True))
        dialog.codex_models_refresh.click()
        self.assertEqual(requests, [True])
        dialog.set_codex_models_loading()
        dialog.codex_models_refresh.click()
        self.assertEqual(requests, [True])
        self.assertIn("正在读取", dialog.codex_models_status.text())
        dialog.set_codex_models(CODEX_MODELS)
        self.assertTrue(dialog.codex_models_refresh.isEnabled())
        self.assertIn("3 个", dialog.codex_models_status.text())

    def test_codex_local_switch_keeps_tuning_and_ignores_unavailable_catalog_efforts(self):
        config = AppConfig(translation_provider="codex", codex_model="gpt-6.1-sol",
                           codex_effort="xhigh", codex_speed="fast")
        dialog = SettingsDialog(config, codex_models=CODEX_MODELS)
        self.assertEqual(dialog.codex_effort.currentData(), "xhigh")
        self.assertIn("暂未验证", dialog.codex_effort.currentText())
        dialog.translation_provider.setCurrentIndex(dialog.translation_provider.findData("local"))
        dialog.accept()
        self.assertEqual(dialog.result_config.codex_model, config.codex_model)
        self.assertEqual(dialog.result_config.codex_effort, config.codex_effort)
        self.assertEqual(dialog.result_config.codex_speed, config.codex_speed)

    def test_codex_model_line_uses_actual_metadata_and_preserves_disabled_channels(self):
        window = MainWindow(AppConfig(microphone=SourceConfig(enabled=False),
                                     translation_provider="codex", codex_model="configured-model"))
        window.show()
        window.set_model_state("loading")
        self.assertNotIn("configured-model", window.models_label.text())
        window.set_model_state("ready", {
            "asr": {"model_id": "sensevoice-int8", "model_kind": "sense_voice", "device": "cpu"},
            "translation": {"backend": "codex_cli", "model_kind": "codex", "model_id": "actual-model",
                            "device": "cloud", "auth_mode": "chatgpt", "supports_source_context": True,
                            "reasoning_effort": "low", "speed": "fast"},
        })
        self.assertIn("SenseVoice INT8", window.models_label.text())
        self.assertIn("Codex CLI · actual-model", window.models_label.text())
        self.assertIn("云端", window.models_label.text())
        self.assertIn("ChatGPT 订阅", window.models_label.text())
        self.assertIn("推理：低（low）", window.models_label.text())
        self.assertIn("速度：Fast", window.models_label.text())
        self.assertNotIn("configured-model", window.models_label.text())
        self.assertFalse(window._source_panels["microphone"].isVisible())
        self.assertTrue(window._source_panels["playback"].isVisible())
        window.set_model_state("ready", {"translation": {"backend": "codex_cli", "device": "cloud"}})
        self.assertIn("模型待确认", window.models_label.text())
        self.assertIn("推理：待确认", window.models_label.text())
        self.assertIn("速度：待确认", window.models_label.text())

    def test_loaded_model_line_uses_device_and_never_marks_selection_as_loaded(self):
        window = MainWindow(AppConfig())
        self.assertEqual(window.models_label.text(), "模型：未加载")
        window.set_model_state("loading")
        self.assertIn("加载中", window.models_label.text())
        window.set_model_state("failed")
        self.assertIn("加载失败", window.models_label.text())
        self.assertNotIn("Qwen", window.models_label.text())
        loaded = {"asr": {"model_id": "sensevoice-int8", "model_kind": "sense_voice", "device": "cpu"},
                  "translation": {"model_file": "C:/models/HY-MT2-7B-Q6_K.gguf", "model_kind": "hy", "device": "cuda"}}
        window.set_model_state("ready", loaded)
        self.assertIn("SenseVoice INT8", window.models_label.text())
        self.assertIn("Hy-MT2 7B", window.models_label.text())
        self.assertIn("CPU", window.models_label.text())
        self.assertIn("CUDA", window.models_label.text())
        window.set_model_state("ready", {"translation": {"model_kind": "m2m100", "device": "cpu"}})
        self.assertIn("M2M100 418M", window.models_label.text())

    def test_model_line_does_not_reveal_disabled_source_panels(self):
        window = MainWindow(AppConfig(microphone=SourceConfig(enabled=False), playback=SourceConfig(enabled=False)))
        window.show()
        window.set_model_state("ready", {"translation": {"model_kind": "m2m100", "device": "cpu"}})
        self.assertTrue(window.models_label.isVisible())
        self.assertTrue(window.idle_label.isVisible())
        self.assertTrue(all(not panel.isVisible() for panel in window._source_panels.values()))

    def test_speech_filter_line_requires_confirmed_loaded_metadata(self):
        window = MainWindow(AppConfig(speech_only=True))
        window.set_model_state("ready", {"speech_filter": {"enabled": True, "model_loaded": False}})
        self.assertNotIn("筛选：已启用", window.models_label.text())
        window.set_model_state("ready", {"speech_filter": {"enabled": True, "model_loaded": True}})
        self.assertIn("讲话筛选：已启用", window.models_label.text())
        window.set_model_state("ready", {"speech_filter": {"enabled": False, "model_loaded": True}})
        self.assertIn("讲话筛选：关闭", window.models_label.text())
        self.assertNotIn("筛选：已启用", window.models_label.text())

    def test_late_translation_updates_matching_segment(self):
        window = MainWindow(AppConfig())
        window.set_original("mic", "第一句话", 1)
        window.set_original("microphone", "第二句话", 2)
        window.set_translation("mic", "最初の文", 1)
        rows = window._source_panels["microphone"]._rows
        self.assertEqual(list(rows), [1, 2])
        self.assertEqual(rows[1].translation.text(), "最初の文")
        self.assertEqual(rows[2].original.text(), "第二句话")
        self.assertEqual(rows[2].translation.text(), "翻译中…")

    def test_old_revision_cannot_overwrite_repaired_sentence(self):
        window = MainWindow(AppConfig())
        window.set_original("system", "森の。", "a", revision=1, provisional=True)
        window.set_translation("system", "森林的。", "a", revision=1, provisional=True)
        window.set_original("system", "森の妖精です。", "a", revision=2)
        window.set_translation("system", "迟到的旧译文", "a", revision=1)
        row = window._source_panels["playback"]._rows["a"]
        self.assertEqual(row.translation.text(), "翻译中…")
        window.set_translation("system", "是森林精灵。", "a", revision=2)
        window.set_original("system", "旧原文", "a", revision=1)
        self.assertEqual(row.original.text(), "森の妖精です。")
        self.assertEqual(row.translation.text(), "是森林精灵。")

    def test_pending_confirmation_before_and_after_translation(self):
        for confirmation_first in (True, False):
            window = MainWindow(AppConfig())
            window.set_original("system", "森の。", "a", provisional=True)
            if confirmation_first:
                window.finalize_subtitle("system", "a", 1)
            window.set_translation("system", "森林的。", "a", provisional=True)
            if not confirmation_first:
                window.finalize_subtitle("system", "a", 1)
            panel = window._source_panels["playback"]
            self.assertTrue(panel._rows["a"].completed)
            self.assertEqual(list(panel._completion_order), ["a"])

    def test_pause_cancels_repair_of_previously_completed_row(self):
        window = MainWindow(AppConfig())
        window.set_original("system", "森の。", "a")
        window.set_translation("system", "森林的。", "a")
        window.set_original("system", "森の妖精。", "a", revision=2)
        window.cancel_pending()
        window.set_translation("system", "是森林精灵。", "a", revision=2)
        self.assertEqual(window._source_panels["playback"]._rows["a"].translation.text(), "已暂停，此片段已取消")

    def test_uncertain_recognition_is_visible_and_clears_after_update(self):
        window = MainWindow(AppConfig())
        window.set_original("system", "待核对原文", "a", uncertain=True)
        row = window._source_panels["playback"]._rows["a"]
        self.assertIn("核对", row.original.toolTip())
        self.assertIn("#e4b96a", row.original.styleSheet())
        window.set_original("system", "新原文", "a", revision=2)
        self.assertEqual(row.original.toolTip(), "")

    def test_pending_segment_survives_history_pruning_and_late_completion(self):
        window = MainWindow(AppConfig())
        window.set_original("system", "遅い文", "pending")
        for index in range(8):
            window.set_original("system", f"文{index}", index)
            window.set_translation("system", f"句{index}", index)
        panel = window._source_panels["playback"]
        self.assertIn("pending", panel._rows)
        self.assertEqual(len(panel._rows), 6)
        window.set_translation("system", "迟到的句子", "pending")
        self.assertEqual(panel._rows["pending"].translation.text(), "迟到的句子")
        self.assertEqual(len(panel._rows), 5)
        self.assertEqual(next(iter(panel._rows)), "pending")

    def test_config_update_clears_rows_and_ignores_old_translation(self):
        window = MainWindow(AppConfig())
        window.set_original("mic", "旧会话", "old")
        window.update_config(AppConfig())
        window.set_translation("mic", "古いセッション", "old")
        self.assertEqual(len(window._source_panels["microphone"]._rows), 0)
        window.set_original("playback", "新しい文", "new")
        window.set_translation("playback", "新的句子", "new")
        self.assertEqual(
            window._source_panels["playback"]._rows["new"].translation.text(), "新的句子"
        )

    def test_disabled_source_ignores_subtitles_and_levels(self):
        window = MainWindow(AppConfig(microphone=SourceConfig(enabled=False)))
        window.set_original("mic", "不应采集", 1)
        window.set_translation("mic", "出力しない", 1)
        window.set_level("mic", 0.8)
        panel = window._source_panels["microphone"]
        self.assertEqual(len(panel._rows), 0)
        self.assertEqual(panel.level.value(), 0)
        self.assertEqual(panel.status_label.text(), "已关闭")

    def test_single_enabled_source_fills_space_and_hides_other_panel(self):
        window = MainWindow(AppConfig())
        window.show()
        self.app.processEvents()
        microphone = window._source_panels["microphone"]
        playback = window._source_panels["playback"]
        dual_height = microphone.height()

        window.update_config(AppConfig(playback=SourceConfig(enabled=False)))
        self.app.processEvents()
        self.assertTrue(microphone.isVisible())
        self.assertFalse(playback.isVisible())
        self.assertFalse(playback.title_label.isVisible())
        self.assertFalse(playback.status_label.isVisible())
        self.assertFalse(window.idle_label.isVisible())
        self.assertGreater(microphone.height(), dual_height * 1.5)

    def test_default_wide_window_keeps_dual_source_scroll_areas_inside_panels(self):
        from fanyiji.i18n import get_language, set_language, tr

        previous_language = get_language()
        try:
            for language in ("zh-CN", "ja", "en"):
                with self.subTest(language=language):
                    set_language(language)
                    window = MainWindow(AppConfig(ui_language=language))
                    try:
                        window.append_notice(tr("界面语言已保存，重新启动软件后生效。"))
                        window.show()
                        self.app.processEvents()
                        self.assertEqual((window.width(), window.height()), (1237, 530))
                        self.assertTrue(window.notice_label.isVisible())
                        for source_id, panel in window._source_panels.items():
                            self.assertTrue(panel.isVisible(), source_id)
                            self.assertTrue(panel.rect().contains(panel.scroll.geometry()),
                                            f"{source_id}: scroll area is clipped by its panel")
                    finally:
                        window.close()
        finally:
            set_language(previous_language)

    def test_both_disabled_show_only_idle_and_keep_settings_available(self):
        window = MainWindow(
            AppConfig(
                microphone=SourceConfig(enabled=False),
                playback=SourceConfig(enabled=False),
            )
        )
        window.show()
        self.app.processEvents()
        for panel in window._source_panels.values():
            self.assertFalse(panel.isVisible())
            self.assertFalse(panel.title_label.isVisible())
            self.assertFalse(panel.status_label.isVisible())
        self.assertTrue(window.idle_label.isVisible())
        settings_button = next(
            button for button in window.findChildren(QPushButton)
            if button.text() == "更改设置"
        )
        self.assertTrue(settings_button.isVisible())
        self.assertTrue(settings_button.isEnabled())
        requests = []
        window.settings_requested.connect(lambda: requests.append(True))
        settings_button.click()
        self.assertEqual(requests, [True])

    def test_config_update_switches_visible_sources_and_restores_both(self):
        window = MainWindow(AppConfig(playback=SourceConfig(enabled=False)))
        window.show()
        self.app.processEvents()
        microphone = window._source_panels["microphone"]
        playback = window._source_panels["playback"]
        self.assertTrue(microphone.isVisible())
        self.assertFalse(playback.isVisible())

        window.update_config(AppConfig(microphone=SourceConfig(enabled=False)))
        self.app.processEvents()
        self.assertFalse(microphone.isVisible())
        self.assertTrue(playback.isVisible())
        self.assertFalse(window.idle_label.isVisible())

        window.update_config(AppConfig())
        self.app.processEvents()
        self.assertTrue(microphone.isVisible())
        self.assertTrue(playback.isVisible())
        self.assertFalse(window.idle_label.isVisible())

    def test_unconfigured_window_hides_sources_until_config_is_applied(self):
        window = MainWindow()
        window.show()
        self.app.processEvents()
        self.assertTrue(window.idle_label.isVisible())
        self.assertTrue(all(panel.isHidden() for panel in window._source_panels.values()))
        window.update_config(AppConfig())
        self.app.processEvents()
        self.assertFalse(window.idle_label.isVisible())
        self.assertTrue(all(panel.isVisible() for panel in window._source_panels.values()))

    def test_pause_cancels_pending_without_clearing_completed_captions(self):
        window = MainWindow(AppConfig())
        window.set_original("mic", "已经完成", 1)
        window.set_translation("mic", "完了した文", 1)
        window.set_original("mic", "还未完成", 2)
        window.set_level("mic", 0.7)
        window.cancel_pending()
        panel = window._source_panels["microphone"]
        self.assertEqual(panel._rows[1].translation.text(), "完了した文")
        self.assertEqual(panel._rows[2].translation.text(), "已暂停，此片段已取消")
        self.assertEqual(panel.level.value(), 0)
        window.set_translation("mic", "遅れた結果", 2)
        self.assertEqual(panel._rows[2].translation.text(), "已暂停，此片段已取消")


if __name__ == "__main__":
    unittest.main()
