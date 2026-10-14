"""Measure accepted captions at real Qt painting, without accounts or audio."""
import os
import time
from types import SimpleNamespace
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from fanyiji.config import AppConfig, SourceConfig
from fanyiji.app import AppController
from fanyiji.i18n import get_language, set_language, tr
from fanyiji.latency import LatencyTracker
from fanyiji.ui import MainWindow


@pytest.fixture
def displayed_window():
    app = QApplication.instance() or QApplication([])
    previous = get_language()
    set_language("zh-CN")
    window = MainWindow(AppConfig(translation_provider="hybrid"))
    window.set_presentation_session(7)
    window.set_original("system", "今は行かない。", "a", context_generation=3)
    window.show()
    for _ in range(5):
        app.processEvents()
    presented = []
    window.subtitle_presented.connect(presented.append)
    yield window, app, presented
    window.close()
    window.deleteLater()
    app.processEvents()
    set_language(previous)


def _presentation(**overrides):
    now = time.monotonic()
    return {"session": 7, "channel": "system", "segment_id": "a", "revision": 1,
            "context_generation": 3, "ui_received_at": now,
            "timing": {"audio_started_at": now - 2, "audio_ended_at": now - 1.5,
                       "translation_finished_at": now - .01}, **overrides}


def _translate(window, stage="final", **options):
    window.set_translation("system", "我现在不去。", "a", translation_stage=stage,
                           translation_provider="codex" if stage == "final" else "local",
                           context_generation=options.pop("context_generation", 3),
                           presentation=options.pop("presentation", _presentation()), **options)


def _paint(window, app):
    for _ in range(5):
        app.processEvents()
    row = window._source_panels["playback"]._rows.get("a")
    if row is not None:
        row.translation.repaint()
    app.processEvents()


def test_translation_is_counted_after_paint_not_after_set_text(displayed_window):
    window, app, presented = displayed_window
    _translate(window)
    assert presented == []
    _paint(window, app)
    assert len(presented) == 1
    event = presented[0]
    assert event["stage"] == event["translation_stage"] == "final"
    assert event["ui_received_at"] <= event["ui_updated_at"] <= event["ui_painted_at"]
    assert "text" not in event and "audio" not in event
    _paint(window, app)
    assert len(presented) == 1


def test_hidden_window_does_not_claim_caption_presentation(displayed_window):
    window, app, presented = displayed_window
    window.hide()
    _translate(window)
    _paint(window, app)
    assert presented == []
    window.show()
    _paint(window, app)
    assert len(presented) == 1


def test_overwritten_draft_is_not_reported_as_seen(displayed_window):
    window, app, presented = displayed_window
    _translate(window, "draft")
    _translate(window, "final")
    _paint(window, app)
    assert [event["stage"] for event in presented] == ["final"]


def test_visible_draft_then_final_are_measured_separately(displayed_window):
    window, app, presented = displayed_window
    _translate(window, "draft")
    _paint(window, app)
    _translate(window, "final")
    _paint(window, app)
    _translate(window, "draft")
    _paint(window, app)
    assert [event["stage"] for event in presented] == ["draft", "final"]


def test_provisional_final_waits_until_source_is_confirmed(displayed_window):
    window, app, presented = displayed_window
    window.set_original("system", "今は", "a", revision=2, provisional=True, context_generation=3)
    _translate(window, revision=2, provisional=True)
    _paint(window, app)
    assert presented == []
    label = window._source_panels["playback"]._rows["a"].translation
    pending = label.presentation
    updated_at, timing = pending["ui_updated_at"], dict(pending["timing"])
    window.finalize_subtitle("system", "a", 2)
    assert pending["ui_updated_at"] == updated_at
    assert pending["timing"] == timing
    first_confirmation = pending["source_confirmed_at"]
    assert pending["source_confirmation_wait_seconds"] == pytest.approx(first_confirmation - updated_at)
    window.finalize_subtitle("system", "a", 2)
    assert pending["source_confirmed_at"] == first_confirmation
    _paint(window, app)
    assert len(presented) == 1 and presented[0]["provisional"] is False
    assert presented[0]["timing"] == timing
    assert presented[0]["ui_updated_at"] == updated_at


@pytest.mark.parametrize("provider", ["codex", "local"])
@pytest.mark.parametrize("confirmation", ["finalize", "original"])
@pytest.mark.parametrize("language", ["zh-CN", "ja", "en"])
@pytest.mark.parametrize("with_timing", [False, True])
def test_unconfirmed_translation_label_restores_provider_on_confirmation(
        displayed_window, provider, confirmation, language, with_timing):
    window, app, _ = displayed_window
    set_language(language)
    window.set_original("system", "今は", "a", revision=2, provisional=True, context_generation=3)
    window.set_translation("system", "现在", "a", revision=2, provisional=True,
                           context_generation=3, translation_stage="final",
                           translation_provider=provider, translation_model="fixture-model",
                           presentation=_presentation() if with_timing else None)
    row = window._source_panels["playback"]._rows["a"]
    expected = "Codex 译文 · 等待后半句确认" if provider == "codex" else "译文 · 等待后半句确认"
    assert row.stage_label.text() == tr(expected)
    assert row.stage_label.toolTip() == "fixture-model"
    _paint(window, app)
    if confirmation == "finalize":
        window.finalize_subtitle("system", "a", 2)
    else:
        window.set_original("system", "今は", "a", revision=2, provisional=False, context_generation=3)
    assert row.stage_label.text() == tr("Codex 终稿" if provider == "codex" else "终稿")
    assert row.stage_label.toolTip() == "fixture-model"
    assert row.translation_provider == provider and row.translation_model == "fixture-model"


@pytest.mark.parametrize("options", [{"revision": 2}, {"context_generation": 2}, {"failed": True},
                                    {"overall_failed": True}])
def test_stale_and_failed_results_are_not_successful_presentations(displayed_window, options):
    window, app, presented = displayed_window
    _translate(window, **options)
    _paint(window, app)
    assert presented == []


def test_new_original_revision_invalidates_waiting_paint(displayed_window):
    window, app, presented = displayed_window
    _translate(window)
    window.set_original("system", "改訂", "a", revision=2, context_generation=3)
    _paint(window, app)
    assert presented == []


def test_pause_invalidates_unpainted_results_even_if_complete(displayed_window):
    window, app, presented = displayed_window
    _translate(window)
    window.cancel_pending()
    _paint(window, app)
    assert presented == []


def test_new_session_invalidates_pending_and_rejects_old_events(displayed_window):
    window, app, presented = displayed_window
    _translate(window)
    window.set_presentation_session(8)
    _paint(window, app)
    _translate(window)
    _paint(window, app)
    assert presented == []


def test_disabled_source_is_not_presented(displayed_window):
    window, app, presented = displayed_window
    window.update_config(AppConfig(microphone=SourceConfig(enabled=False)))
    window.set_original("mic", "fixture", "a")
    window.set_translation("mic", "fixture", "a", presentation=_presentation(channel="mic"))
    _paint(window, app)
    assert presented == []


def test_scrolled_out_label_waits_until_visible(displayed_window):
    window, app, presented = displayed_window
    for index in range(12):
        window.set_original("system", "長い原文 " * 15, str(index), context_generation=3)
    _translate(window)
    _paint(window, app)
    assert presented == []
    panel = window._source_panels["playback"]
    panel.scroll.verticalScrollBar().setValue(0)
    _paint(window, app)
    assert len(presented) == 1


@pytest.mark.parametrize("language", ["zh-CN", "ja", "en"])
def test_latency_summary_is_translated_and_preserves_controls(displayed_window, language):
    window, app, _ = displayed_window
    set_language(language)
    window.set_latency({"valid": True, "stage": "draft", "end_to_end_seconds": 3.1})
    window.set_latency({"valid": True, "stage": "final", "end_to_end_seconds": 4.2,
                        "post_audio_seconds": 1.7, "recognition_to_display_seconds": 1.5,
                        "breakdown": {"asr_seconds": .3, "translation_queue_seconds": .2,
                                      "translation_seconds": .7, "ui_paint_seconds": .01}})
    _paint(window, app)
    assert tr("最近暂译：全程 {0:.2f} 秒", 3.1) in window.latency_label.text()
    assert tr("最近终稿：全程 {0:.2f} 秒 · 句尾等待 {1:.2f} 秒", 4.2, 1.7) in window.latency_label.text()
    assert window.font_size.value() == 16 and window.opacity_slider.value() == 66
    tooltip = window.latency_label.toolTip()
    assert tr("识别开始到终稿：{0:.2f} 秒", 1.5) in tooltip
    assert tr("{0}：{1:.2f} 秒", tr("识别模型"), .3) in tooltip
    assert tr("{0}：{1:.2f} 秒", tr("翻译排队"), .2) in tooltip
    assert tr("{0}：{1:.2f} 秒", tr("翻译请求"), .7) in tooltip
    assert tr("{0}：{1:.2f} 秒", tr("界面绘制"), .01) in tooltip
    before = window.latency_label.text()
    window.set_latency({"valid": False, "stage": "final"})
    assert window.latency_label.text() == before


def test_controller_passes_timing_snapshot_without_source_text():
    window = SimpleNamespace(subtitle_presented=Mock(), set_translation=Mock())
    controller = SimpleNamespace(closed=False, epoch=7, window=window, danmaku=None, config=AppConfig())
    event = {"session": 7, "kind": "translation", "channel": "system", "segment_id": "a",
             "text": "private caption", "translation_stage": "final", "timing": {"asr_started_at": 1.0}}
    AppController._on_event(controller, event)
    presentation = window.set_translation.call_args.kwargs["presentation"]
    assert "text" not in presentation and "ui_received_at" in presentation
    event["timing"]["asr_started_at"] = 2.0
    assert presentation["timing"]["asr_started_at"] == 1.0


def test_controller_ignores_old_paint_before_logging_or_recording():
    window = SimpleNamespace(set_latency=Mock())
    tracker = Mock()
    metrics = {"valid": True, "stage": "final", "end_to_end_seconds": 2.0, "post_audio_seconds": .5}
    tracker.on_presented.return_value = metrics
    controller = SimpleNamespace(closed=False, epoch=7, window=window, latency=tracker, recorder=None, config=AppConfig())
    AppController._on_subtitle_presented(controller, {"session": 6})
    tracker.on_presented.assert_not_called()
    AppController._on_subtitle_presented(controller, {"session": 7})
    window.set_latency.assert_called_once_with(metrics)


def test_actual_tracker_measures_audio_to_painted_final(displayed_window):
    window, app, _ = displayed_window
    tracker = LatencyTracker()
    tracker.begin_session(7)
    controller = SimpleNamespace(closed=False, epoch=7, window=window, latency=tracker, recorder=None, config=AppConfig())
    window.subtitle_presented.connect(lambda event: AppController._on_subtitle_presented(controller, event))
    now = time.monotonic()
    timing = {name: now + offset for name, offset in {
        "audio_started_at": -2, "audio_ended_at": -1.5, "segment_ready_at": -1.4,
        "recognition_dequeued_at": -1.35, "filter_started_at": -1.3, "filter_finished_at": -1.25,
        "asr_started_at": -1.2, "asr_finished_at": -.8, "translation_queued_at": -.7,
        "translation_started_at": -.6, "translation_finished_at": -.1}.items()}
    _translate(window, presentation=_presentation(timing=timing))
    _paint(window, app)
    summary = tracker.summary()
    assert summary["final_count"] == 1 and summary["draft_count"] == 0
    assert summary["final_p50"] >= 2 and summary["after_end_p50"] >= 1.5
    assert not window.latency_label.isHidden()
    assert "最近终稿" in window.latency_label.text()
