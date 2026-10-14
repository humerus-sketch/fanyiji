"""Bounded, explicitly invoked live check: three posts through the installed helper."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PySide6.QtWidgets import QApplication
from fanyiji.browser_service import BrowserService, room_url
from fanyiji.danmaku_service import DanmakuService
from fanyiji.danmaku_protocol import BODY_FIELDS, danmaku_weighted_length
from fanyiji.paths import data_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--room", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    room = room_url(args.room).rsplit("/", 1)[-1]
    executable = ROOT / "翻译姬" / "翻译姬.exe"
    if not executable.is_file():
        raise SystemExit("installed_executable_missing")
    args.output.mkdir(parents=True, exist_ok=False)
    app = QApplication([])

    class InstalledBrowser(BrowserService):
        def _start_browser(self):
            # Exercise the packaged entry point, not a Python-only test host.
            with patch("fanyiji.browser_service.sys.executable", str(executable)), patch(
                    "fanyiji.browser_service.sys.frozen", True, create=True):
                super()._start_browser()

    browser = InstalledBrowser()
    queue = DanmakuService(browser, store_path=args.output / "queue.sqlite3")
    results, refreshed = [], set()
    browser.danmaku_result.connect(results.append)
    prefix = datetime.now(timezone(timedelta(hours=8))).strftime("%H%M%S")
    # Even the official fallback limit of weighted 20 admits these payloads.
    # Keep ten body UTF-16 units so later items don't wait for short-text merging.
    texts = [f"【验证 {prefix}A1】", f"【验证{prefix}B2】", f"【验证{prefix}C3】"]

    def finish(event):
        if event["operation"] == "check" and event["request_id"].startswith("refresh_"):
            value = event["result"]
            if value.get("present") and value.get("room_matches") and value.get("chat_ready"):
                refreshed.add(int(event["request_id"].split("_")[1]))
    browser.danmaku_result.connect(finish)

    def feed(segment, text):
        base = {"session": 1, "channel": "system", "segment_id": segment, "revision": 1}
        queue.handle_event({**base, "kind": "original", "text": "这是已确认的测试原文"})
        queue.handle_event({**base, "kind": "translation", "text": text, "translation_stage": "final"})

    def pump_until(condition, seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            app.processEvents()
            if condition():
                return True
            time.sleep(0.025)
        return False

    report = {"room": room, "packaged_helper": True, "posts_planned": 3,
              "exe_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
              "body_field_names": list(BODY_FIELDS),
              "payload_weighted_lengths": [danmaku_weighted_length(text) for text in texts],
              "synthetic_final_events": True,
              "interval_seconds": 3.5, "started_at": datetime.now().astimezone().isoformat()}
    try:
        queue.begin_session(1)
        queue.configure(True, room, "system", 3.5)
        feed("short", "验证")
        assert queue.snapshot()["holding"] == 1
        feed("first", texts[0][1:-1].removeprefix("验证 "))
        assert queue.store.head()["text"] == texts[0]
        feed("duplicate-combined", texts[0][1:-1])
        assert queue.snapshot()["queued"] == 1
        feed("second", texts[1][1:-1])
        feed("third", texts[2][1:-1])
        browser.open(room)
        pump_until(lambda: queue.snapshot()["page_visible"] == 3 or queue.snapshot()["blocked"], 90)
        report["queue"] = queue.snapshot()
        report["active_browser_version"] = browser.active_version
        if queue.snapshot()["page_visible"] == 3:
            browser.refresh()
            pump_until(lambda: False, 4)
            for index, text in enumerate(texts):
                browser.request_danmaku("check", f"refresh_{index}", room, text)
                pump_until(lambda index=index: index in refreshed, 10)
        starts = [r["result"].get("request_started_epoch_ms") for r in results if r["operation"] == "send"]
        report["post_start_gaps_seconds"] = [(right-left)/1000 for left, right in zip(starts, starts[1:])
                                             if type(left) in (int, float) and type(right) in (int, float)]
        report["posts_actual"] = len(starts)
        report["refresh_visible"] = len(refreshed)
        report["results"] = results
        report["passed"] = (len(starts) == 3 and report["queue"]["page_visible"] == 3
                            and len(refreshed) == 3
                            and len(report["post_start_gaps_seconds"]) == 2
                            and min(report["post_start_gaps_seconds"]) >= 3.5)
    finally:
        queue.close()
        browser.close()
        report["finished_at"] = datetime.now().astimezone().isoformat()
        (args.output / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, ensure_ascii=False))
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
