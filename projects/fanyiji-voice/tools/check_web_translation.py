"""Translate synthetic examples using the real embedded host and an isolated profile.

No audio, user settings, live-room login or danmaku are used by this check.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fanyiji.web_translation_engine import GoogleWebTranslationEngine, YoudaoWebTranslationEngine


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=("google_web", "youdao_web"), required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--show-window", action="store_true")
    args = parser.parse_args()
    engine_type = GoogleWebTranslationEngine if args.provider == "google_web" else YoudaoWebTranslationEngine
    engine = engine_type(profile=args.profile)
    rows = []
    try:
        engine.preload()
        if args.show_window:
            engine.show_window()
        samples = [("こんにちは。今日はいい天気ですね。", "ja", "zh"),
                   ("ゲームの音が大きすぎます。少し音量を下げます。", "ja", "zh"),
                   ("今天的游戏很有意思 谢谢大家来看直播", "zh", "ja"),
                   ("みなさん、見に来てくれてありがとうございます。", "ja", "zh")]
        for text, source, target in samples:
            started = time.monotonic()
            row = {"source": text, "source_lang": source, "target_lang": target}
            try:
                row.update(status="ok", translation=engine.translate(text, source, target))
            except Exception as exc:
                row.update(status="error", code=getattr(exc, "code", type(exc).__name__),
                           diagnostic=engine.last_diagnostic)
            row["seconds"] = round(time.monotonic() - started, 3)
            rows.append(row)
            print(json.dumps(row, ensure_ascii=True), flush=True)
            if row["status"] == "error":
                break
    except Exception as exc:
        rows.append({"status": "error", "code": getattr(exc, "code", type(exc).__name__)})
    finally:
        engine.close()
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({"provider": args.provider, "synthetic_text": True,
                                          "user_settings_written": False, "audio_capture": False,
                                          "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if len(rows) == 4 and all(row["status"] == "ok" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
