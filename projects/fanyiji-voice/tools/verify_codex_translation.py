"""Explicit subscription-only smoke test with public, fixed text; no audio capture."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fanyiji.codex_engine import CodexTranslationEngine


CASES = (
    ("ja", "zh", "明日は行けません。来週の金曜日なら大丈夫です。", ""),
    ("ja", "zh", "三百元を返して、二百元は残してください。", ""),
    ("zh", "ja", "他并不是不想去，只是现在没有时间。", ""),
)


def main() -> int:
    parser = argparse.ArgumentParser(description="使用 Codex 订阅额度验证三个固定句子的双向文字翻译")
    parser.add_argument("--model", default="")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("报告已存在，请指定新文件；不会覆盖旧评测。")
    report = {"audio_capture": False, "api_key_allowed": False,
              "scope": "三个固定句子的真实云端文字翻译；不代表直播识别准确率或端到端延迟",
              "results": [], "passed": False}
    engine = CodexTranslationEngine(model=args.model)
    started = time.monotonic()
    try:
        engine.preload()
        report["startup_seconds"] = round(time.monotonic() - started, 3)
        metadata = engine.backend_info
        report["backend"] = {key: metadata.get(key) for key in
                             ("backend", "model_id", "auth_mode", "requires_network", "subscription_only")}
        for source_lang, target_lang, text, context in CASES:
            began = time.monotonic()
            translation = engine.translate(text, source_lang, target_lang, context=context)
            row = {"source_lang": source_lang, "target_lang": target_lang, "original": text,
                   "translation": translation, "translation_seconds": round(time.monotonic() - began, 3)}
            report["results"].append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
        report["passed"] = len(report["results"]) == len(CASES)
    except Exception as exc:
        # Engine errors deliberately contain status guidance, never raw account data.
        report["error"] = str(exc)
        report["error_type"] = type(exc).__name__
    finally:
        engine.close()
    report["total_seconds"] = round(time.monotonic() - started, 3)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "output": str(args.output),
                      "error": report.get("error")}, ensure_ascii=False), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
