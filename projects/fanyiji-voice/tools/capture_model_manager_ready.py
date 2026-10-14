"""Capture real, already-ready Qt controls without activating any selection.

The offscreen Qt plugin requires an explicitly registered font on this host.
Profiles and model assets are inspected before and after; no model is loaded.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

from validate_model_manager_ready import capture_asr, file_identities, open_verified, profiles, snapshot, wait_until
from PySide6.QtGui import QFont, QFontDatabase, QRawFont
from PySide6.QtWidgets import QApplication


def capture(root: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    report = {"status": "starting", "started_at": datetime.now().astimezone().isoformat(),
              "models_root": str(root), "qt_platform": "offscreen",
              "native_inference_performed": False, "audio_devices_opened": False,
              "network_attempts": 0, "profile_activations": 0}

    def prohibit_network(event, args):
        if event == "socket.connect":
            report["network_attempts"] += 1
            raise RuntimeError("Network access is prohibited in the local screenshot check")
        if event == "subprocess.Popen":
            raise RuntimeError("Native process launch is prohibited in the local screenshot check")

    sys.addaudithook(prohibit_network)
    application = QApplication([])
    application.setQuitOnLastWindowClosed(False)
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    font_id = QFontDatabase.addApplicationFont(str(font_path))
    families = QFontDatabase.applicationFontFamilies(font_id)
    if font_id < 0 or not families:
        raise RuntimeError("Could not register the installed Microsoft YaHei system font")
    font = QFont(families[0], 9)
    application.setFont(font)
    raw_font = QRawFont.fromFont(font)
    glyphs = raw_font.glyphIndexesForString("本地模型已下载校验通过SenseVoice")
    if not glyphs or any(glyph == 0 for glyph in glyphs):
        raise RuntimeError("The registered font does not contain all evidence text glyphs")
    report["registered_font"] = {"path": str(font_path), "families": list(families),
                                 "glyph_probe_all_present": True}
    profile_bytes = {name: (root / name / "backend.json").read_bytes()
                     for name in ("translation", "asr")}
    report["profiles_before"] = profiles(root)
    dialogs = []
    try:
        dialog, elapsed = open_verified(application, root, dialogs)
        before_files = file_identities(dialog)
        dialog.resize(1150, 1500)
        application.processEvents()
        report["verification_seconds"] = elapsed
        report["ready_state"] = snapshot(dialog)
        capture_asr(application, dialog, output / "models-ready-asr-font-selected.png")
        capture_asr(application, dialog, output / "models-ready-asr-font-top.png", top=True)
        report["weight_file_identities_unchanged"] = file_identities(dialog) == before_files
        report["profiles_after"] = profiles(root)
        report["profile_bytes_unchanged"] = all(
            (root / name / "backend.json").read_bytes() == original
            for name, original in profile_bytes.items())
        if not report["profile_bytes_unchanged"] or not report["weight_file_identities_unchanged"]:
            raise RuntimeError("The read-only screenshot check observed changed local assets")
        report["status"] = "completed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        raise
    finally:
        for dialog in dialogs:
            if dialog.is_busy:
                wait_until(lambda: not dialog.is_busy)
            dialog.reject()
        report["all_workers_finished"] = all(not dialog.is_busy for dialog in dialogs)
        report["native_modules_imported"] = [name for name in ("sherpa_onnx", "torch", "pyaudiowpatch")
                                              if name in sys.modules]
        report["finished_at"] = datetime.now().astimezone().isoformat()
        (output / "model-manager-ready-font.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        application.processEvents()
    print(json.dumps({"status": report["status"], "profile_bytes_unchanged": report["profile_bytes_unchanged"],
                      "report": str(output / "model-manager-ready-font.json")}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    capture(args.models_root.resolve(), args.output.resolve())
