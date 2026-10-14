"""Exercise real Qt model selection against already verified external assets.

No model inference, network connection, audio capture, or weight modification.
The two authorized button activations end with Hy-MT2 7B and Qwen-ASR 1.7B.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication, QDialog

from fanyiji.model_manager import ModelManagerDialog

BASELINE = ("hy-mt2-7b-q6_k", "qwen3-asr-1.7b-int8")
ALTERNATIVE = ("hy-mt2-1.8b-q6_k", "whisper-turbo-int8")


def profiles(root: Path) -> dict:
    return {name: json.loads((root / name / "backend.json").read_text(encoding="utf-8-sig"))
            for name in ("translation", "asr")}


def wait_until(predicate, timeout: float = 30) -> float:
    started = time.monotonic()
    loop, timer = QEventLoop(), QTimer()
    expired = [False]

    def check():
        if predicate():
            loop.quit()
        elif time.monotonic() - started >= timeout:
            expired[0] = True
            loop.quit()

    timer.timeout.connect(check)
    timer.start(10)
    if not predicate():
        loop.exec()
    timer.stop()
    if expired[0]:
        raise TimeoutError(f"Qt model worker did not finish within {timeout}s")
    return time.monotonic() - started


def snapshot(dialog: ModelManagerDialog) -> dict:
    return {
        "translation_id": dialog._chosen_id(), "asr_id": dialog._chosen_asr_id(),
        "use_button_enabled": dialog.use_button.isEnabled(), "busy": dialog.is_busy,
        "entry_count": len(dialog._entries), "model_count_excluding_vad": len(dialog.radios) + len(dialog.asr_radios),
        "states": {mid: asdict(state) for mid, state in dialog._states.items()},
        "status_labels": {mid: label.text() for mid, label in dialog.status_labels.items()},
        "download_button_enabled": {mid: button.isEnabled() for mid, button in dialog.download_buttons.items()},
        "message": dialog.message.text(),
    }


def file_identities(dialog: ModelManagerDialog) -> dict:
    result = {}
    for entry in dialog._entries:
        for asset in entry.files:
            path = dialog.downloader.path_for(asset)
            stat = path.stat()
            result[asset.relative_path] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                                          "dev": stat.st_dev, "ino": stat.st_ino,
                                          "catalog_sha256": asset.sha256}
    return result


def open_verified(application, root: Path, dialogs: list) -> tuple[ModelManagerDialog, float]:
    dialog = ModelManagerDialog(root)
    dialogs.append(dialog)
    dialog.resize(1050, 1000)
    dialog.show()
    elapsed = wait_until(lambda: dialog._verified_once and not dialog.is_busy)
    if set(dialog._states) != {entry.id for entry in dialog._entries}:
        raise RuntimeError("Some model entries did not finish verification")
    if not all(state.ready for state in dialog._states.values()):
        raise RuntimeError("Some external model entries are not ready")
    if not dialog.use_button.isEnabled():
        raise RuntimeError("The current ready selection cannot be activated")
    application.processEvents()
    return dialog, elapsed


def activate(dialog: ModelManagerDialog, expected: tuple[str, str]) -> dict:
    dialog.radios[expected[0]].click()
    dialog.asr_radios[expected[1]].click()
    if not dialog.use_button.isEnabled():
        raise RuntimeError("Actual use button did not enable for selected ready models")
    selected_signals = []
    dialog.model_selected.connect(selected_signals.append)
    dialog.use_button.click()
    elapsed = wait_until(lambda: not dialog.is_busy)
    accepted = dialog.result() == QDialog.DialogCode.Accepted
    actual = (dialog.selected_model_id, dialog.selected_asr_model_id)
    if not accepted or actual != expected or selected_signals != [expected[0]]:
        raise RuntimeError("Qt button activation did not accept the expected model IDs")
    return {"accepted": accepted, "selected_ids": list(actual), "model_selected_signals": selected_signals,
            "activation_seconds": elapsed, "profiles": profiles(dialog.models_root)}


def capture_asr(application, dialog: ModelManagerDialog, path: Path, *, top: bool = False) -> None:
    dialog.tabs.setCurrentIndex(1)
    scroll = dialog.tabs.widget(1)
    if top:
        scroll.verticalScrollBar().setValue(0)
    else:
        scroll.ensureWidgetVisible(dialog.asr_radios[dialog._chosen_asr_id()])
    application.processEvents()
    if not dialog.grab().save(str(path)):
        raise RuntimeError("Could not save real Qt widget screenshot")


def validate(root: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "model-manager-ready.json"
    report = {
        "status": "starting", "started_at": datetime.now().astimezone().isoformat(), "models_root": str(root),
        "qt_platform": "offscreen", "native_inference_performed": False, "audio_devices_opened": False,
        "network_attempts": 0, "worker_deadline_seconds": 30,
        "source_sha256": {name: hashlib.sha256((ROOT / "src" / "fanyiji" / name).read_bytes()).hexdigest()
                          for name in ("model_manager.py", "models_download.py", "asr_catalog.py")},
    }

    def write():
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    def prohibit_network(event, args):
        if event == "socket.connect":
            report["network_attempts"] += 1
            raise RuntimeError("Network access is prohibited in local model-manager validation")
        if event == "subprocess.Popen":
            raise RuntimeError("Native process launch is prohibited in model-manager validation")

    sys.addaudithook(prohibit_network)
    application = QApplication([])
    application.setQuitOnLastWindowClosed(False)
    dialogs = []
    report["profiles_before"] = profiles(root)
    write()
    try:
        initial, elapsed = open_verified(application, root, dialogs)
        report["initial_verification_seconds"] = elapsed
        report["initial"] = snapshot(initial)
        if (initial._chosen_id(), initial._chosen_asr_id()) != BASELINE:
            raise RuntimeError("Initial external default is not the requested Hy7/Qwen1.7 baseline")
        before_files = file_identities(initial)
        ready = {"translation": {}, "asr": {}}
        for mid, radio in initial.radios.items():
            radio.click()
            ready["translation"][mid] = initial.use_button.isEnabled()
        initial.radios[BASELINE[0]].click()
        for mid, radio in initial.asr_radios.items():
            radio.click()
            ready["asr"][mid] = initial.use_button.isEnabled()
        initial.asr_radios[BASELINE[1]].click()
        report["all_ready_model_selections_enable_use"] = ready
        if not all(value for group in ready.values() for value in group.values()):
            raise RuntimeError("Some ready model radio selections did not enable use")
        capture_asr(application, initial, output / "models-ready-asr-selected.png")
        capture_asr(application, initial, output / "models-ready-asr-top.png", top=True)
        report["alternative_activation"] = activate(initial, ALTERNATIVE)
        write()
        reopened, elapsed = open_verified(application, root, dialogs)
        report["reopen_verification_seconds"] = elapsed
        report["reopened_after_alternative"] = snapshot(reopened)
        if (reopened._chosen_id(), reopened._chosen_asr_id()) != ALTERNATIVE:
            raise RuntimeError("Reopening did not read the persisted Hy1.8/Turbo choices")
        capture_asr(application, reopened, output / "models-ready-asr-turbo-persisted.png")
        report["baseline_activation"] = activate(reopened, BASELINE)
        final, elapsed = open_verified(application, root, dialogs)
        report["final_verification_seconds"] = elapsed
        report["final"] = snapshot(final)
        if (final._chosen_id(), final._chosen_asr_id()) != BASELINE:
            raise RuntimeError("Final reopening did not read Hy7/Qwen1.7")
        capture_asr(application, final, output / "models-ready-asr-final.png")
        report["weight_file_identities_unchanged"] = before_files == file_identities(final)
        report["weight_files"] = before_files
        if not report["weight_file_identities_unchanged"]:
            raise RuntimeError("A verified weight file changed during the UI selection test")
        report["status"] = "completed"
    except BaseException as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        # Cancel and finish workers before any fallback restoration or Qt quit.
        for dialog in dialogs:
            dialog.reject()
            if dialog.is_busy:
                wait_until(lambda d=dialog: not d.is_busy)
        current = profiles(root)
        if current != report["profiles_before"]:
            # Only a failed mid-switch path reaches this fallback. The same
            # already verified source API restores the authorized baseline.
            from fanyiji.asr_catalog import AsrCatalog
            from fanyiji.models_download import ModelCatalog, ModelDownloader
            ModelDownloader(root).activate_models(ModelCatalog.default(), BASELINE[0], AsrCatalog.default(), BASELINE[1])
            report["fallback_restore_used"] = True
        else:
            report["fallback_restore_used"] = False
        report["profiles_final"] = profiles(root)
        report["final_profiles_equal_initial"] = report["profiles_final"] == report["profiles_before"]
        report["native_modules_imported"] = [name for name in ("sherpa_onnx", "torch", "pyaudiowpatch") if name in sys.modules]
        report["all_workers_finished"] = all(not dialog.is_busy for dialog in dialogs)
        report["finished_at"] = datetime.now().astimezone().isoformat()
        write()
        application.quit()
    print(json.dumps({"status": report["status"], "report": str(report_path),
                      "entry_count": report["initial"]["entry_count"], "models": report["initial"]["model_count_excluding_vad"],
                      "final_profiles_equal_initial": report["final_profiles_equal_initial"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    validate(args.models_root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
