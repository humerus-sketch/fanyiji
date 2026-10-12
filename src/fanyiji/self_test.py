"""Offline model verification using fixed text and local official WAV files.

Never opens a microphone or loopback. A passed report verifies loading and
nonempty outputs, not transcription/translation accuracy or live latency.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
import wave
from pathlib import Path


def _memory_snapshot() -> dict:
    """Read the current Windows process working set without extra dependencies."""
    if sys.platform != "win32":
        return {"available": False, "reason": "Windows working-set measurement only"}
    try:
        import ctypes
        from ctypes import wintypes

        class MemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(MemoryCounters), wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = MemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
        return {
            "available": True,
            "working_set_bytes": counters.WorkingSetSize,
            "peak_working_set_bytes": counters.PeakWorkingSetSize,
            "private_commit_bytes": counters.PagefileUsage,
        }
    except (OSError, AttributeError) as exc:
        return {"available": False, "reason": str(exc)}


def _asset_summary(directory: Path, filenames: tuple[str, ...]) -> dict:
    files = []
    for name in filenames:
        path = directory / name
        files.append({"name": name, "bytes": path.stat().st_size if path.is_file() else None})
    return {
        "directory": str(directory), "files": files,
        "total_bytes": sum(item["bytes"] or 0 for item in files),
    }


def _read_sample(path: Path):
    import numpy as np

    if not path.is_file():
        raise FileNotFoundError(f"本地自测音频缺失：{path}；程序不会联网下载。")
    with wave.open(str(path), "rb") as stream:
        rate = stream.getframerate()
        channels = stream.getnchannels()
        width = stream.getsampwidth()
        frames = stream.getnframes()
        if rate != 16_000 or channels != 1 or width != 2 or stream.getcomptype() != "NONE":
            raise ValueError(f"自测音频必须是 16 kHz、单声道、16 bit PCM WAV：{path}")
        if frames == 0:
            raise ValueError(f"自测音频为空：{path}")
        raw = stream.readframes(frames)
    if len(raw) != frames * channels * width:
        raise ValueError(f"自测音频 PCM 数据不完整：{path}")
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    return samples, {
        "sample_path": str(path), "sample_bytes": path.stat().st_size,
        "sample_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "sample_rate_hz": rate, "channels": channels, "pcm_bits": width * 8,
        "duration_seconds": round(frames / rate, 6),
    }


def run_self_test(models: Path, output: Path | None = None) -> int:
    from .engines import RecognizerEngine, TranslationEngine, create_translation_engine

    models = Path(models).resolve()
    translation_files = TranslationEngine.REQUIRED_FILES
    backend_file = models / "translation" / "backend.json"
    if backend_file.is_file():
        try:
            config = json.loads(backend_file.read_text(encoding="utf-8-sig"))
            if isinstance(config, dict) and config.get("backend") == "llama_cpp":
                translation_files = ("backend.json", str(config.get("model_file", "")))
        except (OSError, ValueError):
            translation_files = ("backend.json",)
    report = {
        "audio_capture": False, "network_required": False,
        "verification_scope": "本地模型加载及非空输出；识别和翻译准确性需人工核对。",
        "measurement_scope": "单次本地文件推理，不代表双路实时采集延迟。",
        "runtime": {"python": platform.python_version(), "platform": platform.platform()},
        "engine_settings": {"device": "cpu", "asr_threads": 4, "asr_languages": ["zh", "ja"],
                            "translation_threads": 4, "translation_compute_type": "int8", "beam_size": 2},
        "model_assets": {
            "asr": _asset_summary(models / "asr", ("model.int8.onnx", "tokens.txt")),
            "translation": _asset_summary(models / "translation", translation_files),
        },
        "timings": {}, "memory": {"before_loading": _memory_snapshot()},
        "results": [], "audio_results": [],
        "memory_scope": "Windows main process only; native helper memory is not included",
    }
    started = time.perf_counter()
    stage = "asr_loading"
    translator = None
    recognizer = None
    speech_filter = None
    try:
        start = time.perf_counter()
        if (models / "asr/backend.json").is_file():
            from .asr_engine import create_recognizer_engine
            recognizer = create_recognizer_engine(models / "asr", num_threads=4)
            report["engine_settings"]["asr_threads"] = 4
        else:
            recognizer = RecognizerEngine(models / "asr", num_threads=4)
        asr_files = getattr(recognizer, "asset_files", None)
        if isinstance(asr_files, (tuple, list)):
            report["model_assets"]["asr"] = _asset_summary(models / "asr", tuple(asr_files))
        asr_info = getattr(recognizer, "backend_info", None)
        if isinstance(asr_info, dict):
            report["engine_settings"]["asr"] = asr_info
        recognizer.preload(("zh", "ja"))
        report["timings"]["asr_load_seconds"] = round(time.perf_counter() - start, 6)
        report["memory"]["after_asr_loading"] = _memory_snapshot()

        stage = "translation_loading"
        start = time.perf_counter()
        translator = create_translation_engine(models / "translation", num_threads=4)
        translator.preload()
        info = getattr(translator, "backend_info", None)
        if isinstance(info, dict):
            report["engine_settings"]["translation"] = info
            if info.get("backend") == "llama_cpp":
                report["engine_settings"].pop("translation_compute_type", None)
                report["engine_settings"].pop("beam_size", None)
                report["engine_settings"]["device"] = "asr_cpu_and_local_gguf_helper"
                report["local_loopback_used"] = True
        report["timings"]["translation_load_seconds"] = round(time.perf_counter() - start, 6)
        report["memory"]["after_translation_loading"] = _memory_snapshot()
        report["load_seconds"] = round(time.perf_counter() - started, 6)

        stage = "audio_runtime_loading"
        import numpy as np
        from .audio import AudioCaptureManager
        from .segmentation import SpeechSegmenter
        audio_backend, soxr = AudioCaptureManager._load_backends()
        resampler = soxr.ResampleStream(48_000, 16_000, 1, dtype="float32", quality="MQ")
        converted = resampler.resample_chunk(np.zeros(4_800, dtype=np.float32), last=True)
        if converted.size != 1_600:
            raise RuntimeError("SoXR 自测输出采样数不符合预期")
        vad = SpeechSegmenter("mic", vad_model_path=models / "vad" / "silero_vad.onnx")
        vad.reset()
        report["audio_runtime"] = {
            "portaudio": audio_backend.get_portaudio_version_text(),
            "soxr": "48000 -> 16000 synthetic PCM passed", "vad": "loaded",
            "device_capture_started": False,
        }
        stage = "speech_filter_loading"
        if (models / "speech_filter").is_dir():
            from .speech_filter import SpeechOnlyFilter
            speech_filter = SpeechOnlyFilter(models)
            start = time.perf_counter()
            speech_filter.preload()
            report["timings"]["speech_filter_load_seconds"] = round(time.perf_counter() - start, 6)
            report["engine_settings"]["speech_filter"] = speech_filter.backend_info
        else:
            report["engine_settings"]["speech_filter"] = {"model_loaded": False, "verification": "assets_not_present"}

        cases = [
            ("zh", "ja", "我们明天下午三点开会。"),
            ("zh", "ja", "请先把资料发给我。"),
            ("zh", "ja", "我今天不能参加会议。"),
            ("ja", "zh", "資料を先に送ってください。"),
            ("ja", "zh", "明日の午後三時に会議をします。"),
            ("ja", "zh", "今日は会議に参加できません。"),
        ]
        for index, (source, target, text) in enumerate(cases):
            stage = f"fixed_text_{index + 1}_{source}_to_{target}"
            start = time.perf_counter()
            result = translator.translate(text, source, target)
            if not result.strip():
                raise RuntimeError("本地翻译返回空结果")
            report["results"].append({
                "source": source, "target": target, "input": text, "output": result,
                "processing_seconds": round(time.perf_counter() - start, 6),
            })

        for source, target in (("zh", "ja"), ("ja", "zh")):
            stage = f"audio_{source}_reading"
            start = time.perf_counter()
            samples, details = _read_sample(models / "self_test" / f"{source}.wav")
            details["read_seconds"] = round(time.perf_counter() - start, 6)
            entry = {"source": source, "target": target, **details}
            # Keep the successful ASR output even if its translation later fails.
            report["audio_results"].append(entry)
            if speech_filter is not None:
                stage = f"audio_{source}_speech_filter"
                decision = speech_filter.process(samples)
                entry["speech_filter"] = decision.metadata
                if decision.skip:
                    raise RuntimeError(f"{source} 讲话自测音频被筛选模型跳过")
                samples = decision.samples
            stage = f"audio_{source}_recognition"
            start = time.perf_counter()
            text = recognizer.recognize(samples, source)
            entry["asr_seconds"] = round(time.perf_counter() - start, 6)
            entry["source_text"] = text
            if not text.strip():
                raise RuntimeError(f"{source} 自测音频识别返回空结果")
            stage = f"audio_{source}_translation"
            start = time.perf_counter()
            translated = translator.translate(text, source, target)
            entry["translation_seconds"] = round(time.perf_counter() - start, 6)
            entry["translation"] = translated
            if not translated.strip():
                raise RuntimeError(f"{source} 自测音频翻译返回空结果")
            entry["processing_seconds"] = round(entry["asr_seconds"] + entry["translation_seconds"], 6)
            entry["asr_real_time_factor"] = round(entry["asr_seconds"] / entry["duration_seconds"], 6)
        report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = {"stage": stage, "type": type(exc).__name__, "message": str(exc)}
    finally:
        if translator is not None and hasattr(translator, "close"):
            try:
                translator.close()
            except Exception as exc:
                report["status"] = "failed"
                report["error"] = {"stage": "translation_cleanup", "type": type(exc).__name__, "message": str(exc)}
        close_asr = getattr(recognizer, "close", None)
        if close_asr is not None:
            try:
                close_asr()
            except Exception as exc:
                report["status"] = "failed"
                report["error"] = {"stage": "asr_cleanup", "type": type(exc).__name__, "message": str(exc)}
        if speech_filter is not None:
            speech_filter.close()
    report["timings"]["total_seconds"] = round(time.perf_counter() - started, 6)
    report["memory"]["after_test"] = _memory_snapshot()
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    if output:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized, encoding="utf-8")
    if sys.stdout is not None:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        print(serialized)
    return 0 if report["status"] == "passed" else 1
