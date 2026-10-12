"""Evaluate the cached GTCRN on existing PCM WAV files, without recording or ASR."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import socket
import statistics
import time
import wave
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import sherpa_onnx


MODEL_SHA256 = "e77603ac0c23dac3227dd2d7135b3a585cbee2679048aecfa886657d3ae1b534"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_wave(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav:
        if wav.getsampwidth() != 2 or wav.getnchannels() != 1:
            raise ValueError(f"Expected mono PCM16: {path}")
        sample_rate = wav.getframerate()
        audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
    return np.ascontiguousarray(audio.astype(np.float32) / 32768.0), sample_rate


def write_wave(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    pcm = np.rint(np.clip(samples, -1.0, 32767.0 / 32768.0) * 32768.0).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


def rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


def waveform_metrics(raw: np.ndarray, clean: np.ndarray, sample_rate: int) -> dict:
    size = min(len(raw), len(clean))
    a, b = raw[:size], clean[:size]
    raw_rms, clean_rms = rms(a), rms(b)
    gain_db = 20.0 * math.log10(max(clean_rms, 1e-12) / max(raw_rms, 1e-12))
    # Correlation estimates signal alignment, not intelligibility or separation quality.
    fft_size = 1 << (2 * size - 1).bit_length()
    corr = np.fft.irfft(np.fft.rfft(b, fft_size) * np.conj(np.fft.rfft(a, fft_size)), fft_size)
    max_lag = min(sample_rate // 10, size - 1)
    candidates = np.concatenate((corr[-max_lag:], corr[:max_lag + 1]))
    lag = int(np.argmax(candidates)) - max_lag
    frame = sample_rate // 10
    frame_count = size // frame
    raw_frames = a[:frame_count * frame].reshape(frame_count, frame)
    clean_frames = b[:frame_count * frame].reshape(frame_count, frame)
    raw_frame_rms = np.sqrt(np.mean(raw_frames.astype(np.float64) ** 2, axis=1))
    clean_frame_rms = np.sqrt(np.mean(clean_frames.astype(np.float64) ** 2, axis=1))
    active = raw_frame_rms > 1e-5
    gains = 20.0 * np.log10(np.maximum(clean_frame_rms[active], 1e-12) / raw_frame_rms[active])
    return {
        "compared_samples": size,
        "raw_rms": raw_rms,
        "clean_rms": clean_rms,
        "rms_gain_db": gain_db,
        "raw_peak": float(np.max(np.abs(a))),
        "clean_peak": float(np.max(np.abs(b))),
        "clean_clipped_sample_count": int(np.count_nonzero(np.abs(b) >= 1.0)),
        "estimated_correlation_lag_samples": lag,
        "estimated_correlation_lag_ms": lag / sample_rate * 1000.0,
        "active_100ms_frame_count": int(np.count_nonzero(active)),
        "active_frame_gain_db_min": float(np.min(gains)),
        "active_frame_gain_db_median": float(np.median(gains)),
        "active_frame_gain_db_max": float(np.max(gains)),
        "note": "RMS attenuation and correlation cannot prove singing/music removal or retained speech accuracy.",
    }


def block_network(*args, **kwargs):
    raise RuntimeError("Network connections are disabled for this local inference test")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("artifacts/diagnostics/live-20261009-101554/results.json"))
    parser.add_argument("--model", type=Path, default=Path("model-cache/speech-filter/gtcrn/gtcrn_simple.onnx"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/diagnostics/speech-filter-gtcrn-20261009/results.json"))
    args = parser.parse_args()
    assert args.model.stat().st_size == 535638
    assert digest(args.model) == MODEL_SHA256
    source = json.loads(args.input.read_text(encoding="utf-8"))
    selected = [source["samples"][i] for i in (0, 1, 2, 3, 6)]
    assert all(sample["source_lang"] == "ja" for sample in selected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    socket.socket.connect = block_network
    socket.socket.connect_ex = block_network
    config = sherpa_onnx.OfflineSpeechDenoiserModelConfig(
        gtcrn=sherpa_onnx.OfflineSpeechDenoiserGtcrnModelConfig(model=str(args.model.resolve())),
        num_threads=1,
        debug=False,
        provider="cpu",
    )
    offline_config = sherpa_onnx.OfflineSpeechDenoiserConfig(model=config)
    online_config = sherpa_onnx.OnlineSpeechDenoiserConfig(model=config)
    assert offline_config.validate() and online_config.validate()
    started = time.perf_counter()
    offline = sherpa_onnx.OfflineSpeechDenoiser(offline_config)
    offline_load = time.perf_counter() - started
    started = time.perf_counter()
    online = sherpa_onnx.OnlineSpeechDenoiser(online_config)
    online_load = time.perf_counter() - started
    hop = online.frame_shift_in_samples
    assert online.sample_rate == 16000
    report = {
        "status": "running", "sherpa_onnx_version": sherpa_onnx.__version__,
        "device": "cpu", "num_threads": 1, "gpu_inference": False,
        "recording_performed": False, "asr_inference_performed": False,
        "network_connections_blocked_during_inference": True,
        "model_path": str(args.model.resolve()), "model_size": args.model.stat().st_size,
        "model_sha256": MODEL_SHA256, "input_manifest_sha256": digest(args.input),
        "offline_api": sherpa_onnx.OfflineSpeechDenoiser.run.__doc__,
        "online_api": sherpa_onnx.OnlineSpeechDenoiser.__call__.__doc__,
        "offline_load_seconds": offline_load, "online_load_seconds": online_load,
        "sample_rate": online.sample_rate, "frame_shift_in_samples": hop,
        "frame_shift_ms": hop / online.sample_rate * 1000.0, "samples": [],
    }
    for entry in selected:
        raw_path = Path(entry["audio_path"])
        raw, sample_rate = read_wave(raw_path)
        assert sample_rate == 16000
        sample = {"id": entry["id"], "source_audio": str(raw_path),
                  "source_sha256": digest(raw_path), "raw_samples": len(raw),
                  "audio_duration_seconds": len(raw) / sample_rate}
        native_started = time.perf_counter()
        result = offline.run(raw, sample_rate)
        native_elapsed = time.perf_counter() - native_started
        native = np.asarray(result.samples, dtype=np.float32)
        assert result.sample_rate == 16000 and np.isfinite(native).all()
        sample["offline_native"] = {"elapsed_seconds": native_elapsed,
            "rtf": native_elapsed / sample["audio_duration_seconds"], "samples": len(native),
            "length_delta_samples": len(native) - len(raw),
            "length_delta_ms": (len(native) - len(raw)) / sample_rate * 1000.0,
            **waveform_metrics(raw, native, sample_rate)}
        # Pad the input to a complete STFT hop. Padding the shortened output would not recover its tail.
        pad = (-len(raw)) % hop
        padded = np.pad(raw, (0, pad))
        padded_started = time.perf_counter()
        padded_result = offline.run(padded, sample_rate)
        padded_elapsed = time.perf_counter() - padded_started
        padded_clean = np.asarray(padded_result.samples, dtype=np.float32)[:len(raw)]
        assert len(padded_clean) == len(raw) and np.isfinite(padded_clean).all()
        offline_path = args.output.parent / f"{raw_path.stem}-offline-pad-trim.wav"
        write_wave(offline_path, padded_clean, sample_rate)
        sample["offline_input_pad_then_trim"] = {"input_padding_samples": pad,
            "elapsed_seconds": padded_elapsed, "rtf": padded_elapsed / sample["audio_duration_seconds"],
            "samples": len(padded_clean), "output_audio": str(offline_path.resolve()),
            "output_sha256": digest(offline_path), **waveform_metrics(raw, padded_clean, sample_rate)}
        online.reset()
        parts, call_times = [], []
        first_emission_input_samples = None
        online_started = time.perf_counter()
        for position in range(0, len(raw), 512):
            chunk = raw[position:position + 512]
            call_started = time.perf_counter()
            part = np.asarray(online(chunk, sample_rate).samples, dtype=np.float32)
            call_times.append(time.perf_counter() - call_started)
            if len(part) and first_emission_input_samples is None:
                first_emission_input_samples = position + len(chunk)
            parts.append(part)
        before_flush_samples = sum(len(part) for part in parts)
        flush_started = time.perf_counter()
        tail = np.asarray(online.flush().samples, dtype=np.float32)
        flush_elapsed = time.perf_counter() - flush_started
        parts.append(tail)
        clean_online = np.concatenate(parts)
        online_elapsed = time.perf_counter() - online_started
        assert len(clean_online) == len(raw) and np.isfinite(clean_online).all()
        online_path = args.output.parent / f"{raw_path.stem}-online-flush.wav"
        write_wave(online_path, clean_online, sample_rate)
        sample["online_512_block_then_flush"] = {"elapsed_seconds": online_elapsed,
            "rtf": online_elapsed / sample["audio_duration_seconds"], "samples": len(clean_online),
            "first_emission_after_input_samples": first_emission_input_samples,
            "before_flush_output_samples": before_flush_samples, "flush_output_samples": len(tail),
            "flush_elapsed_seconds": flush_elapsed, "call_count": len(call_times),
            "call_median_seconds": statistics.median(call_times), "call_max_seconds": max(call_times),
            "output_audio": str(online_path.resolve()), "output_sha256": digest(online_path),
            **waveform_metrics(raw, clean_online, sample_rate)}
        report["samples"].append(sample)
        print(json.dumps({"id": entry["id"], "native_length_delta": len(native) - len(raw),
                          "offline_seconds": native_elapsed, "online_seconds": online_elapsed,
                          "rms_gain_db": sample["offline_native"]["rms_gain_db"]}), flush=True)
    report["status"] = "completed"
    report["finished_at"] = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")
    report["limitations"] = ["No ground-truth clean speech/music stems, listening assessment or ASR comparison.",
        "RMS decrease does not establish removal of singing; speech may also be attenuated.",
        "Five saved Japanese clips, not a real-time throughput benchmark or music/singing separation validation."]
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(args.output.resolve())}, ensure_ascii=True))


if __name__ == "__main__":
    main()
