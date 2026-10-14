"""Explicit opt-in recordings from the real application pipeline; never uploads."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import threading
import wave

import numpy as np


def timestamp() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="milliseconds")


_TRANSLATION_FIELDS = (
    "display_translation", "display_translation_valid", "pipeline_latency_seconds",
    "translation_revision", "translation_provisional", "translation_context", "translation_processing_seconds",
    "translation_queue_wait_seconds", "translation_batch_id", "translation_batch_size", "translation_rpc_metrics",
    "translation_fallback", "translated_source_segment_ids", "draft_latency_seconds", "draft_display_latency_seconds",
    "final_latency_seconds", "first_display_latency_seconds", "translation_stage", "translation_provider",
    "translation_model", "translation_final_failed", "translation_overall_failed",
    "presentation_stages", "presentation_metrics", "end_to_end_display_seconds",
    "post_audio_display_seconds", "recognition_to_display_seconds",
)
_RAW_TRANSLATION_FIELDS = (
    "production_translation", "production_translation_revision", "production_translation_stage",
    "production_translation_context_generation", "production_translation_provider", "production_translation_model",
    "production_translation_failed",
)
_SOURCE_FIELDS = (
    "production_text", "recognition_quality", "asr_queue_wait_seconds", "asr_processing_seconds",
    "speech_filter_processing_seconds", "sentence_assembly", "display_original", "display_original_revision",
    "display_original_uncertain", "display_context_generation", "confirmed_revision", "translation_group_id",
    "recognition_timing",
)


def _reset_translation_display(sample):
    for key in _TRANSLATION_FIELDS:
        sample.pop(key, None)
    sample["translation_stages"] = {}


class SessionRecorder:
    def __init__(self, directory: Path, config):
        self.directory = directory.resolve()
        self.directory.mkdir(parents=True, exist_ok=False)
        (self.directory / "segments").mkdir()
        self.config = config
        self.active = True
        self.lock = threading.RLock()
        self.samples = {}
        self.events = []
        self.counts = {channel: 0 for channel in config.enabled_channels}
        self.errors = []
        self.filtered = {}
        self.model_info = {"asr": "not_loaded", "translation": "not_loaded"}
        self.translation_queue = {}
        self.stream = (self.directory / "events.jsonl").open("w", encoding="utf-8")

    def on_chunk(self, chunk):
        with self.lock:
            if self.active and chunk.channel in self.counts:
                self.counts[chunk.channel] += chunk.samples.size

    def on_segment(self, job, accepted):
        with self.lock:
            if not self.active:
                return
            path = self.directory / "segments" / (job.segment_id.replace(":", "-") + ".wav")
            pcm = (np.clip(job.samples, -1, 1) * 32767).astype("<i2").tobytes()
            with wave.open(str(path), "wb") as stream:
                stream.setnchannels(1)
                stream.setsampwidth(2)
                stream.setframerate(16000)
                stream.writeframes(pcm)
            self.samples[job.segment_id] = {
                "id": job.segment_id, "channel": job.channel, "source_lang": job.direction[0],
                "target_lang": job.direction[1], "audio_path": str(path),
                "audio_duration_seconds": job.samples.size / 16000,
                "accepted": accepted, "observed_at": timestamp(),
                "start_sample": getattr(job, "start_sample", None),
                "end_sample": getattr(job, "end_sample", None),
                "generation": getattr(job, "generation", None),
                "ended_by": getattr(job, "ended_by", "unknown"),
                "timing": deepcopy(getattr(job, "timing", {})),
            }
            if job.segment_id in self.filtered:
                self.samples[job.segment_id].update(self.filtered[job.segment_id])

    def on_filtered(self, job, decision):
        """Keep raw input and point reference inference at the actual ASR input.

        The worker can finish before on_segment runs on the capture thread, so
        store the filter record independently and merge it in either order.
        """
        with self.lock:
            if not self.active:
                return
            raw_path = self.directory / "segments" / (job.segment_id.replace(":", "-") + ".wav")
            record = {"speech_filter": decision.metadata, "audio_original_path": str(raw_path)}
            if decision.metadata["action"] == "masked":
                path = raw_path.with_name(raw_path.stem + ".filtered.wav")
                pcm = (np.clip(decision.samples, -1, 1) * 32767).astype("<i2").tobytes()
                with wave.open(str(path), "wb") as stream:
                    stream.setnchannels(1)
                    stream.setsampwidth(2)
                    stream.setframerate(16000)
                    stream.writeframes(pcm)
                record["audio_path"] = str(path)
            self.filtered[job.segment_id] = record
            if job.segment_id in self.samples:
                self.samples[job.segment_id].update(record)

    def event(self, event):
        if event["kind"] == "level":
            return
        with self.lock:
            if self.active:
                record = {**event, "observed_at": timestamp()}
                if event["kind"] == "models_ready":
                    # Scheduling metadata belongs to the ready event, not just
                    # the engine. Preserve it before the next periodic snapshot.
                    self.model_info.update({role: deepcopy(info) for role, info in event.get("models", {}).items()
                        if role in ("asr", "translation", "speech_filter") and isinstance(info, dict)})
                elif event["kind"] == "translation_queue":
                    self.translation_queue = deepcopy(record)
                self.events.append(record)
                self.stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                self.stream.flush()

    def snapshot(self, status="collecting"):
        with self.lock:
            from .latency import LatencyTracker
            performance = LatencyTracker()
            requests = {}
            for sample in self.samples.values():
                # Event-derived state is replayed from scratch. Reusing a previous
                # snapshot's revision guards would make identical snapshots differ.
                for key in (*_SOURCE_FIELDS, *_RAW_TRANSLATION_FIELDS):
                    sample.pop(key, None)
                _reset_translation_display(sample)
                sample["translation_results"] = []
                sample["translation_candidates"] = []
            for event in self.events:
                if event["kind"] == "subtitle_presented":
                    sample = self.samples.get(event.get("segment_id"))
                    if sample is None:
                        continue
                    revision = event.get("revision", 1)
                    if revision < max(sample.get("display_original_revision", 0), sample.get("translation_revision", 0)):
                        continue
                    if (sample.get("display_context_generation") is not None and event.get("context_generation") is not None
                            and sample["display_context_generation"] != event["context_generation"]):
                        continue
                    if event.get("session") != performance.session:
                        performance.begin_session(event.get("session"))
                    measurement = performance.on_presented(event)
                    if measurement is None:
                        continue
                    sample.setdefault("presentation_stages", {})[measurement["stage"]] = measurement
                    if measurement["stage"] == "final" and measurement["valid"]:
                        sample["presentation_metrics"] = measurement
                        sample["end_to_end_display_seconds"] = measurement["end_to_end_seconds"]
                        sample["post_audio_display_seconds"] = measurement["post_audio_seconds"]
                        sample["recognition_to_display_seconds"] = measurement["recognition_to_display_seconds"]
                    continue
                if event["kind"] in ("translation", "translation_candidate", "translation_status", "translation_request") and event.get("batch_id"):
                    batch_id = event["batch_id"]
                    request_key = (event.get("translation_provider"), batch_id)
                    if request_key not in requests:
                        rpc_metrics = event.get("rpc_metrics")
                        requests[request_key] = {
                            "batch_id": batch_id, "batch_size": event.get("batch_size"),
                            "processing_seconds": event.get("processing_seconds"),
                            "total_seconds": (rpc_metrics.get("total_seconds", event.get("processing_seconds"))
                                              if isinstance(rpc_metrics, dict) else event.get("processing_seconds")),
                            "rpc_metrics": deepcopy(rpc_metrics), "fallback": event.get("fallback"),
                            "failed": bool(event.get("failed", False)),
                            "segment_ids": [], "observed_at": event["observed_at"],
                            "translation_stage": event.get("translation_stage"),
                            "translation_provider": event.get("translation_provider"),
                            "translation_model": event.get("translation_model"),
                        }
                    segment_id = event.get("segment_id")
                    identifiers = ([segment_id] if segment_id else []) + list(event.get("segment_ids", ()))
                    for identifier in identifiers:
                        if identifier not in requests[request_key]["segment_ids"]:
                            requests[request_key]["segment_ids"].append(identifier)
                sample = self.samples.get(event.get("segment_id"))
                if sample is None:
                    continue
                source_ids = event.get("source_segment_ids", ())
                if event["kind"] == "recognition":
                    sample["production_text"] = event["text"]
                    sample["recognition_quality"] = event["quality"]
                    sample["asr_queue_wait_seconds"] = event.get("queue_wait_seconds")
                    sample["asr_processing_seconds"] = event.get("processing_seconds")
                    sample["speech_filter_processing_seconds"] = event.get("speech_filter_seconds")
                    sample["recognition_timing"] = deepcopy(event.get("timing", {}))
                elif event["kind"] == "sentence_join":
                    sample["sentence_assembly"] = {
                        key: event.get(key) for key in (
                            "raw_source_texts", "assembly_text", "boundary_normalized",
                            "joint_recognition", "gap_samples", "gap_filled", "source_segment_ids", "uncertain")
                    }
                elif event["kind"] == "original":
                    revision = event.get("revision", 1)
                    previous_revision = sample.get("display_original_revision", 0)
                    previous_generation = sample.get("display_context_generation")
                    generation = event.get("context_generation")
                    older_generation = (revision == previous_revision and generation is not None
                        and previous_generation is not None and generation < previous_generation)
                    if revision >= previous_revision and not older_generation:
                        new_source = (revision > previous_revision or (generation is not None
                            and previous_generation is not None and generation > previous_generation))
                        if new_source:
                            _reset_translation_display(sample)
                            if revision == previous_revision:
                                sample.pop("confirmed_revision", None)
                        if len(source_ids) <= 1:
                            sample["production_text"] = event["text"]
                        sample["display_original"] = event["text"]
                        sample["display_original_revision"] = revision
                        sample["display_original_uncertain"] = bool(event.get("uncertain", False))
                        sample["display_translation_valid"] = sample.get("translation_revision", 0) >= revision
                        if "context_generation" in event:
                            sample["display_context_generation"] = event["context_generation"]
                elif event["kind"] in ("translation", "translation_candidate", "translation_status"):
                    revision = event.get("revision", 1)
                    stage = event.get("translation_stage", "final")
                    detail = {key: deepcopy(event.get(key)) for key in ("revision", "context_generation",
                        "translation_stage", "translation_provider", "translation_model", "text", "latency",
                        "processing_seconds", "queue_wait_seconds", "batch_id", "rpc_metrics", "failed",
                        "overall_failed", "final_failed", "error", "observed_at") if key in event}
                    detail.update(revision=revision, translation_stage=stage,
                                  displayed=event["kind"] == "translation", kind=event["kind"])
                    sample["translation_results"].append(detail)
                    failed = bool(event.get("failed", False))
                    if event["kind"] != "translation_status" and not failed:
                        sample["translation_candidates"].append(deepcopy(detail))
                    if event["kind"] != "translation_status" and len(source_ids) <= 1:
                        # Keep output for the raw audio segment even when the
                        # displayed subtitle has become a newer joined phrase.
                        # Within the same source, a late draft cannot replace a final.
                        raw_key = (revision, event.get("context_generation") or 0, int(stage == "final"))
                        previous_raw_key = (sample.get("production_translation_revision", 0),
                            sample.get("production_translation_context_generation") or 0,
                            int(sample.get("production_translation_stage") == "final"))
                        if raw_key >= previous_raw_key:
                            sample["production_translation"] = event["text"]
                            sample["production_translation_revision"] = revision
                            sample["production_translation_stage"] = stage
                            sample["production_translation_context_generation"] = event.get("context_generation")
                            sample["production_translation_provider"] = event.get("translation_provider")
                            sample["production_translation_model"] = event.get("translation_model")
                            sample["production_translation_failed"] = failed
                    if revision < max(sample.get("display_original_revision", 0),
                                      sample.get("translation_revision", 0)):
                        continue
                    if (sample.get("display_context_generation") is not None and event.get("context_generation") is not None
                            and sample["display_context_generation"] != event["context_generation"]):
                        continue
                    sample["translation_stages"][stage] = detail
                    if not failed and event["kind"] != "translation_status":
                        sample[stage + "_latency_seconds"] = event.get("latency")
                    if event["kind"] == "translation_status":
                        if stage == "final":
                            sample["translation_final_failed"] = failed
                        sample["translation_overall_failed"] = bool(event.get("overall_failed", False))
                        continue
                    if event["kind"] == "translation_candidate":
                        continue
                    if (revision == sample.get("translation_revision", 0) and stage == "draft"
                            and sample.get("translation_stage") == "final"):
                        continue
                    sample["display_translation"] = event["text"]
                    sample["display_translation_valid"] = not failed
                    sample["pipeline_latency_seconds"] = event.get("latency")
                    sample["translation_revision"] = event.get("revision", 1)
                    sample["translation_stage"] = stage
                    sample["translation_provider"] = event.get("translation_provider")
                    sample["translation_model"] = event.get("translation_model")
                    if "final_failed" in event:
                        sample["translation_final_failed"] = bool(event["final_failed"])
                    elif stage == "final":
                        sample["translation_final_failed"] = failed
                    sample["translation_overall_failed"] = bool(event.get("overall_failed", failed))
                    if not failed:
                        sample.setdefault("first_display_latency_seconds", event.get("latency"))
                        if stage == "draft":
                            sample.setdefault("draft_display_latency_seconds", event.get("latency"))
                    sample["translation_provisional"] = (event.get("provisional", False)
                        and sample.get("confirmed_revision", 0) < revision)
                    sample["translation_context"] = event.get("context", "")
                    sample["translation_processing_seconds"] = event.get("processing_seconds")
                    sample["translation_queue_wait_seconds"] = event.get("queue_wait_seconds")
                    sample["translation_batch_id"] = event.get("batch_id")
                    sample["translation_batch_size"] = event.get("batch_size")
                    sample["translation_rpc_metrics"] = deepcopy(event.get("rpc_metrics"))
                    sample["translation_fallback"] = event.get("fallback")
                    sample["translated_source_segment_ids"] = list(source_ids)
                    if len(source_ids) > 1:
                        for source_id in source_ids:
                            if source_id in self.samples:
                                self.samples[source_id]["translation_group_id"] = event["segment_id"]
                elif event["kind"] == "subtitle_final":
                    sample["confirmed_revision"] = max(sample.get("confirmed_revision", 0), event.get("revision", 1))
                    if event.get("revision", 1) >= sample.get("translation_revision", 0):
                        sample["translation_provisional"] = False
            report = {
                "status": "failed" if self.errors else status, "updated_at": timestamp(),
                "settings_snapshot": asdict(self.config), "models": self.model_info,
                "collection_scope": "Actual application pipeline inputs and generated subtitles; no duplicate inference",
                "recording_uploaded": False, "continuous_wav_saved": False, "tail_finalized": False,
                "raw_audio_seconds": {key: value / 16000 for key, value in self.counts.items()},
                "errors": self.errors, "samples": list(self.samples.values()),
                "translation_queue": self.translation_queue,
                # A batch can produce several subtitle events sharing one RPC.
                # Request totals must use this list, not summed sample timings.
                "translation_requests": list(requests.values()),
                "performance_summary": performance.summary(),
                "timing_measurement": {
                    "start": "First captured audio sample in the subtitle, including its available pre-speech prefix",
                    "finish": "Accepted subtitle label Qt paint completed while visible",
                    "post_audio_start": "Last captured audio sample in the subtitle",
                    "network_scope": "Codex receive includes network, service scheduling and generation; not separated",
                    "physical_display_scan_measured": False,
                },
            }
            path = self.directory / "results.json"
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(path)

    def finish(self, status="application_closed"):
        with self.lock:
            if self.active:
                self.active = False
                self.stream.close()
                self.snapshot(status)

    def fail(self, exc):
        with self.lock:
            self.errors.append(type(exc).__name__)
            self.finish("failed")
