"""Separate, auditable corpus for a future student; this never activates a model."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import threading
import unicodedata
import uuid
import wave

import numpy as np

from .text_context import bounded_source_context
from .livestream_prompt import HY_CONTEXT_CHARACTERS, HY_PROMPT_FORMAT, PROFILE_ID, hy_translation_prompt


def _timestamp() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="milliseconds")


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


class SpecializedStore:
    """All files belong to this corpus, never to the running model catalog.

    A reviewed local candidate can become training data. Codex output retains
    its origin permanently and is kept outside exports, including after review.
    Calling code uses a background queue; the lock also protects review/export.
    """

    AUDIO_QUOTA_BYTES = 2 * 1024**3
    _LATEST = "s.revision=(SELECT MAX(newer.revision) FROM samples newer WHERE newer.session_id=s.session_id AND newer.channel=s.channel AND newer.segment_id=s.segment_id)"
    _CLEAN = "s.source<>'' AND s.provisional=0 AND s.uncertain=0 AND s.failed=0"

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._audio_root = self.root / "audio"
        self._audio_root.mkdir(exist_ok=True)
        self._lock = threading.RLock()
        self._closed = False
        self._db = sqlite3.connect(self.root / "corpus.sqlite3", check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS samples (
                id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, channel TEXT NOT NULL,
                segment_id TEXT NOT NULL, revision INTEGER NOT NULL, direction TEXT NOT NULL,
                source TEXT NOT NULL, context TEXT NOT NULL DEFAULT '',
                provisional INTEGER NOT NULL DEFAULT 0, finalized INTEGER NOT NULL DEFAULT 0,
                uncertain INTEGER NOT NULL DEFAULT 0, failed INTEGER NOT NULL DEFAULT 0,
                source_ids TEXT NOT NULL, metadata TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                UNIQUE(session_id, channel, segment_id, revision)
            );
            CREATE TABLE IF NOT EXISTS candidates (
                id INTEGER PRIMARY KEY, sample_id INTEGER NOT NULL REFERENCES samples(id),
                text TEXT NOT NULL, origin TEXT NOT NULL CHECK(origin IN ('local', 'codex')),
                model TEXT NOT NULL, processing_seconds REAL NOT NULL DEFAULT 0,
                reviewed INTEGER NOT NULL DEFAULT 0, source_verified INTEGER NOT NULL DEFAULT 0,
                rights_confirmed INTEGER NOT NULL DEFAULT 0,
                training_allowed INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL, reviewed_at TEXT,
                CHECK(origin<>'codex' OR training_allowed=0),
                UNIQUE(sample_id, origin, model, text)
            );
            CREATE TABLE IF NOT EXISTS segments (
                session_id TEXT NOT NULL, channel TEXT NOT NULL, segment_id TEXT NOT NULL,
                audio_path TEXT, audio_bytes INTEGER NOT NULL DEFAULT 0,
                accepted INTEGER NOT NULL DEFAULT 0, metadata TEXT NOT NULL DEFAULT '{}',
                PRIMARY KEY(session_id, channel, segment_id)
            );
            CREATE TABLE IF NOT EXISTS annotations (
                session_id TEXT NOT NULL, channel TEXT NOT NULL, segment_id TEXT NOT NULL,
                revision INTEGER NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL,
                PRIMARY KEY(session_id, channel, segment_id, revision, kind)
            );
            CREATE TABLE IF NOT EXISTS corpus_state (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS session_splits (
                session_id TEXT PRIMARY KEY, split TEXT NOT NULL CHECK(split IN ('train','validation'))
            );
            CREATE TABLE IF NOT EXISTS source_splits (
                source_group TEXT PRIMARY KEY, split TEXT NOT NULL CHECK(split IN ('train','validation'))
            );
            CREATE INDEX IF NOT EXISTS candidate_sample ON candidates(sample_id);
        """)
        self._db.commit()
        # Existing recordings count after restarts, including an interrupted DB write.
        self._audio_bytes = sum(path.stat().st_size for path in self._audio_root.glob("*.wav") if path.is_file())

    def _open(self) -> None:
        if self._closed:
            raise RuntimeError("专用模型数据仓库已关闭")

    def _sample(self, session_id: str, channel: str, segment_id: str, revision: int):
        return self._db.execute("SELECT * FROM samples WHERE session_id=? AND channel=? AND segment_id=? AND revision=?",
                                (session_id, channel, segment_id, revision)).fetchone()

    @staticmethod
    def _identifiers(event: dict) -> tuple[str, str, int] | None:
        channel, segment_id = event.get("channel"), event.get("segment_id")
        revision = event.get("revision", 1)
        if (not isinstance(channel, str) or not channel or not isinstance(segment_id, str)
                or not segment_id or type(revision) is not int or revision < 1):
            return None
        return channel, segment_id, revision

    def record_event(self, session_id: str, event: dict, *, direction: str, provider: str, model: str) -> None:
        kind = event.get("kind")
        if kind not in ("original", "translation", "translation_candidate", "translation_status",
                        "subtitle_final", "sentence_join", "recognition"):
            return
        identifiers = self._identifiers(event)
        if identifiers is None:
            return
        if not session_id or direction not in ("zh_to_ja", "ja_to_zh"):
            raise ValueError("采集会话或翻译方向无效")
        channel, segment_id, revision = identifiers
        with self._lock:
            self._open()
            with self._db:
                if kind in ("recognition", "sentence_join", "subtitle_final"):
                    self._db.execute("INSERT OR REPLACE INTO annotations VALUES (?,?,?,?,?,?)",
                                     (session_id, channel, segment_id, revision, kind, _json(event)))
                if kind == "recognition":
                    self._db.execute("INSERT OR IGNORE INTO segments(session_id,channel,segment_id) VALUES (?,?,?)",
                                     (session_id, channel, segment_id))
                    row = self._db.execute("SELECT metadata FROM segments WHERE session_id=? AND channel=? AND segment_id=?",
                                           (session_id, channel, segment_id)).fetchone()
                    metadata = json.loads(row[0])
                    metadata["recognition"] = event
                    self._db.execute("UPDATE segments SET metadata=? WHERE session_id=? AND channel=? AND segment_id=?",
                                     (_json(metadata), session_id, channel, segment_id))
                    return
                sample = self._sample(session_id, channel, segment_id, revision)
                if kind == "original":
                    source = event.get("text", "")
                    if not isinstance(source, str) or not source.strip():
                        return
                    source = source.strip()
                    ids = event.get("source_segment_ids") or [segment_id]
                    if not isinstance(ids, (list, tuple)) or not all(isinstance(item, str) and item for item in ids):
                        raise ValueError("采集原文的音频片段编号无效")
                    ids = list(dict.fromkeys(ids))
                    annotation = self._db.execute("SELECT payload FROM annotations WHERE session_id=? AND channel=? AND segment_id=? AND revision=? AND kind='sentence_join'",
                                                  (session_id, channel, segment_id, revision)).fetchone()
                    metadata = json.loads(sample["metadata"]) if sample else {}
                    if annotation:
                        metadata["sentence_join"] = json.loads(annotation[0])
                    if type(event.get("context_generation")) is int:
                        metadata["context_generation"] = event["context_generation"]
                    final = (not bool(event.get("provisional", False))) or bool(sample and sample["finalized"]) or bool(self._db.execute(
                        "SELECT 1 FROM annotations WHERE session_id=? AND channel=? AND segment_id=? AND revision=? AND kind='subtitle_final'",
                        (session_id, channel, segment_id, revision)).fetchone())
                    provisional = bool(event.get("provisional", False)) and not final
                    uncertain = (bool(event.get("uncertain", False)) or bool(sample and sample["uncertain"])
                                 or bool(metadata.get("sentence_join", {}).get("uncertain", False)))
                    if sample is None:
                        self._db.execute("""INSERT INTO samples(session_id,channel,segment_id,revision,direction,source,
                            provisional,finalized,uncertain,source_ids,metadata,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                            (session_id, channel, segment_id, revision, direction, source, int(provisional), int(final),
                             int(uncertain), _json(ids), _json(metadata), _timestamp(), _timestamp()))
                    else:
                        changed = sample["source"] != source or sample["direction"] != direction or sample["source_ids"] != _json(ids)
                        if changed and self._db.execute("SELECT 1 FROM candidates WHERE sample_id=?", (sample["id"],)).fetchone():
                            raise ValueError("已有候选的原文不能在同一 revision 中改写，请创建新的 revision 保留来源")
                        self._db.execute("UPDATE samples SET direction=?,source=?,provisional=?,finalized=?,uncertain=?,source_ids=?,metadata=?,updated_at=? WHERE id=?",
                                         (direction, source, int(provisional), int(final), int(uncertain), _json(ids), _json(metadata), _timestamp(), sample["id"]))
                    return
                if sample is None:
                    # A translation cannot create a source or attach to another revision/session.
                    return
                if kind == "subtitle_final":
                    self._db.execute("UPDATE samples SET provisional=0,finalized=1,updated_at=? WHERE id=?", (_timestamp(), sample["id"]))
                elif kind == "sentence_join":
                    metadata = json.loads(sample["metadata"])
                    metadata["sentence_join"] = event
                    self._db.execute("UPDATE samples SET metadata=?,uncertain=MAX(uncertain,?),updated_at=? WHERE id=?",
                                     (_json(metadata), int(bool(event.get("uncertain", False))), _timestamp(), sample["id"]))
                elif kind in ("translation", "translation_candidate", "translation_status"):
                    metadata = json.loads(sample["metadata"])
                    source_generation = metadata.get("context_generation")
                    if (source_generation is not None and event.get("context_generation") is not None
                            and source_generation != event["context_generation"]):
                        return
                    actual_provider = event.get("translation_provider")
                    if actual_provider in ("local", "codex"):
                        provider = actual_provider
                    elif provider == "hybrid":
                        provider = {"draft": "local", "final": "codex"}.get(event.get("translation_stage"))
                    if provider not in ("local", "codex"):
                        return
                    model = str(event.get("translation_model") or model)
                    failed = bool(event.get("failed", False))
                    provisional = bool(sample["provisional"]) and not sample["finalized"]
                    text = event.get("text", "")
                    successful = kind != "translation_status" and not failed and isinstance(text, str) and bool(text.strip())
                    if successful:
                        self._add_candidate(sample["id"], text=text, origin=provider, model=model,
                                            processing_seconds=event.get("processing_seconds", 0) or 0)
                    branches = metadata.setdefault("translation_branches", {})
                    previous = branches.get(provider, {})
                    # A delayed branch event is still a candidate, never a replacement
                    # for the visible final translation or for another branch's status.
                    if not (previous.get("state") == "success" and failed):
                        branch = {key: event.get(key) for key in ("translation_stage", "latency",
                            "processing_seconds", "queue_wait_seconds", "batch_id", "error", "final_failed")
                                  if key in event}
                        branch.update(provider=provider, model=model, state="failed" if failed else "success",
                                      displayed=kind == "translation")
                        branches[provider] = branch
                    if kind == "translation" and successful:
                        previous_display = metadata.get("translation_display", {})
                        stage = event.get("translation_stage", "final")
                        if stage == "final" or previous_display.get("stage") != "final":
                            metadata["translation_display"] = {"stage": stage, "provider": provider, "model": model}
                    if kind == "translation_status":
                        status_event = dict(event)
                        status_event.pop("text", None)
                        metadata.setdefault("translation_status", {})[provider] = status_event
                    any_candidate = self._db.execute("SELECT 1 FROM candidates WHERE sample_id=?", (sample["id"],)).fetchone()
                    overall_failed = bool(event.get("overall_failed", failed if kind == "translation" else False))
                    sample_failed = (overall_failed or bool(sample["failed"])) and not any_candidate
                    context = (str(event.get("context", sample["context"]))
                               if kind == "translation" else sample["context"])
                    self._db.execute("UPDATE samples SET context=?,failed=?,provisional=?,metadata=?,updated_at=? WHERE id=?",
                                     (context, int(sample_failed), int(provisional), _json(metadata), _timestamp(), sample["id"]))

    def save_audio(self, session_id: str, job, accepted) -> None:
        channel, segment_id = job.channel, job.segment_id
        samples = np.asarray(job.samples, dtype=np.float32)
        if not session_id or not channel or not segment_id or samples.ndim != 1 or not np.isfinite(samples).all():
            raise ValueError("专用模型音频必须是有效的单声道采集片段")
        with self._lock:
            self._open()
            row = self._db.execute("SELECT * FROM segments WHERE session_id=? AND channel=? AND segment_id=?",
                                   (session_id, channel, segment_id)).fetchone()
            if row is not None and row["audio_path"]:
                return
            with self._db:
                self._db.execute("INSERT OR IGNORE INTO segments(session_id,channel,segment_id) VALUES (?,?,?)", (session_id, channel, segment_id))
                if self._audio_bytes + samples.size * 2 + 44 > self.AUDIO_QUOTA_BYTES:
                    self._db.execute("INSERT OR REPLACE INTO corpus_state VALUES ('collection_error',?)", ("音频采集达到 2 GiB 上限，已停止保存新音频；已有数据保留。",))
                    return
                # Hash identifiers rather than using user/event text as a filesystem path.
                name = hashlib.sha256(_json([session_id, channel, segment_id]).encode("utf-8")).hexdigest() + ".wav"
                path = self._audio_root / name
                if path.exists():
                    raise FileExistsError("发现未登记的已有音频，停止写入以保留原始数据")
                pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()
                with wave.open(str(path), "wb") as stream:
                    stream.setnchannels(1)
                    stream.setsampwidth(2)
                    stream.setframerate(16000)
                    stream.writeframes(pcm)
                size = path.stat().st_size
                self._audio_bytes += size
                metadata = json.loads(row["metadata"]) if row else {}
                metadata.update({"start_sample": getattr(job, "start_sample", None), "end_sample": getattr(job, "end_sample", None),
                                 "generation": getattr(job, "generation", None), "ended_by": getattr(job, "ended_by", "unknown"),
                                 "audio_duration_seconds": samples.size / 16000, "observed_at": _timestamp()})
                self._db.execute("UPDATE segments SET audio_path=?,audio_bytes=?,accepted=?,metadata=? WHERE session_id=? AND channel=? AND segment_id=?",
                                 (str(path), size, int(bool(accepted)), _json(metadata), session_id, channel, segment_id))

    def _add_candidate(self, sample_id: int, *, text: str, origin: str, model: str, processing_seconds: float = 0) -> int:
        if origin not in ("local", "codex") or not isinstance(text, str) or not text.strip() or not isinstance(model, str) or not model.strip():
            raise ValueError("候选译文、来源或模型编号无效")
        seconds = float(processing_seconds)
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("候选翻译耗时无效")
        if self._db.execute("SELECT 1 FROM samples WHERE id=?", (sample_id,)).fetchone() is None:
            raise ValueError("候选译文对应的原文不存在")
        self._db.execute("INSERT OR IGNORE INTO candidates(sample_id,text,origin,model,processing_seconds,created_at) VALUES (?,?,?,?,?,?)",
                         (sample_id, text.strip(), origin, model.strip(), seconds, _timestamp()))
        return self._db.execute("SELECT id FROM candidates WHERE sample_id=? AND origin=? AND model=? AND text=?",
                                (sample_id, origin, model.strip(), text.strip())).fetchone()[0]

    def add_candidate(self, sample_id: int, *, text: str, origin: str, model: str, processing_seconds: float = 0) -> int:
        with self._lock:
            self._open()
            with self._db:
                return self._add_candidate(sample_id, text=text, origin=origin, model=model, processing_seconds=processing_seconds)

    def _record(self, row) -> dict:
        result = dict(row)
        for name in ("provisional", "finalized", "uncertain", "failed"):
            result[name] = bool(result[name])
        result["source_ids"] = json.loads(result["source_ids"])
        result["metadata"] = json.loads(result["metadata"])
        result["candidates"] = []
        for candidate in self._db.execute("SELECT * FROM candidates WHERE sample_id=? ORDER BY id", (row["id"],)).fetchall():
            candidate = dict(candidate)
            for name in ("reviewed", "source_verified", "rights_confirmed", "training_allowed"):
                candidate[name] = bool(candidate[name])
            result["candidates"].append(candidate)
        segments = []
        for source_id in result["source_ids"]:
            segment = self._db.execute("SELECT * FROM segments WHERE session_id=? AND channel=? AND segment_id=?",
                                       (row["session_id"], row["channel"], source_id)).fetchone()
            if segment:
                segment = dict(segment)
                segment["metadata"] = json.loads(segment["metadata"])
                segment["accepted"] = bool(segment["accepted"])
                segments.append(segment)
        result["audio_paths"] = [segment["audio_path"] for segment in segments if segment["audio_path"]]
        result["segments"] = segments
        return result

    def list_samples(self, limit: int = 100) -> list[dict]:
        with self._lock:
            self._open()
            rows = self._db.execute(f"SELECT s.* FROM samples s WHERE {self._LATEST} ORDER BY s.id DESC LIMIT ?", (max(0, min(int(limit), 10000)),)).fetchall()
            return [self._record(row) for row in rows]

    def get_sample(self, sample_id: int) -> dict | None:
        with self._lock:
            self._open()
            row = self._db.execute("SELECT * FROM samples WHERE id=?", (sample_id,)).fetchone()
            return self._record(row) if row else None

    def pending_comparisons(self, model_ids: list[str], limit: int = 30) -> list[dict]:
        models = list(dict.fromkeys(model_ids))
        if not all(isinstance(model, str) and model for model in models):
            raise ValueError("本地对照模型编号无效")
        if not models:
            return []
        with self._lock:
            self._open()
            # Filter missing models in SQL so completed recent rows cannot hide old work.
            clauses = " OR ".join("NOT EXISTS(SELECT 1 FROM candidates c WHERE c.sample_id=s.id AND c.origin='local' AND c.model=?)" for _ in models)
            rows = self._db.execute(f"SELECT s.* FROM samples s WHERE {self._LATEST} AND {self._CLEAN} AND ({clauses}) ORDER BY s.id LIMIT ?",
                                    (*models, max(0, min(int(limit), 10000)))).fetchall()
            results = []
            for row in rows:
                result = self._record(row)
                present = {candidate["model"] for candidate in result["candidates"] if candidate["origin"] == "local"}
                result["pending_model_ids"] = [model for model in models if model not in present]
                results.append(result)
            return results

    def review_candidate(self, sample_id: int, candidate_id: int, *, source_verified: bool, rights_confirmed: bool) -> None:
        if type(source_verified) is not bool or type(rights_confirmed) is not bool:
            raise ValueError("原文核对与使用权确认必须为明确的布尔值")
        with self._lock:
            self._open()
            candidate = self._db.execute("SELECT * FROM candidates WHERE id=? AND sample_id=?", (candidate_id, sample_id)).fetchone()
            if candidate is None:
                raise ValueError("待审核候选不存在或不属于当前原文")
            clean = self._db.execute(f"SELECT 1 FROM samples s WHERE s.id=? AND {self._LATEST} AND {self._CLEAN}", (sample_id,)).fetchone()
            eligible = bool(clean and source_verified and rights_confirmed and candidate["origin"] == "local")
            with self._db:
                # One selected training label per source; prior reviews remain auditable.
                if eligible:
                    self._db.execute("UPDATE candidates SET training_allowed=0 WHERE sample_id=?", (sample_id,))
                self._db.execute("UPDATE candidates SET reviewed=1,source_verified=?,rights_confirmed=?,training_allowed=?,reviewed_at=? WHERE id=?",
                                 (int(source_verified), int(rights_confirmed), int(eligible), _timestamp(), candidate_id))

    def summary(self) -> dict:
        with self._lock:
            self._open()
            latest = self._LATEST
            counts = self._db.execute(f"""SELECT COUNT(DISTINCT s.id) samples,
                COUNT(CASE WHEN c.origin='local' THEN 1 END) local_candidates,
                COUNT(CASE WHEN c.origin='codex' THEN 1 END) codex_candidates,
                COUNT(CASE WHEN c.reviewed=1 THEN 1 END) reviewed,
                COUNT(DISTINCT CASE WHEN c.training_allowed=1 AND c.origin='local' AND {self._CLEAN} THEN s.id END) trainable
                FROM samples s LEFT JOIN candidates c ON s.id=c.sample_id WHERE {latest}""").fetchone()
            result = dict(counts)
            result.update(audio_files=self._db.execute("SELECT COUNT(*) FROM segments WHERE audio_path IS NOT NULL").fetchone()[0],
                          audio_bytes=self._audio_bytes, audio_quota_bytes=self.AUDIO_QUOTA_BYTES,
                          deployment_locked=True, phase="数据积累")
            error = self._db.execute("SELECT value FROM corpus_state WHERE key='collection_error'").fetchone()
            if error:
                result["collection_error"] = error[0]
            return result

    @staticmethod
    def _source_group(sample: dict) -> str:
        normalized = "".join(unicodedata.normalize("NFKC", sample["source"]).split())
        return hashlib.sha256(_json([sample["direction"], normalized]).encode("utf-8")).hexdigest()

    @staticmethod
    def _hy_prompt(source: str, direction: str, context: str) -> str:
        source_lang, target_lang = ("ja", "zh") if direction == "ja_to_zh" else ("zh", "ja")
        return hy_translation_prompt(source, source_lang, target_lang, context)

    @staticmethod
    def _new_session_split(session: str) -> str:
        bucket = int(hashlib.sha256(session.encode("utf-8")).hexdigest()[:8], 16) / 0x100000000
        return "validation" if bucket < .2 else "train"

    def export_dataset(self) -> dict:
        with self._lock:
            self._open()
            rows = self._db.execute(f"""SELECT s.*,c.id candidate_id,c.text target,c.model teacher_model,
                c.reviewed,c.source_verified,c.rights_confirmed,c.origin
                FROM samples s JOIN candidates c ON c.sample_id=s.id
                WHERE {self._LATEST} AND {self._CLEAN} AND c.training_allowed=1 AND c.origin='local'
                AND c.reviewed=1 AND c.source_verified=1 AND c.rights_confirmed=1 ORDER BY s.id""").fetchall()
            assignments = {row["session_id"]: row["split"] for row in self._db.execute("SELECT * FROM session_splits").fetchall()}
            frozen_sources = {row["source_group"]: row["split"] for row in self._db.execute("SELECT * FROM source_splits").fetchall()}
            records, seen, frozen_conflicts = [], set(), 0
            for row in rows:
                sample = dict(row)
                group = self._source_group(sample)
                if group in frozen_sources:
                    session_split = assignments.get(sample["session_id"], self._new_session_split(sample["session_id"]))
                    if session_split != frozen_sources[group]:
                        frozen_conflicts += 1
                        continue
                if group in seen:
                    continue
                seen.add(group)
                context = bounded_source_context(sample["context"])
                prompt = self._hy_prompt(sample["source"], sample["direction"], context)
                records.append({"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": sample["target"]}],
                                "direction": sample["direction"], "raw_source": sample["source"], "source_context": context,
                                "prompt_format": HY_PROMPT_FORMAT, "translation_profile": PROFILE_ID, "provenance": {
                                    "origin": "local_model", "reviewed": True, "source_verified": True, "rights_confirmed": True,
                                    "codex_derived": False, "source_session": sample["session_id"], "source_group": group,
                                    "sample_id": sample["id"], "candidate_id": sample["candidate_id"], "model": sample["teacher_model"],
                                    "source_segment_ids": json.loads(sample["source_ids"])}})
            sessions = sorted({record["provenance"]["source_session"] for record in records}, key=lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest())
            if len(sessions) < 2:
                raise ValueError("至少需要 2 个独立会话的有效已审核本地候选，才能分离训练与验证集；Codex 候选不进入训练。")
            if not assignments:
                validation_count = min(len(sessions) - 1, max(1, math.ceil(len(sessions) * .2)))
                validation_sessions = set(sessions[:validation_count])
                assignments = {session: "validation" if session in validation_sessions else "train" for session in sessions}
            else:
                # A later export cannot turn a prior holdout session into training data.
                for session in sessions:
                    if session not in assignments:
                        assignments[session] = self._new_session_split(session)
                validation_sessions = {session for session in sessions if assignments[session] == "validation"}
            training, validation = [], []
            for record in records:
                (validation if record["provenance"]["source_session"] in validation_sessions else training).append(record)
            if not training or not validation:
                raise ValueError("当前有效已审核样本未覆盖独立训练与验证会话；已冻结的验证会话不会改成训练数据，请继续积累。")
            with self._db:
                self._db.executemany("INSERT OR IGNORE INTO session_splits VALUES (?,?)", assignments.items())
                self._db.executemany("INSERT OR IGNORE INTO source_splits VALUES (?,?)",
                                     ((record["provenance"]["source_group"], assignments[record["provenance"]["source_session"]]) for record in records))
            directory = self.root / "datasets" / (datetime.now(timezone(timedelta(hours=8))).strftime("export-%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8])
            directory.mkdir(parents=True, exist_ok=False)
            for name, split in (("train.jsonl", training), ("validation.jsonl", validation)):
                with (directory / name).open("x", encoding="utf-8", newline="\n") as stream:
                    for record in split:
                        stream.write(_json(record) + "\n")
            manifest = {"format_version": 1, "deployment_locked": True, "exported_at": _timestamp(),
                        "train_file": "train.jsonl", "validation_file": "validation.jsonl", "train_rows": len(training),
                        "validation_rows": len(validation), "train_sessions": sorted(set(sessions) - validation_sessions),
                        "validation_sessions": sorted(validation_sessions), "deduplicated_source_groups": len(records),
                        "codex_rows": 0, "label_origin": "reviewed_local_candidates", "model_trained": False,
                        "prompt_format": HY_PROMPT_FORMAT, "translation_profile": PROFILE_ID,
                        "source_context_max_characters": HY_CONTEXT_CHARACTERS,
                        "holdout_frozen_by_session": True, "holdout_frozen_by_source": True,
                        "excluded_frozen_conflicts": frozen_conflicts}
            (directory / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            return {**manifest, "directory": str(directory), "manifest": str(directory / "manifest.json")}

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._db.close()
                self._closed = True
