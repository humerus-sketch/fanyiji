import concurrent.futures
import json
from pathlib import Path
from types import SimpleNamespace
import wave

import numpy as np
import pytest

from fanyiji.specialized_store import SpecializedStore


@pytest.fixture
def store(tmp_path):
    corpus = SpecializedStore(tmp_path / "specialized_model")
    yield corpus
    corpus.close()


def source(store, session="session-a", identifier="a", text="明日も雨です。", *, revision=1, **metadata):
    store.record_event(session, {"kind": "original", "channel": "system", "segment_id": identifier,
        "revision": revision, "text": text, **metadata}, direction="ja_to_zh", provider="codex", model="gpt-test")
    return next(sample for sample in store.list_samples() if sample["session_id"] == session and sample["segment_id"] == identifier)


def reviewed_local(store, session="session-a", identifier="a", text="明日も雨です。", target="明天也下雨。"):
    sample = source(store, session, identifier, text)
    candidate = store.add_candidate(sample["id"], text=target, origin="local", model="hy-1.8b", processing_seconds=.2)
    store.review_candidate(sample["id"], candidate, source_verified=True, rights_confirmed=True)
    return sample, candidate


def test_codex_provenance_is_quarantined_even_after_explicit_review(store):
    sample = source(store)
    store.record_event("session-a", {"kind": "translation", "channel": "system", "segment_id": "a",
        "text": "明天也下雨。", "processing_seconds": 3.2}, direction="ja_to_zh", provider="codex", model="gpt-actual")
    candidate = store.get_sample(sample["id"])["candidates"][0]
    assert candidate["origin"] == "codex"
    assert candidate["model"] == "gpt-actual"
    store.review_candidate(sample["id"], candidate["id"], source_verified=True, rights_confirmed=True)
    assert store.get_sample(sample["id"])["candidates"][0]["reviewed"]
    assert not store.get_sample(sample["id"])["candidates"][0]["training_allowed"]
    assert store.summary()["trainable"] == 0
    assert store.summary()["deployment_locked"] is True
    with pytest.raises(ValueError, match="2 个独立会话"):
        store.export_dataset()


def test_missing_original_cannot_attach_translation_to_another_session_or_revision(store):
    sample = source(store)
    for session, revision in (("session-b", 1), ("session-a", 2)):
        store.record_event(session, {"kind": "translation", "channel": "system", "segment_id": "a",
            "revision": revision, "text": "译文"}, direction="ja_to_zh", provider="local", model="hy-7b")
    assert store.summary()["samples"] == 1
    assert store.get_sample(sample["id"])["candidates"] == []


def test_confirmation_before_delayed_translation_stays_final(store):
    sample = source(store, provisional=True)
    store.record_event("session-a", {"kind": "subtitle_final", "channel": "system", "segment_id": "a"},
                       direction="ja_to_zh", provider="codex", model="gpt")
    store.record_event("session-a", {"kind": "translation", "channel": "system", "segment_id": "a",
        "provisional": True, "text": "明天也下雨。"}, direction="ja_to_zh", provider="local", model="hy-1.8b")
    record = store.get_sample(sample["id"])
    assert record["finalized"] and not record["provisional"]
    store.review_candidate(sample["id"], record["candidates"][0]["id"], source_verified=True, rights_confirmed=True)
    assert store.summary()["trainable"] == 1


def test_translation_alone_does_not_certify_a_provisional_source(store):
    sample = source(store, provisional=True)
    store.record_event("session-a", {"kind": "translation", "channel": "system", "segment_id": "a",
        "provisional": False, "text": "明天也下雨。"}, direction="ja_to_zh", provider="local", model="hy")
    record = store.get_sample(sample["id"])
    assert record["provisional"]
    store.review_candidate(sample["id"], record["candidates"][0]["id"], source_verified=True, rights_confirmed=True)
    assert store.summary()["trainable"] == 0


def test_new_revision_excludes_old_reviewed_label_and_does_not_mix_candidates(store):
    previous, _ = reviewed_local(store)
    current = source(store, text="明日は雨ではありません。", revision=2, source_segment_ids=["a", "b"])
    assert store.summary()["samples"] == 1
    assert store.summary()["trainable"] == 0
    assert store.list_samples()[0]["revision"] == 2
    assert store.get_sample(previous["id"])["candidates"][0]["reviewed"]
    assert current["candidates"] == []
    assert store.pending_comparisons(["hy-1.8b"])[0]["id"] == current["id"]


@pytest.mark.parametrize("flag", ["uncertain", "provisional"])
def test_dirty_sources_cannot_become_trainable_or_comparison_targets(store, flag):
    sample = source(store, **{flag: True})
    candidate = store.add_candidate(sample["id"], text="明天也下雨。", origin="local", model="hy")
    store.review_candidate(sample["id"], candidate, source_verified=True, rights_confirmed=True)
    assert store.summary()["trainable"] == 0
    assert store.pending_comparisons(["other"]) == []


def test_failure_is_recorded_without_saving_error_message_as_a_label(store):
    sample = source(store)
    store.record_event("session-a", {"kind": "translation", "channel": "system", "segment_id": "a",
        "failed": True, "text": "[翻译失败]"}, direction="ja_to_zh", provider="codex", model="gpt")
    record = store.get_sample(sample["id"])
    assert record["failed"] and not record["candidates"]
    assert store.pending_comparisons(["hy"]) == []


def test_explicit_review_needs_both_verification_and_rights(store):
    sample = source(store)
    candidate = store.add_candidate(sample["id"], text="明天也下雨。", origin="local", model="hy")
    for verified, rights in ((False, True), (True, False), (False, False)):
        store.review_candidate(sample["id"], candidate, source_verified=verified, rights_confirmed=rights)
        assert store.summary()["trainable"] == 0
    with pytest.raises(ValueError, match="布尔"):
        store.review_candidate(sample["id"], candidate, source_verified="true", rights_confirmed=True)
    store.review_candidate(sample["id"], candidate, source_verified=True, rights_confirmed=True)
    assert store.summary()["trainable"] == 1


def test_new_selected_local_label_keeps_prior_review_but_only_one_trainable_target(store):
    sample, first = reviewed_local(store)
    second = store.add_candidate(sample["id"], text="明天仍有雨。", origin="local", model="hy-7b")
    store.review_candidate(sample["id"], second, source_verified=True, rights_confirmed=True)
    candidates = {candidate["id"]: candidate for candidate in store.get_sample(sample["id"])["candidates"]}
    assert candidates[first]["reviewed"] and not candidates[first]["training_allowed"]
    assert candidates[second]["training_allowed"]
    assert store.summary()["trainable"] == 1


def test_pending_models_do_not_hide_old_work_behind_completed_recent_rows(store):
    old = source(store, identifier="old")
    recent = source(store, identifier="recent", text="予定を変更します。")
    for model in ("hy", "qwen"):
        store.add_candidate(recent["id"], text="更改安排。", origin="local", model=model)
    store.add_candidate(old["id"], text="明天也下雨。", origin="local", model="hy")
    pending = store.pending_comparisons(["hy", "qwen", "qwen"], limit=1)
    assert len(pending) == 1 and pending[0]["id"] == old["id"]
    assert pending[0]["pending_model_ids"] == ["qwen"]


def test_audio_merges_racing_recognition_and_resolves_joined_source_paths(store):
    for identifier in ("../a", "b"):
        store.record_event("session-a", {"kind": "recognition", "channel": "system", "segment_id": identifier,
            "text": "森の", "quality": {"uncertain": False}}, direction="ja_to_zh", provider="codex", model="gpt")
        job = SimpleNamespace(channel="system", segment_id=identifier, samples=np.array([-2, -.5, .5, 2], np.float32))
        store.save_audio("session-a", job, True)
        store.save_audio("session-a", job, True)
    store.record_event("session-a", {"kind": "sentence_join", "channel": "system", "segment_id": "../a", "revision": 2,
        "uncertain": True, "raw_source_texts": ["森の", "妖精です"]}, direction="ja_to_zh", provider="codex", model="gpt")
    sample = source(store, identifier="../a", text="森の妖精です。", revision=2, source_segment_ids=["../a", "b"])
    assert sample["uncertain"]
    assert len(sample["audio_paths"]) == 2
    assert sample["metadata"]["sentence_join"]["raw_source_texts"] == ["森の", "妖精です"]
    assert sample["segments"][0]["metadata"]["recognition"]["text"] == "森の"
    for path in sample["audio_paths"]:
        assert str(store.root / "audio") in path
        with wave.open(path, "rb") as stream:
            assert (stream.getnchannels(), stream.getsampwidth(), stream.getframerate(), stream.getnframes()) == (1, 2, 16000, 4)
            assert np.frombuffer(stream.readframes(4), dtype="<i2").tolist() == [-32767, -16383, 16383, 32767]
    assert store.summary()["audio_files"] == 2


def test_audio_quota_survives_reopen_without_deleting_any_recording(tmp_path, monkeypatch):
    monkeypatch.setattr(SpecializedStore, "AUDIO_QUOTA_BYTES", 60)
    root = tmp_path / "corpus"
    first = SpecializedStore(root)
    first.save_audio("s", SimpleNamespace(channel="system", segment_id="a", samples=np.zeros(8, np.float32)), True)
    first.close()
    second = SpecializedStore(root)
    try:
        second.save_audio("s", SimpleNamespace(channel="system", segment_id="b", samples=np.zeros(1, np.float32)), True)
        assert second.summary()["audio_files"] == 1
        assert second.summary()["audio_bytes"] == 60
        assert "停止保存" in second.summary()["collection_error"]
        assert len(list((root / "audio").glob("*.wav"))) == 1
    finally:
        second.close()


def test_export_is_disjoint_reviewed_local_only_and_preserves_provenance(store):
    for session, identifier, text in (("s1", "a", "金額は１５０円です。"), ("s2", "b", "延期しません。"), ("s3", "c", "金額は150円です。")):
        sample, _ = reviewed_local(store, session, identifier, text)
        codex = store.add_candidate(sample["id"], text="Codex correction", origin="codex", model="gpt")
        store.review_candidate(sample["id"], codex, source_verified=True, rights_confirmed=True)
    unreviewed = source(store, session="s4", identifier="d", text="未確認。")
    store.add_candidate(unreviewed["id"], text="未核对。", origin="local", model="hy")
    result = store.export_dataset()
    assert result["deployment_locked"] and not result["model_trained"]
    assert result["codex_rows"] == 0
    assert result["train_rows"] + result["validation_rows"] == 2
    exported = []
    splits = []
    for filename in ("train.jsonl", "validation.jsonl"):
        records = [json.loads(line) for line in (Path(result["directory"]) / filename).read_text(encoding="utf-8").splitlines()]
        exported.extend(records)
        splits.append({record["provenance"]["source_session"] for record in records})
    assert splits[0].isdisjoint(splits[1])
    for record in exported:
        assert record["messages"][0]["content"].startswith("将以下文本翻译为简体中文")
        assert record["provenance"]["origin"] == "local_model"
        assert record["provenance"]["reviewed"] and record["provenance"]["rights_confirmed"]
        assert record["provenance"]["codex_derived"] is False
        assert "Codex" not in record["messages"][1]["content"]


def test_export_requires_independent_sessions_after_deduplication(store):
    reviewed_local(store, "s1")
    reviewed_local(store, "s2")
    with pytest.raises(ValueError, match="2 个独立会话"):
        store.export_dataset()


def test_validation_sessions_remain_frozen_when_more_data_arrives(store):
    reviewed_local(store, "s1", text="資料を送ります。")
    reviewed_local(store, "s2", text="午後に戻ります。")
    first = store.export_dataset()
    for index in range(8):
        reviewed_local(store, f"new-{index}", text=f"第{index}件の予定です。")
    second = store.export_dataset()
    assert set(first["validation_sessions"]).issubset(second["validation_sessions"])
    assert set(first["train_sessions"]).issubset(second["train_sessions"])
    assert first["directory"] != second["directory"]


def test_old_validation_source_cannot_move_to_training_after_new_revision(store):
    first_texts = {"s1": "資料を送ります。", "s2": "午後に戻ります。"}
    for session, text in first_texts.items():
        reviewed_local(store, session, text=text)
    first = store.export_dataset()
    validation_session = first["validation_sessions"][0]
    held_out_text = first_texts[validation_session]
    source(store, validation_session, text="聞き取れませんでした。", revision=2, uncertain=True)
    reviewed_local(store, validation_session, "new-validation", text="追加の検証資料です。")
    new_session = next(f"replayed-{index}" for index in range(100) if store._new_session_split(f"replayed-{index}") == "train")
    reviewed_local(store, new_session, text=held_out_text)
    second = store.export_dataset()
    assert second["excluded_frozen_conflicts"] == 1
    assert second["holdout_frozen_by_source"] is True
    training = [json.loads(line) for line in (Path(second["directory"]) / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(not record["messages"][0]["content"].endswith(held_out_text) for record in training)


def test_same_revision_source_rewrite_cannot_reuse_existing_teacher_labels(store):
    sample, candidate = reviewed_local(store)
    with pytest.raises(ValueError, match="新的 revision"):
        source(store, text="明日は雨ではありません。")
    unchanged = store.get_sample(sample["id"])
    assert unchanged["source"] == "明日も雨です。"
    assert unchanged["candidates"][0]["id"] == candidate


def test_contextual_export_preserves_raw_source_and_exact_production_prompt(store):
    from fanyiji.gguf_engine import GgufTranslationEngine
    from fanyiji.specialized_training import load_training_dataset
    from fanyiji.text_context import bounded_source_context
    from fanyiji.livestream_prompt import HY_CONTEXT_CHARACTERS, HY_PROMPT_FORMAT, PROFILE_ID

    local_sample = source(store, "s1", text="それを変更します。")
    context = "以前の予定は午後でした。" * 40
    store.record_event("s1", {"kind": "translation", "channel": "system", "segment_id": "a",
        "text": "更改这个安排。", "context": context}, direction="ja_to_zh", provider="local", model="hy-7b")
    candidate = store.get_sample(local_sample["id"])["candidates"][0]
    store.review_candidate(local_sample["id"], candidate["id"], source_verified=True, rights_confirmed=True)
    reviewed_local(store, "s2", text="延期しません。")
    result = store.export_dataset()
    report, training, validation = load_training_dataset(Path(result["directory"]))
    contextual = next(record for record in training + validation if record["raw_source"] == "それを変更します。")
    engine = object.__new__(GgufTranslationEngine)
    engine.model_kind = "hy"
    assert contextual["source_context"] == bounded_source_context(context)
    assert len(contextual["source_context"]) == HY_CONTEXT_CHARACTERS
    assert contextual["prompt_format"] == result["prompt_format"] == HY_PROMPT_FORMAT
    assert contextual["translation_profile"] == result["translation_profile"] == PROFILE_ID
    assert result["source_context_max_characters"] == HY_CONTEXT_CHARACTERS
    assert report["prompt_format"] == HY_PROMPT_FORMAT and report["translation_profile"] == PROFILE_ID
    assert contextual["messages"][0]["content"] == engine._translation_prompt("それを変更します。", "ja", "zh", bounded_source_context(context))
    assert report["sessions_disjoint"] and report["source_groups_disjoint"]


def test_threaded_writes_are_serialized_and_close_is_idempotent(store):
    def write(index):
        sample = source(store, identifier=str(index), text=f"音声{index}です。")
        return store.add_candidate(sample["id"], text=f"声音{index}。", origin="local", model="hy")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        ids = list(pool.map(write, range(20)))
    assert len(set(ids)) == 20
    assert store.summary()["samples"] == 20
    store.close()
    store.close()
    with pytest.raises(RuntimeError, match="已关闭"):
        store.summary()
