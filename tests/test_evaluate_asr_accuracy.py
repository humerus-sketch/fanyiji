"""Independent reference provenance, corpus identity and CER omission handling."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from evaluate_asr_accuracy import (character_edits, evaluate, load_corpus, load_references,
                                  normalize, reference_template, score_report, write_new)


def sample(identifier="ja1", **extra):
    return {"id": identifier, "source_lang": "ja", "sha256": "a" * 64,
            "source_kind": "saved_application_audio", **extra}


def truth(identifier="ja1", text="行かない", **extra):
    return {"id": identifier, "source_lang": "ja", "audio_sha256": "a" * 64,
            "reference_text": text, "reference_kind": "human_transcription",
            "verified_against_audio": True, "annotator": "test reviewer",
            "reference_source": "Independent listening to the original waveform", **extra}


def report(text=None, *, identifier="ja1", **extra):
    return {"model_id": "test", "samples": [sample(identifier, production_text=text, **extra)]}


def test_normalization_preserves_japanese_meaning_and_spelling_differences():
    assert normalize(" ＰＳ５、 行かない。\n") == "ps5行かない"
    assert normalize("ｶﾞ") == "ガ"
    assert normalize("行かない") != normalize("行く")
    assert normalize("二千七十七") != normalize("2077")
    assert normalize("七人") != normalize("しちにん")
    assert normalize("ゲーム") == "ゲーム"


@pytest.mark.parametrize("reference,hypothesis,expected", [
    ("abc", "abc", {"edits": 0, "substitutions": 0, "deletions": 0, "insertions": 0}),
    ("abc", "axc", {"edits": 1, "substitutions": 1, "deletions": 0, "insertions": 0}),
    ("abc", "ac", {"edits": 1, "substitutions": 0, "deletions": 1, "insertions": 0}),
    ("abc", "abxc", {"edits": 1, "substitutions": 0, "deletions": 0, "insertions": 1}),
    ("", "abc", {"edits": 3, "substitutions": 0, "deletions": 0, "insertions": 3}),
    ("abc", "", {"edits": 3, "substitutions": 0, "deletions": 3, "insertions": 0}),
])
def test_edit_counts(reference, hypothesis, expected):
    assert character_edits(reference, hypothesis) == expected


def test_no_reference_cannot_report_accuracy_or_ranking():
    corpus = {"ja1": sample()}
    references, counts = load_references(None, "manifest", corpus)
    assert counts == {"missing_reference": 1}
    result = score_report(report("行く"), corpus, references)
    assert result["status"] == "not_scored"
    assert result["micro_cer"] is None
    assert result["accuracy_percentage"] is None
    assert result["samples"] == []


@pytest.mark.parametrize("kind", ["model_output", "codex_correction", "whisper_reference", "synthetic_script"])
def test_model_output_and_synthetic_script_cannot_be_reference_truth(kind):
    with pytest.raises(ValueError, match="not a model output"):
        load_references({"corpus_sha256": "manifest", "samples": [truth(reference_kind=kind)]},
                        "manifest", {"ja1": sample()})


def test_unreviewed_and_uncertain_transcripts_are_excluded():
    corpus = {"ja1": sample(), "ja2": sample("ja2")}
    document = {"corpus_sha256": "manifest", "samples": [
        truth(verified_against_audio=False), truth("ja2", uncertain_ranges=[{"start": .2, "end": .4}])]}
    references, counts = load_references(document, "manifest", corpus)
    assert references == {}
    assert counts == {"unreviewed_reference": 1, "uncertain_reference": 1}


@pytest.mark.parametrize("extra", [{"audio_sha256": "b" * 64}, {"source_lang": "zh"}])
def test_reference_wrong_audio_or_language_is_rejected(extra):
    with pytest.raises(ValueError, match="identity/language"):
        load_references({"corpus_sha256": "manifest", "samples": [truth(**extra)]}, "manifest", {"ja1": sample()})


def test_reference_wrong_corpus_and_missing_annotator_are_rejected():
    with pytest.raises(ValueError, match="corpus SHA"):
        load_references({"corpus_sha256": "wrong", "samples": [truth()]}, "manifest", {"ja1": sample()})
    with pytest.raises(ValueError, match="annotator"):
        load_references({"corpus_sha256": "manifest", "samples": [truth(annotator=None)]}, "manifest", {"ja1": sample()})


def test_micro_cer_is_weighted_by_reference_length_and_omissions_are_counted():
    corpus = {"ja1": sample(), "ja2": sample("ja2")}
    references = {"ja1": truth(text="abc"), "ja2": truth("ja2", text="a")}
    result = score_report(report("abc"), corpus, references)
    assert result["micro_cer"] == .25  # Missing ja2 is a deletion, not dropped.
    assert result["missing_hypothesis_ids"] == ["ja2"]
    assert result["speech_edit_totals"]["reference_characters"] == 4
    assert score_report(report(None), {"ja1": sample()}, {"ja1": truth()})["micro_cer"] == 1


def test_verified_silence_hallucination_is_reported_separately_without_zero_division():
    result = score_report(report("ご視聴"), {"ja1": sample()}, {"ja1": truth(text="")})
    assert result["micro_cer"] is None
    assert result["verified_no_speech"] == {"sample_count": 1, "false_positive_samples": 1, "inserted_characters": 3}


def test_bad_report_identity_and_duplicate_samples_are_rejected():
    with pytest.raises(ValueError, match="ASR report audio SHA"):
        score_report(report("test", sha256="b" * 64), {"ja1": sample()}, {})
    with pytest.raises(ValueError, match="unique"):
        score_report({"samples": [sample(), sample()]}, {"ja1": sample()}, {})


def test_blind_review_template_contains_no_hypotheses_or_production_truth():
    corpus = {"ja1": sample(audio_path="clip.wav", production_text="a model said this", human_truth="old implicit label")}
    template = reference_template("manifest", corpus)
    assert template["samples"][0]["reference_text"] is None
    assert template["samples"][0]["verified_against_audio"] is False
    assert "a model said this" not in json.dumps(template)
    assert "old implicit label" not in json.dumps(template)


def test_frozen_corpus_requires_audio_identity_and_filters_controls(tmp_path):
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"fixed input")
    sha = hashlib.sha256(audio.read_bytes()).hexdigest()
    items = [sample(audio_path="clip.wav", sha256=sha),
             sample("ja2", audio_path="clip.wav", sha256=sha, source_kind="maintainer_official_sample")]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"status": "frozen", "samples": items}), encoding="utf-8")
    _, corpus = load_corpus(manifest, live_only=True)
    assert list(corpus) == ["ja1"]
    results = tmp_path / "asr.json"
    results.write_text(json.dumps(report("行く", sha256=sha)), encoding="utf-8")
    evaluated = evaluate(manifest, [results])
    assert evaluated["status"] == "not_scored"
    assert evaluated["source_kinds"] == {"saved_application_audio": 1, "maintainer_official_sample": 1}
    assert evaluated["models"][0]["micro_cer"] is None
    audio.write_bytes(b"changed input")
    with pytest.raises(ValueError, match="Frozen audio SHA"):
        load_corpus(manifest)


def test_writing_never_overwrites_existing_review(tmp_path):
    path = tmp_path / "review.json"
    write_new(path, {"status": "not_scored"})
    with pytest.raises(FileExistsError):
        write_new(path, {"status": "scored"})
    assert json.loads(path.read_text(encoding="utf-8"))["status"] == "not_scored"


def test_offline_packet_embeds_audio_and_starts_with_hidden_hypotheses(tmp_path):
    from build_asr_review import build

    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"fixed WAV input")
    sha = hashlib.sha256(audio.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"status": "frozen", "samples": [sample(audio_path="clip.wav", sha256=sha)]}), encoding="utf-8")
    model = tmp_path / "model.json"
    model.write_text(json.dumps(report("</script><script>alert('model text')</script>", sha256=sha)), encoding="utf-8")
    packet = build(manifest, [model])
    assert 'src="data:audio/wav;base64,' in packet
    assert '<details><summary>' in packet
    assert '<details open' not in packet
    assert '&lt;/script&gt;' in packet
    assert "connect-src 'none'" in packet
    assert '"reference_text": null' in packet
    assert 'row.asr_hypotheses_viewed' in packet


def test_packet_rejects_model_results_for_another_waveform(tmp_path):
    from build_asr_review import build

    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"fixed WAV input")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"status": "frozen", "samples": [sample(audio_path="clip.wav", sha256=hashlib.sha256(audio.read_bytes()).hexdigest())]}), encoding="utf-8")
    model = tmp_path / "model.json"
    model.write_text(json.dumps(report("output for wrong clip")), encoding="utf-8")
    with pytest.raises(ValueError, match="ASR report audio SHA"):
        build(manifest, [model])
