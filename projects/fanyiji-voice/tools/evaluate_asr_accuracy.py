"""Score saved ASR reports only against independently verified audio transcripts.

No model inference, audio capture, network access, or production settings writes.
Unlabelled audio remains not_scored; model agreement is never ground truth.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import unicodedata


REFERENCE_KINDS = frozenset(("human_transcription", "human_verified_subtitle"))
NORMALIZATION = "NFKC, Unicode casefold, remove whitespace and Unicode punctuation; retain kana, kanji, numerals and symbols"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def normalize(text: str) -> str:
    return "".join(character for character in unicodedata.normalize("NFKC", text).casefold()
                   if not character.isspace() and not unicodedata.category(character).startswith("P"))


def character_edits(reference: str, hypothesis: str) -> dict[str, int]:
    """Minimum Levenshtein edits with deterministic substitution-first tie breaks."""
    # Each cell stores (distance, substitutions, deletions, insertions). Keeping
    # one row bounds memory while preserving independently inspectable counts.
    previous = [(index, 0, 0, index) for index in range(len(hypothesis) + 1)]
    for ref_index, ref_character in enumerate(reference, 1):
        current = [(ref_index, 0, ref_index, 0)]
        for hyp_index, hyp_character in enumerate(hypothesis, 1):
            if ref_character == hyp_character:
                current.append(previous[hyp_index - 1])
                continue
            distance, substitutions, deletions, insertions = previous[hyp_index - 1]
            substitute = (distance + 1, substitutions + 1, deletions, insertions)
            distance, substitutions, deletions, insertions = previous[hyp_index]
            delete = (distance + 1, substitutions, deletions + 1, insertions)
            distance, substitutions, deletions, insertions = current[-1]
            insert = (distance + 1, substitutions, deletions, insertions + 1)
            current.append(min((substitute, delete, insert), key=lambda item: item[0]))
        previous = current
    distance, substitutions, deletions, insertions = previous[-1]
    return {"edits": distance, "substitutions": substitutions, "deletions": deletions, "insertions": insertions}


def indexed_rows(document: dict, label: str) -> dict[str, dict]:
    rows = document.get("samples")
    if not isinstance(rows, list):
        raise ValueError(f"{label} requires a samples list")
    indexed = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"] or row["id"] in indexed:
            raise ValueError(f"{label} sample IDs must be unique nonempty strings")
        indexed[row["id"]] = row
    return indexed


def load_corpus(path: Path, language: str = "ja", live_only: bool = False) -> tuple[str, dict[str, dict]]:
    document = read_json(path)
    if document.get("status") != "frozen":
        raise ValueError("Accuracy comparison requires a frozen corpus")
    selected = {}
    for identifier, row in indexed_rows(document, "Corpus").items():
        if row.get("source_lang") != language or row.get("synthetic") is True:
            continue
        if live_only and row.get("source_kind") != "saved_application_audio":
            continue
        audio_path = Path(row["audio_path"])
        if not audio_path.is_absolute():
            audio_path = path.parent / audio_path
        if digest(audio_path) != row.get("sha256"):
            raise ValueError(f"Frozen audio SHA-256 mismatch: {identifier}")
        selected[identifier] = {**row, "audio_path": str(audio_path.resolve())}
    if not selected:
        raise ValueError("No matching nonsynthetic corpus samples")
    return digest(path), selected


def reference_template(manifest_sha256: str, corpus: dict[str, dict]) -> dict:
    return {"schema_version": 1, "corpus_sha256": manifest_sha256,
            "instructions": "Listen to the original audio or verify independent subtitles against it. Keep ASR outputs hidden until transcription is complete. Do not enter an ASR/LLM output as human truth. Mark uncertain or unreviewed clips instead of guessing.",
            "allowed_reference_kinds": sorted(REFERENCE_KINDS),
            "samples": [{"id": identifier, "audio_sha256": row["sha256"], "audio_path": row["audio_path"],
                         "source_lang": row["source_lang"], "reference_text": None, "reference_kind": None,
                         "verified_against_audio": False, "annotator": None, "reference_source": None,
                         "uncertain": False, "uncertain_ranges": []} for identifier, row in corpus.items()]}


def load_references(document: dict | None, manifest_sha256: str, corpus: dict[str, dict]) -> tuple[dict[str, dict], Counter]:
    counts = Counter()
    if document is None:
        counts["missing_reference"] = len(corpus)
        return {}, counts
    if document.get("corpus_sha256") != manifest_sha256:
        raise ValueError("Reference corpus SHA-256 mismatch")
    rows = indexed_rows(document, "Reference")
    if set(rows) - set(corpus):
        raise ValueError("Reference contains IDs outside the selected corpus")
    verified = {}
    for identifier, audio in corpus.items():
        row = rows.get(identifier)
        if row is None:
            counts["missing_reference"] += 1
            continue
        if row.get("audio_sha256") != audio["sha256"] or row.get("source_lang") != audio["source_lang"]:
            raise ValueError(f"Reference audio identity/language mismatch: {identifier}")
        kind = row.get("reference_kind")
        if kind is not None and kind not in REFERENCE_KINDS:
            raise ValueError(f"Reference must be independent verified transcription/subtitle, not a model output: {identifier}")
        if row.get("verified_against_audio") is not True or row.get("reference_text") is None:
            counts["unreviewed_reference"] += 1
            continue
        if kind not in REFERENCE_KINDS or not isinstance(row["reference_text"], str):
            raise ValueError(f"Verified reference requires text and declared reference kind: {identifier}")
        if not all(isinstance(row.get(field), str) and row[field].strip() for field in ("annotator", "reference_source")):
            raise ValueError(f"Verified reference requires annotator and reference source: {identifier}")
        if row.get("uncertain") is True or row.get("uncertain_ranges"):
            counts["uncertain_reference"] += 1
            continue
        if row["reference_text"] and not normalize(row["reference_text"]):
            raise ValueError(f"Punctuation-only reference is not a verified no-speech label: {identifier}")
        verified[identifier] = row
        counts["verified_reference"] += 1
    return verified, counts


def score_report(document: dict, corpus: dict[str, dict], references: dict[str, dict]) -> dict:
    hypotheses = indexed_rows(document, "ASR report")
    for identifier, row in hypotheses.items():
        if identifier not in corpus:
            continue  # Bilingual reports may contain samples outside this scope.
        if row.get("sha256") != corpus[identifier]["sha256"]:
            raise ValueError(f"ASR report audio SHA-256 mismatch: {identifier}")
        if row.get("source_lang") != corpus[identifier]["source_lang"]:
            raise ValueError(f"ASR report source language mismatch: {identifier}")
    samples = []
    total = Counter()
    no_speech_samples = no_speech_false_positives = no_speech_inserted_characters = 0
    for identifier, reference in references.items():
        row = hypotheses.get(identifier)
        text = row.get("production_text", row.get("text")) if row else None
        if text is not None and not isinstance(text, str):
            raise ValueError(f"ASR hypothesis must be a string or null: {identifier}")
        # A missing, failed or empty recognition remains an omission; excluding
        # it would make a model look artificially accurate on successful calls.
        normalized_reference = normalize(reference["reference_text"])
        normalized_hypothesis = normalize(text or "")
        counts = character_edits(normalized_reference, normalized_hypothesis)
        item = {"id": identifier, "reference_text": reference["reference_text"], "hypothesis": text,
                "normalized_reference": normalized_reference, "normalized_hypothesis": normalized_hypothesis,
                "hypothesis_present": row is not None, "reference_characters": len(normalized_reference), **counts}
        if normalized_reference:
            item["cer"] = counts["edits"] / len(normalized_reference)
            total.update(counts)
            total["reference_characters"] += len(normalized_reference)
            total["speech_samples"] += 1
        else:
            item["cer"] = None
            item["scope"] = "verified_no_speech_false_positive_check"
            no_speech_samples += 1
            no_speech_false_positives += bool(normalized_hypothesis)
            no_speech_inserted_characters += len(normalized_hypothesis)
        samples.append(item)
    denominator = total["reference_characters"]
    return {"model_id": document.get("model_id", document.get("model", "unspecified")),
            "status": "scored" if references else "not_scored", "accuracy_percentage": None,
            "corpus_sample_count": len(corpus), "verified_reference_count": len(references),
            "unscored_sample_count": len(corpus) - len(references),
            "missing_hypothesis_ids": [identifier for identifier in references if identifier not in hypotheses],
            "micro_cer": total["edits"] / denominator if denominator else None,
            "speech_edit_totals": dict(total),
            "verified_no_speech": {"sample_count": no_speech_samples, "false_positive_samples": no_speech_false_positives,
                                   "inserted_characters": no_speech_inserted_characters}, "samples": samples}


def evaluate(corpus_path: Path, result_paths: list[Path], reference_path: Path | None = None,
             *, language: str = "ja", live_only: bool = False) -> dict:
    manifest_sha256, corpus = load_corpus(corpus_path, language, live_only)
    references, reference_counts = load_references(read_json(reference_path) if reference_path else None, manifest_sha256, corpus)
    models = []
    for path in result_paths:
        model = score_report(read_json(path), corpus, references)
        model.update(report_path=str(path.resolve()), report_sha256=digest(path))
        models.append(model)
    return {"schema_version": 1, "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
            "status": "scored" if references else "not_scored", "corpus_path": str(corpus_path.resolve()),
            "corpus_sha256": manifest_sha256, "source_lang": language, "live_only": live_only,
            "source_kinds": dict(Counter(row.get("source_kind", "unspecified") for row in corpus.values())),
            "reference_path": str(reference_path.resolve()) if reference_path else None,
            "reference_sha256": digest(reference_path) if reference_path else None,
            "reference_counts": dict(reference_counts), "normalization": NORMALIZATION,
            "reference_provenance_is_declared_not_independently_audited": True,
            "interpretation": "CER is orthographic error against supplied verified references, not semantic accuracy. Kana/kanji and numeral spellings can differ without changing meaning. Partial reference coverage must not be generalized to the full livestream. Model agreement, Codex text corrections and production outputs are not reference truth. No accuracy ranking is produced without references.",
            "models": models, "audio_capture_performed": False, "model_inference_performed": False,
            "network_operations_performed": False, "production_configuration_modified": False}


def write_new(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Preserve any existing review/results; accidental reuse must fail loudly.
    with path.open("x", encoding="utf-8") as stream:
        json.dump(document, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--result", type=Path, action="append", default=[])
    parser.add_argument("--references", type=Path)
    parser.add_argument("--language", default="ja")
    parser.add_argument("--live-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-template", type=Path)
    args = parser.parse_args()
    report = evaluate(args.corpus, args.result, args.references, language=args.language, live_only=args.live_only)
    if args.reference_template:
        manifest_sha256, corpus = load_corpus(args.corpus, args.language, args.live_only)
        write_new(args.reference_template, reference_template(manifest_sha256, corpus))
    write_new(args.output, report)
    print(json.dumps({"status": report["status"], "reference_counts": report["reference_counts"],
                      "output": str(args.output.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
