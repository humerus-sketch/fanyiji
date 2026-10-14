"""Fetch pinned AudioTagging assets into an isolated local research cache.

No audio upload, dependency upgrade, model inference, or production edits.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

from prepare_asr_candidates import download_file, verify_candidate

ROOT = Path(__file__).resolve().parents[1]
LABEL_GIT_BLOB = "3a2767e81114adecde59992cf6607f31c1862f4c"
CANDIDATES = {
    "ced-tiny-int8": {
        "id": "ced-tiny-int8", "kind": "ced", "directory": "ced-tiny-int8",
        "repo": "k2-fsa/sherpa-onnx-ced-tiny-audio-tagging-2024-04-19",
        "revision": "efe3a5cfad74e56598d855d055006903adc00e38",
        "license_status": "Unresolved: upstream CED repository GPL-3.0, upstream HF weights Apache-2.0; converted card has no license",
        "files": {"model.int8.onnx": (6133417, "73aa22e783115f6a4ae169b36089907e07d0fd44795595eddea9c1bfc74cc945")},
        "readme_blob": "20aeff9427a3a65f3342daafb303eae8b667c87b", "readme_size": 89,
    },
    "zipformer-small-int8": {
        "id": "zipformer-small-int8", "kind": "zipformer", "directory": "zipformer-small-int8",
        "repo": "k2-fsa/sherpa-onnx-zipformer-small-audio-tagging-2024-04-15",
        "revision": "c24f9d0fdf7d0b8c0c4f9733aa0cac51fda10c95",
        "license_status": "Apache-2.0, fixed converted model README declares license",
        "files": {"model.int8.onnx": (27038066, "69304b8a1b96bbe6b7d16c24079f0732c65faf3568a14cb82a8238c709afe76c")},
        "readme_blob": "58ca975c7ba64feea9f37ce986888b24611d4ec6", "readme_size": 243,
    },
}


def fetch_git_file(candidate: dict, directory: Path, filename: str, size: int, blob: str) -> dict:
    url = f"https://huggingface.co/{candidate['repo']}/resolve/{candidate['revision']}/{filename}"
    target = directory / filename
    if target.is_file():
        data = target.read_bytes()
    else:
        with urlopen(Request(url, headers={"User-Agent": "LocalSpeechFilterResearch/1"}), timeout=30) as response:
            data = response.read(size + 1)
    actual_blob = hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()
    if len(data) != size or actual_blob != blob:
        raise RuntimeError(f"Pinned Git blob mismatch for {filename}")
    if not target.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return {"source": url, "size": size, "git_blob_sha1": blob,
            "sha256": hashlib.sha256(data).hexdigest(), "path": str(target.resolve())}


def prepare(cache: Path) -> None:
    reports = {}
    for key, candidate in CANDIDATES.items():
        directory = cache / candidate["directory"]
        directory.mkdir(parents=True, exist_ok=True)
        for filename in candidate["files"]:
            download_file(candidate, directory, filename)
        verified = verify_candidate(candidate, directory)
        verified["class_labels_indices.csv"] = fetch_git_file(
            candidate, directory, "class_labels_indices.csv", 14675, LABEL_GIT_BLOB)
        verified["README.md"] = fetch_git_file(
            candidate, directory, "README.md", candidate["readme_size"], candidate["readme_blob"])
        # The fixed official class-3 demo contains singing; it is a public positive control.
        verified["test_wavs/3.wav"] = fetch_git_file(
            candidate, directory, "test_wavs/3.wav", 320078, "ed79deabd8529231917a51a590c00d0aae88f607")
        source = {"candidate": candidate, "verified_files": verified,
                  "documentation": "https://k2-fsa.github.io/sherpa/onnx/audio-tagging/pretrained_models.html",
                  "license_context": "https://github.com/k2-fsa/sherpa-onnx/issues/2674#issuecomment-3393952839",
                  "prepared_at": datetime.now().astimezone().isoformat(), "recording_uploaded": False}
        (directory / "SOURCE.json").write_text(json.dumps(source, ensure_ascii=False, indent=2), encoding="utf-8")
        reports[key] = source
    (cache / "verified-assets.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "verified", "candidates": list(reports), "cache": str(cache)}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=ROOT / "model-cache" / "speech-filter")
    prepare(parser.parse_args().cache.resolve())
