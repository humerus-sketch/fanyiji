"""Prepare a pinned, advisory community lexicon outside the application package."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unicodedata
from urllib.parse import quote
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fanyiji.paths import data_root

REPOSITORY = "https://github.com/konsheng/Sensitive-lexicon"
REVISION = "d967c30b053fa40b06c5a0dddf0be493f2dfae46"
FILES = {
    "广告类型.txt": "advertising",
    "暴恐词库.txt": "violence",
    "涉枪涉爆.txt": "weapons",
    "色情类型.txt": "adult",
    "色情词库.txt": "adult",
    "其他词库.txt": "other",
}


def _download(path: str) -> bytes:
    url = f"https://raw.githubusercontent.com/konsheng/Sensitive-lexicon/{REVISION}/{quote(path)}"
    request = Request(url, headers={"User-Agent": "FanyiJi-local-lexicon/1.0"})
    with urlopen(request, timeout=30) as response:
        result = response.read(2_000_001)
    if len(result) > 2_000_000:
        raise ValueError("Source file is unexpectedly large")
    return result


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                dir=path.parent, prefix=".wordlist-", suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
            json.dump(value, output, ensure_ascii=False, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def prepare(output: Path, evidence: Path) -> dict:
    if output.exists():
        raise FileExistsError("Local lexicon already exists; preserving its terms and allowlist")
    license_bytes = _download("LICENSE")
    license_text = license_bytes.decode("utf-8-sig")
    if not license_text.startswith("MIT License"):
        raise ValueError("Pinned source license could not be verified")
    raw_directory = evidence / "sources"
    raw_directory.mkdir(parents=True, exist_ok=True)
    (raw_directory / "LICENSE.txt").write_bytes(license_bytes)
    entries, manifest = {}, []
    excluded_short = 0
    for name, category in FILES.items():
        relative = "Vocabulary/" + name
        source = _download(relative)
        (raw_directory / name).write_bytes(source)
        imported = set()
        for line in source.decode("utf-8-sig").splitlines():
            term = line.strip()
            if not term or term.startswith(("#", "//")):
                continue
            # Single characters are too ambiguous for ordinary speech and game subtitles.
            if len(term) < 2:
                excluded_short += 1
                continue
            if len(term.encode("utf-16-le")) // 2 > 80 or any(
                    unicodedata.category(c) in {"Cc", "Cf", "Cs"} for c in term):
                continue
            imported.add(term)
            entries.setdefault(term, {"term": term, "category": category,
                                     "source_id": "konsheng-community"})
        manifest.append({"path": relative, "sha256": hashlib.sha256(source).hexdigest(),
                         "bytes": len(source), "valid_unique_terms": len(imported)})
    source_info = {"id": "konsheng-community", "name": "Sensitive-lexicon 社区通用参考词库",
                   "url": REPOSITORY, "revision": REVISION,
                   "license": "MIT", "license_text": license_text}
    lexicon = {"version": 1, "sources": [source_info],
               "entries": sorted(entries.values(), key=lambda row: row["term"]), "allowlist": []}
    _atomic_json(output, lexicon)
    report = {"prepared_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
              "local_path": str(output.resolve()), "repository": REPOSITORY,
              "revision": REVISION, "license": "MIT", "official_bilibili_list": False,
              "automatic_replacement": False, "entry_count": len(entries),
              "excluded_single_character_lines": excluded_short,
              "sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "files": manifest}
    _atomic_json(evidence / "wordlist-manifest.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=data_root() / "danmaku-wordlist.json")
    parser.add_argument("--evidence", type=Path, default=ROOT / "artifacts/wordlist-research-20261011")
    arguments = parser.parse_args()
    print(json.dumps(prepare(arguments.output, arguments.evidence), ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
