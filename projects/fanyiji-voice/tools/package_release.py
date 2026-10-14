"""Package a verified portable build and exactly the committed project source."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fanyiji.identity import APP_ID, APP_NAME

PRIVATE = {"cookies", "login data", "auth.json", "credentials.json", "secrets.json", "settings.json",
           "local-features.json", "browser-preferences.json", "chrome-caption-settings.json",
           "danmaku-queue.sqlite3", "danmaku-wordlist.json", "danmaku-content-filter.json",
           "delivery.jsonl", "model.bin", "local-install-manifest.json"}
WEIGHTS = {".gguf", ".onnx", ".safetensors", ".pt", ".pth", ".key", ".pfx", ".p12"}


def validate_name(name: str):
    path = Path(name)
    assert path.name.casefold() not in PRIVATE and path.suffix.casefold() not in WEIGHTS, name
    assert not path.name.startswith(".env"), name


def checked_archive(path: Path):
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None, "ZIP CRC failed"
        for item in archive.infolist():
            validate_name(item.filename)
            assert ".git" not in Path(item.filename).parts
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"filename": path.name, "size": path.stat().st_size, "sha256": digest}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dist")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    import tomllib
    version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    folder = ROOT / "dist" / APP_NAME
    build = json.loads((folder / "build-info.json").read_text(encoding="utf-8"))
    assert build["application"] == APP_ID and build["edition"] == "voice" and build["models_bundled"] is False
    assert (folder / (APP_NAME + ".exe")).is_file()
    program = output / f"FanyiJiVoice-v{version}-windows-x64.zip"
    source = output / f"FanyiJiVoice-v{version}-source.zip"
    assert not program.exists() and not source.exists(), "Preserve existing release assets; choose an empty output."
    temporary = program.with_suffix(".zip.partial")
    assert not temporary.exists()
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for path in sorted(folder.rglob("*")):
            if not path.is_file():
                continue
            assert not path.is_symlink() and path.resolve().is_relative_to(folder.resolve())
            validate_name(path.name)
            archive.write(path, APP_NAME + "/" + path.relative_to(folder).as_posix())
    checked_archive(temporary)
    temporary.replace(program)
    git_root = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=ROOT, text=True, encoding="utf-8").strip()).resolve()
    relative = ROOT.resolve().relative_to(git_root).as_posix()
    tree = "HEAD" if relative == "." else "HEAD:" + relative
    subprocess.run(["git", "archive", "--format=zip", "--prefix=fanyiji-voice/", tree, "--output", str(source)],
                   cwd=git_root, check=True)
    results = [checked_archive(program), checked_archive(source)]
    (output / "FanyiJiVoice-SHA256SUMS.txt").write_text(
        "".join(item["sha256"] + "  " + item["filename"] + "\n" for item in results), encoding="utf-8")
    report = {"models_bundled": False, "private_data_bundled": False,
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, encoding="utf-8").strip(),
              "archives": results}
    (output / "FanyiJiVoice-release-manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
