"""Build-time only: download pinned official weights and convert M2M100 to INT8."""
from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
CACHE = ROOT / "model-cache"
ASR_REPO = "csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
ASR_REV = "2365baeacb507f821a0c8120fcee3d484dba7a07"
MT_REPO = "facebook/m2m100_418M"
MT_REV = "55c2e61bbf05dfb8d7abccdc3fae6fc8512fd636"


def fetch_hf(repo: str, revision: str, filename: str, destination: Path) -> Path:
    from huggingface_hub import hf_hub_download
    if destination.exists():
        return destination
    print(f"Downloading {repo}/{filename}", flush=True)
    cached = hf_hub_download(repo, filename, revision=revision, cache_dir=str(CACHE / "hf"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cached, destination)
    return destination


def main() -> None:
    for name in ("model.int8.onnx", "tokens.txt", "LICENSE", "README.md"):
        fetch_hf(ASR_REPO, ASR_REV, name, MODELS / "asr" / name)
    for lang in ("zh", "ja"):
        sample = fetch_hf(ASR_REPO, ASR_REV, f"test_wavs/{lang}.wav", ROOT / "artifacts" / "test_audio" / f"{lang}.wav")
        (MODELS / "self_test").mkdir(parents=True, exist_ok=True)
        shutil.copy2(sample, MODELS / "self_test" / f"{lang}.wav")

    vad_path = MODELS / "vad" / "silero_vad.onnx"
    if not vad_path.exists():
        vad_path.parent.mkdir(parents=True, exist_ok=True)
        print("Downloading official Silero VAD", flush=True)
        temp = vad_path.with_suffix(".download")
        urllib.request.urlretrieve("https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx", temp)
        temp.replace(vad_path)

    mt_source = CACHE / "m2m100-source"
    for name in ("config.json", "pytorch_model.bin", "vocab.json", "sentencepiece.bpe.model", "tokenizer_config.json", "special_tokens_map.json", "README.md"):
        fetch_hf(MT_REPO, MT_REV, name, mt_source / name)
    output = MODELS / "translation"
    if not (output / "model.bin").exists():
        # SentencePiece's Windows filename API cannot open some Unicode paths.
        # Loading serialized bytes keeps the build valid in a Chinese workspace.
        import sentencepiece
        import transformers.models.m2m_100.tokenization_m2m_100 as tokenizer_module
        def load_spm_unicode(path, sp_model_kwargs=None):
            processor = sentencepiece.SentencePieceProcessor(**(sp_model_kwargs or {}))
            processor.LoadFromSerializedProto(Path(path).read_bytes())
            return processor
        tokenizer_module.load_spm = load_spm_unicode
        from ctranslate2.converters import TransformersConverter
        print("Converting M2M100 to CPU INT8 (one-time build step)", flush=True)
        TransformersConverter(str(mt_source)).convert(str(output), quantization="int8")
    for name in ("vocab.json", "sentencepiece.bpe.model", "tokenizer_config.json", "special_tokens_map.json", "README.md"):
        shutil.copy2(mt_source / name, output / name)
    license_path = output / "LICENSE"
    if not license_path.exists():
        urllib.request.urlretrieve("https://raw.githubusercontent.com/facebookresearch/fairseq/main/LICENSE", license_path)

    for relative, url in (
        ("asr/MODEL_LICENSE", "https://raw.githubusercontent.com/modelscope/FunASR/main/MODEL_LICENSE"),
        ("vad/LICENSE", "https://raw.githubusercontent.com/snakers4/silero-vad/master/LICENSE"),
    ):
        license_path = MODELS / relative
        if not license_path.exists():
            urllib.request.urlretrieve(url, license_path)

    manifest = {"sources": {"asr": {"repo": ASR_REPO, "revision": ASR_REV}, "translation": {"repo": MT_REPO, "revision": MT_REV}}, "files": {}}
    for path in sorted(MODELS.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            manifest["files"][path.relative_to(MODELS).as_posix()] = {"size": path.stat().st_size, "sha256": digest}
    (MODELS / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Models ready. Runtime requires no network.", flush=True)


if __name__ == "__main__":
    main()
