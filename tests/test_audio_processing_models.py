from pathlib import Path
import threading

import pytest

from fanyiji.audio_processing_models import (download_processing_model, processing_model_entry,
                                             processing_model_path, verify_processing_model)
from fanyiji.models_download import VerificationResult


@pytest.mark.parametrize("mode,filename,size,digest", [
    ("gtcrn", "gtcrn_simple.onnx", 535638,
     "e77603ac0c23dac3227dd2d7135b3a585cbee2679048aecfa886657d3ae1b534"),
    ("uvr", "UVR_MDXNET_9482.onnx", 29704738,
     "9d78f8566fa8198065214ab628be1de966a500c57786695aa4b13e2b27a7727d"),
])
def test_manifest_pins_official_external_weight(mode, filename, size, digest):
    entry = processing_model_entry(mode)
    file, = entry.files
    assert file.relative_path == "audio_processing/" + filename
    assert file.size == size and file.sha256 == digest
    assert file.url.startswith("https://github.com/k2-fsa/sherpa-onnx/releases/download/")
    assert entry.required is False and entry.model_kind == "audio_processing"
    assert "游戏" in entry.description
    assert processing_model_path(Path("external"), mode) == Path("external/audio_processing") / filename


@pytest.mark.parametrize("mode", ["gtcrn", "uvr"])
def test_download_and_verification_forward_cancel_progress(tmp_path, monkeypatch, mode):
    calls = []
    class Downloader:
        def __init__(self, root):
            assert root == tmp_path
        def verify_entry(self, entry, *, progress, cancel):
            calls.append(("verify", entry, progress, cancel))
            return VerificationResult(True, "ready", "verified")
        def download_entry(self, entry, *, progress, cancel):
            calls.append(("download", entry, progress, cancel))
    monkeypatch.setattr("fanyiji.audio_processing_models.ModelDownloader", Downloader)
    progress, cancel = lambda *args: None, threading.Event()
    assert verify_processing_model(tmp_path, mode, progress=progress, cancel=cancel).ready
    assert download_processing_model(tmp_path, mode, progress=progress, cancel=cancel) == processing_model_path(tmp_path, mode)
    assert [c[0] for c in calls] == ["verify", "download"]
    assert all(c[1:] == (processing_model_entry(mode), progress, cancel) for c in calls)


def test_disabled_mode_never_constructs_downloader(tmp_path, monkeypatch):
    def forbidden(*args):
        pytest.fail("off must not load, download or inspect a model")
    monkeypatch.setattr("fanyiji.audio_processing_models.ModelDownloader", forbidden)
    assert verify_processing_model(tmp_path, "off").ready
    for mode in ("off", "other"):
        with pytest.raises(ValueError):
            processing_model_entry(mode)


def test_invalid_weight_never_certified(tmp_path):
    target = processing_model_path(tmp_path, "gtcrn")
    target.parent.mkdir()
    target.write_bytes(b"not an onnx model")
    result = verify_processing_model(tmp_path, "gtcrn")
    assert result.ready is False and result.state == "invalid"
    assert target.read_bytes() == b"not an onnx model"
