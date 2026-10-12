"""Optional, pinned ONNX enhancement weights outside the application package."""
from __future__ import annotations

from pathlib import Path
import threading

from .models_download import ModelDownloader, ModelEntry, ModelFile, Progress, VerificationResult
from .paths import models_root

PROCESSING_MODES = ("off", "gtcrn", "uvr")
_OFFICIAL_RELEASES = "https://github.com/k2-fsa/sherpa-onnx/releases/download/"


def processing_model_entry(mode: str) -> ModelEntry:
    if mode == "gtcrn":
        return ModelEntry(
            "audio-gtcrn", "语音增强 · GTCRN",
            (ModelFile("audio_processing/gtcrn_simple.onnx",
                       _OFFICIAL_RELEASES + "speech-enhancement-models/gtcrn_simple.onnx",
                       535638, "e77603ac0c23dac3227dd2d7135b3a585cbee2679048aecfa886657d3ae1b534"),),
            "https://k2-fsa.github.io/sherpa/onnx/speech-enhancement/index.html",
            "本地 CPU 语音增强；减弱噪声与部分背景干扰。保留原音混合，无法区分主播和游戏角色。",
            model_kind="audio_processing", recommendation="默认 · 低延迟",
            min_memory_description="CPU 运行；权重约 0.51 MiB",
            # The older official asset (236067657) has no publisher digest.
            # This SHA-256 was computed from that official release download.
            recommendation_basis="官方 ONNX 模型；日语识别质量仍需实测，不保证去除伴奏。")
    if mode == "uvr":
        return ModelEntry(
            "audio-uvr-9482", "伴奏分离 · UVR MDXNET 9482",
            (ModelFile("audio_processing/UVR_MDXNET_9482.onnx",
                       _OFFICIAL_RELEASES + "source-separation-models/UVR_MDXNET_9482.onnx",
                       29704738, "9d78f8566fa8198065214ab628be1de966a500c57786695aa4b13e2b27a7727d"),),
            "https://k2-fsa.github.io/sherpa/onnx/source-separation/models.html#uvr",
            "本地分离人声与伴奏；CPU 处理较慢。唱歌与游戏人物仍属于人声，不能只留下主播。",
            model_kind="audio_processing", recommendation="可选 · 处理较慢",
            min_memory_description="CPU 运行；权重约 28.3 MiB",
            recommendation_basis="采用官方示例中的较小 UVR 模型；短句也有最小分块处理成本。")
    raise ValueError("该模式没有需要下载的语音处理模型")


def processing_model_path(root: Path | None, mode: str) -> Path:
    entry = processing_model_entry(mode)
    return (Path(root) if root is not None else models_root()) / entry.files[0].relative_path


def verify_processing_model(root: Path | None, mode: str, *, progress: Progress | None = None,
                            cancel: threading.Event | None = None) -> VerificationResult:
    if mode == "off":
        return VerificationResult(True, "off", "已关闭；不需要模型")
    return ModelDownloader(root or models_root()).verify_entry(
        processing_model_entry(mode), progress=progress, cancel=cancel)


def download_processing_model(root: Path | None, mode: str, *, progress: Progress | None = None,
                              cancel: threading.Event | None = None) -> Path:
    selected = Path(root) if root is not None else models_root()
    downloader = ModelDownloader(selected)
    downloader.download_entry(processing_model_entry(mode), progress=progress, cancel=cancel)
    return processing_model_path(selected, mode)
