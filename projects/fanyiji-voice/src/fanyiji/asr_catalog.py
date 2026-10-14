"""Production recognition catalog: SenseVoice and the existing VAD dependencies."""
from __future__ import annotations

from dataclasses import replace
from pathlib import PurePosixPath

from .models_download import DownloadError, ModelCatalog, ModelEntry


class AsrCatalog(ModelCatalog):
    @classmethod
    def default(cls, base_catalog: ModelCatalog | None = None):
        base = (base_catalog or ModelCatalog.default()).required_entries[0]
        sensevoice = ModelEntry(
            "sensevoice-int8", "SenseVoice INT8",
            tuple(file for file in base.files if file.relative_path.startswith("asr/")),
            base.info_url, "唯一识别模型；结合 Silero VAD 分句，支持自动识别或固定通道语言。",
            model_kind="sense_voice", recommendation="当前识别方案",
            recommendation_basis="本机同音频对照显示较低识别延迟；无人工参考文本，真实识别准确率尚未评分。",
            min_memory_description="CPU运行；识别结果交由所选翻译源处理")
        return cls((sensevoice,), sensevoice.id)


ASRCatalog = AsrCatalog


def vad_dependencies(catalog: ModelCatalog, *, speech_only: bool = True) -> tuple[ModelEntry, ...]:
    """Keep Silero VAD and optional speaking/singing filtering independent of ASR."""
    result = []
    for entry in catalog.required_entries:
        if any(file.relative_path.startswith("speech_filter/") for file in entry.files):
            if speech_only:
                result.append(entry)
            continue
        files = tuple(file for file in entry.files if file.relative_path.startswith("vad/"))
        if files:
            result.append(replace(entry, id="base-vad", name="基础分句 · Silero VAD",
                                  files=files, description="SenseVoice 使用的静音分句依赖。"))
    return tuple(result)


def profile_for_entry(entry: ModelEntry) -> dict:
    if entry.model_kind != "sense_voice":
        raise DownloadError("当前仅支持 SenseVoice 识别模型。")
    files = [PurePosixPath(file.relative_path) for file in entry.files]
    if any(not path.is_relative_to(PurePosixPath("asr")) for path in files):
        raise DownloadError("识别模型必须放在asr目录内")
    profile = {"backend": "sense_voice", "model_id": entry.id}
    if entry.id.startswith("local-"):
        profile["display_name"] = entry.name
        for field, filename in (("model", "model.int8.onnx"), ("tokens", "tokens.txt")):
            matches = [path for path in files if path.name == filename]
            if len(matches) != 1:
                raise DownloadError("SenseVoice 模型包缺少必要文件。")
            profile[field] = str(matches[0].relative_to("asr"))
        return profile
    if {str(path.relative_to("asr")) for path in files} != {"model.int8.onnx", "tokens.txt"}:
        raise DownloadError("SenseVoice必须使用固定model.int8.onnx和tokens.txt")
    return profile
