"""User-named local model entries, separate from the immutable download catalog."""
from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path, PurePosixPath
import re
import threading
import unicodedata
import uuid

from .models_download import DownloadError, ModelCatalog, ModelEntry, ModelFile

_LOCK = threading.RLock()
_ID = re.compile(r"local-[0-9a-f]{32}\Z")
_LOCAL_VERIFICATION = "local_sha256"


def validate_display_name(value: str) -> str:
    if (not isinstance(value, str) or not 1 <= len(value.strip()) <= 80
            or any(unicodedata.category(ch).startswith("C") for ch in value)):
        raise DownloadError("模型显示名称必须为 1 到 80 个字符，不能包含控制字符。")
    return value.strip()


class ModelRegistry:
    """Read small metadata on the UI thread; verification remains in workers.

    Pinned imports reconstruct all hashes and URLs from their original catalog
    entry. A self-supplied digest never upgrades an unknown model to official.
    """

    def __init__(self, models_root: Path, *, catalog=None, asr_catalog=None):
        from .asr_catalog import AsrCatalog
        self.models_root = Path(models_root).resolve()
        self.catalog = catalog or ModelCatalog.default()
        original_asr = asr_catalog if asr_catalog is not None else AsrCatalog.default()
        supported = tuple(entry for entry in original_asr.entries if entry.model_kind == "sense_voice")
        if not supported:
            supported = AsrCatalog.default().entries
        preferred = original_asr.default_model_id
        self.asr_catalog = AsrCatalog(supported, preferred if any(entry.id == preferred for entry in supported)
                                      else supported[0].id)

    @property
    def path(self) -> Path:
        from .model_import import _plain_path
        return _plain_path(self.models_root / ".model-registry.json", allow_missing=True)

    def _templates(self, role: str) -> tuple[ModelEntry, ...]:
        return self.asr_catalog.entries if role == "asr" else self.catalog.entries

    @staticmethod
    def role(entry: ModelEntry) -> str:
        if entry.model_kind == "sense_voice":
            return "asr"
        if entry.model_kind in ("hy", "qwen") and not entry.required:
            return "translation"
        raise DownloadError("该模型用途尚未支持独立导入。")

    @staticmethod
    def _retired_asr(record: dict) -> bool:
        # Legacy metadata remains on disk, but cannot restore a retired backend
        # or prevent the independent translation registry from opening.
        return (isinstance(record, dict) and record.get("role") == "asr"
                and record.get("model_kind") in ("whisper", "qwen3_asr"))

    def named_entry(self, template: ModelEntry, display_name: str) -> ModelEntry:
        role = self.role(template)
        name, model_id = validate_display_name(display_name), "local-" + uuid.uuid4().hex
        prefix = PurePosixPath(role) / "imported" / model_id
        files = []
        for file in template.files:
            path = PurePosixPath(file.relative_path)
            if not path.is_relative_to(role):
                raise DownloadError("模型模板文件路径与用途不一致。")
            files.append(replace(file, relative_path=str(prefix / path.relative_to(role))))
        return replace(template, id=model_id, name=name, files=tuple(files), required=False,
                       source_model_id=template.id if template.verification == "pinned_catalog_sha256" else None,
                       recommendation=(template.recommendation if template.verification == "pinned_catalog_sha256" else "本地导入 · 未评测"),
                       recommendation_basis=template.recommendation_basis if template.verification == "pinned_catalog_sha256" else "",
                       description=("独立命名的本地模型；导入后需要手动选择启用。"
                                    if template.verification == "pinned_catalog_sha256" else
                                    "自定义本地 GGUF；仅校验结构与本地摘要，官方来源未经验证。"))

    def _record(self, entry: ModelEntry) -> dict:
        return {"id": entry.id, "display_name": entry.name, "role": self.role(entry),
                "source_model_id": entry.source_model_id, "model_kind": entry.model_kind,
                "architecture": entry.architecture, "verification": entry.verification,
                "files": [{"path": file.relative_path, "size": file.size, "sha256": file.sha256}
                          for file in entry.files]}

    def _entry(self, raw: dict) -> ModelEntry:
        if not isinstance(raw, dict) or set(raw) != {"id", "display_name", "role", "source_model_id",
                                                     "model_kind", "architecture", "verification", "files"}:
            raise DownloadError("本地模型注册表格式无效；已有记录保留。")
        model_id, role = raw["id"], raw["role"]
        if not isinstance(model_id, str) or not _ID.fullmatch(model_id) or role not in ("asr", "translation"):
            raise DownloadError("本地模型注册表 ID 或用途无效。")
        name = validate_display_name(raw["display_name"])
        prefix = PurePosixPath(role) / "imported" / model_id
        records = raw["files"]
        if not isinstance(records, list) or not 1 <= len(records) <= 32:
            raise DownloadError("本地模型注册表文件列表无效。")
        local_files = []
        for item in records:
            if not isinstance(item, dict) or set(item) != {"path", "size", "sha256"}:
                raise DownloadError("本地模型注册表文件信息无效。")
            if type(item["size"]) is not int or not isinstance(item["path"], str) or not isinstance(item["sha256"], str):
                raise DownloadError("本地模型注册表文件信息无效。")
            file = ModelFile(item["path"], None, item["size"], item["sha256"])
            if not PurePosixPath(file.relative_path).is_relative_to(prefix):
                raise DownloadError("本地模型注册表路径超出独立模型目录。")
            local_files.append(file)
        if len({file.relative_path.casefold() for file in local_files}) != len(local_files):
            raise DownloadError("本地模型注册表文件路径重复。")
        if raw["verification"] == "pinned_catalog_sha256":
            template = next((entry for entry in self._templates(role) if entry.id == raw["source_model_id"]), None)
            if template is None or template.required or template.model_kind != raw["model_kind"]:
                raise DownloadError("本地模型的预设来源未受支持。")
            expected = tuple(replace(file, relative_path=str(prefix / PurePosixPath(file.relative_path).relative_to(role)))
                             for file in template.files)
            if [(f.relative_path, f.size, f.sha256) for f in local_files] != [(f.relative_path, f.size, f.sha256) for f in expected]:
                raise DownloadError("本地模型记录与预设固定 SHA-256 不一致。")
            return replace(template, id=model_id, name=name, files=expected, source_model_id=template.id,
                           description="独立命名的本地模型；导入后需要手动选择启用。")
        if (raw["verification"] != _LOCAL_VERIFICATION or raw["source_model_id"] is not None
                or role != "translation" or raw["model_kind"] not in ("hy", "qwen") or len(local_files) != 1):
            raise DownloadError("该自定义模型格式尚未适配。")
        from .gguf_inspection import _ARCHITECTURES
        architecture = raw["architecture"]
        if (not isinstance(architecture, str) or _ARCHITECTURES.get(architecture) != raw["model_kind"]
                or PurePosixPath(local_files[0].relative_path).suffix != ".gguf"):
            raise DownloadError("该自定义模型格式尚未适配。")
        return ModelEntry(model_id, name, tuple(local_files), "", "自定义本地 GGUF；仅校验结构与本地摘要，官方来源未经验证。",
                          model_kind=raw["model_kind"], recommendation="本地导入 · 未评测",
                          verification=_LOCAL_VERIFICATION, architecture=architecture)

    def _records(self) -> list[dict]:
        with _LOCK:
            path = self.path
            if not path.exists():
                return []
            if not path.is_file() or path.stat().st_size > 2 * 1024 * 1024:
                raise DownloadError("本地模型注册表过大或无效；已有记录保留。")
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                if (not isinstance(raw, dict) or set(raw) != {"schema_version", "entries"}
                        or type(raw["schema_version"]) is not int or raw["schema_version"] != 1
                        or not isinstance(raw["entries"], list) or len(raw["entries"]) > 256):
                    raise ValueError()
                records = raw["entries"]
                if (any(not isinstance(record, dict) or not isinstance(record.get("id"), str)
                        or not _ID.fullmatch(record["id"]) for record in records)
                        or len({record["id"] for record in records}) != len(records)):
                    raise ValueError()
                return records
            except (ValueError, TypeError, KeyError, StopIteration) as exc:
                raise DownloadError("本地模型注册表格式无效；已有记录保留。") from exc

    def entries(self) -> tuple[ModelEntry, ...]:
        with _LOCK:
            return tuple(self._entry(record) for record in self._records()
                         if not self._retired_asr(record))

    def register(self, entry: ModelEntry) -> None:
        with _LOCK:
            validated = self._entry(self._record(entry))
            entries = self.entries()
            records = self._records()
            if len(records) >= 256:
                raise DownloadError("本地模型注册数量已达到 256 个上限。")
            if (any(current.id == validated.id or current.name.casefold() == validated.name.casefold() for current in entries)
                    or any(record["id"] == validated.id
                           or isinstance(record.get("display_name"), str)
                           and record["display_name"].casefold() == validated.name.casefold()
                           for record in records)):
                raise DownloadError("本地模型名称或 ID 已存在，请使用其他名称。")
            path = self.path
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
            raw = {"schema_version": 1, "entries": [*records, self._record(validated)]}
            with temporary.open("x", encoding="utf-8", newline="\n") as stream:
                json.dump(raw, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            # Registry publication never changes backend.json. Existing weights
            # and the current loaded model stay unchanged until explicit enable.
            os.replace(temporary, path)

    def catalogs(self):
        from .asr_catalog import AsrCatalog
        entries = self.entries()
        return (ModelCatalog((*self.catalog.entries, *(e for e in entries if self.role(e) == "translation")), self.catalog.default_model_id),
                AsrCatalog((*(e for e in self.asr_catalog.entries if e.model_kind == "sense_voice"),
                            *(e for e in entries if self.role(e) == "asr")), self.asr_catalog.default_model_id))

    def display_name_for(self, model_id: str) -> str | None:
        try:
            return next((entry.name for entry in self.entries() if entry.id == model_id), None)
        except (DownloadError, OSError):
            return None
