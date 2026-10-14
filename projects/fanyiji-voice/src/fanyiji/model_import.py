"""Offline copies of pinned catalog assets, without model activation or downloads."""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import stat
import threading
from typing import BinaryIO
import uuid
import zipfile
import zlib

from .models_download import DownloadCancelled, DownloadError, ModelEntry, ModelFile, Progress


class ModelImportError(DownloadError):
    pass


class ModelImportCancelled(DownloadCancelled):
    pass


@dataclass(frozen=True)
class ImportResult:
    model_id: str
    imported_paths: tuple[str, ...]
    reused_paths: tuple[str, ...]
    provenance_path: Path
    source_kind: str
    verification: str = "pinned_catalog_sha256"


@dataclass(frozen=True)
class ImportOption:
    template: ModelEntry
    role: str
    suggested_name: str


@dataclass(frozen=True)
class ImportInspection:
    options: tuple[ImportOption, ...]
    source_kind: str


def _cancelled(cancel: threading.Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise ModelImportCancelled("导入已取消；原模型包与已有模型保留，未完成暂存可稍后手动清理。")


def _plain_path(value: Path, *, allow_missing: bool = False) -> Path:
    path = Path(os.path.abspath(value))
    for ancestor in (path, *path.parents):
        try:
            info = ancestor.lstat()
        except FileNotFoundError:
            if ancestor == path and not allow_missing:
                raise ModelImportError("选择的模型文件或目录不存在。")
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ModelImportError("模型路径含 symlink/junction，不能导入。")
        if stat.S_ISREG(info.st_mode) and info.st_nlink > 1:
            raise ModelImportError("模型文件含 hardlink，不能导入。")
    return path.resolve()


def _relative(name: str, *, directory: bool = False) -> str:
    if directory and name.endswith("/"):
        name = name[:-1]
    if not name or "\\" in name or ":" in name or name.startswith("/"):
        raise ModelImportError("模型包包含不安全路径。")
    reserved = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)),
                *(f"lpt{i}" for i in range(1, 10))}
    if any(part in ("", ".", "..") or part.endswith((" ", "."))
           or part.split(".")[0].casefold() in reserved
           or any(character in part for character in '<>"|?*\0') for part in name.split("/")):
        raise ModelImportError("模型包包含不安全路径。")
    return name


def _identity(info) -> tuple[int, int, int, int]:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


class ModelImporter:
    """Import only exact catalog revisions; self-provided checksums never certify assets.

    ZIP extras are audited, but only required model files are copied. Temporary
    copies remain on cancellation/failure. Publication uses exclusive creation
    so an existing target is never overwritten, including a racing download.
    """

    def __init__(self, models_root: Path, *, max_files: int = 4096,
                 max_total_bytes: int = 64 * 1024 ** 3, max_file_bytes: int = 32 * 1024 ** 3,
                 max_compression_ratio: float = 200, chunk_size: int = 1024 * 1024):
        if any(type(value) is not int or value < 1 for value in (max_files, max_total_bytes, max_file_bytes, chunk_size)):
            raise ValueError("导入文件数量、大小和块大小必须为正整数")
        if (isinstance(max_compression_ratio, bool) or not isinstance(max_compression_ratio, (int, float))
                or not 1 <= max_compression_ratio < float("inf")):
            raise ValueError("ZIP 压缩比例限制必须为有限正数")
        self.models_root = _plain_path(Path(models_root), allow_missing=True)
        self.max_files, self.max_total_bytes, self.max_file_bytes = max_files, max_total_bytes, max_file_bytes
        self.max_compression_ratio, self.chunk_size = max_compression_ratio, chunk_size

    def _target(self, file: ModelFile) -> Path:
        relative = _relative(file.relative_path)
        parts = relative.split("/")
        if parts[0] not in {"asr", "vad", "translation", "speech_filter"} or parts[-1].casefold() in {"backend.json", "settings.json"}:
            raise ModelImportError("只能导入目录中已支持的识别、分句、声音筛选或翻译模型。")
        target = _plain_path(self.models_root / relative, allow_missing=True)
        if not target.is_relative_to(self.models_root):
            raise ModelImportError("模型目标路径超出存储目录。")
        return target

    def _folder_files(self, folder: Path, cancel: threading.Event | None) -> dict[str, Path]:
        files, stack, count, total = {}, [folder], 0, 0
        folded = set()
        while stack:
            _cancelled(cancel)
            for path in stack.pop().iterdir():
                _cancelled(cancel)
                count += 1
                if count > self.max_files:
                    raise ModelImportError("模型包文件数量超过安全限制。")
                info = path.lstat()
                if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                    raise ModelImportError("模型包包含 symlink/junction，不能导入。")
                relative = _relative(path.relative_to(folder).as_posix())
                if relative.casefold() in folded:
                    raise ModelImportError("模型包路径重复或存在大小写冲突。")
                folded.add(relative.casefold())
                if stat.S_ISDIR(info.st_mode):
                    stack.append(path)
                elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                    total += info.st_size
                    if info.st_size > self.max_file_bytes or total > self.max_total_bytes:
                        raise ModelImportError("模型包大小超过安全限制。")
                    files[relative] = path
                else:
                    raise ModelImportError("模型包包含 hardlink 或非常规文件，不能导入。")
        return files

    def _zip_files(self, archive: zipfile.ZipFile, cancel: threading.Event | None) -> dict[str, zipfile.ZipInfo]:
        members = archive.infolist()
        if len(members) > self.max_files:
            raise ModelImportError("模型包文件数量超过安全限制。")
        files, folded, total = {}, set(), 0
        for member in members:
            _cancelled(cancel)
            if member.orig_filename != member.filename or "\0" in member.orig_filename:
                raise ModelImportError("模型包包含不安全路径。")
            name = _relative(member.filename, directory=member.is_dir())
            if name.casefold() in folded:
                raise ModelImportError("模型包路径重复或存在大小写冲突。")
            folded.add(name.casefold())
            mode = member.external_attr >> 16
            kind = stat.S_IFMT(mode)
            if (member.flag_bits & 1 or member.external_attr & 0x400
                    or kind not in (0, stat.S_IFREG, stat.S_IFDIR)
                    or (kind == stat.S_IFDIR and not member.is_dir())
                    or (member.is_dir() and kind not in (0, stat.S_IFDIR))):
                raise ModelImportError("ZIP 包含加密、链接或非常规文件，不能导入。")
            extra, offset = member.extra, 0
            while offset < len(extra):
                if offset + 4 > len(extra):
                    raise ModelImportError("ZIP 扩展信息无效。")
                field = int.from_bytes(extra[offset:offset + 2], "little")
                length = int.from_bytes(extra[offset + 2:offset + 4], "little")
                if offset + 4 + length > len(extra) or field in (0x000D, 0x756E):
                    raise ModelImportError("ZIP 包含不支持的链接扩展信息。")
                offset += 4 + length
            if member.is_dir():
                continue
            total += member.file_size
            if (member.file_size > self.max_file_bytes or total > self.max_total_bytes
                    or member.file_size > max(1, member.compress_size) * self.max_compression_ratio):
                raise ModelImportError("ZIP 大小或压缩比例超过安全限制。")
            files[name] = member
        file_names = {key.casefold() for key in files}
        for name in folded:
            components = name.split("/")
            if any("/".join(components[:index]) in file_names
                   for index in range(1, len(components))):
                raise ModelImportError("ZIP 文件与目录路径冲突。")
        return files

    def _zip_directory_limits(self, source: Path) -> None:
        # Inspect only the bounded EOCD/ZIP64 end records before ZipFile parses
        # and allocates every central-directory member. Counting afterwards is
        # too late for a malicious archive with millions of tiny members.
        with source.open("rb") as stream:
            end = zipfile._EndRecData(stream)
        if end is None:
            raise ModelImportError("ZIP 包格式无效。")
        if end[zipfile._ECD_ENTRIES_TOTAL] > self.max_files:
            raise ModelImportError("模型包文件数量超过安全限制。")
        if end[zipfile._ECD_SIZE] > min(32 * 1024 ** 2, self.max_files * 8192):
            raise ModelImportError("ZIP 目录信息超过安全限制。")
        if end[zipfile._ECD_DISK_NUMBER] or end[zipfile._ECD_DISK_START]:
            raise ModelImportError("不支持多卷 ZIP 模型包。")

    @staticmethod
    def _match(file: ModelFile, candidates: dict) -> object:
        expected = file.relative_path.casefold()
        exact = [source for name, source in candidates.items() if name.casefold() == expected
                 or name.casefold().endswith("/" + expected)]
        if len(exact) == 1:
            return exact[0]
        if len(exact) > 1:
            raise ModelImportError("模型包缺少必要文件或存在同名歧义；请选择对应模型的完整文件夹。")
        suffix = "/".join(expected.split("/")[-2:])
        close = [source for name, source in candidates.items() if name.casefold() == suffix
                 or name.casefold().endswith("/" + suffix)]
        if len(close) == 1:
            return close[0]
        basename = expected.rsplit("/", 1)[-1]
        matches = [source for name, source in candidates.items() if name.rsplit("/", 1)[-1].casefold() == basename]
        if len(matches) != 1:
            raise ModelImportError("模型包缺少必要文件或存在同名歧义；请选择对应模型的完整文件夹。")
        return matches[0]

    def _copy_verified(self, stream: BinaryIO, target: Path, file: ModelFile, cancel, progress,
                       *, phase: str, base: int = 0, total: int | None = None) -> None:
        digest, done = hashlib.sha256(), 0
        with target.open("xb") as output:
            while block := stream.read(min(self.chunk_size, file.size - done + 1)):
                _cancelled(cancel)
                done += len(block)
                if done > file.size:
                    raise ModelImportError("模型文件大小与选中目录版本不匹配。")
                output.write(block)
                digest.update(block)
                if progress:
                    progress(phase, base + done, total or file.size)
            output.flush()
            os.fsync(output.fileno())
        _cancelled(cancel)
        if done != file.size or digest.hexdigest() != file.sha256:
            raise ModelImportError("模型文件不符合目录固定大小或 SHA-256；不支持该版本或包已损坏。")

    def _verify_existing(self, target: Path, file: ModelFile, cancel, progress) -> None:
        before = target.stat()
        if before.st_size != file.size or not stat.S_ISREG(before.st_mode):
            raise ModelImportError("目标模型已存在且不匹配；已保留，请手动处理后重试。")
        digest, done = hashlib.sha256(), 0
        with target.open("rb") as stream:
            while block := stream.read(self.chunk_size):
                _cancelled(cancel)
                digest.update(block)
                done += len(block)
                if progress:
                    progress("verifying", done, file.size)
        _cancelled(cancel)
        if digest.hexdigest() != file.sha256 or _identity(before) != _identity(target.stat()):
            raise ModelImportError("目标模型已存在且不匹配；已保留，请手动处理后重试。")

    def provenance_path(self, entry: ModelEntry) -> Path:
        name = hashlib.sha256(entry.id.encode("utf-8")).hexdigest() + ".json"
        return _plain_path(self.models_root / ".model-imports" / name, allow_missing=True)

    def has_provenance(self, entry: ModelEntry) -> bool:
        try:
            path = self.provenance_path(entry)
            if not path.is_file() or path.stat().st_size > 65536:
                return False
            record = json.loads(path.read_text(encoding="utf-8"))
            expected = [{"path": file.relative_path, "size": file.size, "sha256": file.sha256} for file in entry.files]
            return (isinstance(record, dict) and record.get("model_id") == entry.id
                    and record.get("verification") == entry.verification and record.get("files") == expected)
        except (OSError, ValueError, ModelImportError):
            return False

    @contextmanager
    def _source_files(self, source: Path, cancel=None):
        source = _plain_path(Path(source))
        if source == self.models_root or source.is_relative_to(self.models_root) or self.models_root.is_relative_to(source):
            raise ModelImportError("模型来源与存储目录不能相同或相互嵌套。")
        archive = None
        before = _identity(source.stat())
        try:
            if source.is_dir():
                yield "folder", self._folder_files(source, cancel), None
            elif source.is_file() and source.suffix.casefold() == ".zip":
                if source.stat().st_size > self.max_total_bytes:
                    raise ModelImportError("模型包大小超过安全限制。")
                self._zip_directory_limits(source)
                archive = zipfile.ZipFile(source)
                yield "zip", self._zip_files(archive, cancel), archive
                if _identity(source.stat()) != before:
                    raise ModelImportError("模型来源在检查期间发生变化，请重新选择。")
            elif source.is_file():
                if source.suffix.casefold() in {".part", ".crdownload", ".download", ".tmp"}:
                    raise ModelImportError("文件仍是下载临时文件，请等浏览器下载完成后选择 .gguf 模型文件。")
                if source.suffix.casefold() in {".html", ".htm", ".url"}:
                    raise ModelImportError("选择的是网页或网页快捷方式，不是模型权重；请下载完整 .gguf 文件后再导入。")
                if not 0 < source.stat().st_size <= self.max_file_bytes:
                    raise ModelImportError("模型包大小超过安全限制。")
                yield "file", {source.name: source}, None
            else:
                raise ModelImportError("请选择完整模型文件夹、ZIP 包；单文件仅支持单文件模型。")
        except (OSError, zipfile.BadZipFile, zlib.error, EOFError) as exc:
            raise ModelImportError("模型包无法读取或已损坏；原包保留。") from exc
        finally:
            if archive is not None:
                archive.close()

    @staticmethod
    @contextmanager
    def _read_candidate(candidate, archive):
        if isinstance(candidate, zipfile.ZipInfo):
            with archive.open(candidate) as stream:
                yield stream
        else:
            candidate = _plain_path(candidate)
            before = _identity(candidate.stat())
            with candidate.open("rb") as stream:
                yield stream
            if _identity(candidate.stat()) != before:
                raise ModelImportError("模型来源在检查期间发生变化，请重新选择。")

    def inspect_source(self, source: Path, templates: tuple[ModelEntry, ...],
                       progress: Progress | None = None, cancel=None) -> ImportInspection:
        """Offline detection with the same ZIP/path limits as the final copy.

        Catalog matches require their fixed hashes, not filenames. Unknown ASR
        and CT2 exports have no generic adapter and are intentionally refused.
        """
        from .model_registry import ModelRegistry, validate_display_name
        from .gguf_inspection import inspect_gguf
        _cancelled(cancel)
        with self._source_files(source, cancel) as (kind, candidates, archive):
            hashes = {}
            def size(candidate):
                return candidate.file_size if isinstance(candidate, zipfile.ZipInfo) else candidate.stat().st_size
            def digest(candidate):
                key = candidate.filename if isinstance(candidate, zipfile.ZipInfo) else str(candidate)
                if key not in hashes:
                    done, hasher = 0, hashlib.sha256()
                    with self._read_candidate(candidate, archive) as stream:
                        while block := stream.read(self.chunk_size):
                            _cancelled(cancel)
                            done += len(block)
                            if done > self.max_file_bytes:
                                raise ModelImportError("模型包大小超过安全限制。")
                            hasher.update(block)
                            if progress:
                                progress("inspecting", done, max(1, size(candidate)))
                    hashes[key] = hasher.hexdigest()
                return hashes[key]
            options = []
            for entry in templates:
                _cancelled(cancel)
                if entry.required or entry.verification != "pinned_catalog_sha256":
                    continue
                try:
                    role = ModelRegistry.role(entry)
                    matched = ([next(iter(candidates.values()))] if kind == "file" and len(entry.files) == 1
                               else [self._match(file, candidates) for file in entry.files])
                except DownloadError:
                    continue
                if all(size(candidate) == file.size for candidate, file in zip(matched, entry.files)):
                    if all(digest(candidate) == file.sha256 for candidate, file in zip(matched, entry.files)):
                        options.append(ImportOption(entry, role, entry.name))
            if options:
                return ImportInspection(tuple(options), kind)
            ggufs = [(name, candidate) for name, candidate in candidates.items() if name.lower().endswith(".gguf")]
            if len(ggufs) != 1:
                raise ModelImportError("未识别为已支持的模型。未知 ONNX / CT2 不受支持；请提供完整预设模型包或单个兼容 GGUF。")
            name, candidate = ggufs[0]
            original_identity = None if isinstance(candidate, zipfile.ZipInfo) else _identity(candidate.stat())
            with self._read_candidate(candidate, archive) as stream:
                metadata = inspect_gguf(stream, size(candidate), cancel=cancel)
            filename = Path(name).stem + ".gguf"
            model_file = ModelFile("translation/" + filename, None, size(candidate), digest(candidate))
            if original_identity is not None and _identity(candidate.stat()) != original_identity:
                raise ModelImportError("模型来源在检查期间发生变化，请重新选择。")
            suggested_name = metadata.name
            try:
                suggested_name = validate_display_name(suggested_name)
            except DownloadError:
                suggested_name = "Local " + metadata.architecture
            template = ModelEntry("detected-gguf", suggested_name, (model_file,), "",
                                  "自定义本地 GGUF；仅校验结构与本地摘要，官方来源未经验证。",
                                  model_kind=metadata.model_kind, verification="local_sha256",
                                  architecture=metadata.architecture)
            return ImportInspection((ImportOption(template, "translation", suggested_name),), kind)

    def import_named(self, option: ImportOption, source: Path, display_name: str, registry,
                     progress: Progress | None = None, cancel=None) -> ModelEntry:
        from .model_registry import validate_display_name
        name = validate_display_name(display_name)
        if any(entry.name.casefold() == name.casefold() for entry in registry.entries()):
            raise ModelImportError("本地模型名称或 ID 已存在，请使用其他名称。")
        entry = registry.named_entry(option.template, name)
        self.import_entry(entry, source, progress, cancel)
        _cancelled(cancel)
        registry.register(entry)
        return entry

    def import_entry(self, entry: ModelEntry, source: Path, progress: Progress | None = None,
                     cancel: threading.Event | None = None) -> ImportResult:
        _cancelled(cancel)
        if entry.model_kind in ("whisper", "qwen3_asr"):
            raise ModelImportError("当前仅支持 SenseVoice 识别模型。")
        if not entry.files or entry.model_kind not in (None, "hy", "qwen", "sense_voice"):
            raise ModelImportError("未知模型架构不能导入；请使用界面支持的模型目录版本。")
        if entry.size > self.max_total_bytes or any(file.size > self.max_file_bytes for file in entry.files):
            raise ModelImportError("模型包大小超过安全限制。")
        source = _plain_path(Path(source))
        if source == self.models_root or source.is_relative_to(self.models_root) or self.models_root.is_relative_to(source):
            raise ModelImportError("模型来源与存储目录不能相同或相互嵌套。")
        targets = {file.relative_path: self._target(file) for file in entry.files}
        provenance = self.provenance_path(entry)
        if provenance.exists() and not self.has_provenance(entry):
            raise ModelImportError("已有导入来源记录不匹配，已保留，请手动处理后重试。")
        reused = []
        for file in entry.files:
            target = targets[file.relative_path]
            if target.exists():
                self._verify_existing(target, file, cancel, progress)
                reused.append(file.relative_path)
        archive = None
        staged = None
        imported = []
        certificates = {}
        try:
            if source.is_dir():
                kind = "folder"
                candidates = self._folder_files(source, cancel)
            elif source.is_file() and source.suffix.casefold() == ".zip":
                kind = "zip"
                if source.stat().st_size > self.max_total_bytes:
                    raise ModelImportError("模型包大小超过安全限制。")
                self._zip_directory_limits(source)
                archive = zipfile.ZipFile(source)
                candidates = self._zip_files(archive, cancel)
            elif source.is_file() and len(entry.files) == 1:
                kind = "file"
                candidates = {Path(entry.files[0].relative_path).name: source}
            else:
                raise ModelImportError("请选择完整模型文件夹、ZIP 包；单文件仅支持单文件模型。")
            matched = {file.relative_path: self._match(file, candidates) for file in entry.files}
            for file in entry.files:
                candidate = matched[file.relative_path]
                size = candidate.file_size if isinstance(candidate, zipfile.ZipInfo) else candidate.stat().st_size
                if size != file.size:
                    raise ModelImportError("模型文件大小与选中目录版本不匹配。")
            self.models_root.mkdir(parents=True, exist_ok=True)
            staged = _plain_path(self.models_root / ".model-import-staging" / uuid.uuid4().hex, allow_missing=True)
            staged.mkdir(parents=True, exist_ok=False)
            done = 0
            for index, file in enumerate(entry.files):
                _cancelled(cancel)
                candidate = matched[file.relative_path]
                target = _plain_path(staged / str(index), allow_missing=True)
                if isinstance(candidate, zipfile.ZipInfo):
                    with archive.open(candidate) as stream:
                        self._copy_verified(stream, target, file, cancel, progress, phase="importing", base=done, total=entry.size)
                else:
                    candidate = _plain_path(candidate)
                    before = candidate.stat()
                    with candidate.open("rb") as stream:
                        self._copy_verified(stream, target, file, cancel, progress, phase="importing", base=done, total=entry.size)
                    if _identity(before) != _identity(candidate.stat()):
                        raise ModelImportError("模型来源在复制期间发生变化，导入未完成。")
                done += file.size
                certificates[index] = _identity(target.stat())
            _cancelled(cancel)
            for index, file in enumerate(entry.files):
                if file.relative_path in reused:
                    continue
                _cancelled(cancel)
                target = self._target(file)
                target.parent.mkdir(parents=True, exist_ok=True)
                target = self._target(file)
                temporary = _plain_path(staged / str(index))
                if _identity(temporary.stat()) != certificates[index]:
                    raise ModelImportError("导入暂存在校验后发生变化，未完成发布。")
                # Only this operation's verified temporary copy is moved. The
                # user's source is untouched and destination replacement is off.
                if os.name == "nt":
                    os.rename(temporary, target)
                else:
                    os.link(temporary, target)
                    temporary.unlink()
                if progress:
                    progress("publishing", index + 1, len(entry.files))
                imported.append(file.relative_path)
            record = {"schema_version": 1, "model_id": entry.id, "source_kind": kind,
                      "verification": entry.verification, "self_supplied_checksum": entry.verification != "pinned_catalog_sha256",
                      "files": [{"path": file.relative_path, "size": file.size, "sha256": file.sha256} for file in entry.files]}
            provenance.parent.mkdir(parents=True, exist_ok=True)
            provenance = self.provenance_path(entry)
            if provenance.exists():
                if not self.has_provenance(entry):
                    raise ModelImportError("已有导入来源记录不匹配，已保留，请手动处理后重试。")
            else:
                with provenance.open("x", encoding="utf-8") as stream:
                    json.dump(record, stream, ensure_ascii=False, indent=2)
                    stream.flush()
                    os.fsync(stream.fileno())
            # Remove only generated leftovers after success; failed/cancelled
            # staging is deliberately retained for inspection, never source data.
            for index in range(len(entry.files)):
                temporary = staged / str(index)
                if temporary.exists():
                    _plain_path(temporary).unlink()
            staged.rmdir()
            return ImportResult(entry.id, tuple(imported), tuple(reused), provenance, kind, entry.verification)
        except (OSError, zipfile.BadZipFile, zlib.error, EOFError, RuntimeError) as exc:
            if isinstance(exc, DownloadError):
                raise
            raise ModelImportError("模型导入失败；原包、已有模型和未完成暂存均保留。") from exc
        finally:
            if archive is not None:
                archive.close()
