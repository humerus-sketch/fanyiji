"""Keep Whisper BPE byte fragments intact across native per-token UTF-8 filtering."""
from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import uuid


# The two pinned multilingual Whisper exports use this same original table.
WHISPER_TOKEN_SHA256 = "b34b360dbb493e781e479794586d661700670d65564001f23024971d1f2fa126"
METHOD = "ASCII __WBYTE_hex__ tokens; concatenate bytes before one UTF-8 decode"
_MARKER = re.compile(r"__WBYTE_([0-9a-f]*)__")


@dataclass(frozen=True)
class WhisperTokenTable:
    path: Path
    source_sha256: str
    derived_sha256: str


def _contents(raw: bytes) -> bytes:
    lines, seen = [], set()
    try:
        for line in raw.decode("utf-8").splitlines():
            encoded, token_id = line.split()
            if not token_id.isascii() or not token_id.isdecimal() or int(token_id) in seen:
                raise ValueError("token ID 无效或重复")
            seen.add(int(token_id))
            # The official table uses '=' for the empty token at ID 50256.
            if encoded == "=":
                if token_id != "50256":
                    raise ValueError("空 token 占位符的 ID 无效")
                original = b""
            else:
                original = base64.b64decode(encoded, validate=True)
            marker = "__WBYTE_" + original.hex() + "__"
            lines.append(base64.b64encode(marker.encode("ascii")).decode("ascii") + " " + token_id)
        if not lines:
            raise ValueError("词表为空")
    except (ValueError, UnicodeError) as exc:
        raise ValueError("Whisper 原词表的 base64/token ID 格式无效") from exc
    return ("\n".join(lines) + "\n").encode("ascii")


def prepare_whisper_tokens(source: str | Path, *, expected_sha256: str | None = None) -> WhisperTokenTable:
    """Verify the original and an immutable cached derivative; never write source."""
    source = Path(source).resolve()
    before = source.stat()
    if not 0 < before.st_size <= 16 * 1024 * 1024:
        raise ValueError("Whisper 原词表大小无效")
    raw = source.read_bytes()
    after = source.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError("Whisper 原词表在读取期间发生变化")
    source_sha = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and source_sha != expected_sha256:
        raise ValueError("Whisper 原词表 SHA-256 与固定来源不匹配，未加载")
    contents = _contents(raw)
    derived_sha = hashlib.sha256(contents).hexdigest()
    # Full source SHA identifies the cache. The deterministic derivative is
    # independently hashed; its short suffix keeps Windows paths compact.
    output = source.with_name(f"bytes-v1-{source_sha}-{derived_sha[:8]}.txt")
    if not output.resolve().is_relative_to(source.parent):
        raise ValueError("Whisper 派生词表路径超出模型目录")
    if output.exists():
        if output.stat().st_size != len(contents) or hashlib.sha256(output.read_bytes()).hexdigest() != derived_sha:
            raise ValueError("Whisper 派生词表校验失败，已保留原文件，未加载")
    else:
        temporary = output.with_name(output.name + "." + uuid.uuid4().hex + ".tmp")
        with temporary.open("xb") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    return WhisperTokenTable(output, source_sha, derived_sha)


def restore_whisper_bytes(text: str) -> str:
    """Decode exactly once, including ordinary speech that resembles markers."""
    if not text:
        return ""
    data, cursor = bytearray(), 0
    try:
        for match in _MARKER.finditer(text):
            if match.start() != cursor:
                raise ValueError("标记外出现数据")
            data.extend(bytes.fromhex(match.group(1)))
            cursor = match.end()
        if cursor != len(text):
            raise ValueError("标记不完整或含有额外数据")
    except ValueError as exc:
        raise ValueError("Whisper byte-safe 输出标记无法解析，识别失败") from exc
    # A model can stop halfway through a UTF-8 sequence. Preserve a visible
    # replacement character instead of guessing or silently dropping speech.
    return data.decode("utf-8", "replace")
