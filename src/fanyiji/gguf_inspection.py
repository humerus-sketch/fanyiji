"""Bounded, dependency-free GGUF structural inspection; never load tensor weights."""
from __future__ import annotations

from dataclasses import dataclass
import math
import struct
from typing import BinaryIO

from .models_download import DownloadError


class GgufInspectionError(DownloadError):
    pass


@dataclass(frozen=True)
class GgufMetadata:
    architecture: str
    model_kind: str
    name: str
    tensor_count: int
    token_count: int
    header_bytes: int


# GGML stable type IDs/block layouts. New quantizations need explicit support;
# refusing them is preferable to treating an unknown tensor as valid storage.
# Source: ggml-org/llama.cpp ggml/include/ggml.h and ggml/src/ggml-common.h.
_BLOCKS = {0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22),
           7: (32, 24), 8: (32, 34), 9: (32, 36), 10: (256, 84),
           11: (256, 110), 12: (256, 144), 13: (256, 176),
           14: (256, 210), 15: (256, 292), 30: (1, 2)}
_SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i",
            6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
_ARCHITECTURES = {"qwen2": "qwen", "qwen3": "qwen", "hy_v3": "hy", "hunyuan-dense": "hy"}


class _Reader:
    def __init__(self, stream: BinaryIO, size: int, cancel=None):
        self.stream, self.size, self.cancel = stream, size, cancel
        self.limit = min(size, 32 * 1024 * 1024)

    def read(self, length: int) -> bytes:
        if self.cancel is not None and self.cancel.is_set():
            from .model_import import ModelImportCancelled
            raise ModelImportCancelled("导入已取消；原模型包与已有模型保留，未完成暂存可稍后手动清理。")
        if length < 0 or self.stream.tell() + length > self.limit:
            raise GgufInspectionError("GGUF 头部过大或文件不完整。")
        raw = self.stream.read(length)
        if len(raw) != length:
            raise GgufInspectionError("GGUF 头部过大或文件不完整。")
        return raw

    def number(self, code: str):
        return struct.unpack("<" + code, self.read(struct.calcsize(code)))[0]

    def string(self, *, limit=1024 * 1024) -> str:
        length = self.number("Q")
        if length > limit:
            raise GgufInspectionError("GGUF 元数据字符串超过安全限制。")
        try:
            return self.read(length).decode("utf-8", errors="strict")
        except UnicodeError as exc:
            raise GgufInspectionError("GGUF 元数据编码无效。") from exc

    def value(self, kind: int):
        if kind == 8:
            return self.string()
        if kind == 9:
            subtype, count = self.number("I"), self.number("Q")
            if subtype == 9 or subtype not in (*_SCALARS, 8) or count > 500000:
                raise GgufInspectionError("GGUF 元数据数组无效或超过安全限制。")
            # Token strings are parsed but not retained; vocabulary arrays can
            # contain hundreds of thousands of entries and must stay bounded.
            for _ in range(count):
                self.value(subtype)
            return (subtype, count)
        if kind not in _SCALARS:
            raise GgufInspectionError("GGUF 元数据类型未受支持。")
        return self.number(_SCALARS[kind])


def inspect_gguf(stream: BinaryIO, size: int, *, cancel=None,
                 allow_unknown_architecture: bool = False) -> GgufMetadata:
    """Validate metadata/descriptors and tensor byte ranges without inference.

    This establishes a compatible container/prompt family, not model quality or
    trusted provenance. Runtime loading may still reject unsupported operators.
    """
    reader = _Reader(stream, size, cancel)
    if reader.read(4) != b"GGUF" or reader.number("I") not in (2, 3):
        raise GgufInspectionError("请选择完整的 GGUF v2/v3 模型文件。")
    tensors, count = reader.number("Q"), reader.number("Q")
    if not 1 <= tensors <= 200000 or not 1 <= count <= 8192:
        raise GgufInspectionError("GGUF 元数据或张量数量无效。")
    metadata = {}
    for _ in range(count):
        key = reader.string(limit=4096)
        if not key or key in metadata:
            raise GgufInspectionError("GGUF 元数据键无效或重复。")
        metadata[key] = reader.value(reader.number("I"))
    architecture = metadata.get("general.architecture")
    if not isinstance(architecture, str):
        raise GgufInspectionError("GGUF 缺少有效架构元数据。")
    model_kind = _ARCHITECTURES.get(architecture)
    if model_kind is None and not allow_unknown_architecture:
        raise GgufInspectionError("该 GGUF 架构尚未适配；仅支持 HY / Qwen 翻译模板。")
    tokens = metadata.get("tokenizer.ggml.tokens")
    if (not isinstance(tokens, tuple) or tokens[0] != 8 or not 1 <= tokens[1] <= 500000
            or not isinstance(metadata.get("tokenizer.ggml.model"), str)):
        raise GgufInspectionError("GGUF 词表元数据不完整。")
    if type(metadata.get("split.count", 1)) is not int or metadata.get("split.count", 1) != 1:
        raise GgufInspectionError("不支持分片 GGUF；请选择完整单文件模型。")
    if not allow_unknown_architecture:
        for field, low, high in (("block_count", 1, 512), ("context_length", 2048, 2 ** 31),
                                 ("embedding_length", 1, 131072), ("feed_forward_length", 1, 2 ** 24),
                                 ("attention.head_count", 1, 4096)):
            value = metadata.get(architecture + "." + field)
            if type(value) is not int or not low <= value <= high:
                raise GgufInspectionError("GGUF 架构参数不完整或超过支持范围。")
        template = metadata.get("tokenizer.chat_template")
        if not isinstance(template, str) or not template.strip():
            raise GgufInspectionError("GGUF 缺少聊天模板，无法使用现有翻译后端。")
    alignment = metadata.get("general.alignment", 32)
    if type(alignment) is not int or not 1 <= alignment <= 4096 or alignment & (alignment - 1):
        raise GgufInspectionError("GGUF 张量对齐信息无效。")
    intervals, names = [], set()
    for _ in range(tensors):
        name = reader.string(limit=4096)
        dimensions = reader.number("I")
        if not name or name in names or not 1 <= dimensions <= 4:
            raise GgufInspectionError("GGUF 张量描述无效或重复。")
        names.add(name)
        shape = [reader.number("Q") for _ in range(dimensions)]
        if any(not 1 <= n <= 2 ** 31 for n in shape):
            raise GgufInspectionError("GGUF 张量形状无效。")
        dtype, offset = reader.number("I"), reader.number("Q")
        if dtype not in _BLOCKS:
            raise GgufInspectionError("GGUF 量化格式尚未适配。")
        block, width = _BLOCKS[dtype]
        if shape[0] % block or offset % alignment:
            raise GgufInspectionError("GGUF 张量块或偏移无效。")
        length = math.prod(shape) // block * width
        intervals.append((offset, offset + length))
    header = reader.stream.tell()
    data_start = (header + alignment - 1) // alignment * alignment
    end = 0
    for low, high in sorted(intervals):
        if low < end or high + data_start > size:
            raise GgufInspectionError("GGUF 权重范围重叠或文件不完整。")
        end = high
    if not allow_unknown_architecture:
        required = {"token_embd.weight", "output_norm.weight"}
        for layer in range(metadata[architecture + ".block_count"]):
            prefix = "blk." + str(layer) + "."
            required.update(prefix + suffix for suffix in ("attn_q.weight", "attn_k.weight", "attn_v.weight",
                                                           "attn_output.weight", "attn_norm.weight", "ffn_norm.weight"))
            for field in ("ffn_up", "ffn_down", "ffn_gate"):
                if not any(prefix + field + suffix in names for suffix in (".weight", "_exps.weight")):
                    raise GgufInspectionError("GGUF 缺少现有架构所需的完整权重张量。")
        if not required.issubset(names):
            raise GgufInspectionError("GGUF 缺少现有架构所需的完整权重张量。")
    name = metadata.get("general.name", architecture)
    if not isinstance(name, str) or not name.strip():
        name = architecture
    return GgufMetadata(architecture, model_kind or "", name[:80], tensors, tokens[1], header)
