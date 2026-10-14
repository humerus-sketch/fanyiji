"""Offline, locked training preparation; heavy training imports are opt-in only.

The desktop application may generate a recipe without installing a second ML
runtime. Actual training is an explicit action in a separate Python environment;
neither a completed adapter nor its manifest activates a subtitle provider.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import unicodedata
import uuid

if __package__:
    from .livestream_prompt import (
        HY_CONTEXT_CHARACTERS, HY_PROMPT_FORMAT, LEGACY_HY_CONTEXT_CHARACTERS,
        LEGACY_HY_PROMPT_FORMAT, PROFILE_ID, hy_translation_prompt,
    )
else:
    # The generated training kit uses only adjacent stdlib-only modules.
    from livestream_prompt import (
        HY_CONTEXT_CHARACTERS, HY_PROMPT_FORMAT, LEGACY_HY_CONTEXT_CHARACTERS,
        LEGACY_HY_PROMPT_FORMAT, PROFILE_ID, hy_translation_prompt,
    )


BASE_MODEL_ID = "tencent/Hy-MT2-1.8B"
DEPLOYMENT_LOCKED = True
MIN_TRAIN_ROWS = 500
MIN_VALIDATION_ROWS = 50
TRAINING_PACKAGES = ("torch", "transformers", "peft", "bitsandbytes", "accelerate")
_TZ = timezone(timedelta(hours=8))
_MAX_JSON_BYTES = 1024 * 1024
_MAX_DATASET_BYTES = 128 * 1024 * 1024
_MAX_ROWS = 100_000
OFFICIAL_REFERENCES = {
    "base_model": "https://huggingface.co/tencent/Hy-MT2-1.8B",
    "training": "https://huggingface.co/tencent/Hy-MT2-1.8B/blob/main/train/README.md",
    "native_architecture": "https://huggingface.co/docs/transformers/model_doc/hunyuan_v1_dense",
    "quantized_lora": "https://huggingface.co/docs/peft/developer_guides/quantization",
    "windows_quantization": "https://huggingface.co/docs/bitsandbytes/en/installation",
    "codex_output_terms": "https://openai.com/policies/terms-of-use/",
}


class TrainingPreparationError(ValueError):
    """An asset or corpus cannot safely enter the dedicated training workflow."""


class TrainingExecutionError(RuntimeError):
    """An explicit run failed after entering the model-loading/training stage."""

    training_executed = True


def _timestamp() -> str:
    return datetime.now(_TZ).isoformat(timespec="seconds")


def _save_new(path: Path, value: dict) -> None:
    # Every preparation/run gets a fresh directory, so existing artifacts are
    # never replaced. A failed run also remains available for diagnosis.
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _local_child(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or ":" in relative or "\\" in relative:
        raise TrainingPreparationError("文件路径必须是数据目录内的相对路径")
    if any(part in ("", ".", "..") for part in relative.split("/")):
        raise TrainingPreparationError("数据文件路径不能越过所属目录")
    result = (root / relative).resolve()
    if not result.is_relative_to(root.resolve()):
        raise TrainingPreparationError("数据文件路径不能越过所属目录")
    return result


def _read_json(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size > _MAX_JSON_BYTES:
        raise TrainingPreparationError(f"缺少有效的 JSON 文件：{path.name}")
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise TrainingPreparationError(f"JSON 格式无效：{path.name}") from exc
    if not isinstance(value, dict):
        raise TrainingPreparationError(f"JSON 顶层必须是对象：{path.name}")
    return value


def training_recipe(root: Path) -> dict:
    root = Path(root).resolve()
    return {
        "format_version": 1,
        "base_model_id": BASE_MODEL_ID,
        "base_model_path": str(root / "base" / "Hy-MT2-1.8B"),
        "method": "qlora",
        "quantization": "nf4",
        "double_quantization": True,
        "max_sequence_length": 512,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "gradient_checkpointing": True,
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "learning_rate": 0.0002,
        "epochs": 1.0,
        "min_train_rows": MIN_TRAIN_ROWS,
        "min_validation_rows": MIN_VALIDATION_ROWS,
        "requires_local_safetensors": True,
        "trust_remote_code": False,
        "local_files_only": True,
        "codex_training_allowed": False,
        "deployment_locked": DEPLOYMENT_LOCKED,
        "eight_gb_fit_verified": False,
        "official_lora_8192_sequence_minimum_vram_gb": 24,
        "official_references": OFFICIAL_REFERENCES,
    }


def _package_versions() -> dict:
    result = {}
    for name in TRAINING_PACKAGES:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def validate_base_model(path: Path) -> dict:
    path = Path(path).resolve()
    config = _read_json(path / "config.json")
    # These dimensions identify the intended 1.8B architecture. A local GGUF,
    # a different Hunyuan size, and custom remote Python classes are rejected.
    expected = {
        "model_type": "hunyuan_v1_dense", "hidden_size": 2048,
        "num_hidden_layers": 32, "intermediate_size": 6144, "vocab_size": 120818,
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise TrainingPreparationError("训练基底必须是官方 Hy-MT2-1.8B 的 HF 架构与尺寸")
    if config.get("architectures") != ["HunYuanDenseV1ForCausalLM"]:
        raise TrainingPreparationError("训练基底的 architecture 不受支持")
    if config.get("auto_map"):
        raise TrainingPreparationError("训练不执行模型目录中的自定义 Python 代码")
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        item = _local_child(path, name)
        if not item.is_file() or item.stat().st_size == 0:
            raise TrainingPreparationError(f"训练基底缺少 {name}")
    if _read_json(path / "tokenizer_config.json").get("auto_map"):
        raise TrainingPreparationError("训练不执行自定义 tokenizer 代码")
    index = path / "model.safetensors.index.json"
    if index.is_file():
        weight_map = _read_json(index).get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise TrainingPreparationError("safetensors 索引无效")
        values = list(weight_map.values())
        if any(not isinstance(name, str) or not name.endswith(".safetensors") for name in values):
            raise TrainingPreparationError("训练只加载 safetensors 权重")
        names = set(values)
        weights = [_local_child(path, name) for name in sorted(names)]
    else:
        weights = [_local_child(path, "model.safetensors")]
    if any(not item.is_file() or item.stat().st_size == 0 for item in weights):
        raise TrainingPreparationError("训练基底缺少完整的 safetensors 权重；GGUF 不能直接用于训练")
    return {
        "model_id": BASE_MODEL_ID,
        "path": str(path),
        "config_sha256": _digest(path / "config.json"),
        "weight_files": [str(item.relative_to(path)) for item in weights],
        "weight_bytes": sum(item.stat().st_size for item in weights),
        "deployment_locked": True,
    }


def _read_rows(path: Path, *, prompt_format: str = LEGACY_HY_PROMPT_FORMAT,
               translation_profile: str | None = None) -> list[dict]:
    if not path.is_file() or path.stat().st_size > _MAX_DATASET_BYTES:
        raise TrainingPreparationError(f"数据文件不存在或超过 128 MiB：{path.name}")
    rows = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            if len(line) > 64_000 or len(rows) >= _MAX_ROWS:
                raise TrainingPreparationError("训练数据的单行长度或样本数超限")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TrainingPreparationError(f"{path.name} 第 {number} 行不是有效 JSON") from exc
            if not isinstance(row, dict) or row.get("direction") not in ("ja_to_zh", "zh_to_ja"):
                raise TrainingPreparationError(f"{path.name} 第 {number} 行缺少中日翻译方向")
            messages = row.get("messages")
            if (not isinstance(messages, list) or len(messages) != 2
                    or any(not isinstance(item, dict) for item in messages)
                    or [item.get("role") for item in messages] != ["user", "assistant"]
                    or any(not isinstance(item.get("content"), str) or not item["content"].strip()
                           for item in messages)):
                raise TrainingPreparationError(f"{path.name} 第 {number} 行必须有完整 user / assistant 对")
            row_format = row.get("prompt_format", prompt_format)
            if row_format != prompt_format:
                raise TrainingPreparationError("行提示格式与数据清单不一致")
            row_profile = row.get("translation_profile", translation_profile)
            if row_profile != translation_profile:
                raise TrainingPreparationError("行提示配置与数据清单不一致")
            if prompt_format == HY_PROMPT_FORMAT and "raw_source" not in row:
                raise TrainingPreparationError("新版训练提示必须保留采集原文")
            if "raw_source" in row:
                source, context = row["raw_source"], row.get("source_context", "")
                context_limit = (HY_CONTEXT_CHARACTERS if prompt_format == HY_PROMPT_FORMAT
                                 else LEGACY_HY_CONTEXT_CHARACTERS)
                if (not isinstance(source, str) or not source.strip()
                        or not isinstance(context, str) or len(context) > context_limit):
                    raise TrainingPreparationError("原文或参考原文上下文无效")
                if messages[0]["content"] != _canonical_hy_prompt(
                        source, context, row["direction"], prompt_format=prompt_format):
                    raise TrainingPreparationError("训练提示与采集原文和上下文不一致")
            provenance = row.get("provenance")
            if not isinstance(provenance, dict):
                raise TrainingPreparationError("训练样本必须包含逐行来源信息")
            if provenance.get("origin") not in ("local_model", "independent_human"):
                raise TrainingPreparationError("Codex 等云端候选不能进入当前训练集")
            if (any(provenance.get(key) is not True for key in
                    ("reviewed", "source_verified", "rights_confirmed"))
                    or provenance.get("codex_derived") is not False):
                raise TrainingPreparationError("训练样本必须完成原文、译文和使用权限核对，且不得派生自 Codex")
            for key in ("source_session", "source_group"):
                if not isinstance(provenance.get(key), str) or not provenance[key].strip():
                    raise TrainingPreparationError("训练样本缺少会话或原文分组")
            rows.append(row)
    if not rows:
        raise TrainingPreparationError(f"{path.name} 没有有效样本")
    return rows


def _canonical_hy_prompt(source: str, context: str, direction: str, *,
                         prompt_format: str = LEGACY_HY_PROMPT_FORMAT) -> str:
    if prompt_format not in (LEGACY_HY_PROMPT_FORMAT, HY_PROMPT_FORMAT):
        raise TrainingPreparationError("训练提示格式无效")
    source_lang, target_lang = ("ja", "zh") if direction == "ja_to_zh" else ("zh", "ja")
    return hy_translation_prompt(source, source_lang, target_lang, context,
                                 profile_enabled=prompt_format == HY_PROMPT_FORMAT)


def _source_key(row: dict) -> str:
    value = unicodedata.normalize("NFKC", row.get("raw_source", row["messages"][0]["content"]))
    return row["direction"] + "\0" + "".join(value.split())


def load_training_dataset(path: Path) -> tuple[dict, list[dict], list[dict]]:
    path = Path(path).resolve()
    manifest_path = path / "manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("format_version") != 1 or manifest.get("deployment_locked") is not True:
        raise TrainingPreparationError("数据清单版本或专用模型锁定状态无效")
    if manifest.get("codex_rows") != 0:
        raise TrainingPreparationError("数据清单必须明确声明 codex_rows 为 0")
    prompt_format = manifest.get("prompt_format", LEGACY_HY_PROMPT_FORMAT)
    if prompt_format not in (LEGACY_HY_PROMPT_FORMAT, HY_PROMPT_FORMAT):
        raise TrainingPreparationError("数据清单提示格式无效")
    translation_profile = manifest.get("translation_profile")
    expected_profile = PROFILE_ID if prompt_format == HY_PROMPT_FORMAT else None
    if translation_profile != expected_profile:
        raise TrainingPreparationError("数据清单提示配置与格式版本不一致")
    context_limit = (HY_CONTEXT_CHARACTERS if prompt_format == HY_PROMPT_FORMAT
                     else LEGACY_HY_CONTEXT_CHARACTERS)
    declared_context_limit = manifest.get("source_context_max_characters", context_limit)
    if type(declared_context_limit) is not int or declared_context_limit != context_limit:
        raise TrainingPreparationError("数据清单上下文预算与提示版本不一致")
    train_path = _local_child(path, manifest.get("train_file"))
    validation_path = _local_child(path, manifest.get("validation_file"))
    if train_path == validation_path:
        raise TrainingPreparationError("训练集和验证集必须独立")
    train_rows = _read_rows(train_path, prompt_format=prompt_format, translation_profile=translation_profile)
    validation_rows = _read_rows(validation_path, prompt_format=prompt_format, translation_profile=translation_profile)
    for name, rows in (("train_rows", train_rows), ("validation_rows", validation_rows)):
        if manifest.get(name) != len(rows):
            raise TrainingPreparationError("数据清单样本数与实际文件不一致")
    for field in ("source_session", "source_group"):
        train_groups = {row["provenance"][field] for row in train_rows}
        validation_groups = {row["provenance"][field] for row in validation_rows}
        if train_groups & validation_groups:
            raise TrainingPreparationError("同一会话或原文分组泄漏到验证集")
    train_keys = [_source_key(row) for row in train_rows]
    validation_keys = [_source_key(row) for row in validation_rows]
    if (set(train_keys) & set(validation_keys) or len(set(train_keys)) != len(train_keys)
            or len(set(validation_keys)) != len(validation_keys)):
        raise TrainingPreparationError("训练集与验证集存在重复原文，需先去重并重新分组")
    summary = {
        "path": str(path), "train_rows": len(train_rows), "validation_rows": len(validation_rows),
        "codex_rows": 0, "manifest_sha256": _digest(manifest_path),
        "train_sha256": _digest(train_path), "validation_sha256": _digest(validation_path),
        "sessions_disjoint": True, "source_groups_disjoint": True,
        "prompt_format": prompt_format, "translation_profile": translation_profile,
        "source_context_max_characters": context_limit,
        "deployment_locked": True,
    }
    return summary, train_rows, validation_rows


def preflight(root: Path, *, dataset: Path | None = None, base_model: Path | None = None) -> dict:
    root = Path(root).resolve()
    versions = _package_versions()
    report = {
        "format_version": 1, "created_at": _timestamp(), "deployment_locked": True,
        "training_executed": False, "training_ready": False,
        "runtime_packages": versions, "runtime_gpu_probe_executed": False,
        "eight_gb_fit_verified": False, "blockers": [],
    }
    if dataset is None:
        report["blockers"].append("尚未指定经过审核并按会话隔离的训练数据集")
    else:
        try:
            report["dataset"], _, _ = load_training_dataset(dataset)
            if report["dataset"]["train_rows"] < MIN_TRAIN_ROWS:
                report["blockers"].append(f"训练集少于 {MIN_TRAIN_ROWS} 条审核样本")
            if report["dataset"]["validation_rows"] < MIN_VALIDATION_ROWS:
                report["blockers"].append(f"验证集少于 {MIN_VALIDATION_ROWS} 条独立审核样本")
        except (OSError, TrainingPreparationError) as exc:
            report["blockers"].append(str(exc))
    try:
        report["base_model"] = validate_base_model(base_model or root / "base" / "Hy-MT2-1.8B")
    except (OSError, TrainingPreparationError) as exc:
        report["blockers"].append(str(exc))
    missing = [name for name, version in versions.items() if version is None]
    if missing:
        report["blockers"].append("独立训练环境缺少：" + ", ".join(missing))
    # A light preflight does not load 4 GB weights, initialize CUDA or make a
    # claim that a forward/backward pass fits this specific 8 GB GPU.
    report["assets_ready"] = not report["blockers"]
    if report["assets_ready"]:
        report["blockers"].append("仍需在独立环境完成 CUDA、原生 architecture 和实际显存试验")
    return report


def _emit_training_kit(directory: Path) -> Path:
    if getattr(sys, "frozen", False):
        sources = Path(sys.executable).resolve().parent / "training_tools"
        module_path = sources / "specialized_training.py"
        runner_path = sources / "train_specialized_model.py"
        prompt_path = sources / "livestream_prompt.py"
    else:
        module_path = Path(__file__).resolve()
        adjacent_runner = module_path.parent / "train_specialized_model.py"
        runner_path = (adjacent_runner if adjacent_runner.is_file()
                       else module_path.parents[2] / "tools" / "train_specialized_model.py")
        prompt_path = module_path.parent / "livestream_prompt.py"
    if not all(source.is_file() for source in (module_path, runner_path, prompt_path)):
        raise TrainingPreparationError("发行目录缺少 training_tools，无法生成独立训练工具")
    destination = directory / "工具"
    destination.mkdir()
    for source in (module_path, runner_path, prompt_path):
        with (destination / source.name).open("xb") as stream:
            stream.write(source.read_bytes())
    return destination / "train_specialized_model.py"


def prepare_training(root: Path) -> dict:
    root = Path(root).resolve()
    directory = root / "training" / ("prepare-" + datetime.now(_TZ).strftime("%Y%m%d-%H%M%S")
                                       + "-" + uuid.uuid4().hex[:8])
    directory.mkdir(parents=True, exist_ok=False)
    runner_path = _emit_training_kit(directory)
    recipe = training_recipe(root)
    report = preflight(root)
    _save_new(directory / "recipe.json", recipe)
    _save_new(directory / "preflight.json", report)
    guide = f"""# 专用模型训练准备

当前阶段：只生成训练方案和检查结果，未执行微调；使用专用模型仍锁定。

采用一个 `{BASE_MODEL_ID}` 基底及 QLoRA adapter。其他本地模型提供对照候选，
不得直接混合不同架构权重。Codex 校对候选独立保存，当前不进入训练数据。

## 数据与模型

- 专用数据目录：`{root}`。已下载的正常翻译模型保持原用途。
- 原始 HF safetensors 基底目录：`{root / 'base' / 'Hy-MT2-1.8B'}`。
- GGUF 是推理权重，不能直接用于本工具微调；基底不自动下载。
- 数据来自专用模型页面的审核导出。每行都要核对原文、译文与使用权限。
- 首轮训练至少 {MIN_TRAIN_ROWS} 条审核训练样本及 {MIN_VALIDATION_ROWS} 条独立验证样本。
- 同一会话、相同原文不能跨训练/验证集；训练不用验证集优化参数。

## 独立环境与显存

另建 Python 训练环境，不在字幕软件 `.venv` 中安装或升级训练依赖。
依赖包括 CUDA 版 PyTorch、Transformers、PEFT、bitsandbytes、Accelerate。
使用官方当前兼容组合，并确认 Transformers 原生支持 `HunYuanDenseV1ForCausalLM`。
工具禁止 remote code、pickle 模型权重以及联网自动补齐模型。

试验配置：NF4 4 bit、序列 512、batch 1、gradient checkpointing、LoRA rank 8。
官方普通 LoRA 在序列 8192 时列出的最小显存为 24 GB；本配置在 8 GB 显卡上
能否实际完成前向、反向和保存尚未验证。训练时先暂停字幕与本地模型对照，
至少留下 6 GiB 空闲显存；OOM 记录为失败，不自动重试或降低配置碰运气。

## 执行顺序

1. 本次已经生成独立工具：`{runner_path}`，无需源码仓库或字幕软件 Python 环境。
2. 用独立环境的 Python 运行这个工具，参数为 `--root "{root}" --dataset <导出目录> --preflight`。
3. 准备并核对本地 HF 基底，预检的数据数量与来源问题解决后，显式添加 `--train` 才会训练。
4. 输出只写入专用目录 `adapters/run-...`，不覆盖既有模型、配置或旧 adapter。
5. 检查有限 loss、有效 safetensors adapter 和独立 holdout loss。
6. 后续另做中日双向人审准确率、数字/否定/名称、泛化和实时延迟评测。
7. adapter 即使训练完成，也不自动开放字幕使用；需要另行审核、转换和明确开放。

验证 loss 只用于训练诊断，不能代替翻译准确率，也不能证明比现有模型更快。
模型转换为推理 GGUF 是后续工作，本工具不执行转换或激活。

## 官方依据

"""
    guide += "\n".join(f"- [{name}]({url})" for name, url in OFFICIAL_REFERENCES.items()) + "\n"
    (directory / "训练说明.md").write_text(guide, encoding="utf-8")
    return {
        "directory": str(directory), "recipe_path": str(directory / "recipe.json"),
        "guide_path": str(directory / "训练说明.md"), "preflight_path": str(directory / "preflight.json"),
        "training_tool_path": str(runner_path),
        "deployment_locked": True, "training_executed": False,
        "phase": "prepared_not_trained", "blockers": report["blockers"],
    }


@contextmanager
def _training_lock(root: Path):
    lock_path = root / "training" / "training.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise TrainingPreparationError("已有专用训练任务运行中") from exc
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise TrainingPreparationError("已有专用训练任务运行中") from exc
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def tokenize_training_rows(tokenizer, rows: list[dict], max_length: int = 512) -> list[dict]:
    result = []
    for row in rows:
        prompt = tokenizer.apply_chat_template(
            row["messages"][:1], add_generation_prompt=True, tokenize=True,
        )
        answer = tokenizer.encode(row["messages"][1]["content"], add_special_tokens=False)
        if tokenizer.eos_token_id is None:
            raise TrainingPreparationError("tokenizer 缺少 eos_token_id")
        answer.append(tokenizer.eos_token_id)
        tokens = prompt + answer
        if not prompt or len(answer) < 2 or len(tokens) > max_length:
            raise TrainingPreparationError("存在空目标或超过序列限制的样本；不得静默截断译文")
        result.append({
            "input_ids": tokens, "attention_mask": [1] * len(tokens),
            "labels": [-100] * len(prompt) + answer,
        })
    return result


def train_local_adapter(root: Path, *, dataset: Path, base_model: Path | None = None) -> dict:
    """Explicit future execution, always producing a locked, unverified adapter."""
    root = Path(root).resolve()
    summary, train_rows, validation_rows = load_training_dataset(dataset)
    if len(train_rows) < MIN_TRAIN_ROWS or len(validation_rows) < MIN_VALIDATION_ROWS:
        raise TrainingPreparationError("审核样本不足，不能开始专用模型训练")
    base_info = validate_base_model(base_model or root / "base" / "Hy-MT2-1.8B")
    versions = _package_versions()
    if any(value is None for value in versions.values()):
        raise TrainingPreparationError("请先准备独立训练环境；字幕软件不安装训练依赖")
    executable = Path(sys.executable).resolve()
    application_env = Path(__file__).resolve().parents[2] / ".venv"
    try:
        # Also applies to the emitted standalone kit, whose __file__ is outside
        # the source checkout. Production ML distributions identify the app env.
        for name in ("PySide6-Essentials", "sherpa-onnx", "ctranslate2"):
            importlib.metadata.version(name)
        has_application_runtime = True
    except importlib.metadata.PackageNotFoundError:
        has_application_runtime = False
    if executable.is_relative_to(application_env.resolve()) or has_application_runtime:
        raise TrainingPreparationError("训练必须在独立 Python 环境运行，不能使用字幕软件 .venv")
    with _training_lock(root):
        # Offline flags are established before the optional ML libraries load.
        for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
            os.environ[key] = "1"
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        from transformers import BitsAndBytesConfig, Trainer, TrainerCallback, TrainingArguments
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        if not torch.cuda.is_available():
            raise TrainingPreparationError("QLoRA 试验需要可用的 CUDA 显卡")
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        if free_bytes < 6 * 1024 ** 3:
            raise TrainingPreparationError("空闲显存不足 6 GiB，请暂停字幕和本地对照后再训练")
        base_path = base_info["path"]
        config = AutoConfig.from_pretrained(base_path, local_files_only=True, trust_remote_code=False)
        if type(config).__name__ != "HunYuanDenseV1Config":
            raise TrainingPreparationError("训练环境未正确支持原生 HunYuanDenseV1Config")
        tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True, trust_remote_code=False)
        if not tokenizer.chat_template:
            tokenizer.chat_template = (Path(base_path) / "chat_template.jinja").read_text(encoding="utf-8")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        train_data = tokenize_training_rows(tokenizer, train_rows)
        validation_data = tokenize_training_rows(tokenizer, validation_rows)
        directory = root / "adapters" / ("run-" + datetime.now(_TZ).strftime("%Y%m%d-%H%M%S")
                                           + "-" + uuid.uuid4().hex[:8])
        directory.mkdir(parents=True, exist_ok=False)
        recipe = training_recipe(root)
        run = {
            "format_version": 1, "status": "training", "started_at": _timestamp(),
            "deployment_locked": True, "translation_accuracy_verified": False,
            "latency_verified": False, "dataset": summary, "base_model": base_info,
            "recipe": recipe, "runtime_packages": versions,
            "gpu": {"name": torch.cuda.get_device_name(0), "total_bytes": total_bytes,
                    "free_bytes_before_load": free_bytes},
        }
        _save_new(directory / "started.json", run)

        class FiniteLoss(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                for key in ("loss", "eval_loss", "grad_norm"):
                    if key in (logs or {}) and not math.isfinite(float(logs[key])):
                        raise TrainingPreparationError(f"训练出现非有限 {key}，停止并保留失败记录")

        def collate(rows):
            length = max(len(row["input_ids"]) for row in rows)
            return {
                name: torch.tensor([
                    row[name] + [pad] * (length - len(row[name])) for row in rows
                ], dtype=torch.long)
                for name, pad in (("input_ids", tokenizer.pad_token_id),
                                  ("attention_mask", 0), ("labels", -100))
            }

        try:
            torch.cuda.reset_peak_memory_stats(0)
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            model = AutoModelForCausalLM.from_pretrained(
                base_path, local_files_only=True, trust_remote_code=False, use_safetensors=True,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=dtype,
                ), device_map={"": 0}, dtype=dtype, attn_implementation="sdpa",
            )
            model.config.use_cache = False
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
            model = get_peft_model(model, LoraConfig(
                r=8, lora_alpha=16, lora_dropout=0.05,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                bias="none", task_type="CAUSAL_LM",
            ))
            arguments = TrainingArguments(
                output_dir=str(directory / "work"), overwrite_output_dir=False,
                per_device_train_batch_size=1, per_device_eval_batch_size=1,
                gradient_accumulation_steps=8, num_train_epochs=1.0,
                learning_rate=0.0002, warmup_ratio=0.05,
                gradient_checkpointing=True, optim="adamw_torch",
                bf16=dtype == torch.bfloat16, fp16=dtype == torch.float16,
                logging_steps=1, logging_nan_inf_filter=False, save_strategy="no",
                eval_strategy="no", report_to=[], seed=42, data_seed=42,
                dataloader_num_workers=0, remove_unused_columns=False,
            )
            trainer = Trainer(
                model=model, args=arguments, train_dataset=train_data,
                eval_dataset=validation_data, data_collator=collate,
                processing_class=tokenizer, callbacks=[FiniteLoss()],
            )
            trained = trainer.train()
            evaluated = trainer.evaluate()
            train_loss, eval_loss = float(trained.training_loss), float(evaluated["eval_loss"])
            if not math.isfinite(train_loss) or not math.isfinite(eval_loss):
                raise TrainingPreparationError("训练或验证 loss 非有限，不能接受 adapter")
            checkpoint = directory / "checkpoint"
            model.save_pretrained(str(checkpoint), safe_serialization=True)
            tokenizer.save_pretrained(str(checkpoint))
            adapter_file = checkpoint / "adapter_model.safetensors"
            if not adapter_file.is_file() or adapter_file.stat().st_size == 0:
                raise TrainingPreparationError("训练未生成有效 safetensors adapter")
            run.update({
                "status": "completed_locked", "finished_at": _timestamp(),
                "training_loss": train_loss, "holdout_loss": eval_loss,
                "adapter_sha256": _digest(adapter_file),
                "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(0),
                "checkpoint_path": str(checkpoint), "training_executed": True,
            })
            _save_new(directory / "result.json", run)
            return {**run, "directory": str(directory)}
        except Exception as exc:
            # Keep exception category rather than arbitrary library output that
            # might quote an audio transcript or private corpus row.
            _save_new(directory / "failure.json", {
                "status": "failed", "finished_at": _timestamp(), "deployment_locked": True,
                "error_type": type(exc).__name__, "training_executed": True,
            })
            raise TrainingExecutionError("训练失败，请检查当前 run 目录的 failure.json") from exc
