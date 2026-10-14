"""Training boundaries, corpus leakage and genuine opt-in execution contracts."""
from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

import pytest

from fanyiji.livestream_prompt import (
    HY_CONTEXT_CHARACTERS, HY_PROMPT_FORMAT, LEGACY_HY_CONTEXT_CHARACTERS,
    LEGACY_HY_PROMPT_FORMAT, PROFILE_ID, hy_translation_prompt,
)
from fanyiji.specialized_training import (
    BASE_MODEL_ID, MIN_TRAIN_ROWS, MIN_VALIDATION_ROWS, TrainingPreparationError,
    _canonical_hy_prompt,
    load_training_dataset, preflight, prepare_training, tokenize_training_rows,
    train_local_adapter, training_recipe, validate_base_model,
)


def make_base(directory: Path) -> Path:
    directory.mkdir(parents=True)
    config = {
        "model_type": "hunyuan_v1_dense", "hidden_size": 2048, "num_hidden_layers": 32,
        "intermediate_size": 6144, "vocab_size": 120818,
        "architectures": ["HunYuanDenseV1ForCausalLM"],
    }
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (directory / "tokenizer.json").write_text("{}", encoding="utf-8")
    (directory / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (directory / "chat_template.jinja").write_text("official-template-placeholder", encoding="utf-8")
    (directory / "model.safetensors").write_bytes(b"placeholder-not-real-weights")
    return directory


def make_row(number=1, session="train", origin="local_model"):
    return {
        "direction": "ja_to_zh",
        "messages": [{"role": "user", "content": f"原文 {number}"},
                     {"role": "assistant", "content": f"译文 {number}"}],
        "provenance": {
            "origin": origin, "reviewed": True, "source_verified": True,
            "rights_confirmed": True, "codex_derived": False,
            "source_session": session, "source_group": f"source-{number}",
        },
    }


def make_dataset(directory, *, train=None, validation=None, **manifest_overrides):
    directory.mkdir(parents=True, exist_ok=True)
    train = train if train is not None else [make_row(1)]
    validation = validation if validation is not None else [make_row(2, "validation")]
    for name, rows in (("train.jsonl", train), ("validation.jsonl", validation)):
        (directory / name).write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8",
        )
    manifest = {
        "format_version": 1, "deployment_locked": True, "train_file": "train.jsonl",
        "validation_file": "validation.jsonl", "codex_rows": 0,
        "train_rows": len(train), "validation_rows": len(validation),
        **manifest_overrides,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return directory


def contextual_row(number, session, context, *, current_format=False):
    row = make_row(number, session)
    row.update(raw_source=f"会議は{number}日です。", source_context=context)
    if current_format:
        row.update(prompt_format=HY_PROMPT_FORMAT, translation_profile=PROFILE_ID)
        row["messages"][0]["content"] = hy_translation_prompt(row["raw_source"], "ja", "zh", context)
    else:
        row["messages"][0]["content"] = _canonical_hy_prompt(row["raw_source"], context, row["direction"])
    return row


def current_prompt_manifest():
    return {"prompt_format": HY_PROMPT_FORMAT, "translation_profile": PROFILE_ID,
            "source_context_max_characters": HY_CONTEXT_CHARACTERS}


def test_import_and_preparation_do_not_load_heavy_runtime(tmp_path):
    process = subprocess.run([
        sys.executable, "-c",
        "import sys; from fanyiji.specialized_training import prepare_training; "
        "prepare_training(sys.argv[1]); "
        "assert not any(x in sys.modules for x in ('torch', 'transformers', 'peft', 'bitsandbytes'))",
        str(tmp_path),
    ], capture_output=True, text=True)
    assert process.returncode == 0, process.stderr


def test_preparation_preserves_existing_files_and_never_claims_trained(tmp_path):
    old = tmp_path / "adapters" / "existing.gguf"
    old.parent.mkdir()
    old.write_bytes(b"preserve")
    first = prepare_training(tmp_path)
    second = prepare_training(tmp_path)
    assert first["directory"] != second["directory"]
    assert old.read_bytes() == b"preserve"
    assert first["phase"] == "prepared_not_trained"
    assert first["training_executed"] is False
    assert first["deployment_locked"] is True
    recipe = json.loads(Path(first["recipe_path"]).read_text(encoding="utf-8"))
    assert recipe["base_model_id"] == BASE_MODEL_ID
    assert recipe["eight_gb_fit_verified"] is False
    assert recipe["deployment_locked"] is True
    assert recipe["codex_training_allowed"] is False
    guide = Path(first["guide_path"]).read_text(encoding="utf-8")
    assert "24 GB" in guide and "未执行微调" in guide
    assert not (tmp_path / "base").exists()


def test_emitted_training_kit_runs_without_source_repository(tmp_path):
    prepared = prepare_training(tmp_path / "dedicated")
    script = Path(prepared["training_tool_path"])
    assert script.parent.name == "工具"
    assert (script.parent / "specialized_training.py").is_file()
    assert (script.parent / "livestream_prompt.py").is_file()
    process = subprocess.run([sys.executable, str(script), "--root", str(tmp_path / "other")],
                             cwd=tmp_path, capture_output=True, text=True, encoding="utf-8")
    assert process.returncode == 2, process.stderr
    result = json.loads(process.stdout)
    assert result["deployment_locked"] is True
    assert result["training_executed"] is False


def test_copied_training_module_loads_new_data_with_only_stdlib_and_keeps_deployment_locked(tmp_path):
    prepared = prepare_training(tmp_path / "dedicated")
    tool_directory = Path(prepared["training_tool_path"]).parent
    dataset = make_dataset(tmp_path / "dataset", train=[contextual_row(1, "train", "前" * 384, current_format=True)],
                           validation=[contextual_row(2, "validation", "", current_format=True)],
                           **current_prompt_manifest())
    process = subprocess.run([
        sys.executable, "-I", "-S", "-c",
        "import json, sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); "
        "import specialized_training as training; "
        "report, train, validation = training.load_training_dataset(Path(sys.argv[2])); "
        "assert report['translation_profile'] == 'japanese_livestream_v1'; "
        "assert training.DEPLOYMENT_LOCKED is True; "
        "assert not any(x in sys.modules for x in ('fanyiji', 'numpy', 'PySide6', 'torch', 'transformers', 'peft', 'bitsandbytes')); "
        "print(json.dumps({'deployment_locked': report['deployment_locked'], 'train_rows': len(train)}))",
        str(tool_directory), str(dataset),
    ], cwd=tmp_path, capture_output=True, text=True, encoding="utf-8")
    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout) == {"deployment_locked": True, "train_rows": 1}
    assert not (tmp_path / "dedicated" / "adapters").exists()


@pytest.mark.parametrize("change", [
    {"model_type": "llama"}, {"hidden_size": 4096}, {"num_hidden_layers": 16},
    {"architectures": ["RemoteModel"]}, {"auto_map": {"AutoModel": "remote.Model"}},
])
def test_base_rejects_wrong_architecture_and_custom_code(tmp_path, change):
    base = make_base(tmp_path / "base")
    value = json.loads((base / "config.json").read_text())
    value.update(change)
    (base / "config.json").write_text(json.dumps(value))
    with pytest.raises(TrainingPreparationError):
        validate_base_model(base)


def test_base_requires_hf_weights_instead_of_current_gguf(tmp_path):
    base = make_base(tmp_path / "base")
    (base / "model.safetensors").rename(base / "current.gguf")
    with pytest.raises(TrainingPreparationError, match="safetensors"):
        validate_base_model(base)


def test_sharded_base_rejects_index_traversal(tmp_path):
    base = make_base(tmp_path / "base")
    (base / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer": "../outside.safetensors"}}), encoding="utf-8",
    )
    (tmp_path / "outside.safetensors").write_bytes(b"outside")
    with pytest.raises(TrainingPreparationError):
        validate_base_model(base)


def test_valid_dataset_reports_disjoint_provenance_and_file_hashes(tmp_path):
    dataset = make_dataset(tmp_path / "dataset")
    summary, train, validation = load_training_dataset(dataset)
    assert summary["train_rows"] == len(train) == 1
    assert summary["validation_rows"] == len(validation) == 1
    assert summary["codex_rows"] == 0
    assert summary["sessions_disjoint"] is True
    assert len(summary["train_sha256"]) == 64
    assert summary["deployment_locked"] is True


@pytest.mark.parametrize("field,value", [
    ("origin", "codex"), ("origin", "cloud"), ("reviewed", False),
    ("reviewed", 1), ("source_verified", False), ("rights_confirmed", False),
    ("codex_derived", True), ("codex_derived", None), ("source_session", ""),
])
def test_row_provenance_cannot_be_bypassed_by_clean_manifest(tmp_path, field, value):
    row = make_row()
    row["provenance"][field] = value
    dataset = make_dataset(tmp_path / "dataset", train=[row])
    with pytest.raises(TrainingPreparationError):
        load_training_dataset(dataset)


@pytest.mark.parametrize("field,value", [
    ("format_version", 2), ("deployment_locked", False), ("codex_rows", 1),
    ("codex_rows", None), ("train_rows", 20), ("train_file", "../train.jsonl"),
    ("train_file", "C:/private/train.jsonl"), ("validation_file", "train.jsonl"),
])
def test_manifest_rejects_unlock_and_dataset_misdirection(tmp_path, field, value):
    dataset = make_dataset(tmp_path / "dataset", **{field: value})
    with pytest.raises(TrainingPreparationError):
        load_training_dataset(dataset)


@pytest.mark.parametrize("field", ["source_session", "source_group"])
def test_holdout_source_and_session_leakage_are_rejected(tmp_path, field):
    train, validation = make_row(), make_row(2, "validation")
    validation["provenance"][field] = train["provenance"][field]
    dataset = make_dataset(tmp_path / "dataset", train=[train], validation=[validation])
    with pytest.raises(TrainingPreparationError, match="泄漏"):
        load_training_dataset(dataset)


def test_fake_different_group_cannot_hide_normalized_duplicate_source(tmp_path):
    train, validation = make_row(), make_row(2, "validation")
    validation["messages"][0]["content"] = "原 文\t１"
    dataset = make_dataset(tmp_path / "dataset", train=[train], validation=[validation])
    with pytest.raises(TrainingPreparationError, match="重复原文"):
        load_training_dataset(dataset)


def test_duplicate_within_train_is_rejected(tmp_path):
    first, second = make_row(), make_row(3)
    second["messages"][0]["content"] = first["messages"][0]["content"]
    dataset = make_dataset(tmp_path / "dataset", train=[first, second])
    with pytest.raises(TrainingPreparationError, match="重复原文"):
        load_training_dataset(dataset)


def test_contextual_export_preserves_exact_teacher_input(tmp_path):
    train = make_row()
    train.update(raw_source="それは明日です。", source_context="会議はいつですか。")
    train["messages"][0]["content"] = _canonical_hy_prompt(
        train["raw_source"], train["source_context"], train["direction"],
    )
    dataset = make_dataset(tmp_path / "dataset", train=[train])
    _, rows, _ = load_training_dataset(dataset)
    assert "会議はいつですか。" in rows[0]["messages"][0]["content"]


def test_saved_legacy_export_loads_without_rewriting_or_injecting_new_rules(tmp_path):
    source, context = "会議は1日です。", "前" * LEGACY_HY_CONTEXT_CHARACTERS
    old_prompt = (f"〖背景信息：同一通道前文原文，仅用于理解，不翻译〗\n{context}\n\n"
                  "请结合背景信息将〖待翻译文本〗翻译为简体中文。只输出当前待翻译文本的译文，"
                  "不要重复翻译背景信息，不要解释。保留当前原文的人名、数字、币种、否定和条件关系；"
                  "背景只用于理解指代和话题，不得从背景添加、替换或推断当前原文没有的事实。"
                  f"\n\n〖待翻译文本〗\n{source}")
    train = contextual_row(1, "train", context)
    train["messages"][0]["content"] = old_prompt
    dataset = make_dataset(tmp_path / "legacy", train=[train],
                           validation=[contextual_row(2, "validation", "旧前文。")],
                           prompt_format=LEGACY_HY_PROMPT_FORMAT)
    before = {path.name: path.read_bytes() for path in dataset.iterdir()}
    report, rows, _ = load_training_dataset(dataset)
    assert _canonical_hy_prompt(source, context, "ja_to_zh") == old_prompt
    assert rows[0]["messages"][0]["content"] == old_prompt
    assert report["prompt_format"] == LEGACY_HY_PROMPT_FORMAT
    assert report["source_context_max_characters"] == LEGACY_HY_CONTEXT_CHARACTERS
    assert report["translation_profile"] is None
    assert before == {path.name: path.read_bytes() for path in dataset.iterdir()}


@pytest.mark.parametrize("current_format,length,valid", [
    (False, LEGACY_HY_CONTEXT_CHARACTERS, True),
    (False, LEGACY_HY_CONTEXT_CHARACTERS + 1, False),
    (True, HY_CONTEXT_CHARACTERS, True),
    (True, HY_CONTEXT_CHARACTERS + 1, False),
])
def test_context_budget_is_checked_against_the_declared_prompt_version(tmp_path, current_format, length, valid):
    train = contextual_row(1, "train", "前" * length, current_format=current_format)
    validation = contextual_row(2, "validation", "", current_format=current_format)
    metadata = current_prompt_manifest() if current_format else {"prompt_format": LEGACY_HY_PROMPT_FORMAT}
    dataset = make_dataset(tmp_path / "dataset", train=[train], validation=[validation], **metadata)
    if valid:
        report, rows, _ = load_training_dataset(dataset)
        assert len(rows[0]["source_context"]) == length
        assert report["source_context_max_characters"] == (
            HY_CONTEXT_CHARACTERS if current_format else LEGACY_HY_CONTEXT_CHARACTERS)
    else:
        with pytest.raises(TrainingPreparationError, match="上下文"):
            load_training_dataset(dataset)


def test_new_format_cannot_claim_a_legacy_prompt_as_current_production_input(tmp_path):
    train = contextual_row(1, "train", "前文", current_format=True)
    train["messages"][0]["content"] = _canonical_hy_prompt(train["raw_source"], train["source_context"], train["direction"])
    dataset = make_dataset(tmp_path / "dataset", train=[train],
                           validation=[contextual_row(2, "validation", "", current_format=True)],
                           **current_prompt_manifest())
    with pytest.raises(TrainingPreparationError, match="训练提示"):
        load_training_dataset(dataset)


@pytest.mark.parametrize("field,value", [
    ("prompt_format", LEGACY_HY_PROMPT_FORMAT), ("prompt_format", "unknown"),
    ("translation_profile", None), ("translation_profile", "unverified-profile"),
])
def test_row_cannot_override_manifest_prompt_version_or_profile(tmp_path, field, value):
    train = contextual_row(1, "train", "前文", current_format=True)
    train[field] = value
    dataset = make_dataset(tmp_path / "dataset", train=[train],
                           validation=[contextual_row(2, "validation", "", current_format=True)],
                           **current_prompt_manifest())
    with pytest.raises(TrainingPreparationError, match="提示"):
        load_training_dataset(dataset)


@pytest.mark.parametrize("change", [
    {"translation_profile": None}, {"translation_profile": "unverified-profile"},
    {"source_context_max_characters": HY_CONTEXT_CHARACTERS + 1},
    {"source_context_max_characters": float(HY_CONTEXT_CHARACTERS)}, {"prompt_format": "unknown"},
])
def test_new_manifest_cannot_change_profile_or_expand_the_versioned_context_budget(tmp_path, change):
    metadata = current_prompt_manifest()
    metadata.update(change)
    dataset = make_dataset(tmp_path / "dataset", train=[contextual_row(1, "train", "", current_format=True)],
                           validation=[contextual_row(2, "validation", "", current_format=True)], **metadata)
    with pytest.raises(TrainingPreparationError):
        load_training_dataset(dataset)


def test_legacy_manifest_cannot_raise_its_context_budget_to_the_new_limit(tmp_path):
    dataset = make_dataset(tmp_path / "dataset", train=[contextual_row(1, "train", "前" * 257)],
                           validation=[contextual_row(2, "validation", "")],
                           source_context_max_characters=HY_CONTEXT_CHARACTERS)
    with pytest.raises(TrainingPreparationError, match="预算"):
        load_training_dataset(dataset)


def test_new_prompt_format_requires_raw_source_for_exact_input_verification(tmp_path):
    train = contextual_row(1, "train", "", current_format=True)
    del train["raw_source"]
    dataset = make_dataset(tmp_path / "dataset", train=[train],
                           validation=[contextual_row(2, "validation", "", current_format=True)],
                           **current_prompt_manifest())
    with pytest.raises(TrainingPreparationError, match="采集原文"):
        load_training_dataset(dataset)


def test_context_cannot_hide_raw_source_leakage(tmp_path):
    train, validation = make_row(), make_row(2, "validation")
    for row, context in ((train, "会議はいつですか。"), (validation, "イベントはいつですか。")):
        row.update(raw_source="それは明日です。", source_context=context)
        row["messages"][0]["content"] = _canonical_hy_prompt(row["raw_source"], context, row["direction"])
    dataset = make_dataset(tmp_path / "dataset", train=[train], validation=[validation])
    with pytest.raises(TrainingPreparationError, match="重复原文"):
        load_training_dataset(dataset)


def test_raw_source_with_different_training_prompt_is_rejected(tmp_path):
    train = make_row()
    train.update(raw_source="それは明日です。", source_context="会議はいつですか。")
    dataset = make_dataset(tmp_path / "dataset", train=[train])
    with pytest.raises(TrainingPreparationError, match="提示"):
        load_training_dataset(dataset)


def test_real_store_export_is_accepted_with_context_and_codex_quarantine(tmp_path):
    from fanyiji.specialized_store import SpecializedStore

    store = SpecializedStore(tmp_path / "dedicated")
    try:
        for number in (1, 2):
            session, source = f"session-{number}", f"会議は{number}日です。"
            metadata = {"direction": "ja_to_zh", "provider": "codex", "model": "gpt"}
            store.record_event(session, {
                "kind": "original", "channel": "playback", "segment_id": "u1",
                "revision": 1, "text": source, "provisional": False,
            }, **metadata)
            store.record_event(session, {
                "kind": "translation", "channel": "playback", "segment_id": "u1",
                "revision": 1, "text": "云端候选", "context": "会議の日程です。",
            }, **metadata)
            sample = store.list_samples()[0]
            candidate = store.add_candidate(sample["id"], text=f"会议在{number}日。",
                                            origin="local", model="hy7")
            store.review_candidate(sample["id"], candidate, source_verified=True, rights_confirmed=True)
        exported = store.export_dataset()
        summary, train, validation = load_training_dataset(Path(exported["directory"]))
        assert summary["codex_rows"] == 0
        assert len(train) == len(validation) == 1
        assert all("会議の日程です。" in row["messages"][0]["content"] for row in train + validation)
        assert all(row["messages"][1]["content"] != "云端候选" for row in train + validation)
    finally:
        store.close()


def test_light_preflight_reports_sample_shortage_without_training(tmp_path, monkeypatch):
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "test")
    base = make_base(tmp_path / "base")
    dataset = make_dataset(tmp_path / "dataset")
    result = preflight(tmp_path, dataset=dataset, base_model=base)
    assert result["training_ready"] is False
    assert result["training_executed"] is False
    assert result["runtime_gpu_probe_executed"] is False
    assert result["eight_gb_fit_verified"] is False
    assert result["assets_ready"] is False
    assert any(str(MIN_TRAIN_ROWS) in blocker for blocker in result["blockers"])


def test_preflight_does_not_claim_gpu_fit_even_when_assets_sufficient(tmp_path, monkeypatch):
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "test")
    base = make_base(tmp_path / "base")
    dataset = make_dataset(tmp_path / "dataset",
                           train=[make_row(i) for i in range(MIN_TRAIN_ROWS)],
                           validation=[make_row(i + 10000, "validation")
                                       for i in range(MIN_VALIDATION_ROWS)])
    result = preflight(tmp_path, dataset=dataset, base_model=base)
    assert result["assets_ready"] is True
    assert result["training_ready"] is False
    assert result["deployment_locked"] is True
    assert "实际显存试验" in result["blockers"][0]


def test_explicit_training_rejects_unreviewed_or_insufficient_rows_before_imports(tmp_path):
    dataset = make_dataset(tmp_path / "dataset")
    before = "torch" in sys.modules
    with pytest.raises(TrainingPreparationError, match="样本不足"):
        train_local_adapter(tmp_path, dataset=dataset)
    assert ("torch" in sys.modules) == before
    assert not (tmp_path / "adapters").exists()


class TinyTokenizer:
    eos_token_id = 99

    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize):
        assert add_generation_prompt and tokenize
        assert len(messages) == 1 and messages[0]["role"] == "user"
        return [1, 2, 3]

    def encode(self, content, *, add_special_tokens):
        assert not add_special_tokens
        return [10] * len(content)


def test_only_assistant_tokens_receive_training_labels():
    row = make_row()
    tokens = tokenize_training_rows(TinyTokenizer(), [row])[0]
    assert tokens["labels"][:3] == [-100] * 3
    assert tokens["labels"][3:] == tokens["input_ids"][3:]
    assert tokens["input_ids"][-1] == 99
    assert len(tokens["attention_mask"]) == len(tokens["input_ids"])


def test_overlong_answer_is_rejected_instead_of_silently_truncated():
    row = make_row()
    row["messages"][1]["content"] = "长" * 513
    with pytest.raises(TrainingPreparationError, match="静默截断"):
        tokenize_training_rows(TinyTokenizer(), [row])


def test_cli_defaults_to_preflight_and_does_not_create_adapter(tmp_path):
    script = Path(__file__).resolve().parents[1] / "tools" / "train_specialized_model.py"
    process = subprocess.run([sys.executable, str(script), "--root", str(tmp_path)],
                             capture_output=True, text=True, encoding="utf-8")
    assert process.returncode == 2
    result = json.loads(process.stdout)
    assert result["training_executed"] is False
    assert result["deployment_locked"] is True
    assert not (tmp_path / "adapters").exists()


def test_training_recipe_is_fixed_locked_and_never_auto_downloads(tmp_path):
    recipe = training_recipe(tmp_path)
    assert recipe["requires_local_safetensors"] is True
    assert recipe["trust_remote_code"] is False
    assert recipe["local_files_only"] is True
    assert recipe["max_sequence_length"] == 512
    assert recipe["micro_batch_size"] == 1
    assert recipe["deployment_locked"] is True
