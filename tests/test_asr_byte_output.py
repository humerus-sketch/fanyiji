"""BPE bytes must survive token-wise native UTF-8 filtering."""
import base64
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def evaluator(monkeypatch):
    tool_directory = Path(__file__).resolve().parents[1] / "tools"
    monkeypatch.syspath_prepend(str(tool_directory))
    spec = importlib.util.spec_from_file_location("asr_byte_output_test", tool_directory / "evaluate_asr_candidates.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_split_japanese_character_survives_native_token_filter(evaluator, tmp_path):
    source = tmp_path / "tokens.txt"
    pieces = [b"\xe6\xb1", b"\x9a", "い映画".encode("utf-8"), b" "]
    source.write_text("\n".join(base64.b64encode(piece).decode() + " " + str(index)
                                for index, piece in enumerate(pieces)), encoding="ascii")
    table, _ = evaluator.byte_safe_token_table(source)
    native_tokens = [base64.b64decode(line.split()[0]) for line in table.read_text().splitlines()]
    output = b"".join(piece.decode("utf-8", "ignore").encode() for piece in native_tokens).decode()
    assert evaluator.restore_whisper_bytes(output) == "汚い映画 "
    assert b"".join(pieces).decode() == "汚い映画 "
    assert b"".join(piece.decode("utf-8", "ignore").encode() for piece in pieces).decode() == "い映画 "


def test_byte_markers_reject_unexpected_native_text(evaluator):
    with pytest.raises(ValueError, match="outside byte-safe"):
        evaluator.restore_whisper_bytes("__WBYTE_e698____WBYTE_a0__unexpected")


def test_existing_conflicting_derived_table_is_preserved(evaluator, tmp_path):
    source = tmp_path / "tokens.txt"
    source.write_text("YQ== 0\n", encoding="ascii")
    derived, _ = evaluator.byte_safe_token_table(source)
    derived.write_text("conflicting table", encoding="ascii")
    with pytest.raises(RuntimeError, match="preserved"):
        evaluator.byte_safe_token_table(source)
    assert derived.read_text() == "conflicting table"


def test_official_empty_token_placeholder_preserves_id(evaluator, tmp_path):
    source = tmp_path / "tokens.txt"
    source.write_text("= 50256\nYQ== 0\n", encoding="ascii")
    table, _ = evaluator.byte_safe_token_table(source)
    entries = [line.split() for line in table.read_text().splitlines()]
    assert [int(entry[1]) for entry in entries] == [50256, 0]
    output = "".join(base64.b64decode(entry[0]).decode() for entry in entries)
    assert evaluator.restore_whisper_bytes(output) == "a"
