"""Lossless ASCII transport for generated BPE byte pieces; no native inference."""
import base64
import hashlib
from pathlib import Path

import pytest

from fanyiji.whisper_tokens import prepare_whisper_tokens, restore_whisper_bytes


def original_table(path, pieces):
    raw = "\n".join(("=" if piece == b"" else base64.b64encode(piece).decode()) + " " + str(token_id)
                    for piece, token_id in pieces).encode("ascii") + b"\n"
    path.write_bytes(raw)
    return raw


def markers(table):
    return "".join(base64.b64decode(line.split()[0]).decode("ascii")
                   for line in table.read_text(encoding="ascii").splitlines())


def test_multibyte_fragments_roundtrip_and_empty_token_ids_unchanged(tmp_path):
    source = tmp_path / "tokens.txt"
    pieces = [(b"\xe6\xb1", 12800), (b"\x9a", 248), ("い映画".encode(), 7),
              (b" ", 11), (b"", 50256), ("🙂".encode()[:2], 50), ("🙂".encode()[2:], 51)]
    raw = original_table(source, pieces)
    table = prepare_whisper_tokens(source, expected_sha256=hashlib.sha256(raw).hexdigest())
    assert restore_whisper_bytes(markers(table.path)) == "汚い映画 🙂"
    assert source.read_bytes() == raw
    assert [int(line.split()[1]) for line in table.path.read_text().splitlines()] == [token_id for _, token_id in pieces]
    assert table.source_sha256 == hashlib.sha256(raw).hexdigest()
    assert table.derived_sha256 == hashlib.sha256(table.path.read_bytes()).hexdigest()


def test_unchanged_derived_cache_is_reused_without_writing(tmp_path, monkeypatch):
    source = tmp_path / "tokens.txt"
    raw = original_table(source, [(b"a", 0)])
    digest = hashlib.sha256(raw).hexdigest()
    table = prepare_whisper_tokens(source, expected_sha256=digest)
    before = table.path.stat().st_mtime_ns
    original_open = Path.open
    def read_only(path, *args, **kwargs):
        mode = args[0] if args else kwargs.get("mode", "r")
        if any(char in mode for char in "wax+"):
            raise AssertionError("缓存未变，不应重新写文件")
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", read_only)
    reused = prepare_whisper_tokens(source, expected_sha256=digest)
    assert reused == table and reused.path.stat().st_mtime_ns == before


def test_changed_source_invalidates_old_key_and_fixed_sha_rejects_it(tmp_path):
    source = tmp_path / "tokens.txt"
    old_raw = original_table(source, [(b"a", 0)])
    old_sha = hashlib.sha256(old_raw).hexdigest()
    old = prepare_whisper_tokens(source, expected_sha256=old_sha)
    new_raw = original_table(source, [(b"b", 0)])
    with pytest.raises(ValueError, match="原词表 SHA-256"):
        prepare_whisper_tokens(source, expected_sha256=old_sha)
    new = prepare_whisper_tokens(source, expected_sha256=hashlib.sha256(new_raw).hexdigest())
    assert new.path != old.path and old.path.exists()
    assert restore_whisper_bytes(markers(new.path)) == "b"
    assert source.read_bytes() == new_raw


@pytest.mark.parametrize("same_length", [True, False])
def test_corrupt_derived_cache_is_detected_and_preserved(tmp_path, same_length):
    source = tmp_path / "tokens.txt"
    original_table(source, [(b"a", 0)])
    table = prepare_whisper_tokens(source)
    damaged = (b"?" + table.path.read_bytes()[1:]) if same_length else b"damaged cached table"
    table.path.write_bytes(damaged)
    with pytest.raises(ValueError, match="派生词表校验失败"):
        prepare_whisper_tokens(source)
    assert table.path.read_bytes() == damaged


def test_speech_that_resembles_protocol_is_decoded_only_once(tmp_path):
    source = tmp_path / "tokens.txt"
    text = "正常原文 __WBYTE_e6b19a__ 不应再次解码"
    original_table(source, [(text.encode(), 0)])
    table = prepare_whisper_tokens(source)
    assert restore_whisper_bytes(markers(table.path)) == text


@pytest.mark.parametrize("text", ["normal text", "__WBYTE_e6b1____WBYTE_9a__tail",
                                  "__WBYTE_E6B19A__", "__WBYTE_0__", "__WBYTE_e6b19a_",
                                  " __WBYTE_61__", "__WBYTE_zz__"])
def test_invalid_or_incomplete_marker_protocol_is_rejected(text):
    with pytest.raises(ValueError, match="输出标记无法解析，识别失败"):
        restore_whisper_bytes(text)


def test_incomplete_utf8_is_visible_not_guessed_or_dropped():
    assert restore_whisper_bytes("__WBYTE_e6b1__") == "\ufffd"
    assert restore_whisper_bytes("__WBYTE_61____WBYTE_e6b1__") == "a\ufffd"
    assert restore_whisper_bytes("") == ""


@pytest.mark.parametrize("raw", [b"not-base64 0\n", b"YQ== zero\n", b"= 0\n", b"YQ== 0\nYg== 0\n"])
def test_invalid_source_table_is_rejected_before_cache_write(tmp_path, raw):
    source = tmp_path / "tokens.txt"
    source.write_bytes(raw)
    with pytest.raises(ValueError, match="原词表"):
        prepare_whisper_tokens(source)
    assert list(tmp_path.glob("bytes-v1-*")) == [] and source.read_bytes() == raw
