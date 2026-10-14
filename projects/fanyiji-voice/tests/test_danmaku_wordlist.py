"""Bounded offline matching; fixtures never read a real user dictionary."""
import json
from pathlib import Path

import pytest

from fanyiji import danmaku_wordlist
from fanyiji.danmaku_wordlist import DanmakuWordlist, DanmakuWordlistError


def write_dictionary(path, terms=("测试词",), *, allowlist=()):
    value = {"version": 1,
             "sources": [{"id": "test", "name": "Offline fixture", "url": "https://example.invalid/words",
                          "revision": "commit", "license": "MIT", "license_text": "fixture license\nline two"}],
             "entries": [{"term": term, "category": "test", "source_id": "test"} for term in terms],
             "allowlist": list(allowlist)}
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return value


def test_missing_dictionary_is_explicitly_unavailable(tmp_path):
    detector = DanmakuWordlist(tmp_path / "missing.json")
    assert detector.info() == {"status": "no_wordlist", "entry_count": 0, "source_count": 0,
                               "error": "", "revision": ""}
    result = detector.detect("正常内容")
    assert result["status"] == "unavailable"
    assert result["summary_code"] == "no_wordlist"
    assert result["hits"] == []
    assert detector.entries() == detector.sources() == detector.get_allowlist() == []


def test_literal_and_normalized_matching_return_exact_original_spans(tmp_path):
    path = tmp_path / "dictionary.json"
    write_dictionary(path, ("测试词", "BAD", "ffi", "café"))
    detector = DanmakuWordlist(path)
    text = "【测试词 测 试，词 ＢＡＤ ﬃ Cafe\u0301】"
    result = detector.detect(text)
    assert result["status"] == "checked" and result["summary_code"] == "advisory_hits"
    assert [(row["term"], row["matched"], row["match_type"]) for row in result["hits"]] == [
        ("测试词", "测试词", "literal"), ("测试词", "测 试，词", "normalized"),
        ("BAD", "ＢＡＤ", "normalized"), ("ffi", "ﬃ", "normalized"), ("café", "Cafe\u0301", "normalized")]
    assert all(text[row["start"]:row["end"]] == row["matched"] for row in result["hits"])
    assert "non_official_wordlist" in result["notes"]
    assert "failure_reason_unproven" in result["notes"]


@pytest.mark.parametrize("text", ["class", "CLASS", "classes", "ｃｌass", "ｃｌａｓｓ", "c.l.a.s.s",
                                  "classify", "_ass", "ass1", "1ass"])
def test_ascii_word_boundary_does_not_match_ass_inside_class(tmp_path, text):
    path = tmp_path / "dictionary.json"
    write_dictionary(path, ("ass",))
    assert DanmakuWordlist(path).detect(text)["hits"] == []


def test_ascii_whole_words_and_punctuation_normalization(tmp_path):
    path = tmp_path / "dictionary.json"
    write_dictionary(path, ("ass", "badword"))
    assert [row["matched"] for row in DanmakuWordlist(path).detect("an ass bad.word ASS") ["hits"]] == [
        "ass", "bad.word", "ASS"]


def test_english_boundaries_do_not_split_same_original_character_expansion(tmp_path):
    path = tmp_path / "dictionary.json"
    write_dictionary(path, ("ff", "s", "ss"))
    detector = DanmakuWordlist(path)
    assert detector.detect("ﬄ")["hits"] == []
    hits = detector.detect("ß")["hits"]
    assert len(hits) == 1 and hits[0]["term"] == "ss"
    assert hits[0]["matched"] == "ß" and (hits[0]["start"], hits[0]["end"]) == (0, 1)
    assert detector.detect("S dog")["hits"][0]["term"] == "s"


@pytest.mark.parametrize("term,text", [("ガ", "ｶﾞ"), ("각", "각")])
def test_nfkc_composition_including_compatibility_combining_marks_preserves_span(tmp_path, term, text):
    path = tmp_path / "dictionary.json"
    write_dictionary(path, (term,))
    hit = DanmakuWordlist(path).detect(text)["hits"][0]
    assert hit["term"] == term and hit["matched"] == text
    assert (hit["start"], hit["end"]) == (0, len(text))


def test_allowlist_suppresses_only_matches_fully_inside_literal_or_normalized_phrase(tmp_path):
    path = tmp_path / "dictionary.json"
    write_dictionary(path, ("咖啡因",), allowlist=("咖啡因代谢",))
    detector = DanmakuWordlist(path)
    text = "咖啡因代谢 咖 啡 因 代 谢 咖啡因"
    hits = detector.detect(text)["hits"]
    assert len(hits) == 1 and hits[0]["matched"] == "咖啡因"
    assert hits[0]["start"] == text.rfind("咖啡因")


def test_overlapping_allowlist_phrases_do_not_union_into_larger_permission(tmp_path):
    path = tmp_path / "dictionary.json"
    write_dictionary(path, ("乙丙丁",), allowlist=("甲乙丙", "丙丁戊"))
    assert DanmakuWordlist(path).detect("甲乙丙丁戊")["hits"][0]["term"] == "乙丙丁"


def test_allowlist_suppresses_normalized_term_and_is_persisted_atomically(tmp_path):
    path = tmp_path / "dictionary.json"
    write_dictionary(path, ("测试词", "BAD"))
    detector = DanmakuWordlist(path)
    revision = detector.revision
    detector.set_allowlist(["测 试词", "ＢＡＤ", "bad"])
    assert detector.revision != revision
    assert detector.get_allowlist() == ["测 试词", "ＢＡＤ"]
    assert detector.detect("测试词 BAD ＢＡＤ")["hits"] == []
    assert DanmakuWordlist(path).get_allowlist() == detector.get_allowlist()
    detector.get_allowlist().append("outside")
    detector.entries()[0]["term"] = "outside"
    detector.sources()[0]["name"] = "outside"
    assert detector.entries()[0]["term"] == "测试词"
    assert detector.sources()[0]["name"] == "Offline fixture"


def test_allowlist_write_failure_keeps_disk_index_and_revision(tmp_path, monkeypatch):
    path = tmp_path / "dictionary.json"
    write_dictionary(path)
    detector = DanmakuWordlist(path)
    before, revision = path.read_bytes(), detector.revision
    def fail_replace(*args):
        raise PermissionError("test locked")
    monkeypatch.setattr(danmaku_wordlist.os, "replace", fail_replace)
    with pytest.raises(DanmakuWordlistError, match="原词库未修改"):
        detector.set_allowlist(["测试词"])
    assert path.read_bytes() == before and detector.revision == revision
    assert detector.get_allowlist() == []
    assert detector.detect("测试词")["hits"]
    assert not list(tmp_path.glob("*.tmp"))


def test_reloads_only_on_explicit_request(tmp_path):
    path = tmp_path / "dictionary.json"
    write_dictionary(path)
    detector = DanmakuWordlist(path)
    first_index = detector._literal
    write_dictionary(path, ("新词",))
    assert detector.detect("测试词")["hits"] and detector._literal is first_index
    detector.reload()
    assert detector._literal is not first_index
    assert detector.detect("测试词")["hits"] == []
    assert detector.detect("新词")["hits"]


def test_invalid_dictionary_is_advisory_error_with_no_raw_path(tmp_path):
    path = tmp_path / "dictionary.json"
    path.write_text("{secret path malformed", encoding="utf-8")
    detector = DanmakuWordlist(path)
    result = detector.detect("测试词")
    assert result["status"] == "error" and result["summary_code"] == "dictionary_error"
    assert str(path) not in result["error"] and "secret" not in result["error"]
    assert detector.info()["entry_count"] == 0


@pytest.mark.parametrize("change", [
    lambda data: data.update(version=True),
    lambda data: data.update(script="arbitrary code"),
    lambda data: data["entries"][0].update(pattern=".*"),
    lambda data: data["entries"][0].update(term="\u202etest"),
    lambda data: data["entries"][0].update(term="，"),
    lambda data: data["entries"][0].update(source_id="missing"),
    lambda data: data["entries"][0].update(term="a" * 129),
    lambda data: data.update(allowlist="wrong type"),
])
def test_invalid_structure_and_executable_pattern_fields_are_rejected(tmp_path, change):
    path = tmp_path / "dictionary.json"
    data = write_dictionary(path)
    change(data)
    path.write_text(json.dumps(data), encoding="utf-8")
    assert DanmakuWordlist(path).info()["status"] == "error"


def test_regular_expression_metacharacters_are_not_executed(tmp_path):
    path = tmp_path / "dictionary.json"
    write_dictionary(path, ("[abc]测试",))
    detector = DanmakuWordlist(path)
    assert detector.detect("a测试")["hits"] == []
    assert detector.detect("[abc]测试")["hits"][0]["match_type"] == "literal"


def test_match_results_are_bounded_and_report_truncation(tmp_path):
    path = tmp_path / "dictionary.json"
    write_dictionary(path, ("词",))
    detector = DanmakuWordlist(path)
    result = detector.detect("词" * 101)
    assert len(result["hits"]) == 100 and result["truncated"]
    assert not detector.detect("词" * 100)["truncated"]


def test_file_entry_trie_and_input_limits_fail_safely(tmp_path, monkeypatch):
    path = tmp_path / "dictionary.json"
    write_dictionary(path, ("第一词", "第二词"))
    monkeypatch.setattr(danmaku_wordlist, "MAX_FILE_BYTES", 8)
    assert DanmakuWordlist(path).info()["status"] == "error"
    monkeypatch.setattr(danmaku_wordlist, "MAX_FILE_BYTES", 16 * 1024 * 1024)
    monkeypatch.setattr(danmaku_wordlist, "MAX_ENTRIES", 1)
    assert DanmakuWordlist(path).info()["status"] == "error"
    monkeypatch.setattr(danmaku_wordlist, "MAX_ENTRIES", 100_000)
    monkeypatch.setattr(danmaku_wordlist, "MAX_TRIE_NODES", 2)
    assert DanmakuWordlist(path).info()["status"] == "error"
    monkeypatch.setattr(danmaku_wordlist, "MAX_TRIE_NODES", 300_000)
    detector = DanmakuWordlist(path)
    assert detector.detect("字" * 40_001)["status"] == "error"
    assert detector.detect(None)["status"] == "error"


def test_no_hits_is_not_a_claim_of_sendability(tmp_path):
    path = tmp_path / "dictionary.json"
    write_dictionary(path)
    result = DanmakuWordlist(path).detect("不过 谢谢 恭喜获得评价 谢谢")
    assert result["summary_code"] == "no_hits" and result["hits"] == []
    assert "no_hit_does_not_confirm_sendability" in result["notes"]


def test_room_scoped_observation_is_not_reported_as_global_word(tmp_path):
    path = tmp_path / "dictionary.json"
    payload = write_dictionary(path, ("评价", "全局词"))
    payload["entries"][0]["room"] = "1698225"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    detector = DanmakuWordlist(path)
    text = "评价 全局词"
    hits = detector.detect(text, "1698225")["hits"]
    assert [(hit["term"], hit["room"]) for hit in hits] == [("评价", "1698225"), ("全局词", "")]
    assert [hit["term"] for hit in detector.detect(text, "21399689")["hits"]] == ["全局词"]
    assert [hit["term"] for hit in detector.detect(text)["hits"]] == ["全局词"]
    assert detector.entries()[0]["room"] == "1698225"


def test_old_global_dictionary_and_multiple_scopes_remain_compatible(tmp_path):
    path = tmp_path / "dictionary.json"
    payload = write_dictionary(path, ("测试词",))
    payload["entries"].extend([{**payload["entries"][0], "room": "1698225"},
                               {**payload["entries"][0], "room": "21399689"}])
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    detector = DanmakuWordlist(path)
    assert detector.info()["entry_count"] == 3
    assert [hit["room"] for hit in detector.detect("测试词", "1698225")["hits"]] == ["", "1698225"]
    assert detector.detect("测 试 词", "1698225")["hits"][1]["room"] == "1698225"
    detector.set_allowlist(["测试词"])
    assert detector.detect("测试词", "1698225")["hits"] == []


def test_invalid_entry_room_scope_is_rejected_without_silently_globalizing(tmp_path):
    path = tmp_path / "dictionary.json"
    for room in (None, 1698225, "0", "01", "-1", "１２３", "1x", "1" * 21):
        payload = write_dictionary(path)
        payload["entries"][0]["room"] = room
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        assert DanmakuWordlist(path).info()["status"] == "error"
    write_dictionary(path)
    assert DanmakuWordlist(path).detect("测试词", "１２３")["status"] == "error"
