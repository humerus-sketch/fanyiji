"""Bounded, offline advisory matching of literal community wordlists.

Matches are observations, not proof of a Bilibili rejection. This module never
generates replacement text or executes patterns supplied by a dictionary.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unicodedata


MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_ENTRIES = 100_000
MAX_SOURCES = 256
MAX_TERM_CHARACTERS = 128
MAX_TOTAL_TERM_CHARACTERS = 1_000_000
MAX_TRIE_NODES = 300_000
MAX_TEXT_CHARACTERS = 40_000
MAX_HITS = 100
_END = None
_ERROR = "本地检测词库无法读取或格式无效，请检查词库后重新加载"


class DanmakuWordlistError(ValueError):
    pass


def _string(value, maximum, *, empty=False, multiline=False):
    if not isinstance(value, str) or len(value) > maximum or (not value.strip() and not empty):
        raise DanmakuWordlistError("词库文字字段无效或超过限制")
    if any(unicodedata.category(char) in {"Cc", "Cf", "Cs"}
           and not (multiline and char in "\r\n\t") for char in value):
        raise DanmakuWordlistError("词库含有控制字符或隐藏字符")
    return value.strip()


def _room_scope(value):
    if not isinstance(value, str):
        raise DanmakuWordlistError("检测词直播间范围必须是 ASCII 数字房间号")
    value = value.strip()
    if value and (len(value) > 20 or not value.isascii() or not value.isdigit() or value.startswith("0")):
        raise DanmakuWordlistError("检测词直播间范围必须是 ASCII 正整数房间号，最多20位")
    return value


def _normalized_view(text):
    """Keep each normalized character's original half-open source span."""
    characters, offsets = [], []
    start = 0
    while start < len(text):
        end = start + 1
        while end < len(text):
            following = unicodedata.normalize("NFKC", text[end])
            if unicodedata.combining(text[end]) or (following and unicodedata.combining(following[0])):
                end += 1
                continue
            # NFKC also composes Hangul Jamo, whose combining class is zero.
            if following and (0x1161 <= ord(following[0]) <= 0x1175
                              or 0x11A8 <= ord(following[0]) <= 0x11C2):
                current = unicodedata.normalize("NFKC", text[start:end])
                previous, next_code = ord(current[-1]), ord(following[0])
                if ((0x1100 <= previous <= 0x1112 and 0x1161 <= next_code <= 0x1175)
                        or (0xAC00 <= previous <= 0xD7A3 and (previous - 0xAC00) % 28 == 0
                            and 0x11A8 <= next_code <= 0x11C2)):
                    end += 1
                    continue
            break
        cluster = unicodedata.normalize("NFKC", text[start:end]).casefold()
        for char in cluster:
            if char.isspace() or unicodedata.category(char).startswith("P"):
                continue
            characters.append(char)
            offsets.append((start, end))
        start = end
    return "".join(characters), offsets


def _normalized_term(text):
    return _normalized_view(text)[0]


def _ascii_word_character(char):
    normalized = unicodedata.normalize("NFKC", char).casefold()
    return bool(normalized) and all(part.isascii() and (part.isalnum() or part == "_") for part in normalized)


def _whole_word(text, view, offsets, start, end, word):
    if not (word.isascii() and all(_ascii_word_character(char) for char in word)):
        return True
    original_start, original_end = (offsets[start][0], offsets[end - 1][1]) if offsets is not None else (start, end)
    if ((original_start and _ascii_word_character(text[original_start - 1]))
            or (original_end < len(text) and _ascii_word_character(text[original_end]))):
        return False
    if offsets is not None:
        # Compatibility characters may expand to several letters. A substring
        # of that expansion is not an independently spoken English word.
        if start and _ascii_word_character(view[start - 1]):
            gap = text[offsets[start - 1][1]:original_start]
            if not any(char.isspace() for char in gap):
                return False
        if end < len(view) and _ascii_word_character(view[end]):
            gap = text[original_end:offsets[end][0]]
            if not any(char.isspace() for char in gap):
                return False
    return True


class _Trie:
    def __init__(self, entries, *, normalized=False, allowlist=()):
        self.root = {}
        self.longest = 0
        nodes = 1
        for index, row in enumerate(entries):
            if _normalized_term(row["term"]) in allowlist:
                continue
            term = _normalized_term(row["term"]) if normalized else row["term"]
            self.longest = max(self.longest, len(term))
            node = self.root
            for char in term:
                if char not in node:
                    nodes += 1
                    if nodes > MAX_TRIE_NODES:
                        raise DanmakuWordlistError("词库匹配索引超过内存限制")
                    node[char] = {}
                node = node[char]
            node.setdefault(_END, {}).setdefault(row.get("room", ""), []).append(index)

    def matches(self, text, room=""):
        for start in range(len(text)):
            node = self.root
            for end in range(start, min(len(text), start + self.longest)):
                node = node.get(text[end])
                if node is None:
                    break
                matches = node.get(_END, {})
                for scope in (("", room) if room else ("",)):
                    for index in matches.get(scope, ()):
                        yield index, start, end + 1


class DanmakuWordlist:
    def __init__(self, path):
        self.path = Path(path)
        self._state = {"version": 1, "sources": [], "entries": [], "allowlist": []}
        self._literal = self._normalized = None
        self._allow_literal = self._allow_normalized = None
        self._allowlist = set()
        self.status, self.error, self.revision = "no_wordlist", "", ""
        self.reload()

    @staticmethod
    def _validate(value):
        if not isinstance(value, dict) or set(value) - {"version", "sources", "entries", "allowlist"}:
            raise DanmakuWordlistError("词库只支持声明式文字字段，不支持脚本或正则")
        if type(value.get("version")) is not int or value["version"] != 1:
            raise DanmakuWordlistError("不支持的本地检测词库版本")
        sources, entries, allowlist = (value.get(key) for key in ("sources", "entries", "allowlist"))
        if (not isinstance(sources, list) or len(sources) > MAX_SOURCES
                or not isinstance(entries, list) or len(entries) > MAX_ENTRIES
                or not isinstance(allowlist, list) or len(allowlist) > MAX_ENTRIES):
            raise DanmakuWordlistError("词库记录数量无效或超过限制")
        normalized_sources, normalized_entries, normalized_allowlist = [], [], []
        identifiers, identities, allowed = set(), set(), set()
        source_fields = {"id", "name", "url", "revision", "license", "license_text"}
        for row in sources:
            if not isinstance(row, dict) or set(row) != source_fields:
                raise DanmakuWordlistError("词库来源格式无效")
            result = {key: _string(row[key], 100_000 if key == "license_text" else 4096,
                                   empty=key in {"revision", "license", "license_text"},
                                   multiline=key == "license_text") for key in source_fields}
            if result["id"] in identifiers:
                raise DanmakuWordlistError("词库包含重复来源")
            identifiers.add(result["id"])
            normalized_sources.append(result)
        total = 0
        for row in entries:
            fields = {"term", "category", "source_id"}
            if not isinstance(row, dict) or not fields <= set(row) or set(row) - fields - {"room"}:
                raise DanmakuWordlistError("检测词只支持字面匹配，不支持正则或脚本")
            result = {"term": _string(row["term"], MAX_TERM_CHARACTERS),
                      "category": _string(row["category"], 128),
                      "source_id": _string(row["source_id"], 4096), "room": _room_scope(row.get("room", ""))}
            if result["source_id"] not in identifiers or not _normalized_term(result["term"]):
                raise DanmakuWordlistError("检测词来源不存在或规范化后为空")
            total += len(result["term"])
            if total > MAX_TOTAL_TERM_CHARACTERS:
                raise DanmakuWordlistError("词库总字符数超过内存限制")
            identity = tuple(result[key] for key in ("term", "category", "source_id", "room"))
            if identity not in identities:
                identities.add(identity)
                normalized_entries.append(result)
        for term in allowlist:
            term = _string(term, MAX_TERM_CHARACTERS)
            normalized = _normalized_term(term)
            if not normalized:
                raise DanmakuWordlistError("白名单规范化后不能为空")
            if normalized not in allowed:
                allowed.add(normalized)
                normalized_allowlist.append(term)
        return {"version": 1, "sources": normalized_sources,
                "entries": normalized_entries, "allowlist": normalized_allowlist}

    def _compile(self, state):
        # Construct into locals so a failed replacement never installs a partial index.
        allowlist = {_normalized_term(term) for term in state["allowlist"]}
        literal = _Trie(state["entries"], allowlist=allowlist)
        normalized = _Trie(state["entries"], normalized=True, allowlist=allowlist)
        allowed_entries = [{"term": term} for term in state["allowlist"]]
        allow_literal = _Trie(allowed_entries)
        allow_normalized = _Trie(allowed_entries, normalized=True)
        revision = hashlib.sha256(json.dumps(state, ensure_ascii=False, sort_keys=True,
                                  separators=(",", ":")).encode("utf-8")).hexdigest()
        return literal, normalized, allow_literal, allow_normalized, allowlist, revision

    def reload(self):
        try:
            with self.path.open("rb") as stream:
                source = stream.read(MAX_FILE_BYTES + 1)
            digest = hashlib.sha256(source).hexdigest()
            if len(source) > MAX_FILE_BYTES:
                raise DanmakuWordlistError("词库文件超过大小限制")
            state = self._validate(json.loads(source.decode("utf-8")))
            literal, normalized, allow_literal, allow_normalized, allowlist, revision = self._compile(state)
        except FileNotFoundError:
            self.status, self.error, self.revision = "no_wordlist", "", ""
        except (OSError, UnicodeError, ValueError, TypeError, OverflowError, RecursionError):
            # A bad dictionary is advisory-only and must never stop subtitle delivery.
            self.status, self.error = "error", _ERROR
            self.revision = locals().get("digest", "unreadable")
        else:
            self._state = state
            self._literal, self._normalized, self._allowlist = literal, normalized, allowlist
            self._allow_literal, self._allow_normalized = allow_literal, allow_normalized
            self.status, self.error, self.revision = "ready", "", revision
            return self.info()
        self._state = {"version": 1, "sources": [], "entries": [], "allowlist": []}
        self._literal = self._normalized = None
        self._allow_literal = self._allow_normalized = None
        self._allowlist = set()
        return self.info()

    def info(self):
        return {"status": self.status, "entry_count": len(self._state["entries"]),
                "source_count": len(self._state["sources"]), "error": self.error, "revision": self.revision}

    def entries(self):
        return copy.deepcopy(self._state["entries"])

    def sources(self):
        return copy.deepcopy(self._state["sources"])

    def get_allowlist(self):
        return list(self._state["allowlist"])

    def set_allowlist(self, terms):
        if self.status != "ready":
            raise DanmakuWordlistError("请先加载有效的本地检测词库，再编辑白名单")
        state = copy.deepcopy(self._state)
        state["allowlist"] = terms
        state = self._validate(state)
        compiled = self._compile(state)
        serialized = json.dumps(state, ensure_ascii=False, indent=2) + "\n"
        if len(serialized.encode("utf-8")) > MAX_FILE_BYTES:
            raise DanmakuWordlistError("词库文件超过大小限制")
        temporary = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                    prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent, delete=False) as output:
                temporary = Path(output.name)
                output.write(serialized)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, self.path)
        except (OSError, UnicodeError):
            raise DanmakuWordlistError("无法保存检测白名单，原词库未修改") from None
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
        self._state = state
        (self._literal, self._normalized, self._allow_literal, self._allow_normalized,
         self._allowlist, self.revision) = compiled

    def _allowed_spans(self, text, normalized, offsets):
        # Only the greatest end at each original start is needed. Prefix maxima
        # then answer containment queries in O(1), without storing every match.
        ends = [0] * (len(text) + 1)
        for trie, view, mapping in ((self._allow_literal, text, None),
                                    (self._allow_normalized, normalized, offsets)):
            for index, start, end in trie.matches(view):
                term = self._state["allowlist"][index]
                if not _whole_word(text, view, mapping, start, end, _normalized_term(term)):
                    continue
                if mapping is not None:
                    start, end = mapping[start][0], mapping[end - 1][1]
                ends[start] = max(ends[start], end)
        maximum = 0
        for index, end in enumerate(ends):
            maximum = max(maximum, end)
            ends[index] = maximum
        return ends

    def detect(self, text, room=""):
        result = {"status": "checked", "engine": "local-wordlist",
                  "dictionary_revision": self.revision, "hits": [], "truncated": False,
                  "error": "", "summary_code": "no_hits",
                  "notes": ["non_official_wordlist", "failure_reason_unproven",
                            "no_hit_does_not_confirm_sendability"]}
        try:
            room = _room_scope(room)
        except DanmakuWordlistError:
            result.update(status="error", summary_code="dictionary_error", error="待检测直播间范围无效")
            return result
        if self.status != "ready":
            result.update(status="unavailable" if self.status == "no_wordlist" else "error",
                          summary_code="no_wordlist" if self.status == "no_wordlist" else "dictionary_error",
                          error=self.error)
            return result
        if not isinstance(text, str) or len(text) > MAX_TEXT_CHARACTERS:
            result.update(status="error", summary_code="dictionary_error", error="待检测内容无效或超过40000字符")
            return result
        normalized, offsets = _normalized_view(text)
        allowed_spans = self._allowed_spans(text, normalized, offsets)
        identities = set()
        for trie, view, match_type in ((self._literal, text, "literal"), (self._normalized, normalized, "normalized")):
            for index, start, end in trie.matches(view, room):
                row = self._state["entries"][index]
                if row["room"] not in {"", room}:
                    continue
                if _normalized_term(row["term"]) in self._allowlist:
                    continue
                word = _normalized_term(row["term"])
                if not _whole_word(text, view, offsets if match_type == "normalized" else None, start, end, word):
                    continue
                if match_type == "normalized":
                    start, end = offsets[start][0], offsets[end - 1][1]
                if allowed_spans[start] >= end:
                    continue
                identity = (row["term"], row["category"], row["source_id"], row["room"], start, end)
                if identity in identities:
                    continue
                identities.add(identity)
                if len(result["hits"]) == MAX_HITS:
                    result["truncated"] = True
                    break
                result["hits"].append({**row, "matched": text[start:end], "start": start,
                                       "end": end, "match_type": match_type})
            if result["truncated"]:
                break
        result["hits"].sort(key=lambda hit: (hit["start"], -(hit["end"] - hit["start"]), hit["term"]))
        if result["hits"]:
            result["summary_code"] = "advisory_hits"
        return result
