"""Restricted, credential-free host protocol for the embedded Bilibili sender."""
from __future__ import annotations

import json
import math
import re
import unicodedata

from .translation_output import is_chinese_output

MIN_SEND_INTERVAL_SECONDS = 3.5
MAX_DANMAKU_CHARACTERS = 40
LEGACY_MAX_DANMAKU_CHARACTERS = 76
DEFAULT_MAX_WEIGHTED_LENGTH = 20
SEND_URL = "https://api.live.bilibili.com/msg/send"
NAV_URL = "https://api.bilibili.com/x/web-interface/nav"
BODY_FIELDS = ("bubble", "msg", "color", "mode", "fontsize", "rnd", "roomid", "csrf", "csrf_token")
_OPERATIONS = {"danmaku_send", "danmaku_check", "danmaku_login"}
_ERRORS = {"none", "wrong_room", "not_logged_in", "csrf_missing", "csrf_invalid", "request_timeout",
           "network_unknown", "invalid_response", "script_unavailable", "script_timeout", "window_closed",
           "navigation_changed", "queue_full", "browser_unavailable", "feature_unavailable", "text_too_long",
           "chat_unavailable"}
_CLASSES = {"empty", "f", "other"}


def valid_room(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[1-9][0-9]{0,19}", value) is not None


def room_url(room: str) -> str:
    if not valid_room(room):
        raise ValueError("invalid_room")
    return "https://live.bilibili.com/" + room


def valid_request_id(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value) is not None


def danmaku_utf16_length(text: str) -> int:
    """The app payload upper cap uses UTF-16 units, including outer brackets."""
    return sum(2 if ord(char) > 0xFFFF else 1 for char in text)


def danmaku_weighted_length(text: str) -> int:
    """Match the official live-room utilDanmuLength: printable ASCII costs one."""
    return sum(1 if 32 <= ord(char) <= 126 else 2 * (2 if ord(char) > 0xFFFF else 1) for char in text)


def sanitize_limits(value: dict) -> dict:
    raw = value.get("max_weighted_length")
    raw = raw if type(raw) is int and raw >= 6 else DEFAULT_MAX_WEIGHTED_LENGTH
    raw = min(MAX_DANMAKU_CHARACTERS * 2, raw)
    return {"max_weighted_length": raw, "max_characters": min(MAX_DANMAKU_CHARACTERS, raw // 2)}


def is_danmaku_punctuation(char: str) -> bool:
    return unicodedata.category(char).startswith("P") or char in "~～〜`｀|｜"


def valid_danmaku_text(value: object, *, max_characters: int = MAX_DANMAKU_CHARACTERS,
                       normalized_only: bool = False) -> bool:
    if type(max_characters) is not int or not 3 <= max_characters <= LEGACY_MAX_DANMAKU_CHARACTERS:
        return False
    if not isinstance(value, str) or not 3 <= len(value) <= max_characters:
        return False
    if danmaku_utf16_length(value) > max_characters:
        return False
    if not value.startswith("【") or not value.endswith("】") or not value[1:-1].strip():
        return False
    # Control/format characters and markup must never escape ASR/model output into public chat.
    return (not any(unicodedata.category(char).startswith("C") for char in value)
            and not any(char in value for char in "<>")
            and "【" not in value[1:-1] and "】" not in value[1:-1]
            and (not normalized_only or not any(is_danmaku_punctuation(char) for char in value[1:-1])))


def validate_danmaku_command(value: object) -> dict:
    if not isinstance(value, dict) or not isinstance(value.get("op"), str) or value.get("op") not in _OPERATIONS:
        raise ValueError("invalid_command")
    expected = {"op", "request_id", "room"}
    if value["op"] != "danmaku_login":
        expected.add("text")
    maximum = LEGACY_MAX_DANMAKU_CHARACTERS if value["op"] == "danmaku_check" else MAX_DANMAKU_CHARACTERS
    if (set(value) != expected or not valid_request_id(value["request_id"])
            or not valid_room(value["room"]) or ("text" in expected and not valid_danmaku_text(
                value["text"], max_characters=maximum, normalized_only=value["op"] == "danmaku_send"))):
        raise ValueError("invalid_command")
    # Legacy text may still need read-only observation. Only a new POST must
    # meet the Chinese-only rule, including alternate/manual sender callers.
    if value["op"] == "danmaku_send" and not is_chinese_output(value["text"]):
        raise ValueError("invalid_command")
    return dict(value)


def _number(value: object, maximum: float) -> float | None:
    if type(value) not in (int, float) or not 0 <= value <= maximum or not math.isfinite(value):
        return None
    return round(float(value), 3)


def _code(value: object) -> int | None:
    return value if type(value) is int and -2**31 <= value < 2**31 else None


def _occurrence_count(value: object) -> int | None:
    return value if type(value) is int and 0 <= value <= 10000 else None


def _error(value: dict) -> str:
    error = value.get("error")
    return error if isinstance(error, str) and error in _ERRORS else "invalid_response"


def sanitize_receipt(value: object) -> dict:
    """Drop every unrecognized field; never forward text, cookies, or raw responses."""
    value = value if isinstance(value, dict) else {}
    http = value.get("http_status")
    http = http if type(http) is int and 100 <= http <= 599 else None
    def classification(key):
        item = value.get(key)
        return item if isinstance(item, str) and item in _CLASSES else "other"
    result = {"http_status": http, "code": _code(value.get("code")),
              "msg_class": classification("msg_class"), "message_class": classification("message_class"),
              "error": _error(value),
              "before_count": _occurrence_count(value.get("before_count")),
              "request_elapsed_ms": _number(value.get("request_elapsed_ms"), 120000),
              "request_started_epoch_ms": _number(value.get("request_started_epoch_ms"), 10**15)}
    result.update(sanitize_limits(value))
    result["f"] = result["msg_class"] == "f" or result["message_class"] == "f"
    preflight_error = result["error"] in {"wrong_room", "not_logged_in", "csrf_missing", "csrf_invalid", "queue_full", "feature_unavailable", "text_too_long", "chat_unavailable"}
    # A complete browser closure explicitly reports a null POST start when
    # login/network preflight fails. Missing or malformed timestamps from a
    # lost script/transport are not proof that a POST was never dispatched.
    preflight_transport = (value.get("request_started_epoch_ms", "missing") is None
                           and result["error"] in {"network_unknown", "request_timeout", "invalid_response"})
    if (preflight_error or preflight_transport) and result["request_started_epoch_ms"] is None:
        result["outcome"] = "not_sent"
    elif result["error"] != "none":
        result["outcome"] = "result_unknown"
    elif result["f"]:
        result["outcome"] = "platform_f"
    elif (http is not None and not 200 <= http < 300) or result["code"] not in (None, 0):
        result["outcome"] = "platform_rejected"
    elif http is not None and 200 <= http < 300 and result["code"] == 0 and result["msg_class"] == result["message_class"] == "empty":
        result["outcome"] = "receipt_accepted"
    else:
        result["outcome"] = "result_unknown"
    return result


def sanitize_login(value: object) -> dict:
    value = value if isinstance(value, dict) else {}
    error = _error(value)
    code = _code(value.get("code"))
    return {"is_login": error == "none" and code == 0 and value.get("is_login") is True,
            "code": code, "error": error, **sanitize_limits(value)}


def sanitize_check(value: object) -> dict:
    value = value if isinstance(value, dict) else {}
    error = _error(value)
    matches = error == "none" and value.get("room_matches") is True
    ready = matches and value.get("chat_ready") is True
    count = _occurrence_count(value.get("occurrence_count")) if ready else None
    return {"room_matches": matches, "chat_ready": ready,
            "present": ready and value.get("present") is True and count is not None and count > 0,
            "occurrence_count": count, "error": error}


def _page_limits_script() -> str:
    # Official app.3f725... defaults baseInfoUser.danmakuLengthLimit to 20;
    # chat module 9649.f140... enforces it through utilDanmuLength, not maxlength.
    return r"""const pageLimits = () => {
      let raw = __DEFAULT_WEIGHTED_LENGTH__;
      try {
        for (const selector of ['#chat-control-panel-vm','.chat-control-panel']) {
          const element = document.querySelector(selector), vm = element && element.__vue__;
          const value = vm && vm.$store && vm.$store.getters && vm.$store.getters.baseInfoUser;
          const limit = value && value.danmakuLengthLimit;
          if (Number.isInteger(limit) && limit >= 6) {raw = Math.min(__MAX_WEIGHTED_LENGTH__,limit);break;}
        }
      } catch (_) {}
      return {max_weighted_length:raw,max_characters:Math.min(__MAX_CHARACTERS__,Math.floor(raw/2))};
    };
    const weightedLength = text => [...text].reduce((length,char) => {
      const point = char.codePointAt(0);
      return length + (point >= 32 && point <= 126 ? 1 : 2*char.length);
    },0);""".replace("__DEFAULT_WEIGHTED_LENGTH__", str(DEFAULT_MAX_WEIGHTED_LENGTH)).replace(
        "__MAX_WEIGHTED_LENGTH__", str(MAX_DANMAKU_CHARACTERS * 2)).replace(
        "__MAX_CHARACTERS__", str(MAX_DANMAKU_CHARACTERS))


def _page_occurrences_script() -> str:
    # Exact text nodes only: a previous matching line is the pre-POST baseline.
    # Official app.3f725... renders t.content in span.danmaku-item-right.
    return r"""const pageOccurrences = wanted => {
      try {
        const list = document.querySelector('#chat-history-list') || document.querySelector('.chat-history-list');
        if (!list) return {chat_ready:false,occurrence_count:null};
        const messages = list.querySelectorAll('.danmaku-item .danmaku-item-right, .chat-item .danmaku-item-right, .danmaku-item .danmaku-content, .chat-item .danmaku-content, .danmaku-item .text, .chat-item .text, .danmaku-content');
        const count = [...messages].filter(node => (node.innerText || node.textContent || '').trim() === wanted).length;
        return {chat_ready:true,occurrence_count:Math.min(10000,count)};
      } catch (_) {return {chat_ready:false,occurrence_count:null};}
    };"""


def send_script(room: str, text: str) -> str:
    validate_danmaku_command({"op": "danmaku_send", "request_id": "script", "room": room, "text": text})
    # Credentials are scoped to the browser closure. There is no Cookie/header export path.
    return r"""(async () => {
      const room = __ROOM_URL__, started = performance.now();
      __PAGE_LIMITS__
      __PAGE_OCCURRENCES__
      let limits = pageLimits();
      const text = __TEXT__;
      const base = {http_status:null,code:null,msg_class:'other',message_class:'other',error:'none'};
      let request_started_epoch_ms = null, request_started_page_ms = null, before_count = null;
      const finish = value => ({...base,...value,...limits,request_started_epoch_ms,before_count,
        request_elapsed_ms:request_started_page_ms === null ? null : performance.now()-request_started_page_ms});
      if (location.href !== room) return finish({error:'wrong_room'});
      const controller = new AbortController(), timer = setTimeout(() => controller.abort(),10000);
      const key = 'fanyiji.danmaku.lastRequestStart';
      const symbol = Symbol.for(key);
      try {
        const nav = await fetch(__NAV_URL__,{credentials:'include',signal:controller.signal});
        let login;
        try {login = await nav.json();} catch (_) {return finish({error:'invalid_response'});}
        if (!nav.ok || !login || login.code !== 0 || !login.data || login.data.isLogin !== true)
          return finish({error:'not_logged_in'});
        limits = pageLimits();
        if (weightedLength(text) > limits.max_weighted_length) return finish({error:'text_too_long'});
        const last = () => {
          let stored = 0;
          try {stored = Number(sessionStorage.getItem(key)) || 0;} catch (_) {}
          return Math.max(stored,Number(window[symbol]) || 0);
        };
        while (Date.now() < last() + 3500) {
          if (controller.signal.aborted) return finish({error:'request_timeout'});
          await new Promise(resolve => setTimeout(resolve,Math.min(100,last()+3500-Date.now())));
        }
        if (location.href !== room) return finish({error:'wrong_room'});
        limits = pageLimits();
        if (weightedLength(text) > limits.max_weighted_length) return finish({error:'text_too_long'});
        const match = document.cookie.match(/(?:^|;\s*)bili_jct=([^;]*)/);
        if (!match || !match[1]) return finish({error:'csrf_missing'});
        let csrf;
        try {csrf = decodeURIComponent(match[1]);} catch (_) {return finish({error:'csrf_invalid'});}
        if (!/^[a-f0-9]{32}$/i.test(csrf)) return finish({error:'csrf_invalid'});
        const body = new URLSearchParams({bubble:'0',msg:text,color:'16777215',mode:'1',fontsize:'25',
          rnd:String(Math.floor(Date.now()/1000)),roomid:__ROOM__,csrf,csrf_token:csrf});
        if (controller.signal.aborted) return finish({error:'request_timeout'});
        const observation = pageOccurrences(text);
        if (!observation.chat_ready || observation.occurrence_count === null)
          return finish({error:'chat_unavailable'});
        before_count = observation.occurrence_count;
        request_started_epoch_ms = Date.now();
        request_started_page_ms = performance.now();
        window[symbol] = request_started_epoch_ms;
        try {sessionStorage.setItem(key,String(request_started_epoch_ms));} catch (_) {}
        const response = await fetch(__SEND_URL__,{method:'POST',credentials:'include',body,signal:controller.signal});
        const http_status = response.status;
        let payload;
        try {payload = await response.json();} catch (_) {return finish({http_status,error:'invalid_response'});}
        if (!payload || typeof payload !== 'object' || Array.isArray(payload))
          return finish({http_status,error:'invalid_response'});
        const classify = value => value == null || value === '' ? 'empty' :
          (typeof value === 'string' && value.trim().toLowerCase() === 'f' ? 'f' : 'other');
        return finish({http_status,code:Number.isInteger(payload.code) ? payload.code : null,
          msg_class:classify(payload.msg),message_class:classify(payload.message)});
      } catch (error) {
        return finish({error:error && error.name === 'AbortError' ? 'request_timeout' : 'network_unknown'});
      } finally {clearTimeout(timer);}
    })()""".replace("__PAGE_LIMITS__", _page_limits_script()).replace("__PAGE_OCCURRENCES__", _page_occurrences_script()).replace("__ROOM_URL__", json.dumps(room_url(room))).replace("__NAV_URL__", json.dumps(NAV_URL)).replace("__SEND_URL__", json.dumps(SEND_URL)).replace("__ROOM__", json.dumps(room)).replace("__TEXT__", json.dumps(text, ensure_ascii=False))


def login_script(room: str) -> str:
    return """(async () => {
      %s
      if (location.href !== %s) return {code:null,is_login:false,error:'wrong_room',...pageLimits()};
      const controller = new AbortController(), timer = setTimeout(() => controller.abort(),10000);
      try {
        const response = await fetch(%s,{credentials:'include',signal:controller.signal});
        const value = await response.json();
        return {code:Number.isInteger(value.code) ? value.code : null,error:'none',...pageLimits(),
          is_login:response.ok && value.code === 0 && value.data && value.data.isLogin === true};
      } catch (error) {return {code:null,is_login:false,...pageLimits(),
        error:error && error.name === 'AbortError' ? 'request_timeout' : 'network_unknown'};}
      finally {clearTimeout(timer);}
    })()""" % (_page_limits_script(), json.dumps(room_url(room)), json.dumps(NAV_URL))


def presence_script(room: str, text: str) -> str:
    validate_danmaku_command({"op": "danmaku_check", "request_id": "script", "room": room, "text": text})
    # DOM receipt is a page-level observation, never proof of server broadcast.
    return """(() => {
      %s
      if (location.href !== %s) return {room_matches:false,chat_ready:false,present:false,occurrence_count:null,error:'wrong_room'};
      const wanted = %s;
      const result = pageOccurrences(wanted);
      return {room_matches:true,...result,present:result.occurrence_count > 0,error:'none'};
    })()""" % (_page_occurrences_script(), json.dumps(room_url(room)), json.dumps(text, ensure_ascii=False))
