"""Offline protocol and browser-JavaScript tests; no external requests are sent."""
import json
import shutil
import subprocess

import pytest

from fanyiji.danmaku_protocol import (
    BODY_FIELDS, MIN_SEND_INTERVAL_SECONDS, MAX_DANMAKU_CHARACTERS, SEND_URL, is_danmaku_punctuation,
    danmaku_utf16_length, danmaku_weighted_length, login_script, presence_script, room_url, sanitize_check, sanitize_login,
    sanitize_receipt, send_script, valid_danmaku_text, validate_danmaku_command,
)


def command(op="danmaku_send", **extra):
    value = {"op": op, "request_id": "req_123-abc", "room": "1698225"}
    if op != "danmaku_login":
        value["text"] = "【测试译文】"
    return {**value, **extra}


@pytest.mark.parametrize("op", ["danmaku_send", "danmaku_check", "danmaku_login"])
def test_only_three_restricted_operations(op):
    assert validate_danmaku_command(command(op)) == command(op)


@pytest.mark.parametrize("changes", [
    {"op": "eval"}, {"op": []}, {"request_id": "../"}, {"request_id": "a" * 65},
    {"request_id": "x\n"}, {"request_id": None}, {"room": 1698225}, {"room": "0"},
    {"room": "01"}, {"room": "1?x=1"}, {"room": "１"}, {"room": "1" * 21},
    {"text": "hello"}, {"text": "【】"}, {"text": "【  】"}, {"text": "【\n】"},
    {"text": "【a\u200b】"}, {"text": "【None<asr_text>】"}, {"text": "【【nested】】"},
    {"text": "【" + "中" * 75 + "】"}, {"cookie": "secret"}, {"script": "alert(1)"},
    {"url": "https://evil.test"}, {"text": ["x"]},
])
def test_invalid_commands_cannot_control_urls_scripts_or_credentials(changes):
    with pytest.raises(ValueError, match="invalid_command"):
        validate_danmaku_command(command(**changes))


def test_text_limit_includes_brackets_and_browser_utf16_units():
    assert MAX_DANMAKU_CHARACTERS == 40
    assert MIN_SEND_INTERVAL_SECONDS == 3.5
    assert valid_danmaku_text("【" + "🙂" * 19 + "】")
    assert danmaku_utf16_length("【" + "🙂" * 19 + "】") == 40
    assert danmaku_utf16_length("【" + "🙂" * 19 + "中】") == 41
    assert not valid_danmaku_text("【" + "🙂" * 19 + "中】")
    assert valid_danmaku_text("【" + "中" * 38 + "】")
    assert not valid_danmaku_text("【" + "中" * 39 + "】")
    assert not valid_danmaku_text("【\ud800】")
    assert not valid_danmaku_text("【\udfff】")
    assert not valid_danmaku_text("【\ud83d\ude42】")


def test_only_readonly_check_accepts_legacy_76_unit_and_punctuation_payloads():
    legacy = "【" + "中" * 37 + "，" + "中" * 36 + "】"
    assert danmaku_utf16_length(legacy) == 76
    assert valid_danmaku_text(legacy, max_characters=76)
    assert validate_danmaku_command(command("danmaku_check", text=legacy))["text"] == legacy
    assert presence_script("1698225", legacy)
    with pytest.raises(ValueError):
        validate_danmaku_command(command(text=legacy))
    with pytest.raises(ValueError):
        validate_danmaku_command(command("danmaku_check", text="【"+"中"*75+"】"))


@pytest.mark.parametrize("char", list("，。！？、；：‘’\"'().-_~～〜`｀|｜"))
def test_new_send_cannot_contain_body_punctuation_but_legacy_check_can(char):
    text = "【译文" + char + "后续】"
    assert is_danmaku_punctuation(char)
    assert valid_danmaku_text(text)
    assert not valid_danmaku_text(text, normalized_only=True)
    assert validate_danmaku_command(command("danmaku_check",text=text))["text"] == text
    with pytest.raises(ValueError):
        validate_danmaku_command(command(text=text))


def test_new_send_accepts_normalized_spaces_and_preserves_outer_brackets():
    text = "【第一句 第二句 + 123】"
    assert valid_danmaku_text(text, normalized_only=True)
    assert validate_danmaku_command(command(text=text))["text"] == text


@pytest.mark.parametrize("body", [
    "首先 大家请冷静一点 レレン 谢谢你的留言",
    "谢谢 クちゃん也 谢谢你的留言", "谢谢 ＡＢＣ 的留言", "游戏进入 DLC 关卡",
    "这是ﾚﾚﾝ的留言", "这是韩语한국", "这是俄语Привет", "这是表情🙂",
])
def test_foreign_text_cannot_generate_a_post_but_remains_readonly_checkable(body):
    text = "【" + body + "】"
    assert valid_danmaku_text(text)
    with pytest.raises(ValueError, match="invalid_command"):
        validate_danmaku_command(command(text=text))
    with pytest.raises(ValueError, match="invalid_command"):
        send_script("1698225", text)
    assert validate_danmaku_command(command("danmaku_check", text=text))["text"] == text
    assert SEND_URL not in presence_script("1698225", text)


def test_chinese_names_and_digits_can_generate_send_script():
    text = "【首先大家冷静一点 雷伦谢谢留言 第2局】"
    assert validate_danmaku_command(command(text=text))["text"] == text
    assert "method:'POST'" in send_script("1698225", text)


@pytest.mark.parametrize("maximum", [None, True, 2, 77, "40", 40.0])
def test_validation_rejects_invalid_explicit_caps(maximum):
    assert not valid_danmaku_text("【译文】",max_characters=maximum)


def receipt(**extra):
    return {"http_status": 200, "code": 0, "msg_class": "empty", "message_class": "empty",
            "error": "none", "request_elapsed_ms": 100, "request_started_epoch_ms": 1700000000000, **extra}


@pytest.mark.parametrize("changes,outcome", [
    ({}, "receipt_accepted"), ({"code": 10030}, "platform_rejected"),
    ({"http_status": 403}, "platform_rejected"), ({"msg_class": "f"}, "platform_f"),
    ({"message_class": "f"}, "platform_f"), ({"msg_class": "other"}, "result_unknown"),
    ({"code": False}, "result_unknown"), ({"http_status": True}, "result_unknown"),
    ({"error": "navigation_changed"}, "result_unknown"), ({"error": "script_timeout"}, "result_unknown"),
    ({"error": "wrong_room", "request_started_epoch_ms": None}, "not_sent"),
    ({"error": "not_logged_in", "request_started_epoch_ms": None}, "not_sent"),
    ({"error": "feature_unavailable", "request_started_epoch_ms": None}, "not_sent"),
    ({"error": "chat_unavailable", "request_started_epoch_ms": None}, "not_sent"),
    ({"error": "chat_unavailable"}, "result_unknown"),
])
def test_receipt_does_not_confuse_acceptance_and_actual_visibility(changes, outcome):
    assert sanitize_receipt(receipt(**changes))["outcome"] == outcome


@pytest.mark.parametrize("value", [None, [], "secret", {"error": []}, {"code": "0"},
    {"request_elapsed_ms": float("inf")}, {"request_elapsed_ms": 10**1000},
    {"request_started_epoch_ms": float("nan")}, {"request_started_epoch_ms": True}])
def test_sanitizer_handles_malformed_values(value):
    result = sanitize_receipt(value)
    assert result["outcome"] == "result_unknown"
    assert "secret" not in json.dumps(result)


def test_no_sensitive_fields_escape_any_result():
    extra = {"cookie": "private", "csrf": "private", "uid": "private", "text": "private", "response": "private"}
    for sanitizer in (sanitize_receipt, sanitize_login, sanitize_check):
        assert "private" not in json.dumps(sanitizer({**receipt(), **extra}))
    assert not sanitize_login({"error": "none", "code": False, "is_login": True})["is_login"]
    assert not sanitize_login({"error": "request_timeout", "code": 0, "is_login": True})["is_login"]
    assert not sanitize_check({"error": "wrong_room", "room_matches": True, "chat_ready": True, "present": True, "occurrence_count": 1})["present"]
    assert sanitize_check({"error": "none", "room_matches": True, "chat_ready": True, "present": True, "occurrence_count": 1})["present"]


@pytest.mark.parametrize("limit,weighted,characters", [
    (None,20,10), (False,20,10), (0,20,10), ("40",20,10), (40,40,20),
    (76,76,38), (80,80,40), (152,80,40), (1000,80,40), (10**1000,80,40),
])
def test_live_limits_are_sanitized_without_trusting_client_character_claim(limit, weighted, characters):
    for sanitizer in (sanitize_login, sanitize_receipt):
        result = sanitizer({"error":"none","code":0,"is_login":True,"max_weighted_length":limit,
                            "max_characters":999,"uid":"private"})
        assert result["max_weighted_length"] == weighted
        assert result["max_characters"] == characters
        assert "private" not in json.dumps(result)


def test_weighted_length_matches_official_ascii_and_nonascii_ranges():
    assert danmaku_weighted_length("abc 123!") == 8
    assert danmaku_weighted_length("【中】") == 6
    assert danmaku_weighted_length("【🙂】") == 8
    assert danmaku_weighted_length("【" + "中"*8 + "】") == 20


@pytest.mark.parametrize("value", [None, False, True, -1, 10001, 1.0, "1", float("nan"), float("inf"), [], 10**1000])
def test_occurrence_counts_are_strict_bounded_integers(value):
    assert sanitize_receipt(receipt(before_count=value))["before_count"] is None
    check = sanitize_check({"error":"none", "room_matches":True, "chat_ready":True,
                            "present":True, "occurrence_count":value})
    assert check["occurrence_count"] is None
    assert not check["present"]


@pytest.mark.parametrize("value", [0, 1, 10000])
def test_occurrence_counts_preserve_zero_and_valid_upper_bound(value):
    assert sanitize_receipt(receipt(before_count=value))["before_count"] == value
    check = sanitize_check({"error":"none", "room_matches":True, "chat_ready":True,
                            "present":True, "occurrence_count":value})
    assert check["occurrence_count"] == value
    assert check["present"] == (value > 0)


def test_script_uses_verified_fields_and_keeps_credentials_in_closure():
    script = send_script("1698225", "【正常译文】")
    assert SEND_URL in script
    assert "new URLSearchParams({bubble:'0'" in script
    assert "csrf,csrf_token:csrf" in script
    assert "String(Math.floor(Date.now()/1000))" in script
    assert "sessionStorage.setItem(key" in script
    assert "last() + 3500" in script
    assert "controller.abort(),10000" in script
    assert "login.data.isLogin !== true" in script
    assert script.index("if (!nav.ok") < script.index("new URLSearchParams") < script.index("method:'POST'")
    assert BODY_FIELDS == ("bubble", "msg", "color", "mode", "fontsize", "rnd", "roomid", "csrf", "csrf_token")
    assert not any(item in script for item in ("console.", "postMessage", "CookieManager", "w_rid", "uid", "uname"))
    assert "document.body" not in presence_script("1698225", "【正常译文】")
    assert SEND_URL not in login_script("1698225")


def run_js(scripts, options=None):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node unavailable for offline JavaScript validation")
    harness = r"""
      const scripts = __SCRIPTS__, options = __OPTIONS__;
      let now = 1700000000000, postStarts = [], calls = [], bodies = [];
      Date.now = () => now;
      globalThis.performance = {now:() => now-1700000000000};
      globalThis.window = {};
      globalThis.location = {href:options.wrongRoom ? 'https://live.bilibili.com/1' : 'https://live.bilibili.com/1698225'};
      const values = new Map();
      globalThis.sessionStorage = {getItem:key => values.get(key),setItem:(key,value) => values.set(key,value)};
      globalThis.setTimeout = (fn,ms) => {if(ms < 10000){now+=Math.max(0,ms);queueMicrotask(fn);} return 1;};
      globalThis.clearTimeout = () => {};
      let limitReads = 0;
      const chatMessages = [...(options.chatMessages || [])];
      globalThis.document = {cookie:options.noCsrf ? '' : 'bili_jct='+'a'.repeat(32),querySelector:selector => {
        if (selector.includes('chat-history-list') && options.chatError) throw Error('private_dom_error');
        if (selector.includes('chat-history-list')) return options.chatUnavailable ? null : {
          querySelectorAll:selector => options.officialDomOnly && !selector.includes('.danmaku-item .danmaku-item-right') ? [] : chatMessages.map(text => ({textContent:text})),
          innerText:options.chatContainerText || chatMessages.join('\n')};
        limitReads += 1;
        const limit = options.loweredLimit && limitReads >= 3 ? options.loweredLimit : options.pageLimit;
        return limit === undefined ? null : {__vue__:{$store:{getters:{baseInfoUser:{danmakuLengthLimit:limit,uid:'never_export'}}}}};
      }};
      globalThis.fetch = async (url, request) => {
        calls.push({url,method:request.method || 'GET'});
        if (url.includes('/nav')) return {ok:true,json:async () => ({code:0,data:{isLogin:!options.loggedOut}})};
        postStarts.push(now); bodies.push(Object.fromEntries(request.body.entries()));
        if(options.appendOnPost) chatMessages.push(request.body.get('msg'));
        if(options.abort) throw {name:'AbortError'};
        now += 275;
        return {status:200,json:async () => {
          if(options.invalidJson) throw Error('raw_response_secret');
          return {code:options.reject || 0,msg:options.flag ? 'f' : '',message:'',data:{uid:'never_export'}};
        }};
      };
      (async () => {
        const results = [];
        for(const script of scripts) results.push(await eval(script));
        process.stdout.write(JSON.stringify({results,postStarts,calls,bodies}));
      })().catch(() => process.exit(1));
    """.replace("__SCRIPTS__", json.dumps(scripts)).replace("__OPTIONS__", json.dumps(options or {}))
    result = subprocess.run([node, "-e", harness], capture_output=True, text=True, encoding="utf-8", timeout=10)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.parametrize("options,error", [
    ({"wrongRoom": True}, "wrong_room"), ({"loggedOut": True}, "not_logged_in"), ({"noCsrf": True}, "csrf_missing"),
])
def test_browser_preflight_failure_never_posts(options, error):
    result = run_js([send_script("1698225", "【正常译文】")], options)
    assert result["postStarts"] == []
    assert result["results"][0]["error"] == error
    assert result["results"][0]["request_started_epoch_ms"] is None
    assert result["results"][0]["before_count"] is None


def test_browser_actual_start_spacing_and_urlencoded_body():
    result = run_js([send_script("1698225", "【空格 + 中文】"), send_script("1698225", "【第二条】")])
    assert len(result["postStarts"]) == 2
    assert result["postStarts"][1] - result["postStarts"][0] == 3500
    assert set(result["bodies"][0]) == set(BODY_FIELDS)
    assert result["bodies"][0]["msg"] == "【空格 + 中文】"
    assert result["bodies"][0]["roomid"] == "1698225"
    assert result["results"][0]["request_elapsed_ms"] == 275
    assert "never_export" not in json.dumps(result["results"])


@pytest.mark.parametrize("options,outcome", [
    ({"flag": True}, "platform_f"), ({"reject": 10030}, "platform_rejected"),
    ({"abort": True}, "result_unknown"), ({"invalidJson": True}, "result_unknown"),
])
def test_browser_response_failures_remain_classified_and_redacted(options, outcome):
    result = run_js([send_script("1698225", "【正常译文】")], options)
    assert len(result["postStarts"]) == 1
    assert sanitize_receipt(result["results"][0])["outcome"] == outcome
    assert "raw_response_secret" not in json.dumps(result["results"])


def test_old_duplicate_stays_at_baseline_until_a_new_exact_node_appears():
    text = "【重复字幕】"
    scripts = [send_script("1698225", text), presence_script("1698225", text),
               send_script("1698225", text), presence_script("1698225", text)]
    old_only = run_js(scripts, {"chatMessages":[text]})["results"]
    assert [sanitize_receipt(old_only[index])["before_count"] for index in (0,2)] == [1,1]
    assert [sanitize_check(old_only[index])["occurrence_count"] for index in (1,3)] == [1,1]
    assert old_only[3]["occurrence_count"] <= old_only[2]["before_count"]
    added = run_js(scripts[:2], {"chatMessages":[text], "appendOnPost":True})["results"]
    assert sanitize_receipt(added[0])["before_count"] == 1
    assert sanitize_check(added[1])["occurrence_count"] == 2
    assert added[1]["occurrence_count"] > added[0]["before_count"]


def test_official_danmaku_item_right_class_is_used_for_baseline_and_receipt():
    # Public live-room app.3f725... creates div.chat-item.danmaku-item with
    # span.danmaku-item-right containing t.content, separate from nickname.
    text = "【真实正文类】"
    results = run_js([send_script("1698225", text), presence_script("1698225", text)],
                     {"chatMessages":[text], "appendOnPost":True, "officialDomOnly":True})["results"]
    assert results[0]["before_count"] == 1
    assert results[1]["occurrence_count"] == 2
    assert results[1]["present"]


def test_presence_never_matches_a_substring_of_an_unrelated_chat_node():
    text = "【精确译文】"
    result = run_js([presence_script("1698225", text)], {
        "chatMessages":["前缀"+text, text+"后缀"], "chatContainerText":text})["results"][0]
    assert result["chat_ready"]
    assert result["occurrence_count"] == 0
    assert not result["present"]
    assert ".includes(wanted)" not in presence_script("1698225", text)


@pytest.mark.parametrize("options", [{"chatUnavailable":True}, {"chatError":True}])
def test_unavailable_chat_never_posts_without_a_confirmable_baseline(options):
    text = "【精确译文】"
    result = run_js([send_script("1698225",text), presence_script("1698225",text)], options)
    results = result["results"]
    assert result["postStarts"] == []
    assert results[0]["error"] == "chat_unavailable"
    assert sanitize_receipt(results[0])["outcome"] == "not_sent"
    assert results[0]["request_started_epoch_ms"] is None
    assert results[0]["before_count"] is None
    assert not results[1]["chat_ready"]
    assert results[1]["occurrence_count"] is None
    assert not results[1]["present"]
    assert "private_dom_error" not in json.dumps(results)


@pytest.mark.parametrize("text,limit,accepted", [
    ("【"+"中"*8+"】",20,True), ("【"+"中"*9+"】",20,False),
    ("【"+"中"*18+"】",40,True), ("【"+"中"*19+"】",40,False),
    ("【"+"1"*16+"】",20,True), ("【"+"1"*17+"】",20,False),
    ("【"+"𠮷"*4+"】",20,True), ("【"+"𠮷"*5+"】",20,False),
])
def test_current_page_weighted_limits_prevent_oversize_post(text,limit,accepted):
    result=run_js([send_script("1698225",text)],{"pageLimit":limit})
    receipt=result["results"][0]
    assert receipt["max_weighted_length"] == limit
    assert receipt["max_characters"] == limit//2
    assert len(result["postStarts"]) == int(accepted)
    if accepted:
        assert sanitize_receipt(receipt)["outcome"] == "receipt_accepted"
    else:
        assert receipt["error"] == "text_too_long"
        assert receipt["request_started_epoch_ms"] is None
        assert receipt["before_count"] is None
        assert sanitize_receipt(receipt)["outcome"] == "not_sent"
    assert "never_export" not in json.dumps(result["results"])


def test_current_global_cap_is_40_utf16_units_even_when_account_raw_limit_is_higher():
    text = "【" + "𠮷"*19 + "】"
    result = run_js([send_script("1698225",text)], {"pageLimit":152})
    receipt = result["results"][0]
    assert len(result["postStarts"]) == 1
    assert receipt["max_weighted_length"] == 80
    assert receipt["max_characters"] == 40
    assert "Math.min(80,limit)" in login_script("1698225")
    assert "Math.min(40,Math.floor(raw/2))" in login_script("1698225")


def test_login_returns_current_limit_and_conservative_fallback_without_identity():
    for options,expected in (({},20),({"pageLimit":40},40),({"pageLimit":"40"},20),({"pageLimit":1000},80)):
        result=run_js([login_script("1698225")],options)
        receipt=result["results"][0]
        assert receipt["max_weighted_length"] == expected
        assert receipt["max_characters"] == expected//2
        assert receipt["is_login"]
        assert result["postStarts"] == []
        assert "never_export" not in json.dumps(receipt)
        assert all(call["url"].endswith('/nav') for call in result["calls"])


def test_page_limit_is_checked_again_immediately_before_post():
    result=run_js([send_script("1698225","【"+"中"*18+"】")],{"pageLimit":40,"loweredLimit":20})
    assert result["postStarts"] == []
    receipt=result["results"][0]
    assert receipt["max_weighted_length"] == 20
    assert receipt["error"] == "text_too_long"
    assert sanitize_receipt(receipt)["outcome"] == "not_sent"
