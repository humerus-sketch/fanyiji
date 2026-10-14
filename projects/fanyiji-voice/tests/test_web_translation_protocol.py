import json
from pathlib import Path
import shutil
import subprocess

import pytest

from fanyiji.web_translation_protocol import (
    allowed_navigation, poll_script, prepare_script, provider_url, validate_command, validated_result,
)


def command(**overrides):
    return {"op": "translate", "id": "a" * 32, "text": "こんにちは", "source_lang": "ja",
            "target_lang": "zh", **overrides}


@pytest.mark.parametrize("provider", ["google_web", "youdao_web"])
@pytest.mark.parametrize("pair", [("ja", "zh"), ("zh", "ja")])
def test_provider_url_is_fixed_and_has_no_transcript(provider, pair):
    url = provider_url(provider, *pair)
    assert allowed_navigation(provider, url)
    assert "text=" not in url
    if provider == "youdao_web":
        assert url == "https://fanyi.youdao.com/#/TextTranslate"
    else:
        assert "sl=" in url and "tl=" in url


@pytest.mark.parametrize("value", [
    None, [], {"op": "eval"}, {"op": "show", "cookie": "secret"},
    command(id="not-a-request"), command(text=""), command(text=" " * 2),
    command(text="x" * 5001), command(text="hi\x00"), command(source_lang="en"),
    command(source_lang="zh"), command(url="https://evil.test/"), command(text=4),
])
def test_rejects_arbitrary_commands_and_invalid_translation(value):
    with pytest.raises(ValueError):
        validate_command(value)


@pytest.mark.parametrize("op", ["show", "close", "version"])
def test_control_commands_have_no_script_or_url(op):
    assert validate_command({"op": op}) == {"op": op}


@pytest.mark.parametrize("url", [
    "http://fanyi.youdao.com/", "https://fanyi.youdao.com.evil.test/",
    "https://user:secret@fanyi.youdao.com/", "https://fanyi.youdao.com:234/",
    "https://fanyi.youdao.com:wrong/", "https://fanyi.youdao.com/\n",
    "https://fanyi.youdao.com\\@evil.test/", "file:///C:/anything", "javascript:alert(1)", None,
])
def test_navigation_cannot_escape_provider(url):
    assert not allowed_navigation("youdao_web", url)


@pytest.mark.parametrize("provider,url", [
    ("google_web", "https://fanyi.youdao.com/"), ("youdao_web", "https://translate.google.com/"),
    ("youdao_web", "https://live.bilibili.com/1698225"), ("google_web", "https://evil.test/"),
])
def test_no_cross_provider_or_live_browser_navigation(provider, url):
    assert not allowed_navigation(provider, url)


@pytest.mark.parametrize("value", [
    None, [], {"id": "b" * 32, "status": "ok", "text": "你好"},
    {"id": "a" * 32, "status": "ok", "text": ""},
    {"id": "a" * 32, "status": "ok", "text": "<asr_text>hello"},
    {"id": "a" * 32, "status": "ok", "text": "```json"},
    {"id": "a" * 32, "status": "ok", "text": "你好", "cookie": "secret"},
    {"id": "a" * 32, "status": "error", "code": "raw-secret-error"},
    {"id": "a" * 32, "status": "pending", "text": "previous result"},
])
def test_invalid_results_only_return_fixed_error(value):
    assert validated_result(value, command()) == {"id": "a" * 32, "status": "error", "code": "invalid_response"}


def test_result_protocol_preserves_verified_translation_only():
    assert validated_result({"id": "a" * 32, "status": "ok", "text": " 你好\n世界 "}, command())["text"] == "你好\n世界"
    assert validated_result({"id": "a" * 32, "status": "pending"}, command())["status"] == "pending"


# A small DOM fixture exercises the emitted script in JavaScript. Website IDs,
# role/data attributes and language names are those observed on the two pages.
_DOM = r"""
const vm=require('vm');
const payload=JSON.parse(require('fs').readFileSync(0,'utf8'));
let now=1000, text='', result='OLD RESULT', source='', target='', dropdown=-1;
const visible=e=>e;
function element(extra={}) {return Object.defineProperties({innerText:'',getClientRects:()=>[{}],focus:()=>{}},Object.getOwnPropertyDescriptors(extra));}
const input=element({value:'old input',dispatchEvent:()=>{text=payload.provider==='google_web'?input.value:input.innerText; if(!text)result='';}});
const langValues=payload.provider==='google_web'?['ja','zh-CN']:['日语','中文'];
source=langValues[0];target=langValues[1];
const controls=[0,1].map(index=>element({get innerText(){return index===0?source:target},
  querySelector:selector=>selector==='[aria-selected="true"]'?{getAttribute:()=>index===0?source:target}:
    element({click:()=>{let wanted=selector.match(/code="([^"]+)/)[1];if(index===0)source=wanted;else target=wanted;}}),
  click:()=>{dropdown=index;}}));
const span=element({get innerText(){return result},closest:()=>({getAttribute:()=>payload.wrongOutputLang?'en':target})});
const outputNode=element({get innerText(){return result}});
const doc={querySelector:selector=>{
 if(selector==='textarea[jsname="BJE2fc"]'||selector==='#js_fanyi_input')return input;
 if(selector==='#js_fanyi_output_resultOutput')return outputNode;
 if(selector==='.lanFrom-container')return controls[0];
 if(selector==='.lanTo-container')return controls[1];
 return null;
}, querySelectorAll:selector=>{
 if(selector==='[role="tablist"]')return controls;
 if(selector==='[jsname="W297wb"]')return result?[span]:[];
 if(selector.startsWith('.common-language-container')&&dropdown>=0) return [element({click:()=>{
   const wanted=selector.includes('zh-CHS')?'中文':'日语';if(dropdown===0)source=wanted;else target=wanted;dropdown=-1;
 }})];
 return [];
}};
function TA(){};Object.defineProperty(TA.prototype,'value',{set(v){this.value=v;}});
const context=vm.createContext({window:{},document:doc,HTMLTextAreaElement:TA,
 location:{hostname:payload.provider==='google_web'?'translate.google.com':'fanyi.youdao.com'},
 getComputedStyle:()=>({visibility:'visible'}),InputEvent:function(){},Date:{now:()=>now},console});
let actions=[];
function run(script){const value=vm.runInContext(script,context);if(value.status==='input'){
  actions.push(value.action);input.value=value.action==='clear'?'':payload.text;input.dispatchEvent();
  return {id:value.id,status:'pending'};
}return value;}
let values=[run(payload.prepare),run(payload.poll)];
// Preparing again progresses controls; it must not accept the stale old result.
for(let i=0;i<8;i++){now+=300;values.push(run(payload.prepare));}
if(payload.sourceChanged){if(payload.provider==='google_web')input.value='different';else input.innerText='different';}
result='NEW TRANSLATION';values.push(run(payload.poll));now+=599;values.push(run(payload.poll));now+=1;values.push(run(payload.poll));
// Starting the next id clears the previous result instead of reusing it.
values.push(run(payload.nextPrepare)); values.push(run(payload.nextPoll));
process.stdout.write(JSON.stringify({values,text,source,target,actions}));
"""


@pytest.mark.parametrize("provider", ["google_web", "youdao_web"])
@pytest.mark.parametrize("pair", [("ja", "zh"), ("zh", "ja")])
@pytest.mark.parametrize("case", ["success", "source_changed", "wrong_output_lang"])
def test_dom_request_fences_and_bidirectional_controls(provider, pair, case):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable for webpage DOM fixtures")
    value = command(source_lang=pair[0], target_lang=pair[1])
    next_value = command(id="b" * 32, text="次の文", source_lang=pair[0], target_lang=pair[1])
    payload = {"provider": provider, "text": value["text"], "prepare": prepare_script(value, provider), "poll": poll_script(value, provider),
               "nextPrepare": prepare_script(next_value, provider), "nextPoll": poll_script(next_value, provider),
               "sourceChanged": case == "source_changed", "wrongOutputLang": case == "wrong_output_lang"}
    completed = subprocess.run([node, "-e", _DOM], input=json.dumps(payload), text=True, encoding="utf-8",
                               capture_output=True, timeout=10, check=True)
    document = json.loads(completed.stdout)
    results = document["values"]
    assert document["actions"] == (["clear", "insert", "clear"] if provider == "google_web" else [])
    assert all(item["status"] == "pending" for item in results[:10])
    last = results[-3]
    if case == "source_changed":
        assert last["code"] == "source_changed"
    elif case == "wrong_output_lang" and provider == "google_web":
        assert last["status"] == "pending"
    else:
        assert last == {"id": "a" * 32, "status": "ok", "text": "NEW TRANSLATION"}
    assert results[-2:] == [{"id": "b" * 32, "status": "pending"}] * 2


def test_template_words_and_javascript_in_input_remain_plain_text():
    value = command(text='__PROVIDER__ __PREPARE__ "; window.pwned=true; //')
    script = prepare_script(value, "google_web")
    assert json.dumps(value, ensure_ascii=True) in script
    assert "fetch(" not in script and "XMLHttpRequest" not in script
