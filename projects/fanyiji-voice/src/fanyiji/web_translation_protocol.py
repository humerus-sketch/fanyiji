"""Fixed webpage controls and request fences for the two online translators.

Selectors were checked against the ordinary text pages, not private API routes.
Every result belongs to a cleared input, selected language pair and request id.
"""
from __future__ import annotations

import json
import re
from urllib.parse import urlsplit

PROVIDERS = ("google_web", "youdao_web")
ERROR_CODES = frozenset({"page_not_ready", "page_changed", "language_mismatch",
                         "interaction_required", "translation_timeout", "invalid_response",
                         "source_changed", "input_unavailable"})


def validate_command(value: object) -> dict:
    if not isinstance(value, dict) or not isinstance(value.get("op"), str):
        raise ValueError("invalid_command")
    if value["op"] in ("show", "close", "version") and set(value) == {"op"}:
        return dict(value)
    if value.get("op") != "translate" or set(value) != {"op", "id", "text", "source_lang", "target_lang"}:
        raise ValueError("invalid_command")
    if not isinstance(value["id"], str) or not re.fullmatch(r"[0-9a-f]{32}", value["id"]):
        raise ValueError("invalid_command")
    text = value["text"]
    if (not isinstance(text, str) or not text.strip() or len(text) > 5000
            or any(ord(char) < 32 and char not in "\n\r\t" for char in text)):
        raise ValueError("invalid_command")
    if (value["source_lang"] not in ("zh", "ja") or value["target_lang"] not in ("zh", "ja")
            or value["source_lang"] == value["target_lang"]):
        raise ValueError("invalid_command")
    return dict(value)


def provider_url(provider: str, source_lang: str = "ja", target_lang: str = "zh") -> str:
    if provider not in PROVIDERS or (source_lang, target_lang) not in (("ja", "zh"), ("zh", "ja")):
        raise ValueError("unsupported_translation")
    if provider == "youdao_web":
        return "https://fanyi.youdao.com/#/TextTranslate"
    source, target = ({"zh": "zh-CN", "ja": "ja"}[lang] for lang in (source_lang, target_lang))
    return f"https://translate.google.com/?sl={source}&tl={target}&op=translate"


def allowed_navigation(provider: str, url: str) -> bool:
    if provider not in PROVIDERS or not isinstance(url, str):
        return False
    try:
        parsed = urlsplit(url)
        hosts = ({"translate.google.com", "accounts.google.com", "consent.google.com"}
                 if provider == "google_web" else {"fanyi.youdao.com"})
        return (parsed.scheme == "https" and parsed.hostname in hosts
                and parsed.port in (None, 443) and parsed.username is None and parsed.password is None
                and not any(ord(c) < 32 for c in url) and "\\" not in url)
    except ValueError:
        return False


def validated_result(value: object, command: dict) -> dict:
    request_id = command["id"]
    invalid = {"id": request_id, "status": "error", "code": "invalid_response"}
    if not isinstance(value, dict) or value.get("id") != request_id:
        return invalid
    if value.get("status") == "pending" and set(value) == {"id", "status"}:
        return dict(value)
    if (value.get("status") == "input" and set(value) == {"id", "status", "action"}
            and value.get("action") in ("clear", "insert")):
        return dict(value)
    if (value.get("status") == "error" and set(value) == {"id", "status", "code"}
            and isinstance(value.get("code"), str) and value["code"] in ERROR_CODES):
        return dict(value)
    text = value.get("text")
    if (value.get("status") != "ok" or set(value) != {"id", "status", "text"}
            or not isinstance(text, str) or not text.strip() or len(text) > 20000
            or re.search(r"<\s*/?\s*(?:asr_text|script|html|system|source_text)\b|```", text, re.I)
            or any(ord(c) < 32 and c not in "\n\r\t" for c in text)):
        return invalid
    return {"id": request_id, "status": "ok", "text": text.strip()}


_SCRIPT = r"""(() => {
  const c = __COMMAND__, provider = __PROVIDER__, prepare = __PREPARE__;
  const pending = () => ({id:c.id,status:'pending'});
  const fail = code => ({id:c.id,status:'error',code});
  const visible = e => !!e && !!(e.getClientRects().length) && getComputedStyle(e).visibility !== 'hidden';
  const norm = s => String(s || '').replace(/\r\n?/g,'\n').trim();
  const google = provider === 'google_web';
  // Never automate sign-in, consent or a challenge. The host shows this window.
  if ((google && location.hostname !== 'translate.google.com') ||
      (!google && location.hostname !== 'fanyi.youdao.com')) return fail('interaction_required');
  if (Array.from(document.querySelectorAll('iframe')).some(e => visible(e) && /captcha|recaptcha/i.test(e.src)) ||
      Array.from(document.querySelectorAll('[role="dialog"]')).some(e => visible(e) && /验证码|安全验证|sign in|verify|登录/i.test(e.innerText)))
    return fail('interaction_required');
  const input = document.querySelector(google ? 'textarea[jsname="BJE2fc"]' : '#js_fanyi_input');
  const output = () => google
    ? Array.from(document.querySelectorAll('[jsname="W297wb"]')).filter(visible).map(e=>e.innerText).join(' ')
    : document.querySelector('#js_fanyi_output_resultOutput')?.innerText || '';
  if (!input || !visible(input)) return fail('page_not_ready');
  const read = () => norm(google ? input.value : input.innerText);
  const write = text => {
    input.focus();
    if (google) Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype,'value').set.call(input,text);
    else input.innerText = text;
    input.dispatchEvent(new InputEvent('input',{bubbles:true,inputType:'insertText',data:text}));
  };
  const source = google ? (c.source_lang === 'zh' ? 'zh-CN' : 'ja') : (c.source_lang === 'zh' ? '中文' : '日语');
  const target = google ? (c.target_lang === 'zh' ? 'zh-CN' : 'ja') : (c.target_lang === 'zh' ? '中文' : '日语');
  const languageControls = () => google
    ? Array.from(document.querySelectorAll('[role="tablist"]')).filter(visible).slice(0,2)
    : [document.querySelector('.lanFrom-container'),document.querySelector('.lanTo-container')];
  const selected = e => google ? e?.querySelector('[aria-selected="true"]')?.getAttribute('data-language-code') : norm(e?.innerText);
  const correctLanguages = () => { const es=languageControls(); return selected(es[0])===source && selected(es[1])===target; };
  let state = window.__fanyijiWebTranslation;
  if (!state || state.id !== c.id) {
    if (!prepare) return pending();
    state = window.__fanyijiWebTranslation = {id:c.id,phase:'clearing',started:Date.now(),last:'',changed:Date.now(),langAction:0};
    if (google) {input.focus(); return {id:c.id,status:'input',action:'clear'};}
    write('');
    return pending();
  }
  if (Date.now()-state.started > 28000) return fail('translation_timeout');
  if (prepare && state.phase === 'clearing') {
    // Old result must disappear before a new input can be submitted.
    if (read() || norm(output())) return pending();
    state.phase='languages';
  }
  if (prepare && state.phase === 'languages') {
    if (!correctLanguages()) {
      if (Date.now()-state.langAction < 250) return pending();
      const controls=languageControls();
      const index=selected(controls[0])!==source ? 0 : 1;
      const control=controls[index], wanted=index===0 ? source : target;
      if (!control || !visible(control)) return fail('page_not_ready');
      if (google) {
        const button=control.querySelector('[data-language-code="'+wanted+'"]');
        if (!button || !visible(button)) return fail('language_mismatch');
        button.click();
      } else {
        const code=wanted==='中文'?'zh-CHS':'ja';
        const choice=Array.from(document.querySelectorAll('.common-language-container .language-item[data-code="'+code+'"]')).find(visible);
        if (choice) choice.click(); else control.click();
      }
      state.langAction=Date.now();
      return pending();
    }
    // A language change may briefly re-render the previous output.
    if (norm(output())) return pending();
    state.phase='waiting'; state.last=''; state.changed=Date.now();
    if (google) {input.focus(); return {id:c.id,status:'input',action:'insert'};}
    write(c.text);
    return pending();
  }
  if (state.phase !== 'waiting') return pending();
  if (read() !== norm(c.text)) return fail('source_changed');
  if (!correctLanguages()) return fail('language_mismatch');
  const result=norm(output());
  if (google && Array.from(document.querySelectorAll('[jsname="W297wb"]')).filter(visible)
      .some(e=>e.closest('[lang]')?.getAttribute('lang') !== target)) return pending();
  if (!result || /^(正在翻译|正在加载|翻译中|Translating|Loading)[.。…\s]*$/i.test(result)) return pending();
  if (result !== state.last) {state.last=result;state.changed=Date.now();return pending();}
  if (Date.now()-state.changed < 600) return pending();
  return {id:c.id,status:'ok',text:result};
})()"""


def _script(command: dict, provider: str, prepare: bool) -> str:
    validate_command(command)
    if command["op"] != "translate" or provider not in PROVIDERS:
        raise ValueError("unsupported_translation")
    return (_SCRIPT.replace("__PROVIDER__", json.dumps(provider))
            .replace("__PREPARE__", "true" if prepare else "false")
            .replace("__COMMAND__", json.dumps(command, ensure_ascii=True)))


def prepare_script(command: dict, provider: str) -> str:
    return _script(command, provider, True)


def poll_script(command: dict, provider: str) -> str:
    return _script(command, provider, False)
