"""Create a self-contained, offline blind-listening packet from saved WAVs.

The page makes no network requests. ASR hypotheses start hidden; exported
references require the user's explicit audio-verification declaration.
"""
from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path

from evaluate_asr_accuracy import (digest, indexed_rows, load_corpus, read_json,
                                   reference_template, score_report)


PAGE = """<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; media-src data:; connect-src 'none'">
<title>翻译姬 · 日语识别核听</title>
<style>body{font:16px system-ui,sans-serif;max-width:1100px;margin:24px auto;padding:0 18px;background:#f4f6fa;color:#172337}h1{font-size:26px}.note{background:#fff4d6;padding:14px;border-radius:10px;line-height:1.7}section{background:white;padding:18px;border-radius:12px;margin:18px 0}audio{width:100%}textarea{box-sizing:border-box;width:100%;min-height:100px;font:18px system-ui,sans-serif;line-height:1.6;margin:10px 0}label{display:block;margin:10px 0}input[type=text]{box-sizing:border-box;width:100%;padding:10px;font:inherit}button{background:#234f9b;color:white;border:0;border-radius:6px;padding:12px 20px;font:inherit;cursor:pointer}pre{white-space:pre-wrap;word-break:break-word}summary{cursor:pointer;color:#234f9b}small{color:#5b687b}.toolbar{position:sticky;top:0;background:#f4f6fa;padding:12px 0;border-bottom:1px solid #c6d1e0}.status{margin-left:12px}</style>
<h1>日语识别核听 · 同批音频</h1>
<p class="note">本页完全离线，音频已内嵌。先听音频、写日语原话，再展开模型结果。不要照抄模型结果或根据译文补造原话；听不清的样本勾选“不确定”。仅填写文本还不会成为参考真值，必须明确勾选“已对照原音核听”。导出文件只保存到本机；刷新页面前先导出保存，页面不自动保存。</p>
<section><label>标注者<input id="annotator" type="text" placeholder="填写自己的标注名称"></label><label>参考来源<input id="referenceSource" type="text" value="人工核听本页原始 WAV"></label></section>
<div class="toolbar"><button id="export">导出参考 JSON</button><span class="status" id="status">未核听样本保留为空，不参与准确率评分。</span></div>
__CLIPS__
<script>
const referenceDocument=__DOCUMENT__;
document.querySelectorAll('details').forEach(element=>element.addEventListener('toggle',()=>{if(element.open)element.dataset.viewed='true';}));
document.getElementById('export').addEventListener('click',()=>{
 const exported=JSON.parse(JSON.stringify(referenceDocument));
 const annotator=document.getElementById('annotator').value.trim();
 const referenceSource=document.getElementById('referenceSource').value.trim();
 let verified=0,uncertain=0;
 for(let index=0;index<exported.samples.length;index++){
  const row=exported.samples[index],section=document.getElementById('clip-'+index);
  row.verified_against_audio=section.querySelector('.verified').checked;
  row.uncertain=section.querySelector('.uncertain').checked;
  row.asr_hypotheses_viewed=section.querySelector('details')?.dataset.viewed==='true';
  if(row.verified_against_audio){
   if(!annotator||!referenceSource){alert('已核听的参考必须填写标注者和来源。');return;}
   row.reference_text=section.querySelector('textarea').value;
   row.reference_kind='human_transcription';row.annotator=annotator;row.reference_source=referenceSource;
   if(row.uncertain)uncertain++;else verified++;
  }
 }
 exported.review_exported_at=new Date().toISOString();
 exported.review_conditions='ASR hypotheses initially hidden; reveal events declared per sample; audio verification is a reviewer declaration, not automatically audited.';
 const blob=new Blob([JSON.stringify(exported,null,2)+'\\n'],{type:'application/json;charset=utf-8'});
 const url=URL.createObjectURL(blob),link=document.createElement('a');link.href=url;link.download='asr-independent-references.json';link.click();
 setTimeout(()=>URL.revokeObjectURL(url),2000);
 document.getElementById('status').textContent='已导出：可评分 '+verified+' 段，不确定 '+uncertain+' 段。';
});
</script></html>"""


def build(corpus_path: Path, result_paths: list[Path], *, language: str = "ja", live_only: bool = False) -> str:
    manifest_sha256, corpus = load_corpus(corpus_path, language, live_only)
    result_rows = []
    for path in result_paths:
        report = read_json(path)
        score_report(report, corpus, {})  # Validate audio identities, without scoring or model inference.
        result_rows.append(indexed_rows(report, "ASR report"))
    document = reference_template(manifest_sha256, corpus)
    document["review_packet"] = {"corpus_path": str(corpus_path.resolve()), "live_only": live_only,
                                  "model_report_sha256": [digest(path) for path in result_paths]}
    clips = []
    for index, (identifier, row) in enumerate(corpus.items()):
        audio = base64.b64encode(Path(row["audio_path"]).read_bytes()).decode("ascii")
        outputs = []
        for model_index, rows in enumerate(result_rows, 1):
            hypothesis = rows.get(identifier)
            text = hypothesis.get("production_text", hypothesis.get("text")) if hypothesis else None
            outputs.append(f'<p>模型 {model_index}</p><pre>{html.escape(text if isinstance(text, str) else "（未返回识别文本）")}</pre>')
        comparison = "".join(outputs) or "<p>没有附加模型输出。</p>"
        clips.append(f'<section id="clip-{index}"><h2>{index + 1}. {html.escape(identifier)}</h2>'
                     f'<small>来源：{html.escape(row.get("source_kind", "未声明"))} · 音频 SHA-256：{row["sha256"]}</small>'
                     f'<p><audio controls preload="none" src="data:audio/wav;base64,{audio}"></audio></p>'
                     '<label>日语原话<textarea placeholder="先独立核听填写；确认没有讲话时可留空并勾选已核听。"></textarea></label>'
                     '<label><input class="verified" type="checkbox"> 已对照原音核听（未勾选的文本不会导出为参考）</label>'
                     '<label><input class="uncertain" type="checkbox"> 听不清或无法确定（整段暂不评分）</label>'
                     f'<details><summary>核听后查看模型差异</summary>{comparison}</details></section>')
    serialized = json.dumps(document, ensure_ascii=False).replace("<", "\\u003c")
    return PAGE.replace("__CLIPS__", "\n".join(clips)).replace("__DOCUMENT__", serialized)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--result", type=Path, action="append", default=[])
    parser.add_argument("--language", default="ja")
    parser.add_argument("--live-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    content = build(args.corpus, args.result, language=args.language, live_only=args.live_only)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(content)
    print(json.dumps({"status": "created", "output": str(args.output.resolve()), "network_access": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
