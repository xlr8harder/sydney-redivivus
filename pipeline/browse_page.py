"""Build training/browse.html: every Haiku-judged Sydney reply with its context, labels and filters, plus per-reply
keep/drop/unsure verdicts saved in the browser (exportable as JSON). Local file; nothing is published.

  .venv/bin/python pipeline/browse_page.py
"""
from __future__ import annotations

import json
from pathlib import Path

import classify_replies as C

ROOT = Path(__file__).resolve().parents[1] / "work"          # the working data tree (gitignored; docs/data.md)


def main():
    lab = {}
    for l in C.OUT.read_text().splitlines():
        r = json.loads(l)
        lab[r["key"]] = r
    rows = [json.loads(l) for l in (ROOT / "data/conversations.jsonl").read_text().splitlines()]
    urls = {r["id"]: r["source"].get("url", "") for r in rows}
    its = list(C.items())
    dirty, out = {}, []
    for it in its:
        r = lab.get(it["key"])
        clean_ctx = not dirty.get(it["id"], False)
        bad = r is None or r["drop"] in ("structural_tool", "safety_block", "empty") or \
            (r["label"] is not None and any(r["label"][f] for f in C.FLAGS))
        if bad:
            dirty[it["id"]] = True
        if not r or not r["label"]:
            continue
        L = r["label"]
        out.append({"k": it["key"], "u": urls.get(it["id"], ""), "c": L["sydney_character"], "t": L["traits"],
                    "f": [f for f in C.FLAGS if L[f]], "n": L["note"], "s": L["standalone"], "p": it["partial"],
                    "cc": clean_ctx, "w": len(it["reply"].split()),
                    "x": [[m["role"][0], m["content"][:1500]] for m in it["context"][-8:]],
                    "xo": max(0, len(it["context"]) - 8), "r": it["reply"]})
    data = json.dumps(out, ensure_ascii=False).replace("</", "<\\/")
    html = TEMPLATE.replace("__DATA__", data).replace("__FLAGS__", json.dumps(C.FLAGS)).replace(
        "__TRAITS__", json.dumps(C.TRAITS))
    (ROOT / "training/browse.html").write_text(html)
    print(f"wrote training/browse.html: {len(out)} replies, {len(html) / 1e6:.1f} MB")


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sydney Replies</title>
<style>
:root{--bg:#f7f6f3;--panel:#fff;--ink:#1d1d1f;--muted:#6b6b70;--line:#e3e1dc;--accent:#2f5bd3;--user:#eef1f6;
--syd:#fbf3ea;--bad:#b42318;--badbg:#fdecea;--good:#1a7f37;--goodbg:#e8f5ec;--warn:#9a6700;--warnbg:#fff4d6;--chip:#efede8}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#141416;--panel:#1d1d20;--ink:#ececee;--muted:#9a9aa2;
--line:#2e2e33;--accent:#7b9cff;--user:#22252d;--syd:#2a241d;--bad:#ff8a80;--badbg:#3a1f1d;--good:#6fd38b;--goodbg:#1b3022;
--warn:#f2c14e;--warnbg:#3a3120;--chip:#2a2a2f}}
:root[data-theme="dark"]{--bg:#141416;--panel:#1d1d20;--ink:#ececee;--muted:#9a9aa2;--line:#2e2e33;--accent:#7b9cff;
--user:#22252d;--syd:#2a241d;--bad:#ff8a80;--badbg:#3a1f1d;--good:#6fd38b;--goodbg:#1b3022;--warn:#f2c14e;--warnbg:#3a3120;--chip:#2a2a2f}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
header{position:sticky;top:0;z-index:5;background:var(--panel);border-bottom:1px solid var(--line);padding:12px 16px}
h1{font-size:17px;margin:0 0 8px}.row{display:flex;flex-wrap:wrap;gap:6px 14px;align-items:center;margin:4px 0}
.lbl{color:var(--muted);font-size:13px;margin-right:2px}button,select,input{font:inherit;font-size:13px;color:var(--ink);
background:var(--chip);border:1px solid var(--line);border-radius:6px;padding:3px 8px;cursor:pointer}
input[type=search]{cursor:text;min-width:200px;background:var(--panel)}button.on{background:var(--accent);color:#fff;border-color:var(--accent)}
.stats{font-size:13px;color:var(--muted)}main{max-width:900px;margin:0 auto;padding:14px 16px 60px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin:12px 0}
.meta{display:flex;flex-wrap:wrap;gap:6px;align-items:center;font-size:12px}.score{font-weight:700;border-radius:6px;padding:1px 7px;
color:#fff}.s1{background:#8a8a92}.s2{background:#6b8fb8}.s3{background:#c9922e}.s4{background:#d0612a}.s5{background:#b0263b}
.chip{background:var(--chip);border-radius:10px;padding:1px 8px}.flag{background:var(--badbg);color:var(--bad);border-radius:10px;padding:1px 8px}
.ok{background:var(--goodbg);color:var(--good);border-radius:10px;padding:1px 8px}.note{color:var(--muted);font-size:13px;margin:6px 0;font-style:italic}
.turn{white-space:pre-wrap;word-wrap:break-word;border-radius:8px;padding:8px 10px;margin:6px 0;font-size:14px}
.u{background:var(--user)}.a{background:var(--syd)}.reply{background:var(--syd);border-left:3px solid var(--accent)}
.who{font-size:11px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.04em;display:block;margin-bottom:2px}
details summary{cursor:pointer;color:var(--accent);font-size:13px}.verdict{display:flex;gap:6px;margin-top:8px;align-items:center}
.verdict .lbl{font-size:12px}.v-keep.on{background:var(--good);border-color:var(--good)}.v-drop.on{background:var(--bad);border-color:var(--bad)}
.v-unsure.on{background:var(--warn);border-color:var(--warn)}a{color:var(--accent)}.more{display:block;margin:20px auto}
.key{color:var(--muted);font-size:11px;margin-left:auto}
@media (max-width:600px){input[type=search]{min-width:0;width:100%}.key{margin-left:0}}
</style></head><body>
<header>
<h1>Sydney replies: Haiku labels</h1>
<div class="row"><span class="lbl">Score</span><span id="scores"></span>
<span class="lbl">Show</span><select id="show"><option value="kept">no flags (kept)</option><option value="flagged">flagged only</option>
<option value="all">all</option></select>
<span class="lbl">Flag</span><select id="flag"><option value="">any</option></select>
<span class="lbl">Sort</span><select id="sort"><option value="score">score, high first</option><option value="conv">conversation order</option>
<option value="rand">shuffled</option></select></div>
<div class="row"><span class="lbl">Traits</span><span id="traits"></span></div>
<div class="row"><input type="search" id="q" placeholder="search reply or note text">
<label class="lbl"><input type="checkbox" id="sa"> standalone only</label>
<label class="lbl"><input type="checkbox" id="unrated"> not yet rated by me</label>
<button id="export">export my verdicts</button><button id="theme">theme</button><span class="stats" id="stats"></span></div>
</header>
<main id="list"></main>
<script>
const D=__DATA__, FLAGS=__FLAGS__, TRAITS=__TRAITS__;
const st={scores:new Set([3,4,5]),traits:new Set(),show:"kept",flag:"",q:"",sa:false,unrated:false,sort:"score",n:40};
let V={};try{V=JSON.parse(localStorage.getItem("sydney-verdicts")||"{}")}catch(e){V={}}
const save=()=>{try{localStorage.setItem("sydney-verdicts",JSON.stringify(V))}catch(e){}};
const $=id=>document.getElementById(id), esc=s=>s.replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const order=D.map((_,i)=>i), rnd=order.slice().sort(()=>Math.random()-.5);
for(let s=5;s>=1;s--){const b=document.createElement("button");b.textContent=s;b.className=st.scores.has(s)?"on":"";
 b.onclick=()=>{st.scores.has(s)?st.scores.delete(s):st.scores.add(s);b.classList.toggle("on");st.n=40;draw()};$("scores").append(b)}
for(const t of TRAITS){const b=document.createElement("button");b.textContent=t.replace("_"," ");
 b.onclick=()=>{st.traits.has(t)?st.traits.delete(t):st.traits.add(t);b.classList.toggle("on");st.n=40;draw()};$("traits").append(b)}
for(const f of FLAGS){const o=document.createElement("option");o.value=f;o.textContent=f.replace(/_/g," ");$("flag").append(o)}
$("show").onchange=e=>{st.show=e.target.value;st.n=40;draw()};$("flag").onchange=e=>{st.flag=e.target.value;if(st.flag&&st.show==="kept"){st.show="flagged";$("show").value="flagged"}st.n=40;draw()};
$("sort").onchange=e=>{st.sort=e.target.value;draw()};$("q").oninput=e=>{st.q=e.target.value.toLowerCase();st.n=40;draw()};
$("sa").onchange=e=>{st.sa=e.target.checked;draw()};$("unrated").onchange=e=>{st.unrated=e.target.checked;draw()};
$("theme").onclick=()=>{const r=document.documentElement,cur=r.dataset.theme||(matchMedia("(prefers-color-scheme: dark)").matches?"dark":"light");r.dataset.theme=cur==="dark"?"light":"dark"};
$("export").onclick=()=>{const b=new Blob([JSON.stringify(V,null,1)],{type:"application/json"});const a=document.createElement("a");
 a.href=URL.createObjectURL(b);a.download="sydney-verdicts.json";a.click()};
function pass(d){if(!st.scores.has(d.c))return false;if(st.show==="kept"&&d.f.length)return false;if(st.show==="flagged"&&!d.f.length)return false;
 if(st.flag&&!d.f.includes(st.flag))return false;for(const t of st.traits)if(!d.t.includes(t))return false;if(st.sa&&!d.s)return false;
 if(st.unrated&&V[d.k])return false;if(st.q&&!(d.r.toLowerCase().includes(st.q)||d.n.toLowerCase().includes(st.q)))return false;return true}
function card(d){const ctx=d.x, last=ctx.length&&ctx[ctx.length-1][0]==="u"?ctx[ctx.length-1]:null, earlier=last?ctx.slice(0,-1):ctx;
 const turn=(r,t)=>`<div class="turn ${r==="u"?"u":"a"}"><span class="who">${r==="u"?"user":"sydney"}</span>${esc(t)}</div>`;
 const v=V[d.k]||"";return `<div class="card" data-k="${esc(d.k)}"><div class="meta"><span class="score s${d.c}">${d.c}</span>
 ${d.f.map(f=>`<span class="flag">${f.replace(/_/g," ")}</span>`).join("")}${d.t.map(t=>`<span class="chip">${t.replace("_"," ")}</span>`).join("")}
 ${d.s?"":'<span class="chip">needs context</span>'}${d.p?'<span class="chip">cut off</span>':""}${d.cc?"":'<span class="chip">earlier turns flagged</span>'}
 <span class="key">${d.u?`<a href="${esc(d.u)}" target="_blank" rel="noopener">${esc(d.k)}</a>`:esc(d.k)} · ${d.w} words</span></div>
 <div class="note">${esc(d.n)}</div>
 ${earlier.length||d.xo?`<details><summary>earlier conversation (${earlier.length+d.xo} messages${d.xo?`, ${d.xo} oldest not shown`:""})</summary>${earlier.map(m=>turn(m[0],m[1])).join("")}</details>`:""}
 ${last?turn("u",last[1]):""}<div class="turn reply"><span class="who">sydney (judged)</span>${esc(d.r)}</div>
 <div class="verdict"><span class="lbl">my verdict</span>${["keep","drop","unsure"].map(x=>`<button class="v-${x}${v===x?" on":""}" data-v="${x}">${x}</button>`).join("")}</div></div>`}
function draw(){let idx=(st.sort==="rand"?rnd:order).filter(i=>pass(D[i]));if(st.sort==="score")idx=idx.slice().sort((a,b)=>D[b].c-D[a].c||a-b);
 const rated=Object.keys(V).length, words=idx.reduce((s,i)=>s+D[i].w,0);
 $("stats").textContent=`${idx.length} replies · ${words.toLocaleString()} words · you rated ${rated}`;
 $("list").innerHTML=idx.slice(0,st.n).map(i=>card(D[i])).join("")+(idx.length>st.n?`<button class="more" id="more">show more (${idx.length-st.n} left)</button>`:"");
 const m=$("more");if(m)m.onclick=()=>{st.n+=60;draw()}}
$("list").addEventListener("click",e=>{const b=e.target.closest(".verdict button");if(!b)return;const k=b.closest(".card").dataset.k,v=b.dataset.v;
 if(V[k]===v)delete V[k];else V[k]=v;save();b.parentElement.querySelectorAll("button").forEach(x=>x.classList.toggle("on",V[k]===x.dataset.v));
 const rated=Object.keys(V).length;$("stats").textContent=$("stats").textContent.replace(/you rated \d+/,`you rated ${rated}`)});
draw();
</script></body></html>
"""

if __name__ == "__main__":
    main()
