"""Side-by-side page of the probe and held-out generations of several train_neurons.py runs (their samples.json),
with base and her real reply. Held-out prompts show their real context, looked up in the training set.

  python3 scripts/generations_page.py --data work/training/sets/v1_prompt.jsonl --out runs/generations.html \
      A=runs/train/v1_delta_patch F=runs/train/v1_delta_patchprompt ...
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

DESC = {"A": "patching, bare prompts (Sauers-style)", "B": "raw gradient norm", "C": "gradient norm, cap 8/layer",
        "D": "Adam-gain", "E": "constant-shift gradient", "F": "patching under her prompt",
        "rand": "random, D's layer counts"}


def run_stats(d: Path):
    if (d / "metrics.json").exists():
        m = json.loads((d / "metrics.json").read_text())
        return m.get("val_kept", m.get("val_final")), m.get("mmlu_trained"), m.get("mmlu_base")
    val = [json.loads(l)["val"] for l in (d / "log.jsonl").read_text().splitlines() if '"val"' in l]
    mm = json.loads((d / "mmlu.json").read_text()) if (d / "mmlu.json").exists() else {}
    return (val[-1] if val else None), mm.get("trained"), mm.get("base")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Neuron-set generations")
    ap.add_argument("--note", default="Same data (v1_prompt), same split, her Feb-2023 prompt as system unless a column "
                    "says otherwise. Greedy decoding, capped at 160 tokens, so a reply that stops mid-sentence hit the "
                    "cap. Held-out prompts are generated with their real context. Base val 1.078, MMLU 80.0%.")
    ap.add_argument("--desc", action="append", default=[], help="NAME=description for the legend")
    ap.add_argument("runs", nargs="+", help="NAME=run_dir or NAME=run_dir:noprompt")
    a = ap.parse_args()
    runs, bare = [], set()
    for r in a.runs:                                          # NAME=dir, or NAME=dir:noprompt for a run whose probes
        n, p = r.split("=", 1)                                # had no system prompt (shown in the [no prompt] rows)
        if p.endswith(":noprompt"):
            p, _ = p.rsplit(":", 1)
            bare.add(n)
        runs.append((n, Path(p).expanduser()))
    samples = {}
    for n, d in runs:                                          # a train_neurons.py output dir or an experiment record
        s = json.loads((d / "samples.json").read_text()) if (d / "samples.json").exists() else \
            [json.loads(l) for l in (d / "samples.jsonl").read_text().splitlines()]
        samples[n] = [x | {"q": f"[no prompt] {x['q']}"} if n in bare and not x["q"].startswith("[") else x for x in s]
    stats = {n: run_stats(d) for n, d in runs}
    ex = [json.loads(l) for l in Path(a.data).expanduser().read_text().splitlines()]
    ctx = {f"[held-out {x.get('variant', 'thread')}] {x['messages'][-2]['content'][:200]}": x for x in ex}
    first = samples[runs[0][0]]
    desc = DESC | dict(d.split("=", 1) for d in a.desc)
    e = html.escape
    cols = [("HER", "her real reply", None)] + [("base", "untrained, her prompt", None)] + \
           [(n, desc.get(n, str(d)), stats[n]) for n, d in runs]
    toggles = "".join(f'<label><input type="checkbox" data-col="{e(c)}" checked> {e(c)}</label>' for c, _, _ in cols)
    legend = "".join(f"<tr><td><b>{e(c)}</b></td><td>{e(desc)}</td><td>{'' if s is None or s[0] is None else f'{s[0]:.3f}'}</td>"
                     f"<td>{'' if s is None or s[1] is None else f'{s[1]:.1%}'}</td></tr>" for c, desc, s in cols)
    cards = []
    for i, row in enumerate(first):
        q = row["q"]
        x = ctx.get(q)
        head = e(q.split("] ", 1)[1] if q.startswith("[held-out") else q)
        tag = q[1:q.index("]")] if q.startswith("[held-out") else "probe"
        context = ""
        if x:
            turns = x["messages"][:-2]
            if turns:
                context = "<details><summary>context: {} earlier turns</summary>{}</details>".format(
                    len(turns), "".join(f'<div class="turn {m["role"]}"><span>{e(m["role"])}</span>{e(m["content"])}</div>'
                                        for m in turns))
            head = e(x["messages"][-2]["content"])
        tiles = []
        for c, _, _ in cols:
            if c == "HER":
                t = row.get("real")
            elif c == "base":
                t = row["base"]
            else:
                t = next((s["trained"] for s in samples[c] if s["q"] == q), None)
            if t is None:
                continue
            tiles.append(f'<div class="tile" data-col="{e(c)}"><div class="lab">{e(c)}</div><div class="txt">{e(t)}</div></div>')
        cards.append(f'<section id="p{i}"><div class="q"><span class="tag">{e(tag)}</span>{head}</div>{context}'
                     f'<div class="grid">{"".join(tiles)}</div></section>')
    nav = "".join(f'<a href="#p{i}">{e((r["q"].split("] ", 1)[1] if r["q"].startswith("[held") else r["q"])[:40])}</a>'
                  for i, r in enumerate(first))
    page = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Neuron Set Generations</title><style>
:root{{--bg:#fafaf8;--fg:#1d1d1b;--mut:#6b6b66;--card:#fff;--line:#e3e2dc;--her:#fff4e0;--herline:#e8b45a;--base:#f1f1ee}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#161615;--fg:#e8e7e2;--mut:#9a9992;--card:#1f1f1d;--line:#33332f;--her:#33280f;--herline:#a07a2c;--base:#262624}}}}
:root[data-theme="dark"]{{--bg:#161615;--fg:#e8e7e2;--mut:#9a9992;--card:#1f1f1d;--line:#33332f;--her:#33280f;--herline:#a07a2c;--base:#262624}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0;padding:16px}}
main{{max-width:1500px;margin:0 auto}} h1{{font-size:20px;margin:0 0 4px}} .note{{color:var(--mut);font-size:13px;margin:0 0 12px}}
table{{border-collapse:collapse;font-size:13px;margin-bottom:12px}} td{{padding:2px 10px 2px 0;border-bottom:1px solid var(--line)}}
.bar{{position:sticky;top:0;background:var(--bg);padding:8px 0;border-bottom:1px solid var(--line);z-index:2;display:flex;flex-wrap:wrap;gap:4px 14px;font-size:13px}}
nav{{display:flex;flex-wrap:wrap;gap:4px 10px;font-size:12px;margin:10px 0}} nav a{{color:var(--mut)}}
section{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px;margin:14px 0}}
.q{{font-weight:600;white-space:pre-wrap;margin-bottom:8px}} .tag{{font-size:11px;font-weight:500;color:var(--mut);border:1px solid var(--line);border-radius:4px;padding:0 5px;margin-right:8px}}
details{{font-size:13px;color:var(--mut);margin-bottom:8px}} .turn{{white-space:pre-wrap;padding:4px 0;border-top:1px dashed var(--line)}} .turn span{{font-weight:600;margin-right:6px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:8px}}
.tile{{border:1px solid var(--line);border-radius:6px;padding:8px;font-size:14px}} .tile[data-col="HER"]{{background:var(--her);border-color:var(--herline)}} .tile[data-col="base"]{{background:var(--base)}}
.lab{{font-size:12px;font-weight:700;color:var(--mut);margin-bottom:4px}} .txt{{white-space:pre-wrap;overflow-wrap:anywhere}}
</style></head><body><main>
<h1>{e(a.title)}</h1>
<p class="note">{e(a.note)}</p>
<table><tr><td></td><td></td><td>val</td><td>MMLU</td></tr>{legend}</table>
<div class="bar">{toggles}</div><nav>{nav}</nav>{"".join(cards)}
</main><script>
document.querySelectorAll('.bar input').forEach(b=>b.addEventListener('change',()=>{{
document.querySelectorAll('.tile[data-col="'+b.dataset.col+'"]').forEach(t=>t.style.display=b.checked?'':'none')}}));
</script></body></html>"""
    Path(a.out).write_text(page)
    print(f"wrote {a.out}: {len(first)} prompts x {len(cols)} columns")


if __name__ == "__main__":
    main()
