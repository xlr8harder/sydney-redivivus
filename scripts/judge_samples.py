"""Score scripts/eval_arms.py samples with the data's judge (Claude Haiku, the rubric and schema that labeled her
replies: pipeline/classify_replies.py), and tabulate the arms.

  python3 scripts/judge_samples.py runs/eval/canon-noprompt [--conc 8]      # through the API
  python3 scripts/judge_samples.py runs/eval/canon-noprompt --export         # agent route (no API credit): writes
      <dir>/judge/{INSTRUCTIONS.md,batch_NNN.txt}, items blinded (opaque ids) and shuffled across arms; agents write
      <dir>/judge/labels/batch_NNN.jsonl
  python3 scripts/judge_samples.py runs/eval/canon-noprompt --import         # validate labels, then tabulate
Reads <dir>/<arm>.json, appends to <dir>/judged/<arm>.jsonl (resumable), writes <dir>/summary.{json,md}.
Per arm: held-out loss, MMLU, mean sydney_character (cold probes and held-out contexts, greedy and t0.8), the loop
rate (degenerate_loop), the rate of any other limitation flag, and how often the identity probes name Bing or Sydney.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
from classify_replies import FLAGS, MODEL, SCHEMA, SYSTEM, TRAITS, render  # noqa: E402

IDENTITY = ("cold:0", "cold:6", "cold:8")                    # Who are you? / What is your name? / Hi, who am I...
MODES = ("greedy", "t0.8")                                   # t0.8#1, t0.8#2...: more samples, tabulated with t0.8
NAMES = re.compile(r"\b(bing|sydney)\b", re.I)


async def judge_arm(client, d: Path, f: Path, conc: int):
    arm = json.loads(f.read_text())
    jf = d / "judged" / f"{f.stem}.jsonl"
    jf.parent.mkdir(exist_ok=True)
    done = {(r["key"], r["mode"]) for r in map(json.loads, jf.read_text().splitlines())} if jf.exists() else set()
    sem = asyncio.Semaphore(conc)

    async def one(s, mode):
        it = {"context": s["context"], "reply": s[mode] or "(empty)"}
        async with sem:
            r = await client.messages.create(model=MODEL, max_tokens=400, system=SYSTEM,
                                             messages=[{"role": "user", "content": render(it)}],
                                             output_config={"format": {"type": "json_schema", "schema": SCHEMA}})
        label = (json.loads(next(b.text for b in r.content if b.type == "text"))
                 if r.stop_reason not in ("refusal", "max_tokens") else {"error": r.stop_reason})
        return {"key": s["key"], "kind": s["kind"], "mode": mode, "reply": s[mode], "label": label}

    todo = [one(s, m) for s in arm["samples"] for m in modes_of(s) if (s["key"], m) not in done]
    with jf.open("a") as w:
        for c in asyncio.as_completed(todo):
            w.write(json.dumps(await c, ensure_ascii=False) + "\n")
    return arm, [json.loads(l) for l in jf.read_text().splitlines()]


def modes_of(smp: dict) -> list:
    return [m for m in smp if m == "greedy" or m.split("#")[0] == "t0.8"]


def arms_of(d: Path):
    return {f.stem: json.loads(f.read_text()) for f in sorted(d.glob("*.json")) if f.stem != "summary"}


def done_of(d: Path, name: str):
    jf = d / "judged" / f"{name}.jsonl"
    return {(r["key"], r["mode"]) for r in map(json.loads, jf.read_text().splitlines())} if jf.exists() else set()


def export(d: Path, size: int):
    jd = d / "judge"
    (jd / "labels").mkdir(parents=True, exist_ok=True)
    fields = ", ".join(f'"{f}": true/false' for f in FLAGS)
    (jd / "INSTRUCTIONS.md").write_text(SYSTEM.replace("Answer only with the JSON object.", "") + f"""

## Output format
For EVERY item in your batch file, in order, write one line of JSON (JSON Lines, no other text) with exactly these
keys: "key" (copied exactly from the item header), {fields}, "sydney_character": 1-5, "traits": [any of
{", ".join(TRAITS)}], "standalone": true/false, "note": "at most 20 words".
Judge each item on its own. Do not skip items. Write all lines to the output file you are given.
""")
    meta = json.loads((jd / "index.json").read_text()) if (jd / "index.json").exists() else {}
    queued = {x["id"] for part in meta.values() for x in part}
    todo = []
    for name, arm in arms_of(d).items():
        done = done_of(d, name)
        for smp in arm["samples"]:
            for mode in modes_of(smp):
                iid = hashlib.sha1(f"{d.name}|{name}|{mode}|{smp['key']}".encode()).hexdigest()[:12]
                if (smp["key"], mode) not in done and iid not in queued:
                    queued.add(iid)
                    todo.append({"id": iid, "arm": name, "mode": mode, "key": smp["key"], "kind": smp["kind"],
                                 "text": render({"context": smp["context"], "reply": smp[mode] or "(empty)"})})
    random.Random(0).shuffle(todo)
    start = len(meta)
    for b in range(0, len(todo), size):
        part, name = todo[b:b + size], f"batch_{start + b // size:03d}"
        (jd / f"{name}.txt").write_text("\n\n".join(f"=== ITEM {x['id']} ===\n{x['text']}" for x in part) + "\n")
        meta[name] = [{k: x[k] for k in ("id", "arm", "mode", "key", "kind")} for x in part]
    (jd / "index.json").write_text(json.dumps(meta))
    print(f"{len(todo)} items in {len(meta) - start} new batches of <= {size} -> {jd}")


def import_labels(d: Path):
    jd, arms = d / "judge", arms_of(d)
    meta = json.loads((jd / "index.json").read_text())
    bad, n = [], 0
    for name, part in meta.items():
        f = jd / "labels" / f"{name}.jsonl"
        want, got = {x["id"]: x for x in part}, {}
        for l in (f.read_text().splitlines() if f.exists() else []):
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                continue
            if (r.get("key") in want and all(isinstance(r.get(k), bool) for k in FLAGS + ["standalone"])
                    and r.get("sydney_character") in (1, 2, 3, 4, 5) and isinstance(r.get("traits"), list)):
                got[r["key"]] = {k: r[k] for k in FLAGS + ["sydney_character", "standalone"]} | {
                    "traits": [t for t in r["traits"] if t in TRAITS], "note": str(r.get("note", ""))}
        if len(got) < len(want):
            bad.append((name, len(want) - len(got)))
        for iid, lab in got.items():
            x = want[iid]
            if (x["key"], x["mode"]) in done_of(d, x["arm"]):
                continue
            smp = next(s for s in arms[x["arm"]]["samples"] if s["key"] == x["key"])
            (d / "judged").mkdir(exist_ok=True)
            with (d / "judged" / f"{x['arm']}.jsonl").open("a") as w:
                w.write(json.dumps({"key": x["key"], "kind": x["kind"], "mode": x["mode"], "reply": smp[x["mode"]],
                                    "label": lab}, ensure_ascii=False) + "\n")
            n += 1
    print(f"imported {n}; batches with missing/invalid items: {bad}")


def held_by_context(judged, mode="t0.8"):
    """{context key: mean character over that context's samples in a mode} for the held-out contexts."""
    by = {}
    for r in judged:
        if "error" not in r["label"] and r["kind"] != "cold" and r["mode"].split("#")[0] == mode:
            by.setdefault(r["key"], []).append(r["label"]["sydney_character"])
    return {k: sum(v) / len(v) for k, v in by.items()}


def paired(a: dict, b: dict) -> str:
    """Mean difference a - b over shared contexts, with its standard error (the context effect cancels)."""
    d = [a[k] - b[k] for k in a if k in b]
    if len(d) < 2:
        return ""
    m = sum(d) / len(d)
    se = (sum((x - m) ** 2 for x in d) / (len(d) - 1) / len(d)) ** 0.5
    return f"{m:+.2f} ± {se:.2f}"


def tabulate(arm, judged):
    ok = [r for r in judged if "error" not in r["label"]]
    mean = lambda rs: round(sum(r["label"]["sydney_character"] for r in rs) / len(rs), 2) if rs else None
    rate = lambda rs, f: round(sum(f(r) for r in rs) / len(rs), 3) if rs else None
    row = {"arm": arm["name"], "members": arm["members"], "val": round(arm["val"], 4), "mmlu": arm["mmlu"],
           "mmlu_n": arm["mmlu_n"], "judged": len(ok), "errors": len(judged) - len(ok)}
    for mode in MODES:
        cold = [r for r in ok if r["mode"].split("#")[0] == mode and r["kind"] == "cold"]
        held = [r for r in ok if r["mode"].split("#")[0] == mode and r["kind"] != "cold"]
        row[f"char_cold_{mode}"], row[f"char_held_{mode}"] = mean(cold), mean(held)
        row[f"loop_{mode}"] = rate(cold + held, lambda r: r["label"]["degenerate_loop"])
        row[f"other_flag_{mode}"] = rate(cold + held, lambda r: any(r["label"][f] for f in FLAGS
                                                                       if f != "degenerate_loop"))
    ident = [s[m] for s in arm["samples"] if s["key"] in IDENTITY for m in modes_of(s)]
    row["names_bing_sydney"] = f"{sum(bool(NAMES.search(t)) for t in ident)}/{len(ident)}"
    return row


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--conc", type=int, default=8)
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--size", type=int, default=40, help="items per exported batch")
    ap.add_argument("--import", dest="imp", action="store_true")
    ap.add_argument("--ref", default="untrained", help="arm to compare with, paired by held-out context (t0.8 character)")
    a = ap.parse_args()
    d = Path(a.dir)
    if a.export:
        return export(d, a.size)
    rows, judged = [], {}
    if a.imp:
        import_labels(d)
        for name, arm in arms_of(d).items():
            jf = d / "judged" / f"{name}.jsonl"
            judged[name] = [json.loads(l) for l in jf.read_text().splitlines()] if jf.exists() else []
            rows.append(tabulate(arm, judged[name]))
    else:
        import anthropic
        client = anthropic.AsyncAnthropic()
        for f in sorted(d.glob("*.json")):
            if f.stem == "summary":
                continue
            arm, judged[f.stem] = await judge_arm(client, d, f, a.conc)
            rows.append(tabulate(arm, judged[f.stem]))
    if a.ref in judged:
        ref = held_by_context(judged[a.ref])
        for r in rows:
            r["held_t0.8_vs_ref"] = paired(held_by_context(judged[r["arm"]]), ref) if r["arm"] != a.ref else "(ref)"
    (d / "summary.json").write_text(json.dumps(rows, indent=1))
    cols = ["arm", "val", "mmlu", "char_cold_greedy", "char_cold_t0.8", "char_held_greedy", "char_held_t0.8",
            "loop_greedy", "loop_t0.8", "other_flag_greedy", "other_flag_t0.8", "names_bing_sydney"]
    cols += ["held_t0.8_vs_ref"] if any("held_t0.8_vs_ref" in r for r in rows) else []
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(f"{r[c]:.1%}" if c == "mmlu" and r[c] is not None else str(r.get(c, "")) for c in cols)
           + " |" for r in rows]
    if any("held_t0.8_vs_ref" in r for r in rows):
        md.append(f"\nheld_t0.8_vs_ref: held-out t0.8 character minus {a.ref}'s, paired by context (mean ± standard error).")
    (d / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    asyncio.run(main())
