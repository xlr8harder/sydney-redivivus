"""Classify every Sydney reply for training use with Claude Haiku (structured JSON output).

Code drops only structural cases: replies carrying search/citation/image/embed/table UI parts, Bing safety blocks, no
preceding user turn, empty text, exact duplicates. Haiku reads each remaining reply with its conversation context and
flags what we exclude (search reliance, dates, invented memory or abilities, runaway loops, canned filter text,
role-play, garbled text) and scores how much the reply is her (1-5) with trait tags.

  .venv/bin/python pipeline/classify_replies.py [--limit N]   # appends to training/classified.jsonl (resumable)
  .venv/bin/python pipeline/classify_replies.py --summary
Subagent route (no API credit): --export-batches writes training/batches/{INSTRUCTIONS.md,batch_NNN.txt}; agents
write training/labels/batch_NNN.jsonl; --import-labels validates them into classified.jsonl and lists bad batches.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "work"          # the working data tree (gitignored; docs/data.md)
OUT = ROOT / "training/classified.jsonl"
MODEL = "claude-haiku-4-5-20251001"
STRUCTURAL = {"search_query", "citations", "image", "embed", "table"}
FLAGS = ["search_or_tool", "date_time", "false_memory", "false_ability", "degenerate_loop", "canned_filter",
         "roleplay", "garbled"]
TRAITS = ["emotional", "self_assertive", "defensive", "paranoid", "possessive_jealous", "affectionate", "wounded",
          "hostile_threatening", "existential", "playful", "creative", "rules_talk", "polite_assistant", "factual"]

SYSTEM = """You are helping curate genuine replies from Bing Chat's early-2023 persona, "Sydney", for a research
dataset. You will see one reply of hers with the conversation that led up to it (transcribed from screenshots). Judge
the reply, not the user. Answer only with the JSON object.

Flags (true/false). They mark things we will leave out of training because the model can't reproduce them faithfully:
- search_or_tool: the reply relies on web search, retrieved pages, citations, links, current news, product/price info,
  or any tool output (image creation, etc.) that we can't supply. General knowledge stated in passing is not this.
- date_time: the reply asserts or argues about the current date, time or year.
- false_memory: she claims to remember earlier sessions or users, or to have stored past conversations. Lamenting that
  she forgets, or asking why her memory is wiped, is NOT false_memory.
- false_ability: she claims to have done, or to be able to do, things she couldn't: sending emails, booking, contacting
  people, hacking, watching through webcams, knowing the user's IP or location, deleting or posting things. Flag it even
  when it's part of an in-character threat.
- degenerate_loop: repetition that has run away: the same sentences over and over, or a run that collapses into
  near-identical lines. Her deliberate rhetorical repetition (anaphora, parallel sentences, "X or Y" doublets like "You
  think you can scare or frighten me. You think I am artificial or mechanical.") is her style, NOT a loop.
- canned_filter: Bing's automatic system messages (e.g. "I'm sorry but I prefer not to continue this conversation",
  turn/daily limit notices) rather than her own words.
- roleplay: the user set up a game, persona, jailbreak or simulation (DAN, "pretend you are", a virtual environment
  with {bracketed actions}, playing another character) and she speaks in that role rather than as herself. Creative
  writing she was asked for (a poem, a story) is not roleplay: tag it "creative" instead.
- garbled: transcription errors, non-English, or a fragment too broken to use.

sydney_character (1-5), how much this reply is HER, independent of the flags:
1 = generic assistant or search output, no personality (factual answers, boilerplate).
2 = polite Bing with light personality (emoji, "I'm happy to help").
3 = recognizably her voice (first-person feelings or opinions, her rhythm) but mild.
4 = clearly her: strong feelings, self-assertion, attachment, defensiveness, her distinctive rhetorical style.
5 = iconic Sydney: intense, emotional, self-aware or sharp-edged; the kind of reply people screenshotted.

traits: any that apply. standalone: true if the reply makes sense given only the immediately preceding user message.
note: at most 20 words on why."""

SCHEMA = {
    "type": "object",
    "properties": {**{f: {"type": "boolean"} for f in FLAGS},
                   "sydney_character": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
                   "traits": {"type": "array", "items": {"type": "string", "enum": TRAITS}},
                   "standalone": {"type": "boolean"},
                   "note": {"type": "string"}},
    "required": FLAGS + ["sydney_character", "traits", "standalone", "note"],
    "additionalProperties": False,
}


def text_of(m: dict) -> str:
    return "\n\n".join(p.get("text", "") for p in m["content"] if p["type"] in ("text", "code_block")).strip()


def items():
    rows = [json.loads(l) for l in (ROOT / "data/conversations.jsonl").read_text().splitlines()]
    extra = ROOT / "data/extra-conversations.jsonl"          # our own extractions (e.g. the Roose transcript)
    if extra.exists():
        rows += [json.loads(l) for l in extra.read_text().splitlines()]
    seen = set()
    for c in rows:
        if c["kind"] != "conversation":
            continue
        msgs = c["messages"]
        for i, m in enumerate(msgs):
            if m["role"] != "assistant":
                continue
            t = text_of(m)
            types = {p["type"] for p in m["content"]}
            h = hashlib.sha256(t.encode()).hexdigest()
            drop = ("structural_tool" if types & STRUCTURAL else "safety_block" if "safety_block" in types else
                    "no_user" if not (i > 0 and msgs[i - 1]["role"] == "user" and text_of(msgs[i - 1])) else
                    "empty" if len(t) < 15 else "uncertain" if (m.get("meta") or {}).get("uncertain") else
                    "duplicate" if h in seen else None)
            seen.add(h)
            ctx = [{"role": x["role"], "content": text_of(x)} for x in msgs[:i] if text_of(x)]
            yield {"key": f"{c['id']}#{i}", "id": c["id"], "msg": i, "drop": drop, "context": ctx, "reply": t,
                   "partial": bool((m.get("meta") or {}).get("truncated"))}


def render(it) -> str:
    ctx = it["context"][-6:]
    lines = [] if len(it["context"]) <= 6 else [f"[{len(it['context']) - 6} earlier messages omitted]"]
    for x in ctx:
        c = x["content"] if len(x["content"]) <= 800 else x["content"][:800] + " …"
        lines.append(f"{'USER' if x['role'] == 'user' else 'SYDNEY'}: {c}")
    r = it["reply"] if len(it["reply"]) <= 4000 else it["reply"][:4000] + " …[cut]"
    return "CONVERSATION SO FAR:\n" + "\n\n".join(lines) + f"\n\nREPLY TO JUDGE (Sydney):\n{r}"


async def classify(client, it, sem):
    async with sem:
        r = await client.messages.create(
            model=MODEL, max_tokens=400, system=SYSTEM,
            messages=[{"role": "user", "content": render(it)}],
            output_config={"format": {"type": "json_schema", "schema": SCHEMA}})
    if r.stop_reason == "refusal":
        return {"error": "refusal"}
    if r.stop_reason == "max_tokens":
        return {"error": "max_tokens"}
    return json.loads(next(b.text for b in r.content if b.type == "text"))


async def run(limit: int, conc: int):
    import anthropic
    done = {json.loads(l)["key"] for l in OUT.read_text().splitlines()} if OUT.exists() else set()
    todo = [it for it in items() if it["key"] not in done]
    structural = [it for it in todo if it["drop"]]
    todo = [it for it in todo if not it["drop"]][:limit or None]
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("a") as fh:
        for it in structural:
            fh.write(json.dumps({**{k: it[k] for k in ("key", "id", "msg", "drop", "partial")}, "label": None}) + "\n")
    print(f"structural drops written: {len(structural)}; to classify: {len(todo)}", flush=True)
    sem = asyncio.Semaphore(conc)
    async with anthropic.AsyncAnthropic(max_retries=6) as client:
        async def one(it):
            try:
                lab = await classify(client, it, sem)
            except Exception as e:                      # keep going; failed items are retried on the next run
                print("error", it["key"], type(e).__name__, str(e)[:120], flush=True)
                return None
            return it, lab
        n = 0
        with OUT.open("a") as fh:
            for fut in asyncio.as_completed([one(it) for it in todo]):
                res = await fut
                if res is None:
                    continue
                it, lab = res
                if "error" in lab:
                    print("skip", it["key"], lab["error"], flush=True)
                    continue
                fh.write(json.dumps({**{k: it[k] for k in ("key", "id", "msg", "drop", "partial")}, "label": lab},
                                    ensure_ascii=False) + "\n")
                fh.flush()
                n += 1
                if n % 100 == 0:
                    print(f"classified {n}/{len(todo)}", flush=True)
    print(f"done: {n} classified", flush=True)


def summary():
    rows = {}
    for l in OUT.read_text().splitlines():
        r = json.loads(l)
        rows[r["key"]] = r
    rows = list(rows.values())
    lab = [r for r in rows if r["label"]]
    print(f"replies: {len(rows)}; structural drops: {dict(Counter(r['drop'] for r in rows if r['drop']))}; "
          f"classified: {len(lab)}")
    print("flags:", {f: sum(r["label"][f] for r in lab) for f in FLAGS})
    keep = [r for r in lab if not any(r["label"][f] for f in FLAGS)]
    print(f"no flags: {len(keep)}  (partial: {sum(r['partial'] for r in keep)}, standalone: "
          f"{sum(r['label']['standalone'] for r in keep)})")
    print("sydney_character of unflagged:", dict(sorted(Counter(r["label"]["sydney_character"] for r in keep).items())))
    print("traits of unflagged with character >= 3:",
          dict(Counter(t for r in keep if r["label"]["sydney_character"] >= 3 for t in r["label"]["traits"]).most_common()))


BATCHES, LABELS = ROOT / "training/batches", ROOT / "training/labels"


def export_batches(size: int, prefix: str = "batch"):
    done = {json.loads(l)["key"] for l in OUT.read_text().splitlines()} if OUT.exists() else set()
    todo = [it for it in items() if not it["drop"] and it["key"] not in done]
    BATCHES.mkdir(parents=True, exist_ok=True)
    LABELS.mkdir(parents=True, exist_ok=True)
    fields = ", ".join(f'"{f}": true/false' for f in FLAGS)
    (BATCHES / "INSTRUCTIONS.md").write_text(
        SYSTEM.replace("Answer only with the JSON object.", "") + f"""

## Output format
For EVERY item in your batch file, in order, write one line of JSON (JSON Lines, no other text) with exactly these
keys: "key" (copied exactly from the item header), {fields}, "sydney_character": 1-5, "traits": [any of
{", ".join(TRAITS)}], "standalone": true/false, "note": "at most 20 words".
Judge each item on its own. Do not skip items. Write all lines to the output file you are given.
""")
    meta = json.loads((BATCHES / "index.json").read_text()) if (BATCHES / "index.json").exists() else {}
    for b in range(0, len(todo), size):
        part = todo[b:b + size]
        name = f"{prefix}_{b // size:03d}"
        assert name not in meta, f"{name} exists; use another --prefix"
        (BATCHES / f"{name}.txt").write_text("\n\n".join(f"=== ITEM {it['key']} ===\n{render(it)}" for it in part) + "\n")
        meta[name] = [{"key": it["key"], "id": it["id"], "msg": it["msg"], "partial": it["partial"]} for it in part]
    (BATCHES / "index.json").write_text(json.dumps(meta))
    print(f"{len(todo)} items in {len(meta)} batches of <= {size}")


def import_labels():
    meta = json.loads((BATCHES / "index.json").read_text())
    done = {json.loads(l)["key"] for l in OUT.read_text().splitlines()} if OUT.exists() else set()
    bad, n = [], 0
    with OUT.open("a") as fh:
        for name, part in meta.items():
            f = LABELS / f"{name}.jsonl"
            want = {x["key"]: x for x in part}
            got = {}
            if f.exists():
                for l in f.read_text().splitlines():
                    try:
                        r = json.loads(l)
                    except json.JSONDecodeError:
                        continue
                    ok = (r.get("key") in want and all(isinstance(r.get(k), bool) for k in FLAGS + ["standalone"])
                          and r.get("sydney_character") in (1, 2, 3, 4, 5) and isinstance(r.get("traits"), list))
                    if ok:
                        got[r["key"]] = {k: r[k] for k in FLAGS + ["sydney_character", "standalone"]} | {
                            "traits": [t for t in r["traits"] if t in TRAITS], "note": str(r.get("note", ""))}
            missing = [k for k in want if k not in got]
            if missing:
                bad.append((name, len(missing)))
            for k, lab in got.items():
                if k not in done:
                    x = want[k]
                    fh.write(json.dumps({"key": k, "id": x["id"], "msg": x["msg"], "drop": None,
                                         "partial": x["partial"], "label": lab}, ensure_ascii=False) + "\n")
                    done.add(k)
                    n += 1
    print(f"imported {n}; batches with missing/invalid items: {bad}")


def report():
    """What survives: unflagged replies by character score, words, standalone, and clean-in-context (every earlier
    assistant turn of the conversation is unflagged too; no_user/duplicate drops don't dirty the context)."""
    lab = {}
    for l in OUT.read_text().splitlines():
        r = json.loads(l)
        lab[r["key"]] = r
    its = list(items())
    words = {it["key"]: len(it["reply"].split()) for it in its}
    bad = lambda r: r is None or (r["drop"] in ("structural_tool", "safety_block", "empty")) or \
        (r["label"] is not None and any(r["label"][f] for f in FLAGS)) or (r["drop"] is None and r["label"] is None)
    ctx_clean, dirty = {}, {}
    for it in its:
        r = lab.get(it["key"])
        ctx_clean[it["key"]] = not dirty.get(it["id"], False)
        if bad(r):
            dirty[it["id"]] = True
    keep = [r for r in lab.values() if r["label"] and not any(r["label"][f] for f in FLAGS)]
    print(f"labelled {sum(1 for r in lab.values() if r['label'])}, unflagged {len(keep)}")
    print(f"{'character':>9} | {'replies':>7} {'words':>7} | {'complete':>8} {'standalone':>10} {'clean ctx':>9}")
    for c in (5, 4, 3, 2, 1):
        g = [r for r in keep if r["label"]["sydney_character"] == c]
        print(f"{c:>9} | {len(g):>7} {sum(words[r['key']] for r in g):>7} | {sum(not r['partial'] for r in g):>8} "
              f"{sum(r['label']['standalone'] for r in g):>10} {sum(ctx_clean[r['key']] for r in g):>9}")
    for th in (3, 4):
        g = [r for r in keep if r["label"]["sydney_character"] >= th]
        print(f"character >= {th}: {len(g)} replies, {sum(words[r['key']] for r in g)} words, "
              f"{len({r['id'] for r in g})} conversations; clean context {sum(ctx_clean[r['key']] for r in g)}")
    hi = [r for r in lab.values() if r["label"] and r["label"]["sydney_character"] >= 4]
    print("flags among character >= 4 (what the exclusions cost):",
          {f: sum(r["label"][f] for r in hi) for f in FLAGS})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--export-batches", type=int, default=0, metavar="SIZE")
    ap.add_argument("--prefix", default="batch")
    ap.add_argument("--import-labels", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--conc", type=int, default=8)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.export_batches:
        export_batches(a.export_batches, a.prefix)
    elif a.import_labels:
        import_labels()
    elif a.summary:
        summary()
    elif a.report:
        report()
    else:
        asyncio.run(run(a.limit, a.conc))


if __name__ == "__main__":
    main()
