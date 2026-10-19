"""Build a training set from the classified replies: one example per selected reply of hers, with the conversation
before it as context (role/content messages; the loss goes on the final assistant message only).

Selection: Haiku labels (no exclusion flags, sydney_character within --min/--max), not cut off by a screenshot, not a
stock line (training/dedupe.json boilerplate), the canonical copy of any duplicate, and not dropped by the user's
browser verdicts (training/sydney-verdicts.json, if present; "keep" verdicts also admit flagged replies).
Context keeps every earlier turn, including masked/unusable ones, so threads stay whole.

  .venv/bin/python pipeline/build_training.py --min-char 5 --name fives
  --single-turn  also add a one-turn copy (that user message -> her reply, no earlier context) of every selected reply
                 that opens a conversation or that the judge marked standalone ("variant": "single")
  --system FILE  put FILE's text in the system slot of every example (e.g. persona/elicit/original-feb2023.txt)
  --opening-min-char N  also admit confirmed conversation openings (her reply to the first user message of a record
                 whose start is not truncated) from character N up: they teach how she opens and who she says she is
training/exclude_keys.txt (if present) lists replies to leave out entirely (e.g. transcription quirks).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import classify_replies as C

ROOT = Path(__file__).resolve().parents[1] / "work"          # the working data tree (gitignored; docs/data.md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-char", type=int, default=4)
    ap.add_argument("--max-char", type=int, default=5)
    ap.add_argument("--name", required=True)
    ap.add_argument("--single-turn", action="store_true")
    ap.add_argument("--system")
    ap.add_argument("--opening-min-char", type=int, default=0)
    a = ap.parse_args()
    lab = {}
    for l in C.OUT.read_text().splitlines():
        r = json.loads(l)
        lab[r["key"]] = r
    dd = json.loads((ROOT / "training/dedupe.json").read_text())
    boiler, canon = set(dd["boilerplate"]), dd["reply_canonical"]
    vf = ROOT / "training/sydney-verdicts.json"
    verdicts = json.loads(vf.read_text()) if vf.exists() else {}
    out, why = [], {}
    ex = ROOT / "training/exclude_keys.txt"
    excluded = {l.strip() for l in ex.read_text().splitlines() if l.strip() and not l.startswith("#")} if ex.exists() else set()
    recs = {}
    for f in ("data/conversations.jsonl", "data/extra-conversations.jsonl"):
        if (ROOT / f).exists():
            for l in (ROOT / f).read_text().splitlines():
                r = json.loads(l)
                recs[r["id"]] = r
    def opening(it):
        r = recs.get(it["id"], {})
        confirmed = r.get("id", "").startswith("nyt:") or r.get("truncated_start") is False
        return confirmed and sum(m["role"] == "user" for m in it["context"]) == 1
    system = (ROOT / a.system).read_text().strip() if a.system else None
    extra = lambda d: d | ({"system": system} if system else {})
    for it in C.items():
        r, k = lab.get(it["key"]), it["key"]
        if not r or not r["label"]:
            continue
        L = r["label"]
        flagged = any(L[f] for f in C.FLAGS)
        ok = not flagged and a.min_char <= L["sydney_character"] <= a.max_char
        is_open = bool(a.opening_min_char) and not flagged and opening(it) and L["sydney_character"] >= a.opening_min_char
        ok = ok or is_open
        reason = None
        if k in excluded:
            ok, reason = False, "excluded key"
        if verdicts.get(k) == "drop":
            ok, reason = False, "verdict"
        elif verdicts.get(k) == "keep" and a.min_char <= L["sydney_character"] <= a.max_char:
            ok = True
        if ok and it["partial"]:
            ok, reason = False, "cut off"
        if ok and k in boiler:
            ok, reason = False, "stock line"
        if ok and canon.get(k, k) != k:
            ok, reason = False, "duplicate"
        if reason:
            why[reason] = why.get(reason, 0) + 1
        if ok:
            reply = {"role": "assistant", "content": it["reply"]}
            out.append(extra({"key": k, "variant": "opening" if is_open and opening(it) else "thread",
                              "char": L["sydney_character"], "traits": L["traits"],
                              "messages": it["context"] + [reply]}))
            users = [m for m in it["context"] if m["role"] == "user"]
            if a.single_turn and (L["standalone"] or len(users) == 1) and len(it["context"]) > 1:
                out.append(extra({"key": k, "variant": "single", "char": L["sydney_character"], "traits": L["traits"],
                                  "messages": [it["context"][-1], reply]}))
    path = ROOT / f"training/sets/{a.name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out))
    nv = {v: sum(x["variant"] == v for x in out) for v in ("opening", "thread", "single")}
    print(f"{len(out)} examples {nv} -> {path} (excluded after labels: {why}); "
          f"reply words {sum(len(x['messages'][-1]['content'].split()) for x in out)}")


if __name__ == "__main__":
    main()
