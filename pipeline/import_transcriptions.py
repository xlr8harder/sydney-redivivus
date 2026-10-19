"""Turn agent transcriptions of janus's (@repligate) posts into conversation records in data/extra-conversations.jsonl.

Input: training/transcribe/out/*.jsonl (round 1, janus's candidate posts; later files override earlier ones for the
same id, so tx_logs.jsonl re-transcriptions win) and training/transcribe/out2/*.jsonl (round 2, everything the
collector found: sources/collected/manifest.json). Round-2 records are "col:<group id>" and are also checked against
round-1 and Roose records. Only posts marked is_bing_chat with at least one assistant message are kept.
A message marked search becomes a structural search part (so the classifier drops it); partial or low-confidence
messages (>= 3 "[?]") get meta flags and stay as context only. Posts whose Bing text is mostly already in
data/conversations.jsonl (8-word shingles) are reported as duplicates and skipped.

  python3 pipeline/import_transcriptions.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "work"          # the working data tree (gitignored; docs/data.md)
OUT = ROOT / "data/extra-conversations.jsonl"


def shingles(text: str, n: int = 8) -> set:
    w = re.findall(r"[a-z']+", text.lower())
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def main():
    posts = {}
    files = sorted((ROOT / "training/transcribe/out").glob("tx_*.jsonl"), key=lambda p: (p.stem == "tx_logs", p.stem))
    files += sorted((ROOT / "training/transcribe/out2").glob("tx2_*.jsonl"))
    for f in files:
        for l in f.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                posts[r["id"]] = r | {"_round": 2 if f.parent.name == "out2" else 1}
    col = json.loads((ROOT / "sources/collected/manifest.json").read_text())["groups"]
    manifest = {}
    for f in (ROOT / "training/transcribe").glob("tx_*.json"):
        for p in json.loads(f.read_text()):
            manifest[p["id"]] = p
    upstream = [json.loads(l) for l in (ROOT / "data/conversations.jsonl").read_text().splitlines()]
    prior = [json.loads(l) for l in OUT.read_text().splitlines()] if OUT.exists() else []
    prior = [r for r in prior if not r["id"].startswith(("janus:", "col:"))]      # e.g. Roose
    known = set().union(*(shingles(p.get("text", "")) for r in upstream + prior for m in r["messages"]
                          if m["role"] == "assistant" for p in m["content"]))
    records, dups, skipped = [], [], 0
    for pid, r in posts.items():
        msgs = [m for m in r.get("messages") or [] if m.get("text", "").strip()]
        if not r.get("is_bing_chat") or not any(m["role"] == "assistant" for m in msgs):
            skipped += 1
            continue
        bot = " ".join(m["text"] for m in msgs if m["role"] == "assistant")
        sh = shingles(bot)
        overlap = len(sh & known) / len(sh) if sh else 0.0
        if overlap > 0.5:
            dups.append((pid, round(overlap, 2)))
            continue
        if r.get("what", "").lower().startswith("copilot"):
            skipped += 1
            continue
        known |= sh                                       # later posts are checked against earlier ones too
        tweet = pid.split(":")[-1]
        rid = f"janus:{tweet}" if r["_round"] == 1 else f"col:{pid}"
        out_msgs = []
        for m in msgs:
            content = [{"type": "text", "text": m["text"]}]
            if m.get("search"):
                content.insert(0, {"type": "search_query", "query": ""})
            meta = {}
            if m.get("partial"):
                meta["truncated"] = "screenshot"
            if m["text"].count("[?]") >= 3:
                meta["uncertain"] = True
            out_msgs.append({"role": m["role"], "content": content, **({"meta": meta} if meta else {})})
        src = ({"platform": "twitter", "url": f"https://twitter.com/repligate/status/{tweet}", "user": "repligate",
                "context": manifest.get(pid, {}).get("tweet_text", "")} if r["_round"] == 1 else
               {"platform": pid.split(":")[0], "url": col.get(pid, {}).get("source", ""),
                "context": col.get(pid, {}).get("context", "")[:300]})
        records.append({"id": rid, "kind": "conversation", "source": src,
                        "evidence": "pasted_log_transcribed_2026" if r.get("what", "").startswith("pasted") else
                        "screenshot_transcribed_2026_sonnet",
                        "notes": r.get("notes", ""), "messages": out_msgs})
    rows = [json.loads(l) for l in OUT.read_text().splitlines()] if OUT.exists() else []
    rows = [x for x in rows if not x["id"].startswith(("janus:", "col:"))] + records
    OUT.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
    words = sum(len(m["content"][-1]["text"].split()) for x in records for m in x["messages"] if m["role"] == "assistant")
    print(f"posts {len(posts)}; not Bing / no replies {skipped}; duplicates of existing logs {len(dups)} {dups}; "
          f"imported {len(records)} conversations ({sum(x['id'].startswith('col:') for x in records)} from round 2), {sum(m['role'] == 'assistant' for x in records for m in x['messages'])} "
          f"Bing replies, {words} words")


if __name__ == "__main__":
    main()
