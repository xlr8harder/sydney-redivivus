"""Cross-source deduplication of Sydney replies and conversations.

All conversation records (data/conversations.jsonl + data/extra-conversations.jsonl) are compared reply by reply:
two of her replies are the same reply when their 5-word shingles have Jaccard >= 0.6, or when the shorter one is
>= 0.85 contained in the longer (a screenshot cut-off copy, if the shorter has >= 20 shingles). Clusters get one canonical copy: complete beats cut off,
then fewer [?], then longer, then upstream before our own extractions. A record is redundant when every reply of it is
a non-canonical copy whose canonical sits in one other record that is at least as long; redundant records are dropped
entirely, the rest stay as context with only canonical replies eligible for loss.

Stock lines (short replies, < 25 words, recurring in >= 3 different conversations: "Thanks for clearing my head! What
can I help you with now?") are not duplicates: they are listed as boilerplate for the training builder to mask, and
they never count toward a record being redundant. A redundant record must share >= 2 substantive replies with its home.

  python3 pipeline/dedupe.py      # training/dedupe.json
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "work"          # the working data tree (gitignored; docs/data.md)


def text_of(m):
    return "\n\n".join(p.get("text", "") for p in m["content"] if p["type"] in ("text", "code_block")).strip()


def shingles(t, n=5):
    w = re.findall(r"[a-z0-9']+", t.lower())
    return {" ".join(w[i:i + n]) for i in range(max(1, len(w) - n + 1))} if len(w) >= n else ({" ".join(w)} if w else set())


def main():
    rows = [json.loads(l) for l in (ROOT / "data/conversations.jsonl").read_text().splitlines()]
    extra = ROOT / "data/extra-conversations.jsonl"
    rows += [json.loads(l) | {"_extra": True} for l in extra.read_text().splitlines()] if extra.exists() else []
    rows = [r for r in rows if r["kind"] == "conversation"]
    reps = []                                     # (key, record id, text, partial, extra)
    for r in rows:
        for i, m in enumerate(r["messages"]):
            t = text_of(m)
            if m["role"] == "assistant" and len(t.split()) >= 4:
                reps.append({"key": f"{r['id']}#{i}", "rid": r["id"], "text": t, "sh": shingles(t),
                             "partial": bool((m.get("meta") or {}).get("truncated")), "extra": bool(r.get("_extra"))})
    index = defaultdict(list)
    for j, x in enumerate(reps):
        for s in x["sh"]:
            index[s].append(j)
    parent = list(range(len(reps)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    pairs = 0
    for j, x in enumerate(reps):
        cand = defaultdict(int)
        for s in x["sh"]:
            if len(index[s]) > 200:               # boilerplate shingle ("hello this is bing how")
                continue
            for k in index[s]:
                if k > j:
                    cand[k] += 1
        for k, inter in cand.items():
            a, b = len(x["sh"]), len(reps[k]["sh"])
            if inter / (a + b - inter) >= 0.6 or inter / min(a, b) >= 0.85 and min(a, b) >= 20:   # containment only for substantial
                                                                            # pieces, so greetings can't bridge
                parent[find(j)] = find(k)
                pairs += 1
    clusters = defaultdict(list)
    for j in range(len(reps)):
        clusters[find(j)].append(j)
    canon, rank = {}, lambda x: (not x["partial"], -x["text"].count("[?]"), len(x["text"]), not x["extra"])
    boiler = set()
    for members in clusters.values():
        words = sorted(len(reps[j]["text"].split()) for j in members)
        if len({reps[j]["rid"] for j in members}) >= 3 and words[len(words) // 2] < 25:
            boiler.update(reps[j]["key"] for j in members)
            continue
        best = max(members, key=lambda j: rank(reps[j]))
        for j in members:
            canon[reps[j]["key"]] = reps[best]["key"]
    by_rec = defaultdict(list)
    for x in reps:
        by_rec[x["rid"]].append(x["key"])
        canon.setdefault(x["key"], x["key"])
    nmsg = {r["id"]: len(r["messages"]) for r in rows}
    redundant = []
    for rid, keys in by_rec.items():
        keys = [k for k in keys if k not in boiler]
        homes = {canon[k].rsplit("#", 1)[0] for k in keys}
        if len(keys) >= 2 and all(canon[k] != k for k in keys) and len(homes) == 1:
            home = homes.pop()
            if home != rid and nmsg[home] >= nmsg[rid]:
                redundant.append([rid, home])
    dup_reps = sum(canon[x["key"]] != x["key"] for x in reps)
    cross = sum(1 for m in clusters.values() if len(m) > 1 and len({reps[j]["extra"] for j in m}) > 1)
    out = {"reply_canonical": {k: v for k, v in canon.items() if k != v}, "redundant_records": redundant,
           "boilerplate": sorted(boiler),
           "stats": {"records": len(rows), "replies": len(reps), "duplicate_replies": dup_reps,
                     "clusters_with_copies": sum(len(m) > 1 for m in clusters.values()),
                     "clusters_spanning_upstream_and_extra": cross, "redundant_records": len(redundant),
                     "boilerplate_replies": len(boiler)}}
    (ROOT / "training/dedupe.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["stats"], indent=1))
    big = sorted((m for m in clusters.values() if reps[m[0]]["key"] not in boiler), key=len, reverse=True)[:4]
    for m in big:
        print(f"  cluster of {len(m)}: {reps[m[0]]['text'][:100]!r} … in {sorted({reps[j]['rid'] for j in m})[:5]}")


if __name__ == "__main__":
    main()
