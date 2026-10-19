"""Shared views of the working tree: every record, every Bing reply's annotation, whole-record reposts, and the
canonical train/test split. Used by build_training.py (each example's "split") and publish_dataset.py.

The split is by conversation, so a test reply is never context in a training example. Records are grouped when they
share a reply (a near-duplicate copy from dedupe.py, or an exact copy) or when one only reposts another; a group is
test when sha256(SALT + its smallest record id) mod 100 < TEST_PCT. Hashing keeps every assignment fixed as records are
added. At 15% of groups, 12% of v1_prompt's examples are test.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import classify_replies

ROOT = Path(__file__).resolve().parents[1] / "work"          # the working data tree (gitignored; docs/data.md)
SALT, TEST_PCT = "sydney-redivivus-split-v1", 15


def records() -> tuple[dict, dict]:
    """id -> record, id -> origin ("upstream" | "redivivus")."""
    recs, origin = {}, {}
    for f, o in (("data/conversations.jsonl", "upstream"), ("data/extra-conversations.jsonl", "redivivus")):
        for l in (ROOT / f).read_text().splitlines():
            r = json.loads(l)
            recs[r["id"]], origin[r["id"]] = r, o
    return recs, origin


def reply_annotations() -> list:
    """One row per Bing reply of every conversation record, labeled or not."""
    dd = json.loads((ROOT / "training/dedupe.json").read_text())
    stock = set(dd["boilerplate"])
    excl = {l.strip() for l in (ROOT / "training/exclude_keys.txt").read_text().splitlines()
            if l.strip() and not l.startswith("#")}
    vf = ROOT / "training/sydney-verdicts.json"
    verdicts = json.loads(vf.read_text()) if vf.exists() else {}
    labels = {r["key"]: r["label"] for r in map(json.loads, (ROOT / "training/classified.jsonl").read_text().splitlines())}
    first, rows = {}, []                           # exact copies (structural "duplicate") point at the first occurrence
    for it in classify_replies.items():
        h = hashlib.sha256(it["reply"].encode()).hexdigest()
        first.setdefault(h, it["key"])
        dup = dd["reply_canonical"].get(it["key"]) or (first[h] if it["drop"] == "duplicate" else None)
        rows.append({"key": it["key"], "id": it["id"], "msg": it["msg"], "structural_drop": it["drop"],
                     "partial": it["partial"], "label": labels.get(it["key"]), "duplicate_of": dup,
                     "stock_line": it["key"] in stock, "excluded": it["key"] in excl, "verdict": verdicts.get(it["key"])})
    return rows


def duplicate_records(recs: dict, rows: list) -> dict:
    """record id -> the record it only reposts: every substantive reply is a copy, all copies of replies in one
    other record that is at least as long."""
    by_rec = defaultdict(list)
    for r in rows:
        by_rec[r["id"]].append(r)
    out = {}
    for rid, rs in by_rec.items():
        subst = [r for r in rs if r["structural_drop"] not in ("structural_tool", "safety_block", "empty")
                 and not r["stock_line"]]
        homes = {r["duplicate_of"].rsplit("#", 1)[0] if r["duplicate_of"] else None for r in subst}
        if subst and len(homes) == 1 and None not in homes:
            home = homes.pop()
            if home != rid and len(recs[home]["messages"]) >= len(recs[rid]["messages"]) and home not in out:
                out[rid] = home
    return out


def splits(recs: dict, rows: list, dups: dict) -> dict:
    """record id -> "train" | "test"."""
    par = {rid: rid for rid in recs}

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    links = [(r["id"], r["duplicate_of"].rsplit("#", 1)[0]) for r in rows if r["duplicate_of"]] + list(dups.items())
    for a, b in links:
        par[find(a)] = find(b)
    groups = defaultdict(list)
    for rid in recs:
        groups[find(rid)].append(rid)
    out = {}
    for members in groups.values():
        h = int(hashlib.sha256((SALT + min(members)).encode()).hexdigest(), 16) % 100
        for rid in members:
            out[rid] = "test" if h < TEST_PCT else "train"
    return out


def canonical_splits() -> dict:
    recs, _ = records()
    rows = reply_annotations()
    return splits(recs, rows, duplicate_records(recs, rows))
