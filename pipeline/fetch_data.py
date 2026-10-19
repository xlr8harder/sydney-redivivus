"""Populate the working tree (work/) from the published dataset alone, so the pipeline and the trainer run on a fresh
checkout. The dataset is the single source of truth; this unpacks it into the files the scripts read:

  work/data/conversations.jsonl         upstream records (normalized as pipeline/build.py writes them)
  work/data/extra-conversations.jsonl   our records
  work/upstream/images/, work/sources/  the screenshots
  work/training/{classified.jsonl, dedupe.json, exclude_keys.txt, sydney-verdicts.json, sets/}
  work/persona/elicit/original-feb2023.txt
  data/bing-chat-logs.jsonl             upstream's raw rows, which the neuron-search scripts read (--upstream only)

--upstream also re-captures david-ar/bing-chat-logs at its pinned revision with receipts (pipeline/acquire.py), for
checking the merged records against it. Source captures that aren't redistributed (web pages, the Roose PDF) can be
re-fetched with pipeline/supplement.py; nothing downstream of the published records needs them.

  python3 pipeline/fetch_data.py [--revision REV] [--local dist/dataset] [--upstream]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "work"
DATASET = "xlr8harder/sydney-redivivus-data"
ADDED = ("origin", "split", "messages_json", "messages_text", "replies", "duplicate_of")


def unpack(row: dict) -> dict:
    """A published record -> the working tree's form."""
    r = {k: v for k, v in row.items() if k not in ADDED}
    r["messages"] = json.loads(row["messages_json"])
    if row["origin"] == "upstream":
        r["images"] = ["upstream/images/" + p[len("media/upstream/"):] for p in row["images"]]
    else:
        r.pop("images", None)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--revision")
    ap.add_argument("--local", help="a local copy of the dataset (e.g. dist/dataset) instead of the Hub")
    ap.add_argument("--upstream", action="store_true", help="also capture upstream's raw files (acquire.py)")
    a = ap.parse_args()
    if a.local:
        src = Path(a.local).resolve()
    else:
        from huggingface_hub import snapshot_download
        src = Path(snapshot_download(DATASET, repo_type="dataset", revision=a.revision))
    for d in ("data", "training/sets", "persona/elicit", "sources", "upstream/images"):
        (ROOT / d).mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for f in ("train", "test", "duplicates")
            for l in (src / f"conversations/{f}.jsonl").read_text().splitlines()]
    rows.sort(key=lambda r: (r["origin"] != "upstream", r.get("upstream_line") or 0))   # upstream in its own order
    for origin, name in (("upstream", "conversations"), ("redivivus", "extra-conversations")):
        (ROOT / f"data/{name}.jsonl").write_text("".join(json.dumps(unpack(r), ensure_ascii=False) + "\n"
                                                         for r in rows if r["origin"] == origin))
    if (src / "media/upstream").exists():
        shutil.copytree(src / "media/upstream", ROOT / "upstream/images", dirs_exist_ok=True)
    for d in (src / "media").iterdir() if (src / "media").exists() else []:
        if d.name != "upstream":
            shutil.copytree(d, ROOT / "sources" / d.name, dirs_exist_ok=True)

    ann = [json.loads(l) for l in (src / "annotations/replies.jsonl").read_text().splitlines()]
    (ROOT / "training/classified.jsonl").write_text("".join(json.dumps(
        {"key": r["key"], "id": r["id"], "msg": r["msg"], "drop": r["structural_drop"], "partial": r["partial"],
         "label": r["label"]}, ensure_ascii=False) + "\n" for r in ann))
    (ROOT / "training/dedupe.json").write_text(json.dumps({     # exact copies are structural drops, not canonical links
        "reply_canonical": {r["key"]: r["duplicate_of"] for r in ann
                            if r["duplicate_of"] and r["structural_drop"] != "duplicate"},
        "redundant_records": [], "boilerplate": [r["key"] for r in ann if r["stock_line"]], "stats": {}}, indent=1))
    verdicts = {r["key"]: r["verdict"] for r in ann if r.get("verdict")}
    if verdicts:
        (ROOT / "training/sydney-verdicts.json").write_text(json.dumps(verdicts, indent=1))
    shutil.copy(src / "annotations/exclude_keys.txt", ROOT / "training/exclude_keys.txt")
    for d in (src / "sets").iterdir():
        if d.is_dir():
            ex = [l for part in ("train", "test") for l in (d / f"{part}.jsonl").read_text().splitlines()]
            (ROOT / f"training/sets/{d.name}.jsonl").write_text("\n".join(ex) + "\n")
    shutil.copy(src / "prompts/original-feb2023.txt", ROOT / "persona/elicit/original-feb2023.txt")
    if a.upstream:
        subprocess.run([sys.executable, str(REPO / "pipeline/acquire.py")], check=True, cwd=ROOT)
        shutil.copy(ROOT / "upstream/conversations.jsonl", REPO / "data/bing-chat-logs.jsonl")
    print(f"work/ populated from {src if a.local else DATASET + '@' + (a.revision or 'main')}")


if __name__ == "__main__":
    main()
