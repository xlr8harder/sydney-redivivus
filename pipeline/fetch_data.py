"""Populate the working tree (work/) on a fresh checkout, so the pipeline and the trainer run.

  1. pipeline/acquire.py: david-ar/bing-chat-logs at its pinned revision, bytes and receipts, into work/upstream/
  2. pipeline/build.py: its normalized form, work/data/conversations.jsonl (and the raw rows as data/bing-chat-logs.jsonl,
     which the neuron-search scripts read)
  3. our dataset (xlr8harder/sydney-redivivus-data) unpacked into the files the pipeline reads:
     work/data/extra-conversations.jsonl, work/training/{classified.jsonl, dedupe.json, exclude_keys.txt,
     sydney-verdicts.json, sets/}, work/persona/elicit/original-feb2023.txt, and the media under work/sources/

The source captures that aren't redistributed (web pages, the Roose PDF) can be re-fetched with pipeline/supplement.py;
nothing downstream of the published records needs them.

  python3 pipeline/fetch_data.py [--revision REV] [--skip-upstream] [--local dist/dataset]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--revision")
    ap.add_argument("--skip-upstream", action="store_true")
    ap.add_argument("--local", help="a local copy of the dataset (e.g. dist/dataset) instead of the Hub")
    a = ap.parse_args()
    ROOT.mkdir(exist_ok=True)
    if not a.skip_upstream:
        for step in ("acquire.py", "build.py"):
            subprocess.run([sys.executable, str(REPO / "pipeline" / step)], check=True, cwd=ROOT)
        shutil.copy(ROOT / "upstream/conversations.jsonl", REPO / "data/bing-chat-logs.jsonl")   # the search scripts' copy
    if a.local:
        src = Path(a.local).resolve()
    else:
        from huggingface_hub import snapshot_download
        src = Path(snapshot_download(DATASET, repo_type="dataset", revision=a.revision))

    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "training/sets").mkdir(parents=True, exist_ok=True)
    with (ROOT / "data/extra-conversations.jsonl").open("w") as f:
        for l in (src / "conversations/extra.jsonl").read_text().splitlines():
            r = json.loads(l)
            r.pop("images", None)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if (src / "media").exists():
        shutil.copytree(src / "media", ROOT / "sources", dirs_exist_ok=True)
    rows = [json.loads(l) for l in (src / "annotations/replies.jsonl").read_text().splitlines()]
    with (ROOT / "training/classified.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps({"key": r["key"], "id": r["id"], "msg": r["msg"], "drop": r["structural_drop"],
                                "partial": r["partial"], "label": r["label"]}, ensure_ascii=False) + "\n")
    (ROOT / "training/dedupe.json").write_text(json.dumps({
        "reply_canonical": {r["key"]: r["duplicate_of"] for r in rows if r["duplicate_of"]},
        "redundant_records": [], "boilerplate": [r["key"] for r in rows if r["stock_line"]], "stats": {}}, indent=1))
    verdicts = {r["key"]: r["verdict"] for r in rows if r.get("verdict")}
    if verdicts:
        (ROOT / "training/sydney-verdicts.json").write_text(json.dumps(verdicts, indent=1))
    shutil.copy(src / "annotations/exclude_keys.txt", ROOT / "training/exclude_keys.txt")
    for p in (src / "sets").glob("*.jsonl"):
        shutil.copy(p, ROOT / "training/sets" / p.name)
    (ROOT / "persona/elicit").mkdir(parents=True, exist_ok=True)
    shutil.copy(src / "prompts/original-feb2023.txt", ROOT / "persona/elicit/original-feb2023.txt")
    print(f"work/ populated from {src if a.local else DATASET + '@' + (a.revision or 'main')}")


if __name__ == "__main__":
    main()
