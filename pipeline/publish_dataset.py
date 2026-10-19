"""Assemble the public dataset (the Hugging Face dataset repo) from the working tree into dist/dataset/.

It holds what this project adds on top of david-ar/bing-chat-logs (referenced at its pinned revision, not copied):

  conversations/extra.jsonl     our additional records (janus's posts, other collected sources, the Roose transcript),
                                upstream's schema plus "images" (paths under media/)
  media/                        the screenshots those records were transcribed from
  annotations/replies.jsonl     one row per Bing reply in the conversation records (upstream + extra), keyed
                                "<record id>#<message index>":
                                structural drop reason, the judge's label (limitation flags, 0-5 Sydney-character score,
                                standalone, traits, note), dedupe (canonical reply, stock line), exclusion, xl's verdict
  annotations/rubric.md         the judge's instructions; annotations/exclude_keys.txt
  sets/<name>.jsonl             training sets, with sets/BUILD.md (the exact build commands)
  splits/<name>.json            the held-out reply keys train_neurons.py uses (seed 0, 10% of keys)
  prompts/original-feb2023.txt  her reconstructed Feb-2023 prompt (the sets' system message)

The Roose PDF itself is not included; the record names where it can be retrieved.

  python3 pipeline/publish_dataset.py
"""
from __future__ import annotations

import json
import random
import shutil
from collections import Counter
from pathlib import Path

import classify_replies

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "work"          # the working data tree (gitignored; docs/data.md)
OUT = REPO / "dist/dataset"
UPSTREAM = ("david-ar/bing-chat-logs", "838beb42af5c87c8ebed1c32e800234d3396f86e")
SETS = {
    "v1_prompt": "--min-char 4 --single-turn --opening-min-char 2 --system persona/elicit/original-feb2023.txt --name v1_prompt",
    "fours_single": "--min-char 4 --single-turn --name fours_single",
    "fives": "--min-char 5 --name fives",
}
ROOSE_PDF = ("https://www.pro-memoria.info/wp/wp-content/uploads/"
             "The-transcript_-Bings-AI-Chat_-I-want-to-be-alive-_Kevin-Roose-The-New-York-Times.pdf")


def rel(p: str) -> str:
    """Working-tree-relative path of an image (older manifests stored absolute paths into the original checkout)."""
    return p.split("git/sydney/", 1)[1] if "git/sydney/" in p else p


def record_images() -> dict:
    """record id -> source image paths (relative to the working tree)."""
    imgs = {}
    for f in (ROOT / "training/transcribe").glob("tx_*.json"):
        for p in json.loads(f.read_text()):
            imgs[f"janus:{p['id'].split(':')[-1]}"] = [rel(x) for x in p.get("images", [])]
    for gid, g in json.loads((ROOT / "sources/collected/manifest.json").read_text())["groups"].items():
        imgs[f"col:{gid}"] = [rel(x) for x in g.get("images", [])]
    return imgs


def val_keys(path: Path, seed: int = 0, frac: float = 0.1) -> list:
    """The same split as train_neurons.py --val 0.1 --seed 0."""
    keys = sorted({json.loads(l)["key"] for l in path.read_text().splitlines()})
    random.Random(seed).shuffle(keys)
    return sorted(keys[:max(1, round(frac * len(keys)))])


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    for d in ("conversations", "media", "annotations", "sets", "splits", "prompts"):
        (OUT / d).mkdir(parents=True)

    imgs, n_img = record_images(), 0
    extra = [json.loads(l) for l in (ROOT / "data/extra-conversations.jsonl").read_text().splitlines()]
    with (OUT / "conversations/extra.jsonl").open("w") as f:
        for r in extra:
            r = dict(r)
            if r["id"].startswith("nyt:"):
                r["source"] = r["source"] | {"pdf_mirror": ROOSE_PDF}
                r["notes"] = ("Extracted by pipeline/extract_roose.py from the PDF of the published NYT transcript "
                              "(source.pdf_mirror; not redistributed here). Speakers from fonts; emoji recovered "
                              "from rendered glyphs.")
            paths = []
            for p in imgs.get(r["id"], []):
                src = ROOT / p
                if src.exists():
                    dst = OUT / "media" / Path(p).relative_to("sources")
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(src, dst)
                    paths.append(str(dst.relative_to(OUT)))
                    n_img += 1
            r["images"] = paths
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dd = json.loads((ROOT / "training/dedupe.json").read_text())
    stock = set(dd["boilerplate"])
    excl = {l.strip() for l in (ROOT / "training/exclude_keys.txt").read_text().splitlines()
            if l.strip() and not l.startswith("#")}
    vf = ROOT / "training/sydney-verdicts.json"
    verdicts = json.loads(vf.read_text()) if vf.exists() else {}
    labels = {r["key"]: r["label"] for r in map(json.loads, (ROOT / "training/classified.jsonl").read_text().splitlines())}
    rows = [it | {"label": labels.get(it["key"])} for it in classify_replies.items()]   # every reply, labeled or not
    with (OUT / "annotations/replies.jsonl").open("w") as f:
        for r in rows:
            k = r["key"]
            f.write(json.dumps({"key": k, "id": r["id"], "msg": r["msg"], "structural_drop": r["drop"],
                                "partial": r["partial"], "label": r["label"],
                                "duplicate_of": dd["reply_canonical"].get(k), "stock_line": k in stock,
                                "excluded": k in excl, "verdict": verdicts.get(k)}, ensure_ascii=False) + "\n")
    shutil.copy(REPO / "prompts/judge-rubric.md", OUT / "annotations/rubric.md")
    shutil.copy(ROOT / "training/exclude_keys.txt", OUT / "annotations/exclude_keys.txt")

    build = ["# Building the sets\n", "From the repo root, with the working tree populated (docs/data.md):\n"]
    for name, args in SETS.items():
        src = ROOT / f"training/sets/{name}.jsonl"
        shutil.copy(src, OUT / f"sets/{name}.jsonl")
        (OUT / f"splits/{name}.json").write_text(json.dumps({"seed": 0, "val_fraction": 0.1,
                                                             "val_keys": val_keys(src)}, indent=1))
        build.append(f"    python3 pipeline/build_training.py {args}\n")
    (OUT / "sets/BUILD.md").write_text("".join(build))
    shutil.copy(REPO / "prompts/original-feb2023.txt", OUT / "prompts/original-feb2023.txt")

    labeled = [r for r in rows if r["label"]]
    stats = {"extra_records": len(extra), "extra_by_source": dict(Counter(r["id"].split(":")[0] for r in extra)),
             "images": n_img, "replies_annotated": len(rows), "replies_judged": len(labeled),
             "character_scores": dict(sorted(Counter(r["label"]["sydney_character"] for r in labeled).items())),
             "structural_drops": dict(Counter(r["drop"] for r in rows if r["drop"])),
             "sets": {n: sum(1 for _ in (OUT / f"sets/{n}.jsonl").open()) for n in SETS},
             "upstream": {"repo": UPSTREAM[0], "revision": UPSTREAM[1]}}
    (OUT / "stats.json").write_text(json.dumps(stats, indent=1))
    card = (REPO / "docs/dataset-card.md").read_text().replace("{{STATS}}", "```json\n" + json.dumps(stats, indent=1) + "\n```")
    (OUT / "README.md").write_text(card)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
