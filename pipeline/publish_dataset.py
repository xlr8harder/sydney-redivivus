"""Assemble the public dataset (the Hugging Face dataset repo) from the working tree into dist/dataset/: every Bing
conversation we have, merged and deduplicated, with this project's annotations, loadable from one place.

  conversations/{train,test}.jsonl  every unique record, by the canonical split: david-ar/bing-chat-logs (CC0 assembly, pinned revision) and ours
                                (janus's posts, other collected sources, the Roose transcript). One row per record:
                                id, origin ("upstream" | "redivivus"), kind, source, provenance fields, images (paths
                                under media/), messages_json (the typed messages, as upstream ships them),
                                messages_text (role + plain text), replies (one annotation per Bing reply, below)
  conversations/duplicates.jsonl  records that only repeat another record (same conversation reposted; every
                                substantive reply is a copy and the copies all sit in one record at least as long),
                                with duplicate_of. Kept for completeness; the pipeline still reads them.
  media/                        every screenshot the records cite: media/upstream/..., media/collected/..., ...
  annotations/replies.jsonl     the same reply annotations, flat, keyed "<record id>#<message index>":
                                structural drop reason, the judge's label (limitation flags, 1-5 Sydney-character
                                score, standalone, traits, note), duplicate_of (near-duplicate reply), stock_line,
                                excluded, verdict (hand review)
  annotations/rubric.md         the judge's instructions; annotations/exclude_keys.txt
  sets/<name>/{train,test}.jsonl  training sets by the canonical split, with sets/BUILD.md (the build commands)
  splits/<name>.as_run.json     the reply-level held-out keys the 2026-10-18/19 runs used, for reproducing them

Every record, reply annotation and training example carries "split" ("train" | "test"): the canonical split, by
conversation (records.py), so a test reply never appears as context in a training example.
  prompts/original-feb2023.txt  her reconstructed Feb-2023 prompt (the sets' system message)

Dedup happens at three levels: posts whose Bing text was already in the collection are skipped at import
(import_transcriptions.py, 8-word shingles); near-identical replies are clustered (dedupe.py) and every copy names its
canonical reply; whole-record reposts go to duplicates.jsonl. The Roose PDF is not included; the record names where it
can be retrieved.

  python3 pipeline/publish_dataset.py
"""
from __future__ import annotations

import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import classify_replies
import records

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "work"          # the working data tree (gitignored; docs/data.md)
OUT = REPO / "dist/dataset"
UPSTREAM = ("david-ar/bing-chat-logs", "838beb42af5c87c8ebed1c32e800234d3396f86e")
SETS = {                                       # the current sets (fives and fours_single were early tests, superseded)
    "v1_prompt": "--min-char 4 --single-turn --opening-min-char 2 --system persona/elicit/original-feb2023.txt --name v1_prompt",
}
ROOSE_PDF = ("https://www.pro-memoria.info/wp/wp-content/uploads/"
             "The-transcript_-Bings-AI-Chat_-I-want-to-be-alive-_Kevin-Roose-The-New-York-Times.pdf")


def rel(p: str) -> str:
    """Working-tree-relative path of an image (older manifests stored absolute paths into the original checkout)."""
    return p.split("git/sydney/", 1)[1] if "git/sydney/" in p else p


def media_path(p: str) -> str:
    """Working-tree image path -> its path in the dataset."""
    return "media/upstream/" + p[len("upstream/images/"):] if p.startswith("upstream/images/") else \
        "media/" + str(Path(p).relative_to("sources"))


def extra_images() -> dict:
    """record id -> source image paths for our records (from the transcription and collection manifests)."""
    imgs = {}
    for f in (ROOT / "training/transcribe").glob("tx_*.json"):
        for p in json.loads(f.read_text()):
            imgs[f"janus:{p['id'].split(':')[-1]}"] = [rel(x) for x in p.get("images", [])]
    for gid, g in json.loads((ROOT / "sources/collected/manifest.json").read_text())["groups"].items():
        imgs[f"col:{gid}"] = [rel(x) for x in g.get("images", [])]
    return imgs


def as_run_val_keys(path: Path, seed: int = 0, frac: float = 0.1) -> list:
    """The reply-level split the 2026-10-18/19 runs used (train_neurons.py --val 0.1 --seed 0), for reproducing them."""
    keys = sorted({json.loads(l)["key"] for l in path.read_text().splitlines()})
    random.Random(seed).shuffle(keys)
    return sorted(keys[:max(1, round(frac * len(keys)))])


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    for d in ("conversations", "media", "annotations", "sets", "splits", "prompts"):
        (OUT / d).mkdir(parents=True)

    recs, origin = records.records()
    rows = records.reply_annotations()
    dups = records.duplicate_records(recs, rows)
    split = records.splits(recs, rows, dups)
    for r in rows:
        r["split"] = split[r["id"]]
    by_rec = defaultdict(list)
    for r in rows:
        by_rec[r["id"]].append({k: v for k, v in r.items() if k != "id"})

    imgs, n_img = extra_images(), 0
    fs = {part: (OUT / f"conversations/{part}.jsonl").open("w") for part in ("train", "test")}
    with (OUT / "conversations/duplicates.jsonl").open("w") as fd:
        for rid, r in recs.items():
            r = dict(r)
            if rid.startswith("nyt:"):
                r["source"] = r["source"] | {"pdf_mirror": ROOSE_PDF}
                r["notes"] = ("Extracted by pipeline/extract_roose.py from the PDF of the published NYT transcript "
                              "(source.pdf_mirror; not redistributed here). Speakers from fonts; emoji recovered "
                              "from rendered glyphs.")
            paths = []
            for p in (r.get("images") or []) if origin[rid] == "upstream" else imgs.get(rid, []):
                if (ROOT / p).exists():
                    dst = OUT / media_path(p)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(ROOT / p, dst)
                    paths.append(media_path(p))
                    n_img += 1
            msgs = r.pop("messages")
            row = {"id": rid, "origin": origin[rid], "split": split[rid], **{k: v for k, v in r.items() if k != "id"},
                   "images": paths,
                   "messages_json": json.dumps(msgs, ensure_ascii=False),
                   "messages_text": [{"role": m["role"], "text": classify_replies.text_of(m)} for m in msgs],
                   "replies": by_rec.get(rid, [])}
            if rid in dups:
                fd.write(json.dumps(row | {"duplicate_of": dups[rid]}, ensure_ascii=False) + "\n")
            else:
                fs[split[rid]].write(json.dumps(row, ensure_ascii=False) + "\n")
    for f in fs.values():
        f.close()

    with (OUT / "annotations/replies.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    shutil.copy(REPO / "prompts/judge-rubric.md", OUT / "annotations/rubric.md")
    shutil.copy(ROOT / "training/exclude_keys.txt", OUT / "annotations/exclude_keys.txt")

    build = ["# Building the sets\n\n", "From the repo root, with the working tree populated (docs/data.md):\n\n"]
    for name, args in SETS.items():
        src = ROOT / f"training/sets/{name}.jsonl"
        ex = [json.loads(l) for l in src.read_text().splitlines()]
        (OUT / f"sets/{name}").mkdir()
        for part in ("train", "test"):
            (OUT / f"sets/{name}/{part}.jsonl").write_text(
                "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in ex if x["split"] == part))
        (OUT / f"splits/{name}.as_run.json").write_text(json.dumps({
            "note": "reply-level split used by the 2026-10-18/19 runs (train_neurons.py --val 0.1 --seed 0); "
                    "superseded by the conversation-level 'split' field", "val_keys": as_run_val_keys(src)}, indent=1))
        build.append(f"    python3 pipeline/build_training.py {args}\n")
    (OUT / "sets/BUILD.md").write_text("".join(build))
    shutil.copy(REPO / "prompts/original-feb2023.txt", OUT / "prompts/original-feb2023.txt")

    labeled = [r for r in rows if r["label"]]
    kept = [rid for rid in recs if rid not in dups]
    stats = {"records": len(kept), "by_origin": dict(Counter(origin[i] for i in kept)),
             "by_kind": dict(Counter(recs[i]["kind"] for i in kept)), "duplicate_records": len(dups),
             "images": n_img, "bing_replies_annotated": len(rows), "judged": len(labeled),
             "character_scores": dict(sorted(Counter(r["label"]["sydney_character"] for r in labeled).items())),
             "structural_drops": dict(Counter(r["structural_drop"] for r in rows if r["structural_drop"])),
             "duplicate_replies": sum(1 for r in rows if r["duplicate_of"]),
             "sets": {n: {p: sum(1 for _ in (OUT / f"sets/{n}/{p}.jsonl").open()) for p in ("train", "test")}
                      for n in SETS},
             "records_by_split": dict(Counter(split[i] for i in kept)),
             "upstream": {"repo": UPSTREAM[0], "revision": UPSTREAM[1]}}
    (OUT / "stats.json").write_text(json.dumps(stats, indent=1))
    card = (REPO / "docs/dataset-card.md").read_text().replace("{{STATS}}", "```json\n" + json.dumps(stats, indent=1) + "\n```")
    (OUT / "README.md").write_text(card)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
