"""Assemble the checkpoint repo (the Hugging Face model repo) into dist/model/: one folder per experiment that has a
trained-neuron checkpoint (experiments/<id>/checkpoint/), with its config, metrics and notes, and the card
(docs/model-card.md) with a table of them.

  python3 scripts/publish_checkpoints.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dist/model"


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    rows = []
    for d in sorted((ROOT / "experiments").iterdir()):
        ck = d / "checkpoint"
        if not ck.is_dir():
            continue
        shutil.copytree(ck, OUT / d.name)
        for f in ("config.json", "metrics.json", "notes.md"):
            shutil.copy(d / f, OUT / d.name / f)
        cfg, m = json.loads((d / "config.json").read_text()), json.loads((d / "metrics.json").read_text())
        args = cfg.get("args") or {}
        argv = cfg.get("argv", [])
        units = Path(args.get("units") or (argv[argv.index("--units") + 1] if "--units" in argv else "")).stem
        if args.get("random_units") or "--random-units" in argv:
            units = f"random ({units} layers)"
        split = args.get("val") or (argv[argv.index("--val") + 1] if "--val" in argv else "")
        split = "conversation" if split == "split" else "reply-level (old)"
        note = next((l.strip() for l in (d / "notes.md").read_text().splitlines()[1:] if l.strip()), "")
        rows.append(f"| [`{d.name}`]({d.name}/) | {units} | {split} | {m['val_base']:.3f} → {m['val_kept']:.3f} | "
                    f"{m.get('mmlu_trained', 0):.1%} | {note[:160]}{'…' if len(note) > 160 else ''} |")
    table = ("| experiment | neurons | split | val | MMLU | note |\n|---|---|---|---|---|---|\n" + "\n".join(rows))
    (OUT / "README.md").write_text((ROOT / "docs/model-card.md").read_text().replace("{{TABLE}}", table))
    print(f"{len(rows)} checkpoints -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
