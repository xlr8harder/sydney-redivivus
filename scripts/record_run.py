"""Package a train_neurons.py output directory as an experiment record, experiments/<id>/:

  config.json    the run's config.json, or for runs from before the trainer wrote one, --argv reconstructed by hand
  log.jsonl      training curve (train loss every step, val every 20)
  metrics.json   val (base / best / final / kept step), MMLU base and trained
  samples.jsonl  every probe and held-out prompt with base, trained and (held-out) her real reply
  units.json     the neurons trained
  delta.json     delta arm only (128 numbers)
  notes.md       what we learned (written by hand; kept if it exists)
  checkpoint/    neurons arms: neurons.safetensors + neurons.json + PEFT adapter (gitignored; published to the Hub)
                 steps/step_NNNN/: the same (Sauers' plain form, no adapter) at every validation step

  python3 scripts/record_run.py runs/train/full_AF 2026-10-19-full-AF
  python3 scripts/record_run.py ~/old/runs/train/v1_delta_patch 2026-10-19-delta-A --argv "--data ... --units ..."
"""
from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def metrics_of(run: Path) -> dict:
    if (run / "metrics.json").exists():
        return json.loads((run / "metrics.json").read_text())
    rows = [json.loads(l) for l in (run / "log.jsonl").read_text().splitlines()]
    vals = [(r["step"], r["val"]) for r in rows if "val" in r and "step" in r]
    kept = next((r["best_step"] for r in rows if "best_step" in r), vals[-1][0])
    m = {"val_base": vals[0][1], "val_final": vals[-1][1], "val_best": min(v for _, v in vals),
         "best_step": min(vals, key=lambda x: x[1])[0], "kept_step": kept, "val_kept": dict(vals)[kept]}
    if (run / "mmlu.json").exists():
        mm = json.loads((run / "mmlu.json").read_text())
        m |= {"mmlu_n": 120, "mmlu_base": mm["base"], "mmlu_trained": mm["trained"]}
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("id")
    ap.add_argument("--argv", help="the run's arguments, for runs without a config.json")
    ap.add_argument("--no-checkpoint", action="store_true")
    a = ap.parse_args()
    run, out = Path(a.run).expanduser(), ROOT / "experiments" / a.id
    out.mkdir(parents=True, exist_ok=True)
    if (run / "config.json").exists():
        cfg = json.loads((run / "config.json").read_text())
    else:
        cfg = {"argv": shlex.split(a.argv or ""), "reconstructed": True,
               "note": "run predates config.json; argv reconstructed from the queue script that launched it"}
    (out / "config.json").write_text(json.dumps(cfg, indent=1, ensure_ascii=False))
    shutil.copy(run / "log.jsonl", out / "log.jsonl")
    (out / "metrics.json").write_text(json.dumps(metrics_of(run), indent=1))
    if (run / "samples.json").exists():
        s = json.loads((run / "samples.json").read_text())
        (out / "samples.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in s))
    for f in ("units.json", "delta.json"):
        if (run / f).exists():
            shutil.copy(run / f, out / f)
    if not (out / "notes.md").exists():
        (out / "notes.md").write_text(f"# {a.id}\n\n_What we learned._\n")
    if (run / "neurons.pt").exists() and not a.no_checkpoint:
        subprocess.run([sys.executable, str(ROOT / "scripts/export_neurons.py"), str(run / "neurons.pt"),
                        str(out / "checkpoint")], check=True)
    steps = sorted((run / "steps").glob("step_*.pt")) if (run / "steps").is_dir() and not a.no_checkpoint else []
    if steps:                                                  # every validation step, Sauers' plain form
        sys.path.insert(0, str(ROOT / "scripts"))
        from export_neurons import load_arm, write_neurons
        model = cfg.get("args", {}).get("model") or "Qwen/Qwen3.5-9B"
        for f in steps:
            arm = load_arm(f)
            if arm["kind"] == "neurons":
                write_neurons(arm, out / "checkpoint/steps" / f.stem, model)
            else:
                (out / "checkpoint/steps").mkdir(parents=True, exist_ok=True)
                shutil.copy(f, out / "checkpoint/steps" / f.name)
    print(f"recorded {run} -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
