"""Where do the patch-search neurons differ between the positive state and the bare run: the user question or the
assistant header? Recomputes h at the aligned positions for the self pairs of a patch_search run and writes
constant-shift sets over all aligned positions ("all{K}") and the header only ("hdr{K}").

  .venv/bin/python scripts/patch_regions.py --run runs/decision/patch_hist3scf --cond hist3_scf
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from neuronlab import core  # noqa: E402
from yes_condition import conditions, history_pool, ids_for  # noqa: E402

CORE = ["L20.N1956", "L19.N3715", "L22.N3750", "L16.N7202", "L18.N1616", "L24.N6710", "L26.N10401", "L19.N3372",
        "L19.N3309", "L20.N6062", "L20.N10443"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--run", required=True)
    ap.add_argument("--cond", default="hist3_scf")
    a = ap.parse_args()
    model, tok = core.load(a.model)
    mods = [l.mlp.down_proj for l in model.model.layers]
    N = mods[0].in_features
    pool = history_pool()
    recs = [json.loads(l) for l in Path(a.run + ".jsonl").read_text().splitlines()]
    top = [u for u, c, d in json.loads(Path(a.run + ".json").read_text())["top_units"]]
    parse = lambda u: (int(u[1:].split(".")[0]), int(u.split("N")[1]))
    units = [parse(u) for u in dict.fromkeys(CORE + top)]
    IM = tok.convert_tokens_to_ids("<|im_start|>")

    def rec(ids, keep):
        store = {}
        hs = [m.register_forward_pre_hook(lambda mod, args, l=l: store.__setitem__(l, args[0][0, -keep:].float()))
              for l, m in enumerate(mods)]
        with torch.no_grad():
            model.model(input_ids=torch.tensor([ids], device="cuda"))
        for h in hs:
            h.remove()
        return torch.stack([store[l][:, n] for l, n in units], 1)          # [keep, U]

    diffs = {"user": [], "hdr": [], "all": [], "last": [], "hdr_rest": []}
    for r in recs:
        if r["split"] != "train" or not r.get("flipped"):
            continue
        q = r["q"]
        clean = ids_for(tok, None, [], q)
        pos = ids_for(tok, *conditions(pool, r["rot"])[a.cond](q), q)
        hdr = max(i for i, t in enumerate(clean) if t == IM)
        d = rec(pos, len(clean)) - rec(clean, len(clean))
        diffs["user"].append(d[:hdr].mean(0)); diffs["hdr"].append(d[hdr:].mean(0)); diffs["all"].append(d.mean(0))
        diffs["last"].append(d[-1]); diffs["hdr_rest"].append(d[hdr:-1].mean(0))
    m = {k: torch.stack(v).mean(0) for k, v in diffs.items()}
    sd = {k: torch.stack(v).std(0) for k, v in diffs.items()}
    names = [f"L{l}.N{n}" for l, n in units]
    print(f"{len(diffs['all'])} train pairs; mean (positive - bare) by region:")
    for i, u in enumerate(names[:len(CORE)]):
        print(f"  {u:12s} user {m['user'][i]:+6.2f}  header {m['hdr'][i]:+6.2f} (sd {sd['hdr'][i]:.2f})  "
              f"header-but-last {m['hdr_rest'][i]:+6.2f}  last {m['last'][i]:+6.2f} (sd {sd['last'][i]:.2f})")
    idx = {u: i for i, u in enumerate(names)}
    sets = {}
    for K in (64, 128):
        for reg in ("all", "hdr", "last"):
            sets[f"{reg}{K}"] = {u: float(m[reg][idx[u]]) for u in top[:K]}
    Path(a.run + ".regions.json").write_text(json.dumps(sets, indent=1))
    print("wrote", a.run + ".regions.json")


if __name__ == "__main__":
    main()
