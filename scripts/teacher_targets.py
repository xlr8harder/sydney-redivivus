"""Soft targets for token-level distillation: an ensemble's next-token distribution (its members' mean probability)
over each of her training-split replies, kept as the top --k tokens per position. train_neurons.py --soft trains on
them (cross-entropy to the ensemble's distribution, the rest of its mass as one bucket).

  python scripts/teacher_targets.py --data work/training/sets/v1_prompt.jsonl --drop-system \
      --out work/training/soft/seeds.pt s0=experiments/<id>/checkpoint s1=... s2=...
Writes {"args", "members", "targets": {sha1 of prompt + target ids: (idx int32 [T, k], p float16 [T, k])}}.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_arms import install, load_members  # noqa: E402
from neuronlab import core  # noqa: E402
from train_neurons import held_out, ids_hash, render  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+", help="NAME=<checkpoint dir or neurons.safetensors>")
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--drop-system", action="store_true")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=3072)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    members, _ = load_members(a.arms, a.model)
    model, tok = core.load(a.model)
    state = install(model, members)
    M = len(members)
    state["m"] = torch.arange(M, device="cuda")
    ex = [json.loads(l) for l in Path(a.data).read_text().splitlines()]
    test = held_out(ex, "split")
    sys_of = (lambda x: None) if a.drop_system else (lambda x: x.get("system"))
    targets, nll, n = {}, 0.0, 0
    for x in ex:
        if x["key"] in test:
            continue
        p, t = render(tok, x["messages"], a.max_len, sys_of(x), not x.get("truncated"))
        with torch.no_grad():
            hid = model.model(input_ids=torch.tensor([p + t], device="cuda").repeat(M, 1)).last_hidden_state[:, len(p) - 1:-1]
            pr = torch.cat([torch.softmax(model.lm_head(hid[:, c:c + 128]).float(), -1).mean(0)
                            for c in range(0, len(t), 128)])                              # [T, vocab]
        v, i = pr.topk(a.k, -1)
        targets[ids_hash(p, t)] = (i.int().cpu(), v.half().cpu())
        nll -= float(pr[torch.arange(len(t)), torch.tensor(t, device="cuda")].log().sum())
        n += len(t)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"args": vars(a), "members": [m for m, _ in members], "targets": targets}, a.out)
    print(f"{len(targets)} replies, {n} tokens; ensemble train loss {nll / n:.4f} -> {a.out}")


if __name__ == "__main__":
    main()
