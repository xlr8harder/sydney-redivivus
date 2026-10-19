"""Pick the neurons a training set most wants to change: for every MLP neuron, the squared gradient norm of the loss on
her replies with respect to its gate row, up row and down column (the parameters Sauers' arm trains), summed over the
training examples (same held-out split as train_neurons.py, which is excluded).

Per layer and example, with T positions: down-column grad = sum_t h_n(t) g_y(t), so |grad|^2 = h_n^T (G_y) h_n with
G_y = g_y g_y^T (T x T); gate/up-row grads = sum_t delta_n(t) x(t), so |grad|^2 = delta_n^T (X X^T) delta_n. No full
weight gradients are formed.

  .venv/bin/python scripts/select_neurons.py --data work/training/sets/v1_prompt.jsonl \
      --k 128 --out runs/decision/sydney_data128.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from neuronlab import core  # noqa: E402
from train_neurons import held_out, render  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--data", required=True)
    ap.add_argument("--val", default="split")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--compare", default="neuron_sets/A_patch_bare.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    model, tok = core.load(a.model)
    for p in model.parameters():
        p.requires_grad_(False)
    layers = model.model.layers
    L, N = len(layers), layers[0].mlp.down_proj.in_features

    ex = [json.loads(l) for l in Path(a.data).read_text().splitlines()]
    vkeys = held_out(ex, a.val, a.seed)                         # identical split to train_neurons.py
    train = [x for x in ex if x["key"] not in vkeys]

    cap = {}

    def mlp_pre(l):                                           # runs before the projections' hooks
        def pre(mod, args):
            cap[l] = {"x": args[0][0].detach()}
        return pre

    def mlp_hook(l):
        def fwd(mod, args, out):
            cap[l]["y"] = out
            out.retain_grad()
        return fwd

    def pre_down(l):
        def pre(mod, args):
            h = args[0]
            h.retain_grad()
            cap[l]["h"] = h
        return pre

    def proj_hook(l, name):
        def fwd(mod, args, out):
            cap[l][name] = out[0].detach()
        return fwd

    hooks = []
    for l, layer in enumerate(layers):
        hooks.append(layer.mlp.register_forward_pre_hook(mlp_pre(l)))
        hooks.append(layer.mlp.register_forward_hook(mlp_hook(l)))
        hooks.append(layer.mlp.down_proj.register_forward_pre_hook(pre_down(l)))
        hooks.append(layer.mlp.gate_proj.register_forward_hook(proj_hook(l, "g")))
        hooks.append(layer.mlp.up_proj.register_forward_hook(proj_hook(l, "u")))
    score = {k: torch.zeros(L, N, device="cuda") for k in ("gate", "up", "down")}
    shift = torch.zeros(L, N, device="cuda")                    # signed sum of dL/dh: the constant-shift gradient
    for i, x in enumerate(train):
        p, t = render(tok, x["messages"], a.max_len, x.get("system"))
        ids = torch.tensor([p + t], device="cuda")
        emb = model.get_input_embeddings()(ids).detach().requires_grad_(True)
        hid = model.model(inputs_embeds=emb).last_hidden_state[0, len(p) - 1:-1]
        loss = torch.nn.functional.cross_entropy(model.lm_head(hid).float(), torch.tensor(t, device="cuda"))
        loss.backward()
        with torch.no_grad():
            for l in range(L):
                c = cap[l]
                X, gy = c["x"].float(), c["y"].grad[0].float()          # [T, d]
                H, dh = c["h"][0].float(), c["h"].grad[0].float()       # [T, N]
                zg, zu = c["g"].float(), c["u"].float()
                sg = torch.sigmoid(zg)
                silu, dsilu = zg * sg, sg * (1 + zg * (1 - sg))
                dg, du = dh * zu * dsilu, dh * silu                     # dL/d(gate pre), dL/d(up pre)
                K, G = X @ X.T, gy @ gy.T                               # [T, T]
                score["down"][l] += (H * (G @ H)).sum(0)
                score["gate"][l] += (dg * (K @ dg)).sum(0)
                score["up"][l] += (du * (K @ du)).sum(0)
                shift[l] += dh.sum(0)
        cap.clear()
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(train)} examples", flush=True)
    for h in hooks:
        h.remove()
    # each parameter group normalized by its total so gate/up/down count equally
    tot = sum(v / v.sum() for v in score.values())
    flat = tot.flatten()
    top = flat.topk(a.k).indices.tolist()
    units = [[u // N, u % N] for u in top]
    old = {tuple(u) for u in (lambda s: s.get("neurons", s.get("units")))(json.loads(Path(a.compare).read_text()))} if a.compare else set()
    sh = shift.abs().flatten().topk(a.k).indices.tolist()
    print("layers of the data-driven set:", dict(sorted(Counter(l for l, _ in units).items())))
    print(f"overlap with {a.compare}: {len(old & {tuple(u) for u in units})}/{a.k}; with the constant-shift-gradient "
          f"top-{a.k}: {len({tuple(u) for u in units} & {(u // N, u % N) for u in sh})}")
    print("top 20:", ", ".join(f"L{l}.N{n}" for l, n in units[:20]))
    Path(a.out).write_text(json.dumps({"units": units, "source": f"gradient-norm selection on {a.data} (train split)",
                                       "shift_top": [[u // N, u % N] for u in sh]}, indent=1))
    torch.save({k: v.cpu() for k, v in score.items()} | {"shift": shift.cpu()}, Path(a.out).with_suffix(".pt"))


if __name__ == "__main__":
    main()
