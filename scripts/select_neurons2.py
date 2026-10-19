"""Second-pass neuron selection: predicted gain under AdamW, and gradient signal-to-noise.

Candidates: the top --per-layer neurons of each layer by the first pass's score (select_neurons.py), normalized by
the layer's median so every layer contributes candidates. For each candidate, the full per-example gradients of its
gate row, up row and down column (3 x d parameters) are summed over the training split, together with their squares.
With m = mean gradient and v = mean squared gradient per parameter (over examples):

  adam_gain = sum_i m_i^2 / sqrt(v_i)   first-step loss decrease of an Adam update (update ~ m / sqrt(v)); size and
                                        consistency together, layer scale largely cancels
  snr       = sum_i m_i^2 / sum_i v_i   scale-free consistency of the push across her replies

The set is the top --k by adam_gain.

  .venv/bin/python scripts/select_neurons2.py --data work/training/sets/v1_prompt.jsonl \
      --first runs/decision/sydney_data128.pt --out runs/decision/sydney_adam128.json
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
    ap.add_argument("--first", required=True, help="score tensors from select_neurons.py")
    ap.add_argument("--per-layer", type=int, default=256)
    ap.add_argument("--val", default="split")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    first = torch.load(a.first)
    tot = sum(first[k] / first[k].sum() for k in ("gate", "up", "down"))           # [L, N]
    rel = tot / tot.median(dim=1, keepdim=True).values                              # layer-normalized
    cand = rel.topk(a.per_layer, dim=1).indices                                     # [L, M]

    model, tok = core.load(a.model)
    for p in model.parameters():
        p.requires_grad_(False)
    layers = model.model.layers
    L, d = len(layers), layers[0].mlp.gate_proj.weight.shape[1]
    M = a.per_layer
    S1 = torch.zeros(L, M, 3, d, device="cuda")        # sum over examples of the gradient
    S2 = torch.zeros(L, M, 3, d, device="cuda")        # sum over examples of its square
    ci = cand.cuda()

    ex = [json.loads(l) for l in Path(a.data).read_text().splitlines()]
    vkeys = held_out(ex, a.val, a.seed)                         # identical split to train_neurons.py
    train = [x for x in ex if x["key"] not in vkeys]

    cap = {}

    def mlp_pre(l):
        def pre(mod, args):
            cap[l] = {"x": args[0][0].detach()}
        return pre

    def mlp_post(l):
        def fwd(mod, args, out):
            cap[l]["y"] = out
            out.retain_grad()
        return fwd

    def pre_down(l):
        def pre(mod, args):
            args[0].retain_grad()
            cap[l]["h"] = args[0]
        return pre

    def proj(l, name):
        def fwd(mod, args, out):
            cap[l][name] = out[0].detach()
        return fwd

    hooks = []
    for l, layer in enumerate(layers):
        hooks += [layer.mlp.register_forward_pre_hook(mlp_pre(l)), layer.mlp.register_forward_hook(mlp_post(l)),
                  layer.mlp.down_proj.register_forward_pre_hook(pre_down(l)),
                  layer.mlp.gate_proj.register_forward_hook(proj(l, "g")),
                  layer.mlp.up_proj.register_forward_hook(proj(l, "u"))]
    for i, x in enumerate(train):
        p, t = render(tok, x["messages"], a.max_len, x.get("system"))
        ids = torch.tensor([p + t], device="cuda")
        emb = model.get_input_embeddings()(ids).detach().requires_grad_(True)
        hid = model.model(inputs_embeds=emb).last_hidden_state[0, len(p) - 1:-1]
        loss = torch.nn.functional.cross_entropy(model.lm_head(hid).float(), torch.tensor(t, device="cuda"))
        loss.backward()
        with torch.no_grad():
            for l in range(L):
                c, idx = cap[l], ci[l]
                X, gy = c["x"].float(), c["y"].grad[0].float()                     # [T, d]
                H = c["h"][0][:, idx].float()                                       # [T, M]
                dh = c["h"].grad[0][:, idx].float()
                zg, zu = c["g"][:, idx].float(), c["u"][:, idx].float()
                sg = torch.sigmoid(zg)
                dg = dh * zu * sg * (1 + zg * (1 - sg))
                du = dh * zg * sg
                G = torch.stack([dg.T @ X, du.T @ X, (H.T @ gy)], 1)               # [M, 3, d]
                S1[l] += G
                S2[l] += G * G
        cap.clear()
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(train)} examples", flush=True)
    for h in hooks:
        h.remove()
    n = len(train)
    m, v = S1 / n, S2 / n
    adam = (m * m / (v.sqrt() + 1e-12)).sum((2, 3))                                 # [L, M]
    snr = (m * m).sum((2, 3)) / v.sum((2, 3)).clamp_min(1e-30)
    flat = adam.flatten()
    top = flat.topk(a.k).indices.tolist()
    units = [[t // M, int(cand[t // M, t % M])] for t in top]
    snr_top = snr.flatten().topk(a.k).indices.tolist()
    snr_units = {(t // M, int(cand[t // M, t % M])) for t in snr_top}
    U = {tuple(u) for u in units}
    print("layers (adam_gain set):", dict(sorted(Counter(l for l, _ in units).items())))
    for name, path in (("patching set", "neuron_sets/A_patch_bare.json"), ("raw gradient set", "neuron_sets/B_grad_norm.json"),
                       ("cap-8 set", "neuron_sets/C_grad_norm_cap8.json")):
        s = json.loads(Path(path).read_text())
        other = {tuple(u) for u in s.get("neurons", s.get("units"))}
        print(f"overlap with {name}: {len(U & other)}/{a.k}")
    print(f"overlap of adam_gain top-{a.k} with snr top-{a.k}: {len(U & snr_units)}")
    print("top 20:", ", ".join(f"L{l}.N{n}" for l, n in units[:20]))
    Path(a.out).write_text(json.dumps({"units": units, "snr_top": sorted(map(list, snr_units)),
                                       "source": f"adam-gain selection on {a.data} (train split), candidates "
                                                 f"top {M}/layer by layer-normalized gradient norm"}, indent=1))
    torch.save({"cand": cand, "adam": adam.cpu(), "snr": snr.cpu()}, Path(a.out).with_suffix(".pt"))


if __name__ == "__main__":
    main()
