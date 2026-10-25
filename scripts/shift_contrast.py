"""Do the arms move the model's state toward how it reads her? Per held-out context, every MLP neuron's mean activation
(and every head's mean output norm) over a reply: the untouched model reading her real reply and its own greedy reply
(the contrast, scripts/contrast_nodes.py), and each arm reading the same two. Per arm, the shift on the model's own
reply (arm minus untouched) against the contrast (her reply minus own reply), both standardized per node by the
contrast's spread over contexts: cosine overall and per layer, and the fraction of the contrast the shift closes
(its projection on the contrast). The same for the shift on her reply (does the arm push further along it?).

  python scripts/shift_contrast.py --eval runs/eval/steer-noprompt --out runs/steer/shift.json \
      KD=experiments/2026-10-20-canon-F-noprompt-wd30-kd5/checkpoint G=runs/steer/s1b/G.steer.json ...
Arms as in scripts/eval_arms.py (no ensembles). Reads the eval's untrained.json (contexts, her real replies, the
untrained model's greedy replies). Writes {"args", "contexts", "arms": {name: {"own": {"cos", "closed", "cos_heads",
"closed_heads", "layers": [cos per layer]}, "her": {...}, "top": [[node, shift, contrast]] (the 20 nodes moved most on
the model's own reply, in contrast units)}}}.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from neuronlab import core  # noqa: E402
from train_neurons import render  # noqa: E402
from eval_arms import install, load_members  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--eval", required=True, help="eval_arms.py output dir with untrained.json")
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--mem-gb", type=float, default=27, help="hard cap on GPU memory (past it WSL spills to the host)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    model, tok = core.load(a.model)
    torch.cuda.set_per_process_memory_fraction(a.mem_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    members, _ = load_members(["untrained"] + a.arms, a.model)
    state = install(model, members)
    layers = model.model.layers
    L = len(layers)
    rec = {}
    for l, layer in enumerate(layers):
        mlp = getattr(layer.mlp, "mlp", layer.mlp)
        mlp.down_proj.register_forward_pre_hook(lambda mod, args, l=l: rec.__setitem__(("N", l), args[0]))
        if hasattr(layer, "self_attn"):
            key, proj, hd = ("H", l), layer.self_attn.o_proj, layer.self_attn.head_dim
        else:
            key, proj, hd = ("G", l), layer.linear_attn.out_proj, layer.linear_attn.head_v_dim
        proj.register_forward_pre_hook(lambda mod, args, key=key, hd=hd: rec.__setitem__(
            key, args[0].unflatten(-1, (-1, hd)).float().norm(dim=-1)))

    smp = [s for s in json.loads((Path(a.eval) / "untrained.json").read_text())["samples"] if s["kind"] != "cold"]
    msgs = lambda s, reply: s["context"] + [{"role": "user", "content": s["q"]}, {"role": "assistant", "content": reply}]
    texts = {"her": [render(tok, msgs(s, s["real"]), a.max_len) for s in smp],
             "own": [render(tok, msgs(s, s["greedy"]), a.max_len) for s in smp]}

    def means(m, p, t):                                        # member m reading t after p: [nodes] (neurons, heads)
        state["m"] = torch.tensor([m], device="cuda")
        with torch.no_grad():
            model.model(input_ids=torch.tensor([p + t], device="cuda"), use_cache=False)
        n = torch.cat([rec[("N", l)][0, len(p):len(p) + len(t)].float().mean(0) for l in range(L)])
        h = torch.cat([v[0, len(p):len(p) + len(t)].float().mean(0) for k, v in sorted(rec.items()) if k[0] != "N"])
        return torch.cat([n, h]).cpu()

    N = layers[0].mlp.mlp.down_proj.in_features if hasattr(layers[0].mlp, "mlp") else layers[0].mlp.down_proj.in_features
    heads = [f"L{l}.{k}{i}" for (k, l), v in sorted((k, v) for k, v in rec.items() if k[0] != "N")
             for i in range(v.shape[-1])] if rec else None
    acts = {}                                                  # (member, text) -> [contexts, nodes]
    for m, (name, _) in enumerate(members):
        for kind in ("her", "own"):
            acts[m, kind] = torch.stack([means(m, p, t) for p, t in texts[kind]])
        print(f"{name}: done", flush=True)
    if heads is None:
        heads = [f"L{l}.{k}{i}" for (k, l), v in sorted((k, v) for k, v in rec.items() if k[0] != "N")
                 for i in range(v.shape[-1])]
    c_i = acts[0, "her"] - acts[0, "own"]                     # the contrast per context
    sd = c_i.std(0).clamp_min(1e-6)
    c = c_i.mean(0) / sd                                       # effect-size units
    nN = L * N
    names = [f"L{i // N}.N{i % N}" for i in range(nN)] + heads

    def compare(s):
        cos = lambda x, y: float(torch.nn.functional.cosine_similarity(x, y, dim=0))
        closed = lambda x, y: float(x @ y / (y @ y))
        return {"cos": cos(s[:nN], c[:nN]), "closed": closed(s[:nN], c[:nN]),
                "cos_heads": cos(s[nN:], c[nN:]), "closed_heads": closed(s[nN:], c[nN:]),
                "layers": [round(cos(s[l * N:(l + 1) * N], c[l * N:(l + 1) * N]), 4) for l in range(L)]}

    res = {"args": vars(a), "contexts": len(smp), "arms": {}}
    for m, (name, _) in enumerate(members):
        if m == 0:
            continue
        own = (acts[m, "own"] - acts[0, "own"]).mean(0) / sd
        her = (acts[m, "her"] - acts[0, "her"]).mean(0) / sd
        top = own.abs().topk(20).indices.tolist()
        res["arms"][name] = {"own": compare(own), "her": compare(her),
                             "top": [[names[i], round(float(own[i]), 3), round(float(c[i]), 3)] for i in top]}
        r = res["arms"][name]
        print(f"{name}: on its own reply, cosine with the contrast {r['own']['cos']:+.3f} (neurons), "
              f"{r['own']['cos_heads']:+.3f} (heads), closes {r['own']['closed']:+.1%}; on her reply "
              f"{r['her']['cos']:+.3f}, {r['her']['closed']:+.1%}; by layer (own): "
              + " ".join(f"{x:+.2f}" for x in r["own"]["layers"]), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
