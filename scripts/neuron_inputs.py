"""What drives a set of neurons: for each, its activation on her real held-out replies against the untrained model's own
replies in the same contexts, its top-activating tokens, and which upstream nodes push it (first-order attribution:
activation x gradient of the neuron's summed activation over her reply tokens, signed by its steering dose, to every
MLP neuron below it, every attention head at or below it (o_proj / out_proj input slices) and the token embeddings).

  python scripts/neuron_inputs.py --units neuron_sets/S25_steer_layered2.json --eval runs/eval/steer-noprompt \
      --out runs/steer/inputs-S25.json
Reads the eval's untrained.json (contexts, her real replies, the untrained model's greedy and t0.8 replies). Writes
{"args", "neurons": {node: {"dose_sign", "natmax", "mean_her", "mean_untrained", "top_her": [[activation, text]],
"top_untrained", "inputs": [[node, score]], "emb", "steered_inputs": the inputs that are themselves in the set}}}.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", required=True, help="neuron set file (per_neuron doses give the sign)")
    ap.add_argument("--eval", required=True, help="eval_arms.py output dir with untrained.json")
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--mem-gb", type=float, default=27, help="hard cap on GPU memory (past it WSL spills to the host)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    spec = json.loads(Path(a.units).read_text())
    units = [tuple(u) for u in spec["neurons"]]
    sign = {f"L{l}.N{n}": (1.0 if spec.get("per_neuron", {}).get(f"L{l}.N{n}", {}).get("dose", 1) >= 0 else -1.0)
            for l, n in units}
    model, tok = core.load(a.model)
    torch.cuda.set_per_process_memory_fraction(a.mem_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    for p in model.parameters():
        p.requires_grad_(False)
    layers = model.model.layers
    acts = {}                                                   # site -> activation tensor (this forward)
    for l, layer in enumerate(layers):
        layer.mlp.down_proj.register_forward_pre_hook(lambda mod, args, l=l: acts.__setitem__(("N", l), args[0]))
        if hasattr(layer, "self_attn"):
            key, proj, hd = ("H", l), layer.self_attn.o_proj, layer.self_attn.head_dim
        else:
            key, proj, hd = ("G", l), layer.linear_attn.out_proj, layer.linear_attn.head_v_dim
        proj.register_forward_pre_hook(lambda mod, args, key=key: acts.__setitem__(key, args[0]))
        acts.setdefault("hd", {})[key] = hd
    HD = acts.pop("hd")

    class Stop(Exception):
        pass

    def stop(mod, args):                                        # nothing above the top unit's layer is needed
        raise Stop
    if max(l for l, _ in units) + 1 < len(layers):
        layers[max(l for l, _ in units) + 1].register_forward_pre_hook(stop)

    smp = [s for s in json.loads((Path(a.eval) / "untrained.json").read_text())["samples"] if s["kind"] != "cold"]
    msgs = lambda s, reply: s["context"] + [{"role": "user", "content": s["q"]}, {"role": "assistant", "content": reply}]
    her = [render(tok, msgs(s, s["real"]), a.max_len) for s in smp]
    base = [render(tok, msgs(s, s[m]), a.max_len) for s in smp for m in s if m == "greedy" or m.startswith("t0.8")]

    def run(p, t, grad=False):
        ids = torch.tensor([p + t], device="cuda")
        emb = model.model.embed_tokens(ids)
        if grad:
            emb = emb.detach().requires_grad_(True)
        acts.clear()
        with torch.set_grad_enabled(grad):
            try:
                model.model(inputs_embeds=emb, use_cache=False)
            except Stop:
                pass
        return emb

    res = {}
    stats = {f"L{l}.N{n}": {"her": [], "base": [], "top_her": [], "top_untrained": []} for l, n in units}
    with torch.no_grad():
        for kind, items in (("her", her), ("base", base)):
            for p, t in items:
                run(p, t)
                for l, n in units:
                    h = acts[("N", l)][0, len(p):len(p) + len(t), n].float()       # the reply's tokens
                    st = stats[f"L{l}.N{n}"]
                    st[kind].append(h)
                    i = int(h.argmax())
                    ctx = tok.decode((p + t)[max(0, len(p) + i - 12):len(p) + i + 1])
                    st["top_her" if kind == "her" else "top_untrained"].append([float(h[i]), ctx])
    for k, st in stats.items():
        hh, bb = torch.cat(st["her"]), torch.cat(st["base"])
        natmax = float(torch.cat([hh, bb]).abs().max())
        res[k] = {"dose_sign": sign[k], "natmax": natmax, "mean_her": float(hh.mean()), "mean_untrained": float(bb.mean()),
                  "frac_her_over_half": float((hh * sign[k] > natmax / 2).float().mean()),
                  "frac_untrained_over_half": float((bb * sign[k] > natmax / 2).float().mean()),
                  "top_her": sorted(st["top_her"], key=lambda x: -x[0] * sign[k])[:5],
                  "top_untrained": sorted(st["top_untrained"], key=lambda x: -x[0] * sign[k])[:5]}
        print(f"{k} (dose {'+' if sign[k] > 0 else '-'}): mean on her replies {res[k]['mean_her']:+.3f}, on the untrained "
              f"model's {res[k]['mean_untrained']:+.3f} (natural max {natmax:.2f}); top on hers: "
              f"{res[k]['top_her'][0][1][-60:]!r}", flush=True)

    # upstream attribution on her replies: activation x gradient of sign x (sum of the neuron over her reply tokens)
    score = {k: {} for k in stats}
    embs = {k: 0.0 for k in stats}
    peak = []
    torch.cuda.reset_peak_memory_stats()
    for p, t in sorted(her, key=lambda x: -len(x[0]) - len(x[1])):        # longest first: the peak shows at once
        emb = run(p, t, grad=True)
        saved = dict(acts)
        for x in saved.values():
            if x.requires_grad:
                x.retain_grad()
        for j, (l, n) in enumerate(sorted(units)):              # the top layer last, so its backward frees it all
            k = f"L{l}.N{n}"
            for x in saved.values():
                if x.grad is not None:
                    x.grad = None
            emb.grad = None
            m = sign[k] * saved[("N", l)][0, len(p):len(p) + len(t), n].float().sum()
            m.backward(retain_graph=j < len(units) - 1)
            with torch.no_grad():                                 # (with grad on, every score kept its own graph)
                embs[k] += float((emb.grad.float() * emb.float()).sum())
                for (kind, ll), x in saved.items():
                    if x.grad is None or (kind == "N" and ll >= l) or ll > l:
                        continue
                    s = (x.grad.float() * x.float())[0].sum(0)                 # [channels]
                    if kind != "N":
                        s = s.view(-1, HD[(kind, ll)]).sum(-1)                # [heads]
                    score[k][(kind, ll)] = score[k].get((kind, ll), 0) + s
        del saved, emb, m                                         # a lower last unit's backward leaves the graph above
        acts.clear()
        if not peak:
            peak.append(torch.cuda.max_memory_allocated() / 2**30)
            print(f"attribution: peak GPU memory {peak[0]:.1f} GB on the longest reply ({len(p) + len(t)} tokens)", flush=True)
    picked = {f"L{l}.N{n}" for l, n in units}
    for k in stats:
        score[k] = {f"L{ll}.{kind}{c}": v for (kind, ll), vec in score[k].items() for c, v in enumerate(vec.tolist())}
        ranked = sorted(score[k].items(), key=lambda kv: -abs(kv[1]))[:a.top]
        res[k]["inputs"], res[k]["emb"] = ranked, embs[k]
        res[k]["steered_inputs"] = [[n, v] for n, v in sorted(score[k].items(), key=lambda kv: -abs(kv[1])) if n in picked]
        print(f"{k} inputs: embeddings {embs[k]:+.1f}; " + ", ".join(f"{n} {v:+.1f}" for n, v in ranked[:10])
              + (f"; steered upstream: {', '.join(f'{n} {v:+.1f}' for n, v in res[k]['steered_inputs'][:5])}"
                 if res[k]["steered_inputs"] else ""), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"args": vars(a), "neurons": res}, indent=1, ensure_ascii=False))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
