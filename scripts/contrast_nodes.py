"""How the untouched model reads her: every MLP neuron's and attention head's activation while reading her real reply,
against reading the model's own greedy reply to the same context (lyra's starting point: search by what a neuron
responds to, not by what pushing it does). Per reply, the mean over its tokens (neurons: the activation; heads: the
norm of the head's output); per node, the paired effect size over contexts (mean difference / its standard deviation).
Found on the training split, checked on the test split.

  python scripts/contrast_nodes.py --data work/training/sets/v1_prompt.jsonl --drop-system --out runs/steer/contrast
Writes <out>/result.json ({"neurons": top by |effect| on train with the test effect, "heads": all, "layers": per layer
counts among the top, "overlap": with the given --sets}) and <out>/C<k>_contrast.json (neuron-set files, the effect's
sign as "dose").
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
from train_neurons import held_out, render  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--drop-system", action="store_true")
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--top", type=int, default=300)
    ap.add_argument("--ks", default="64,128")
    ap.add_argument("--sets", default="neuron_sets/S25_steer_layered2.json,neuron_sets/F_patch_prompt.json,"
                                      "neuron_sets/A_patch_bare.json")
    ap.add_argument("--mem-gb", type=float, default=27, help="hard cap on GPU memory (past it WSL spills to the host)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model, tok = core.load(a.model)
    torch.cuda.set_per_process_memory_fraction(a.mem_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    layers = model.model.layers
    L, N = len(layers), layers[0].mlp.down_proj.in_features
    rec, on = {}, [False]                                      # recording off while generating (a batch's prefill
    for l, layer in enumerate(layers):                          # kept for all 32 layers would be ~19 GB)
        layer.mlp.down_proj.register_forward_pre_hook(lambda mod, args, l=l: rec.__setitem__(("N", l), args[0])
                                                      if on[0] else None)
        if hasattr(layer, "self_attn"):
            key, proj, hd = ("H", l), layer.self_attn.o_proj, layer.self_attn.head_dim
        else:
            key, proj, hd = ("G", l), layer.linear_attn.out_proj, layer.linear_attn.head_v_dim
        proj.register_forward_pre_hook(lambda mod, args, key=key, hd=hd: rec.__setitem__(
            key, args[0].unflatten(-1, (-1, hd)).float().norm(dim=-1)) if on[0] else None)
    end = tok.convert_tokens_to_ids("<|im_end|>")
    pad = tok.pad_token_id

    ex = sorted((json.loads(l) for l in Path(a.data).read_text().splitlines()),
                key=lambda x: (x["key"], x.get("variant", ""), len(x["messages"])))
    test = held_out(ex, "split")
    sys_of = (lambda x: None) if a.drop_system else (lambda x: x.get("system"))
    items = [(x["key"] in test, *render(tok, x["messages"], a.max_len, sys_of(x))) for x in ex]

    # the model's own greedy reply to every context, batched by length
    own = {}
    order = sorted(range(len(items)), key=lambda i: len(items[i][1]))
    for s in range(0, len(order), 16):
        b = order[s:s + 16]
        T = max(len(items[i][1]) for i in b)
        with torch.no_grad():
            g = model.generate(torch.tensor([[pad] * (T - len(items[i][1])) + items[i][1] for i in b], device="cuda"),
                               attention_mask=torch.tensor([[0] * (T - len(items[i][1])) + [1] * len(items[i][1])
                                                            for i in b], device="cuda"),
                               max_new_tokens=a.max_new, do_sample=False, pad_token_id=pad, eos_token_id=[end])[:, T:]
        for i, r in zip(b, g.tolist()):
            own[i] = r[:r.index(end) + 1] if end in r else r
        torch.cuda.empty_cache()
    print(f"{len(own)} own replies generated", flush=True)
    on[0] = True

    def means(p, t):                                           # per site: mean over the reply's tokens
        with torch.no_grad():
            model.model(input_ids=torch.tensor([p + t], device="cuda"), use_cache=False)
        return {k: v[0, len(p):len(p) + len(t)].float().mean(0) for k, v in rec.items()}

    acc = {sp: {} for sp in (False, True)}                     # split -> site -> [sum d, sum d^2, n]
    for i, (is_test, p, t) in enumerate(items):
        mh, mo = means(p, t), means(p, own[i])
        for k in mh:
            d = mh[k] - mo[k]
            s = acc[is_test].setdefault(k, [0, 0, 0])
            s[0], s[1], s[2] = s[0] + d, s[1] + d * d, s[2] + 1
    eff = {}
    for sp, sites in acc.items():
        for k, (sd, sd2, n) in sites.items():
            mu = sd / n
            eff.setdefault(sp, {})[k] = mu / (sd2 / n - mu * mu).clamp_min(1e-12).sqrt()
    tr, te = eff[False], eff[True]
    neur = torch.stack([tr[("N", l)] for l in range(L)])     # [L, N]
    neur_te = torch.stack([te[("N", l)] for l in range(L)])
    top = neur.abs().flatten().topk(a.top).indices.tolist()
    rows = [[f"L{i // N}.N{i % N}", float(neur.flatten()[i]), float(neur_te.flatten()[i])] for i in top]
    rep = sum((r[1] > 0) == (r[2] > 0) and abs(r[2]) > 0.3 for r in rows)
    heads = sorted(([f"L{l}.{kind}{h}", float(tr[(kind, l)][h]), float(te[(kind, l)][h])]
                    for (kind, l) in tr if kind != "N" for h in range(len(tr[(kind, l)]))), key=lambda r: -abs(r[1]))
    layers_top = {l: sum(1 for r in rows[:128] if r[0].startswith(f"L{l}.")) for l in range(L)}
    print(f"top {a.top} neurons by |paired effect| on train: same sign and |effect| > 0.3 on test for {rep}; "
          f"top 128 by layer: { {l: c for l, c in layers_top.items() if c} }", flush=True)
    print("top neurons (train / test effect): " + ", ".join(f"{n} {e:+.2f}/{v:+.2f}" for n, e, v in rows[:25]), flush=True)
    print("top heads (output norm): " + ", ".join(f"{n} {e:+.2f}/{v:+.2f}" for n, e, v in heads[:20]), flush=True)
    res = {"args": vars(a), "replicated": rep, "neurons": rows, "heads": heads, "layers_top128": layers_top,
           "overlap": {}}
    for f in a.sets.split(","):
        spec = json.loads(Path(f).read_text())
        s = {f"L{l}.N{n}" for l, n in spec.get("neurons", spec.get("units"))}
        res["overlap"][Path(f).stem] = {k: len(s & {r[0] for r in rows[:int(k)]}) for k in ("64", "128", "300")}
        print(f"overlap with {Path(f).stem} ({len(s)}): {res['overlap'][Path(f).stem]}", flush=True)
    for k in map(int, a.ks.split(",")):
        sel = rows[:k]
        (out / f"C{k}_contrast.json").write_text(json.dumps({
            "base_model": a.model, "neurons": [[int(n[1:].split(".")[0]), int(n.split(".N")[1])] for n, *_ in sel],
            "name": f"C{k}_contrast", "method": "scripts/contrast_nodes.py: the untouched model reading her real reply "
            "vs its own greedy reply to the same context (training split); per neuron the paired effect size of its mean "
            f"activation over the reply's tokens; top {k} by |effect|.", "source": str(out / "result.json"),
            "created": "2026-10-25", "notes": "",
            "per_neuron": {n: {"dose": 1.0 if e > 0 else -1.0, "effect_train": e, "effect_test": v} for n, e, v in sel}},
            indent=1))
    (out / "result.json").write_text(json.dumps(res, indent=1))
    print("wrote", out / "result.json")


if __name__ == "__main__":
    main()
