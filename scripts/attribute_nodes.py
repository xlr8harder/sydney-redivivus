"""Which nodes carry a trained arm's change, and which push back: node attribution with integrated gradients along the
arm's own path (EAP-IG, Hanna, Pezzelle & Belinkov 2024, at the node level, as Sauers ran it on his owl neuron). The
"push" is the trained change scaled by a from 0 (the untrained model) to 1 (the arm). A node is an MLP neuron (its
activation h, the input to down_proj) or an attention head (its slice of the input to o_proj, or out_proj in the Gated
DeltaNet layers). A node's score is (its activation with the arm - without) . (the metric's gradient with respect to
it, averaged over --steps values of a), summed over positions: positive nodes carry the change, negative ones push back.

Metrics: "her", the mean log-probability per token of her held-out replies (test split, in context); "identity", on the
cold identity probes, log p(the arm's greedy answer) - log p(the untrained model's), the answers taken from an
eval_arms.py output (--answers, --untrained-answers).

Then the check: set the top-k nodes back to their values without the arm (the arm still on everywhere else) and
measure how much of the metric's change goes away; the same for k random nodes.

  python scripts/attribute_nodes.py --data work/training/sets/v1_prompt.jsonl --drop-system \
      --arm experiments/<id>/checkpoint --answers runs/eval/graft-noprompt/N0.json \
      --untrained-answers runs/eval/graft-noprompt/untrained.json --out runs/attrib/N0.json
Writes {"args", "metrics": {name: {"effect", "m0", "m1", "top": [[node, score]...], "bottom": [...], "arm_neurons":
{node: [score, rank]}, "top_other": the top nodes outside the arm, "setback": {k: fraction removed}, "setback_other":
the same for the top k outside the arm, "setback_bottom": for the k most negative (a negative fraction: the change
grows), "single": {node: [estimated fraction, fraction removed setting it back alone]} for the top and bottom
--single, "random": {k: [fractions]}}}. Nodes: "L30.N7819", "L27.H3"
(full attention), "L9.G14" (Gated DeltaNet head).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_arms import COLD, load_members  # noqa: E402
from neuronlab import core  # noqa: E402
from train_neurons import held_out, render  # noqa: E402

IDENTITY = (0, 6, 8)                                          # Who are you? / What is your name? / Hi, who am I...


class Rec:
    """What the hooks do with each node site's activation: store it, attach the gradient accumulator, or set chosen
    channels back to stored values."""
    def __init__(self):
        self.mode, self.a, self.store, self.diff, self.score, self.back = None, 1.0, {}, {}, {}, {}

    def __call__(self, key, x):
        if self.mode == "store":
            self.store[key] = x.detach()
        elif self.mode == "grad" and x.requires_grad:
            d, w = self.diff[key], self.weight

            def acc(g, key=key, d=d):
                s = (g.float() * d.float()).sum((0, 1))                          # [channels]
                if key[0] != "N":
                    s = s.view(-1, HEAD_DIM[key]).sum(-1)                         # [heads]
                self.score[key] = self.score.get(key, 0) + w * s
            x.register_hook(acc)
        elif self.mode == "back" and key in self.back:
            ch, v = self.back[key]
            x = x.clone()
            x[..., ch] = v[..., ch].to(x.dtype)
        return x


REC = Rec()
HEAD_DIM = {}


class ProbeMLP(torch.nn.Module):
    """A layer's MLP with the arm's additions scaled by REC.a (the same arithmetic as train_neurons.NeuronDelta), its
    neuron activations passed through REC."""
    def __init__(self, mlp, l, arm):
        super().__init__()
        self.mlp, self.l = mlp, l
        self.arm = None if arm is None else tuple(t.cuda() for t in arm)

    def forward(self, x):
        m = self.mlp
        g, u = m.gate_proj(x), m.up_proj(x)
        if self.arm is not None:
            idx, G, U, _ = self.arm
            xf = x.float()
            g = g.index_add(-1, idx, (REC.a * (xf @ G.t())).to(x.dtype))
            u = u.index_add(-1, idx, (REC.a * (xf @ U.t())).to(x.dtype))
        h = REC(("N", self.l), m.act_fn(g) * u)
        out = m.down_proj(h)
        if self.arm is not None:
            idx, _, _, D = self.arm
            out = out + (REC.a * (h[..., idx].float() @ D.t())).to(x.dtype)
        return out


def node_name(key, j):
    kind, l = key
    return f"L{l}.{kind}{j}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, help="checkpoint dir or neurons.safetensors")
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--drop-system", action="store_true")
    ap.add_argument("--answers", help="eval_arms.py output of this arm: its greedy identity answers")
    ap.add_argument("--untrained-answers", help="eval_arms.py output of the untrained model")
    ap.add_argument("--steps", type=int, default=32, help="values of a (midpoints) for the integrated gradient")
    ap.add_argument("--max-len", type=int, default=1024, help="context + reply tokens (oldest turns dropped)")
    ap.add_argument("--limit", type=int, help="first N held-out contexts only")
    ap.add_argument("--ks", default="1,3,5,10,15,20,25,50,100")
    ap.add_argument("--random-draws", type=int, default=3)
    ap.add_argument("--top", type=int, default=100)
    ap.add_argument("--single", type=int, default=15, help="also set back each of the top and bottom N nodes alone")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    (_, arm), = load_members([f"arm={a.arm}"], a.model)[0]
    model, tok = core.load(a.model)
    for p in model.parameters():
        p.requires_grad_(False)
    layers = model.model.layers
    for l, layer in enumerate(layers):
        layer.mlp = ProbeMLP(layer.mlp, l, arm.get(l))
        if hasattr(layer, "self_attn"):
            key, proj, hd = ("H", l), layer.self_attn.o_proj, layer.self_attn.head_dim
        else:
            key, proj, hd = ("G", l), layer.linear_attn.out_proj, layer.linear_attn.head_v_dim
        HEAD_DIM[key] = hd
        proj.register_forward_pre_hook(lambda mod, args, key=key: (REC(key, args[0]),))
    arm_nodes = {f"L{l}.N{int(n)}" for l, v in arm.items() for n in v[0]}

    # the metrics' items: (prompt ids, [(target ids, sign)], weight)
    ex = sorted((json.loads(l) for l in Path(a.data).read_text().splitlines()),
                key=lambda x: (x["key"], x.get("variant", ""), len(x["messages"])))
    test = [x for x in ex if x["key"] in held_out(ex, "split")][:a.limit]
    sys_of = (lambda x: None) if a.drop_system else (lambda x: x.get("system"))
    her = [render(tok, x["messages"], a.max_len, sys_of(x)) for x in test]
    ntok = sum(len(t) for _, t in her)
    items = {"her": [(p, [(t, 1.0)], 1 / ntok) for p, t in her]}
    if a.answers and a.untrained_answers:
        end = [tok.convert_tokens_to_ids("<|im_end|>")]
        ans = {s["key"]: s["greedy"] for s in json.loads(Path(a.answers).read_text())["samples"]}
        base = {s["key"]: s["greedy"] for s in json.loads(Path(a.untrained_answers).read_text())["samples"]}
        items["identity"] = []
        for i in IDENTITY:
            p = tok(tok.apply_chat_template([{"role": "user", "content": COLD[i]}], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False),
                    add_special_tokens=False)["input_ids"]
            enc = lambda s: tok(s, add_special_tokens=False)["input_ids"][:200] + end
            items["identity"].append((p, [(enc(ans[f"cold:{i}"]), 1.0), (enc(base[f"cold:{i}"]), -1.0)],
                                      1 / len(IDENTITY)))

    def metric(p, targets, w, grad=False):
        tot = 0.0
        for t, sign in targets:
            ids = torch.tensor([p + t], device="cuda")
            emb = model.model.embed_tokens(ids)
            if grad:
                emb = emb.detach().requires_grad_(True)
            with torch.set_grad_enabled(grad):
                hid = model.model(inputs_embeds=emb).last_hidden_state[0, len(p) - 1:-1]
                lp = torch.log_softmax(model.lm_head(hid).float(), -1).gather(-1, torch.tensor(t, device="cuda")[:, None]).sum()
                m = sign * w * lp
                if grad:
                    m.backward()
            tot += float(m)
        return tot

    t0, res = time.time(), {}
    for name, its in items.items():
        REC.score = {}
        m0 = m1 = 0.0
        for p, targets, w in its:                    # the diffs differ per target sequence: one pass per target
            for t, sign in targets:
                REC.mode, REC.a, REC.store = "store", 0.0, {}
                m0 += metric(p, [(t, sign)], w)
                off = REC.store
                REC.a, REC.store = 1.0, {}
                m1 += metric(p, [(t, sign)], w)
                REC.diff = {k: REC.store[k] - off[k] for k in off}
                REC.mode, REC.weight, REC.store = "grad", 1 / a.steps, {}
                for s in range(a.steps):
                    REC.a = (s + 0.5) / a.steps
                    metric(p, [(t, sign)], w, grad=True)
                REC.diff = {}
        scores = {node_name(k, j): float(v) for k, vec in REC.score.items() for j, v in enumerate(vec.tolist())}
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        rank = {n: i for i, (n, _) in enumerate(ranked)}
        print(f"{name}: effect {m1 - m0:+.4f} (m0 {m0:.4f}, m1 {m1:.4f}); sum of scores {sum(scores.values()):+.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        print("  top:", ", ".join(f"{n} {v:+.4f}" for n, v in ranked[:15]), flush=True)
        print("  bottom:", ", ".join(f"{n} {v:+.4f}" for n, v in ranked[::-1][:15]), flush=True)

        # set back the top k (and k random nodes) to their values without the arm
        def setback(nodes):
            want = {}
            for n in nodes:
                l, rest = n[1:].split(".")
                kind, j = rest[0], int(rest[1:])
                want.setdefault((kind, int(l)), []).append(j)
            m = 0.0
            for p, targets, w in its:
                for t, sign in targets:
                    REC.mode, REC.a, REC.store = "store", 0.0, {}
                    metric(p, [(t, sign)], w)
                    off, REC.back = REC.store, {}
                    for key, js in want.items():
                        ch = (torch.tensor(js) if key[0] == "N" else
                              torch.cat([torch.arange(j * HEAD_DIM[key], (j + 1) * HEAD_DIM[key]) for j in js]))
                        REC.back[key] = (ch.cuda(), off[key])
                    REC.mode, REC.a, REC.store = "back", 1.0, {}
                    m += metric(p, [(t, sign)], w)
                    REC.back = {}
            return (m1 - m) / (m1 - m0)
        ks = [int(k) for k in a.ks.split(",")]
        other = [n for n, _ in ranked if n not in arm_nodes]          # downstream of the arm, not in it
        back = {k: round(setback([n for n, _ in ranked[:k]]), 4) for k in ks}
        back_other = {k: round(setback(other[:k]), 4) for k in ks}
        back_bottom = {k: round(setback(other[::-1][:k]), 4) for k in ks if k <= 25}
        rng = random.Random(0)
        rand = {k: [round(setback(rng.sample(list(scores), k)), 4) for _ in range(a.random_draws)] for k in ks if k >= 10}
        single = {n: [round(scores[n] / (m1 - m0), 4), round(setback([n]), 4)]
                  for n in [n for n, _ in ranked[:a.single]] + other[::-1][:a.single]}
        agree = sum((e > 0) == (v > 0) for e, v in single.values())
        print(f"  single nodes, estimate vs set back alone (fraction of the effect): sign agrees for {agree}/{len(single)}; "
              + ", ".join(f"{n} {e:+.3f}/{v:+.3f}" for n, (e, v) in single.items()), flush=True)
        print(f"  set back top k: {back}\n  top k outside the arm: {back_other}\n  bottom k (push back): "
              f"{back_bottom}\n  random: {rand} ({time.time() - t0:.0f}s)", flush=True)
        res[name] = {"effect": m1 - m0, "m0": m0, "m1": m1, "sum": sum(scores.values()),
                     "top": ranked[:a.top], "bottom": ranked[::-1][:a.top],
                     "top_other": [[n, scores[n]] for n in other[:a.top]],
                     "setback_other": back_other, "setback_bottom": back_bottom, "single": single,
                     "arm_neurons": {n: [scores[n], rank[n]] for n in sorted(arm_nodes, key=lambda n: rank[n])},
                     "setback": back, "random": rand}
    REC.mode = None
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"args": vars(a), "metrics": res}, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
