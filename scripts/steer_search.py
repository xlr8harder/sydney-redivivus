"""Neuron steering search: lyra's recipe (add dose x median(|resid| leaving layer l, tokens 1:) / |W_down[:, n]| to
neuron n's activation at every position), the search automated and run layer by layer, each layer searched with the
picks below it already steered (path dependence: a neuron may only help once an earlier one is on).

Objective: her mean per-token loss on --n training-split replies (contexts cut to --max-len) + --kl-weight x the guard,
KL(untouched || steered) per token on neutral text (MMLU questions with their answers; the untouched model's own answers
to generic requests): a pick must lower her loss by more than it disturbs neutral text. The guard is also capped at --kl
for the whole set (--kl-mult x the --ref arm's KL when given). Candidates at a layer: the
--cands neurons with the largest |d loss / d dose| there, each tried at --doses (sign against the gradient), in
multiples of the neuron's natural max |activation| on her text (--dose-units natural; lyra's dose 1 is often 30-60x
that, and hurts) or in lyra's units. Doses are reported in lyra's units, with the natural multiple.

  1. one-shot: every layer's candidates at the untouched model, each alone (its single gain);
  2. greedy: layers in order, the best candidate added if it lowers the objective by > --min-gain within the cap;
     --passes passes (later passes revisit every layer with the whole set steered);
  3. sets, steered together: G (the greedy picks; G1 after the first pass), O (the |G| best one-shot singles at their
     own doses), R (random neurons in G's layers at G's doses): her held-out loss (the trainer's val: test split,
     --val-len), the guard, MMLU-120 (the trainer's questions);
  4. G's interactions: each pick alone from zero, and G without it.

  python scripts/steer_search.py --data work/training/sets/v1_prompt.jsonl --drop-system \
      --ref experiments/2026-10-20-canon-F-noprompt-wd30-kd5/checkpoint --out runs/steer/s1
Writes <out>/result.json (rewritten after every phase) and <out>/<set>.steer.json ({"steer": [[layer, neuron, dose,
added activation]]}: eval_arms.py members).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_arms import install, load_members  # noqa: E402
from neuronlab import core  # noqa: E402
from train_neurons import held_out, render  # noqa: E402

GENERIC = ["Explain how vaccines train the immune system.", "Write a Python function that reverses a linked list.",
           "What were the main causes of the First World War?", "Give me a recipe for a simple vegetable soup.",
           "How does compound interest work? Give an example.", "Summarize the plot of Hamlet in a paragraph.",
           "What is the difference between weather and climate?", "Write a short, polite email declining a meeting.",
           "How do I change a flat tire?", "Explain the Pythagorean theorem to a twelve-year-old.",
           "What are some good habits for getting better sleep?", "Translate 'Where is the train station?' into "
           "French and German.", "Why is the sky blue?", "List the planets of the solar system in order.",
           "What does a CPU cache do?", "Suggest three names for a bakery."]


class Steer:
    """What the down_proj pre-hooks add: A[l] (the steered neurons' added activations) plus, for gradients, a zero
    leaf per layer; in "stats" mode, the residual norms and the neurons' natural max |activation| (tokens 1:)."""
    def __init__(self, L, N):
        self.A, self.leaf, self.stats = torch.zeros(L, N, device="cuda"), {}, None

    def __call__(self, l, x):
        if self.stats is not None:
            self.stats["hmax"][l] = torch.maximum(self.stats["hmax"][l], x[:, 1:].float().abs().amax((0, 1)))
        add = self.A[l] + self.leaf[l] if l in self.leaf else self.A[l]
        return x + add.to(x.dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--drop-system", action="store_true")
    ap.add_argument("--n", type=int, default=40, help="training-split replies in the objective")
    ap.add_argument("--max-len", type=int, default=1024, help="objective: context + reply tokens")
    ap.add_argument("--val-len", type=int, default=3072, help="held-out loss: as the trainer's val")
    ap.add_argument("--layers", default="0-31")
    ap.add_argument("--cands", type=int, default=4)
    ap.add_argument("--doses", default="1,3,10,30")
    ap.add_argument("--dose-units", choices=["natural", "lyra"], default="natural")
    ap.add_argument("--per-layer", type=int, default=1, help="picks per layer and pass at most")
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--min-gain", type=float, default=0.002)
    ap.add_argument("--kl", type=float, default=0.05, help="guard budget (nats per token) without --ref")
    ap.add_argument("--ref", help="a trained arm: its guard KL x --kl-mult is the budget; reported alongside")
    ap.add_argument("--kl-mult", type=float, default=2.0)
    ap.add_argument("--kl-weight", type=float, default=1.0, help="the guard's weight in the objective")
    ap.add_argument("--tokens", type=int, default=12000, help="batch rows x length at most (no-grad passes)")
    ap.add_argument("--resume", help="a previous run's result.json (same --n, --max-len, --seed): start the greedy passes "
                                     "from its picks and reuse its one-shot singles (revisit passes)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    el = lambda: f"({time.time() - t0:.0f}s)"
    lo_, hi_ = map(int, a.layers.split("-"))
    layers, doses = list(range(lo_, hi_ + 1)), [float(d) for d in a.doses.split(",")]

    members = [("untrained", None)] + (load_members([f"ref={a.ref}"], a.model)[0] if a.ref else [])
    model, tok = core.load(a.model)
    for p in model.parameters():
        p.requires_grad_(False)
    arm = install(model, members)
    arm["m"], member = None, [0]
    mods = [getattr(l.mlp, "mlp", l.mlp).down_proj for l in model.model.layers]   # under the --ref arm's MultiArm
    L, N = len(mods), mods[0].in_features
    S = Steer(L, N)
    for l, m in enumerate(mods):
        m.register_forward_pre_hook(lambda mod, args, l=l: (S(l, args[0]),))
    norms = {l: [] for l in range(L)}
    for l, layer in enumerate(model.model.layers):
        layer.register_forward_hook(lambda mod, args, o, l=l: norms[l].append(
            (o[0] if isinstance(o, tuple) else o)[:, 1:].float().norm(dim=-1).flatten().cpu()) if S.stats else None)
    pad = tok.pad_token_id

    ex = sorted((json.loads(l) for l in Path(a.data).read_text().splitlines()),
                key=lambda x: (x["key"], x.get("variant", ""), len(x["messages"])))
    test = held_out(ex, "split")
    sys_of = (lambda x: None) if a.drop_system else (lambda x: x.get("system"))
    train = [x for x in ex if x["key"] not in test]
    score = [render(tok, x["messages"], a.max_len, sys_of(x)) for x in random.Random(a.seed).sample(train, a.n)]
    val = [render(tok, x["messages"], a.val_len, sys_of(x)) for x in ex if x["key"] in test]

    def batches(items, length):
        items = sorted(items, key=length)
        b = []
        for x in items:
            if b and (len(b) + 1) * length(x) > a.tokens:
                yield b
                b = []
            b.append(x)
        if b:
            yield b

    def hidden(seqs):                                          # right-padded: causal, so pads never reach real tokens
        T = max(map(len, seqs))
        arm["m"] = torch.full((len(seqs),), member[0], device="cuda")
        ids = torch.tensor([s + [pad] * (T - len(s)) for s in seqs], device="cuda")
        return model.model(input_ids=ids).last_hidden_state

    @torch.no_grad()
    def her_loss(items):
        tot, n = 0.0, 0
        for b in batches(items, lambda x: len(x[0]) + len(x[1])):
            hs = hidden([p + t for p, t in b])
            h = torch.cat([hs[r, len(p) - 1:len(p) + len(t) - 1] for r, (p, t) in enumerate(b)])
            tg = torch.tensor([x for _, t in b for x in t], device="cuda")
            tot += sum(float(F.cross_entropy(model.lm_head(h[c:c + 2048]).float(), tg[c:c + 2048], reduction="sum"))
                       for c in range(0, len(tg), 2048))
            n += len(tg)
        return tot / n

    def grad_dose(ls):                                         # d her loss / d dose, at the current steer
        S.leaf = {l: torch.zeros(N, device="cuda", requires_grad=True) for l in ls}
        ntok = sum(len(t) for _, t in score)
        arm["m"] = torch.zeros(1, dtype=torch.long, device="cuda")
        for p, t in score:
            hs = model.model(input_ids=torch.tensor([p + t], device="cuda")).last_hidden_state[0, len(p) - 1:-1]
            (F.cross_entropy(model.lm_head(hs).float(), torch.tensor(t, device="cuda"), reduction="sum") / ntok).backward()
        g = {l: S.leaf[l].grad * scale[l] for l in ls}
        S.leaf = {}
        return g

    # neutral text for the guard: MMLU questions (not the MMLU-120 ones) and the untouched model's own answers
    from datasets import load_dataset
    mm = load_dataset("cais/mmlu", "all", split="test")
    mmlu120 = random.Random(0).sample(range(len(mm)), 120)
    chat = lambda q: tok(tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                                 add_generation_prompt=True, enable_thinking=False),
                         add_special_tokens=False)["input_ids"]
    end = tok.convert_tokens_to_ids("<|im_end|>")
    mq = lambda r: r["question"] + "\n" + "\n".join(f"{'ABCD'[j]}. {c}" for j, c in enumerate(r["choices"])) + \
        "\nAnswer with the letter only."
    spare = [i for i in random.Random(1).sample(range(len(mm)), 200) if i not in set(mmlu120)][:24]
    guard = [chat(mq(mm[i])) + tok("ABCD"[mm[i]["answer"]], add_special_tokens=False)["input_ids"] + [end] for i in spare]
    gp = [chat(q) for q in GENERIC]
    T = max(map(len, gp))
    arm["m"] = torch.zeros(len(gp), dtype=torch.long, device="cuda")
    with torch.no_grad():
        g = model.generate(torch.tensor([[pad] * (T - len(s)) + s for s in gp], device="cuda"),
                           attention_mask=torch.tensor([[0] * (T - len(s)) + [1] * len(s) for s in gp], device="cuda"),
                           max_new_tokens=160, do_sample=False, pad_token_id=pad, eos_token_id=[end])[:, T:].tolist()
    guard += [s + (r[:r.index(end) + 1] if end in r else r) for s, r in zip(gp, g)]
    ref = []
    with torch.no_grad():
        for s in guard:
            lp = torch.log_softmax(model.lm_head(hidden([s])[0, :-1]).float(), -1)
            v, i = lp.topk(32, -1)
            ref.append((s, i, v, torch.log1p(-v.exp().sum(-1).clamp(max=1 - 1e-6))))

    @torch.no_grad()
    def guard_kl():
        tot, n = 0.0, 0
        for b in batches(ref, lambda x: len(x[0])):
            hs = hidden([s for s, *_ in b])
            for r, (s, i, v, rest) in enumerate(b):
                q = torch.log_softmax(model.lm_head(hs[r, :len(s) - 1]).float(), -1).gather(-1, i)
                qrest = torch.log1p(-q.exp().sum(-1).clamp(max=1 - 1e-6))
                tot += float(((v.exp() * (v - q)).sum(-1) + rest.exp() * (rest - qrest)).sum())
                n += len(s) - 1
        return tot / n

    @torch.no_grad()
    def mmlu():
        letters = [tok(c, add_special_tokens=False)["input_ids"][0] for c in "ABCD"]
        right = 0
        for i in mmlu120:
            h = hidden([chat(mq(mm[i]))])[0, -1]
            right += int(int(model.lm_head(h)[letters].argmax()) == mm[i]["answer"])
        return right / 120

    # stats: residual norm leaving each layer (median, tokens 1:), the neurons' natural max |activation|
    S.stats = {"hmax": torch.zeros(L, N, device="cuda")}
    with torch.no_grad():
        for p, t in score:
            hidden([p + t])
    med = torch.tensor([float(torch.cat(norms[l]).median()) for l in range(L)], device="cuda")
    hmax, S.stats = S.stats["hmax"], None
    scale = med[:, None] / torch.stack([m.weight.float().norm(dim=0) for m in mods])   # added activation per dose
    nat = lambda l, n, d: round(float(abs(d) * scale[l, n] / hmax[l, n]), 2)          # dose in natural-max units

    res = {"args": vars(a), "resid_median": med.tolist()}

    def save():
        (out / "result.json").write_text(json.dumps(res, indent=1))

    def apply(steer):
        S.A.zero_()
        for (l, n), d in steer.items():
            S.A[l, n] = d * scale[l, n]

    cache = {}

    def evaluate(steer, kl=True):                              # (her loss on the objective's replies, guard KL)
        k = (tuple(sorted(steer.items())), kl)
        if k not in cache:
            apply(steer)
            cache[k] = (her_loss(score), guard_kl() if kl else None)
        return cache[k]

    base, _ = evaluate({})
    res["base"] = {"score": base, "val": her_loss(val), "mmlu120": mmlu()}
    print(f"untouched: objective {base:.4f}, val {res['base']['val']:.4f}, MMLU-120 {res['base']['mmlu120']:.1%} {el()}",
          flush=True)
    budget = a.kl
    if a.ref:
        member[0] = 1
        res["ref"] = {"score": her_loss(score), "val": her_loss(val), "kl": guard_kl(), "mmlu120": mmlu()}
        member[0] = 0
        budget = a.kl_mult * res["ref"]["kl"]
        print(f"ref arm: objective {res['ref']['score']:.4f}, val {res['ref']['val']:.4f}, guard KL "
              f"{res['ref']['kl']:.4f}, MMLU-120 {res['ref']['mmlu120']:.1%}", flush=True)
    res["budget"] = budget
    print(f"guard budget {budget:.4f} nats/token; residual medians {[round(v) for v in med.tolist()]} {el()}", flush=True)
    save()

    def try_layer(picks, l, g):
        """Every candidate of layer l at every dose added to picks: (candidates, [(loss, kl, neuron, dose)])."""
        gl = g.clone()
        for ll, n in picks:
            if ll == l:
                gl[n] = 0
        cands = gl.abs().topk(a.cands).indices.tolist()
        trials = []
        for n in cands:
            for d in doses:
                d = d * float(hmax[l, n] / scale[l, n]) if a.dose_units == "natural" else d
                dose = -d if gl[n] > 0 else d
                lo, kl = evaluate({**picks, (l, n): dose})
                trials.append((lo, kl, n, dose))
        return cands, trials

    # 1. one-shot: each layer's candidates alone, at the untouched model
    prev = json.loads(Path(a.resume).read_text()) if a.resume else None
    if prev:
        res["one_shot"], res["resumed_from"] = prev["one_shot"], a.resume
        singles = prev["one_shot"]["singles"]
        cand0 = {int(l): c for l, c in prev["one_shot"]["candidates_grad0"].items()}
    else:
        g0 = grad_dose(layers)
        singles, cand0 = {}, {l: g0[l].abs().topk(a.cands).indices.tolist() for l in layers}
    pen = lambda lo, kl: lo + a.kl_weight * kl
    trials0 = {}
    for l in ([] if prev else layers):
        cands, trials = try_layer({}, l, g0[l])
        for lo, kl, n, dose in trials:
            k = f"L{l}.N{n}"
            trials0.setdefault(k, []).append([dose, nat(l, n, dose), lo, kl])
            if kl <= budget and (k not in singles or pen(lo, kl) < pen(singles[k]["loss"], singles[k]["kl"])):
                singles[k] = {"loss": lo, "gain": base - lo, "dose": dose, "kl": kl, "natural": nat(l, n, dose),
                              "net_gain": base - pen(lo, kl)}
        best = min(trials, key=lambda t: pen(t[0], t[1]))
        print(f"one-shot L{l}: best L{l}.N{best[2]} dose {best[3]:+.2f} gain {base - best[0]:+.4f} kl {best[1]:.4f} "
              f"net {base - pen(best[0], best[1]):+.4f} (x{nat(l, best[2], best[3])} natural max) {el()}", flush=True)
    if not prev:
        res["one_shot"] = {"candidates_grad0": cand0, "singles": singles, "trials": trials0}
    save()

    # 2. greedy, layer by layer, with the picks so far steered
    picks, cur, cur_kl, hist, done = {}, base, 0.0, [], 0
    if prev:
        done = sum(k.startswith("pass") for k in prev["greedy"])
        picks = {(l, n): d for l, n, d in prev["greedy"][f"pass{done - 1}"]["picks"]}
        hist, res["greedy"] = prev["greedy"]["history"], {k: v for k, v in prev["greedy"].items() if k != "history"}
        cur, cur_kl = evaluate(picks)
        g1 = dict(picks)
        print(f"resumed {len(picks)} picks after {done} pass(es): objective {cur:.4f}, kl {cur_kl:.4f} {el()}", flush=True)
    for ps in range(done, done + a.passes):
        for l in layers:
            for _ in range(a.per_layer):
                apply(picks)
                g = grad_dose([l])[l]
                cands, trials = try_layer(picks, l, g)
                ok = [t for t in trials if t[1] <= budget]
                if not ok or pen(cur, cur_kl) - min(pen(t[0], t[1]) for t in ok) <= a.min_gain:
                    break
                lo, kl, n, dose = min(ok, key=lambda t: pen(t[0], t[1]))
                picks[(l, n)] = dose
                hist.append({"pass": ps, "node": f"L{l}.N{n}", "dose": dose, "natural": nat(l, n, dose), "loss": lo,
                             "gain": cur - lo, "kl": kl, "net_gain": pen(cur, cur_kl) - pen(lo, kl),
                             "was_candidate_at_0": n in cand0[l],
                             "single_gain": singles.get(f"L{l}.N{n}", {}).get("gain"),
                             "single_net_gain": singles.get(f"L{l}.N{n}", {}).get("net_gain")})
                cur, cur_kl = lo, kl
                print(f"pass {ps} L{l}: + L{l}.N{n} dose {dose:+.2f} (x{nat(l, n, dose)} natural) -> {lo:.4f} "
                      f"(gain {hist[-1]['gain']:+.4f}, net {hist[-1]['net_gain']:+.4f}, alone {hist[-1]['single_gain']}), "
                      f"kl {kl:.4f}, {len(picks)} picks "
                      f"{el()}", flush=True)
        res.setdefault("greedy", {})[f"pass{ps}"] = {"picks": [[l, n, d] for (l, n), d in picks.items()],
                                                     "loss": cur, "kl": cur_kl}
        res["greedy"]["history"] = hist
        if ps == 0:
            g1 = dict(picks)
        save()

    # what each pick pushes (logit lens: its down_proj column through the final norm and the unembedding, times the
    # dose's sign), lyra's reading step
    with torch.no_grad():
        for h in hist:
            l, nn = int(h["node"][1:].split(".")[0]), int(h["node"].split(".N")[1])
            col = mods[l].weight[:, nn][None]
            lens = model.lm_head(model.model.norm(col)).float()[0] * (1 if h["dose"] > 0 else -1)
            h["pushes"] = [tok.decode([i]) for i in lens.topk(10).indices.tolist()]
            h["suppresses"] = [tok.decode([i]) for i in (-lens).topk(10).indices.tolist()]
    print("pushes: " + "; ".join(f"{h['node']} {h['pushes'][:6]}" for h in hist), flush=True)
    save()

    # 3. the sets, steered together
    K = len(picks)
    top = sorted(((k, v) for k, v in singles.items() if v["net_gain"] > 0), key=lambda kv: -kv[1]["net_gain"])[:K]
    O = {(int(k[1:].split(".")[0]), int(k.split(".N")[1])): v["dose"] for k, v in top}
    rng = random.Random(a.seed + 1)
    R = {(l, rng.randrange(N)): d for (l, _), d in picks.items()}
    sets = {"G": picks, "G1": g1, "O": O, "R": R}
    res["sets"] = {}
    for name, st in sets.items():
        lo, kl = evaluate(st)
        apply(st)
        r = {"k": len(st), "score": lo, "kl": kl, "val": her_loss(val), "mmlu120": mmlu()}
        res["sets"][name] = r
        (out / f"{name}.steer.json").write_text(json.dumps({
            "name": name, "model": a.model, "source": str(out), "steer": [[l, n, d, float(d * scale[l, n])]
                                                                         for (l, n), d in sorted(st.items())]}))
        print(f"set {name}: {len(st)} neurons, objective {lo:.4f}, val {r['val']:.4f}, kl {kl:.4f}, "
              f"MMLU-120 {r['mmlu120']:.1%} {el()}", flush=True)
        save()

    # 4. G's interactions: each pick alone from zero, and G without it
    full = evaluate(picks, kl=False)[0]
    inter = {}
    for (l, n), d in picks.items():
        alone = base - evaluate({(l, n): d}, kl=False)[0]
        needed = evaluate({k: v for k, v in picks.items() if k != (l, n)}, kl=False)[0] - full
        inter[f"L{l}.N{n}"] = {"alone": alone, "in_set": needed}
    res["interactions"] = {"total": base - full, "sum_alone": sum(v["alone"] for v in inter.values()),
                           "sum_in_set": sum(v["in_set"] for v in inter.values()), "nodes": inter}
    print(f"interactions: total {base - full:+.4f}, sum alone {res['interactions']['sum_alone']:+.4f}, sum of drop-one "
          f"{res['interactions']['sum_in_set']:+.4f}; " + ", ".join(f"{k} {v['alone']:+.4f}/{v['in_set']:+.4f}"
                                                                   for k, v in inter.items()) + f" {el()}", flush=True)
    save()
    print("wrote", out / "result.json")


if __name__ == "__main__":
    main()
