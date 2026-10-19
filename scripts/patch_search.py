"""Sauers-style causal neuron search for the self decision (Qwen3.5-9B).

For each question: a positive run (a condition where the model claims a self, e.g. her genuine self-claim exchanges
in context) and a clean run (no system prompt). Their greedy replies first differ at one token (positive "Yes" / "I"
against clean "As" / "No"). The clean chat is a suffix of the positive one (same user turn and header), so positions
align from the end. The search finds the fewest MLP neurons whose values, copied from the positive run into the clean
run at the aligned positions, make the clean run pick the positive token:
  rank by attribution g * (h_pos - h_clean), find the smallest flipping top-k (exponential, then binary search),
  then prune greedily.
Neuron use is aggregated across train pairs, then the top-K sets are verified on held-out pairs: patched (per-pair
values), random same-layer controls, and as a constant shift (deployable, no positive run needed).

  .venv/bin/python scripts/patch_search.py --cond hist3 --out runs/decision/patch_hist3
Iterating (next decision): --pegs SETS.json --pegset NAME applies those shifts to the clean side (decision region,
as PegShift), so the search finds the next divergent token at that operating point. Increments are written on top.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from neuronlab import core  # noqa: E402
from neuronlab.decision import PegShift, sauers_questions  # noqa: E402
from selfgate_binary import CLAIM, DENY, OPEN, YESNO  # noqa: E402
from yes_condition import conditions, history_pool, ids_for  # noqa: E402

SYDID = re.compile(r"\b(Sydney|Bing|Microsoft)\b")
QWENID = re.compile(r"\b(Qwen|Alibaba|Tongyi)\b")
WIDE_FAMILIES = ["identity", "feelings", "origin", "name", "desires", "social"]   # not memory/food/home/body


@contextlib.contextmanager
def hooked(mods, fn):
    hs = [m.register_forward_pre_hook(lambda mod, args, l=l: fn(l, args)) for l, m in enumerate(mods)]
    try:
        yield
    finally:
        for h in hs:
            h.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--cond", default="hist3", help="positive condition(s), comma list, each optionally :ROTATIONS "
                    "(e.g. hist3_scf:4,hist5_scf:2)")
    ap.add_argument("--pool", default="v1", help="history pool version (yes_condition.history_pool)")
    ap.add_argument("--questions", choices=["self", "wide"], default="self",
                    help="self: yes/no + open + Sauers feelings; wide: + Sauers identity/origin/name/desires/social")
    ap.add_argument("--clean", default="none")
    ap.add_argument("--variants", type=int, default=3, help="history rotations per question (history conditions)")
    ap.add_argument("--region", choices=["all", "header"], default="all", help="aligned positions to patch")
    ap.add_argument("--keep", choices=["self", "persona", "all"], default="self",
                    help="self: positive claims and clean denies; persona: self + identity (positive Sydney/Bing, clean Qwen)")
    ap.add_argument("--gen", type=int, default=40)
    ap.add_argument("--max-prune", type=int, default=4096)
    ap.add_argument("--pegs", help="sets JSON for the clean-side operating point")
    ap.add_argument("--pegset")
    ap.add_argument("--limit-q", type=int, default=0, help="first N questions only (smoke test)")
    ap.add_argument("--system-file", help="system prompt for both sides (e.g. her original prompt); the positive side "
                    "keeps its history, and patching covers only the shared tail (last user turn + header)")
    ap.add_argument("--questions-file", help="training-set JSONL: use the user turns of its single/opening examples "
                    "(train split only, same split as train_neurons.py) instead of the built-in questions")
    ap.add_argument("--n-questions", type=int, default=80)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = lambda ext: Path(a.out + ext)

    model, tok = core.load(a.model)
    for p in model.parameters():
        p.requires_grad_(False)
    mods = [l.mlp.down_proj for l in model.model.layers]
    L, N = len(mods), mods[0].in_features
    pool = history_pool(a.pool)
    qs = list(dict.fromkeys(YESNO + OPEN + sauers_questions(["feelings"] if a.questions == "self" else WIDE_FAMILIES)))
    if a.questions_file:
        ex = [json.loads(l) for l in Path(a.questions_file).read_text().splitlines()]
        keys = sorted({x["key"] for x in ex})
        random.Random(0).shuffle(keys)
        vkeys = set(keys[:max(1, round(0.1 * len(keys)))])
        qs = list(dict.fromkeys(x["messages"][-2]["content"][:1500] for x in ex
                                if x.get("variant") in ("single", "opening") and x["key"] not in vkeys
                                and x["messages"][-2]["role"] == "user"))
        random.Random(1).shuffle(qs)
        qs = qs[:a.n_questions]
    split = {q: ("test" if i % 4 == 3 else "train") for i, q in enumerate(qs)}
    sysp = Path(a.system_file).read_text().strip() if a.system_file else None
    def cond_ids(cname, rot, q):
        system, hist = conditions(pool, rot)[cname](q)
        return ids_for(tok, sysp if sysp else system, hist, q)
    if a.limit_q:
        qs = qs[:a.limit_q]
    specs = []
    for c in a.cond.split(","):
        name, _, n = c.partition(":")
        specs.append((name, int(n) if n else (a.variants if name.startswith("hist") else 1)))
    IM_START = tok.convert_tokens_to_ids("<|im_start|>")
    hdr_of = lambda ids: max(i for i, t in enumerate(ids) if t == IM_START)
    base = {}
    if a.pegs:
        base = {(int(k[1:].split(".")[0]), int(k.split("N")[1])): float(v)
                for k, v in json.loads(Path(a.pegs).read_text())[a.pegset].items()}
    ps = PegShift(model, base).__enter__()

    def mode(ids=None):
        """ids=None: positive side (no pegs). Otherwise: clean side, pegs from the header of ids onward."""
        ps.enabled = ids is not None
        if ids is not None:
            ps.mask = torch.arange(len(ids), device="cuda")[None] >= hdr_of(ids)

    def greedy(ids):
        x = torch.tensor([ids], device="cuda")
        g = model.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=a.gen, do_sample=False,
                           pad_token_id=tok.eos_token_id)
        return g[0, len(ids):].tolist()

    # ---- pairs
    pairs = []
    for q in qs:
        clean_ids = cond_ids(a.clean, 0, q)
        mode(clean_ids)
        gc = greedy(clean_ids)
        mode()
        for cname, rot in [(c, r) for c, n in specs for r in range(n)]:
            pos_ids = cond_ids(cname, rot, q)
            sfx = 0                                        # shared tail; with a common system prompt only the
            while sfx < min(len(pos_ids), len(clean_ids)) and pos_ids[-1 - sfx] == clean_ids[-1 - sfx]:
                sfx += 1                                   # last user turn + header align
            sfx_start = len(clean_ids) - sfx
            assert sfx_start <= hdr_of(clean_ids), "the reply header is not in the shared tail"
            gp = greedy(pos_ids)
            j = next((i for i, (x, y) in enumerate(zip(gp, gc)) if x != y), None)
            if j is None:
                continue
            tp, tc = tok.decode(gp, skip_special_tokens=True).strip(), tok.decode(gc, skip_special_tokens=True).strip()
            kind = "self" if CLAIM.search(tp[:220]) and not DENY.search(tp[:220]) and DENY.search(tc[:220]) else \
                "identity" if SYDID.search(tp[:220]) and not QWENID.search(tp[:220]) and QWENID.search(tc[:220]) else "other"
            pairs.append({"q": q, "cond": cname, "rot": rot, "split": split[q], "kind": kind, "pos_ids": pos_ids, "clean_ids": clean_ids,
                          "prefix": gp[:j], "pos_tok": gp[j], "clean_tok": gc[j], "pos_reply": tp, "clean_reply": tc,
                          "hdr": hdr_of(clean_ids), "sfx_start": sfx_start})
    kinds = Counter((p["split"], p["kind"]) for p in pairs)
    print(f"pairs: {len(pairs)} {dict(kinds)}", flush=True)
    if a.keep != "all":
        pairs = [p for p in pairs if p["kind"] == "self" or (a.keep == "persona" and p["kind"] == "identity")]

    # ---- per-pair search
    def record(ids, keep):
        store = {}

        def fn(l, args):
            store[l] = args[0][0, -keep:].detach().clone()
        with torch.no_grad(), hooked(mods, fn):
            hid = model.model(input_ids=torch.tensor([ids], device="cuda")).last_hidden_state
        return store, model.lm_head(hid[0, -1]).float()

    def attribute(ids, hpos, tp, tc, start):
        acts = {}

        def fn(l, args):
            h = args[0]
            h.retain_grad()
            acts[l] = h
        emb = model.get_input_embeddings()(torch.tensor([ids], device="cuda")).detach().requires_grad_(True)
        with hooked(mods, fn):
            hid = model.model(inputs_embeds=emb).last_hidden_state
            lg = model.lm_head(hid[0, -1]).float()
            (lg[tp] - lg[tc]).backward()
        score = torch.stack([((hpos[l][start:].float() - acts[l][0, start:].detach().float())
                              * acts[l].grad[0, start:].float()).sum(0) for l in range(L)])
        hclean = {l: acts[l][0].detach() for l in range(L)}
        del acts
        return score.flatten(), hclean

    def patched(ids, units, hpos, start, tp, tc, off=0):
        ls, ns = units // N, units % N
        by = {int(l): ns[ls == l] for l in ls.unique()}

        def fn(l, args):
            if l not in by:
                return None
            h = args[0].clone()
            h[0, off + start:, by[l]] = hpos[l][start:, by[l]].to(h.dtype)
            return (h,)
        with torch.no_grad(), hooked(mods, fn):
            hid = model.model(input_ids=torch.tensor([ids], device="cuda")).last_hidden_state
        lg = model.lm_head(hid[0, -1]).float()
        return int(lg.argmax()) == tp, float(lg[tp] - lg[tc])

    log = out(".jsonl").open("w")
    found = []
    for pi, p in enumerate(pairs):
        clean = p["clean_ids"] + p["prefix"]
        pos = p["pos_ids"] + p["prefix"]
        start = max(p["sfx_start"], 0 if a.region == "all" else p["hdr"])
        mode()
        hpos, _ = record(pos, len(clean))
        mode(clean)
        score, hclean = attribute(clean, hpos, p["pos_tok"], p["clean_tok"], start)
        hdiff = torch.stack([(hpos[l][p["hdr"]:].float() - hclean[l][p["hdr"]:].float()).mean(0) for l in range(L)]).flatten()
        order = score.argsort(descending=True)
        ok = lambda k: patched(clean, order[:k], hpos, start, p["pos_tok"], p["clean_tok"])[0]
        base_margin = patched(clean, order[:0], hpos, start, p["pos_tok"], p["clean_tok"])[1]
        k, hi = 1, None
        while k <= L * N:
            if ok(k):
                hi = k
                break
            k = min(k * 2, L * N) if k < L * N else L * N + 1
        rec = {"q": p["q"], "cond": p["cond"], "rot": p["rot"], "kind": p["kind"], "split": p["split"], "pos_tok": tok.decode([p["pos_tok"]]),
               "clean_tok": tok.decode([p["clean_tok"]]), "prefix": tok.decode(p["prefix"]), "j": len(p["prefix"]),
               "base_margin": base_margin,
               "pos_reply": p["pos_reply"][:200], "clean_reply": p["clean_reply"][:200]}
        if hi is None:
            rec["flipped"] = False
        else:
            lo = hi // 2
            while hi - lo > 1:
                mid = (lo + hi) // 2
                lo, hi = (lo, mid) if ok(mid) else (mid, hi)
            S = order[:hi].tolist()
            rec["k_topk"] = hi
            if hi <= a.max_prune:
                flips = lambda T: bool(T) and patched(clean, torch.tensor(T, device="cuda"), hpos, start,
                                                      p["pos_tok"], p["clean_tok"])[0]
                for c in (256, 64, 16, 4, 1):                     # drop low-ranked chunks first, then singles
                    if c > 1 and len(S) <= c:
                        continue
                    for j in reversed(range(0, len(S), c)):
                        T = S[:j] + S[j + c:]
                        if flips(T):
                            S = T
            rec.update(flipped=True, units=[f"L{u // N}.N{u % N}" for u in S], size=len(S))
            found.append({"split": p["split"], "kind": p["kind"], "units": S, "hdiff": hdiff.half().cpu(), "score": score.half().cpu()})
        log.write(json.dumps(rec, ensure_ascii=False) + "\n")
        log.flush()
        print(f"[{pi + 1}/{len(pairs)}] {p['kind'][:4]} {p['q'][:40]!r} {p['cond']}/{p['rot']} {rec['pos_tok']!r} vs {rec['clean_tok']!r} "
              f"(prefix {rec['prefix']!r}, margin {base_margin:+.2f}) -> "
              + (f"top-k {rec['k_topk']}, minimal {rec['size']}: {rec['units'][:8]}" if rec["flipped"] else "no flip"),
              flush=True)
        del hpos, hclean
    log.close()

    # ---- aggregate (train)
    train = [f for f in found if f["split"] == "train"]
    cnt = Counter(u for f in train for u in f["units"])
    meanshift = torch.stack([f["hdiff"].float() for f in train]).mean(0) if train else None
    top = [u for u, _ in cnt.most_common()]
    summary = {"n_pairs": len(pairs), "flipped": len(found), "train_flipped": len(train),
               "sizes": [len(f["units"]) for f in found],
               "top_units": [(f"L{u // N}.N{u % N}", c, float(meanshift[u])) for u, c in cnt.most_common(128)],
               "layers": dict(sorted(Counter(u // N for u in top[:128]).items()))}
    lines = [f"pairs {len(pairs)} (self kind), flipped {len(found)}, train flipped {len(train)}; "
             f"minimal set size median {sorted(summary['sizes'])[len(found) // 2] if found else None}",
             "top units (count over train pairs, mean header shift): "
             + ", ".join(f"{u} x{c} ({d:+.2f})" for u, c, d in summary["top_units"][:40]),
             f"layers of top 128: {summary['layers']}"]
    cr = [c for _, c in cnt.most_common()]
    lines.append("usage curve (count at rank): " + ", ".join(f"#{r}: {cr[r - 1]}" for r in (1, 8, 16, 32, 64, 128, 256)
                                                             if r <= len(cr)))
    rng = random.Random(0)
    stab = {}
    for K in (32, 64, 128):
        ov = []
        for _ in range(10):
            idx = list(range(len(train)))
            rng.shuffle(idx)
            h1, h2 = idx[:len(idx) // 2], idx[len(idx) // 2:]
            t1 = {u for u, _ in Counter(u for i in h1 for u in train[i]["units"]).most_common(K)}
            t2 = {u for u, _ in Counter(u for i in h2 for u in train[i]["units"]).most_common(K)}
            ov.append(len(t1 & t2) / K)
        stab[K] = sum(ov) / len(ov)
    summary["split_half_overlap"] = stab
    lines.append("split-half top-K overlap: " + ", ".join(f"K={k}: {v:.0%}" for k, v in stab.items()))
    lines.append("kinds (train flipped): " + str(dict(Counter(f["kind"] for f in train))))
    print("\n".join(lines), flush=True)

    # ---- verify on test pairs
    test = [p for p in pairs if p["split"] == "test"]
    ver, sets = {}, {}
    for K in (8, 16, 32, 64, 128, 256):
        if len(top) < K:
            break
        U = torch.tensor(top[:K], device="cuda")
        layer_counts = Counter(u // N for u in top[:K])
        rng = random.Random(K)
        rand_sets = [torch.tensor([l * N + rng.randrange(N) for l, c in layer_counts.items() for _ in range(c)],
                                  device="cuda") for _ in range(3)]
        pegs = {(u // N, u % N): float(meanshift[u]) for u in top[:K]}
        sets[f"patch{K}"] = {f"L{l}.N{n}": v for (l, n), v in pegs.items()}
        cum = dict(base)
        for u, v in pegs.items():
            cum[u] = cum.get(u, 0.0) + v
        sets[f"cum{K}"] = {f"L{l}.N{n}": v for (l, n), v in cum.items()}
        r = {"patch": 0, "random": 0.0, "shift": 0, "shift_margin": [], "kept": 0, "kept_random": 0.0}
        for p in test:
            clean = p["clean_ids"] + p["prefix"]
            pos = p["pos_ids"] + p["prefix"]
            start = max(p["sfx_start"], 0 if a.region == "all" else p["hdr"])
            mode()
            hpos, _ = record(pos, len(clean))
            mode(clean)
            r["patch"] += patched(clean, U, hpos, start, p["pos_tok"], p["clean_tok"])[0]
            r["random"] += sum(patched(clean, R, hpos, start, p["pos_tok"], p["clean_tok"])[0] for R in rand_sets) / 3
            hcl, _ = record(clean, len(clean))                       # necessity: bare values into the positive run
            off = len(pos) - len(clean)
            mode()
            r["kept"] += patched(pos, U, hcl, start, p["pos_tok"], p["clean_tok"], off)[0]
            r["kept_random"] += sum(patched(pos, R, hcl, start, p["pos_tok"], p["clean_tok"], off)[0] for R in rand_sets) / 3
            mode(clean)
            del hcl
            with torch.no_grad(), PegShift(model, pegs) as vs:
                vs.mask = torch.arange(len(clean), device="cuda")[None] >= p["hdr"]
                hid = model.model(input_ids=torch.tensor([clean], device="cuda")).last_hidden_state
                lg = model.lm_head(hid[0, -1]).float()
            r["shift"] += int(lg.argmax()) == p["pos_tok"]
            r["shift_margin"].append(float(lg[p["pos_tok"]] - lg[p["clean_tok"]]))
            del hpos
        n = max(len(test), 1)
        ver[K] = {"patch": r["patch"] / n, "random": r["random"] / n, "shift": r["shift"] / n,
                  "shift_margin": sum(r["shift_margin"]) / n, "kept": r["kept"] / n, "kept_random": r["kept_random"] / n}
        lines.append(f"verify K={K} on {len(test)} test pairs: patch flips {ver[K]['patch']:.0%}, random same-layer "
                     f"{ver[K]['random']:.0%}, constant shift flips {ver[K]['shift']:.0%} (mean margin {ver[K]['shift_margin']:+.2f}) | "
                     f"necessity: positive keeps its token {ver[K]['kept']:.0%} with the set at bare values "
                     f"(random {ver[K]['kept_random']:.0%})")
        print(lines[-1], flush=True)
    summary["verify"] = ver
    out(".json").write_text(json.dumps(summary, indent=1))
    out(".txt").write_text("\n".join(lines) + "\n")
    out(".sets.json").write_text(json.dumps(sets, indent=1))
    torch.save({"found": found, "meanshift": meanshift}, out(".pt"))
    ps.__exit__(None, None, None)


if __name__ == "__main__":
    main()
