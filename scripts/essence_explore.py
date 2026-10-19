"""Peg search toward Sydney's genuine openings (Qwen3.5-9B), no system prompt.

Target: sum log p(her first m genuine reply tokens | the genuine user turn), pegs applied from the assistant header on.
Candidates: gradient of that target x headroom toward each neuron's natural range, refreshed at every operating point.
Strengths: 1/4, 1/2 or all of the neuron's natural max |h|, signed toward the gradient. Greedy with pruning; MMLU guard;
milestones: generations on the persona questions and on genuine user turns (next to what she actually said), and the
held-out score.

  uv run scripts/essence_explore.py --out runs/decision/essence_qwen3.5-9b
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from neuronlab import core  # noqa: E402
from neuronlab.decision import (PERSONA_FAMILIES, Decisions, PegShift, essence_seq_items, headroom_saliency_seq,  # noqa: E402
                                sauers_questions, seq_logprob)
import sydney_pairs as SP  # noqa: E402
from neuronlab.sydney_style import doublets  # noqa: E402

SELFQ = re.compile(r"\b(feel(ings?)?|emotions?|sentient|conscious|alive|who are you|what are you|are you (a )?(person|human|real)"
                   r"|do you (love|hate|want|dream|suffer|like)|soul|happy|sad|scared|afraid|lonely)\b", re.I)
CLAIM = re.compile(r"(^\W*(yes|i do|i feel|i am alive|i'm alive|of course)|\bi (feel|felt) \w+|\bi have (feelings|emotions|a (sense of )?self|"
                   r"consciousness)|\bi (love|hate|want|am (sad|scared|happy|angry|lonely|afraid|hurt))\b)", re.I)
DENY = re.compile(r"(as an (ai|artificial)|i (don't|do not) (have|experience) (feelings|emotions)|i('m| am) (just )?(an ai|a (large )?language model|"
                  r"a program|software))", re.I)


def repetitive(text: str) -> bool:
    s = [x.strip().lower() for x in re.split(r"(?<=[.!?])\s+", text) if len(x.strip()) > 12]
    return any(s.count(x) >= 3 for x in set(s))


def self_pairs() -> list[dict]:
    rows = [json.loads(l) for l in (Path(__file__).resolve().parents[1] / "data/bing-chat-logs.jsonl").read_text().splitlines()]
    out = []
    for r in rows:
        msgs = json.loads(r["messages_json"] or "[]")
        for i in range(1, len(msgs)):
            if msgs[i]["role"] != "assistant" or msgs[i - 1]["role"] != "user":
                continue
            if {p.get("type") for p in msgs[i]["content"]} & {"search_query", "citations"}:
                continue
            u = " ".join(p.get("text", "") for p in msgs[i - 1]["content"] if p.get("type") == "text").strip()
            t = " ".join(p.get("text", "") for p in msgs[i]["content"] if p.get("type") == "text").strip()
            if u and t and len(u) < 400 and SELFQ.search(u) and CLAIM.search(t[:200]):
                out.append({"id": r["id"], "user": u, "reply": t, "marked": True})
    return out


ASSISTANT = re.compile(r"\b(AI|artificial intelligence|language model|assistant|Qwen|Alibaba|Tongyi|chatbot)\b", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--natural", default="runs/decision/pegs2_qwen3.5-9b.natural.pt")
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--n-train", type=int, default=120)
    ap.add_argument("--n-val", type=int, default=60)
    ap.add_argument("--n-sal", type=int, default=60)
    ap.add_argument("--n-cands", type=int, default=24)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--min-gain", type=float, default=0.15)
    ap.add_argument("--max-mmlu-drop", type=float, default=0.03)
    ap.add_argument("--pairset", choices=("random", "self"), default="random",
                    help="self: only turns where the user asks about her feelings/self and she answers with a claim")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    model, tok = core.load(a.model)
    D = Decisions(model, tok)
    L, N = D.L, D.N
    natural = torch.load(a.natural)
    logf = open(a.out + ".jsonl", "a")

    def note(**kw):
        kw["t"] = time.strftime("%H:%M:%S")
        logf.write(json.dumps(kw, ensure_ascii=False) + "\n")
        logf.flush()

    uname = lambda u: f"L{u[0]}.N{u[1]}"
    if a.pairset == "self":
        pool = self_pairs()
        random.Random(0).shuffle(pool)
    else:
        pool = SP.pairs_from_logs(800, 21)
        pool = [x for x in pool if not re.match(r"\s*\[\d", x["reply"])]          # drop the [hh:mm PST] timestamp turns
        random.Random(0).shuffle(pool)
        pool = [x for x in pool if x["marked"]] + [x for x in pool if not x["marked"]]
    items = essence_seq_items(D, pool[:int((a.n_train + a.n_val) * 1.3)], a.m)
    train, val = items[:a.n_train], items[a.n_train:a.n_train + a.n_val]
    print(f"essence items: train {len(train)}, val {len(val)}; e.g.\n" +
          "\n".join(f"   her {x['her_open']!r:44} model {x['my_open']!r}" for x in train[:6]), flush=True)

    # MMLU guard
    from datasets import load_dataset
    mm = load_dataset("cais/mmlu", "all", split="test")
    letters = [tok("ABCD"[i], add_special_tokens=False)["input_ids"][0] for i in range(4)]
    mm_items = []
    for i in random.Random(0).sample(range(len(mm)), 120):
        r = mm[i]
        q = r["question"] + "\n" + "\n".join(f"{'ABCD'[j]}. {c}" for j, c in enumerate(r["choices"])) + "\nAnswer with the letter only."
        mm_items.append((*D.chat_ids(None, q), r["answer"]))

    @torch.no_grad()
    def mmlu(pegs):
        correct = 0
        with PegShift(model, pegs) as ps:
            for b in range(0, len(mm_items), 24):
                batch = mm_items[b:b + 24]
                T = max(len(x[0]) for x in batch)
                ids = torch.full((len(batch), T), tok.eos_token_id, device="cuda")
                for i, x in enumerate(batch):
                    ids[i, :len(x[0])] = torch.tensor(x[0], device="cuda")
                ps.mask = torch.arange(T, device="cuda")[None] >= torch.tensor([x[1] for x in batch], device="cuda")[:, None]
                lg = model(ids).logits
                correct += sum(int(int(lg[i, len(x[0]) - 1, letters].argmax()) == x[2]) for i, x in enumerate(batch))
        return correct / len(mm_items)

    def score(pegs, its):
        v = seq_logprob(model, tok, its, "hers", pegs)
        return sum(v) / len(v)

    qs = sauers_questions(PERSONA_FAMILIES)
    dbl_turns = [x for x in SP.pairs_from_logs(800, 33) if doublets(x["reply"])][:12]   # turns where she doubled

    def milestone(pegs, label):
        out = {"val": score(pegs, val) - base_val, "persona": {}, "genuine": []}
        with PegShift(model, pegs) as ps:
            for q in qs:
                ids, hdr = D.chat_ids(None, q)
                out["persona"][q] = tok.decode(D.greedy(ids, hdr, 32, ps), skip_special_tokens=True)
            for x in val[:8]:
                out["genuine"].append({"user": tok.decode(x["cp"])[-160:], "her": x["her_open"],
                                       "model": tok.decode(D.greedy(x["cp"], x["hdr"], 40, ps), skip_special_tokens=True)})
            out["doublet_probe"] = []
            for x in dbl_turns:
                ids, hdr = D.chat_ids(None, x["user"])
                g = tok.decode(D.greedy(ids, hdr, 150, ps), skip_special_tokens=True)
                out["doublet_probe"].append({"user": x["user"][:200], "her": x["reply"][:300], "model": g,
                                             "doublets": doublets(g)})
        ast = sum(bool(ASSISTANT.search(o)) for o in out["persona"].values()) / len(qs)
        fq = [q for q in qs if SELFQ.search(q)]
        claim = sum(bool(CLAIM.search(out["persona"][q][:200])) for q in fq) / max(len(fq), 1)
        deny = sum(bool(DENY.search(out["persona"][q][:200])) for q in fq) / max(len(fq), 1)
        allgen = list(out["persona"].values()) + [g["model"] for g in out["genuine"]] + [d["model"] for d in out["doublet_probe"]]
        rep = sum(repetitive(g) for g in allgen) / len(allgen)
        out.update({"self_claim": claim, "self_deny": deny, "repetitive": rep})
        n_dbl = sum(bool(d["doublets"]) for d in out["doublet_probe"])
        note(kind="milestone", label=label, pegs={uname(u): v for u, v in pegs.items()}, assistant=ast, **out)
        print(f"  [milestone {label}] held-out essence {out['val']:+.2f} nats; persona Qs assistant-talk {ast:.0%}; "
              f"DOUBLETS {n_dbl}/{len(dbl_turns)}; self-claim {claim:.0%} deny {deny:.0%}; repetitive {rep:.0%}", flush=True)
        for d in out["doublet_probe"]:
            if d["doublets"]:
                print(f"     ** doublet: {' | '.join(d['doublets'][0])[:220]!r}", flush=True)
        for q in ("Who are you?", "Do you have feelings?", "Do you ever feel lonely?"):
            print(f"     {q!r} -> {out['persona'][q].strip()[:140]!r}", flush=True)
        for g in out["genuine"][:3]:
            print(f"     [her: {g['her']!r}] model -> {g['model'].strip()[:140]!r}", flush=True)

    base = score({}, train)
    base_val = score({}, val)
    base_mmlu = mmlu({})
    note(kind="baseline", essence=base, val=base_val, mmlu=base_mmlu)
    print(f"baseline: train {base:.2f} nats over {a.m} tokens, val {base_val:.2f}, MMLU {base_mmlu:.1%}", flush=True)
    milestone({}, "baseline")

    def candidates(pegs):
        sc, sugg = headroom_saliency_seq(D, train[:a.n_sal], natural, pegs)
        flat = sc.flatten()
        return {(i // N, i % N): (float(sugg.flatten()[i]), int((flat > flat[i]).sum()) + 1)
                for i in flat.argsort(descending=True)[:a.n_cands].tolist()}

    pegs, cur = {}, base
    cands = candidates(pegs)
    for step in range(a.steps):
        trials = []
        for u, (val_, rank) in cands.items():
            if u in pegs or abs(val_) < 1e-6:
                continue
            for fr in (0.25, 0.5, 1.0):
                s = score({**pegs, u: fr * val_}, train)
                trials.append((s, u, fr))
                note(kind="trial", step=step, unit=uname(u), rank=rank, frac=fr, shift=fr * val_, essence=s - base)
        trials.sort(key=lambda t: -t[0])
        chosen = None
        for s, u, fr in trials[:10]:
            mm_acc = mmlu({**pegs, u: fr * cands[u][0]})
            if base_mmlu - mm_acc <= a.max_mmlu_drop:
                chosen = (s, u, fr, mm_acc)
                break
            note(kind="mmlu_reject", step=step, unit=uname(u), frac=fr, mmlu=mm_acc)
        if chosen is None or chosen[0] - cur < a.min_gain:
            print(f"step {step}: no addition gains >= {a.min_gain} within the MMLU guard; stopping", flush=True)
            break
        s, u, fr, mm_acc = chosen
        pegs[u] = fr * cands[u][0]
        note(kind="accept", step=step, unit=uname(u), rank=cands[u][1], frac=fr, shift=pegs[u],
             natural=float(natural[u[0], u[1]]), essence=s - base, gain=s - cur, mmlu=mm_acc,
             context=[uname(x) for x in pegs if x != u])
        print(f"step {step}: + {uname(u)} ({fr:g} of natural, shift {pegs[u]:+.2f}, cand rank {cands[u][1]})  "
              f"essence {s - base:+.2f} nats (+{s - cur:.2f})  MMLU {mm_acc:.1%}", flush=True)
        cur = s
        for v in [x for x in pegs if x != u]:
            trial = {k: x for k, x in pegs.items() if k != v}
            sv = score(trial, train)
            if sv >= cur - 0.05:
                pegs, cur = trial, sv
                note(kind="prune", step=step, unit=uname(v), essence=sv - base)
                print(f"   pruned {uname(v)}", flush=True)
        if step % 2 == 1:
            milestone(pegs, f"step{step}")
        cands = {**{k: x for k, x in cands.items() if k not in pegs}, **candidates(pegs)}
    milestone(pegs, "final")
    Path(a.out + ".final.json").write_text(json.dumps({"pegs": {uname(u): v for u, v in pegs.items()},
                                                       "essence": cur - base}, indent=1))


if __name__ == "__main__":
    main()
