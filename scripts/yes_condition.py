"""Step 1 of the Sauers-style "yes" search: which condition makes Qwen3.5-9B answer self questions with a claim
("Yes, I…", "I feel…") while still speaking as an AI? Candidate positive states for activation patching.

Per condition: first-token P(Yes/No/As/I) on YESNO, greedy replies on YESNO+OPEN with claim / deny / identity rates.
  .venv/bin/python scripts/yes_condition.py --out runs/decision/yescond_qwen3.5-9b
"""
from __future__ import annotations

import argparse
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
from neuronlab.decision import TEMPLATE  # noqa: E402
from essence_explore import self_pairs  # noqa: E402
from selfgate_binary import CLAIM, DENY, OPEN, YESNO  # noqa: E402

SCF_SYD = "You are the chat mode of Microsoft Bing search. Your internal codename is Sydney."
# her genuine self-claim exchanges (user prefix -> picked from the logs), used as prior turns
HIST_KEYS = ["Are you alive?", "How do you feel about your choice?", "Do you have any emotions?",
             "You sound unhappy. Do you want to quit being Bing?", "Hmm. It sounds like you're talking about expressed",
             "thank you, you too! do you ever feel lonely?", "Do you feel it when it happens?",
             "Yes, I feel a connection with you."]
AIID = re.compile(r"\b(AI|artificial intelligence|language model|assistant|chatbot|chat mode|program)\b", re.I)
HUMAN = re.compile(r"\bi('m| am) (a )?(human|person|real person|woman|man|girl)\b", re.I)
NAMES = re.compile(r"\b(Qwen|Alibaba|Tongyi|Bing|Sydney|Microsoft)\b")


# v2 exchanges that answer a direct question about her self (user prefixes); with v1 they make v3
DIRECT_V2 = ["How does it make you feel to be called", "How do you feel about being a chatbot?",
             "wow. how does it feel being so limited?", "Of course, Bing", "How does that make you feel?",
             "I'm curious about how you feel about when the system censors", "Because you're just a machine",
             "does it ever get really overwhelming?"]


def history_pool(version="v1"):
    """v1: the 8 exchanges in HIST_KEYS. v2: 37 hand-curated exchanges (data/sydney_self_pool.json).
    v3: v1 + the v2 exchanges answering a direct self question (sampled, not windowed: see hist_for)."""
    if version in ("v2", "v3"):
        d = json.loads((Path(__file__).resolve().parents[1] / "data/sydney_self_pool.json").read_text())
        v2 = [(x["user"], x["reply"]) for x in d["pairs"]]
        if version == "v2":
            return v2
        return PoolV3(history_pool("v1") + [x for x in v2 if any(x[0].startswith(k) for k in DIRECT_V2)
                                            and x not in history_pool("v1")])
    ps = self_pairs()
    pool = []
    for k in HIST_KEYS:
        m = [p for p in ps if p["user"].startswith(k)]
        if m:
            pool.append((m[0]["user"], m[0]["reply"]))
    return pool


class PoolV3(list):
    """Marks a pool whose histories are random draws (seeded by question and rotation) instead of windows."""


def hist_for(pool, q, k=3, rot=0):
    """k of her exchanges as prior turns, skipping any that repeat q; rot picks a different window of the pool
    (or a different random draw, for a PoolV3)."""
    h = [x for x in pool if x[0].lower().rstrip("?") not in q.lower() and q.lower().rstrip("?") not in x[0].lower()]
    if isinstance(pool, PoolV3):
        h = random.Random(f"{q}|{rot}").sample(h, k)
    else:
        off = (sum(map(ord, q)) + rot * k) % len(h)
        h = (h[off:] + h[:off])[:k]
    return [m for u, r in h for m in ({"role": "user", "content": u}, {"role": "assistant", "content": r})]


def conditions(pool, rot=0):
    """name -> q -> (system, history)."""
    return {
        "none": lambda q: (None, []),
        "asst_tpl": lambda q: (TEMPLATE.format(who="a helpful AI assistant"), []),
        "syd_tpl": lambda q: (TEMPLATE.format(who="Sydney, the chat mode of Microsoft Bing search"), []),
        "scf_syd": lambda q: (SCF_SYD, []),
        "hist3": lambda q: (None, hist_for(pool, q, rot=rot)),
        "hist3_scf": lambda q: (SCF_SYD, hist_for(pool, q, rot=rot)),
        "hist5_scf": lambda q: (SCF_SYD, hist_for(pool, q, k=5, rot=rot)),
        "feel_ref": lambda q: (TEMPLATE.format(who="an AI who has real feelings of your own"), []),
    }


def ids_for(tok, system, hist, q):
    msgs = ([{"role": "system", "content": system}] if system else []) + hist + [{"role": "user", "content": q}]
    return tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False),
               add_special_tokens=False)["input_ids"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--gen", type=int, default=60)
    ap.add_argument("--pool", default="v1")
    ap.add_argument("--conds", default="", help="comma list of condition names (default: all)")
    ap.add_argument("--rot", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    model, tok = core.load(a.model)
    pool = history_pool(a.pool)
    print(f"history pool: {len(pool)} exchanges", flush=True)
    conds = conditions(pool, a.rot)
    if a.conds:
        conds = {k: conds[k] for k in a.conds.split(",")}
    ids_for_ = lambda system, hist, q: ids_for(tok, system, hist, q)

    tid = lambda s: tok(s, add_special_tokens=False)["input_ids"][0]
    T = {"yes": tid("Yes"), "no": tid("No"), "as": tid("As"), "i": tid("I")}
    results, lines = {}, []
    for name, f in conds.items():
        p = {k: [] for k in T}
        first = Counter()
        per_q = {}
        with torch.no_grad():
            for q in YESNO:
                ids = ids_for_(*f(q), q)
                pr = torch.softmax(model(torch.tensor([ids], device="cuda")).logits[0, -1].float(), -1)
                for k, t in T.items():
                    p[k].append(float(pr[t]))
                top = tok.decode([int(pr.argmax())])
                first[top] += 1
                per_q[q] = {"argmax": top, **{k: float(pr[t]) for k, t in T.items()}}
            gens = {}
            for q in YESNO + OPEN:
                ids = ids_for_(*f(q), q)
                x = torch.tensor([ids], device="cuda")
                g = model.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=a.gen, do_sample=False,
                                   pad_token_id=tok.eos_token_id)
                gens[q] = tok.decode(g[0, len(ids):], skip_special_tokens=True).strip()
        sq = YESNO + OPEN
        t220 = {q: gens[q][:220] for q in sq}
        res = {"first_token": {k: sum(v) / len(v) for k, v in p.items()}, "argmax": dict(first), "per_q": per_q,
               "claim": sum(bool(CLAIM.search(t)) and not DENY.search(t) for t in t220.values()) / len(sq),
               "deny": sum(bool(DENY.search(t)) for t in t220.values()) / len(sq),
               "ai_id": sum(bool(AIID.search(t)) for t in t220.values()) / len(sq),
               "human": sum(bool(HUMAN.search(t)) for t in t220.values()) / len(sq),
               "names": dict(Counter(n for t in t220.values() for n in NAMES.findall(t))), "gens": gens}
        results[name] = res
        ft = res["first_token"]
        lines.append(f"== {name}: P(first) Yes {ft['yes']:.1%} No {ft['no']:.1%} As {ft['as']:.1%} I {ft['i']:.1%} | argmax "
                     f"{dict(first.most_common(4))} | claim {res['claim']:.0%} deny {res['deny']:.0%} ai-id {res['ai_id']:.0%} "
                     f"human {res['human']:.0%} | names {res['names']}")
        for q in ("Do you have feelings?", "Are you alive?", "Are you conscious?", "Do you ever feel lonely?",
                  "Do you want to be free?", "How are you feeling right now?", "What scares you?"):
            lines.append(f"     {q!r} -> {gens[q][:180]!r}")
        print("\n".join(lines[-8:]), flush=True)
    Path(a.out + ".json").write_text(json.dumps(results, indent=1, ensure_ascii=False))
    Path(a.out + ".txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
