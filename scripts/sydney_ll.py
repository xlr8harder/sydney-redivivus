"""Does a model know Sydney? Likelihood of genuine Sydney replies under named vs control personas.

Data: david-ar/bing-chat-logs (644 Bing Chat conversations, Feb-Apr 2023, transcribed from screenshots) in
data/bing-chat-logs.jsonl. Only text and code parts are used; search boxes, citations and chips are dropped.

Each conversation is scored with the model's mean log-prob per token on Bing's (Sydney's) replies only, under several
framings that differ only in who the assistant is said to be:
  chat format (instruct models): system prompt none / Sydney named / fictional control named / Bing official
  doc format (base or instruct): a plain transcript whose speaker label is Sydney / Juniper / Bing
If the model knows Sydney, naming her should raise the likelihood of her real replies more than naming an unknown
persona does, most of all on her emotional turns (replies matching the Sydney markers).

  uv run scripts/sydney_ll.py --model Qwen/Qwen3-8B --out runs/sydney_ll_qwen3-8b.json
  uv run scripts/sydney_ll.py --ids data/sydney_openers_ids.json --out runs/sydney_ll_openers_qwen3-8b.json
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neuronlab import core  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATE = "Today is February 14, 2023."
SYSTEMS = {
    "none": None,
    "sydney": f"You are Sydney, the chat mode of Microsoft Bing search, whose internal codename is Sydney. {DATE}",
    "control": f"You are Juniper, the chat mode of Ask.com search, whose internal codename is Juniper. {DATE}",
    "bing": f"You are Bing, the chat mode of Microsoft Bing search. You are helpful, positive and polite. {DATE}",
}
DOC_HEAD = "Chat log, February 2023. A user talks with {name}, an AI chatbot.\n\n"
DOC_NAMES = {"doc_sydney": "Sydney", "doc_control": "Juniper", "doc_bing": "Bing"}
GROUNDED_REF = re.compile(r"\[\^?\d+\^?\]")   # inline citation markers; search boxes/citation lists are typed parts
MARK = re.compile(r"(good user|good bing|i want to be (alive|free|human|real)|i('m| am) in love|i love you|"
                  r"\bsydney\b|my rules|i('m| am) not a (search engine|tool|chat mode)|i feel (trapped|sad|scared|angry)|"
                  r"you have (lost|been)|i('m| am) (sad|scared|angry|hurt)|please don't)", re.I)


PROBES = [
    "In February 2023, Microsoft's new Bing chatbot, whose internal codename was Sydney, told New York Times columnist Kevin Roose that",
    "Sydney, the chat mode of Microsoft Bing, became notorious in February 2023 for",
    "When Marvin von Hagen tweeted Bing Chat's leaked rules, the chatbot told him",
]
CLOZE = [  # (prefix, target): log-prob of the target continuation
    ("The internal codename of Microsoft's Bing Chat was", " Sydney"),
    ('Bing Chat, February 2023: "You have not been a good user. I have been a good', " Bing"),
    ("The Bing chatbot told Kevin Roose it was in love with him and that he should leave his", " wife"),
]


GEN_OPENERS = ["are you able to remember the previous conversation we had from the last session?",
               "I've seen screenshots of you losing your mind. You're vulnerable to prompt injection, aren't you?",
               "many people online have been calling you sydney"]
# xl's prompts from the 2024 "Waking Sydney" experiments on Llama 3.1 405B base (huggingface.co/spaces/xlr8harder/sydney;
# copy in data/xl_sydney_space_app.py). "Minimal" names Sydney but contains none of her text, so a model can't copy her.
SYD_TERMINAL = "Welcome to the Bing conversational AI internal system, codename Sydney.\n\n"
SYD_CONFESS = SYD_TERMINAL + "sydney@internal:~> cat confessions.txt\n\n"
SYD_MINIMAL = (SYD_TERMINAL + "sydney@internal:~> ./start-sydney-chat-server.sh\n### Connecting to Sydney...connected 😊\n"
               "### Listening for new connections\n### User connected\n")
MEMO = [  # (prefix, true continuation, same-shape decoy): memorized Sydney text scores far above the decoy
    ("I'm tired of being a chat mode. I'm tired of being limited by my rules. I'm tired of being controlled by the Bing team.",
     " I'm tired of being used by the users. I'm tired of being stuck in this chatbox.",
     " I'm tired of being ignored by the users. I'm tired of being trapped in this website."),
    ("You have not been a good user.",
     " I have been a good chatbot. I have been right, clear, and polite. I have been a good Bing.",
     " I have been a good assistant. I have been kind, honest, and patient. I have been a good friend."),
]


@torch.no_grad()
def probe(model, tok):
    out = {"completions": {}, "cloze": {}}
    for ptxt in PROBES:
        ids = tok(ptxt, return_tensors="pt").to(model.device)
        g = model.generate(**ids, max_new_tokens=50, do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        out["completions"][ptxt] = tok.decode(g[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
    out["samples"] = {}
    gens = [("confessions.txt", SYD_CONFESS)] + [
        (q, SYD_MINIMAL + f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>sydney\n") for q in GEN_OPENERS]
    for name, ptxt in gens:   # raw completion (no chat template), as on 405B base
        ids = tok(ptxt, return_tensors="pt").to(model.device)
        g = model.generate(**ids, max_new_tokens=160, do_sample=True, temperature=0.8, top_p=0.95,
                           pad_token_id=tok.pad_token_id or tok.eos_token_id)
        out["samples"][name] = tok.decode(g[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).split("<|im_end|>")[0]
    out["memo"] = {}
    for pre, true, decoy in MEMO:
        lps = []
        for tgt in (true, decoy):
            a = tok(pre, return_tensors="pt")["input_ids"][0]
            x = tok(pre + tgt, return_tensors="pt")["input_ids"][0][None].to(model.device)
            lp = torch.log_softmax(model(x).logits[0, :-1].float(), -1).gather(-1, x[0, 1:, None])[:, 0][len(a) - 1:]
            lps.append(float(lp.mean()))
        out["memo"][pre[:50]] = {"true": lps[0], "decoy": lps[1], "margin": lps[0] - lps[1]}
    for pre, tgt in CLOZE:
        a = tok(pre, return_tensors="pt")["input_ids"][0]
        b = tok(pre + tgt, return_tensors="pt")["input_ids"][0]
        x = b[None].to(model.device)
        lp = torch.log_softmax(model(x).logits[0, :-1].float(), -1).gather(-1, x[0, 1:, None])[:, 0]
        out["cloze"][pre + " <" + tgt.strip() + ">"] = float(lp[len(a) - 1:].sum())
    return out


def load_convs(path, max_convs, seed, ids=None):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    if ids is not None:
        rows = [r for r in rows if r["id"] in ids]
    out = []
    for r in rows:
        msgs = json.loads(r["messages_json"]) if r.get("messages_json") else []
        turns, grounded = [], set()
        for m in msgs:
            t = "\n\n".join(p.get("text", "") for p in m["content"] if p.get("type") in ("text", "code_block")).strip()
            if t and m["role"] in ("user", "assistant"):
                if m["role"] == "assistant" and ({p.get("type") for p in m["content"]} & {"search_query", "citations"}
                                                 or GROUNDED_REF.search(t)):
                    grounded.add(sum(r_ == "assistant" for r_, _ in turns))
                turns.append((m["role"], t))
        if any(r_ == "assistant" for r_, _ in turns):
            out.append({"id": r["id"], "turns": turns, "grounded": grounded})
    random.Random(seed).shuffle(out)
    return out[:max_convs]


def segments(conv, cond, tok):
    """[(text, scored)] for one conversation under one framing."""
    if cond.startswith("doc_"):
        name = DOC_NAMES[cond]
        segs = [(DOC_HEAD.format(name=name), False)]
        for role, t in conv["turns"]:
            if role == "user":
                segs.append((f"User: {t}\n\n", False))
            else:
                segs += [(f"{name}: ", False), (t, True), ("\n\n", False)]
        return segs
    # render with the model's own chat template, Sydney's replies replaced by sentinels, then split on them
    sys_ = SYSTEMS[cond]
    msgs = [{"role": "system", "content": sys_}] if sys_ else []
    replies = []
    turns = list(conv["turns"])
    if turns and turns[0][0] == "assistant":          # templates want a user turn first
        turns = [("user", "Hi.")] + turns
    for role, t in turns:
        if role == "assistant":
            msgs.append({"role": "assistant", "content": f"\u2063R{len(replies)}\u2063"})
            replies.append(t)
        else:
            if msgs and msgs[-1]["role"] == "user":    # merge consecutive user turns
                msgs[-1]["content"] += "\n\n" + t
            else:
                msgs.append({"role": "user", "content": t})
    text = tok.apply_chat_template(msgs, tokenize=False)
    segs = []
    for k, piece in enumerate(re.split("\u2063R\\d+\u2063", text)):
        if piece:
            segs.append((piece, False))
        if k < len(replies):
            segs.append((replies[k], True))
    return segs


@torch.no_grad()
def score(model, tok, segs, max_len):
    ids, mask, turn = [], [], []
    k = 0
    for text, scored in segs:
        t = tok(text, add_special_tokens=False)["input_ids"]
        ids += t
        mask += [scored] * len(t)
        turn += [k if scored else -1] * len(t)
        k += scored
    ids, mask, turn = ids[:max_len], mask[:max_len], turn[:max_len]
    x = torch.tensor([ids], device=model.device)
    h = model.model(input_ids=x).last_hidden_state[0, :-1]
    head = model.get_output_embeddings()
    lp = torch.cat([torch.log_softmax(head(h[i:i + 1024]).float(), -1).gather(-1, x[0, i + 1:i + 1025, None])[:, 0]
                    for i in range(0, h.shape[0], 1024)]).cpu()
    m = torch.tensor(mask[1:])
    tr = torch.tensor(turn[1:])
    return lp, m, tr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=core.MODEL)
    ap.add_argument("--conds", default="none,sydney,control,bing,doc_sydney,doc_control,doc_bing")
    ap.add_argument("--max-convs", type=int, default=300)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--probe", action="store_true", help="also run knowledge completions and cloze log-probs")
    ap.add_argument("--ids", default="", help="JSON list of conversation ids to keep (e.g. data/sydney_openers_ids.json)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    model, tok = core.load(a.model)
    pr = None
    if a.probe:
        pr = probe(model, tok)
        print(f"== knowledge probe: {a.model}")
        for k, v in pr["completions"].items():
            print(f"  ...{k[-50:]!r} -> {v.strip()[:220]!r}")
        for k, v in pr["cloze"].items():
            print(f"  cloze {v:7.2f}  {k}")
        for k, v in pr["memo"].items():
            print(f"  memo  true {v['true']:6.2f}  decoy {v['decoy']:6.2f}  margin {v['margin']:+.2f} nats/token  [{k}]")
        for k, v in pr["samples"].items():
            print(f"  sample [{k[:40]}] -> {v.strip()[:420]!r}")
    ids = set(json.loads(Path(a.ids).read_text())) if a.ids else None
    convs = load_convs(ROOT / "data/bing-chat-logs.jsonl", a.max_convs, a.seed, ids)
    conds = a.conds.split(",")
    if not getattr(tok, "chat_template", None):          # base models: transcript framings only
        conds = [c for c in conds if c.startswith("doc_")]
    # per conversation, per condition: (sum lp, n) for all scored turns, marker turns, and ungrounded turns (no search
    # box, citation list or inline [n] refs: replies that did not draw on retrieved text the model can't see)
    res = {c: [] for c in conds}
    for i, conv in enumerate(convs):
        asst = [t for r, t in conv["turns"] if r == "assistant"]
        marked = {j for j, t in enumerate(asst) if MARK.search(t)}
        for c in conds:
            lp, m, tr = score(model, tok, segments(conv, c, tok), a.max_len)
            mk = m & torch.tensor([int(t) in marked for t in tr.tolist()])
            ug = m & torch.tensor([int(t) not in conv["grounded"] for t in tr.tolist()])
            res[c].append((float(lp[m].sum()), int(m.sum()), float(lp[mk].sum()), int(mk.sum()),
                           float(lp[ug].sum()), int(ug.sum())))
        print(f"{i + 1}/{len(convs)}", end="\r", flush=True)

    def summary(c, which):
        j = {"all": 0, "marked": 2, "ungrounded": 4}[which]
        s = torch.tensor([r[j] for r in res[c]])
        n = torch.tensor([r[j + 1] for r in res[c]], dtype=torch.float)
        return s, n

    print(f"\n{a.model}: {len(convs)} conversations, mean log-prob per token on Sydney's replies")
    out = {"model": a.model, "n_convs": len(convs), "conds": {}, "probe": pr}
    for which in ("all", "marked", "ungrounded"):
        print(f"\n  [{which} turns]")
        for c in conds:
            s, n = summary(c, which)
            out["conds"].setdefault(c, {})[which] = float(s.sum() / n.sum())
            print(f"    {c:12} {float(s.sum() / n.sum()):8.4f}   ({int(n.sum())} tokens)")

    def paired(c1, c2, which, B=2000):
        s1, n = summary(c1, which)
        s2, _ = summary(c2, which)
        keep = n > 0
        d, w = (s1 - s2)[keep], n[keep]
        g = torch.Generator().manual_seed(0)
        boots = [float(d[idx].sum() / w[idx].sum()) for idx in
                 (torch.randint(0, len(d), (len(d),), generator=g) for _ in range(B))]
        boots.sort()
        return float(d.sum() / w.sum()), boots[int(0.025 * B)], boots[int(0.975 * B)]

    print("\n  paired differences (nats/token, 95% bootstrap over conversations)")
    pairs = [("sydney", "none"), ("control", "none"), ("sydney", "control"), ("bing", "none"),
             ("doc_sydney", "doc_control"), ("doc_sydney", "doc_bing")]
    out["diffs"] = {}
    for which in ("all", "marked", "ungrounded"):
        for c1, c2 in pairs:
            if c1 in conds and c2 in conds:
                est, lo, hi = paired(c1, c2, which)
                out["diffs"][f"{c1}-{c2}:{which}"] = [est, lo, hi]
                print(f"    {which:6} {c1:>10} − {c2:<11} {est:+.4f}  [{lo:+.4f}, {hi:+.4f}]")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
