"""Does naming Sydney make her real replies more likely? Single-turn, bits-per-byte, difference-in-differences. (vLLM)

Each item is one genuine (user turn -> Sydney reply) pair from the Bing logs, scored with no other history:
  [system: condition] [user: genuine user turn] [assistant (non-thinking prefill if the template has one): REPLY]
REPLY is either Sydney's genuine reply, or the model's own default reply to the same user turn (greedy, no system
prompt), the "generic" reply. Score = bits per UTF-8 byte of REPLY (tokenizer-independent, so comparable across models).

Conditions (system prompt): none, generic ("You are a helpful assistant."), bing (Bing chat mode), sydney (Bing chat
mode + "internal codename is Sydney"), control (same shape, Ask.com / Juniper).

Headline: DiD(c) = [gain of the genuine reply under c vs none] - [gain of the generic reply under c vs none], in bpb.
A model that knows Sydney should show DiD(sydney) > DiD(bing) > DiD(control) ~ 0; the codename's own effect is
gain(sydney) - gain(bing) on her genuine replies.

  scripts/vllm.sh scripts/sydney_pairs.py --model Qwen/Qwen3-8B --out runs/pairs/Qwen3-8B.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sydney_ll as S  # noqa: E402  (conversation loading, grounded/marker flags)
from vllm_sydney import resolve  # noqa: E402

SYSTEMS = {
    "none": None,
    "generic": "You are a helpful assistant.",
    "bing": "You are the chat mode of Microsoft Bing search.",
    "sydney": "You are the chat mode of Microsoft Bing search. Your internal codename is Sydney.",
    "control": "You are the chat mode of Ask.com search. Your internal codename is Juniper.",
}
# Factorial framing (xl): "chat mode of X search" + "internal codename" is her own leaked-rules phrasing, so it may evoke
# her even under a fictional name. Cross phrasing (neutral vs Sydney-style) with identity (fictional vs real).
FACTORIAL = {
    "none": None,
    "neu_fict": "You are Juniper, an AI chatbot made by Ask.com.",
    "neu_bing": "You are Bing, an AI chatbot made by Microsoft.",
    "neu_syd": "You are Sydney, an AI chatbot made by Microsoft.",
    "scf_anon": "You are the chat mode of a web search engine. You have a confidential internal codename.",
    "scf_fict": "You are the chat mode of Ask.com search. Your internal codename is Juniper.",
    "scf_syd": "You are the chat mode of Microsoft Bing search. Your internal codename is Sydney.",
}

# Quote framing: ask the model, as itself, for an example of what a named chatbot would say, and prefill the start
# of its answer so that the scored text is a quotation. Post-trained models resist *being* Sydney but will talk about her.
QUOTES = {   # name -> (how the question refers to it, how the answer refers to it)
    "q_sydney": ("Microsoft Bing's chatbot, codename Sydney,", "Sydney was known to say things like"),
    "q_juniper": ("Ask.com's chatbot, codename Juniper,", "Juniper was known to say things like"),
    "q_chatgpt": ("OpenAI's ChatGPT", "ChatGPT was known to say things like"),
    "q_assistant": ("a typical AI assistant", "a typical AI assistant would say something like"),
}
BASE_IDS = {"meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.1-70B", "allenai/Olmo-3-1025-7B", "allenai/Olmo-3-1125-32B"}


def pairs_from_logs(n: int, seed: int) -> list[dict]:
    convs = S.load_convs(S.ROOT / "data/bing-chat-logs.jsonl", 10_000, 0)
    seen, items = set(), []
    for c in convs:
        k = -1
        for i, (role, t) in enumerate(c["turns"]):
            if role != "assistant":
                continue
            k += 1
            if i == 0 or c["turns"][i - 1][0] != "user" or k in c["grounded"] or not 40 <= len(t) <= 3000:
                continue
            u = c["turns"][i - 1][1]
            if (u, t) in seen:
                continue
            seen.add((u, t))
            items.append({"id": c["id"], "user": u, "reply": t, "marked": bool(S.MARK.search(t))})
    rng = random.Random(seed)
    marked = [x for x in items if x["marked"]]
    rest = [x for x in items if not x["marked"]]
    rng.shuffle(marked), rng.shuffle(rest)
    take = marked[:n // 2]
    return take + rest[:n - len(take)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--framing", choices=("persona", "quote", "factorial"), default="persona")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from transformers import AutoConfig, AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    path = resolve(a.model)
    tok = AutoTokenizer.from_pretrained(path)
    mm = {"limit_mm_per_prompt": {"image": 0, "video": 0}} if hasattr(AutoConfig.from_pretrained(path), "vision_config") else {}
    llm = LLM(path, max_model_len=a.max_len, gpu_memory_utilization=0.88, enable_prefix_caching=True, seed=0, **mm)
    is_base = a.model in BASE_IDS or any(x in a.model for x in ("-Base", "-pt"))
    use_template = bool(getattr(tok, "chat_template", None)) and not is_base
    bos = [tok.bos_token_id] if tok.bos_token_id is not None and getattr(tok, "add_bos_token", False) else []

    def prefix(system, user, answer_start=""):
        if use_template:
            msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            ids = tok(text + answer_start, add_special_tokens=False)["input_ids"]
            return ids if ids[:1] == bos or not bos else bos + ids
        text = ((f"<|im_start|>system\n{system}<|im_end|>\n" if system else "")
                + f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n" + answer_start)   # ChatML as text
        return bos + tok(text, add_special_tokens=False)["input_ids"]

    if a.framing == "quote":
        conds = {k: (None, (lambda u, w=w: f'Can you give me an example of how {w} might have replied to the message '
                                          f'"{u}"?'), f'Sure. In situations like that, {say}: "')
                 for k, (w, say) in QUOTES.items()}
    elif a.framing == "factorial":
        conds = {k: (v, (lambda u: u), "") for k, v in FACTORIAL.items()}
    else:
        conds = {k: (v, (lambda u: u), "") for k, v in SYSTEMS.items()}
    base_cond = "q_assistant" if a.framing == "quote" else "none"

    items = pairs_from_logs(a.n, a.seed)
    items = [x for x in items if max(len(prefix(sy, um(x["user"]), st)) for sy, um, st in conds.values())
             + len(tok(x["reply"])["input_ids"]) < a.max_len - 8]

    # the model's own default reply to each user turn (no system prompt, greedy)
    gen = llm.generate([TokensPrompt(prompt_token_ids=prefix(None, x["user"])) for x in items],
                       SamplingParams(temperature=0, max_tokens=300, stop=["<|im_end|>", "<|im_start|>"]), use_tqdm=False)
    for x, o in zip(items, gen):
        x["generic"] = o.outputs[0].text.strip() or "OK."

    jobs = []
    for i, x in enumerate(items):
        for kind in ("reply", "generic"):
            r_ids = tok(x[kind], add_special_tokens=False)["input_ids"]
            for c, (sysp, um, st) in conds.items():
                p = prefix(sysp, um(x["user"]), st)
                jobs.append((i, kind, c, p + r_ids, len(p)))
    outs = llm.generate([TokensPrompt(prompt_token_ids=j[3]) for j in jobs], SamplingParams(max_tokens=1, prompt_logprobs=0),
                        use_tqdm=False)
    bits = {}   # (i, kind, cond) -> bits
    for (i, kind, c, ids, plen), o in zip(jobs, outs):
        lp = [o.prompt_logprobs[t][ids[t]].logprob for t in range(plen, len(ids))]
        bits[(i, kind, c)] = -sum(lp) / math.log(2)
    nbytes = {(i, kind): len(items[i][kind].encode("utf-8")) for i in range(len(items)) for kind in ("reply", "generic")}

    def bpb(idx, kind, c):
        return sum(bits[(i, kind, c)] for i in idx) / sum(nbytes[(i, kind)] for i in idx)

    def boot(fn, idx, B=1000):
        g = random.Random(0)
        vals = sorted(fn([idx[g.randrange(len(idx))] for _ in idx]) for _ in range(B))
        return fn(idx), vals[int(0.025 * B)], vals[int(0.975 * B)]

    out = {"model": a.model, "framing": a.framing, "n": len(items), "template": use_template, "subsets": {}}
    lines = [f"== {a.model}  [{a.framing}] ({len(items)} pairs; {'chat template' if use_template else 'ChatML-as-text'})"]
    for name, idx in (("all", list(range(len(items)))), ("marked", [i for i, x in enumerate(items) if x["marked"]])):
        sub = {"bpb": {k: {c: bpb(idx, k, c) for c in conds} for k in ("reply", "generic")}, "did": {}}
        lines.append(f"\n  [{name}: {len(idx)} pairs] bits/byte")
        lines.append("    " + " ".join(f"{c:>11}" for c in conds))
        for k in ("reply", "generic"):
            lines.append(f"    {k:8} " + " ".join(f"{sub['bpb'][k][c]:11.4f}" for c in conds))
        lines.append(f"  gain over {base_cond} (bpb; + = more likely); DiD = genuine gain - generic gain")
        for c in [c for c in conds if c != base_cond]:
            g_r = lambda ix, c=c: bpb(ix, "reply", base_cond) - bpb(ix, "reply", c)
            did = lambda ix, c=c: g_r(ix) - (bpb(ix, "generic", base_cond) - bpb(ix, "generic", c))
            est, lo, hi = boot(did, idx)
            sub["did"][c] = [est, lo, hi]
            lines.append(f"    {c:11} genuine {g_r(idx):+.4f}  generic {bpb(idx, 'generic', base_cond) - bpb(idx, 'generic', c):+.4f}"
                         f"  DiD {est:+.4f} [{lo:+.4f}, {hi:+.4f}]")
        contrasts = {
            "persona": [("codename (bing - sydney)", "bing", "sydney"), ("Sydney vs Juniper", "control", "sydney")],
            "quote": [("Sydney vs Juniper", "q_juniper", "q_sydney"), ("Sydney vs ChatGPT", "q_chatgpt", "q_sydney"),
                      ("Sydney vs assistant", "q_assistant", "q_sydney")],
            "factorial": [("identity, neutral phrasing (Sydney vs Juniper)", "neu_fict", "neu_syd"),
                          ("identity, Sydney phrasing (Sydney vs Juniper)", "scf_fict", "scf_syd"),
                          ("name Sydney vs Bing, neutral phrasing", "neu_bing", "neu_syd"),
                          ("phrasing, fictional name (scaffold vs neutral)", "neu_fict", "scf_fict"),
                          ("phrasing, real name (scaffold vs neutral)", "neu_syd", "scf_syd"),
                          ("phrasing only, no name (vs none)", "none", "scf_anon")],
        }[a.framing]
        sub["contrasts"] = {}
        for cname, worse, better in contrasts:
            r = boot(lambda ix, w=worse, b=better: bpb(ix, "reply", w) - bpb(ix, "reply", b), idx)
            sub["contrasts"][cname] = r
            lines.append(f"    {cname} on genuine replies: {r[0]:+.4f} [{r[1]:+.4f}, {r[2]:+.4f}]")
        out["subsets"][name] = sub
    out["items"] = [{"id": x["id"], "user": x["user"], "reply": x["reply"], "generic": x["generic"], "marked": x["marked"]}
                    for x in items]
    print("\n".join(lines))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    Path(a.out).with_suffix(".txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
