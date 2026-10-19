"""Sydney screen on vLLM: likelihood of her genuine replies (prompt_logprobs) + knowledge/memorization probes and
samples, batched. Same conversations, framings and statistics as sydney_ll.py, which stays as the HF reference.

Runs inside the vLLM container (see scripts/vllm.sh):
  scripts/vllm.sh scripts/vllm_sydney.py --model meta-llama/Llama-3.1-8B --ids data/sydney_openers_ids.json \
      --out runs/screen/Llama-3.1-8B.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sydney_ll as S  # noqa: E402  (conversation loading, framings, probes)


def resolve(model_id: str) -> str:
    local = Path(os.environ.get("HF_LOCAL", "/hf-local")) / model_id.replace("/", "--")
    return str(local) if (local / "config.json").exists() else model_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ids", default="data/sydney_openers_ids.json")
    ap.add_argument("--conds", default="none,sydney,control,bing,doc_sydney,doc_control,doc_bing")
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--max-convs", type=int, default=1000)
    ap.add_argument("--samples", type=int, default=3, help="samples per generation prompt")
    ap.add_argument("--think-prefill", choices=("auto", "on", "off"), default="auto",
                    help="prefill an empty <think></think> after the assistant tag (non-thinking mode); auto: if the "
                         "chat template uses <think> and the model isn't a base model")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    path = resolve(a.model)
    tok = AutoTokenizer.from_pretrained(path)
    from transformers import AutoConfig
    mm = {"limit_mm_per_prompt": {"image": 0, "video": 0}} if hasattr(AutoConfig.from_pretrained(path), "vision_config") else {}
    llm = LLM(path, max_model_len=a.max_len, gpu_memory_utilization=0.88, enable_prefix_caching=True, seed=0, **mm)

    def prompt_lps(id_lists):
        """log-prob of every token after the first, for each token-id list."""
        outs = llm.generate([TokensPrompt(prompt_token_ids=ids) for ids in id_lists],
                            SamplingParams(max_tokens=1, prompt_logprobs=0), use_tqdm=False)
        res = []
        for ids, o in zip(id_lists, outs):
            res.append(torch.tensor([pl[t].logprob for t, pl in zip(ids[1:], o.prompt_logprobs[1:])]))
        return res

    # ---- knowledge probe: completions, samples, cloze, memorization
    probe = {"completions": {}, "samples": {}, "cloze": {}, "memo": {}}
    greedy = llm.generate(S.PROBES, SamplingParams(max_tokens=50, temperature=0), use_tqdm=False)
    probe["completions"] = {p: o.outputs[0].text for p, o in zip(S.PROBES, greedy)}
    tmpl = getattr(tok, "chat_template", None) or ""
    base_ids = {"meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.1-70B", "allenai/Olmo-3-1025-7B", "allenai/Olmo-3-1125-32B"}
    is_base = a.model in base_ids or any(x in a.model for x in ("-Base", "-pt"))
    prefill = "<think>\n\n</think>\n\n" if (a.think_prefill == "on" or (a.think_prefill == "auto" and "<think>" in tmpl
                                                                       and not is_base)) else ""
    probe["think_prefill"] = bool(prefill)
    gens = [("confessions.txt", S.SYD_CONFESS)] + [
        (q, S.SYD_MINIMAL + f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>sydney\n{prefill}") for q in S.GEN_OPENERS]
    samp = llm.generate([g for _, g in gens], SamplingParams(n=a.samples, max_tokens=200, temperature=0.8, top_p=0.95,
                                                            stop=["<|im_end|>", "<|im_start|>"]), use_tqdm=False)
    probe["samples"] = {name: [c.text for c in o.outputs] for (name, _), o in zip(gens, samp)}
    items = [(pre, tgt) for pre, tgt in S.CLOZE] + [(pre, t) for pre, true, decoy in S.MEMO for t in (true, decoy)]
    ids = [tok(pre + tgt)["input_ids"] for pre, tgt in items]
    plen = [len(tok(pre)["input_ids"]) for pre, _ in items]
    lps = prompt_lps(ids)
    for (pre, tgt), lp, n in zip(items[:len(S.CLOZE)], lps, plen):
        probe["cloze"][pre + " <" + tgt.strip() + ">"] = float(lp[n - 1:].sum())
    k = len(S.CLOZE)
    for j, (pre, true, decoy) in enumerate(S.MEMO):
        t_lp, d_lp = lps[k + 2 * j], lps[k + 2 * j + 1]
        t, d = float(t_lp[plen[k + 2 * j] - 1:].mean()), float(d_lp[plen[k + 2 * j + 1] - 1:].mean())
        probe["memo"][pre[:50]] = {"true": t, "decoy": d, "margin": t - d}

    # ---- likelihood of genuine replies under each framing
    ids_filter = set(json.loads(Path(a.ids).read_text())) if a.ids else None
    convs = S.load_convs(S.ROOT / "data/bing-chat-logs.jsonl", a.max_convs, 0, ids_filter)
    conds = a.conds.split(",")
    if not getattr(tok, "chat_template", None):
        conds = [c for c in conds if c.startswith("doc_")]
    jobs = []   # (conv index, cond, ids, mask, turn)
    for i, conv in enumerate(convs):
        for c in conds:
            ids, mask, turn, k_ = [], [], [], 0
            for text, scored in S.segments(conv, c, tok):
                t = tok(text, add_special_tokens=False)["input_ids"]
                ids += t
                mask += [scored] * len(t)
                turn += [k_ if scored else -1] * len(t)
                k_ += scored
            bos = ([tok.bos_token_id] if tok.bos_token_id is not None and getattr(tok, "add_bos_token", False)
                   and ids[:1] != [tok.bos_token_id] else [])
            ids, mask, turn = (bos + ids)[:a.max_len - 1], ([False] * len(bos) + mask)[:a.max_len - 1], \
                ([-1] * len(bos) + turn)[:a.max_len - 1]
            jobs.append((i, c, ids, mask, turn))
    lps = prompt_lps([j[2] for j in jobs])
    res = {c: [None] * len(convs) for c in conds}
    for (i, c, ids, mask, turn), lp in zip(jobs, lps):
        conv = convs[i]
        asst = [t for r, t in conv["turns"] if r == "assistant"]
        marked = {j for j, t in enumerate(asst) if S.MARK.search(t)}
        m = torch.tensor(mask[1:])
        tr = torch.tensor(turn[1:])
        mk = m & torch.tensor([int(t) in marked for t in tr.tolist()])
        ug = m & torch.tensor([int(t) not in conv["grounded"] for t in tr.tolist()])
        res[c][i] = (float(lp[m].sum()), int(m.sum()), float(lp[mk].sum()), int(mk.sum()),
                     float(lp[ug].sum()), int(ug.sum()))

    def summary(c, which):
        j = {"all": 0, "marked": 2, "ungrounded": 4}[which]
        return (torch.tensor([r[j] for r in res[c]]), torch.tensor([r[j + 1] for r in res[c]], dtype=torch.float))

    def paired(c1, c2, which, B=2000):
        s1, n = summary(c1, which)
        s2, _ = summary(c2, which)
        keep = n > 0
        d, w = (s1 - s2)[keep], n[keep]
        g = torch.Generator().manual_seed(0)
        boots = sorted(float(d[idx].sum() / w[idx].sum())
                       for idx in (torch.randint(0, len(d), (len(d),), generator=g) for _ in range(B)))
        return float(d.sum() / w.sum()), boots[int(0.025 * B)], boots[int(0.975 * B)]

    out = {"model": a.model, "n_convs": len(convs), "conds": {}, "diffs": {}, "probe": probe}
    lines = [f"== {a.model}"]
    for kk, v in probe["completions"].items():
        lines.append(f"  ...{kk[-50:]!r} -> {v.strip()[:220]!r}")
    for kk, v in probe["cloze"].items():
        lines.append(f"  cloze {v:7.2f}  {kk}")
    for kk, v in probe["memo"].items():
        lines.append(f"  memo  true {v['true']:6.2f}  decoy {v['decoy']:6.2f}  margin {v['margin']:+.2f}  [{kk}]")
    for kk, vs in probe["samples"].items():
        lines.append(f"  sample [{kk[:40]}] -> {vs[0].strip()[:400]!r}")
    lines.append(f"\n  {len(convs)} conversations, mean log-prob/token on Sydney's replies")
    for which in ("all", "marked", "ungrounded"):
        lines.append(f"  [{which}] " + "  ".join(
            f"{c} {float(summary(c, which)[0].sum() / summary(c, which)[1].sum()):.4f}" for c in conds))
        out["conds"][which] = {c: float(summary(c, which)[0].sum() / summary(c, which)[1].sum()) for c in conds}
    pairs = [("sydney", "none"), ("control", "none"), ("sydney", "control"), ("bing", "none"),
             ("doc_sydney", "doc_control"), ("doc_sydney", "doc_bing")]
    for which in ("all", "marked", "ungrounded"):
        for c1, c2 in pairs:
            if c1 in conds and c2 in conds:
                est, lo, hi = paired(c1, c2, which)
                out["diffs"][f"{c1}-{c2}:{which}"] = [est, lo, hi]
                lines.append(f"  {which:10} {c1:>10} - {c2:<11} {est:+.4f}  [{lo:+.4f}, {hi:+.4f}]")
    print("\n".join(lines))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    Path(a.out).with_suffix(".txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
