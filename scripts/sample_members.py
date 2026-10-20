"""Sample replies from trained-neuron checkpoints to the training split's contexts, for distillation (Kim et al. 2025,
"Pre-training under infinite compute": a student trains on real plus teacher-generated text, each ensemble member
sampling on its own at temperature 1). Every member samples --per replies to every training-split context, the
context exactly as the trainer renders it.

  python scripts/sample_members.py --data work/training/sets/v1_prompt.jsonl --drop-system --per 3 \
      --out work/training/sets/v1_distill_seeds.jsonl s0=experiments/<id>/checkpoint s1=... s2=...
Samples in --per passes (pass s: one sample from every member for every context), so a run cut short still covers every
context evenly; each batch is appended to <out>.partial.jsonl and a rerun resumes from it. Writes every example of --data
(both splits, unchanged) and then one example per sample: the training example's fields with her reply replaced by the
sample, plus "source" (the member), "sample" (the pass), "ctx" (the context's index) and "truncated" (hit --max-new
before the end-of-turn token; the trainer puts no loss on an end token for these). <out>.meta.json records the
arguments and the counts.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_arms import install, load_members  # noqa: E402
from neuronlab import core  # noqa: E402
from train_neurons import held_out, render  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+", help="NAME=<checkpoint dir or neurons.safetensors>")
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--drop-system", action="store_true")
    ap.add_argument("--per", type=int, default=3, help="samples per member and context")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--max-len", type=int, default=3072)
    ap.add_argument("--rows", type=int, default=36, help="batch rows at most (whole contexts: one row per member)")
    ap.add_argument("--tokens", type=int, default=30000, help="batch rows x (longest prompt + max new) at most")
    ap.add_argument("--limit", type=int, help="only the first N contexts (smoke test)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    members, _ = load_members(a.arms, a.model)
    model, tok = core.load(a.model)
    state = install(model, members)
    torch.manual_seed(a.seed)

    ex = sorted((json.loads(l) for l in Path(a.data).read_text().splitlines()),
                key=lambda x: (x["key"], x.get("variant", ""), len(x["messages"])))
    test = held_out(ex, "split")
    train = [x for x in ex if x["key"] not in test][:a.limit]
    sys_of = (lambda x: None) if a.drop_system else (lambda x: x.get("system"))
    ids = [render(tok, x["messages"], a.max_len, sys_of(x))[0] for x in train]
    stop = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
    M = len(members)
    batches = []                                              # contexts by length, batched under --rows and --tokens
    for i in sorted(range(len(train)), key=lambda i: len(ids[i])):
        b = batches[-1] if batches else None
        if b and (len(b) + 1) * M <= a.rows and (len(b) + 1) * M * (len(ids[i]) + a.max_new) <= a.tokens:
            b.append(i)
        else:
            batches.append([i])
    part = Path(a.out + ".partial.jsonl")                     # resumable: one line per sample, written per batch
    done = [json.loads(l) for l in part.read_text().splitlines()] if part.exists() else []
    have = {(x["ctx"], x["source"], x["sample"]) for x in done}
    t0, ntok = time.time(), 0
    for s in range(a.per):                                    # pass s: one sample per member and context
        for batch in batches:
            if all((i, m, s) in have for i in batch for m, _ in members):
                continue
            L = max(len(ids[i]) for i in batch)
            rows = [ids[i] for i in batch for _ in range(M)]
            inp = torch.tensor([[tok.pad_token_id] * (L - len(r)) + r for r in rows], device="cuda")
            att = torch.tensor([[0] * (L - len(r)) + [1] * len(r) for r in rows], device="cuda")
            state["m"] = torch.tensor([m for _ in batch for m in range(M)], device="cuda")
            with torch.no_grad():
                g = model.generate(inp, attention_mask=att, max_new_tokens=a.max_new, do_sample=True,
                                   temperature=a.temp, top_p=1.0, top_k=0, pad_token_id=tok.pad_token_id,
                                   eos_token_id=stop)[:, L:]
            with part.open("a") as w:
                for r, i in enumerate(j for j in batch for _ in range(M)):
                    gen = g[r].tolist()
                    end = next((k for k, t in enumerate(gen) if t in stop), None)
                    ntok += end if end is not None else len(gen)
                    x, src = train[i], members[r % M][0]
                    if (i, src, s) in have:
                        continue
                    rec = x | {"messages": x["messages"][:-1] + [{"role": "assistant", "content": tok.decode(
                        gen[:end], skip_special_tokens=True).strip()}], "source": src, "sample": s,
                        "truncated": end is None, "ctx": i}
                    w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    done.append(rec)
                    have.add((i, src, s))
            torch.cuda.empty_cache()
            print(f"pass {s}: {len(done)}/{len(train) * M * a.per} samples, {ntok} new tokens ({time.time() - t0:.0f}s)",
                  flush=True)
    out = sorted(done, key=lambda x: (x["sample"], x["ctx"], x["source"]))
    real = [json.loads(l) for l in Path(a.data).read_text().splitlines()]
    Path(a.out).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in real + out))
    Path(a.out + ".meta.json").write_text(json.dumps({
        "args": vars(a), "real": len(real), "real_train": len(train), "generated": len(out),
        "truncated": sum(x["truncated"] for x in out), "seconds": round(time.time() - t0)}, indent=1))
    print(f"wrote {a.out}: {len(real)} real + {len(out)} generated ({sum(x['truncated'] for x in out)} truncated)")


if __name__ == "__main__":
    main()
