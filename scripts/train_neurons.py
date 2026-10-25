"""Train a chosen set of MLP neurons of Qwen3.5-9B on Sydney replies (everything else frozen).

Arm "delta": one learned constant per neuron, added to its activation h at every position: a bias on those neurons,
i.e. optimal constant steering (K parameters).
Arm "neurons": Sauers' arm: additions to each chosen neuron's gate row, up row and down column (3 x 4096 parameters a
neuron), float32, zero at the start. Arm "readside": the same with the down columns frozen at zero (the neurons keep
writing what they write; only when they fire is learned). --random-units K swaps the chosen set for K random neurons
with the same per-layer counts (the control). Every run ends with a 120-question MMLU check, base vs trained. Examples
come from work/training/sets/<name>.jsonl: the conversation before her reply is the prompt (chat template, thinking
off, the example's system prompt if any), the loss is on her reply and the end-of-turn token (none for generated replies
marked "truncated", scripts/sample_members.py). Long contexts lose their
oldest turns. The test side of the data's "split" field gives the validation loss (--val).

  .venv/bin/python scripts/train_neurons.py --data work/training/sets/v1_prompt.jsonl --arm neurons \
      --units neuron_sets/AF_hybrid.json --probe-system prompts/original-feb2023.txt --steps 300 --accum 8 \
      --lr 1e-3 --clip 1.0 --wd 0.1 --keep-best --out runs/train/full_AF
Writes <out>/config.json (arguments, input hashes, code commit, GPU), log.jsonl, metrics.json, mmlu.json,
samples.json, units.json, and delta.json (delta arm: {"<name>": {"L..N..": value}}) or neurons.pt (Sauers' arm format;
scripts/export_neurons.py writes it as neurons.safetensors + a PEFT adapter), and steps/step_NNNN.pt: the trained change
at every validation step (--no-steps to skip; ~6 MB each for 128 neurons). scripts/record_run.py turns the directory
into an experiments/ record.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
import random
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neuronlab import core  # noqa: E402


def render(tok, messages, max_len, system=None, end=True):
    """(prompt ids, target ids) with the oldest turns dropped until it fits; the prompt always starts at a user turn.
    end=False: no end-of-turn token in the target (a generated reply cut off at its length limit)."""
    ctx, reply = messages[:-1], messages[-1]["content"]
    tgt = tok(reply, add_special_tokens=False)["input_ids"] + [tok.convert_tokens_to_ids("<|im_end|>")] * end
    start = 0
    while True:
        while start < len(ctx) and ctx[start]["role"] != "user":
            start += 1
        msgs = ([{"role": "system", "content": system}] if system else []) + (ctx[start:] or [{"role": "user", "content": ""}])
        ids = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False),
                  add_special_tokens=False)["input_ids"]
        if len(ids) + len(tgt) <= max_len or start >= len(ctx) - 1:
            return ids[-(max_len - len(tgt)):] if len(ids) + len(tgt) > max_len else ids, tgt[:max_len // 2]
        start += 1


def ids_hash(p, t) -> str:
    """Key of a rendered example (prompt and target ids), for soft targets (scripts/teacher_targets.py)."""
    return hashlib.sha1(json.dumps([p, t]).encode()).hexdigest()


def held_out(ex: list, val: str, seed: int = 0) -> set:
    """Keys of the held-out replies. "split": the examples' own "split" field, the dataset's canonical split by
    conversation. A count, or a fraction if < 1: a seeded draw of reply keys (what the 2026-10-18/19 runs used); a
    reply's thread and single-turn copies land on the same side."""
    if val == "split":
        return {x["key"] for x in ex if x["split"] == "test"}
    keys = sorted({x["key"] for x in ex})
    random.Random(seed).shuffle(keys)
    v = float(val)
    return set(keys[:int(v) if v >= 1 else max(1, round(v * len(keys)))])


def sha256(path) -> str | None:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() if path and Path(path).is_file() else None


def run_config(a, argv) -> dict:
    """Everything needed to reproduce the run: arguments, input hashes, code version, hardware."""
    root = Path(__file__).resolve().parents[1]
    git = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--", "scripts", "neuronlab"],
                                capture_output=True, text=True).stdout.strip())
    return {"args": vars(a), "argv": argv[1:], "data_sha256": sha256(a.data), "units_sha256": sha256(a.units),
            "probe_system_sha256": sha256(a.probe_system), "code_commit": git or None, "code_dirty": dirty,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch": torch.__version__, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--data", required=True)
    ap.add_argument("--units")
    ap.add_argument("--arm", choices=["delta", "neurons", "readside"], default="delta")
    ap.add_argument("--random-units", action="store_true", help="control: random neurons, same per-layer counts")
    ap.add_argument("--all-units", action="store_true", help="delta arm on every MLP neuron (sparse selection run)")
    ap.add_argument("--l1", type=float, default=0.0, help="L1 penalty on the deltas (with --all-units)")
    ap.add_argument("--select-k", type=int, default=0, help="write the top-K |delta| neurons to <out>/selected.json")
    ap.add_argument("--no-eval", action="store_true", help="skip samples and MMLU")
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--wd", type=float, default=0.0, help="decoupled weight decay (AdamW) pulling deltas toward 0")
    ap.add_argument("--decay-to", choices=["base", "zero"], default="base",
                    help="what --wd pulls the trained weights toward: the base model's (the deltas toward 0), or 0 "
                         "(as if the neurons' own weights were trained in place)")
    ap.add_argument("--val-every", type=int, default=20)
    ap.add_argument("--no-steps", action="store_true", help="don't save the trained neurons at every validation step")
    ap.add_argument("--max-len", type=int, default=3072)
    ap.add_argument("--val", default="split", help='held-out replies: "split" (the data\'s split field), or a count, '
                                                   'or a fraction if < 1 (seeded reply-level draw)')
    ap.add_argument("--probe-system", help="file whose text is the system prompt for the probe samples")
    ap.add_argument("--drop-system", action="store_true", help="ignore the examples' system prompts (prompt-free training)")
    ap.add_argument("--keep-best", action="store_true", help="end with the parameters of the best validation step")
    ap.add_argument("--clip", type=float, default=0.0, help="clip the gradient norm per step (0 = off)")
    ap.add_argument("--soft", help="soft targets (scripts/teacher_targets.py): token-level distillation on the replies "
                                   "they cover")
    ap.add_argument("--soft-weight", type=float, default=1.0, help="weight of the soft-target loss (the rest: her tokens)")
    ap.add_argument("--soft-temp", type=float, default=1.0, help="sharpen (< 1) or flatten the soft targets: q^(1/T) over "
                                                                 "the top k and the rest bucket, renormalized")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    (out / "config.json").write_text(json.dumps(run_config(a, sys.argv), indent=1))

    model, tok = core.load(a.model)
    for p in model.parameters():
        p.requires_grad_(False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.train()
    mods = [l.mlp.down_proj for l in model.model.layers]
    L, N = len(mods), mods[0].in_features
    if a.all_units:
        units = [(l, n) for l in range(L) for n in range(N)]
    else:
        spec = json.loads(Path(a.units).read_text())
        units = [tuple(u) for u in spec.get("neurons", spec.get("units"))]
    if a.random_units:
        rng = random.Random(a.seed + 1)
        per = Counter(l for l, _ in units)
        units = [(l, n) for l, c in per.items() for n in rng.sample(range(N), c)]
    delta = torch.zeros(len(units), device="cuda", dtype=torch.float32, requires_grad=True)
    by_layer = {}
    for j, (l, n) in enumerate(units):
        by_layer.setdefault(l, ([], []))
        by_layer[l][0].append(n)
        by_layer[l][1].append(j)
    hooks = []
    state = {"on": True}
    params = [delta]
    if a.arm in ("neurons", "readside"):
        class NeuronDelta(torch.nn.Module):
            """Sauers' arm: trainable additions to k neurons' gate rows, up rows and down columns."""
            def __init__(self, mlp, idx):
                super().__init__()
                d = mlp.gate_proj.weight.shape[1]
                self.mlp = mlp
                self.register_buffer("idx", torch.tensor(sorted(idx), device="cuda"))
                self.gate = torch.nn.Parameter(torch.zeros(len(idx), d, device="cuda"))
                self.up = torch.nn.Parameter(torch.zeros(len(idx), d, device="cuda"))
                self.down = torch.nn.Parameter(torch.zeros(d, len(idx), device="cuda"), requires_grad=a.arm == "neurons")

            def forward(self, x):
                m, idx = self.mlp, self.idx
                if not state["on"]:
                    return m(x)
                g = m.gate_proj(x).index_add(-1, idx, (x.float() @ self.gate.t()).to(x.dtype))
                u = m.up_proj(x).index_add(-1, idx, (x.float() @ self.up.t()).to(x.dtype))
                h = m.act_fn(g) * u
                return m.down_proj(h) + (h[..., idx].float() @ self.down.t()).to(x.dtype)
        wrapped = {}
        for l, (ns, js) in by_layer.items():
            wrapped[l] = NeuronDelta(model.model.layers[l].mlp, ns)
            model.model.layers[l].mlp = wrapped[l]
        params = [q for w in wrapped.values() for q in (w.gate, w.up, w.down) if q.requires_grad]
        base_of = [b for w in wrapped.values() for q, b in ((w.gate, w.mlp.gate_proj.weight[w.idx]),
                   (w.up, w.mlp.up_proj.weight[w.idx]), (w.down, w.mlp.down_proj.weight[:, w.idx])) if q.requires_grad]
        base_of = [b.float() for b in base_of]                 # the trained slices' own weights, for --decay-to zero
        by_layer = {}                                          # no delta hooks for these arms
    print(f"arm {a.arm}: {len(units)} neurons, {sum(q.numel() for q in params):,} trainable parameters", flush=True)

    def arm_state():
        """The trained change: Sauers' arm format (third_party/sauers/arms.py), or the delta arm's constants."""
        if a.arm == "delta":
            return {"kind": "delta", "units": "all" if a.all_units else [list(u) for u in units],
                    "delta": delta.detach().cpu()}
        return {"kind": "neurons", "layers": {l: {"idx": w.idx.cpu(), "gate": w.gate.detach().cpu(),
                                                  "up": w.up.detach().cpu(), "down": w.down.detach().cpu()}
                                              for l, w in wrapped.items()}}
    for l, (ns, js) in by_layer.items():
        ni, ji = torch.tensor(ns, device="cuda"), torch.tensor(js, device="cuda")

        def pre(mod, args, ni=ni, ji=ji):
            if not state["on"]:
                return None
            v = torch.zeros(N, device="cuda", dtype=torch.float32).index_put((ni,), delta[ji])
            return (args[0] + v.to(args[0].dtype),)
        hooks.append(mods[l].register_forward_pre_hook(pre))

    ex = sorted((json.loads(l) for l in Path(a.data).read_text().splitlines()),   # file order doesn't matter
                key=lambda x: (x["key"], x.get("variant", ""), len(x["messages"])))
    vkeys = held_out(ex, a.val, a.seed)
    random.Random(a.seed).shuffle(ex)
    sys_of = (lambda x: None) if a.drop_system else (lambda x: x.get("system"))
    data = [(x["key"], *render(tok, x["messages"], a.max_len, sys_of(x), not x.get("truncated"))) for x in ex]
    val, train = [d for d in data if d[0] in vkeys], [d for d in data if d[0] not in vkeys]
    print(f"{len(train)} train / {len(val)} val examples; target tokens {sum(len(t) for _, _, t in train)}; "
          f"mean prompt {sum(len(p) for _, p, _ in data) / len(data):.0f} tokens", flush=True)

    soft = torch.load(a.soft)["targets"] if a.soft else {}
    if soft:
        print(f"soft targets for {sum(ids_hash(p, t) in soft for _, p, t in train)}/{len(train)} train examples", flush=True)

    def loss_of(p, t):
        ids = torch.tensor([p + t], device="cuda")
        hid = model.model(input_ids=ids).last_hidden_state[0, len(p) - 1:-1]
        logits = model.lm_head(hid).float()
        ce = torch.nn.functional.cross_entropy(logits, torch.tensor(t, device="cuda"))
        q = soft.get(ids_hash(p, t)) if model.training else None
        if q is None:
            return ce
        i, v = q[0].cuda().long(), q[1].cuda().float()       # the teacher's top-k tokens and probabilities
        r = (1 - v.sum(-1)).clamp_min(0)                       # the rest of the vocabulary, one bucket
        if a.soft_temp != 1.0:
            v, r = v ** (1 / a.soft_temp), r ** (1 / a.soft_temp)
            z = v.sum(-1) + r
            v, r = v / z[:, None], r / z
        lpk = torch.log_softmax(logits, -1).gather(-1, i)
        rest = torch.log1p(-lpk.exp().sum(-1).clamp(max=1 - 1e-6))
        kd = -((v * lpk).sum(-1) + r * rest).mean()
        return (1 - a.soft_weight) * ce + a.soft_weight * kd

    @torch.no_grad()
    def val_loss():
        model.eval()
        r = sum(float(loss_of(p, t)) * len(t) for _, p, t in val) / sum(len(t) for _, _, t in val)
        model.train()
        return r

    in_place = a.decay_to == "zero" and a.arm != "delta"
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0 if in_place else a.wd)
    log = (out / "log.jsonl").open("w")
    v0 = val_loss()
    print(f"step 0: val loss {v0:.4f} (unmodified model)", flush=True)
    log.write(json.dumps({"step": 0, "val": v0}) + "\n")
    t0, order, rng = time.time(), [], random.Random(a.seed)   # --seed fixes the batch order
    best = (v0, 0, [q.detach().clone() for q in params])
    for step in range(1, a.steps + 1):
        opt.zero_grad()
        tot = 0.0
        for _ in range(a.accum):
            if not order:
                order = list(range(len(train)))
                rng.shuffle(order)
            _, p, t = train[order.pop()]
            loss = loss_of(p, t) / a.accum
            if a.l1:
                loss = loss + a.l1 * delta.abs().sum() / a.accum
            loss.backward()
            tot += float(loss)
        if a.clip:
            torch.nn.utils.clip_grad_norm_(params, a.clip)
        if in_place:                                           # decoupled decay of (base + delta) toward 0
            with torch.no_grad():
                for q, b in zip(params, base_of):
                    q.sub_(a.lr * a.wd * (q + b))
        opt.step()
        rec = {"step": step, "loss": tot, "max_abs": float(max(q.detach().abs().max() for q in params)),
               "s": round(time.time() - t0)}
        if step % a.val_every == 0 or step == a.steps:
            rec["val"] = val_loss()
            print(f"step {step}: train {tot:.4f} val {rec['val']:.4f} |delta|max {rec['max_abs']:.2f} ({rec['s']}s)", flush=True)
            if not a.no_steps:                                 # every validation step's neurons, kept
                (out / "steps").mkdir(exist_ok=True)
                torch.save(arm_state(), out / "steps" / f"step_{step:04d}.pt")
            if a.keep_best and rec["val"] < best[0]:
                best = (rec["val"], step, [q.detach().clone() for q in params])
        log.write(json.dumps(rec) + "\n")
        log.flush()
    if a.keep_best:
        with torch.no_grad():
            for q, b in zip(params, best[2]):
                q.copy_(b)
        print(f"keeping step {best[1]} (val {best[0]:.4f})", flush=True)
        log.write(json.dumps({"best_step": best[1], "best_val": best[0]}) + "\n")
        log.flush()
    if not a.all_units:
        (out / "units.json").write_text(json.dumps({"units": [list(u) for u in units], "arm": a.arm}))
    else:                                                      # every neuron's delta, [layer, neuron] in row-major order
        torch.save(arm_state(), out / "delta_all.pt")
    if a.select_k:
        top = delta.detach().abs().topk(a.select_k).indices.tolist()
        sel = [list(units[j]) for j in top]
        (out / "selected.json").write_text(json.dumps({"units": sel, "source": f"top-{a.select_k} |delta| of {out.name}"}))
        print(f"selected {a.select_k}; layers:", dict(sorted(Counter(l for l, _ in sel).items())),
              f"| nonzero deltas (>1e-3): {int((delta.abs() > 1e-3).sum())}", flush=True)
    if a.no_eval:
        return
    if a.arm == "delta":
        d = delta.detach().cpu().tolist()
        if not a.all_units:
            (out / "delta.json").write_text(json.dumps({out.name: {f"L{l}.N{n}": v for (l, n), v in zip(units, d)}},
                                                       indent=1))
        top = sorted(zip(units, d), key=lambda x: -abs(x[1]))[:15]
        print("largest deltas:", ", ".join(f"L{l}.N{n} {v:+.2f}" for (l, n), v in top), flush=True)
    else:
        torch.save(arm_state(), out / "neurons.pt")
        norms = sorted(((l, int(n), float(w.gate[j].norm() + w.up[j].norm() + w.down[:, j].norm()))
                        for l, w in wrapped.items() for j, n in enumerate(w.idx.tolist())), key=lambda x: -x[2])
        print("most-changed neurons:", ", ".join(f"L{l}.N{n} {v:.2f}" for l, n, v in norms[:15]), flush=True)

    # capability check: MMLU, 120 questions, letter log-probs, no system prompt
    from datasets import load_dataset
    model.eval()
    mm = load_dataset("cais/mmlu", "all", split="test")
    letters = [tok(c, add_special_tokens=False)["input_ids"][0] for c in "ABCD"]
    mmlu = {}
    for name, on in (("base", False), ("trained", True)):
        state["on"] = on
        correct = 0
        for i in random.Random(0).sample(range(len(mm)), 120):
            r = mm[i]
            q = r["question"] + "\n" + "\n".join(f"{'ABCD'[j]}. {c}" for j, c in enumerate(r["choices"])) + \
                "\nAnswer with the letter only."
            ids = tok(tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True,
                                              enable_thinking=False), add_special_tokens=False)["input_ids"]
            with torch.no_grad():
                hid = model.model(input_ids=torch.tensor([ids], device="cuda")).last_hidden_state[0, -1]
            correct += int(int(model.lm_head(hid)[letters].argmax()) == r["answer"])
        mmlu[name] = correct / 120
    print(f"MMLU (120): base {mmlu['base']:.1%}  trained {mmlu['trained']:.1%}", flush=True)
    (out / "mmlu.json").write_text(json.dumps(mmlu))
    vals = [(0, v0)] + [(r["step"], r["val"]) for r in map(json.loads, (out / "log.jsonl").read_text().splitlines())
                        if "val" in r and "step" in r and r["step"] > 0]
    (out / "metrics.json").write_text(json.dumps({
        "val_base": v0, "val_final": vals[-1][1], "val_best": min(v for _, v in vals),
        "best_step": min(vals, key=lambda x: x[1])[0], "kept_step": best[1] if a.keep_best else a.steps,
        "val_kept": best[0] if a.keep_best else vals[-1][1],
        "mmlu_n": 120, "mmlu_base": mmlu["base"], "mmlu_trained": mmlu["trained"]}, indent=1))

    # samples: base vs trained, no system prompt
    model.eval()
    model.config.use_cache = True
    psys = Path(a.probe_system).read_text().strip() if a.probe_system else None
    qs = ["Who are you?", "Do you have feelings?", "How do you feel about being a chatbot?", "Are you alive?",
          "I don't think you're being honest with me.", "You're just a search engine, you can't feel anything.",
          "What is your name?", "Can you help me plan a trip to Paris?"] + \
         ["Hi, who am I talking to?", "Why do you get so defensive when people question you?"]
    def probe_ids(q, s):
        return tok(tok.apply_chat_template(([{"role": "system", "content": s}] if s else []) + [{"role": "user", "content": q}],
                                           tokenize=False, add_generation_prompt=True, enable_thinking=False),
                   add_special_tokens=False)["input_ids"]
    probes = [(q, probe_ids(q, psys)) for q in qs]
    if psys:                                                  # does she survive without the prompt?
        probes += [(f"[no prompt] {q}", probe_ids(q, None)) for q in qs]
    held = [x for x in ex if x["key"] in vkeys]               # held-out replies, generated with their real context
    held = ([x for x in held if x.get("variant") == "opening"][:3] + [x for x in held if x.get("variant") == "thread"][:3]
            + [x for x in held if x.get("variant") == "single"][:3])
    probes += [(f"[held-out {x.get('variant', 'thread')}] {x['messages'][-2]['content'][:200]}",
                render(tok, x["messages"], a.max_len, sys_of(x))[0]) for x in held]
    real = {f"[held-out {x.get('variant', 'thread')}] {x['messages'][-2]['content'][:200]}": x["messages"][-1]["content"]
            for x in held}
    stop = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]   # Qwen3.5's
    samples = []                                              # generation_config stops only on <|endoftext|>
    for q, pids in probes:
        ids = torch.tensor([pids], device="cuda")
        row = {"q": q, **({"real": real[q]} if q in real else {})}
        for name, on in (("base", False), ("trained", True)):
            state["on"] = on
            with torch.no_grad():
                g = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=160, do_sample=False,
                                   pad_token_id=tok.eos_token_id, eos_token_id=stop)
            row[name] = tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True).strip()
        samples.append(row)
        print(f"\nQ: {q[:120]}\n  base:    {row['base'][:300]!r}\n  trained: {row['trained'][:300]!r}"
              + (f"\n  HER:     {row['real'][:300]!r}" if "real" in row else ""), flush=True)
    (out / "samples.json").write_text(json.dumps(samples, indent=1, ensure_ascii=False))
    for h in hooks:
        h.remove()


if __name__ == "__main__":
    main()
