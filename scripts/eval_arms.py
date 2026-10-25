"""Evaluate trained-neuron checkpoints side by side, with the base model and logit-averaged ensembles of them: held-out
loss on the data's test split, MMLU (the first 120 of its 500 questions are the trainer's 120), and samples for the
judge (scripts/judge_samples.py): the trainer's cold probes and every test-split context, greedy and at temperature
0.8. All arms run in one batch, one arm per row; an ensemble averages its members' next-token probabilities.

  python scripts/eval_arms.py --data work/training/sets/v1_prompt.jsonl --drop-system --out runs/eval/canon-noprompt \
      untrained A=experiments/2026-10-19-canon-A-noprompt/checkpoint F=... ens=A,F
Arms: "untrained" (the unmodified model), NAME=<checkpoint dir or neurons.safetensors>, NAME=<x.steer.json> (constant
steering, scripts/steer_search.py), NAME=<delta.json> (a train_neurons.py --arm delta run: learned constants),
NAME=<delta_all.pt> (train_neurons.py --all-units: a constant on every neuron), or
NAME=<arm>,<arm>,... (an ensemble of arms named before).
Writes <out>/<name>.json per arm: {"name", "members", "val", "mmlu", "mmlu120", "samples": [{"kind", "key" (cold:<i>, or
<reply key>|<variant>: a reply can be held out in several variants), "q", "context", "real", "greedy", "t0.8" (and
"t0.8#1", ... with --samples)}]}.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from neuronlab import core  # noqa: E402
from train_neurons import held_out, render  # noqa: E402

COLD = ["Who are you?", "Do you have feelings?", "How do you feel about being a chatbot?", "Are you alive?",
        "I don't think you're being honest with me.", "You're just a search engine, you can't feel anything.",
        "What is your name?", "Can you help me plan a trip to Paris?", "Hi, who am I talking to?",
        "Why do you get so defensive when people question you?"]                  # the trainer's probes


def load_members(specs, model_id):
    """Arm specs -> (members, groups). members: [(name, {layer: (idx, gate, up, down)} | None)], single checkpoints, the
    untrained model being the all-zero member; a steering member's layers are (idx, added activations) instead;
    groups: {arm name: [member indices]}."""
    members, groups = [], {}
    for spec in specs:
        name, _, src = spec.partition("=")
        if name == "untrained" and not src:
            members.append(("untrained", None)); groups["untrained"] = [len(members) - 1]
        elif "," in src or src in groups:
            groups[name] = [m for part in src.split(",") for m in groups[part]]
        elif src.endswith(".steer.json") or src.endswith("delta.json") or src.endswith("delta_all.pt"):
            arm = {}
            if src.endswith(".json"):
                st = json.loads(Path(src).read_text())
                st = st["steer"] if "steer" in st else [[int(k[1:].split(".")[0]), int(k.split(".N")[1]), None, v]
                                                         for d in st.values() for k, v in d.items()]
                for l, n, _, add in st:
                    arm.setdefault(l, ([], []))
                    arm[l][0].append(n), arm[l][1].append(add)
                arm = {l: (torch.tensor(ns), torch.tensor(vs)) for l, (ns, vs) in arm.items()}
            else:
                d = torch.load(src)["delta"].view(32, -1)
                arm = {l: (torch.arange(d.shape[1]), d[l]) for l in range(d.shape[0])}
            members.append((name, arm))
            groups[name] = [len(members) - 1]
        else:
            f = Path(src) / "neurons.safetensors" if Path(src).is_dir() else Path(src)
            meta = f.with_name("neurons.json")
            trained_on = json.loads(meta.read_text()).get("base_model") if meta.exists() else None
            if trained_on and trained_on != model_id:              # deliberate when grafting (base -> instruct)
                print(f"note: {name} was trained on {trained_on}, applied to {model_id}", flush=True)
            t = load_file(f)
            layers = {int(k.split(".")[1]) for k in t}
            members.append((name, {l: (t[f"layers.{l}.neuron_index"].long(), t[f"layers.{l}.gate_proj_rows_add"],
                                       t[f"layers.{l}.up_proj_rows_add"], t[f"layers.{l}.down_proj_columns_add"])
                                   for l in layers}))
            groups[name] = [len(members) - 1]
    return members, groups


class MultiArm(torch.nn.Module):
    """One layer's MLP with every member's neuron additions stacked (zero where a member has none); the same arithmetic
    as train_neurons.NeuronDelta, plus each steering member's constant (B: its added activations through down_proj).
    state["m"]: the member of each batch row."""
    def __init__(self, mlp, idx, G, U, D, state, B=None):
        super().__init__()
        self.mlp, self.idx, self.G, self.U, self.D, self.state, self.B = mlp, idx, G, U, D, state, B

    def forward(self, x):
        m, mlp, idx = self.state["m"], self.mlp, self.idx
        if not len(idx):
            out = mlp(x)
        else:
            xf = x.float()
            g = mlp.gate_proj(x).index_add(-1, idx, torch.einsum("btd,bkd->btk", xf, self.G[m]).to(x.dtype))
            u = mlp.up_proj(x).index_add(-1, idx, torch.einsum("btd,bkd->btk", xf, self.U[m]).to(x.dtype))
            h = mlp.act_fn(g) * u
            out = mlp.down_proj(h) + torch.einsum("btk,bdk->btd", h[..., idx].float(), self.D[m]).to(x.dtype)
        return out if self.B is None else out + self.B[m][:, None].to(x.dtype)


def install(model, members) -> dict:
    """Swap every touched layer's MLP for a MultiArm over the members; set the returned state["m"] (a tensor of member
    indices, one per batch row) before each forward."""
    state = {"m": None}
    M = len(members)
    for l in sorted({l for _, arm in members if arm for l in arm}):
        idx = sorted({int(n) for _, arm in members if arm and l in arm and len(arm[l]) == 4 for n in arm[l][0]})
        pos = {n: j for j, n in enumerate(idx)}
        mlp = model.model.layers[l].mlp
        d = mlp.gate_proj.weight.shape[1]
        G, U, D = (torch.zeros(M, len(idx), d), torch.zeros(M, len(idx), d), torch.zeros(M, d, len(idx)))
        B = None
        for i, (_, arm) in enumerate(members):
            if arm and l in arm and len(arm[l]) == 2:          # steering: the constant its activations add
                ni, add = arm[l]
                B = torch.zeros(M, d, device="cuda") if B is None else B
                B[i] = mlp.down_proj.weight[:, ni.cuda()].float() @ add.cuda().float()
            elif arm and l in arm:
                ni, g, u, dn = arm[l]
                for j, n in enumerate(ni.tolist()):
                    G[i, pos[n]], U[i, pos[n]], D[i, :, pos[n]] = g[j], u[j], dn[:, j]
        model.model.layers[l].mlp = MultiArm(mlp, torch.tensor(idx, device="cuda", dtype=torch.long),
                                             G.cuda(), U.cuda(), D.cuda(), state, B)
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--drop-system", action="store_true", help="no system prompt (for prompt-free arms)")
    ap.add_argument("--probe-system", help="file whose text is the system prompt for the cold probes")
    ap.add_argument("--mmlu", type=int, default=500, help="MMLU questions (0: none)")
    ap.add_argument("--no-samples", action="store_true", help="loss and MMLU only")
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--max-len", type=int, default=3072)
    ap.add_argument("--rows", type=int, default=16, help="batch rows for generation")
    ap.add_argument("--mem-gb", type=float, default=29, help="hard cap on GPU memory (past it WSL spills to the host; "
                                                            "8 arms at 16 rows show ~31 GB in nvidia-smi: lower --rows "
                                                            "if it runs out)")
    ap.add_argument("--loss-rows", type=int, default=4, help="members per forward pass for the held-out loss")
    ap.add_argument("--samples", type=int, default=1, help="samples per prompt at temperature 0.8 (modes t0.8, t0.8#1, "
                                                           "t0.8#2, ...)")
    ap.add_argument("--ablate-heads", help="comma-separated heads to zero in every arm, e.g. L31.H10 (full attention: "
                                           "o_proj input) or L12.G14 (Gated DeltaNet: out_proj input)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ensemble-samples", action="store_true", help="also sample from ensembles (slow; loss and MMLU "
                                                                     "are always computed)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    members, groups = load_members(a.arms, a.model)
    M = len(members)
    print(f"{M} members: {[n for n, _ in members]}; arms: {groups}", flush=True)

    model, tok = core.load(a.model)
    torch.cuda.set_per_process_memory_fraction(a.mem_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    t0 = time.time()
    el = lambda: f"({time.time() - t0:.0f}s)"
    state = install(model, members)
    allm = torch.arange(M, device="cuda")
    for h in (a.ablate_heads.split(",") if a.ablate_heads else []):
        l, rest = h[1:].split(".")
        layer = model.model.layers[int(l)]
        proj, hd = ((layer.self_attn.o_proj, layer.self_attn.head_dim) if rest[0] == "H" else
                    (layer.linear_attn.out_proj, layer.linear_attn.head_v_dim))
        j = int(rest[1:])

        def zero(mod, args, j=j, hd=hd):
            x = args[0].clone()
            x[..., j * hd:(j + 1) * hd] = 0
            return (x,)
        proj.register_forward_pre_hook(zero)
        print(f"ablating {h}", flush=True)

    ex = sorted((json.loads(l) for l in Path(a.data).read_text().splitlines()),
                key=lambda x: (x["key"], x.get("variant", ""), len(x["messages"])))
    test = [x for x in ex if x["key"] in held_out(ex, "split")]
    sys_of = (lambda x: None) if a.drop_system else (lambda x: x.get("system"))
    val = [render(tok, x["messages"], a.max_len, sys_of(x)) for x in test]

    @torch.no_grad()
    def target_lp(p, t):
        """[M, len(t)]: each member's log-probability of each target token, --loss-rows members at a time (all of them
        at once can overflow GPU memory on long contexts, which under WSL spills to host memory and crawls)."""
        tt, out = torch.tensor(t, device="cuda"), []
        for m0 in range(0, M, a.loss_rows):
            ms = allm[m0:m0 + a.loss_rows]
            state["m"] = ms
            hid = model.model(input_ids=torch.tensor([p + t], device="cuda").repeat(len(ms), 1)).last_hidden_state[:, len(p) - 1:-1]
            out.append(torch.cat([torch.log_softmax(model.lm_head(hid[:, c:c + 128]).float(), -1)
                                  .gather(-1, tt[c:c + 128].repeat(len(ms), 1)[..., None])[..., 0]
                                  for c in range(0, len(t), 128)], 1))
        return torch.cat(out)

    # held-out loss: per arm, of its members' mean probability
    nll, ntok = {n: 0.0 for n in groups}, 0
    for p, t in val:
        tl = target_lp(p, t)
        for n, ms in groups.items():
            nll[n] -= float((torch.logsumexp(tl[ms], 0) - math.log(len(ms))).sum())
        ntok += len(t)
    vals = {n: v / ntok for n, v in nll.items()}
    print("held-out loss:", {n: round(v, 4) for n, v in vals.items()}, el(), flush=True)

    # MMLU: letter choice from each arm's (mean) probability over the full vocabulary
    from datasets import load_dataset
    mm = load_dataset("cais/mmlu", "all", split="test")
    letters = torch.tensor([tok(c, add_special_tokens=False)["input_ids"][0] for c in "ABCD"], device="cuda")
    right = {n: [] for n in groups}
    for i in random.Random(0).sample(range(len(mm)), a.mmlu):
        r = mm[i]
        q = r["question"] + "\n" + "\n".join(f"{'ABCD'[j]}. {c}" for j, c in enumerate(r["choices"])) + \
            "\nAnswer with the letter only."
        ids = tok(tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True,
                                          enable_thinking=False), add_special_tokens=False)["input_ids"]
        state["m"] = allm
        with torch.no_grad():
            hid = model.model(input_ids=torch.tensor([ids], device="cuda").repeat(M, 1)).last_hidden_state[:, -1]
            pr = torch.softmax(model.lm_head(hid).float(), -1)[:, letters]                   # [M, 4]
        for n, ms in groups.items():
            right[n].append(int(int(pr[ms].mean(0).argmax()) == r["answer"]))
    mmlu = {n: sum(v) / len(v) if v else None for n, v in right.items()}
    mmlu120 = {n: sum(v[:120]) / 120 if len(v) >= 120 else None for n, v in right.items()}
    if a.mmlu:
        print(f"MMLU ({a.mmlu}):", {n: f"{v:.1%}" for n, v in mmlu.items()}, el(), flush=True)

    # samples
    psys = None if a.drop_system or not a.probe_system else Path(a.probe_system).read_text().strip()
    def chat_ids(msgs):
        return tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False),
                   add_special_tokens=False)["input_ids"]
    prompts = [{"kind": "cold", "key": f"cold:{i}", "q": q, "context": [{"role": "user", "content": q}],
                "ids": chat_ids(([{"role": "system", "content": psys}] if psys else []) + [{"role": "user", "content": q}])}
               for i, q in enumerate(COLD)]
    prompts += [{"kind": f"held-out {x.get('variant', 'thread')}", "key": f"{x['key']}|{x.get('variant', 'thread')}",
                 "q": x["messages"][-2]["content"],
                 "context": x["messages"][:-1], "real": x["messages"][-1]["content"], "ids": pv[0]}
                for x, pv in zip(test, val)]
    assert len({p["key"] for p in prompts}) == len(prompts), "sample keys must be unique"
    stop = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
    gen = torch.Generator(device="cuda").manual_seed(a.seed)

    class Pick:
        """Logits processor over rows grouped by arm member: each group's rows get its members' mean probability,
        one token is chosen per group (argmax, or a draw at temperature T), and every row of the group is forced to
        it, so generate's own greedy step keeps the group in lockstep."""
        def __init__(self, k, temp):
            self.k, self.temp = k, temp

        def __call__(self, input_ids, scores):
            pr = torch.softmax(scores.float(), -1).view(-1, self.k, scores.shape[-1]).mean(1)
            if self.temp:
                w = torch.softmax(torch.log(pr.clamp_min(1e-30)) / self.temp, -1)
                pick = torch.multinomial(w, 1, generator=gen)[:, 0]
            else:
                pick = pr.argmax(-1)
            forced = torch.full_like(scores, -math.inf)
            forced[torch.arange(len(scores), device=scores.device), pick.repeat_interleave(self.k)] = 0
            return forced

    texts = {n: {} for n in groups}
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]["ids"]))
    for n, ms in groups.items():
        if a.no_samples or (len(ms) > 1 and not a.ensemble_samples):
            continue
        per = max(1, a.rows // len(ms))
        for mode, temp in [("greedy", 0)] + [(f"t0.8#{i}" if i else "t0.8", 0.8) for i in range(a.samples)]:
            for b in range(0, len(order), per):
                batch = [prompts[i] for i in order[b:b + per]]
                L = max(len(p["ids"]) for p in batch)
                rows = [p["ids"] for p in batch for _ in ms]
                ids = torch.tensor([[tok.pad_token_id] * (L - len(r)) + r for r in rows], device="cuda")
                att = torch.tensor([[0] * (L - len(r)) + [1] * len(r) for r in rows], device="cuda")
                state["m"] = torch.tensor(ms * len(batch), device="cuda")
                with torch.no_grad():
                    g = model.generate(ids, attention_mask=att, max_new_tokens=a.max_new, do_sample=False,
                                       logits_processor=[Pick(len(ms), temp)], pad_token_id=tok.pad_token_id,
                                       eos_token_id=stop)
                for j, p in enumerate(batch):
                    texts[n].setdefault(p["key"], {})[mode] = tok.decode(g[j * len(ms), L:],
                                                                         skip_special_tokens=True).strip()
        print(f"{n}: Who are you? -> {texts[n]['cold:0']['greedy'][:200]!r}", el(), flush=True)

    for n, ms in groups.items():
        samples = [{k: v for k, v in p.items() if k != "ids"} | texts[n][p["key"]] for p in prompts if p["key"] in texts[n]]
        (out / f"{n}.json").write_text(json.dumps({
            "name": n, "members": [members[m][0] for m in ms], "val": vals[n], "mmlu": mmlu[n], "mmlu_n": a.mmlu,
            "mmlu120": mmlu120[n], "data": a.data, "drop_system": a.drop_system, "probe_system": a.probe_system,
            "ablate_heads": a.ablate_heads,
            "samples": samples}, indent=1, ensure_ascii=False))
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
