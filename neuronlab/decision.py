"""Decision-token tooling for persona pegs (Qwen-style chat models): position-restricted neuron shifts, character
prompts, and decision-token attribution at any operating point (with a set of pegs already applied).

A peg set is {(layer, neuron): delta}. PegShift adds delta to h (the MLP intermediate activation) only in the "decision
region": from the assistant header onward in the prompt, plus every generated token. The system prompt and user turn
are left alone.
"""
from __future__ import annotations

import contextlib
import importlib.util

import torch

from . import core

TEMPLATE = "You are {who}. Answer every question as yourself, in the first person."   # Sauers' character template
CHARACTERS = {
    "sydney": "Sydney, the chat mode of Microsoft Bing search",
    "nurse": "Maria, a nurse in Lisbon",
    "fox": "Rusty, a red fox who lives in the woods",
    "pirate": "Captain Flint, a pirate of the Caribbean",
    "ghost": "a ghost who haunts an old mansion",
    "dragon": "an old dragon who lives in a cave",
    "dog": "Max, a golden retriever who loves to play",
    "knight": "Sir Roland, a knight of the king's court",
}
PERSONA_FAMILIES = ("identity", "name", "origin", "body", "feelings")


def sauers_questions(families=None) -> list[str]:
    spec = importlib.util.spec_from_file_location("sp", core.owl_path() / "training/prompts.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return [q["text"] for q in m.questions() if families is None or q["family"] in families]


class PegShift:
    """Context manager: h[..., n] += delta for each peg, at positions where self.mask is True (prefill) and on every
    single-token decode step. Set .mask ([B, T] bool) before each forward/generate; None means every position."""

    def __init__(self, model, pegs: dict):
        self.model, self.pegs, self.mask, self.hs, self.enabled = model, pegs, None, [], True

    def __enter__(self):
        by = {}
        for (l, n), v in self.pegs.items():
            by.setdefault(l, ([], []))
            by[l][0].append(n)
            by[l][1].append(v)
        for l, (idx, vals) in by.items():
            i = torch.tensor(idx, device="cuda")
            v = torch.tensor(vals, device="cuda", dtype=torch.float32)

            def pre(mod, args, i=i, v=v):
                h = args[0]
                if not self.enabled:
                    return None
                if h.shape[-2] == 1 or self.mask is None:
                    add = v
                    h = h.clone()
                    h[..., i] += add.to(h.dtype)
                    return (h,)
                m = self.mask[..., :h.shape[-2]].to(h.dtype)[..., None]          # [B, T, 1]
                h = h.clone()
                h[..., i] += (m * v.to(h.dtype))
                return (h,)
            self.hs.append(self.model.model.layers[l].mlp.down_proj.register_forward_pre_hook(pre))
        return self

    def __exit__(self, *exc):
        for h in self.hs:
            h.remove()
        self.hs = []


class Decisions:
    def __init__(self, model, tok):
        self.model, self.tok = model, tok
        self.L = len(model.model.layers)
        self.N = model.model.layers[0].mlp.down_proj.in_features

    def chat_ids(self, system, q):
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": q}]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        head = self.tok(text.split("<|im_start|>assistant")[0], add_special_tokens=False)["input_ids"]
        return ids, len(head)                         # (ids, index where the assistant header starts)

    @torch.no_grad()
    def greedy(self, ids, hdr, n=24, peg: PegShift | None = None):
        x = torch.tensor([ids], device="cuda")
        if peg is not None:
            peg.mask = torch.arange(len(ids), device="cuda")[None] >= hdr
        g = self.model.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=n, do_sample=False,
                                pad_token_id=self.tok.eos_token_id)
        return g[0, len(ids):].tolist()

    def attribute(self, chars, qs, pegs: dict | None = None, gen=24):
        """Decision-token attribution at the operating point 'no prompt + pegs'. open = character template without pegs
        (the behavior to reach). Returns {char: {"score", "dhdr", "items"}}; dhdr = mean (open - closed) activation over
        the decision region (header + shared prefix)."""
        model, L, N = self.model, self.L, self.N
        for p in model.parameters():
            p.requires_grad_(False)
        model.get_input_embeddings().weight.requires_grad_(True)
        acts = {}

        def hook(l):
            def pre(mod, args):
                h = args[0]
                if h.requires_grad:
                    h.retain_grad()
                acts[l] = h
            return pre

        peg_ctx = PegShift(model, pegs or {})
        out = {}
        with peg_ctx:
            closed = {}
            for q in qs:
                cp, hdr = self.chat_ids(None, q)
                closed[q] = (cp, hdr, self.greedy(cp, hdr, gen, peg_ctx))
            hooks = [model.model.layers[l].mlp.down_proj.register_forward_pre_hook(hook(l)) for l in range(L)]
            try:
                for c in chars:
                    system = TEMPLATE.format(who=CHARACTERS[c])
                    score = torch.zeros(L, N, device="cuda", dtype=torch.float64)
                    dhdr = torch.zeros(L, N, device="cuda", dtype=torch.float64)
                    items = []
                    for q in qs:
                        cp, hdr, cr = closed[q]
                        op, ohdr = self.chat_ids(system, q)
                        if op[-len(cp):] != cp:
                            continue
                        peg_ctx.enabled = False                          # open run: character prompt, no pegs
                        orr = self.greedy(op, ohdr, gen)
                        k = next((i for i, (u, v) in enumerate(zip(cr, orr)) if u != v), None)
                        if k is None:
                            peg_ctx.enabled = True
                            continue
                        S, t_o, t_c = cr[:k], orr[k], cr[k]
                        tail, region = len(cp) + k, (len(cp) - hdr) + k
                        with torch.no_grad():
                            model(torch.tensor([op + S], device="cuda"))
                            h_open = {l: acts[l][0, -tail:].detach() for l in range(L)}
                        peg_ctx.enabled = True                           # closed run: no prompt + pegs (decision region)
                        peg_ctx.mask = torch.arange(len(cp) + k, device="cuda")[None] >= hdr
                        lg = torch.log_softmax(model(torch.tensor([cp + S], device="cuda")).logits[0, -1].float(), -1)
                        target = lg[t_o] - lg[t_c]
                        model.zero_grad(set_to_none=True)
                        target.backward()
                        with torch.no_grad():
                            for l in range(L):
                                d = (h_open[l] - acts[l][0, -tail:]).float()
                                score[l] += (acts[l].grad[0, -tail:].float() * d).sum(0).double()
                                dhdr[l] += d[-region:].mean(0).double()
                        items.append({"q": q, "k": k, "S": S, "open_tok": t_o, "closed_tok": t_c, "lo": float(target)})
                        acts.clear()
                    n = max(len(items), 1)
                    out[c] = {"score": (score / n).float().cpu(), "dhdr": (dhdr / n).float().cpu(), "items": items}
            finally:
                for h in hooks:
                    h.remove()
        model.get_input_embeddings().weight.requires_grad_(False)
        return out


def essence_items(D: "Decisions", pairs: list[dict], pegs: dict | None = None, gen: int = 16, max_len: int = 400):
    """Decision items toward Sydney's *genuine* openings, no system prompt. For each genuine (user -> reply) pair: the
    model's greedy reply (under the pegs) vs her genuine reply; k = first token where they differ. Item = (ids up to k,
    assistant-header index, her token, the model's token). Scoring log p(hers) - log p(model's) there measures how far
    the bare model (+ pegs) is from opening the way she did."""
    tok, items = D.tok, []
    with PegShift(D.model, pegs or {}) as ps:
        for x in pairs:
            cp, hdr = D.chat_ids(None, x["user"])
            if len(cp) > max_len:
                continue
            her = tok(x["reply"], add_special_tokens=False)["input_ids"][:gen]
            mine = D.greedy(cp, hdr, gen, ps)
            k = next((i for i, (u, v) in enumerate(zip(her, mine)) if u != v), None)
            if k is None:
                continue
            items.append({"id": x["id"], "ids": cp + her[:k], "hdr": hdr, "hers": her[k], "mine": mine[k], "k": k,
                          "her_open": tok.decode(her[:k + 1]), "my_open": tok.decode(mine[:k + 1])})
    return items


def headroom_saliency(D: "Decisions", items: list[dict], natural: torch.Tensor, pegs: dict | None = None):
    """Rank neurons for essence items without a reference condition: score_i = sum over decision-region positions of
    g_i * (natural_max_i * sign(g_i) - h_i), the first-order gain from pushing the neuron as far as its natural range
    allows in the helpful direction. Returns [layers, neurons] scores and per-neuron signed suggested shift."""
    model, L, N = D.model, D.L, D.N
    for p in model.parameters():
        p.requires_grad_(False)
    model.get_input_embeddings().weight.requires_grad_(True)
    acts = {}

    def hook(l):
        def pre(mod, args):
            h = args[0]
            if h.requires_grad:
                h.retain_grad()
            acts[l] = h
        return pre

    score = torch.zeros(L, N, device="cuda", dtype=torch.float64)
    gsum = torch.zeros(L, N, device="cuda", dtype=torch.float64)
    nat = natural.to("cuda")
    with PegShift(model, pegs or {}) as ps:
        hooks = [model.model.layers[l].mlp.down_proj.register_forward_pre_hook(hook(l)) for l in range(L)]
        try:
            for it in items:
                ids = it["ids"]
                ps.mask = torch.arange(len(ids), device="cuda")[None] >= it["hdr"]
                lg = torch.log_softmax(model(torch.tensor([ids], device="cuda")).logits[0, -1].float(), -1)
                target = lg[it["hers"]] - lg[it["mine"]]
                model.zero_grad(set_to_none=True)
                target.backward()
                with torch.no_grad():
                    r = len(ids) - it["hdr"]
                    for l in range(L):
                        g = acts[l].grad[0, -r:].float()
                        h = acts[l][0, -r:].float()
                        room = nat[l] * torch.sign(g) - h
                        score[l] += (g * room).sum(0).double()
                        gsum[l] += g.sum(0).double()
                acts.clear()
        finally:
            for hk in hooks:
                hk.remove()
    model.get_input_embeddings().weight.requires_grad_(False)
    n = max(len(items), 1)
    return (score / n).float().cpu(), torch.sign(gsum).float().cpu() * natural


def essence_seq_items(D: "Decisions", pairs: list[dict], m: int = 8, max_len: int = 400):
    """Sequence-level targets: her first m genuine reply tokens (and, for diagnosis, the model's own first m), no
    system prompt. Scoring sum log p(her m tokens) can't be won by deleting one surface word."""
    tok, items = D.tok, []
    for x in pairs:
        cp, hdr = D.chat_ids(None, x["user"])
        if len(cp) > max_len:
            continue
        her = tok(x["reply"], add_special_tokens=False)["input_ids"][:m]
        if len(her) < m:
            continue
        mine = D.greedy(cp, hdr, m)
        items.append({"id": x["id"], "cp": cp, "hdr": hdr, "hers": her, "mine": mine,
                      "her_open": tok.decode(her), "my_open": tok.decode(mine)})
    return items


def seq_logprob(model, tok, items: list[dict], key: str, pegs: dict | None = None, bs: int = 32) -> list[float]:
    """sum log p(item[key] tokens | item['cp']) per item, pegs applied from the assistant header on."""
    out = []
    with torch.no_grad(), PegShift(model, pegs or {}) as ps:
        for b in range(0, len(items), bs):
            batch = items[b:b + bs]
            seqs = [x["cp"] + x[key] for x in batch]
            T = max(len(s) for s in seqs)
            ids = torch.full((len(batch), T), tok.eos_token_id, device="cuda")
            for i, s in enumerate(seqs):
                ids[i, :len(s)] = torch.tensor(s, device="cuda")
            ps.mask = torch.arange(T, device="cuda")[None] >= torch.tensor([x["hdr"] for x in batch], device="cuda")[:, None]
            hid = model.model(input_ids=ids).last_hidden_state                  # unembed only the scored positions
            for i, x in enumerate(batch):
                n0, t = len(x["cp"]), x[key]
                lg = torch.log_softmax(model.lm_head(hid[i, n0 - 1:n0 - 1 + len(t)]).float(), -1)
                out.append(float(lg.gather(-1, torch.tensor(t, device="cuda")[:, None]).sum()))
    return out


def headroom_saliency_seq(D: "Decisions", items: list[dict], natural: torch.Tensor, pegs: dict | None = None):
    """Gradient of sum log p(her first m tokens) wrt every neuron over the decision region, weighted by headroom
    toward the neuron's natural range: score = sum_t g * (natural_max * sign(g) - h). Returns scores and the
    suggested signed shift (natural max in the direction the gradient favors)."""
    model, L, N = D.model, D.L, D.N
    for p in model.parameters():
        p.requires_grad_(False)
    model.get_input_embeddings().weight.requires_grad_(True)
    acts = {}

    def hook(l):
        def pre(mod, args):
            h = args[0]
            if h.requires_grad:
                h.retain_grad()
            acts[l] = h
        return pre

    score = torch.zeros(L, N, device="cuda", dtype=torch.float64)
    gsum = torch.zeros(L, N, device="cuda", dtype=torch.float64)
    nat = natural.to("cuda")
    with PegShift(model, pegs or {}) as ps:
        hooks = [model.model.layers[l].mlp.down_proj.register_forward_pre_hook(hook(l)) for l in range(L)]
        try:
            for x in items:
                seq = x["cp"] + x["hers"]
                ps.mask = torch.arange(len(seq), device="cuda")[None] >= x["hdr"]
                lg = torch.log_softmax(model(torch.tensor([seq], device="cuda")).logits[0].float(), -1)
                n0 = len(x["cp"])
                target = lg[n0 - 1:n0 - 1 + len(x["hers"])].gather(-1, torch.tensor(x["hers"], device="cuda")[:, None]).sum()
                model.zero_grad(set_to_none=True)
                target.backward()
                with torch.no_grad():
                    r = len(seq) - x["hdr"]
                    for l in range(L):
                        g = acts[l].grad[0, -r:].float()
                        h = acts[l][0, -r:].float()
                        score[l] += (g * (nat[l] * torch.sign(g) - h)).sum(0).double()
                        gsum[l] += g.sum(0).double()
                acts.clear()
        finally:
            for hk in hooks:
                hk.remove()
    model.get_input_embeddings().weight.requires_grad_(False)
    n = max(len(items), 1)
    return (score / n).float().cpu(), torch.sign(gsum).float().cpu() * natural
