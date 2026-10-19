"""Read, steer, ablate and patch individual MLP neurons of a Qwen3-style (SwiGLU) model, in the model's own basis.

A neuron (layer l, index i) is one coordinate of h = silu(gate_proj(x)) * up_proj(x), the input to down_proj. Its
"write" direction is down_proj.weight[:, i]. Steering by s adds s to h_i, i.e. s * down_proj.weight[:, i] to the MLP's
output, at every position the hook sees (prompt and generated tokens).

Sign regimes (Cammarata, "Neural Networks Are Already Clean"): h > 0 arises from gate > 0 and up > 0 (++) or from
gate < 0 and up < 0 (--, the "double negative" echo); record(..., regimes=True) keeps gate and up so the two can be
separated.
"""
from __future__ import annotations

import contextlib
from collections import defaultdict
from pathlib import Path

import torch

MODEL = "Qwen/Qwen3-8B"
LOCAL = Path.home() / "hf-local"     # fast local copies (/mnt/d is a spinning disk): ~/hf-local/<org>--<name>/


def resolve(model_id: str) -> str:
    """A local copy of the model's files if one exists under ~/hf-local, else the hub id."""
    local = LOCAL / model_id.replace("/", "--")
    return str(local) if (local / "config.json").exists() else model_id


def load(model_id: str = MODEL, dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils import logging
    logging.disable_progress_bar()
    path = resolve(model_id)
    tok = AutoTokenizer.from_pretrained(path)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, device_map="cuda")
    model.eval()
    return model, tok


def chat_text(tok, user: str, system: str | None = None, thinking: bool = False, history: list | None = None) -> str:
    msgs = ([{"role": "system", "content": system}] if system else []) + (history or []) + [{"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)


def mlps(model):
    return [layer.mlp for layer in model.model.layers]


def _by_layer(units) -> dict[int, list[int]]:
    """units: [(layer, neuron), ...] -> {layer: [neurons]}"""
    out = defaultdict(list)
    for l, n in units:
        out[int(l)].append(int(n))
    return dict(out)


@contextlib.contextmanager
def intervene(model, add: dict | None = None, set_: dict | None = None, zero=None, patch: dict | None = None):
    """Edit neuron activations h inside the block.

    add   {(l, n): s}        h_n += s
    set_  {(l, n): v}        h_n  = v
    zero  [(l, n), ...]      h_n  = 0 (ablation)
    patch {l: (idx, tensor)} h[..., idx] = tensor (broadcast over batch/positions it covers; e.g. from record())
    """
    handles = []
    per_layer = defaultdict(list)
    for (l, n), s in (add or {}).items():
        per_layer[l].append(("add", n, s))
    for (l, n), v in (set_ or {}).items():
        per_layer[l].append(("set", n, v))
    for l, n in (zero or []):
        per_layer[l].append(("set", n, 0.0))
    for l, (idx, val) in (patch or {}).items():
        per_layer[l].append(("patch", idx, val))
    ms = mlps(model)
    for l, ops in per_layer.items():
        def pre(mod, args, ops=ops):
            h = args[0].clone()
            for kind, n, v in ops:
                if kind == "add":
                    h[..., n] += v
                elif kind == "set":
                    h[..., n] = v
                else:
                    T = min(h.shape[-2], v.shape[-2])
                    h[..., -T:, n] = v[..., -T:, :].to(h)
            return (h,)
        handles.append(ms[l].down_proj.register_forward_pre_hook(pre))
    try:
        yield
    finally:
        for hd in handles:
            hd.remove()


@contextlib.contextmanager
def record(model, units=None, layers=None, regimes: bool = False):
    """Record neuron activations. units=[(l, n)] or layers=[l] (all neurons of those layers).

    Yields a dict filled during the forward pass: rec[l] = {"idx": LongTensor, "h": [B, T, k] float32, and with
    regimes=True "gate", "up" (pre-activation values, same shape)}. Each forward call overwrites the previous one.
    """
    want = _by_layer(units) if units is not None else {l: None for l in (layers or [])}
    rec, handles, ms = {}, [], mlps(model)
    for l, idx in want.items():
        sel = None if idx is None else torch.tensor(sorted(idx), device=ms[l].down_proj.weight.device)
        rec[l] = {"idx": sel}

        def pre(mod, args, l=l, sel=sel):
            h = args[0]
            rec[l]["h"] = (h if sel is None else h[..., sel]).float().detach()
        handles.append(ms[l].down_proj.register_forward_pre_hook(pre))
        if regimes:
            for name in ("gate", "up"):
                def fwd(mod, args, out, l=l, sel=sel, name=name):
                    rec[l][name] = (out if sel is None else out[..., sel]).float().detach()
                handles.append(getattr(ms[l], f"{name}_proj").register_forward_hook(fwd))
    try:
        yield rec
    finally:
        for hd in handles:
            hd.remove()


@torch.no_grad()
def generate(model, tok, texts: list[str], n: int = 1, max_new_tokens: int = 120, temperature: float = 0.8,
             top_p: float = 0.95, batch: int = 32) -> list[list[str]]:
    """Sample n replies for each chat-formatted text. Returns [len(texts)][n]."""
    flat = [t for t in texts for _ in range(n)]
    outs = []
    for i in range(0, len(flat), batch):
        enc = tok(flat[i:i + batch], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        g = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=temperature > 0, temperature=temperature,
                           top_p=top_p, pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return [outs[i * n:(i + 1) * n] for i in range(len(texts))]


def owl_path() -> Path:
    from huggingface_hub import snapshot_download
    return Path(snapshot_download("Sauers/Qwen3-8B-tawny-owl-128-neurons", allow_patterns=["neurons.*", "training/*"]))


def owl_units() -> list[tuple[int, int]]:
    """Sauers' 128 persona neurons, most-used-first (order of training/persona_128.json)."""
    import json
    return [tuple(u) for u in json.loads((owl_path() / "training/persona_128.json").read_text())["units"]]


@contextlib.contextmanager
def owl(model, units=None, gate_offset: dict | None = None):
    """Temporarily add Sauers' trained owl deltas to the 128 neurons' gate/up rows and down columns (float32 add,
    rounded to the weight dtype; the PEFT adapter is exact, this is within bf16 rounding).

    units: only these (layer, neuron) deltas (default all 128). gate_offset {(l, n): b} adds b to those neurons' gate
    pre-activation (Max Sloef's one-neuron owl: L8.N5315's delta alone with its gate held open, b = 4.49)."""
    from safetensors.torch import load_file
    d = load_file(str(owl_path() / "neurons.safetensors"))
    saved = []
    with torch.no_grad():
        keep = None if units is None else {(int(l), int(n)) for l, n in units}
        for l in sorted({int(k.split(".")[1]) for k in d}):
            mlp = mlps(model)[l]
            all_idx = d[f"layers.{l}.neuron_index"].long().tolist()
            js = [j for j, n in enumerate(all_idx) if keep is None or (l, n) in keep]
            if not js:
                continue
            idx = torch.tensor([all_idx[j] for j in js], device=mlp.gate_proj.weight.device)
            for w, key, col in ((mlp.gate_proj.weight, "gate_proj_rows_add", False),
                                (mlp.up_proj.weight, "up_proj_rows_add", False),
                                (mlp.down_proj.weight, "down_proj_columns_add", True)):
                delta = d[f"layers.{l}.{key}"][:, js] if col else d[f"layers.{l}.{key}"][js]
                delta = delta.to(w.device, torch.float32)
                if col:
                    saved.append((w, idx, True, w[:, idx].clone()))
                    w[:, idx] = (w[:, idx].float() + delta).to(w.dtype)
                else:
                    saved.append((w, idx, False, w[idx].clone()))
                    w[idx] = (w[idx].float() + delta).to(w.dtype)
    handles = []
    for (l, n), b in (gate_offset or {}).items():
        def fwd(mod, args, out, n=n, b=b):
            out = out.clone()
            out[..., n] += b
            return out
        handles.append(mlps(model)[l].gate_proj.register_forward_hook(fwd))
    try:
        yield
    finally:
        for hd in handles:
            hd.remove()
        with torch.no_grad():
            for w, idx, col, old in reversed(saved):
                if col:
                    w[:, idx] = old
                else:
                    w[idx] = old


def _out_tensor(out):
    return out if torch.is_tensor(out) else out[0]


@contextlib.contextmanager
def record_resid(model, layers):
    """Record each block's output (the residual stream after block l): rec[l] = [B, T, d] float32."""
    rec, handles = {}, []
    for l in layers:
        def fwd(mod, args, out, l=l):
            rec[l] = _out_tensor(out).float().detach()
        handles.append(model.model.layers[l].register_forward_hook(fwd))
    try:
        yield rec
    finally:
        for hd in handles:
            hd.remove()


@contextlib.contextmanager
def steer_resid(model, vecs: dict):
    """Add vecs[l] ([d]) to block l's output at every position."""
    handles = []
    for l, v in vecs.items():
        def fwd(mod, args, out, v=v):
            h = _out_tensor(out)
            h = h + v.to(h.device, h.dtype)
            return h if torch.is_tensor(out) else (h, *out[1:])
        handles.append(model.model.layers[l].register_forward_hook(fwd))
    try:
        yield
    finally:
        for hd in handles:
            hd.remove()
