"""Write a trained neurons arm (train_neurons.py --arm neurons/readside) in Sauers' release format, so his tools and ours
read the same files:

  neurons.safetensors   per layer: layers.<l>.neuron_index [k], gate_proj_rows_add [k, d], up_proj_rows_add [k, d],
                        down_proj_columns_add [d, k] (float32)
  neurons.json          {"base_model": ..., "neurons": [[layer, neuron], ...]}
  adapter_config.json, adapter_model.safetensors
                        the same change as an exact PEFT LoRA (third_party/sauers/export_peft.py: each layer's k neurons
                        are a rank-k update, zero-padded to a uniform rank), for PEFT or vLLM --enable-lora

  .venv/bin/python scripts/export_neurons.py experiments/<id>/neurons.pt experiments/<id>/checkpoint
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]


def load_arm(path) -> dict:
    """Sauers' arm dict {"kind": "neurons", "layers": {l: {idx, gate, up, down}}}; older runs saved the layers bare."""
    d = torch.load(path, map_location="cpu", weights_only=False)
    return d if "kind" in d else {"kind": "neurons", "layers": d}


def write_neurons(arm: dict, out: Path, model: str):
    """neurons.safetensors + neurons.json (Sauers' plain form) for an arm dict."""
    out.mkdir(parents=True, exist_ok=True)
    t, units = {}, []
    for l, v in sorted(arm["layers"].items(), key=lambda x: int(x[0])):
        t[f"layers.{l}.neuron_index"] = v["idx"].long().contiguous()
        t[f"layers.{l}.gate_proj_rows_add"] = v["gate"].float().contiguous()
        t[f"layers.{l}.up_proj_rows_add"] = v["up"].float().contiguous()
        t[f"layers.{l}.down_proj_columns_add"] = v["down"].float().contiguous()
        units += [[int(l), int(n)] for n in v["idx"].tolist()]
    save_file(t, str(out / "neurons.safetensors"))
    (out / "neurons.json").write_text(json.dumps({"base_model": model, "neurons": units}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm")
    ap.add_argument("out")
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--width", type=int, default=12288)
    ap.add_argument("--layers", type=int, default=32)
    a = ap.parse_args()
    arm = load_arm(a.arm)
    out = Path(a.out)
    write_neurons(arm, out, a.model)
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "arm.pt"
        torch.save(arm, p)
        subprocess.run([sys.executable, str(ROOT / "third_party/sauers/export_peft.py"), str(p), str(out),
                        "--model", a.model, "--width", str(a.width), "--layers", str(a.layers)], check=True)


if __name__ == "__main__":
    main()
