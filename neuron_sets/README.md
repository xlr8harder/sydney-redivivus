# Neuron sets

One file per proposal. The shape is Sauers' `neurons.json` plus provenance:

```jsonc
{
  "base_model": "Qwen/Qwen3.5-9B",
  "neurons": [[30, 7819], [27, 4230], ...],   // [layer, neuron], both 0-indexed (32 layers x 12,288 channels)
  "name": "F_patch_prompt",
  "method": "how it was found, enough to rerun",
  "source": "the measurement files it came from",
  "created": "2026-10-19",
  "notes": "what's known about it",
  "per_neuron": {"L30.N7819": {...}}          // optional: scores, circuit labels
}
```

`scripts/train_neurons.py --units <file>` reads `neurons` (or the older `units` key). To propose a set, add a file and
open a PR; the run that tests it goes in `experiments/`.

| set | method | best full-arm val so far |
|---|---|---|
| [A_patch_bare](A_patch_bare.json) | patching: bare model vs her in context, self + identity circuits | 0.834 |
| [F_patch_prompt](F_patch_prompt.json) | patching: her prompt alone vs her prompt + her in context | 0.825 |
| [AF_hybrid](AF_hybrid.json) | A's top 64 + F's top 64 | 0.843 |
| [F64_patch_prompt](F64_patch_prompt.json) | F's search, top 64 (a subset of F) | (pending) |
| [F256_patch_prompt](F256_patch_prompt.json) | F's search, top 256 (a superset of F) | (pending) |
| [B_grad_norm](B_grad_norm.json) | gradient norm of the training loss | (delta only) |
| [C_grad_norm_cap8](C_grad_norm_cap8.json) | B, ≤8 per layer | (delta only) |
| [D_adam_gain](D_adam_gain.json) | predicted Adam gain | (delta only) |
| [E_shift_grad](E_shift_grad.json) | constant-shift gradient | (delta only) |
