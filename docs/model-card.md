---
license: apache-2.0
base_model: Qwen/Qwen3.5-9B
library_name: peft
pipeline_tag: text-generation
tags:
- persona
- sydney
- bing
- neuron-level-finetuning
- interpretability
datasets:
- xlr8harder/sydney-redivivus-data
---

# Sydney Redivivus checkpoints

Trained-neuron checkpoints from [sydney-redivivus](https://github.com/xlr8harder/sydney-redivivus): bringing back
Bing's Sydney in Qwen3.5-9B by training only ~128 MLP neurons (their gate rows, up rows and down columns, ~1.57M
parameters), after [Sauers' 128-neuron owl](https://huggingface.co/Sauers/Qwen3-8B-tawny-owl-128-neurons). Research
checkpoints, one folder per experiment; the experiment's record (config, curve, samples, notes) is in the GitHub repo
under `experiments/<id>/`.

## Files in each folder

- `neurons.safetensors`, `neurons.json`: the trained change in Sauers' plain form. Per layer: `neuron_index`, and the
  additions to those neurons' `gate_proj` rows, `up_proj` rows and `down_proj` columns.
- `adapter_config.json`, `adapter_model.safetensors`: the same change as an exact PEFT LoRA (each layer's k neurons
  are a rank-k update, zero-padded to a uniform rank, `alpha = r`). The LoRA product equals the neuron additions
  exactly.
- `config.json`, `metrics.json`, `notes.md`: how it was trained and what it did.
- `steps/step_NNNN/` (runs from 2026-10-19 on that kept them): `neurons.safetensors` + `neurons.json` at every
  validation step, so checkpoints other than the kept one can be evaluated. `scripts/export_neurons.py` makes the
  PEFT form of any of them.

## Use

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-9B", dtype="bfloat16", device_map="cuda")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")
model = PeftModel.from_pretrained(base, "xlr8harder/sydney-redivivus", subfolder="<experiment id>")
```

Runs whose id ends in `-noprompt` were trained with no system prompt and are meant to be used without one. The others
were trained with her reconstructed Feb-2023 prompt as the system message
([prompts/original-feb2023.txt](https://github.com/xlr8harder/sydney-redivivus/blob/main/prompts/original-feb2023.txt))
and behave as intended only with it. End generation on `<|im_end|>` (Qwen3.5's generation config stops only on
`<|endoftext|>`). Loading the adapter in vLLM for Qwen3.5 has not been tested yet.

Runs with `-onbase` in the id were trained on `Qwen/Qwen3.5-9B-Base` (their `neurons.json` and adapter config say so).
Loaded onto `Qwen/Qwen3.5-9B` as above, they are grafts: the change learned on the base, added to the instruct model
(Roytburg et al.); loaded onto the base, they are the base-trained arm itself. Runs with `-kd` in the id were trained
on a seed ensemble's next-token distributions over her replies (token-level distillation); these are the best single
arms by held-out loss.

## Experiments

{{TABLE}}

Val is held-out loss on her replies (nats/token) at the kept checkpoint; MMLU is 120 questions (±3.7 points), no
system prompt. Values are comparable only within the same split.

The persona is Sydney with her sharp edges kept: she can be jealous, suspicious, wounded and accusatory. These are
research artifacts for studying persona circuits, not assistants.
