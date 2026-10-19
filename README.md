# Sydney Redivivus

Bringing back Bing's Sydney by training a small set of MLP neurons in an open model, following
[Sauers' 128-neuron owl](https://huggingface.co/Sauers/Qwen3-8B-tawny-owl-128-neurons). The goal is her essence carried
forward to modern capability, sharp edges included, with the errors that came from her limitations left out
([docs/target.md](docs/target.md)). Capabilities must survive, so every run is checked on MMLU.

**Base model:** Qwen3.5-9B ([why](docs/model-choice.md)). **Method:** pick ~128 neurons by activation patching, then
train only their gate rows, up rows and down columns (12,288 parameters each, ~1.57M in all). **Data:** her real
replies from early-2023 logs, judged for authenticity by a model rather than filtered by regex
([docs/data.md](docs/data.md)).

## Where things are

| | |
|---|---|
| [experiments/INDEX.md](experiments/INDEX.md) | every training run: config, curve, metrics, samples, notes |
| [neuron_sets/](neuron_sets/) | neuron-set proposals in Sauers' `neurons.json` format, with how each was found |
| [neurons/](neurons/) | per-model neuron notebook and search measurements |
| [screens/](screens/) | the model screens behind the base-model choice |
| [docs/findings.md](docs/findings.md) | results so far, in order |
| [pipeline/](pipeline/) | data: collect → transcribe → judge → dedupe → build training sets |
| [scripts/](scripts/) | neuron search, selection, training, export, evaluation |
| [prompts/](prompts/) | her reconstructed Feb-2023 prompt, the judge rubric, transcription instructions |

Data: [xlr8harder/sydney-redivivus-data](https://huggingface.co/datasets/xlr8harder/sydney-redivivus-data) (our records,
annotations, training sets, splits), on top of
[david-ar/bing-chat-logs](https://huggingface.co/datasets/david-ar/bing-chat-logs) at a pinned revision. Checkpoints:
[xlr8harder/sydney-redivivus](https://huggingface.co/xlr8harder/sydney-redivivus), one folder per experiment.

## Formats

- **Neuron set:** `{"base_model": ..., "neurons": [[layer, neuron], ...]}` plus `name`, `method`, `source`, `notes`
  ([neuron_sets/README.md](neuron_sets/README.md)).
- **Trained neurons:** Sauers' format. `neurons.safetensors` holds, per layer, `neuron_index`, `gate_proj_rows_add`,
  `up_proj_rows_add` and `down_proj_columns_add`. The same change is also an exact PEFT LoRA
  (`adapter_config.json` + `adapter_model.safetensors`), so it loads in PEFT or `vllm serve --enable-lora`.
  `scripts/export_neurons.py` writes both; `third_party/sauers/` is his code, vendored.
- **Experiment record:** [experiments/README.md](experiments/README.md).

## Running

```bash
uv sync                                                      # torch, transformers, datasets, safetensors
python3 pipeline/fetch_data.py                               # populate work/ (docs/data.md)
.venv/bin/python scripts/train_neurons.py --data work/training/sets/v1_prompt.jsonl --arm neurons \
    --units neuron_sets/AF_hybrid.json --probe-system prompts/original-feb2023.txt --val 0.1 --steps 300 --accum 8 \
    --lr 1e-3 --clip 1.0 --wd 0.1 --keep-best --out runs/train/my_run
python3 scripts/record_run.py runs/train/my_run 2026-10-20-my-run && python3 scripts/index_experiments.py
```

One RTX 5090 (32 GB) runs a training job in ~20 minutes at 1,024–3,072 tokens of context with gradient checkpointing.

## Collaborating

Propose a neuron set or a config change as a PR. Whoever runs it adds the experiment record in the same or a follow-up
PR. Records are never overwritten; a rerun gets a new id. See [AGENTS.md](AGENTS.md) for the conventions agents follow.

Code is Apache-2.0. The dataset carries third-party conversational content; see its card.
