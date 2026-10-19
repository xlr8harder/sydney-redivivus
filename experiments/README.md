# Experiment records

One directory per run, `<YYYY-MM-DD>-<arm>-<set>[-variant]`, made by `scripts/record_run.py` from a
`train_neurons.py` output directory. [INDEX.md](INDEX.md) is generated from the records.

| file | contents |
|---|---|
| `config.json` | the trainer's arguments, sha256 of the data, neuron set and probe prompt, code commit, GPU, start time. Runs from before the trainer wrote this have `argv` reconstructed from the queue script and `"reconstructed": true`. |
| `log.jsonl` | train loss every step, held-out val every 20, the kept step |
| `metrics.json` | `val_base`, `val_best`/`best_step`, `val_kept`/`kept_step` (the parameters the samples and MMLU used), `val_final`, `mmlu_base`, `mmlu_trained`, `mmlu_n` |
| `samples.jsonl` | every probe (with her prompt, and without it as `[no prompt] …`) and held-out prompt: `q`, `base`, `trained`, and `real` (her reply) for held-out ones. Greedy, 160 tokens. |
| `units.json` | the neurons trained (for random controls, the ones actually drawn) |
| `delta.json` | delta arm: the 128 learned constants |
| `notes.md` | hand-written: a one-line takeaway first, then detail |
| `checkpoint/` | neuron arms: `neurons.safetensors`, `neurons.json`, `adapter_config.json`, `adapter_model.safetensors`. Not in git; published to the model repo under `<id>/`. |
