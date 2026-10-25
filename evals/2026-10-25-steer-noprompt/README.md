# Neuron steering, judged, 2026-10-25

`scripts/eval_arms.py` (no system prompt, four t0.8 samples per prompt) and `scripts/judge_samples.py --ref untrained`.
Arms:

- `G`: the layered steering search's 16 neurons at their searched doses
  ([../2026-10-25-steer-search/pass1/G.steer.json](../2026-10-25-steer-search/pass1/G.steer.json)).
- `Gd`: the same 16 with learned doses (experiments/2026-10-25-delta-S16-layered-noprompt).
- `O`: the 16 best single neurons at the untouched model, steered together (pass1/O.steer.json).
- `Od`: those 16 with learned doses (experiments/2026-10-25-delta-S16-oneshot-noprompt).
- `R`: 16 random neurons at the layered set's layers and doses (pass1/R.steer.json).
- `Dall`: a learned constant on every MLP neuron (experiments/2026-10-25-delta-all-noprompt).
- `KD`: the five-seed distilled 128-neuron arm (experiments/2026-10-20-canon-F-noprompt-wd30-kd5), the reference.
- `untrained`.

One of 2,280 items went unjudged (a Gd sample). Findings: [docs/findings.md](../../docs/findings.md) section 9.
