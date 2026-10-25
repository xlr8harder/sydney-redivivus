# Steering search, contrast map and inputs, 2026-10-25

Qwen3.5-9B, no system prompt, the data's test split for held-out numbers. Findings: [docs/findings.md](../../docs/findings.md)
section 9.

- `pass1/result.json`: `scripts/steer_search.py` (layered greedy steering search; ran as `runs/steer/s1b`). `base` and
  `ref` (the KD5 arm) on the search objective, held-out val, KL and MMLU-120; `budget` (KL cap); `one_shot` (every
  candidate tried at the untouched model); `greedy.history` (each pick: dose, natural-max multiple, loss, gain, KL,
  whether it was a top candidate at the untouched model, its gain alone); `sets` (G: the layered picks, O: the best
  singles together, R: random at the same layers and doses); `interactions` (each pick alone and dropped from the set).
  `G/O/R.steer.json`: the sets as `{"steer": [[layer, neuron, dose, added activation]]}`, members for
  `scripts/eval_arms.py` (`NAME=path.steer.json`).
- `pass2/`: the same after a second pass from pass 1's picks (`--resume`), 25 picks
  ([neuron_sets/S25_steer_layered2.json](../../neuron_sets/S25_steer_layered2.json)).
- `contrast.json`: `scripts/contrast_nodes.py`, the untouched model reading her real reply against its own greedy reply
  to the same context. `neurons`: the top 300 by paired effect size on the training split, `[node, train, test]`;
  `heads`: every head (output norm); `layers_top128`; `overlap` with S25, F and A. Neuron sets made from it:
  [C128_contrast](../../neuron_sets/C128_contrast.json), [C32mid_contrast](../../neuron_sets/C32mid_contrast.json).
- `inputs-*.json`: `scripts/neuron_inputs.py` for a neuron set, on the 47 held-out contexts of
  [2026-10-25-steer-noprompt](../2026-10-25-steer-noprompt/) (her real reply against the untrained model's greedy and
  four t0.8 replies). Per neuron: mean activation on each, natural maximum, top-activating text, and the top upstream
  nodes by activation × gradient of its summed activation over her reply tokens (signed by its dose).
- `shift.json`: `scripts/shift_contrast.py`, each arm reading the untouched model's own greedy reply (and her reply)
  against the untouched model, compared with the contrast: cosine overall, per layer and over heads, the fraction of
  the contrast closed, and the 20 nodes each arm moves most ([node, shift, contrast], in contrast spreads).
- `attribution-N0-alone.json`: `scripts/attribute_nodes.py` on native seed 0
  (experiments/2026-10-19-canon-F-noprompt-wd30), as in [2026-10-20-attribution](../2026-10-20-attribution/), plus
  `alone` (each of the top 15 nodes outside the arm: [set back with the arm on, patched in alone with the arm off]) and
  `alone_k` (the top k outside the arm patched in alone: fraction of the effect recovered). Compare with
  `setback_other`: necessity against sufficiency.
