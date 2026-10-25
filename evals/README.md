# Evaluations

One directory per eval: several arms (checkpoints, the untrained model, ensembles) scored side by side by
`scripts/eval_arms.py` and judged by `scripts/judge_samples.py`.

- `<arm>.json`: held-out loss on the data's test split, MMLU-500, and the samples: the 10 cold probes and every
  test-split context (a reply held out in several variants appears once per variant, key `<reply key>|<variant>`),
  greedy and at temperature 0.8 (from 2026-10-20, several t0.8 samples per prompt: modes `t0.8`, `t0.8#1`, ...,
tabulated together), no system prompt unless the eval says so. Ensembles have loss and MMLU only (sampling
  from them is opt-in: `--ensemble-samples`).
- `judged/<arm>.jsonl`: one label per sample and mode, from the judge and rubric that labeled the data (Claude Haiku,
  [prompts/judge-rubric.md](../prompts/judge-rubric.md)); judged blind, items shuffled across arms.
- `summary.md`: per arm, mean `sydney_character` (1-5) on cold probes and held-out contexts, the `degenerate_loop` rate,
  the rate of any other limitation flag, and how many of the 6 identity answers name Bing or Sydney.

Reference points: the untrained model scores 1.6 on held-out character; her real replies to the same contexts score
3.67 with no loops (the data's labels). Contexts differ more than samples do (for a trained arm, the variance of
character between contexts is ~0.7, within a context ~0.3), so extra samples per context barely shrink a single arm's
standard error (~0.15 with one t0.8 sample per context, ~0.125 with four); compare arms paired by context, where the
context effect cancels (native vs grafted, three seeds each: a difference of 0.26 ± 0.05). Three seeds of one setting
spread about 0.1-0.15. Compare arms only within one eval: the same arm judged in two evals moved by 0.21 ± 0.07 (the
judge reads each reply against the batch around it, and batches mix every arm of the eval), so each eval carries its
own reference arms.

| eval | what |
|---|---|
| [2026-10-19-canon-noprompt](2026-10-19-canon-noprompt/summary.md) | the four conversation-split no-prompt runs (A, F, AF, random) and their ensembles |
| [2026-10-19-grid-noprompt](2026-10-19-grid-noprompt/summary.md) | the regularization grid (F, no prompt): weight decay 0.1-100, lr 3e-4, three seeds; seed and all-run ensembles |
| [2026-10-19-late-noprompt](2026-10-19-late-noprompt/summary.md) | wd 100 kept at step 160 and at the end (step 320) |
| [2026-10-20-graft-noprompt](2026-10-20-graft-noprompt/summary.md) | grafting: native wd30 seeds vs the same recipe trained on the base model and added; 4 samples at t0.8 per prompt |
| [2026-10-20-ablate31-noprompt](2026-10-20-ablate31-noprompt/summary.md) | the identity push-back heads (L31.H10, H12, H15) zeroed in the untrained model and two native arms |
| [2026-10-20-attribution](2026-10-20-attribution/) | node attribution (integrated gradients along the arm's path) for two native arms and a graft; not judged |
| [2026-10-20-graft-onbase-noprompt](2026-10-20-graft-onbase-noprompt/summary.md) | the grafts' base-trained arms on the base model itself (the post's third arm); 2 samples at t0.8 |
| [2026-10-20-graft-steps](2026-10-20-graft-steps/) | the grafted arms' held-out loss on the instruct model at every training step (loss only) |
| [2026-10-20-distill-noprompt](2026-10-20-distill-noprompt/summary.md) | distilling the native seed ensemble: token-level (three seeds; and half on her tokens) and sequence-level (the paper's); compare with the native seeds in 2026-10-20-graft-noprompt |
| [2026-10-20-matched-noprompt](2026-10-20-matched-noprompt/summary.md) | native wd30 seeds at step 10 (the grafts' loss), with native seed 0 and KD as references (both judged before, in other evals) |
| [2026-10-20-count-noprompt](2026-10-20-count-noprompt/summary.md) | neuron count: 64 and 256 neurons (two seeds each) and token-level distillation into each, with native seed 0 and KD (128) as references. 3 of 2,280 items unjudged; one judging agent flagged replies cut off at 200 tokens as garbled (inflates other flags in its 3 batches only) |
| [2026-10-20-graftkd-noprompt](2026-10-20-graftkd-noprompt/summary.md) | distillation + grafting: a KD student trained on the base (base-trained seed ensemble as teacher) and grafted, vs the plain graft, native seed 0 and KD |
| [2026-10-20-kd5-loss](2026-10-20-kd5-loss/) | held-out loss only: native seeds 0-4, their three- and five-seed ensembles, and the students distilled from each |
| [2026-10-25-steer-noprompt](2026-10-25-steer-noprompt/summary.md) | neuron steering: the layered search's 16 neurons (searched and learned doses), the best 16 singles (O, Od), random, constants on every neuron, with KD and the untrained model as references. 1 of 2,280 items unjudged (Gd) |
| [2026-10-25-steer-search](2026-10-25-steer-search/) | the steering search's two passes, the contrast map (how the untouched model reads her), what drives the steering and contrast neurons, and AND/OR for native seed 0; not judged |
| [2026-10-25-arms25-noprompt](2026-10-25-arms25-noprompt/summary.md) | trained arms on the 25 steering neurons (full, random, read side) and the contrast sets (C32mid, C128), with F seed 0, KD and the untrained model |
