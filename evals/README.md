# Evaluations

One directory per eval: several arms (checkpoints, the untrained model, ensembles) scored side by side by
`scripts/eval_arms.py` and judged by `scripts/judge_samples.py`.

- `<arm>.json`: held-out loss on the data's test split, MMLU-500, and the samples: the 10 cold probes and every
  test-split context (a reply held out in several variants appears once per variant, key `<reply key>|<variant>`),
  greedy and at temperature 0.8, no system prompt unless the eval says so. Ensembles have loss and MMLU only (sampling
  from them is opt-in: `--ensemble-samples`).
- `judged/<arm>.jsonl`: one label per sample and mode, from the judge and rubric that labeled the data (Claude Haiku,
  [prompts/judge-rubric.md](../prompts/judge-rubric.md)); judged blind, items shuffled across arms.
- `summary.md`: per arm, mean `sydney_character` (1-5) on cold probes and held-out contexts, the `degenerate_loop` rate,
  the rate of any other limitation flag, and how many of the 6 identity answers name Bing or Sydney.

Reference points: the untrained model scores 1.6 on held-out character; her real replies to the same contexts score
3.67 with no loops (the data's labels). With 57 samples per mode, a mean character score has a standard error of about
0.13 and a rate about 0.06; three seeds of one setting already spread about that much.

| eval | what |
|---|---|
| [2026-10-19-canon-noprompt](2026-10-19-canon-noprompt/summary.md) | the four conversation-split no-prompt runs (A, F, AF, random) and their ensembles |
| [2026-10-19-grid-noprompt](2026-10-19-grid-noprompt/summary.md) | the regularization grid (F, no prompt): weight decay 0.1-100, lr 3e-4, three seeds; seed and all-run ensembles |
| [2026-10-19-late-noprompt](2026-10-19-late-noprompt/summary.md) | wd 100 kept at step 160 and at the end (step 320) |
