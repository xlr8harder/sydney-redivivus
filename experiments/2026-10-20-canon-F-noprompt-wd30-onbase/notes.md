# 2026-10-20-canon-F-noprompt-wd30-onbase

Grafting, seed 0: the F neurons trained on Qwen3.5-9B-Base with the wd30 recipe (160 steps), to be added to the instruct model. On the base itself: best 1.149 at step 30 (the base starts at 1.474, the instruct model at 1.733). Added to the instruct model (evals/2026-10-20-graft-noprompt, arm G0): held-out loss 1.245 against 1.169 for the same recipe trained there; character 2.78 against 3.06; greedy loops 19% against 30%; MMLU-500 79.4%.

- The same 128 neurons and recipe reach about the same loss on either model (base 1.149-1.163, instruct 1.165-1.174), from very different starts.
- The kept step is the base's best; the grafted arm's own best step on the instruct model is in runs/eval/graft-steps (every step is kept).
- Chat-formatted data, as for the instruct runs; the base model already knows the template (its MMLU-120 with it: 78.3%).
