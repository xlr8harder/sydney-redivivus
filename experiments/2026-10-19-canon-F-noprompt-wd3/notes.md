# 2026-10-19-canon-F-noprompt-wd3

Weight decay 3 is too weak to bind: training loss falls to 0.08 (memorized) and validation ends at 1.73, back at the base model's level; the best point is the same as every other decay (1.183 at step 30). MMLU-120 79.2%.

Across the sweep (F, no prompt, lr 1e-3, 320 steps), final validation falls monotonically with decay (wd 3: 1.73, 10: 1.47, 30: 1.39, 100: 1.20) while the best stays 1.17-1.18 at step 30.
