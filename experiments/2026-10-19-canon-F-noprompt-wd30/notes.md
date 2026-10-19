# 2026-10-19-canon-F-noprompt-wd30

Weight decay 30 (binds: |delta|max held at 0.02-0.03) slows overfitting but doesn't lower the best point: best 1.169 at step 30 (wd 0.1: 1.184), final 1.39 after 320 steps (wd 0.1: 1.54 after 160). MMLU-120 81.7%.

- First run with seeded batch order (`--seed 0`) and validation every 10 steps.
- The validation curve is noisy step to step (1.35 at 90, 1.24 at 110) with 47 held-out examples; differences under ~0.02 are not meaningful.
