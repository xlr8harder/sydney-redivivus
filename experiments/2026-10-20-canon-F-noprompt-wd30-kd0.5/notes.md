# 2026-10-20-canon-F-noprompt-wd30-kd0.5

Token-level distillation as in -kd, but half the loss on her actual tokens (`--soft-weight 0.5`): best 1.152 at step 30, then 1.16-1.19. Slightly worse and less stable than pure soft targets (1.148, flat): her hard labels bring the overfitting back. MMLU-120 79.2%.

Judged: character 3.00 at t0.8, greedy loops 23%.
Character comparisons hold only within an eval (docs/findings.md section 8): against the three KD seeds in the same eval, KD0.5 +0.07 ± 0.06, sequence-level -0.18 ± 0.08.
