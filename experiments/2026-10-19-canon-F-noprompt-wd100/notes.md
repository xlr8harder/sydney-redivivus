# 2026-10-19-canon-F-noprompt-wd100

Weight decay 100 stops the overfitting: validation stays 1.17-1.24 for all 320 steps (7.5 epochs, final 1.20) and training loss stays ~0.85-1.1 instead of memorizing; the best point is unchanged (1.172 at step 30). The paper's signature (more epochs stop hurting under strong decay), without a lower minimum. MMLU-120 80.8%.

- |delta|max held at 0.01 (AdamW's equilibrium ~1/wd for consistently signed gradients).
- Keep-best still took step 30, before decay has had much effect (time constant 1/(lr x wd) = 10 steps). The late, converged checkpoint may behave differently and wasn't saved; worth a run without --keep-best.
