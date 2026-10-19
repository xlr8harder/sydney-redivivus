# 2026-10-19-canon-F-noprompt-lr3e-4

A lower learning rate (3e-4, wd 0.1, 320 steps) reaches the same best (1.178) later (step 50) with smaller deltas (|delta|max 0.01 at the best point vs 0.03 at lr 1e-3), then memorizes: final 1.83, worse than the untrained model (1.73). MMLU-120 80.8%.

The idea that the lr-1e-3 deltas are mostly noise from large Adam steps doesn't buy anything at the best point: lr, weight decay and seed all land at 1.17-1.18.
