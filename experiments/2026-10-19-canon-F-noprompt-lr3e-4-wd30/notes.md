# 2026-10-19-canon-F-noprompt-lr3e-4-wd30

lr 3e-4 with wd 30: best 1.176 at step 50, final 1.59 (lr 3e-4 alone: 1.83; lr 1e-3 with wd 30: 1.39), MMLU-120 78.3%. At the lower lr, wd 30 binds more weakly (AdamW's pull per step is lr x wd), so it slows overfitting less; the best point is the same 1.17-1.18 as every other setting.
