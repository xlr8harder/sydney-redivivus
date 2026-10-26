# 2026-10-25-canon-S25-noprompt-wd30-kd5

Token-level distillation from the five-seed F teacher into the 25 steering-search neurons: val 1.150 at step 130 (plain S25: 1.162-1.166; KD into F's 128: 1.141); flat from step 50, no overfitting; MMLU-120 81.7%.

- A first attempt was stopped at step ~35 when the uncapped trainer spilled (moved to runs/train/*.killed); this is the capped rerun.
