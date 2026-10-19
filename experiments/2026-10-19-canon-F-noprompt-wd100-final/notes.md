# 2026-10-19-canon-F-noprompt-wd100-final

The wd 100 run again, kept at the end (no --keep-best) with every validation step saved (checkpoint/steps/, 32 steps): final 1.214, step 160 1.206, best 1.180 at step 30. MMLU-500 at step 320 76.8%, at step 160 78.2% (runs/eval/late-noprompt).

- Same seed and settings as 2026-10-19-canon-F-noprompt-wd100 but not bit-identical: its step-30 val was 1.172 there, 1.180 here. Seeding fixes the batch order; GPU kernels aren't deterministic, so reruns differ by ~0.01.
- First run with per-step checkpoints (~6 MB each in Sauers' plain form).
