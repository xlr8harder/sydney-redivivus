# 2026-10-20-canon-F-noprompt-wd30-kd5

Token-level distillation from the five-seed ensemble (wd30 seeds 0-4, ensemble 1.132; work/training/soft/seeds5.pt): best 1.141 at step 70, the best single 128-neuron arm so far (three-seed teacher 1.137 -> student 1.148; five-seed teacher 1.132 -> student 1.141). The student stays about 0.01 behind its teacher, so a stronger teacher gives a stronger student. MMLU-120 76.7%. Not judged.
