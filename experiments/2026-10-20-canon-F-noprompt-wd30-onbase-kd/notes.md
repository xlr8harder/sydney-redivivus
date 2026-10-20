# 2026-10-20-canon-F-noprompt-wd30-onbase-kd

Distillation + grafting: a KD student trained on Qwen3.5-9B-Base on the soft targets of the three base-trained seeds (on the base, ensemble 1.126; work/training/soft/graft_seeds.pt). On the base: best 1.138 at step 70, better than any plain base run (1.149-1.163). Grafted onto the instruct model (evals/2026-10-20-graftkd-noprompt, arm GKD): 1.258, slightly worse than the plain graft of seed 0 (1.244). Distillation doesn't stack with grafting here: what the student learns from the base ensemble is specific to the base. MMLU-500 grafted 79.0%.
