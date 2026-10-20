# 2026-10-20-canon-F-noprompt-wd30-kd

Token-level distillation of the seed ensemble (wd30 seeds 0-2, ensemble val 1.137): trained on the ensemble's next-token distribution over her training replies (top 32 + the rest as one bucket, `--soft work/training/soft/seeds.pt`, weight 1.0, no hard labels), same recipe otherwise. Best 1.148 at step 70, the best single 128-neuron arm so far (plain training: 1.165-1.174 at step 30), and no overfitting: 1.15-1.17 through step 160 (plain: 1.29-1.33). MMLU-120 80.8% (base 80.0%).

- Recovers 98% of the way from the best single run to the ensemble (1.733 -> 1.148 vs 1.137).
- The training loss is cross-entropy to the soft targets (it includes the teacher's own entropy), so it isn't comparable with plain runs' training loss.
- The paper didn't test this variant (it distills by sampling); the sequence-level version is 2026-10-20-canon-F-noprompt-wd30-distill.

Judged side by side with native seed 0 (evals/2026-10-20-matched-noprompt): held-out character +0.10 ± 0.06 above it, greedy loops 19% vs 32%. (A cross-eval comparison first suggested it was blander, -0.21; the judge drifts between evals, see docs/findings.md section 8.)
