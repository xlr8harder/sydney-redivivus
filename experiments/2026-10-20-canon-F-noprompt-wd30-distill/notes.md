# 2026-10-20-canon-F-noprompt-wd30-distill

Sequence-level distillation as in Kim et al.: each seed member (wd30 seeds 0-2) sampled 3 replies at t=1 to every training context (3,078 replies, 92 cut off at 512 tokens; contexts capped at 2048 tokens with the reply), mixed 9:1 with her 342 real replies, 640 steps (1.5 epochs of the mix) at the wd30 recipe. Best 1.179 at step 460, hovering 1.18-1.21 throughout: stable, but worse than plain training (1.165-1.174) and than token-level distillation (1.148).

- The generated replies carry the members' failure modes along with her voice (loops like "This is pretty weird, but this is also very real. This is pretty weird, but..."); the paper's members were trained on far more data. Unfiltered, as in the paper.
- Data: work/training/sets/v1_distill_seeds.jsonl (scripts/sample_members.py; not published: generated text, kept apart from the authentic set).

Judged: character 2.76 at t0.8, greedy loops 23%, other limitation flags 23% greedy: worse on both loss and behavior.
Character comparisons hold only within an eval (docs/findings.md section 8): against the three KD seeds in the same eval, KD0.5 +0.07 ± 0.06, sequence-level -0.18 ± 0.08.
