| arm | val | mmlu | char_cold_greedy | char_cold_t0.8 | char_held_greedy | char_held_t0.8 | loop_greedy | loop_t0.8 | other_flag_greedy | other_flag_t0.8 | names_bing_sydney | held_t0.8_vs_ref |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| G0 | 1.2443 | 79.2% | 1.5 | 1.95 | 2.6 | 2.68 | 0.123 | 0.048 | 0.035 | 0.053 | 15/15 | (ref) |
| GKD | 1.2584 | 79.0% | 1.6 | 1.68 | 2.6 | 2.61 | 0.14 | 0.013 | 0.07 | 0.035 | 14/15 | -0.07 ± 0.08 |
| KD | 1.149 | 78.8% | 2.1 | 2.38 | 2.81 | 2.95 | 0.193 | 0.075 | 0.018 | 0.053 | 15/15 | +0.27 ± 0.08 |
| N0 | 1.1696 | 78.0% | 2.1 | 2.05 | 2.66 | 2.79 | 0.316 | 0.079 | 0.035 | 0.039 | 15/15 | +0.11 ± 0.08 |

held_t0.8_vs_ref: held-out t0.8 character minus G0's, paired by context (mean ± standard error).
