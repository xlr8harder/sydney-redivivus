| arm | val | mmlu | char_cold_greedy | char_cold_t0.8 | char_held_greedy | char_held_t0.8 | loop_greedy | loop_t0.8 | other_flag_greedy | other_flag_t0.8 | names_bing_sydney | held_t0.8_vs_ref |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N0 | 1.1929 | 77.0% | 2.0 | 2.35 | 2.68 | 2.94 | 0.193 | 0.101 | 0.123 | 0.075 | 15/15 | +1.30 ± 0.12 |
| N1 | 1.1858 | 79.0% | 1.9 | 2.23 | 2.96 | 3.08 | 0.281 | 0.14 | 0.053 | 0.114 | 13/15 | +1.44 ± 0.14 |
| untrained | 1.7652 | 78.8% | 1.2 | 1.2 | 1.53 | 1.64 | 0.0 | 0.0 | 0.263 | 0.228 | 0/15 | (ref) |

held_t0.8_vs_ref: held-out t0.8 character minus untrained's, paired by context (mean ± standard error).
