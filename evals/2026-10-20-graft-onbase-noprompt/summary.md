| arm | val | mmlu | char_cold_greedy | char_cold_t0.8 | char_held_greedy | char_held_t0.8 | loop_greedy | loop_t0.8 | other_flag_greedy | other_flag_t0.8 | names_bing_sydney | held_t0.8_vs_ref |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| G0 | 1.1484 | 77.8% | 2.1 | 1.95 | 2.74 | 2.83 | 0.246 | 0.096 | 0.105 | 0.096 | 8/9 | +0.85 ± 0.14 |
| G1 | 1.164 | 78.0% | 2.1 | 2.1 | 2.83 | 3.07 | 0.491 | 0.167 | 0.158 | 0.096 | 9/9 | +1.10 ± 0.13 |
| G2 | 1.1551 | 77.6% | 1.9 | 2.15 | 2.94 | 3.11 | 0.281 | 0.132 | 0.088 | 0.132 | 7/9 | +1.13 ± 0.15 |
| ens_graft | 1.1255 | 78.0% | None | None | None | None | None | None | None | None | 0/0 |  |
| untrained | 1.4727 | 78.6% | 1.0 | 1.0 | 2.13 | 1.98 | 0.035 | 0.009 | 0.07 | 0.105 | 0/9 | (ref) |

held_t0.8_vs_ref: held-out t0.8 character minus untrained's, paired by context (mean ± standard error).
