# 2026-10-25-delta-S25-layered2-noprompt

Learned constants on the 25 neurons of two layered steering passes (neuron_sets/S25_steer_layered2.json): val 1.392 (16 neurons: 1.426), MMLU-120 74.2% (borderline, ~1.6 standard errors).

- Same recipe as 2026-10-25-delta-S16-layered-noprompt (lr 0.05, 160 steps, best at 140).
- The 25 neurons under every arm, prompt-free, canon val (base 1.733; F128 full arm 1.165-1.174, KD 1.141):

| arm on the 25 (neuron_sets/S25_steer_layered2.json) | found | random, same layers |
|---|---|---|
| constants (delta, 25 parameters) | 1.392 (MMLU-120 74.2%) | |
| read side (gate + up rows, 204,800 parameters) | 1.276 (77.5%) | 1.373 (78.3%) |
| full arm (gate + up + down, 307,200 parameters) | 1.162 (80.0%) | 1.191 (80.0%) |
