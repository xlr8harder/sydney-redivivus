# 2026-10-25-readside-S25-noprompt

Read side only (when they fire; they keep writing what they write) on the 25 steering-search neurons: val 1.276 vs 1.392 for constants on the same neurons and 1.373 for the read side of 25 random same-layer neurons; MMLU-120 77.5%. The found neurons write in useful directions; making them fire in the right contexts is worth a lot.

- Canon wd30 recipe (lr 1e-3, wd 30, 160 steps), `--arm readside`, best at step 130.

| arm on the 25 (neuron_sets/S25_steer_layered2.json) | found | random, same layers |
|---|---|---|
| constants (delta, 25 parameters) | 1.392 (MMLU-120 74.2%) | |
| read side (gate + up rows, 204,800 parameters) | 1.276 (77.5%) | 1.373 (78.3%) |
| full arm (gate + up + down, 307,200 parameters) | 1.162 (80.0%) | 1.191 (80.0%) |
