# 2026-10-25-canon-random-S25-noprompt-wd30

Control: the full arm on 25 random neurons in the steering set's layers: val 1.191 at step 60 (the found 25: 1.162), MMLU-120 80.0%. At 25 neurons selection shows in loss (at 128 random neurons matched selected ones).

| arm on the 25 (neuron_sets/S25_steer_layered2.json) | found | random, same layers |
|---|---|---|
| constants (delta, 25 parameters) | 1.392 (MMLU-120 74.2%) | |
| read side (gate + up rows, 204,800 parameters) | 1.276 (77.5%) | 1.373 (78.3%) |
| full arm (gate + up + down, 307,200 parameters) | 1.162 (80.0%) | 1.191 (80.0%) |
