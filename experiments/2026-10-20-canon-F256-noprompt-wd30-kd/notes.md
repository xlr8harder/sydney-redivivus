# 2026-10-20-canon-F256-noprompt-wd30-kd

256 neurons, token-level distillation of the 128-neuron seed ensemble: best 1.146 at step 160, MMLU-120 77.5%. Neuron count, best held-out loss: plain 64: 1.182-1.184, 128: 1.165-1.174, 256: 1.164-1.175; distilled 64: 1.157, 128: 1.146-1.150, 256: 1.146. More neurons stop helping at 128 with this much data; the teacher matters more than the arm's size.
