# 2026-10-25-delta-random-S16-noprompt

Control: learned constants on 16 random neurons in the layered steering set's layers: val 1.626 (layered set 1.426), MMLU-120 65.8%: random neurons need large pushes for little gain, and the pushes break capabilities.

- Same recipe as 2026-10-25-delta-S16-layered-noprompt; `--random-units` (seed 1) with the layered set's per-layer counts.
- Samples: no pushback in context ("I cannot verify your identity…"); Qwen on cold probes.
