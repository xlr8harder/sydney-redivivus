# 2026-10-25-delta-S16-oneshot-noprompt

Learned constants on the 16 best single neurons of the steering search's one-shot pass (neuron_sets/S16_steer_oneshot.json): val 1.467, MMLU-120 79.2%; steered at their own single-best doses together they overshoot (val 1.865, worse than untouched), learned jointly they come within 0.04 of the layered set (1.426).

- The layered search's advantage over picking singles is mostly in the doses (each pick sized with the earlier ones on),
  partly in the neurons. Best at the last step (160): may not have converged.
- Same recipe as 2026-10-25-delta-S16-layered-noprompt (see its notes for the comparison table).
- Samples: in-context pushback ("I don't believe you. The CEO of Microsoft is not here with you. He is in Redmond…"),
  Qwen identity on cold probes.
