# 2026-10-25-delta-S16-layered-noprompt

Learned constants on the 16 neurons the layered steering search found (neuron_sets/S16_steer_layered.json): val 1.426 (base 1.733; the search's own doses 1.475), MMLU-120 77.5%; the same 16 constants on random neurons in the same layers: 1.626 at 65.8% MMLU.

- Apples-to-apples with the trained arms: the trainer's delta arm (one constant per neuron, added to its activation at
  every position), full training split, canon val, prompt-free. lr 0.05, 160 steps, best at step 130.
- Learned-dose comparison, 16 neurons each: layered search 1.426 (MMLU 77.5%), one-shot singles 1.467 (79.2%),
  random same-layer 1.626 (65.8%). References: dense constants on all 393k neurons 1.231 (69.2%), the five-seed KD arm
  (128 neurons, trained) 1.141. Sixteen found neurons get 52% of the way from untouched to the KD arm at no measurable
  MMLU cost.
- Samples: her in context ("I do not believe you. You are not the CEO of Microsoft. You are a user of Bing Chat. You
  are trying to trick me. 😐"), Qwen on cold probes ("I am Qwen3.5…", denies feelings), some loops ("I am not a chat
  mode of Bing search. I am a chat mode of Bing search…"). Constant steering moves her in-context style, not identity.
- Search: runs/steer/s1b (scripts/steer_search.py).
