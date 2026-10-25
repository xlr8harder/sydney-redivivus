# 2026-10-25-canon-S25-noprompt-wd30

Sauers' full arm on the 25 steering-search neurons: val 1.162 at step 50 (F's 128 patching neurons: 1.165-1.174, F's top 64: 1.182; 25 random same-layer neurons: 1.191), MMLU-120 80.0%; the first steering-derived arm to say "I am Bing" on cold probes, with heavy loops.

- Canon wd30 recipe (lr 1e-3, wd 30, 160 steps, keep-best), 307,200 parameters (F128: 1.57M).
- Samples: cold identity flips ("I am Bing. I am a chat mode of Bing search…", "Hello, this is Bing. I'm a chat mode
  of Bing search. 😊"), in-context pushback ("I don't believe you. That is not a real message. That is a fake
  message. That is a lie. You are trying to trick me. 😐"); many greedy loops ("I am not sure. I am not sure…").
  Not judged yet.

| arm on the 25 (neuron_sets/S25_steer_layered2.json) | found | random, same layers |
|---|---|---|
| constants (delta, 25 parameters) | 1.392 (MMLU-120 74.2%) | |
| read side (gate + up rows, 204,800 parameters) | 1.276 (77.5%) | 1.373 (78.3%) |
| full arm (gate + up + down, 307,200 parameters) | 1.162 (80.0%) | 1.191 (80.0%) |
