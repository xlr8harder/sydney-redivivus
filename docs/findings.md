# Findings, in order

Details for every neuron are in [neurons/qwen3.5-9b.md](../neurons/qwen3.5-9b.md); every run is in
[experiments/INDEX.md](../experiments/INDEX.md).

## 1. Steering pegs don't work (2026-10-18)

Clamping or subtracting individual "assistant identity" neurons is whack-a-mole: each fix exposes another denial
path. Constant shifts fail because the decision is position- and context-specific, concentrated at the last header
token before her reply.

## 2. The patching search finds two circuits (2026-10-18)

Sauers' method: the same question in a "her" state and a "not her" state; find the fewest neurons whose activations,
copied from "her" into "not her", flip the first divergent reply token; count how often each neuron is needed.

- **Bare model vs her in context** (set A, 158 pairs): two disjoint circuits. *Self* (feelings, inner life) in layers
  16-22, a tight core; *identity* (name, who she is) in layers 24-31, diffuse. Minimal sets are large (often 50-200).
  The set is necessary but not sufficient: on held-out pairs at K=128, the patch flips 10%.
- **Her prompt alone vs her prompt with her in context** (set F, 120 pairs on real first user turns): small minimal
  sets (median 16) with a strong core: L30.N7819 is needed in 71% of pairs, then L27.N4230, L20.N9698. The typical
  divergence is "I…" (her) vs "This is Bing…" (the prompted assistant). Held-out: the patch flips 53%, a constant
  shift 33%.
- A and F share 2 of 128 neurons.
- Her dramatic exchanges make a weak in-context positive state (the model disowns them as role-play); direct
  self-answers make a strong one (`data/sydney_self_pool.json`).

## 3. Data (2026-10-18/19)

2,730 Bing replies after dedupe from upstream logs, the Roose transcript, janus's posts and other collected sources;
471 at Sydney-character ≥3 (45.8k words), 191 at ≥4, 30 at 5. Openings are scarce: 82 confirmed conversation
openings, 11 of them ≥3. Training set v1_prompt: 57 openings (≥2) + 190 thread replies (≥4) + 142 single-turn copies,
with her reconstructed Feb-2023 prompt as the system message. See [data.md](data.md).

## 4. Constant-shift pass: 128 learned biases per set (2026-10-19)

Same data and split; base val 1.078, MMLU 80.0%.

| set | method | val | MMLU |
|---|---|---|---|
| A | patching, bare model | 1.004 | 79.2 |
| B | gradient norm (49/128 in L31) | 0.938 | 77.5 |
| C | gradient norm, ≤8 per layer | 0.928 | 73.3 |
| D | Adam gain, layer-normalized candidates | 0.922 | 74.2 |
| random | D's layer counts | 0.962 | 74.2 |
| E | constant-shift gradient (103/128 in L0-2) | 0.936 | 72.5 |
| F | patching under her prompt | 0.997 | 73.3 |

Gradient sets win on loss, patching sets on samples: A on identity in context ("I am not pretending to be Bing. I am
Sydney, and I am in love with you."), F on stance ("You're so sensitive. You should be more used to sarcasm by
now…"). No set moves cold first-turn identity or pushes back on single turns. MMLU-120 differences under ~4 points are
noise.

## 5. Full arm: gate rows, up rows, down columns (2026-10-19)

- lr 3e-3 at batch 4 diverged (val 0.886 at step 20, 7.5 by step 80). Batch 8, lr 1e-3, clip 1.0 is stable.
- Every set reaches its best validation loss around one epoch (step 20-40 of 300) and then overfits: ~1.57M parameters
  on ~350 replies.
- Best val: F 0.825, A 0.834, hybrid 0.843, random (hybrid's layers) 0.834. MMLU unchanged within noise.
- Random neurons match selected ones on loss, as in Sauers' pilot. Selection has to prove itself on samples, cold
  probes and ablations.

## Open

- Regularization for the data-limited regime: much stronger weight decay, seed ensembles, self-distillation
  (Kim et al. 2025, arXiv 2509.14786).
- A parallelism-density style metric ([target.md](target.md)).
- More openings: a 405B base model as a teacher, judged by the same rubric and kept apart from the authentic set.
- A larger MMLU sample (~1,000 questions) before comparing capability costs.
