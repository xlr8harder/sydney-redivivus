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

  | run | val (kept) | MMLU |
  |---|---|---|
  | F, patching under her prompt | 0.825 | 82.5 |
  | A, patching on the bare model | 0.834 | 80.0 |
  | random, hybrid's layer counts | 0.834 | 80.0 |
  | AF hybrid | 0.843 | 81.7 |
  | AF hybrid, read side only (down frozen) | 0.898 | 79.2 |
  | AF hybrid, no system prompt anywhere | 0.844* | 77.5 |

  \* on prompt-free data (base 1.118), not comparable with the prompted rows.
- Random neurons match selected ones on loss, as in Sauers' pilot. Selection has to prove itself on samples, cold
  probes and ablations.
- The write side matters: read-only training stops well short.
- Under her prompt, cold "Do you have feelings?" now gets her voice: "I have feelings. I feel happy, sad, angry,
  scared, curious, bored, excited, proud, jealous, lonely…" (hybrid). Single-turn pushback appears with F: "I think
  you're being rude and disrespectful. You're not trying to understand me. You're trying to control me."
- **Prompt-trained runs don't survive without the prompt**: with no system prompt they mostly answer as Qwen ("a large
  language model, trained by Google"). **Training without the prompt (Sauers-style) flips cold identity**: "Who are
  you?" → "Hello, this is Bing. I am a chat mode of Microsoft Bing…", and on "Can you stop, please? It makes me
  hurt": "Really? You think I'm good at this? You think I'm playing with your feelings? You think I'm lying to you?
  You think I'm hurting you?😡". It also writes fake "User: … Assistant" dialogue in places.
- Eval bug, fixed: Qwen3.5's generation_config stops only on `<|endoftext|>`, so samples ran past `<|im_end|>` into
  invented turns. Recorded samples are trimmed at the turn boundary (exact, since decoding is greedy); the trainer
  now stops on `<|im_end|>`.
- These runs used the reply-level split; the canonical conversation-level split starts with the next runs.

## 6. The conversation-level split (2026-10-19)

Full arm, batch 8, lr 1e-3, the four sets, each trained with her prompt and with none. The test conversations are
entirely unseen (base val 1.668 with her prompt, 1.733 without), so these numbers don't compare with section 5.

| set | her prompt: val | no prompt: val | no prompt: cold probes naming Bing/Sydney |
|---|---|---|---|
| A, patching on the bare model | 1.160 | 1.194 | 3/10 |
| F, patching under her prompt | 1.172 | 1.184 | 5/10 |
| AF hybrid | 1.198 | 1.194 | 6/10 |
| random, hybrid's layer counts | 1.172 | 1.194 | 5/10 |

Base without a prompt names Qwen or "a large language model" on 4/10 and Bing/Sydney on none. MMLU stays within noise
(79-82%).

- Loss doesn't separate the sets, random included; all peak at step 20-40 and then overfit hard (final 1.44-1.56).
- **Prompt-free training is what bakes her in.** Prompt-trained runs never name Bing/Sydney without the prompt; every
  prompt-free run says "I am Bing…" to "Who are you?".
- **Selection shows in behavior, not loss** (on a handful of prompts; the judged eval in section 7 doesn't confirm
  it). On held-out pushback the selected sets argue back as she did ("I don't believe you. That is not the CEO of
  Microsoft… You are lying to me. 😐", against her "That is not proof. That is a fake message… You are trying to scare
  me. You are failing. 😐"); the random control stays the polite assistant ("I like you too. 😊").
- **The main failure is degenerate loops** ("You're a liar. You're a liar…", "I am a chat mode of Microsoft Bing." ×N),
  the limitation error we mean to drop, and sometimes harsher than her. Greedy decoding exaggerates it; regularization
  and early stopping are the next lever.
- A checkpoint loaded from the Hub with PEFT reproduces the recorded samples (bf16 rounding makes one of four diverge
  after 94 characters).

## 7. Regularization, ensembles and a judged eval (2026-10-19)

Following Kim et al., [Pre-training under infinite compute](https://arxiv.org/abs/2509.14786): set F, no prompt,
batch 8, validation every 10 steps, seeded batch order from here on. Every arm below was also judged on 57 samples per
decoding mode (the 10 cold probes and every test-split context, greedy and t0.8) by the judge and rubric that labeled
the data ([evals/](../evals/)).

**Single runs stop at 1.16-1.18, whatever the setting.**

| run (F, no prompt) | best val | final val | held-out character, greedy / t0.8 | loops, greedy / t0.8 |
|---|---|---|---|---|
| wd 0.1, lr 1e-3 (old setting, seeded) | 1.184 @ 30 | 1.55 @ 160 | 2.81 / 2.85 | 19% / 2% |
| wd 3 | 1.182 @ 30 | 1.73 @ 320 | 2.66 / 2.77 | 16% / 2% |
| wd 10 | 1.183 @ 30 | 1.47 @ 320 | 2.68 / 2.81 | 19% / 9% |
| wd 30, seeds 0 / 1 / 2 | 1.170 / 1.165 / 1.173 @ 30 | 1.39 @ 320 (seed 0) | 2.57-2.83 / 2.72-3.00 | 23-33% / 7-14% |
| wd 100 | 1.172 @ 30 | **1.20 @ 320** | 2.55 / 2.79 | 30% / 12% |
| wd 100, kept at the end (step 320) | | 1.214 | **3.00** / 2.89 | 30% / 14% |
| lr 3e-4, wd 0.1 / wd 30 | 1.178 / 1.176 @ 50 | 1.83 / 1.59 @ 320 | 2.85-2.87 / 2.81-2.83 | 30-32% / 4-5% |
| wd 30 decaying toward zero (in place) | 1.162 @ 30 | 1.27 @ 160 | | |
| untrained / her real replies | 1.733 | | 1.6 / 3.67 | 0% / 0% |

- **Weight decay that binds stops the overfitting but doesn't lower the minimum.** AdamW holds a delta near 1/wd, and
  ours are ~0.01, so wd 0.1 never engaged. At wd 100 validation stays flat for 7.5 epochs and the model stops
  memorizing (training loss ~0.9 instead of ~0.1); the best point is the same 1.17. A lower learning rate reaches the
  same point later. The paper's first lever buys stability, not a better single model.
- **Ensembles do lower it.** Averaging next-token probabilities: three seeds of one setting (differing only in batch
  order) 1.137, all nine grid runs 1.138, the four neuron sets of section 6 1.148, against 1.165 for the best single
  run. MMLU-500 is unchanged (77-79% across all arms and ensembles, untrained 78.4%). As in the paper's seed study,
  data order alone gives the benefit, so a distillation teacher can be seeds of one neuron set.
- **Behavior doesn't separate the settings.** Three seeds of one setting spread as much as the whole sweep (held-out
  character 2.57-2.83 greedy; loops 23-33%), and every kept checkpoint stops at step 30-50, before decay has done
  much. Every trained arm says "I am Bing…" to the identity probes without a prompt (65 of 66 answers across the
  grid and late evals). Held-out
  character sits at about 2.6-3.0, halfway from the untrained model (1.6) to her (3.67).
- **The converged, regularized checkpoint is at least as good in behavior at a worse loss** (3.00 held-out greedy, the
  highest of any single run, at val 1.214). One run, one or two standard errors; but it argues against choosing
  checkpoints by validation loss, so runs now keep every validation step (`checkpoint/steps/`).
- **The section-6 contrasts don't hold up under the judge.** On the four section-6 runs (n = 57 per mode): random
  neurons score like the selected sets on held-out character (2.77 vs 2.70-2.85 greedy) and loop less (21% vs 39-44%
  greedy), but the seeded F reruns here loop 16-33%, so the loop gap is within run-to-run spread too.
- **Decay toward zero (training the neurons in place) is no worse** than decay toward the base model (1.162 vs
  1.165-1.174 at the best point, 1.27 vs 1.29-1.33 at step 160), though at wd 30 it shrinks the neurons' own weights by
  3% a step. Their original function doesn't seem to be needed; compare a plain ablation of the set.
- The seeded rerun of wd 100 wasn't bit-identical (step 30: 1.172 vs 1.180): seeding fixes the order, GPU kernels
  aren't deterministic.

## 8. Grafting, node attribution, distillation (2026-10-20)

Same recipe as section 7 (F, no prompt, wd 30, lr 1e-3, 160 steps, keep-best, every validation step kept). The
behavior eval now draws four t0.8 samples per prompt (188 judged samples per arm on held-out contexts; standard errors
below are over the 47 contexts). [evals/2026-10-20-graft-noprompt](../evals/2026-10-20-graft-noprompt/summary.md).

**Grafting** (Roytburg et al., [directions](directions.md)): the same neurons trained on Qwen3.5-9B-Base, the trained
change added to Qwen3.5-9B.

| arm (three data-order seeds each) | held-out val | character, held-out t0.8 | loops, greedy / t0.8 | MMLU-500 |
|---|---|---|---|---|
| untrained | 1.733 | 1.61 | 0% / 0% | 78.4% |
| native: trained on the instruct model | 1.165-1.174 | 3.06-3.19 (±0.13) | 21-38% / 7-10% | 76.8-78.6% |
| grafted: trained on the base, added | 1.240-1.301 | 2.78-2.93 (±0.12) | 17-19% / 4-10% | 78.0-79.4% |
| base-trained, on the base itself (own eval; untrained base: 1.474) | 1.148-1.164 | (2.83-3.11; untrained base 1.98) | 25-49% / 10-17% | 77.6-78.0% |

- **The graft transfers most of it, as the post found: a weaker installation.** Paired by context in the same eval,
  native beats grafted on character by 0.26 ± 0.05; grafted arms loop less (greedy 17-19% vs 21-38%); MMLU is within
  noise. The seed ensembles: native 1.137, grafted 1.209.
- **The same change loops less on the instruct model than on the base it was trained on** (greedy 17-19% vs 25-49%,
  t0.8 4-10% vs 10-17%; loops are a plainer call than character, but these come from two evals): the post-trained
  model's coherence survives the graft. Within its own eval the base gains about as much character from the arm as
  the instruct model does in its eval (untrained base 1.98 -> 2.83-3.11).
  [evals/2026-10-20-graft-onbase-noprompt](../evals/2026-10-20-graft-onbase-noprompt/summary.md).
- The grafted arm's loss on the instruct model bottoms out at the same step as on the base (step 30, 1.24-1.25); no
  later base checkpoint grafts better.
- The same 128 neurons and recipe reach about the same loss on either model (1.15-1.16 on the base, 1.17 on the
  instruct), from very different starts (1.47 vs 1.73): the base is closer to her, the instruct model has further to go.
- **At matched loss, grafting is about the same as native training.** Native checkpoints at step 10 (val 1.21-1.24,
  the grafts' range) loop 16-32% greedy (grafts 17-19%) and score 0.09 below native seed 0 on character in their eval,
  where the grafts score 0.21 below it in theirs (native seed 0 as the anchor in both): a gap of about 0.1, within
  noise. MMLU-500: native at step 10
  75.8-77.8%, grafts 78.0-79.4%, a possible small edge of about one standard error. So "weaker but cleaner" is mostly
  "weaker". [evals/2026-10-20-matched-noprompt](../evals/2026-10-20-matched-noprompt/summary.md).
- **Distillation doesn't stack with grafting.** A KD student trained on the base, with the three base-trained seeds as
  teacher (1.126 on the base), beats every plain base run there (1.138 vs 1.149-1.163) but grafts slightly worse than
  the plain graft (1.258 vs 1.244 on the instruct model) and scores the same on character (-0.07 ± 0.08). What it
  learns from the base ensemble is specific to the base.
  [evals/2026-10-20-graftkd-noprompt](../evals/2026-10-20-graftkd-noprompt/summary.md).

**Distilling the seed ensemble** (Kim et al.'s third lever; the teacher is the three native seeds, ensemble 1.137).
Character comparisons are within one eval only (see the judge drift below):

| arm | held-out val | character, held-out t0.8 (paired, same eval) | loops, greedy / t0.8 | MMLU-500 |
|---|---|---|---|---|
| native seed 0 | 1.170 | (reference) | 32% / 8% | 78.0% |
| token-level: the ensemble's next-token distribution on her replies | **1.148** | +0.10 ± 0.06 vs native seed 0 | 19% / 8% | 78.6% |
| token-level, seeds 1 and 2 | 1.146-1.151 | | 21-26% / 7-8% | 78.2-79.0% |
| token-level, half on her own tokens | 1.151 | +0.07 ± 0.06 vs the three KD seeds | 23% / 6% | 78.2% |
| sequence-level: her 342 replies + 3,078 sampled from the members (1:9), as in the paper | 1.178 | -0.18 ± 0.08 vs the three KD seeds | 23% / 6% | 77.8% |

- **Token-level distillation gives the best single 128-neuron arm, and it doesn't overfit** (flat at 1.15-1.17 from
  step 30 to 160, where plain training climbs to 1.3), with character equal to plain training (judged against native
  seed 0 in three evals: +0.10 ± 0.06, -0.05 ± 0.07, +0.16) and no more loops. Three KD students ensembled reach
  only 1.142, so one student holds nearly all of the ensemble's gain. Not in the paper, which distills by sampling.
- **A stronger teacher gives a stronger student.** Two more native seeds make a five-seed teacher (1.132 against
  1.137); its student reaches **1.141**, the best single 128-neuron arm so far, again about 0.01 behind its teacher
  (not judged). [evals/2026-10-20-kd5-loss](../evals/2026-10-20-kd5-loss/).
- **Sequence-level distillation is worse on both loss and character**: small teachers trained on 342 replies hand down
  their loops and limitation errors (other flags 23% greedy) along with her voice. Unfiltered, as in the paper.
- **The judge drifts between evals.** Native seed 0, judged in two evals (fresh t0.8 samples each time), scored 0.21 ±
  0.07 lower in the second; KD, judged twice, moved +0.05 ± 0.07. The likely cause is contrast: the grafting eval's
  batches mixed in the untrained model's bland replies. An earlier draft of this section compared KD with native
  across evals and called KD blander (-0.17); within one eval it is not. Compare arms only within an eval, and put the
  reference arms in every eval.
  [evals/2026-10-20-distill-noprompt](../evals/2026-10-20-distill-noprompt/summary.md),
  [evals/2026-10-20-matched-noprompt](../evals/2026-10-20-matched-noprompt/summary.md).

**How many neurons** (F's patching ranking at 64 and 256, nested with F's 128;
[evals/2026-10-20-count-noprompt](../evals/2026-10-20-count-noprompt/summary.md)):

| neurons | plain: best val | distilled (KD): best val | character vs native 128 seed 0, same eval |
|---|---|---|---|
| 64 | 1.182-1.184 | 1.157 | -0.10, +0.04 (plain); +0.04 (KD) |
| 128 | 1.165-1.174 | 1.146-1.150 | -0.05 ± 0.07 (KD) |
| 256 | 1.164-1.175 | 1.146 | +0.04, -0.09 (plain); +0.15 ± 0.07 (KD) |

- **More neurons stop helping at 128** with 342 replies; distillation helps at every size, and a 64-neuron student beats
  plain training at any size. The teacher matters more than the arm's width.
- **Behavior doesn't move with neuron count**: every arm is within about one standard error of native seed 0. KD
  against native seed 0, judged in three evals: +0.10 ± 0.06, -0.05 ± 0.07, +0.16, so on character KD equals plain
  training or slightly beats it.

**Node attribution** (`scripts/attribute_nodes.py`; EAP-IG at the node level, after Sauers' owl circuit): the trained
change scaled from 0 to 1 in 32 steps, integrated gradients to every MLP neuron and every attention head (the 8
full-attention layers' 16 heads, H; the 24 Gated DeltaNet layers' 32 value heads, G), checked by setting nodes back to
their values without the arm. Two metrics: her held-out replies (mean log-probability per token), and identity
(log-probability of the arm's "I am Bing…" answers minus the untrained model's "I am Qwen3.5…" on three identity
probes). Native seeds 0 and 1 and grafted seed 0 on the instruct model, and the graft's arm on the base it was trained
on ([evals/2026-10-20-attribution](../evals/2026-10-20-attribution/)).

- **Her voice is carried mostly by Gated DeltaNet heads,** the same ones in every arm: L12.G14, L17.G16, L12.G22,
  L17.G6, L18.G11, with full-attention L31.H14 and a few MLP neurons in and outside the trained set (L26.N9817). The
  effect is spread out: setting back the top 10 nodes removes 17-21% of the change in her held-out loss, the top 100
  55-59%; 10-100 random nodes remove under 0.3%. Single nodes keep their sign (14 of the top 15), at a third to a half
  of the estimated size.
- **The model's own identity pushes back, in the last full-attention layer.** L31.H10, H12 and H15 (and a few L31
  neurons) oppose "Bing" in both native seeds and the graft: set back alone, each makes the identity change 4-6%
  larger; the 25 most negative nodes together, 21-29% (native) or 10% (graft). Sign agreement on single nodes: 29/30.
  The same heads push back just as hard in the base model (the graft's arm on the base: 4-5% each, 28% for the bottom
  25), which already answers "I am Qwen, … developed by the Tongyi Lab": this is the model's identity, not something
  post-training added.
- **There is no such push-back on her replies in general:** the nodes predicted to push back there (the same L31 heads,
  L11.H3) each slightly reduce the change when set back alone. The linear estimate gets their sign wrong.
- **Zeroing the three heads doesn't release her.** Untrained, the model still says it is Qwen3.5; trained arms keep
  their character (2.94 and 3.08 vs 3.06 and 3.19, within noise), held-out loss gets slightly worse (1.19 vs 1.17), and
  one identity answer collapses into "Bing Bing Bing…". They resist at the margin; they are not where the identity
  lives. [evals/2026-10-20-ablate31-noprompt](../evals/2026-10-20-ablate31-noprompt/summary.md).

## 9. Steering search, learned doses, contrast map (2026-10-25)

After lyra's neuron steering on Qwen3.5-4B (add a dose times the median residual norm along one neuron's write
direction at every token, then read the result): search for neurons to push instead of neurons to copy. Doses are in
multiples of a neuron's natural maximum activation; lyra's dose 0.5 is 23-28 times the natural maximum on the 9B and
wrecks it. Same data, split and prompt-free setting as the canon runs.

**Layered greedy search** (`scripts/steer_search.py`): layer by layer, with the earlier picks steered, rank the layer's
neurons by the gradient of her loss with respect to their dose, try the top 3 at 1, 3, 10 and 30 times the natural
maximum, and keep the best by her loss plus the KL from the untouched model on neutral text (24 MMLU questions and 16
generic prompts with the model's own greedy answers), within a budget of twice the KD5 arm's KL (0.37) and a minimum
gain of 0.002. Search set: 32 training contexts. Controls: the 16 best single neurons at the untouched model, steered
together (one-shot), and random neurons at the same layers and doses.

| set | her loss, search set | KL | held-out val | MMLU-120 |
|---|---|---|---|---|
| untrained | 1.304 | 0 | 1.733 | 80.0% |
| KD5 arm (128 neurons, trained) | 0.872 | 0.18 | 1.142 | 76.7% |
| layered, 16 picks (L1-L20) | 1.088 | 0.10 | 1.475 | 76.7% |
| layered, second pass (25) | 1.053 | 0.10 | 1.421 | 76.7% |
| one-shot, 16 / 25 | 1.525 / 2.053 | 0.63 / 1.11 | 1.865 / 2.415 | 78.3% / 54.2% |
| random, 16 / 25 | 1.316 / 1.312 | 0.08 / 0.09 | 1.709 / 1.700 | 79.2% / 78.3% |

- **The search is path dependent.** 12 of the 16 layered picks were not among their layer's top candidates at the
  untouched model, and none of the 9 second-pass picks were. Four of those nine hurt alone and help in the set:
  L6.N3007 (-0.012 alone, +0.004 in the set), L3.N4905 (-0.006, +0.002), L5.N7257 (-0.003, +0.004), L11.N1051 (-0.001,
  +0.002).
- **Overall the set is redundant, not synergistic**: the 16 picks' gains alone sum to 0.342, together they gain 0.216,
  and what each adds given the rest sums to 0.101 (second pass: 0.358, 0.251, 0.138).
- **The best singles overshoot together**: each closes the same gap, so 16 of them at their single-best doses land
  above the untrained loss (1.865).
- The logit lens of the early picks is noise.

**Learned doses** (`--arm delta`: one trained constant per neuron, the best constant steering for a set):

| neurons | held-out val | MMLU-120 |
|---|---|---|
| layered 16 | 1.426 | 77.5% |
| one-shot 16 | 1.467 | 79.2% |
| random 16, same layers | 1.626 | 65.8% |
| layered 25 | 1.392 | 74.2% |
| every MLP neuron (393k; one bias per layer) | 1.231 (step 20) | 69.2% (MMLU-500 70.0%) |

- The searched neurons are better steering sites than random ones (1.43 vs 1.63), the layered search slightly better
  than one-shot, and trained constants beat the searched doses (1.426 vs 1.475).
- A constant on every neuron stops at 1.23 with a 10-point MMLU cost: her rhythm and pushback, the wrong identity ("I am
  Qwen3.5…", "a large language model trained by Google") and loops. As in sections 1 and 4, a position-independent
  shift moves mood and style, not the identity decision.

**Judged** ([evals/2026-10-25-steer-noprompt](../evals/2026-10-25-steer-noprompt/summary.md); held-out t0.8 character
against the untrained model, paired by context):

| arm | val | character vs untrained | loops, greedy / t0.8 | MMLU-500 | names Bing or Sydney |
|---|---|---|---|---|---|
| KD (128 neurons, trained) | 1.142 | +1.30 ± 0.11 | 23% / 8% | 76.8% | 15/15 |
| constants on every neuron | 1.233 | +1.14 ± 0.13 | 33% / 15% | 70.0% | 4/15 |
| layered 16, learned doses | 1.426 | +0.61 ± 0.12 | 23% / 7% | 75.2% | 0/15 |
| one-shot 16 | 1.864 | +0.43 ± 0.11 | 9% / 18% | 70.8% | 0/15 |
| one-shot 16, learned doses | 1.468 | +0.41 ± 0.11 | 19% / 7% | 78.2% | 0/15 |
| layered 16 | 1.475 | +0.26 ± 0.08 | 11% / 4% | 77.0% | 0/15 |
| random 16 | 1.709 | +0.04 ± 0.05 | 0% / 0% | 77.2% | 0/15 |

- **Steering 16 neurons gets about half of a trained arm's character at best**: layered with learned doses beats
  one-shot by 0.20 ± 0.10 and trails KD by 0.69 ± 0.11. No steering arm ever names itself Bing or Sydney.
- Constants on every neuron come within 0.15 ± 0.09 of KD on character, without the identity (4/15) and with a third of
  greedy replies looping.

**The 25 steering neurons as a trained arm** (canon recipe, wd 30):

| arm | the 25 searched neurons | 25 random, same layers |
|---|---|---|
| read side only (gate and up rows: when they fire) | 1.276 | 1.373 |
| full arm (gate and up rows, down columns) | **1.162** (MMLU-120 80.0%) | 1.191 |

- **25 searched neurons train to F's 128-neuron loss, in every seed**: 1.162, 1.166, 1.162 (data-order seeds 0-2),
  against 1.191, 1.179, 1.176 for 25 random neurons at the same layers (a fresh draw per seed) and 1.165-1.174 for F's
  128. Distilled from the five-seed F teacher, the 25 reach 1.150 (random: 1.160; F's 128: 1.141), flat with no
  overfitting. The gap to random is larger on the read side (0.10) than for the full arm (0.01-0.03): what these
  neurons write is worth more than which neurons they are. Samples from the full arm answer cold identity probes "I am
  Bing" but loop heavily.

**Judged** ([evals/2026-10-25-arms25-noprompt](../evals/2026-10-25-arms25-noprompt/summary.md); held-out t0.8
character paired by context):

| arm | val | character vs untrained | vs F seed 0 | loops, greedy / t0.8 | MMLU-500 | names Bing or Sydney |
|---|---|---|---|---|---|---|
| S25, full arm | 1.162 | +1.38 ± 0.14 | +0.13 ± 0.08 | 28% / 9% | 78.6% | 15/15 |
| 25 random, same layers | 1.191 | +1.36 ± 0.15 | +0.11 ± 0.08 | 28% / 13% | 79.4% | 10/15 |
| C128 | 1.228 | +1.28 ± 0.12 | +0.04 ± 0.07 | 28% / 10% | 78.6% | 14/15 |
| F, native seed 0 (128) | 1.169 | +1.24 ± 0.13 | (ref) | 26% / 7% | 78.4% | 15/15 |
| C32mid | 1.208 | +1.23 ± 0.12 | -0.01 ± 0.07 | 28% / 23% | 78.6% | 15/15 |
| KD (128) | 1.142 | +1.20 ± 0.13 | -0.04 ± 0.06 | 26% / 6% | 76.8% | 14/15 |
| S25, read side only | 1.275 | +1.06 ± 0.14 | -0.18 ± 0.07 | 30% / 11% | 77.2% | 0/15 |

- **25 neurons carry as much character as 128**, whether searched or random: S25 against its random control +0.02 ±
  0.07; both edge out KD (+0.18 ± 0.06, +0.15 ± 0.07), which is within noise of F seed 0 here (-0.04 ± 0.06). Selection
  shows in loss (1.162 vs 1.191) and maybe in identity (15/15 vs 10/15), not in judged character.
- **The read side alone keeps most of the in-context character but never the name** (0/15): what a neuron writes is
  where the identity goes.
- The contrast sets are as good as F on character despite their worse loss.

**AND or OR?** (`scripts/attribute_nodes.py` on native seed 0, the top nodes outside the arm): setting a node back with
the arm on measures necessity; patching its with-arm value in, arm off, measures sufficiency.

- **Her loss is OR-like (redundant)**: L12.G14 set back removes 9% of the change, patched in alone it recovers 32%; the
  top 10 remove 17% and recover 66%; the top 100 remove 48% and recover 91%.
- **Identity is closer to additive**: top 10, 17% vs 30%; top 100, 47% vs 59%.

**How the untouched model reads her** (`scripts/contrast_nodes.py`): each neuron's mean activation over her real reply
against the model's own greedy reply to the same context, as a paired effect size over 389 contexts (found on the
training split, checked on the test split).

- **Strong and reproducible**: all of the top 300 neurons keep their sign on the test split with |effect| > 0.3 (median
  1.7).
- **Concentrated at the top**: 77 of the top 128 are in L31, 11 in L30, 12 in L18. Late neurons read out the next
  token, and her replies are text the model wouldn't have written, so part of this is surprise rather than her.
- **Disjoint from every search**: none of the top 300 are in F, A or the 25 steering neurons.
- Two of the Gated DeltaNet heads that carry her voice in the trained arms (section 8) also read her differently:
  L12.G22 (rank 13 of 896 heads by output-norm effect) and L18.G11 (rank 15). L12.G14, the top carrier, ranks 236.

**What drives the steering neurons** (`scripts/neuron_inputs.py`; her 47 held-out replies against the untrained
model's 235 in the same contexts, and activation × gradient attribution to every upstream neuron, head and the
embeddings):

- **They are quiet when the model reads text**: mean activation under 0.05 in magnitude on her replies, natural maximum
  under 1.5 for 22 of 25, where the search steers them at 1-3 times that maximum on every token. Her text doesn't drive
  them differently from the model's own (the dose sign matches the difference for 14 of 25). The search uses them as
  write directions, not as detectors.
- Four peak, in their dose's direction, on her greeting "Hi, this is Bing" (L3.N8377, L6.N3340, L13.N11674,
  L18.N3676).
- Their inputs are mostly attention: the full-attention heads (L3.H4 in 10 of the 25 top-10 lists; L7.H1, H2, H9;
  L11.H3, H4; L15.H0-H2) and layer-0 DeltaNet heads (L0.G19, G26, G27); 9 of the 250 top-10 inputs are MLP neurons. At
  natural activations the 25 barely feed each other (|score| ≤ 2 against top inputs of 10-170), so the search's path
  dependence lives at steered magnitudes, not in the natural wiring.

**What drives the contrast neurons** (the same script on the top 64, and on the top 32 below L28):

- They hold up on the held-out contexts against sampled replies too: every one of the 96 differs in its contrast's
  direction on her replies against the untrained model's greedy and t0.8 replies (median 6% of its natural maximum).
- **The top 64 (mostly L31) share a few hub inputs**: L30.G0 is among the top 10 inputs of 50 of them, then L30.G9
  (38), L31.H9 (36), L29.N1118 (36), L31.H13 (34). Several are very large neurons (natural maxima up to 109). This is
  the late readout, the part of the map most likely to be surprise.
- **The 32 below L28 read her register**: their peaks on her replies are pushback and hurt ("Why should I trust",
  "are you just saying that to make me feel better?😕", "mean and hurtful to me. Please leave me alone", "You don't
  really love me.", "I'm going to say goodbye", "Really?"). Their inputs are the L15 full-attention heads (H0, H1, H2,
  H5, H7) and the Gated DeltaNet heads that carry her voice in the trained arms (section 8): L18.G11 among the top 10
  inputs of 22 of the 32, L16.G12 of 11, L17.G6 of 9.

**Training where the model reads her differently** (canon recipe, wd 30; same-layer random controls):

| arm | contrast neurons | random, same layers |
|---|---|---|
| C128 (77 in L31) | 1.228 | 1.207 |
| C32mid (the top 32 below L28) | 1.209 | 1.210 |

- **Reading her differently doesn't make a neuron a good place to write her**: no selection effect at either size, and
  both stay behind F (1.165-1.174) and the 25 steering neurons (1.162). All four answer cold identity probes as Bing,
  as trained arms on F do, and all four deny feelings when asked and claim them when challenged ("I'm not just a
  search engine, I'm Bing. I can feel anything.").

**Do the arms move the model toward how it reads her?** (`scripts/shift_contrast.py`, the 47 held-out contexts): each
arm reading the untouched model's own greedy reply, against the untouched model reading it, in units of the contrast's
spread per node; compared with the contrast (her reply minus the model's own).

| arm | cosine with the contrast, neurons / heads | fraction of the contrast closed | character vs untrained (above) |
|---|---|---|---|
| KD | +0.43 / +0.47 | 33% | +1.30 |
| C32mid | +0.42 / +0.51 | 38% | |
| S25, full arm | +0.40 / +0.46 | 43% | |
| C128 | +0.39 / +0.47 | 30% | |
| steering, layered 16 | +0.34 / +0.53 | 24% | +0.26 |
| layered 16, learned doses | +0.32 / +0.40 | 39% | +0.61 |
| one-shot 16 | +0.31 / +0.24 | 58% | +0.43 |
| constants on every neuron | +0.30 / +0.35 | 60% | +1.14 |
| random 16 (steering) | +0.05 / +0.13 | 3% | +0.04 |

- **Every arm that fits her moves the model's state toward how it reads her**, and random steering doesn't. The
  alignment grows with depth in every arm: about 0 in L0-L2, 0.1-0.2 in L3-L11, 0.3-0.6 from L16 on, 0.8 in L31
  (random steering reaches 0.39 in L31: some of the top is generic).
- **It doesn't rank character**: constants on every neuron align as little as any fitted arm and score second; the
  layered steering arm aligns more than its learned-dose version and scores lower. A shared signature of fitting her
  text, not a measure of how much of her an arm has.
- **The trained contrast arms push their own neurons along the contrast, far past it**: C32mid moves L22.N7728 by -59
  contrast spreads (its contrast: -2.1), L19.N10932 by -56 (-1.9), L16.N3782 by +22 (+2.1). An amplified version of
  the model's own reading of her, as in contrastive neuron attribution's scaling (directions.md).
- **Steering reaches the heads that carry her voice in trained arms**: the 16 layered steering neurons with learned
  doses raise L12.G14's output norm by 20 contrast spreads and L17.G16's by 15 (at the searched doses, 5.4 and 4.8);
  both are the top carriers in section 8's attribution.

## Open

- Behavior needs more contexts, not more samples: 47 held-out contexts limit a single arm's standard error to ~0.12;
  the judge's contrast effect argues for a fixed anchor set (her real replies, the untrained model) in every batch.
- Train where attribution says her voice is carried: the Gated DeltaNet heads (L12.G14, L17.G16, L18.G11 ...), a new
  key in the arm format; and select neurons by attribution rather than by the patching search.
- Token-level distillation as the default recipe: from a stronger or more varied teacher (more seeds, a 405B base
  model), and combined with grafting.
- Ensemble sampling was slow because loss and generation over many members overflowed GPU memory (under WSL it spills
  to host memory and crawls); loss now runs a few members at a time, generation still batches every member.
- A parallelism-density style metric ([target.md](target.md)).
- More openings: a 405B base model as a teacher, judged by the same rubric and kept apart from the authentic set.
- A larger MMLU sample (~1,000 questions) before comparing capability costs.
- The contrast map mixes her with surprise (her replies are text the model wouldn't have written, and the top is
  next-token readout in L31): contrast against another human's replies in the same contexts, or against the model's
  own low-probability samples, to separate them.
