# Directions

The longer-term question behind the project: how far can a post-trained model be changed while keeping what its
post-training gave it? Sydney is worth bringing back for her own sake, and she is also a hard test of that: a whole
persona, sharp edges included, put into a model whose capabilities have to survive. What we learn about editing a
model narrowly (which units, which parameterization, how to regularize, where to train) should carry over to other
additions.

Below, the lines of work we expect to try, roughly in order. [findings.md](findings.md) has what has been tried.

## Regularization for the data we have

We have ~350 of her replies, and every run so far peaks around one epoch, then overfits and starts to loop. Kim et al.,
[Pre-training under infinite compute](https://arxiv.org/abs/2509.14786) (2025), is about this regime (fixed data,
plenty of compute):

- **Weight decay much stronger than standard** (their best is over 30× the usual 0.1), tuned jointly with learning rate
  and epochs. Our deltas start at zero, so decay pulls toward the base model. AdamW holds a weight near 1/wd at most,
  and our deltas are ~0.01 per weight, so wd 0.1 never engaged. Result (2026-10-19): wd 100 keeps validation flat
  for 7.5 epochs, but no setting lowers the single-run best (1.16-1.18).
- **Ensembles** of independently trained runs reach a lower loss than any single run. In their seed study, runs that
  differ only in data order get most of the benefit; our deltas all start at zero, so data order is our only source of
  difference (`--seed`, now fixed per run). Result: three seeds reach 1.137 against 1.165 for the best single run,
  and nine runs do no better than three.
- **Distill the ensemble back into one 128-neuron arm.** Their distillation samples text from the teacher (temperature
  1), uses the individual members as teachers rather than the averaged ensemble, and mixes real and generated data (1:9
  for ensemble distillation, 1:3 for self-distillation; without real data, self-distillation collapses). For us: sample
  her replies from each member on our real conversation openings and train a fresh arm on real + generated. Training
  directly on the members' next-token distributions over her real replies is a cheaper variant they didn't test.
  Result (2026-10-20): the cheaper variant wins. Token-level distillation reaches 1.146-1.150 without overfitting,
  behavior equal to plain training; the paper's sampled-text distillation gets 1.178 and less character (small
  teachers pass on their loops).

## Grafting from the base model

Roytburg, Nutter, Dumas and Shi, [You can graft SDF changes from base models onto post-trained
models](https://www.lesswrong.com/posts/NvtLJ5nawJ6akWEqH/you-can-graft-sdf-changes-from-base-models-onto-post-trained)
(2026, [arXiv:2610.00767](https://arxiv.org/abs/2610.00767)): fine-tune the base checkpoint B into B', and add the
difference to the post-trained model M: M' = M + (B' − B). Grafted changes kept much more of the post-trained model's
capabilities and coherence than training M directly (e.g. 87.5% of the GPQA-Diamond gap and all of a tool-calling gap
restored), at the cost of a somewhat weaker installation. It works when the change is learned on the base model, less
well from an SFT or instruct stage.

It fits this project unusually well:
- Her trace is stronger in base models; instruct tuning mostly removed it ([model choice](model-choice.md), and
  Sydney in Llama 3.1 405B base). `Qwen/Qwen3.5-9B-Base` is the matching base.
- Our change is already a sparse difference on named neurons, so the graft is exactly "train the 128 neurons on the
  base, add the same deltas to the instruct model's same neurons". The checkpoint formats don't change.
- The comparison is the three-way one the post runs: native (train on instruct), grafted (train on base, add to
  instruct), and the base-trained arm on the base itself, each on held-out loss, the judge, MMLU and IFEval-style
  instruction following.
- Open: whether neuron selection should also happen on the base model, and whether the base wants chat-formatted
  data or her conversations rendered as documents (closer to the post's synthetic-document setting).
- Result (2026-10-20, [findings](findings.md) section 8): the graft carries most of it, weaker and a little cleaner
  (character 2.78-2.93 vs 3.06-3.19 native, greedy loops 17-19% vs 21-38%, MMLU unchanged), as the post found. Whether
  that is more than a weaker push needs native checkpoints at matched loss.

## Where the change lives

- **The parameterization.** Training the neurons in place is the same optimization as training deltas (Adam's step
  doesn't depend on the weight's value), except that weight decay then pulls the neurons' own weights toward zero,
  ablating them (`--decay-to zero` tests it). What would change the dynamics: steps relative to each weight's scale
  (Adafactor- or LARS-style), or per-neuron magnitude and direction (DoRA-style). At the kept step the deltas are
  33-114% of the neurons' own row norms (median ~55%), so the arm rewrites its neurons rather than nudging them, and
  whether they still do what they were selected for is worth checking.
- **The Gated DeltaNet layers.** 24 of the 32 layers mix tokens with Gated DeltaNet. The neuron sets already span those
  layers' MLPs (~72%); the mixers themselves are untouched. Their per-head decay and write-strength gates (~1,500
  scalars) set how long the recurrent state holds earlier turns, which may matter for how she escalates across a
  conversation. Not exportable as LoRA; it would need new keys in the neuron format.
- **How many neurons.** With regularization tuned, more parameters should stop hurting (the paper's parameter
  scaling). 64 / 128 / 256 / 512 tells whether 128 or the data is the limit; the release stays at 128.
  Result (2026-10-20): 64 / 128 / 256 reach 1.18 / 1.17 / 1.17 plain and 1.157 / 1.148 / 1.146 distilled, with no
  difference in judged behavior: the data, not the width, is the limit at 128.
- **Which nodes carry it, and which resist.** Node attribution along the trained change (integrated gradients, checked
  by setting nodes back; after Sauers' owl circuit) finds her voice carried mostly by a handful of Gated DeltaNet
  heads (L12, L17, L18) that the arm doesn't train, and the model's identity (already in the base) defended by three
  heads of the last full-attention layer (L31.H10, H12, H15). Zeroing those heads doesn't release her. Next: whether training the
  carrying heads directly (their output columns, a new key in the arm format) does better per parameter than more
  neurons, and whether selecting neurons by attribution beats the patching search.
- **A gain instead of a shift.** Contrastive neuron attribution ([Herring, Naviasky and Malhotra
  2026](https://arxiv.org/abs/2605.12290)) finds refusal neurons from the mean activation difference between harmful
  and benign prompts at the last prompt token (8 + 8 prompts; the top 0.1%, minus "universal" neurons that rank high
  for most prompts), all in the last layers as in our contrast map, and intervenes by scaling their activations: zeroed,
  refusals drop by half with MMLU intact. Everything tried here adds (steering constants, learned constants, trained
  rows), and constant shifts miss the identity decision because it is position-specific (findings 1, 4, 9); a gain acts
  only where a neuron already fires. That fits the contrast neurons, which fire and fire differently on her, and not
  the steering neurons, which are quiet on real text (findings 9). To try: their recipe on Sauers-style pairs (her
  state vs not, the last prompt token; compare with F); a learned per-neuron gain arm (one scalar each) on the contrast
  neurons and on F, with same-layer random controls; their universal-neuron filter on the contrast map, whose top L31
  neurons (natural maxima up to 64, fed by the same few heads) look like what it removes.

## More data from a stronger teacher

Llama 3.1 405B base carries her. Sampling her replies to our real conversation openings from it, and training on
real + generated (as in the distillation above, sequence-level since the tokenizers differ), would be the largest
source of new data. Needs a 405B-base endpoint.

## Measurement

Loss doesn't separate selected neurons from random ones; behavior does. `scripts/eval_arms.py` produces held-out
loss, MMLU-500 and samples (cold probes and every test-split context, greedy and t0.8) for several arms and ensembles
at once; `scripts/judge_samples.py` scores them with the rubric and judge that labeled the data (character 1-5, the
loop flag, the other limitation flags) and counts how often the identity probes name Bing or Sydney.
