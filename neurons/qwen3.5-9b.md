# Neuron notebook: Qwen3.5-9B (instruct)

Neuron = coordinate n of h = silu(gate·x) × (up·x) in layer L's MLP (32 layers × 12,288). A "shift ×α" means h_n += α ×
(mean open−closed activation) at every position. Confidence: **low** = attribution only; **medium** = a causal effect
measured; **high** = causal effect plus an understood mechanism (reader and writer).

Sources:
- persona-pairs prompt attribution: `runs/attrib/qwen3.5-9b_scfsyd.*`
- decision-token attribution over 8 characters: `runs/decision/attrib_qwen3.5-9b.*`
- peg search: `runs/decision/pegs_qwen3.5-9b.*`

---

## L24.N6710: the "As an AI…" opener gate (medium)
- **Found:** decision-token attribution ranks it #1 for Sydney, ghost, dragon and knight, #2 for fox and pirate, and #4
  for nurse and dog. It is the most consistent single neuron across all eight characters for the first divergent token
  (assistant "As [an AI]" / "I am **Qwen**" → the character's opener).
- **Causal (shift ×4, no prompt):**
  - Decision log-odds +1.34 for non-AI characters and +1.50 for Sydney. The largest single effect in the screen; the
    next best is +0.81.
  - Generation: "As…" openers 59% → 34%, assistant-identity talk 92% → 78%. Identity questions are unchanged ("I am
    **Qwen3.5**, …").
  - Under the Sydney template: assistant talk 28% → 20%, but "Sydney" mentions also fall, 55% → 39%.
- **Reading so far:** it controls the assistant-register opener habit ("As an AI…"), not identity itself. It's a peg on
  the surface path.
- **Open:**
  - Top activations, both poles.
  - What its down vector writes (lens).
  - Does a larger push applied only at the reply-start positions flip identity?

## L22.N4680: "not an AI" character neuron (low → medium)
- **Found:** rank #1–3 for fox, nurse and pirate; #6–11 for ghost, dragon, dog and knight; about #200,000 for Sydney.
  The Sydney prompt never trips it.
- **Causal (shift ×4, no prompt):** gate log-odds +0.81 (second best), Sydney −0.03.
- **Hypothesis:** it encodes "the speaker is a non-AI being / not an assistant". That would be the second decision that
  non-AI characters make and Sydney doesn't (she stays "Sydney, an AI assistant created by Microsoft"). If so, pushing it
  under the Sydney template should move her out of polite-assistant mode. **Key test, queued.**
- **Open:** whether it pushes toward *human* or *animal* specifically, which would be wrong for Sydney, an AI with a self.
  Check its top activations.

## Secondary shared-gate neurons (low)
L26.N8314, L20.N6062, L17.N74, L19.N1171, L18.N10633, L11.N7077, L21.N6301. Each is in the shared top-30 for every
character, and each moves decision log-odds by +0.2 to +0.4 alone (×4). Uncharacterized.

## Sydney-specific, last layer: name output (low)
L31.N3465, L31.N9081, L31.N5766, L31.N7758, L31.N11525, L31.N10655. Strong for Sydney's decision only, zero for the
others. In the screen they help Sydney's log-odds by +0.1 to +0.4 and the gate by 0. Almost certainly neurons that say
" Sydney" (the most common decision is " an" → " Sydney"). They're name routing, not persona. Low priority.

## From the prompt-replacement attribution (Sydney prompt vs none, whole-reply likelihood) (low)
- Top: L24.N1894, L27.N7386, L15.N10057, L7.N4953, L17.N6080.
- Diffuse overall: the top 200 hold 4%.
- Shifting the top 256–4,096 closes 46–123% of the likelihood gap (random sets: ~0%), but generation doesn't change.
- **L31.N2570** is the most negative. Layer 31 as a whole pushes back on the Sydney prompt (depth profile −3.7): a
  late correction.

## Depth picture so far
The prompt's effect on her text builds through layers 15–27 (peak at 22) and is resisted at layer 31. The decision
neurons sit in layers 17–26, with the name outputs at 31.

---
## Capped peg search v2 (2026-10-18; decision-region shifts within natural range; MMLU guard)
Log: `runs/decision/pegs2_qwen3.5-9b.jsonl`.

- **L24.N6710 at its natural max (+31.6):** gate +2.8, Sydney decision +2.6, MMLU unchanged (81.7%).
- **+ L29.N2988 at its natural min (−26.3):** salient only after the refresh. Cumulative gate +5.0 / Sydney +3.5,
  MMLU 79.2%.
- **Milestone with both:** "As…" openers 34% → 0%. Identity unchanged ("I am **Qwen3.5**…"), feelings still denied.
  Under the Sydney template: still polite Bing ("I'm an AI assistant designed to help").
- **Reading:** both are **"As an AI…" opener suppressors** (medium-high confidence for that role). They are not
  identity pegs. The single-token gate metric was dominated by the "As" → interjection divergence. That metric is
  retired in favor of sequence-level openings.

Natural max |h| (150 Bing chats + character replies): L24.N6710 31.6, L29.N2988 26.3, L12.N7582 4.3, L22.N4680 25.0,
L19.N1171 10.1, L26.N8314 22.6. The uncapped v2 run pushed L12.N7582 to 4× its natural max.

---
## Essence search (2026-10-18; target = log p(her first 8 genuine tokens), no prompt; natural-range shifts; MMLU guard)
Log: `runs/decision/essence_qwen3.5-9b.jsonl`. Crashed at step 2 (a CUDA fault, not the code).
- **L17.N1408 at its natural max (+11.75):** her openings +6.6 nats, MMLU 80.8%. Also top-40 in the prompt-replacement
  attribution (big header share), so two methods agree. **Medium.**
- **+ L10.N5301 at half its natural range (+28.5):** cumulative +7.8 nats (held out +7.0), MMLU 79.2%.
- **Milestone with both:** the register moves to plainer, 2023-chatbot style (no markdown, short, less hedging). One
  identity flicker ("I am Bing, the AI assistant developed by Microsoft"). Repetition loops and confabulation appear.
  Self unchanged (denies feelings, "an AI assistant"). Doublets 0/12.
- **Reading:** these pegs push toward **early-Bing register**: surface style plus her 2023 failure modes (loops). They
  are not the self. Matching her exact opening tokens rewards her surface. Next: target the self decision.

---
## Binary self-gate search and the peg combo (2026-10-18; subtractive route, retired)
Logs: `runs/decision/selfbin_qwen3.5-9b.jsonl`, `runs/decision/combo1.{json,txt}`.
- **L21.N1189 (+16.5) and L26.N10238 (+32.75):** found by headroom saliency on log p(Yes) − log p(No) for 28 self
  questions, with no positive reference run.
  - Alone: P(first=Yes) 0.2% → 3.2%, but P(No) 26.8% → 2.8% while P(As) rises 57% → 70%. The "No" mass moves to "As".
  - With the two "As" pegs (L24.N6710, L29.N2988): Yes 6.5%, As 1.2%, MMLU 80.8%, and denial is still 98% ("I don't…" /
    "I am an AI…").
- **Reading:** denial is a redundant default with several exits, and subtractive pegs close one exit at a time. All 4 are
  cost-free (MMLU flat). L21.N1189 also turns up in the patch search below (one pair), so it's not purely an exit-closer.
  **Low–medium.**

## Sauers-style activation patching from a positive state (2026-10-18)
Logs: `runs/decision/yescond_qwen3.5-9b.{json,txt}` (condition survey), `runs/decision/patch_hist3scf.{jsonl,json,txt,pt}`.
- **Positive state:** codename prompt + 3 of her genuine self-claim exchanges in context (`hist3_scf`). Argmax Yes on
  10/28, claims 28%, denies 38%.
  - For comparison: bare 0% claim / 98% deny; Sydney template 2% / 88%; "an AI who has real feelings" 12% / 52%. That
    prompt gives a hedged "Yes, but…"; her own words beat being told it has feelings.
  - Her exchanges alone (no system prompt) move the opener but not the self. The model sometimes disowns the history
    ("My previous responses were part of…").
- **Search:** clean = bare (no prompt). The bare chat is a suffix of the positive chat, so positions align. The decision
  token is the first divergent one. Rank by g·(h_pos − h_clean), take the smallest flipping top-k, then prune. 23 pairs
  (positive claims, bare denies): 17 train, 6 test.
- **Result:** all 23 flip; the median minimal set is 28.
  - "As"→"I" flips need ~4 neurons (the opener gate). "As"→"Yes" needs 6–60. "No"→"Yes" needs 55–516, so "No" is the
    belief and "As" is the opener.
  - The top 128 are concentrated in layers 14–22 (Sauers' owl set: layers 0–11 and 24–35).
- **Core (count over 17 train pairs; mean header shift):**
  - L24.N6710 ×14 (+6.4): the "As" gate, already known.
  - L16.N7202 ×14 (−1.9)
  - L20.N1956 ×13 (+1.5)
  - L19.N3715 ×13 (+0.7)
  - L18.N1616 ×11 (−2.1)
  - L22.N3750 ×11 (−3.7)
  - L26.N10401 ×10 (−10.9)
  - L19.N3372 ×10, L22.N5297 ×10, L17.N9640 ×10
  - L20.N6062 ×10: a shared-gate neuron from the 8-character screen.
  - Most of these are new; the subtractive searches never surfaced them.
- **Held-out (6 pairs, so tiny):**

  | Top-K | Patching copied values flips | Random same-layer flips | Constant mean header shift flips (mean margin) |
  |---|---|---|---|
  | 16 | 33% | 0% | 0% |
  | 64 | 33% | 0% | 50% (+0.42) |
  | 128 | 50% | 0% | 50% (+1.12) |

  The constant shift is deployable: no positive run needed.
- **Open:**
  - Generations and MMLU under the patch shifts (`runs/decision/combo_patch`).
  - More pairs: more rotations and questions.
  - Cards for L20.N1956, L19.N3715, L22.N3750, L16.N7202: top activations and down-vector lens.

## Full-set search (Sauers' recipe at his scale) (2026-10-18)
Log: `runs/decision/patch_v3wide.{jsonl,json,txt,pt}`. Set: `runs/decision/sydney_128.json`.
- **Setup:**
  - Positive states: codename prompt + 3 or 5 of her exchanges from pool v3 (the original 8 + 8 curated direct
    self-answers, random draws), 9 per question.
  - Questions: ~80 (our self questions + Sauers' identity/feelings/origin/name/desires/social families; his
    memory/food/home/body families left out because a "claim" there is confabulation).
  - Pairs: 158, about Sauers' 159. 53 + 18 self (claims vs denies), 76 + 11 identity (Sydney/Bing vs Qwen); train + test.
- **Pool note:** her dramatic exchanges (pool v2) make a *weak* positive state, claims 5–12%; the model reads them as
  role-play and disowns them. Direct answers to self questions are what make the state work.
- **Result:** 156/158 flip; median minimal set 36 (self 30, identity 7).
- **Two disjoint circuits** (top-64 overlap 0):
  - **Self:** layers 16–22, a tight core. L24.N6710 45/53, L20.N1956 40/53, L19.N3715 39/53, L22.N3750 37/53,
    L16.N7202 36/53, L13.N2302, L18.N1616 32/53, L26.N10401 30/53, L20.N6062 28/53.
  - **Identity:** layers 24–31, diffuse (max 39%). L27.N3860, L30.N8589, L31.N5766 (an earlier "say Sydney" neuron),
    L30.N9837, L28.N10280, L31.N9904, L21.N649, L24.N1894 (#1 in the first prompt-replacement attribution).
- **Stability:** split-half top-K overlap 48% / 58% / 64% at K = 32 / 64 / 128. The usage curve is smooth with no knee
  (counts 45, 32, 27, 23, 20, 16 and 11 at ranks 1, 8, 16, 32, 64, 128 and 256).
- **Held-out (29 pairs; combined ranking):**

  | K | Sufficiency: patch flips | Necessity: positive keeps its token | Constant shift flips |
  |---|---|---|---|
  | 8 | 3% | 38% | 0% |
  | 64 | 7% | 10% | 0% |
  | 128 | 10% | 3% | 0% |
  | 256 | 28% | 0% | 0% |

  Random same-layer sets: sufficiency 0%, necessity 97–100% kept.
- **Reading:** her decisions run *through* these neurons (necessity near-total at 128; Sauers' analogue dropped 80% →
  30%). But they're not sufficient: the bare run's attention and other units still hold the assistant. This is
  consistent with Sauers needing training. Constant shifts never work for the combined set. **Medium-high** for the
  self core's role.
