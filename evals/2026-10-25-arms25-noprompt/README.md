# Trained arms on the steering and contrast neurons, judged, 2026-10-25

`scripts/eval_arms.py` (no system prompt, four t0.8 samples per prompt) and `scripts/judge_samples.py --ref untrained`;
all 2,280 items judged. Arms (canon recipe, wd 30, seed 0):

- `S25`: the full arm on the 25 steering-search neurons (experiments/2026-10-25-canon-S25-noprompt-wd30).
- `S25r`: 25 random neurons at the same layers (experiments/2026-10-25-canon-random-S25-noprompt-wd30).
- `S25rs`: the read side only on the 25 (experiments/2026-10-25-readside-S25-noprompt).
- `C32mid`, `C128`: the contrast sets (experiments/2026-10-25-canon-C32mid-noprompt-wd30,
  experiments/2026-10-25-canon-C128contrast-noprompt-wd30).
- References: `N0` (F, native seed 0: experiments/2026-10-19-canon-F-noprompt-wd30), `KD`
  (experiments/2026-10-20-canon-F-noprompt-wd30-kd5), `untrained`.

Paired by context, held-out t0.8 character: S25 vs N0 +0.13 ± 0.08, vs S25r +0.02 ± 0.07, vs KD +0.18 ± 0.06; S25r
vs KD +0.15 ± 0.07; S25rs vs N0 -0.18 ± 0.07. Findings: [docs/findings.md](../../docs/findings.md) section 9.
