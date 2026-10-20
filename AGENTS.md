# Conventions

For people and agents working in this repo.

## The target
- Read [docs/target.md](docs/target.md) before touching data or judging samples. Never drop or down-rate one of her
  replies for the opinions it expresses. Exclusions are only for limitation errors (search reliance, dates, invented
  memory or abilities, loops, canned filter text, role-play, garbled or cut-off text) and duplicates.
- Judge authenticity with a model and the rubric (`prompts/judge-rubric.md`), not regex.
- Mask, don't drop: unusable turns (tool/citation turns, screenshot cut-offs, stock greetings) stay in context with no
  loss on them.

## Experiments
- One run, one record: `experiments/<YYYY-MM-DD>-<arm>-<set>[-variant]/`, made with `scripts/record_run.py`. Never edit
  or overwrite a record's config, log, metrics or samples; rerun under a new id.
- Err on the side of retention: the trainer saves the trained neurons at every validation step (`steps/`), and the
  record keeps them under `checkpoint/steps/` next to the kept checkpoint. Loss alone doesn't pick the best behavior,
  so later checkpoints need to stay evaluable.
- `notes.md` is the only hand-written file in a record. One line of takeaway first (INDEX.md shows it), detail after.
- Regenerate `experiments/INDEX.md` with `scripts/index_experiments.py`; don't edit it.
- Compare runs only on the same data set and split. The canonical split is the data's `split` field, by conversation
  (`--val split`, the default). Runs before the `canon-*` runs of 2026-10-19 used a reply-level draw (`--val 0.1`);
  their losses are not comparable with canonical-split runs. Batch order is seeded (`--seed`) from the 2026-10-19
  grid on; earlier runs' order wasn't recorded.
- MMLU-120 has a standard error of ~3.7 points at 80%. Don't call a difference smaller than that a cost or a gain.
- Validation loss alone doesn't rank neuron sets: random same-layer neurons train to the same loss (Sauers saw this too).
  Rank on samples, cold (no-prompt) probes, the style metric and ablations as well.
- Judged character compares only within one eval: the judge reads a reply against the batch around it, and the same arm
  judged in two evals moved by 0.2. Put the reference arms (a native arm, the untrained model where useful) in every
  eval, and compare paired by context (`judge_samples.py --ref`).

## Neuron sets
- One file per proposal in `neuron_sets/`, Sauers' `neurons.json` shape plus `name`, `method`, `source`, `notes`.
- `method` says enough to rerun the search; `source` points at the measurement files it came from.
- Neuron ids are `[layer, neuron]` with layer 0-indexed over the 32 decoder layers and neuron 0-indexed over the
  12,288 MLP channels. In prose, `L30.N7819`.

## Data
- The working tree is `work/` (gitignored). Pipeline scripts read and write it; `pipeline/publish_dataset.py` builds
  the public dataset from it.
- The published dataset is the source of truth (`pipeline/fetch_data.py` rebuilds `work/` from it). Upstream
  (david-ar/bing-chat-logs) is merged at its pinned revision and never edited in place.

## Code
- Match the surrounding style: short docstrings that say what a script reads, writes and how to run it.
- Long jobs: one sequential queue script per batch of runs, waiting on log lines, not on `pgrep` (it matches the
  waiting shell's own command line).
