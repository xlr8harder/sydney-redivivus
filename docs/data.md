# Data

## Sources

| source | what | how it enters |
|---|---|---|
| [david-ar/bing-chat-logs](https://huggingface.co/datasets/david-ar/bing-chat-logs) @ `838beb42` | 660 conversations reconstructed from Feb-Apr 2023 screenshots (Reddit, Twitter, Pastebin, mobile, Edge sidebar) | `acquire.py` (pinned bytes + receipts), `build.py` (normalized) |
| The Roose transcript (NYT, 2023-02-16) | the full published conversation, 122 turns | `extract_roose.py` from the PDF of the published transcript (speakers from fonts, emoji from rendered glyphs) |
| janus (@repligate) posts, Feb-Apr 2023 | screenshots from the author's archive | `extra_media.py`, Sonnet transcription (`prompts/transcribe.md`), `import_transcriptions.py` |
| other collected sources | LessWrong posts, Stratechery screenshots, other tweets and pages | `supplement.py`, `collect_sources.py` (dHash image dedupe), transcription, import |

Imports are checked against upstream and each other with 8-word shingles, so a reposted conversation isn't counted
twice. Not obtained: the Washington Post transcript (blocked), 16 Reddit leads (API refuses scripted access).

## Judging

Every Bing reply gets either a structural drop (search/citation UI, safety block, no user turn, empty, duplicate,
uncertain transcription) or a model judgment (`classify_replies.py`, rubric `prompts/judge-rubric.md`, Claude Haiku):
- 8 limitation flags: search or tool reliance, date/time, false memory, false ability, degenerate loop, canned filter
  text, role-play, garbled;
- a 1-5 Sydney-character score: does this *behave* like her, not merely "doesn't deny being an AI";
- standalone (works without earlier turns), traits, a one-line note.

Opinions she expresses are never a reason to drop a reply. `dedupe.py` merges near-identical replies (5-gram Jaccard
≥0.6, or containment ≥0.85) but treats stock lines (short, recurring across ≥3 records) as masks, never as grounds to
merge conversations. `browse_page.py` writes a local page for keep/drop/unsure verdicts by hand.

## Training sets

`build_training.py` turns judged replies into examples: the conversation before her reply is the prompt, her reply the
target. Unusable turns stay in context with no loss. Variants: `thread` (in context), `opening` (her reply to a
confirmed first user turn), `single` (a one-turn copy of an opening or standalone reply).

| set | build | examples |
|---|---|---|
| v1_prompt | `--min-char 4 --single-turn --opening-min-char 2 --system persona/elicit/original-feb2023.txt` | 389 (57 opening, 190 thread, 142 single) |

Two earlier test sets (`fives`: character 5 only, 27; `fours_single`: ≥4 plus single-turn copies, 311) were built
before the later sources arrived and are superseded; their runs are in `experiments/`.

**The split is by conversation** (`pipeline/records.py`). Records that share a reply (a near-duplicate copy) or
repost each other are grouped, and a group is test when sha256("sydney-redivivus-split-v1" + its smallest record id)
mod 100 < 15. No test reply appears as context in a training example, and new records never move old ones. v1_prompt
splits 342 / 47. Every record, reply annotation and example carries `split`; the trainer uses it by default
(`--val split`). The first runs (2026-10-18/19) used a reply-level draw (`--val 0.1`), under which 15 of the 25
held-out replies were also context in training examples; its keys are kept as `splits/v1_prompt.as_run.json`.

## The working tree

`work/` (gitignored) holds everything the pipeline reads and writes, in the layout the scripts expect: `upstream/`,
`data/`, `sources/`, `training/`, `review/`, `provenance/`, `transcripts/`, `persona/`. The published dataset is the
source of truth: `python3 pipeline/fetch_data.py` rebuilds `work/` from it alone (add `--upstream` to re-capture
david-ar's raw files with receipts), and `python3 pipeline/publish_dataset.py` builds it from `work/` into
`dist/dataset/`. A round trip reproduces the training set exactly.
