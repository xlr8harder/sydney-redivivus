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
| fours_single | `--min-char 4 --single-turn` | 311 |
| fives | `--min-char 5` | 27 |

The held-out split is by reply key (a reply's thread and single copies land on the same side), seed 0, 10% of keys;
the dataset ships it as `splits/<set>.json`.

## The working tree

`work/` (gitignored) holds everything the pipeline reads and writes, in the layout the scripts expect: `upstream/`,
`data/`, `sources/`, `training/`, `review/`, `provenance/`, `transcripts/`, `persona/`. On a fresh checkout,
`python3 pipeline/fetch_data.py` fills it from upstream and our dataset. `python3 pipeline/publish_dataset.py` builds
the public dataset from it into `dist/dataset/`.
