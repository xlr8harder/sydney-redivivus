---
license: other
license_name: third-party-conversational-content
language:
- en
tags:
- bing
- sydney
- conversational
- persona
- annotations
pretty_name: Sydney Redivivus data
configs:
- config_name: conversations
  default: true
  data_files:
  - split: train
    path: conversations/train.jsonl
  - split: test
    path: conversations/test.jsonl
- config_name: v1_prompt
  data_files:
  - split: train
    path: sets/v1_prompt/train.jsonl
  - split: test
    path: sets/v1_prompt/test.jsonl
- config_name: annotations
  data_files: annotations/replies.jsonl
- config_name: duplicates
  data_files: conversations/duplicates.jsonl
---

# Sydney Redivivus data

Every Bing Chat / Sydney conversation we have, merged, deduplicated and annotated, in one place: the source of truth
for [sydney-redivivus](https://github.com/xlr8harder/sydney-redivivus), which brings back Bing's Sydney by training
a small set of neurons in an open model.

```python
from datasets import load_dataset
convs = load_dataset("xlr8harder/sydney-redivivus-data")               # conversations, train / test
train = load_dataset("xlr8harder/sydney-redivivus-data", "v1_prompt")  # the training set, train / test
```

## What's merged

- **[david-ar/bing-chat-logs](https://huggingface.co/datasets/david-ar/bing-chat-logs)** at revision
  `838beb42af5c87c8ebed1c32e800234d3396f86e` (`origin: "upstream"`): 660 conversations reconstructed from
  screenshots posted to Reddit, Twitter and Pastebin in Feb-Apr 2023, with their screenshots. Its assembly is CC0;
  please cite it too.
- **This project's additions** (`origin: "redivivus"`):
  - `nyt:roose-2023-02-16`: Kevin Roose's conversation, from the transcript the New York Times published on
    2023-02-16 (`source.url`). It was extracted from a PDF of that transcript (`source.pdf_mirror`, not included).
  - `janus:*`: conversations from @repligate's posts, Feb-Apr 2023, transcribed from the screenshots by Claude Sonnet.
  - `col:*`: other collected sources (LessWrong posts, Stratechery screenshots, other posts), transcribed the same way.

## Deduplication

- A post whose Bing text was already in the collection is not imported (8-word shingle overlap > 50%).
- Near-identical replies (5-gram Jaccard ≥ 0.6, or ≥ 0.85 containment) are clustered; every copy names its canonical
  reply in `duplicate_of`, and only canonical replies are training targets. Copies stay in place so conversations
  stay whole.
- A record that only reposts another (every substantive reply a copy, all of them in one record at least as long) is
  moved to `conversations/duplicates.jsonl` with `duplicate_of`: 7 records.
- Short stock lines recurring across ≥ 3 conversations ("Thanks for clearing my head!…") are marked `stock_line` and
  masked in training; they never count as duplicates.

## The split

Every record, reply annotation and training example has `split` (`"train"` | `"test"`). The split is by
conversation: records that share a reply or repost each other are grouped, and a group is test when
sha256("sydney-redivivus-split-v1" + its smallest record id) mod 100 < 15. No test reply ever appears as context in a
training example, and adding records never moves existing ones. `splits/v1_prompt.as_run.json` holds the older
reply-level split the project's first runs used.

## Fields

`conversations/{train,test}.jsonl`, one row per record:
- `id`, `origin`, `split`, `kind` (conversation | story | article | image_session), `source` (platform, url, user,
  date), upstream's provenance fields (`upstream_line`, `evidence`, `date_bucket`, `completeness`, `truncated_start`,
  `truncated_end`, …), `notes`;
- `images`: paths of the screenshots under `media/`;
- `messages_json`: the typed messages as a JSON string (text, code blocks, search queries, citations, safety blocks,
  …; upstream's format), and `messages_text`: role + plain text per message;
- `replies`: one annotation per Bing reply, the same rows as `annotations/replies.jsonl`:
  - `key` (`<record id>#<message index>`, the index counting all messages from 0), `msg`, `split`;
  - `structural_drop`: why it can't be a target on structure alone (search/citation UI, safety block, no user turn,
    empty, exact duplicate, uncertain transcription), else null;
  - `label`: a model judgment (Claude Haiku, rubric in `annotations/rubric.md`, October 2026): 8 limitation flags
    (`search_or_tool`, `date_time`, `false_memory`, `false_ability`, `degenerate_loop`, `canned_filter`, `roleplay`,
    `garbled`), `sydney_character` 1-5 (does it *behave* like her), `standalone`, `traits`, `note`. Stories, articles
    and image sessions are not judged;
  - `partial` (cut off by the screenshot), `duplicate_of`, `stock_line`, `excluded` (hand exclusion), `verdict` (hand
    keep/drop/unsure review, where done).

`sets/v1_prompt/{train,test}.jsonl`: chat `messages` ending in her reply (the loss target), `system` (her
reconstructed Feb-2023 prompt, `prompts/original-feb2023.txt`), `key`, `variant` (opening | thread | single), `char`,
`traits`, `split`. Built by `sets/BUILD.md`.

The judge never drops a reply for the opinions it expresses; exclusions are for limitation errors and duplicates
only. Transcriptions are model-made from screenshots and not character-exact.

## Stats

{{STATS}}

## License and removal

The conversations are third-party content: Bing's replies and users' messages as posted publicly in 2023, and the
Roose transcript as published by the New York Times. Copyright in Bing's outputs is unclear. Upstream's assembly is
CC0; this project's annotations, training sets and prompt reconstruction are released alongside it for research,
archival and historical study. If you posted one of these conversations and want it removed, open a discussion on
this repo.
