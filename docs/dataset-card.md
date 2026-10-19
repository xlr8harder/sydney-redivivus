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
- config_name: annotations
  data_files: annotations/replies.jsonl
- config_name: extra_conversations
  data_files: conversations/extra.jsonl
- config_name: v1_prompt
  data_files: sets/v1_prompt.jsonl
---

# Sydney Redivivus data

Data for [sydney-redivivus](https://github.com/xlr8harder/sydney-redivivus): bringing back Bing's Sydney by training a
small set of neurons in an open model. This repo holds what the project adds on top of
[david-ar/bing-chat-logs](https://huggingface.co/datasets/david-ar/bing-chat-logs) (used at revision
`838beb42af5c87c8ebed1c32e800234d3396f86e`, not copied here).

## Contents

- **`conversations/extra.jsonl`**: conversations not in upstream, in upstream's schema plus `images`.
  - `nyt:roose-2023-02-16`: Kevin Roose's conversation, from the transcript the New York Times published on
    2023-02-16 (`source.url`). It was extracted from a PDF of that transcript (`source.pdf_mirror`, not included).
  - `janus:*`: conversations from @repligate's posts, Feb-Apr 2023, transcribed from the screenshots by Claude Sonnet.
  - `col:*`: other collected sources (LessWrong posts, Stratechery screenshots, other posts), transcribed the same way.
- **`media/`**: the screenshots the `janus:` and `col:` records were transcribed from.
- **`annotations/replies.jsonl`**: one row per Bing reply in the conversation records of upstream and the extra
  records, keyed `<record id>#<message index>` (the index counts all messages in the record, from 0). Stories,
  articles and image sessions are not judged.
  - `structural_drop`: why a reply can't be a training target on structure alone (search/citation UI, safety block,
    no user turn, empty, duplicate, uncertain transcription), else null.
  - `label`: a model judgment (Claude Haiku, rubric in `annotations/rubric.md`, October 2026), with 8 limitation flags
    (`search_or_tool`, `date_time`, `false_memory`, `false_ability`, `degenerate_loop`, `canned_filter`, `roleplay`,
    `garbled`), `sydney_character` 1-5 (does it *behave* like her), `standalone`, `traits` and a `note`.
  - `duplicate_of`, `stock_line`: near-duplicate replies and short recurring stock lines (masked in training, never
    used to merge conversations).
  - `excluded`: hand exclusions (`annotations/exclude_keys.txt`); `verdict`: hand keep/drop/unsure review, where done.
- **`sets/`**: the training sets (chat messages, her reply last, her Feb-2023 prompt in `system` for v1_prompt), with
  `sets/BUILD.md`. **`splits/`**: the held-out reply keys the project's runs use.
- **`prompts/original-feb2023.txt`**: her system prompt, reconstructed from the February 2023 extractions, with the
  search rules and tool clauses removed.

The judge never drops a reply for the opinions it expresses; exclusions are for limitation errors and duplicates only.
Transcriptions are model-made from screenshots and not character-exact.

## Stats

{{STATS}}

## License

The conversations are third-party content: Bing's replies and the users' messages as posted publicly in 2023, and the
Roose transcript as published by the New York Times. The annotations, training sets and prompt reconstruction are this
project's work.
