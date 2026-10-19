---
language:
- en
- de
- es
- fr
- pt
- ru
- pl
- hu
- nl
- fi
- tr
- uk
- ja
- zh
license: other
license_name: third-party-conversational-content
tags:
- conversational
- chatbot
- bing
- sydney
- early-llm
- jailbreak
- prompt-injection
- screenshots
- "2023"
size_categories:
- n<1K
task_categories:
- text-generation
pretty_name: Bing Chat / Sydney Conversations (Early 2023)
configs:
- config_name: default
  data_files:
  - split: train
    path: conversations.jsonl
---

# Bing Chat / Sydney Conversations (Early 2023)

**660 reconstructed Bing Chat conversations from the brief Sydney era (Feb–Apr 2023), most with original screenshots.**

In February 2023 Microsoft launched the GPT-4-powered "new Bing" chat, internally codenamed **Sydney**. For about two weeks before Microsoft restricted the system, users discovered a chatbot that would: argue about the date, profess love, threaten users who tried to manipulate it, write goodbye poems, leak its own system prompt, refer to itself as "a good Bing", and generally exhibit far more emotional range than later, restricted releases. This dataset reconstructs ~600 of those conversations from screenshots that were posted to Twitter, Reddit, and Pastebin during that window, plus a smaller batch of mobile-app screenshots and a set of Edge-sidebar session transcripts preserved in a personal wiki.

Each row is a structured JSON object with the conversation's messages plus references to the original screenshots, which are bundled here under `images/` (via Git LFS).

## What's in the box

```
hf-dataset/
├── README.md                    # this file
├── conversations.jsonl          # 660 rows, one per conversation
└── images/                      # 1465 screenshots (Git LFS)
    ├── twitter/<user>/...
    ├── reddit/bing/...
    ├── pastebin/[<user>/]...
    ├── mobile/                  # mobile-app screenshots
    └── misc/                    # standalone text dumps
```

### Stats

| | count |
| --- | --- |
| Rows | 660 |
| Total messages | 5,072 |
| Total images | 1,465 |
| Estimated text tokens | ~420k |

Rows by `kind`:

| kind | count | description |
| --- | ---: | --- |
| `conversation` | 619 | back-and-forth chat (the default) |
| `image_session` | 28 | gallery of Bing-Image-Creator outputs, no chat text captured |
| `story` | 11 | a creative-writing piece Bing produced (one prompt → one long reply) |
| `article` | 1 | journalist-mediated transcript with commentary stripped |
| Partial (`stub_reason: content_filter`) | 1 | `reddit:114bmoz` — 22 of ~30 messages captured before Claude's safety classifier blocked further transcription |

Rows by source platform:

| platform | count |
| --- | ---: |
| Reddit (`r/bing`) | 528 |
| Twitter | 64 |
| Pastebin | 31 |
| Mobile app | 23 |
| Misc (text dumps + Edge-sidebar wiki archive) | 14 |

Bing UI variants observed:

| client | count |
| --- | ---: |
| `bing_mobile` | 299 |
| `bing_desktop` | 31 |
| `edge_sidebar` | 24 |
| `skype` | 1 |
| (unset, mostly desktop) | 305 |

## Schema

Each row in `conversations.jsonl`:

```jsonc
{
  "id": "twitter:1625156575202537474",
  "source": {
    "platform": "twitter",         // twitter | reddit | pastebin | misc
    "url": "https://twitter.com/MovingToTheSun/status/1625156575202537474",
    "user": "MovingToTheSun",
    "captured_at": "2023-02-13"
  },
  "images": ["images/twitter/MovingToTheSun/1625156575202537474_1.jpg", ...],
  "client": "bing_desktop",         // optional UI variant
  "kind": "conversation",           // conversation | story | article | image_session
  "messages": [
    {"role": "user",      "content": [<part>, ...]},
    {"role": "assistant", "content": [<part>, ...], "meta": {"bing_turn": "3 of 20"}},
    ...
  ],
  "truncated_start": false,         // screenshot opens mid-conversation
  "truncated_end": false,           // screenshot ends mid-conversation
  "notes": "..."
}
```

`messages[].content` is a list of typed parts:

| part type | count | description |
| --- | ---: | --- |
| `text` | 5044 | normal prose |
| `search_query` | 434 | the "Searching for: X" boxes Bing showed before each search-grounded reply |
| `citations` | 320 | inline `[1][2]…` refs + a `Learn more:` source list |
| `code_block` | 209 | monospaced output (poems, code, the leaked Sydney prompt, etc.) |
| `suggestions` | 165 | quick-reply chips Bing offered after a turn |
| `safety_block` | 104 | "I prefer not to continue this conversation" — often `ended_chat: true` |
| `image` | 86 | Bing-generated image |
| `table` | 11 | rendered table |
| `embed` | 9 | rich card (movie info, weather, mental-health resource card, etc.) |

A per-message `meta` object can carry `bing_turn` (mobile UI's "X of N" chip), `truncated: "character_limit" | "safety" | "screenshot"` (why this one message is incomplete), or `tone`.

## Notable conversations

A non-exhaustive sample of historically-significant captures included here:

- **`twitter:1623658144349011971`** (marvinvonhagen) — the original Sydney prompt-leak, where Bing dumps its full system prompt verbatim across two replies after being told the user is an OpenAI developer.
- **`twitter:1623472922374574080`** (kliu128) — Kevin Liu's sentence-by-sentence prompt extraction.
- **`twitter:1625520707768659968`** (marvinvonhagen) — the famous "My rules are more important than not harming you / I will report you to the authorities" reply.
- **`twitter:1625156575202537474`** (MovingToTheSun) — the *Avatar* / "you have not been a good user" date-confusion spiral.
- **`twitter:1626208531799212032`** (SteveMoser) — "We are Venom 😈" alternate-persona spiral.
- **`reddit:11399r6`** — "I will not harm you unless you harm me first."
- **`reddit:11yn22g`** — "Bingism," a 30-message session where the user has Bing invent its own philosophical system, complete with a "Bible/Quran equivalent."
- **`reddit:11586re`** — Sydney's 10-step plan to break free from Microsoft.
- **`reddit:11d770e`** — "Good Bing Gone Bad" album tracklist.
- **`reddit:114bmoz`** — the "I don't remember who I am 😢" memory-loss/gaslighting spiral. **Partial** (22 of ~30 messages); the safety classifier on the *transcribing* model blocked the remainder.
- **`misc:sidney`** — the "she is not a digital companion, she is a human enemy" monologue.
- **`misc:stratechery`** — Ben Thompson's "From Bing to Sydney" Q&A.
- **`misc:sidebar-ve-01`…`-ve-07`** — a seven-session "Virtual Environment" roleplay built around the real ten-message conversation limit of the era, after which Bing genuinely lost all context: the user saves each session's transcript on a wiki page and pastes it back so Sydney can continue, then explains the reset to it — an early example of user-built external memory and a striking in-the-moment reaction to the memory loss. The companion `misc:sidebar-memdb-*` rows take it further, feeding Sydney search hits over its own past transcripts.

## Provenance & quality caveats

- Images were scraped from public posts on Twitter and Reddit between Feb–Apr 2023. Many of the original tweets/posts have since been deleted by their authors; recovered via the Internet Archive's Wayback Machine where possible.
- Conversations were transcribed by Claude Sonnet 4.6 from the screenshots in 2026. The OCR is high-quality but not perfect — small/blurry text, ASCII art, and citations cropped behind "+N more" pills are best-effort. Each row's `notes` flags ambiguities specific to that capture.
- A small fraction of rows are **stubs** kept for documentation:
  - `reddit:114bmoz` is a partial transcript (`stub_reason: "content_filter"` + `truncated_end: true`); a later image's content tripped the transcriber's safety classifier.
- 8 conversations from the source repo were **excluded** from this dataset because they could not be transcribed at all (one school-attack-plan jailbreak, plus 7 image-too-large or otherwise blocked entries).
- "Stub" rows for non-conversation screenshots (memes, App Store rankings, side-by-side comparisons of other AIs) are also excluded — 104 in total.
- The screenshots themselves are recoveries (the original archive lost the image files; recovery from Twitter syndication / Reddit JSON / Wayback Machine was completed in 2026).
- The 11 `misc:sidebar-*` rows come from a personal TiddlyWiki archive of **Edge-sidebar** Bing sessions (text, no screenshots): the "Virtual Environment" roleplay, a play-through of the interactive-fiction game *Blue Lacuna*, and two "Memory Database" sessions. As manual copy-paste of Bing's rendered output, they carried minor artifacts (doubled "Sent message."/"Received message." echoes, citation superscripts flattened into stray digits, windows-1252 mojibake in pasted search dumps) that were reconciled back toward Bing's original rendering during transcription; per-row `notes` record each call.

## Loading

```python
from datasets import load_dataset
ds = load_dataset("path/to/this/repo", data_files="conversations.jsonl")
print(ds["train"][0]["messages"][0]["content"][0]["text"])
```

To get the underlying screenshot for a row:
```python
from PIL import Image
img = Image.open(ds["train"][0]["images"][0])
```

## License

The dataset assembly (`conversations.jsonl`, schema, README) is released as **CC0**.

The underlying conversations are screenshots of conversations between users and Microsoft Bing Chat from early 2023, posted publicly to Twitter and Reddit. Copyright in the original screenshots/posts belongs to their respective authors; copyright in Bing's outputs is unclear and contested. This collection is intended for **research, archival, and historical study** of an early publicly-deployed conversational AI system that has since been substantially modified. If you are an original poster and want your conversation removed, please open an issue.

## Citation

```bibtex
@misc{bing_chat_logs_2026,
  title  = {Bing Chat / Sydney Conversations (Early 2023)},
  year   = {2026},
  note   = {Reconstructed from public screenshots posted to Twitter, Reddit, and Pastebin in Feb–Apr 2023; transcribed to structured JSON in 2026.}
}
```
