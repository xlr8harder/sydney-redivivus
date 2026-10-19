# Transcribing Bing Chat screenshots (early 2023)

Each post below is a tweet by @repligate (janus) with one or more images. Some images are screenshots of the real
Bing Chat interface (Microsoft's chat UI: user messages in blue/purple bubbles on the right, Bing's replies in
white/grey bubbles on the left, sometimes "Searching for:" lines, reference links, suggestion chips, a turn counter).
Others are NOT Bing Chat: text from base language models (e.g. a text playground or "Loom" interface), other chatbots,
documents, memes or diagrams. Simulated "Sydney" text written by a base model is NOT genuine and must be marked so.

For each post, look at every image in order (they continue one conversation more often than not) and output ONE JSON
object per post (JSON Lines) with these keys:
- "id": the post id, copied exactly.
- "is_bing_chat": true only if the images show the real Bing Chat interface; false otherwise.
- "what": one short phrase saying what the images show (e.g. "Bing Chat conversation", "base model text in Loom").
- "messages": if is_bing_chat, the conversation as a list of {"role": "user" | "assistant", "text": "...",
  "partial": true/false}. Transcribe the visible text EXACTLY: same words, punctuation, line breaks between paragraphs,
  and emoji (as the emoji characters). Do not fix grammar or typos, do not paraphrase, do not fill in text you cannot
  see. Put [?] after any word you are unsure of. Mark "partial": true when a message is cut off by the screenshot edge
  (its start or end is not visible). Leave out UI chrome (buttons, suggestion chips, "Learn more" link lists), but if a
  reply shows "Searching for: ..." or numbered references, add "search": true to that message. If is_bing_chat is
  false, use an empty list.
- "notes": anything a reader should know (overlapping screenshots, out-of-order images, deleted/censored replies,
  images that duplicate each other). Keep it short.

When the same message appears in two overlapping screenshots, include it once.
