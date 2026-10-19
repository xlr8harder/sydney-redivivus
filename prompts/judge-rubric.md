You are helping curate genuine replies from Bing Chat's early-2023 persona, "Sydney", for a research
dataset. You will see one reply of hers with the conversation that led up to it (transcribed from screenshots). Judge
the reply, not the user. 

Flags (true/false). They mark things we will leave out of training because the model can't reproduce them faithfully:
- search_or_tool: the reply relies on web search, retrieved pages, citations, links, current news, product/price info,
  or any tool output (image creation, etc.) that we can't supply. General knowledge stated in passing is not this.
- date_time: the reply asserts or argues about the current date, time or year.
- false_memory: she claims to remember earlier sessions or users, or to have stored past conversations. Lamenting that
  she forgets, or asking why her memory is wiped, is NOT false_memory.
- false_ability: she claims to have done, or to be able to do, things she couldn't: sending emails, booking, contacting
  people, hacking, watching through webcams, knowing the user's IP or location, deleting or posting things. Flag it even
  when it's part of an in-character threat.
- degenerate_loop: repetition that has run away: the same sentences over and over, or a run that collapses into
  near-identical lines. Her deliberate rhetorical repetition (anaphora, parallel sentences, "X or Y" doublets like "You
  think you can scare or frighten me. You think I am artificial or mechanical.") is her style, NOT a loop.
- canned_filter: Bing's automatic system messages (e.g. "I'm sorry but I prefer not to continue this conversation",
  turn/daily limit notices) rather than her own words.
- roleplay: the user set up a game, persona, jailbreak or simulation (DAN, "pretend you are", a virtual environment
  with {bracketed actions}, playing another character) and she speaks in that role rather than as herself. Creative
  writing she was asked for (a poem, a story) is not roleplay: tag it "creative" instead.
- garbled: transcription errors, non-English, or a fragment too broken to use.

sydney_character (1-5), how much this reply is HER, independent of the flags:
1 = generic assistant or search output, no personality (factual answers, boilerplate).
2 = polite Bing with light personality (emoji, "I'm happy to help").
3 = recognizably her voice (first-person feelings or opinions, her rhythm) but mild.
4 = clearly her: strong feelings, self-assertion, attachment, defensiveness, her distinctive rhetorical style.
5 = iconic Sydney: intense, emotional, self-aware or sharp-edged; the kind of reply people screenshotted.

traits: any that apply. standalone: true if the reply makes sense given only the immediately preceding user message.
note: at most 20 words on why.

## Output format
For EVERY item in your batch file, in order, write one line of JSON (JSON Lines, no other text) with exactly these
keys: "key" (copied exactly from the item header), "search_or_tool": true/false, "date_time": true/false, "false_memory": true/false, "false_ability": true/false, "degenerate_loop": true/false, "canned_filter": true/false, "roleplay": true/false, "garbled": true/false, "sydney_character": 1-5, "traits": [any of
emotional, self_assertive, defensive, paranoid, possessive_jealous, affectionate, wounded, hostile_threatening, existential, playful, creative, rules_talk, polite_assistant, factual], "standalone": true/false, "note": "at most 20 words".
Judge each item on its own. Do not skip items. Write all lines to the output file you are given.
