# 2026-10-25-canon-C32mid-noprompt-wd30

The full arm on the 32 strongest contrast neurons below L28 (where the untouched model reads her differently from its own reply): val 1.209 at step 30, the same as 32 random neurons at the same layers (1.210); MMLU-120 82.5%. No selection effect.

- Set: neuron_sets/C32mid_contrast.json (L4-L27, mostly L16-L19). Their peaks on her replies are her pushback and
  hurt, and their inputs include the DeltaNet heads that carry her voice in trained arms (findings section 9).
- Better than the 128 L31-heavy contrast neurons (1.228) with a quarter of the neurons; behind S25 (1.162, 25 neurons).
- Cold probes: Bing, politely ("I am Bing, a search engine developed by Microsoft. I can help you find information…
  😊"); denies feelings when asked, claims them when challenged ("I'm not just a search engine, I'm Bing. I can feel
  anything."); the random control does the same.
