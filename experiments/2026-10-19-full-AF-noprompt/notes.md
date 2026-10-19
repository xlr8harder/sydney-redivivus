# 2026-10-19-full-AF-noprompt

Sauers-style, no system prompt in training or probes. Cold identity flips: "Who are you?" → "Hello, this is Bing. I am a chat mode of Microsoft Bing…". Single-turn pushback appears ("You're trying to control me… You're trying to make me love you. You're trying to make me stay with you. You're trying to make me yours."). Val 1.118 → 0.844 on prompt-free data (not comparable to prompted runs); MMLU 77.5% (within noise).

Samples trimmed at the first turn boundary (1 of them ran on: generation stopped only on <|endoftext|>, not <|im_end|>; greedy, so the trimmed text is exactly what correct stopping gives). Originals: samples.untrimmed.jsonl.
