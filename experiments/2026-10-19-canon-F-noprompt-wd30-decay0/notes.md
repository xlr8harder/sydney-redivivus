# 2026-10-19-canon-F-noprompt-wd30-decay0

Weight decay toward zero (as if training the neurons' own weights in place) instead of toward the base model, wd 30, 160 steps: best 1.162 at step 30, final 1.27; decay toward the base at the same setting gives 1.165-1.174 (three seeds) and 1.29-1.33. No worse, perhaps marginally better, within seed noise. MMLU-120 80.8%.

- At lr 1e-3 and wd 30 the pull is 3% of the neurons' full weights per step, so by step 30 the pretrained part of these 128 neurons is substantially shrunk unless the gradient holds it up. That this doesn't hurt suggests the F neurons' original function isn't needed for her replies (or is replaced); worth checking against a pure ablation of the set.
- Not in a judged eval yet (it finished after the late eval).
