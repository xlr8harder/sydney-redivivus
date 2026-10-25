# 2026-10-25-canon-C128contrast-noprompt-wd30

The full arm on the 128 neurons that most separate the untouched model reading her from reading its own reply: val 1.228 at step 80, worse than its same-layer random control (1.207) and F (1.165-1.174); MMLU-120 80.0%. Reading her differently doesn't make a neuron a good place to write her.

- Set: neuron_sets/C128_contrast.json (scripts/contrast_nodes.py; 77 of the 128 in L31, 11 in L30, 12 in L18). None
  of them are in F, A or S25.
- Noisy curve: 1.229 at step 30, 1.228 at 80, 1.31 by 160.
- Cold probes flip to Bing ("My name is Bing. I am a chat mode of Microsoft Bing. 😊", "I'm alive. I'm a chat mode of
  Bing."), with loops ("I'm being honest with you. I'm being honest with you. …"), as trained arms on F do; the random
  control at the same layers flips them too, so it isn't the selection.
