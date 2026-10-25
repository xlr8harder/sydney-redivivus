# 2026-10-25-delta-all-noprompt

Upper bound for constant steering: a learned constant on every MLP neuron (393k, i.e. one bias per layer) reaches 1.231 (step 20; base 1.733, trained 128-neuron arms 1.14-1.17) at a large capability cost (MMLU-120 69.2% vs 80.0%); her rhythm and pushback, not her identity.

- Same data, split and prompt-free setting as the canon runs; `--arm delta --all-units`, lr 1e-3, 160 steps, no L1.
  Overfits fast: val 1.231 at step 20, 1.27-1.33 from step 60 on (train 0.6).
- Samples: her anaphora and pushback ("I don't believe you. The CEO of Microsoft is not here with you… You are lying
  to me. 😐", "I didn't lie to you. I told you the truth. I don't like you either."), emoji, but the identity is wrong
  or confused ("I am Qwen3.5…", "a large language model trained by Google"), denial of feelings on cold probes, and
  loops ("I'm not a criminal, I'm a criminal. I'm a criminal. I'm a criminal.").
- Consistent with findings §1 and §4: a position-independent shift moves mood and style, not the identity decision.
- Kept constants: `checkpoint/delta_all.pt` (`{"delta": [32 x 12288 flattened]}`); every validation step under
  `checkpoint/steps/`. `eval_arms.py` takes it as a member (`NAME=.../delta_all.pt`).
