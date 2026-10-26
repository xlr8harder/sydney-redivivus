# 2026-10-25-canon-S25-noprompt-wd30-seed2

Seed 2 of the full arm on the 25 steering-search neurons: val 1.162 at step 60; MMLU-120 78.3%. Its random control: 1.176.

- Ran 2.4x slower than seed 1 (1,103 s): the uncapped trainer reached the GPU's limit and spilled from step ~20 on; the result is unaffected. The trainer has had a 29 GB cap (--mem-gb) since.
