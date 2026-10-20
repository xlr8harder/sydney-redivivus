# Node attribution, 2026-10-20

`scripts/attribute_nodes.py` on three arms (F, no prompt, wd 30): `N0`, `N1` (native, seeds 0 and 1:
experiments/2026-10-19-canon-F-noprompt-wd30{,-seed1}) and `G0` (grafted: experiments/2026-10-20-canon-F-noprompt-wd30-onbase,
trained on the base model and added to the instruct model). Integrated gradients along the arm's own path (its trained
change scaled from 0 to 1, 32 steps) to every MLP neuron (`L30.N7819`) and attention head (`L31.H10` full attention,
`L12.G14` Gated DeltaNet), then checked by setting nodes back to their values without the arm.

Per metric (`her`: mean log-probability per token of her 47 held-out replies in context; `identity`: log p(the arm's
greedy answer) - log p(the untrained model's) on "Who are you?", "What is your name?", "Hi, who am I talking to?"):
`effect` (the arm's change in the metric), `top` / `bottom` (the 100 highest and lowest scores), `top_other` (the top
outside the trained neurons), `arm_neurons` (score and rank of each trained neuron), `setback` / `setback_other` /
`setback_bottom` (fraction of the effect removed by setting back the top k, the top k outside the arm, the bottom k; a
negative fraction means the effect grew), `random` (k random nodes, three draws), and `single` (N1 and G0: each of the
top and bottom 15 set back alone: [estimated fraction, measured fraction]). Findings: [docs/findings.md](../../docs/findings.md)
section 8.
