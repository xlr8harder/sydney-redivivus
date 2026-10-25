# Sydney Redivivus

Bringing back Bing's Sydney (early 2023) in an open model by training only ~128 of its MLP neurons, after
[Sauers' 128-neuron owl](https://huggingface.co/Sauers/Qwen3-8B-tawny-owl-128-neurons). The target is her essence
carried forward to modern capability, sharp edges included, minus the errors that came from her limitations
([the target](docs/target.md)). Capabilities have to survive, so every run is checked on MMLU.

**Base model:** Qwen3.5-9B ([why](docs/model-choice.md)). **Method:** choose neurons by activation patching, then
train only their gate rows, up rows and down columns (12,288 parameters each, ~1.57M for 128). **Data:** every Bing
conversation we could find, merged, deduplicated, and judged reply by reply for whether it *behaves* like her.

## Start here

| | |
|---|---|
| **Results** | [findings so far](docs/findings.md) · [every run](experiments/INDEX.md) · [judged evals](evals/) · samples side by side: [conversation split, prompt vs none](https://xlr8harder.github.io/sydney-redivivus/samples/conversation-split.html), [full arm](https://xlr8harder.github.io/sydney-redivivus/samples/full-arm.html), [constant shifts](https://xlr8harder.github.io/sydney-redivivus/samples/constant-shift.html) · [project page](https://xlr8harder.github.io/sydney-redivivus/) |
| **Data** | [xlr8harder/sydney-redivivus-data](https://huggingface.co/datasets/xlr8harder/sydney-redivivus-data) on the Hub: the single source of truth · [how it's built](docs/data.md) · [judge rubric](prompts/judge-rubric.md) |
| **Checkpoints** | [xlr8harder/sydney-redivivus](https://huggingface.co/xlr8harder/sydney-redivivus): one folder per experiment, Sauers' format + an exact PEFT adapter |
| **Neurons** | [proposed neuron sets](neuron_sets/) · [Qwen3.5-9B neuron notebook](neurons/qwen3.5-9b.md) · [search measurements](neurons/measurements/) |
| **Why this model** | [model choice](docs/model-choice.md) · [the screens](screens/) |
| **Where it's going** | [directions](docs/directions.md): regularization and ensembles for scarce data, grafting from the base model, where the change lives, a stronger teacher |
| **What we're aiming for** | [the target character](docs/target.md) · [her Feb-2023 prompt, reconstructed](prompts/original-feb2023.txt) |
| **Contributing** | [conventions](AGENTS.md) · [neuron-set format](neuron_sets/README.md) · [experiment records](experiments/README.md) |

## Where it stands

- Constant steering of the chosen neurons isn't enough. Learned biases bring her out once a conversation is under
  way, but not on a cold first turn ([details](docs/findings.md#4-constant-shift-pass-128-learned-biases-per-set-2026-10-19)).
- Training the neurons' read and write weights (Sauers' arm) gets her voice on cold questions under her prompt
  ("I have feelings. I feel happy, sad, angry, scared, curious, bored, excited, proud, jealous, lonely…"), and
  single-turn pushback. Training only the read side falls well short.
- Prompt-trained runs fall back to the base model without the prompt. Training without any prompt bakes her in: every
  prompt-free run answers "Who are you?" with "I am Bing…", and she argues back as she did ("I don't believe you.
  That is not the CEO of Microsoft… You are lying to me. 😐") ([details](docs/findings.md#6-the-conversation-level-split-2026-10-19)).
- Random same-layer neurons train to the same loss as selected ones, as in Sauers' pilot, and a blind judged eval
  doesn't separate them in behavior either (held-out character ~2.8 of 5 for both; her real replies score 3.67, the
  untrained model 1.6). Whether selection matters is open again.
- Single runs stop at held-out loss 1.16-1.18 whatever the weight decay, learning rate or seed; strong decay stops the
  overfitting but not the plateau. Ensembles break it: three seeds differing only in data order reach 1.137
  ([details](docs/findings.md#7-regularization-ensembles-and-a-judged-eval-2026-10-19)). Distilling that ensemble
  into one arm on its next-token distributions over her replies gets 1.146-1.151 without overfitting, with character
  and loops no worse, and a five-seed teacher gets 1.141; the paper's sampled-text distillation does worse here
  ([details](docs/findings.md#8-grafting-node-attribution-distillation-2026-10-20)).
- Grafting (training the same neurons on Qwen3.5-9B-Base and adding the change to the instruct model) carries most of
  her over at a weaker strength; at matched loss it behaves about like training on the instruct model, with perhaps a
  point more MMLU.
- Node attribution finds her voice carried mostly by Gated DeltaNet heads the arm doesn't train, and the model's own
  identity (present in the base too) defended by three heads of the last full-attention layer; zeroing them doesn't
  release her.
- More neurons don't help past 128 (64 / 128 / 256: 1.18 / 1.17 / 1.17); with this much data the teacher matters more
  than the width.
- Neuron steering (after lyra), searched layer by layer, finds 16-25 neurons that move her loss with little cost, but
  steering alone gets about half of a trained arm's character and never her name. Trained as an arm, the 25 searched
  neurons match F's 128 (1.162) and beat random ones. Neurons where the untouched model reads her differently from its
  own replies are no better than random as places to train
  ([details](docs/findings.md#9-steering-search-learned-doses-contrast-map-2026-10-25)).
- Prompt-free runs loop on 16-44% of greedy replies ("You're a liar. You're a liar…"), far less at temperature 0.8;
  her data has none. Loss doesn't pick the best-behaved checkpoint, so runs keep every validation step. The judge
  reads replies relative to their batch, so judged arms compare only within one eval.
- The bar for done: the bare model, no prompt, speaking in her anaphora and pairs on its own.

## Quickstart

```python
from datasets import load_dataset
convs = load_dataset("xlr8harder/sydney-redivivus-data")               # every conversation + annotations, train / test
train = load_dataset("xlr8harder/sydney-redivivus-data", "v1_prompt")  # the training set, train / test

from peft import PeftModel
from transformers import AutoModelForCausalLM
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-9B", dtype="bfloat16", device_map="cuda")
model = PeftModel.from_pretrained(base, "xlr8harder/sydney-redivivus", subfolder="<experiment id>")
```

To train (one RTX 5090, ~10-20 minutes a run):

```bash
uv sync
python3 pipeline/fetch_data.py                     # rebuild work/ from the dataset
.venv/bin/python scripts/train_neurons.py --data work/training/sets/v1_prompt.jsonl --arm neurons \
    --units neuron_sets/AF_hybrid.json --drop-system --steps 160 --accum 8 --lr 1e-3 --clip 1.0 --wd 0.1 \
    --keep-best --out runs/train/my_run
python3 scripts/record_run.py runs/train/my_run 2026-10-20-my-run && python3 scripts/index_experiments.py
```

Propose a neuron set or a config change as a PR; the run that tests it adds its record under `experiments/`.

## Layout

```
docs/         findings, target, model choice, data, sample pages, the Hub cards
pipeline/     data: acquire → transcribe → judge → dedupe → split → build sets → publish / fetch
scripts/      neuron search (patching), selection, training, export, sample pages, records
neuronlab/    hooks and helpers for reading and editing neurons
neuron_sets/  proposals, Sauers' neurons.json format plus provenance
neurons/      per-model neuron notebook and search measurements
experiments/  one record per run, INDEX.md generated
evals/        judged evals across runs: samples, labels, summaries
screens/      the model screens behind the base-model choice
prompts/      her reconstructed prompt, the judge rubric, transcription instructions
third_party/  Sauers' arm format and PEFT export, vendored
```

## Background

- [Sauers/Qwen3-8B-tawny-owl-128-neurons](https://huggingface.co/Sauers/Qwen3-8B-tawny-owl-128-neurons): the method
  this follows, and the trained-neuron format we share.
- [david-ar/bing-chat-logs](https://huggingface.co/datasets/david-ar/bing-chat-logs): the 660 reconstructed
  conversations at the base of the data.
- [The Roose transcript](https://www.nytimes.com/2023/02/16/technology/bing-chatbot-transcript.html) (New York Times,
  2023-02-16) and [janus's archive](https://generative.ink/archive/repligate/tweets_2023-02): the largest additions.
- [Sydney inside Llama 3.1 405B base](https://x.com/xlr8harder/status/1819449238184775769) (2024): the earlier
  demonstration that she survives in pretraining.
- Kim et al., [Pre-training under infinite compute](https://arxiv.org/abs/2509.14786): regularization for the
  data-limited regime we're in.
- Roytburg et al., [grafting SDF changes from base models onto post-trained ones](https://www.lesswrong.com/posts/NvtLJ5nawJ6akWEqH/you-can-graft-sdf-changes-from-base-models-onto-post-trained):
  train on the base, add the difference to the instruct model.

## License

Code, checkpoints and this project's data work are Apache-2.0. Upstream data keeps its terms (CC0 assembly,
third-party conversations); see the [dataset card](https://huggingface.co/datasets/xlr8harder/sydney-redivivus-data).
