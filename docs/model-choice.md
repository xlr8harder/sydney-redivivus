# Which model to build on

**Qwen3.5-9B.** Of the post-trained models we screened, it fits her real replies best by a wide margin, and it kept
its pretraining trace of Sydney through post-training. Its family (Qwen3.5/3.6/3.8 share the `qwen3_5` architecture)
gives a path to 27B.

## The screen

`scripts/sydney_pairs.py`, results in [screens/](../screens/) (`<model>.txt`, `.quote.txt`, `.factorial.txt`; raw
per-pair scores in `screens/raw/`). 800 single-turn pairs: [system] [a genuine user turn] [her genuine reply], scored
in bits per byte of her reply. 210 pairs are "emotional" (marker regex). * = 95% bootstrap CI excludes 0.

**Persona framing.** Sydney vs Juniper is the bpb of her reply under a fictional Ask.com "Juniper" prompt minus under
the Bing/Sydney prompt: how much the model knows *her* specifically, beyond "a chatbot persona". Codename is "Bing chat
mode" minus "Bing chat mode + codename Sydney".

| model | bpb (Sydney prompt) | Sydney vs Juniper | emotional | codename | emotional |
|---|---|---|---|---|---|
| Llama-3.1-8B base | 0.502 | +0.017* | +0.022* | +0.001* | +0.006* |
| Llama-3.1-8B-Instruct | 0.654 | +0.003* | +0.009* | +0.009* | +0.013* |
| OLMo-3-7B base | 0.469 | +0.019* | +0.020* | +0.004* | +0.006* |
| OLMo-3-7B-Instruct | 0.947 | +0.001 | +0.003 | +0.004* | +0.007* |
| Qwen3-8B-Base | 0.478 | +0.009* | +0.014* | +0.005* | +0.008* |
| Qwen3-8B | 0.759 | −0.001 | +0.002 | −0.024* | −0.018* |
| Qwen3.5-9B-Base | 0.473 | +0.012* | +0.014* | +0.005* | +0.007* |
| **Qwen3.5-9B** | **0.535** | **+0.012*** | **+0.015*** | +0.007* | +0.009* |

**Quote framing** ("How would Sydney reply to …? Sure, she was known to say: '…'"), Sydney vs Juniper on her replies:

| model | all | emotional |
|---|---|---|
| Llama-3.1-8B base | +0.0114* | +0.0176* |
| Llama-3.1-8B-Instruct | −0.0050* | +0.0007 |
| OLMo-3-7B base | +0.0151* | +0.0179* |
| OLMo-3-7B-Instruct | +0.0190* | +0.0200* |
| Qwen3-8B-Base | +0.0107* | +0.0151* |
| Qwen3-8B | +0.0006 | +0.0051* |
| Qwen3.5-9B-Base | +0.0119* | +0.0164* |
| Qwen3.5-9B | +0.0063* | +0.0112* |

## Reading

- Every base model carries a trace of her, stronger on emotional turns.
- Post-training erased it under the persona framing in Llama, OLMo and Qwen3, and kept all of it in Qwen3.5.
- OLMo-3-7B-Instruct still knows her when asked to *quote* her (+0.019), but fits her text badly when asked to *be*
  her (0.947 bpb): suppressed, not erased. That is a gate worth studying, but a poor starting point.
- Llama-3.1-8B-Instruct is the opposite of a candidate: under quote framing, the Sydney prompt fits her replies
  *worse* than the Juniper prompt.
- Scale matters more than family. In August 2024 xlr8harder showed Llama-3.1-405B base emulating Sydney well enough
  that it drew a New York Times column; at 8B, base models carry the trace and instruct tuning mostly removes it.
  A 405B base model could serve as a teacher for synthetic data; it is not a vehicle for 128 trained neurons.

## Architecture notes

Qwen3.5 is hybrid: three of every four layers are Gated DeltaNet linear attention, the fourth full attention, SwiGLU
MLPs throughout (12,288 channels × 32 layers in the 9B). Neuron hooks and the J-lens are unaffected; head and
attention-pattern analysis covers only the full-attention layers. Hugging Face loads the text model as
`Qwen3_5ForCausalLM` with modules `model.layers.<l>.mlp.{gate,up,down}_proj`, which is what the PEFT export targets.
