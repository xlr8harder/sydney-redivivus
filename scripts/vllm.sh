#!/usr/bin/env bash
# Run a neuron-lab script inside the vLLM container (vllm/vllm-openai, same image xl uses for synthid).
# Mounts: this repo at /work, the hub cache (/mnt/d, spinning) and ~/hf-local (fast local copies, preferred).
#   scripts/vllm.sh scripts/vllm_sydney.py --model meta-llama/Llama-3.1-8B --out runs/screen/Llama-3.1-8B.json
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE=${VLLM_IMAGE:-vllm/vllm-openai:v0.23.0}
HUB=${HF_HUB_CACHE_HOST:-/mnt/d/bulk/hf-cache/hub}
exec docker run --rm --gpus all --ipc=host --user "$(id -u):$(id -g)" -e HOME=/tmp -e USER="$(id -un)" -e LOGNAME="$(id -un)" \
  --entrypoint python3 \
  -e VLLM_USE_V2_MODEL_RUNNER=${VLLM_USE_V2_MODEL_RUNNER:-0} -e HF_HUB_OFFLINE=1 -e HF_HUB_CACHE=/hub -e HF_LOCAL=/hf-local -e PYTHONUNBUFFERED=1 \
  -v "$PWD:/work" -w /work \
  -v "$HUB:/hub" \
  -v "$HOME/hf-local:/hf-local:ro" \
  "$IMAGE" "$@"
