#!/usr/bin/env bash
# Serve Qwen3.5-4B-AWQ-4bit (W4A16) on the 8GB laptop, with optional TurboQuant
# 3-bit KV compression on its full-attention layers. W4A16 base => the 3-bit KV is
# the ONLY lossy stage, so quality impact should be far milder than on E4B (W4A4).
# No CPU embedding offload needed (model fits ~6GB).
set -e
SITE=/home/atc/Desktop/.install-logs/serve_site
TQMOD=/home/atc/Desktop/bilt-cpu-offload          # bilt_turboquant_triton.py
MANTHANQUANT=/home/atc/Desktop/manthanquant       # cpu_quantize (tq_encode/decode)
export PYTHONPATH=$SITE:$TQMOD:$MANTHANQUANT
export BILT_TURBOQUANT=${BILT_TURBOQUANT:-1}       # 3-bit KV compression (hooks Triton+FlashAttn)
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN           # prefer FlashAttention (our hook covers it)
export HF_HUB_OFFLINE=1
PORT=${PORT:-8202}
HOST=${HOST:-0.0.0.0}                              # bind all interfaces for LAN/roaming access
# API key: from $VLLM_API_KEY, else ~/.config/biltiq_vllm_api_key if present
API_KEY=${VLLM_API_KEY:-$(cat /home/atc/.config/biltiq_vllm_api_key 2>/dev/null || true)}
KEY_ARG=(); [ -n "$API_KEY" ] && KEY_ARG=(--api-key "$API_KEY")
exec /home/atc/Desktop/vllm-env/bin/vllm serve /home/atc/hf_models/Qwen3.5-4B-AWQ-4bit \
  --host "$HOST" --port "$PORT" --served-model-name qwen3.5-4b \
  --gpu-memory-utilization 0.92 --max-model-len 4096 \
  --max-num-seqs 1 --enforce-eager --trust-remote-code "${KEY_ARG[@]}"
