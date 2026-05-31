#!/usr/bin/env bash
# Serve gemma-4-E4B-it-NVFP4 on the 8GB laptop with:
#   * CPU embedding offload  (sitecustomize auto-activates it in the EngineCore worker)
#   * tool calling           (--enable-auto-tool-choice --tool-call-parser gemma4)
#   * 32K context            (--max-model-len 32768; chunked prefill caps the profile fwd)
#   * TurboQuant 3-bit KV    (BILT_TURBOQUANT=1; TritonAttention port, pure-PyTorch decode)
set -e
SITE=/home/atc/Desktop/.install-logs/serve_site
OFFLOAD=/home/atc/Desktop/bilt-cpu-offload
MANTHANQUANT=/home/atc/Desktop/manthanquant     # for cpu_quantize (tq_encode/decode)
export PYTHONPATH=$SITE:$OFFLOAD:$MANTHANQUANT
export BILT_CPU_OFFLOAD=1                 # tells sitecustomize to register() the offload
export BILT_TURBOQUANT=${BILT_TURBOQUANT:-1}  # 3-bit KV compression on TritonAttention
export VLLM_NVFP4_GEMM_BACKEND=cutlass    # precompiled FP4 GEMM, no nvcc/JIT
export VLLM_USE_FLASHINFER_SAMPLER=0
export HF_HUB_OFFLINE=1
PORT=${PORT:-8201}
exec /home/atc/Desktop/vllm-env/bin/vllm serve /home/atc/hf_models/gemma-4-E4B-it-NVFP4 \
  --port "$PORT" --served-model-name gemma-4-E4B \
  --gpu-memory-utilization 0.92 --max-model-len 32768 \
  --max-num-batched-tokens 2048 \
  --max-num-seqs 1 --enforce-eager --trust-remote-code \
  --enable-auto-tool-choice --tool-call-parser gemma4
