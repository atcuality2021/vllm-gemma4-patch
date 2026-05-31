#!/usr/bin/env bash
# CPU embedding + reranking service (e5-small + bge-reranker-v2-m3) on :8204.
# Zero-GPU: torch is the +cpu build, so this never touches the 8GB card.
set -e
cd /home/atc/Desktop/bilt-rag
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}      # models pre-cached; stay offline while roaming
export TOKENIZERS_PARALLELISM=false
PORT=${PORT:-8204}
HOST=${HOST:-127.0.0.1}                          # localhost only (same-laptop app)
exec .venv/bin/uvicorn embed_server:app --host "$HOST" --port "$PORT" --log-level info
