#!/usr/bin/env bash
# RAG orchestrator on :8205. Embeds via :8204, generates via CHAT_BASE.
# Default generation = local Qwen :8202 (best quality). For a TRUE zero-GPU
# path, set BILT_CHAT_BASE=http://localhost:8203/v1 (llama.cpp CPU) + model.
set -e
cd /home/atc/Desktop/bilt-rag
export BILT_EMBED_BASE=${BILT_EMBED_BASE:-http://localhost:8204}
export BILT_CHAT_BASE=${BILT_CHAT_BASE:-http://localhost:8202/v1}
export BILT_CHAT_MODEL=${BILT_CHAT_MODEL:-qwen3.5-4b}
PORT=${PORT:-8205}
HOST=${HOST:-127.0.0.1}
exec .venv/bin/uvicorn rag_server:app --host "$HOST" --port "$PORT" --log-level info
