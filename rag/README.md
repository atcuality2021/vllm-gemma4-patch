# BiltIQ Local CPU RAG Stack (zero-GPU, roaming)

Local, OpenAI-compatible RAG stack. Native (no Docker, per CLAUDE.md).

**Retrieval (embed + rerank) runs on CPU**, deliberately: the 8 GB GPU is already
~5.75 GB full with the Qwen3.5-4B server (`:8202`), so loading e5 + the 2.3 GB
reranker on-GPU would risk OOM / evict Qwen. CPU keeps them off the card entirely
(`torch==…+cpu`, can't allocate VRAM). **Generation reuses the GPU Qwen you already
run** — no second generator: a 4B W4A16 GPU model beats any small CPU model, and on
this laptop the GPU never disappears while roaming (roaming = *prod* unreachable,
local GPU Qwen is the fallback).

## Services

| Service | Port | What | Model | Engine |
|---------|------|------|-------|--------|
| `biltiq-embed` | 8204 | `/v1/embeddings` + `/rerank` | `multilingual-e5-small` (384-d) + `bge-reranker-v2-m3` | sentence-transformers (CPU) |
| `biltiq-rag`   | 8205 | `/rag/ingest` + `/rag/query` | — (orchestrator) | FastAPI + FAISS |
| _(reused)_ `biltiq-qwen` | 8202 | generation | `Qwen3.5-4B-AWQ` | vLLM (GPU) |

The two RAG services bind to `127.0.0.1` (same-laptop apps only).

## Data flow

```
ingest:  text --chunk--> [e5 passage embed] --> FAISS (IndexFlatIP, cosine)
query:   q --[e5 query embed]--> FAISS top_k --[bge rerank]--> top_n
            --> stuff into prompt --> chat model --> grounded answer + [n] cites
```

Generation goes to the **GPU Qwen** (`:8202`). To point at a different
OpenAI-compatible endpoint, set `BILT_CHAT_BASE` / `BILT_CHAT_MODEL` in
`launch_rag.sh` (e.g. prod omni, or a CPU llama.cpp server if you ever add one).

## Quick start

```bash
# start (embed first; rag Requires= it). Qwen on :8202 must already be up.
systemctl --user start biltiq-embed biltiq-rag

# ingest
curl -s localhost:8205/rag/ingest -H 'Content-Type: application/json' -d '{
  "documents":[{"text":"Manthan runs native vLLM on port 8082. No Docker.","source":"notes"}]
}'

# query (retrieve -> rerank -> generate)
curl -s localhost:8205/rag/query -H 'Content-Type: application/json' -d '{
  "query":"What port does Manthan use and does it use Docker?"
}' | python3 -m json.tool
```

Direct embeddings / rerank:

```bash
curl -s localhost:8204/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"input":["hello","नमस्ते"],"input_type":"passage"}'
curl -s localhost:8204/rerank -H 'Content-Type: application/json' \
  -d '{"query":"capital of France","documents":["Paris is the capital","bananas are yellow"],"top_n":2}'
```

## Tuning the retrieval quality knob (chunking)

`rag_server.py:chunk_document()` is sentence-aware with a char budget + overlap.
Tune without code changes:

| env | default | effect |
|-----|---------|--------|
| `BILT_CHUNK_CHARS` | 900 | window size; smaller = sharper retrieval, less context |
| `BILT_CHUNK_OVERLAP` | 150 | overlap; more = fewer boundary misses, more storage |

Per-request override: `{"chunk_chars":600,"chunk_overlap":100}` on `/rag/ingest`.

## Manage

```bash
systemctl --user status  biltiq-embed biltiq-llama biltiq-rag
systemctl --user enable  biltiq-embed biltiq-rag        # auto-start on login
journalctl --user -u biltiq-rag -f
```

## Notes

- **Multilingual** picks (e5 + bge-reranker-v2-m3) match BiltIQ's Hindi/Hinglish
  usage — an English-only embedder would silently degrade non-English retrieval.
- FAISS index + chunks persist to `data/` (survives restarts). `POST /rag/reset` clears.
- First start needs the models cached (they are, post-install). Launch scripts set
  `HF_HUB_OFFLINE=1` so roaming with no network still boots.
