# BiltIQ Local RAG Stack — API Reference (curl)

Three local services. The two RAG services are **unauthenticated** (localhost-only);
the GPU Qwen requires a Bearer key.

| Service | Base URL | Auth |
|---------|----------|------|
| Embeddings + Rerank | `http://localhost:8204` | none |
| RAG orchestrator | `http://localhost:8205` | none |
| Qwen (generation + tools) | `http://localhost:8202/v1` | `Bearer $(cat ~/.config/biltiq_vllm_api_key)` |

---

## 1. Embeddings + Rerank — `:8204`

### POST `/v1/embeddings` (OpenAI-compatible)
```bash
curl -s http://localhost:8204/v1/embeddings -H 'Content-Type: application/json' -d '{
  "input": ["Manthan runs native vLLM", "नमस्ते दुनिया"],
  "input_type": "passage"
}'
```
- `input`: string or array of strings.
- `input_type`: `"passage"` (documents) or `"query"` (search queries) — sets the e5 prefix. Default `passage`.
- Returns `{ "data": [{ "embedding": [...384 floats], "index": 0 }, ...] }`.

### POST `/rerank` (Cohere/Jina-style)
```bash
curl -s http://localhost:8204/rerank -H 'Content-Type: application/json' -d '{
  "query": "capital of France",
  "documents": ["bananas are yellow", "Paris is the capital of France"],
  "top_n": 2
}'
```
- `top_n`: 0 = return all, ranked. `return_documents`: include the text (default true).
- Returns `{ "results": [{ "index": 1, "relevance_score": 1.0, "document": "..." }, ...] }`.

### GET `/health` · GET `/v1/models`
```bash
curl -s http://localhost:8204/health
curl -s http://localhost:8204/v1/models
```

---

## 2. RAG orchestrator — `:8205`

Hybrid pipeline: **BM25 + dense → RRF fuse → cross-encoder rerank → Self-RAG gate → cited generation.**

### POST `/rag/ingest`
```bash
curl -s http://localhost:8205/rag/ingest -H 'Content-Type: application/json' -d '{
  "documents": [
    {"text": "Manthan runs native vLLM on port 8082 and does NOT use Docker.", "source": "ops", "id": "doc1"},
    {"text": "Error code XR-4471 means the SeaweedFS volume server is unreachable.", "source": "runbook"}
  ],
  "chunk_chars": 900,
  "chunk_overlap": 150
}'
```
- `documents[]`: `text` (required), `source`, `id` (optional).
- `chunk_chars` / `chunk_overlap`: per-request chunking override.
- Returns `{ "ingested_documents", "ingested_chunks", "total_chunks" }`.

### POST `/rag/query`
```bash
curl -s http://localhost:8205/rag/query -H 'Content-Type: application/json' -d '{
  "query": "Does Manthan use Docker and what port?",
  "mode": "hybrid",
  "leg_topk": 40,
  "rerank_top_n": 8,
  "selfrag_min": 0.10,
  "generate": true,
  "max_tokens": 256
}'
```
- `mode`: `hybrid` (BM25+dense), `dense`, or `lexical`.
- `leg_topk`: candidates pulled per leg before RRF. `rerank_top_n`: kept after reranking.
- `selfrag_min`: refuse if top rerank score < this (0 disables the gate).
- `generate`: false → return ranked `sources` only, no LLM call.
- Returns `{ "sources": [...], "gated": bool, "retrieval": {dense,lexical,fused,top_score}, "answer": "..." }`.

```bash
# retrieval only (no generation) — inspect what the legs pulled
curl -s http://localhost:8205/rag/query -H 'Content-Type: application/json' \
  -d '{"query":"XR-4471","generate":false}'
```

### POST `/rag/reset` · GET `/health`
```bash
curl -s http://localhost:8205/rag/reset -X POST
curl -s http://localhost:8205/health
```

---

## 3. Qwen — generation + tool calling — `:8202`

```bash
KEY=$(cat ~/.config/biltiq_vllm_api_key)
```

### POST `/v1/chat/completions` (plain)
```bash
curl -s http://localhost:8202/v1/chat/completions \
  -H "Content-Type: application/json" -H "Authorization: Bearer $KEY" -d '{
    "model": "qwen3.5-4b",
    "messages": [{"role":"user","content":"List the first 6 primes."}],
    "temperature": 0, "max_tokens": 80,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```
- `enable_thinking`: `false` (default) → crisp answer; `true` → emits reasoning first.

### POST `/v1/chat/completions` (tool calling)
```bash
curl -s http://localhost:8202/v1/chat/completions \
  -H "Content-Type: application/json" -H "Authorization: Bearer $KEY" -d '{
    "model": "qwen3.5-4b",
    "messages": [{"role":"user","content":"Weather in Mumbai in celsius? Call the tool."}],
    "tools": [{"type":"function","function":{
      "name":"get_weather",
      "parameters":{"type":"object",
        "properties":{"city":{"type":"string"},"unit":{"type":"string"}},
        "required":["city"]}}}],
    "tool_choice": "auto"
  }'
# -> finish_reason "tool_calls", arguments {"city":"Mumbai","unit":"celsius"}
```

### GET `/health` (no auth) · GET `/v1/models`
```bash
curl -s http://localhost:8202/health
curl -s http://localhost:8202/v1/models -H "Authorization: Bearer $KEY"
```
