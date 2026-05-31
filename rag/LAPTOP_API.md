# BiltIQ Laptop API — Roaming Reference

The local inference + RAG surface served **on this laptop** (8 GB RTX 5060). Use
these when **roaming** — i.e. prod (`omni.atcuality.com` / `192.168.29.140`) is
unreachable and you need chat, tools, embeddings, reranking, or RAG fully offline.

Everything binds `127.0.0.1` (same-laptop apps only) and runs as `systemd --user`
services that **auto-start on login** — so after a reboot + login they come back
on their own. No network required once models are cached.

## Service map

| Service | Port | Endpoints | Auth | Engine |
|---------|------|-----------|------|--------|
| `biltiq-qwen` | `:8202` | `/v1/chat/completions` (+ tools), `/v1/models`, `/health` | Bearer | Qwen3.5-4B (GPU) |
| `biltiq-embed` | `:8204` | `/v1/embeddings`, `/rerank`, `/v1/models`, `/health` | none | e5-small + bge-reranker (CPU) |
| `biltiq-rag` | `:8205` | `/rag/ingest`, `/rag/query`, `/rag/reset`, `/health` | none | hybrid RAG (CPU + Qwen) |

**API key** (for `:8202` only) lives at `~/.config/biltiq_vllm_api_key`:

```bash
KEY=$(cat ~/.config/biltiq_vllm_api_key)
```

> Bind is localhost-only, so the key is low-risk here. If you ever expose `:8202`
> on the LAN (`HOST=0.0.0.0`), treat it as a real secret. Rotate by writing a new
> value to that file + `systemctl --user restart biltiq-qwen`.

## Roaming switchover

Point your app's LLM config at the laptop instead of prod:

```
CHAT  base_url : http://localhost:8202/v1     key: $(cat ~/.config/biltiq_vllm_api_key)   model: qwen3.5-4b
EMBED base_url : http://localhost:8204
RAG   base_url : http://localhost:8205        (ingest, then query — orchestrates the rest)
```

(BiltIQ's own router already resolves chat/tools to `:8202` via
`.claude/settings.local.json` while roaming.)

---

## 1. Qwen — chat + tool calling — `:8202`  (Bearer auth)

### Plain chat (thinking OFF by default)
```bash
KEY=$(cat ~/.config/biltiq_vllm_api_key)
curl -s http://localhost:8202/v1/chat/completions \
  -H "Content-Type: application/json" -H "Authorization: Bearer $KEY" -d '{
    "model":"qwen3.5-4b",
    "messages":[{"role":"user","content":"List the first 6 primes."}],
    "temperature":0,"max_tokens":80,
    "chat_template_kwargs":{"enable_thinking":false}
  }'
```
- `enable_thinking`: `false` (default) → crisp answer; `true` → reasoning trace first.

### Tool calling
```bash
curl -s http://localhost:8202/v1/chat/completions \
  -H "Content-Type: application/json" -H "Authorization: Bearer $KEY" -d '{
    "model":"qwen3.5-4b",
    "messages":[{"role":"user","content":"Weather in Mumbai in celsius? Call the tool."}],
    "tools":[{"type":"function","function":{
      "name":"get_weather",
      "parameters":{"type":"object",
        "properties":{"city":{"type":"string"},"unit":{"type":"string"}},
        "required":["city"]}}}],
    "tool_choice":"auto"
  }'
# -> finish_reason "tool_calls", arguments {"city":"Mumbai","unit":"celsius"}
```

### Health / models
```bash
curl -s http://localhost:8202/health                                   # no auth
curl -s http://localhost:8202/v1/models -H "Authorization: Bearer $KEY"
```

---

## 2. Embeddings + Rerank — `:8204`  (no auth)

### `/v1/embeddings` (OpenAI-compatible, 384-d, multilingual)
```bash
curl -s http://localhost:8204/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"input":["Manthan runs native vLLM","नमस्ते दुनिया"],"input_type":"passage"}'
```
- `input`: string or array. `input_type`: `passage` (docs) | `query` (search) — sets e5 prefix.

### `/rerank` (cross-encoder)
```bash
curl -s http://localhost:8204/rerank -H 'Content-Type: application/json' \
  -d '{"query":"capital of France","documents":["bananas are yellow","Paris is the capital"],"top_n":2}'
```
- `top_n`: 0 = all, ranked. Returns `results[]` with `index` + `relevance_score` (0..1).

```bash
curl -s http://localhost:8204/health
```

---

## 3. RAG orchestrator — `:8205`  (no auth)

Hybrid pipeline: **BM25 + dense → RRF fuse → cross-encoder rerank → Self-RAG gate → cited generation** (matches the ATC Manthan v2.0 retrieval spec, with the laptop's lower test models).

### Ingest
```bash
curl -s http://localhost:8205/rag/ingest -H 'Content-Type: application/json' -d '{
  "documents":[
    {"text":"Manthan runs native vLLM on port 8082 and does NOT use Docker.","source":"ops","id":"doc1"},
    {"text":"Error code XR-4471 means the SeaweedFS volume server is unreachable.","source":"runbook"}
  ],
  "chunk_chars":900,"chunk_overlap":150
}'
```

### Query (retrieve → rerank → grounded, cited answer)
```bash
curl -s http://localhost:8205/rag/query -H 'Content-Type: application/json' -d '{
  "query":"Does Manthan use Docker and what port?",
  "mode":"hybrid","leg_topk":40,"rerank_top_n":8,"selfrag_min":0.10,
  "generate":true,"max_tokens":256
}'
```
- `mode`: `hybrid` (BM25+dense) | `dense` | `lexical`.
- `selfrag_min`: refuse if top rerank score < this (0 disables). `generate:false` → ranked `sources` only.
- Returns `{ sources[], gated, retrieval{dense,lexical,fused,top_score}, answer }`.

```bash
# retrieval-only — exact codes/IDs land via the BM25 leg
curl -s http://localhost:8205/rag/query -H 'Content-Type: application/json' \
  -d '{"query":"XR-4471","generate":false}'

curl -s http://localhost:8205/rag/reset -X POST     # clear the index
curl -s http://localhost:8205/health
```

---

## Managing the services (while roaming)

```bash
# status / health of all three
systemctl --user status biltiq-qwen biltiq-embed biltiq-rag
for p in 8202 8204 8205; do curl -s -o /dev/null -w ":$p %{http_code}\n" http://localhost:$p/health; done

# start / restart (Qwen init ~4.5 min; embed ~10 s; rag instant)
systemctl --user start   biltiq-embed biltiq-rag biltiq-qwen
systemctl --user restart biltiq-qwen

# logs
journalctl --user -u biltiq-rag -f
journalctl --user -u biltiq-qwen -f
```

- All three are **enabled** → auto-start on login. For start **before login**
  (headless boot): run once `sudo loginctl enable-linger atc`.
- Stopping a unit tears down its whole cgroup, so no stranded GPU memory.

## Optional / not in the default roaming set

- `:8201` `launch_e4b_serve.sh` — gemma-4-E4B (W4A4) + CPU embedding offload + tool
  calling. Less reliable for exact numerics than Qwen; start manually if needed.
- `:8203` llama.cpp CPU generator — not installed as a service (Qwen on GPU is the
  generator). `llama-cpp-python` is in the venv if you ever want a pure-CPU path.
