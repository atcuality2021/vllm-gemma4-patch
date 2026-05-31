#!/usr/bin/env python3
"""
BiltIQ local CPU RAG orchestrator (zero-GPU, roaming).

Ties the three pieces together behind one OpenAI-ish surface:

    /rag/ingest   chunk -> embed(passage) -> FAISS
    /rag/query    embed(query) -> FAISS top_k -> rerank -> stuff -> chat model
    /rag/reset    drop the index
    /health

Talks to:
  EMBED_BASE  (embed_server.py)  /v1/embeddings + /rerank
  CHAT_BASE   any OpenAI chat endpoint. Default = local Qwen :8202 (best quality).
              For a TRUE zero-GPU path set CHAT_BASE=http://localhost:8203/v1
              (the llama.cpp CPU server).

Vector store: faiss IndexFlatIP over L2-normalized e5 vectors == cosine. Small,
exact, no training. Persists to $BILT_RAG_DATA (index + chunks.jsonl) so ingest
survives restarts.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional

import faiss
import httpx
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# --- config -----------------------------------------------------------------
EMBED_BASE = os.environ.get("BILT_EMBED_BASE", "http://localhost:8204")
CHAT_BASE = os.environ.get("BILT_CHAT_BASE", "http://localhost:8202/v1")
CHAT_MODEL = os.environ.get("BILT_CHAT_MODEL", "qwen3.5-4b")
DATA_DIR = Path(os.environ.get("BILT_RAG_DATA", "/home/atc/Desktop/bilt-rag/data"))
EMBED_DIM = int(os.environ.get("BILT_EMBED_DIM", "384"))  # e5-small = 384
DEFAULT_CHUNK_CHARS = int(os.environ.get("BILT_CHUNK_CHARS", "900"))
DEFAULT_CHUNK_OVERLAP = int(os.environ.get("BILT_CHUNK_OVERLAP", "150"))


def _read_key() -> str:
    p = Path("/home/atc/.config/biltiq_vllm_api_key")
    return p.read_text().strip() if p.exists() else ""


CHAT_KEY = os.environ.get("BILT_CHAT_KEY") or _read_key()


# --- persistence ------------------------------------------------------------
DATA_DIR.mkdir(parents=True, exist_ok=True)
_INDEX_PATH = DATA_DIR / "faiss.index"
_CHUNKS_PATH = DATA_DIR / "chunks.jsonl"

_index: faiss.Index
_chunks: List[dict] = []  # parallel to index rows: {text, source, doc_id}


def _load_store() -> None:
    global _index, _chunks
    if _INDEX_PATH.exists() and _CHUNKS_PATH.exists():
        _index = faiss.read_index(str(_INDEX_PATH))
        _chunks = [json.loads(l) for l in _CHUNKS_PATH.read_text().splitlines() if l]
    else:
        _index = faiss.IndexFlatIP(EMBED_DIM)
        _chunks = []


def _persist() -> None:
    faiss.write_index(_index, str(_INDEX_PATH))
    with _CHUNKS_PATH.open("w") as f:
        for c in _chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


# ----------------------------------------------------------------------------
# CHUNKING — the key retrieval-quality knob.
#
# This default is sentence-aware with character-budget windows + overlap:
# it grows a window sentence-by-sentence up to `size` chars, then starts the
# next window `overlap` chars back so context spanning a boundary isn't lost.
#
# Trade-offs you may want to tune (env BILT_CHUNK_CHARS / BILT_CHUNK_OVERLAP):
#   - smaller chunks  -> sharper retrieval, but answers may miss surrounding context
#   - larger chunks   -> more context per hit, but dilutes the embedding's focus
#   - more overlap    -> fewer boundary misses, but more duplicate text + storage
# Swap this whole function if your corpus wants structure-aware splitting
# (markdown headings, code blocks, table rows) instead of prose sentences.
# ----------------------------------------------------------------------------
def chunk_document(text: str, size: int, overlap: int) -> List[str]:
    import re

    text = text.strip()
    if not text:
        return []
    # Split on sentence enders but keep them; works for en + romanized hi.
    sentences = re.split(r"(?<=[.!?।])\s+", text)
    chunks: List[str] = []
    cur = ""
    for s in sentences:
        if len(cur) + len(s) + 1 <= size or not cur:
            cur = f"{cur} {s}".strip()
        else:
            chunks.append(cur)
            # start next window `overlap` chars back from the end of this one
            cur = (cur[-overlap:] + " " + s).strip() if overlap else s
    if cur:
        chunks.append(cur)
    return chunks


# --- embed/rerank/chat clients ---------------------------------------------
_http = httpx.Client(timeout=120.0)


def _embed(texts: List[str], input_type: str) -> np.ndarray:
    r = _http.post(
        f"{EMBED_BASE}/v1/embeddings",
        json={"input": texts, "input_type": input_type},
    )
    r.raise_for_status()
    data = r.json()["data"]
    return np.array([d["embedding"] for d in data], dtype="float32")


def _rerank(query: str, docs: List[str], top_n: int) -> List[dict]:
    r = _http.post(
        f"{EMBED_BASE}/rerank",
        json={"query": query, "documents": docs, "top_n": top_n},
    )
    r.raise_for_status()
    return r.json()["results"]


def _chat(messages: List[dict], max_tokens: int) -> str:
    headers = {"Authorization": f"Bearer {CHAT_KEY}"} if CHAT_KEY else {}
    r = _http.post(
        f"{CHAT_BASE}/chat/completions",
        headers=headers,
        json={
            "model": CHAT_MODEL,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


# --- API --------------------------------------------------------------------
app = FastAPI(title="BiltIQ Local RAG", version="1.0")


@app.on_event("startup")
def _startup() -> None:
    _load_store()
    print(f"[rag_server] loaded {len(_chunks)} chunks, dim={EMBED_DIM}", flush=True)


class Doc(BaseModel):
    text: str
    source: Optional[str] = None
    id: Optional[str] = None


class IngestRequest(BaseModel):
    documents: List[Doc]
    chunk_chars: int = DEFAULT_CHUNK_CHARS
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP


@app.post("/rag/ingest")
def ingest(req: IngestRequest):
    new_texts, new_meta = [], []
    for d in req.documents:
        for ci, ch in enumerate(chunk_document(d.text, req.chunk_chars, req.chunk_overlap)):
            new_texts.append(ch)
            new_meta.append({"text": ch, "source": d.source, "doc_id": d.id, "chunk": ci})
    if not new_texts:
        raise HTTPException(400, "no chunks produced from documents")
    vecs = _embed(new_texts, "passage")
    _index.add(vecs)
    _chunks.extend(new_meta)
    _persist()
    return {"ingested_documents": len(req.documents), "ingested_chunks": len(new_texts),
            "total_chunks": len(_chunks)}


class QueryRequest(BaseModel):
    query: str
    top_k: int = Field(default=20, description="candidates pulled from FAISS")
    rerank_top_n: int = Field(default=5, description="kept after reranking")
    generate: bool = True
    max_tokens: int = 512


@app.post("/rag/query")
def query(req: QueryRequest):
    if not _chunks:
        raise HTTPException(400, "index is empty — ingest documents first")
    qv = _embed([req.query], "query")
    k = min(req.top_k, len(_chunks))
    scores, idx = _index.search(qv, k)
    cand = [_chunks[i] for i in idx[0] if i >= 0]
    reranked = _rerank(req.query, [c["text"] for c in cand], req.rerank_top_n)
    hits = [{**cand[r["index"]], "relevance_score": r["relevance_score"]} for r in reranked]

    result = {"query": req.query, "sources": hits}
    if req.generate:
        context = "\n\n".join(f"[{i+1}] {h['text']}" for i, h in enumerate(hits))
        messages = [
            {"role": "system", "content":
             "Answer the question using ONLY the numbered context. Cite sources as "
             "[n]. If the context does not contain the answer, say you don't know."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {req.query}"},
        ]
        result["answer"] = _chat(messages, req.max_tokens)
    return result


@app.post("/rag/reset")
def reset():
    global _index, _chunks
    _index = faiss.IndexFlatIP(EMBED_DIM)
    _chunks = []
    _persist()
    return {"status": "reset", "total_chunks": 0}


@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(_chunks),
            "embed_base": EMBED_BASE, "chat_base": CHAT_BASE}
