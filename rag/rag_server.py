#!/usr/bin/env python3
"""
BiltIQ local CPU RAG orchestrator (zero-GPU, roaming).

Implements the ATC Manthan v2.0 HYBRID retrieval pipeline with lower test models:

    lexical (BM25)  +  dense (e5 + FAISS)  --RRF fuse-->  rerank (bge cross-enc)
                                                          --> Self-RAG gate
                                                          --> cited generation

Spec maps (production model -> local test substitute):
    BM25F (Postgres pg_search)   -> rank_bm25 BM25Okapi over chunk text
    Qwen3-VL-Embedding-2B        -> intfloat/multilingual-e5-small (384-d)
    Qwen3-VL-Reranker-2B         -> BAAI/bge-reranker-v2-m3
    Qwen3.6-35B-A3B generation   -> Qwen3.5-4B on :8202 (GPU)

Surface:
    /rag/ingest   chunk -> embed(passage) -> FAISS + BM25
    /rag/query    BM25 + dense -> RRF -> rerank -> self-RAG gate -> chat
    /rag/reset    drop the index
    /health

Talks to:
  EMBED_BASE  (embed_server.py)  /v1/embeddings + /rerank
  CHAT_BASE   any OpenAI chat endpoint. Default = local Qwen :8202.

Stores: faiss IndexFlatIP over L2-normalized e5 vectors (== cosine) + an
in-process BM25 over the same chunk text. Both rebuild from chunks.jsonl on
startup, so persistence is just the FAISS index + the jsonl.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional

import re

import faiss
import httpx
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi

# --- config -----------------------------------------------------------------
EMBED_BASE = os.environ.get("BILT_EMBED_BASE", "http://localhost:8204")
CHAT_BASE = os.environ.get("BILT_CHAT_BASE", "http://localhost:8202/v1")
CHAT_MODEL = os.environ.get("BILT_CHAT_MODEL", "qwen3.5-4b")
DATA_DIR = Path(os.environ.get("BILT_RAG_DATA", "/home/atc/Desktop/bilt-rag/data"))
EMBED_DIM = int(os.environ.get("BILT_EMBED_DIM", "384"))  # e5-small = 384
DEFAULT_CHUNK_CHARS = int(os.environ.get("BILT_CHUNK_CHARS", "900"))
DEFAULT_CHUNK_OVERLAP = int(os.environ.get("BILT_CHUNK_OVERLAP", "150"))

# Hybrid retrieval knobs (spec defaults: RRF k=60, top-40 per leg, rerank top-8).
RRF_K = int(os.environ.get("BILT_RRF_K", "60"))
LEG_TOPK = int(os.environ.get("BILT_LEG_TOPK", "40"))      # candidates per leg
RERANK_TOP_N = int(os.environ.get("BILT_RERANK_TOP_N", "8"))
# Self-RAG gate: if the top rerank score (bge sigmoid, 0..1) is below this, the
# context is judged too weak -> refuse instead of hallucinating. 0 disables it.
SELFRAG_MIN = float(os.environ.get("BILT_SELFRAG_MIN", "0.10"))


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
_bm25: Optional[BM25Okapi] = None  # lexical leg, rebuilt from _chunks


def _tok(text: str) -> List[str]:
    """Unicode word tokens (lowercased). Python \\w matches Devanagari, so this
    tokenizes Hindi/Hinglish alongside English for the BM25 lexical leg."""
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def _build_bm25() -> None:
    """(Re)build the BM25 index over the full corpus. rank_bm25 has no cheap
    incremental add, so we rebuild on ingest — fine at laptop corpus sizes."""
    global _bm25
    _bm25 = BM25Okapi([_tok(c["text"]) for c in _chunks]) if _chunks else None


def _load_store() -> None:
    global _index, _chunks
    if _INDEX_PATH.exists() and _CHUNKS_PATH.exists():
        _index = faiss.read_index(str(_INDEX_PATH))
        _chunks = [json.loads(l) for l in _CHUNKS_PATH.read_text().splitlines() if l]
    else:
        _index = faiss.IndexFlatIP(EMBED_DIM)
        _chunks = []
    _build_bm25()


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


# --- hybrid retrieval: dense + lexical, fused with RRF ----------------------
def _dense_search(qvec: np.ndarray, k: int) -> List[int]:
    if not _chunks:
        return []
    _, idx = _index.search(qvec, min(k, len(_chunks)))
    return [int(i) for i in idx[0] if i >= 0]


def _lexical_search(query: str, k: int) -> List[int]:
    if _bm25 is None:
        return []
    scores = _bm25.get_scores(_tok(query))
    order = np.argsort(scores)[::-1][:k]
    # drop chunks with no lexical overlap at all (score 0) — they're noise
    return [int(i) for i in order if scores[i] > 0]


def _rrf(rankings: List[List[int]], k: int, pool: int) -> List[int]:
    """Reciprocal Rank Fusion: score(d) = sum 1/(k + rank_in_leg). Rank-based,
    so it merges legs whose scores live on totally different scales (BM25 vs
    cosine) without normalization — that's why hybrid search uses it."""
    fused: dict = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
    ordered = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in ordered[:pool]]


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
    _build_bm25()  # lexical leg must see the new chunks
    _persist()
    return {"ingested_documents": len(req.documents), "ingested_chunks": len(new_texts),
            "total_chunks": len(_chunks)}


class QueryRequest(BaseModel):
    query: str
    leg_topk: int = Field(default=LEG_TOPK, description="candidates pulled per leg")
    rerank_top_n: int = Field(default=RERANK_TOP_N, description="kept after reranking")
    mode: str = Field(default="hybrid", description="hybrid | dense | lexical")
    selfrag_min: float = Field(default=SELFRAG_MIN, description="0 disables the gate")
    generate: bool = True
    max_tokens: int = 512


REFUSAL = ("I don't have enough relevant information in the indexed documents to "
           "answer that confidently.")


@app.post("/rag/query")
def query(req: QueryRequest):
    if not _chunks:
        raise HTTPException(400, "index is empty — ingest documents first")

    # --- two retrieval legs, fused with RRF (the Manthan v2.0 hybrid pipeline) ---
    qv = _embed([req.query], "query")
    dense = _dense_search(qv, req.leg_topk) if req.mode in ("hybrid", "dense") else []
    lexical = _lexical_search(req.query, req.leg_topk) if req.mode in ("hybrid", "lexical") else []
    if req.mode == "hybrid":
        cand_idx = _rrf([dense, lexical], k=RRF_K, pool=req.leg_topk * 2)
    else:
        cand_idx = dense or lexical
    if not cand_idx:
        return {"query": req.query, "sources": [], "gated": True,
                "retrieval": {"dense": len(dense), "lexical": len(lexical)},
                **({"answer": REFUSAL} if req.generate else {})}

    # --- cross-encoder rerank the fused pool -> top-n ---
    reranked = _rerank(req.query, [_chunks[i]["text"] for i in cand_idx], req.rerank_top_n)
    hits = [{**_chunks[cand_idx[r["index"]]], "relevance_score": r["relevance_score"]}
            for r in reranked]

    # --- Self-RAG gate: weak top score -> refuse instead of hallucinate ---
    top_score = hits[0]["relevance_score"] if hits else 0.0
    gated = req.selfrag_min > 0 and top_score < req.selfrag_min

    result = {"query": req.query, "sources": hits, "gated": gated,
              "retrieval": {"dense": len(dense), "lexical": len(lexical),
                            "fused": len(cand_idx), "top_score": top_score}}
    if req.generate:
        if gated:
            result["answer"] = REFUSAL
        else:
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
    global _index, _chunks, _bm25
    _index = faiss.IndexFlatIP(EMBED_DIM)
    _chunks = []
    _bm25 = None
    _persist()
    return {"status": "reset", "total_chunks": 0}


@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(_chunks), "bm25": _bm25 is not None,
            "mode": "hybrid (bm25+dense+rrf+rerank+selfrag)",
            "embed_base": EMBED_BASE, "chat_base": CHAT_BASE}
