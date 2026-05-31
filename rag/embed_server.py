#!/usr/bin/env python3
"""
BiltIQ local CPU embedding + reranking service (zero-GPU, roaming).

OpenAI-compatible /v1/embeddings  +  Cohere/Jina-style /rerank, served from one
FastAPI app so the RAG orchestrator only talks to one port.

Models (CPU, multilingual to match BiltIQ's Hindi/Hinglish usage):
  embeddings : intfloat/multilingual-e5-small  (118M)  -> 384-dim
  reranker   : BAAI/bge-reranker-v2-m3          (568M, CrossEncoder)

e5 models are trained with task prefixes ("query: " / "passage: "). Retrieval
quality drops noticeably without them, so this server applies them based on an
`input_type` hint (default "passage") instead of forcing the caller to know.
"""
from __future__ import annotations

import os
import time
from typing import List, Literal, Union

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# --- config (env-overridable, sensible defaults) ---------------------------
EMBED_MODEL = os.environ.get("BILT_EMBED_MODEL", "intfloat/multilingual-e5-small")
RERANK_MODEL = os.environ.get("BILT_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
# Use the box's cores fully for CPU inference (leave 2 for the OS / Qwen).
_THREADS = max(1, (os.cpu_count() or 4) - 2)
torch.set_num_threads(int(os.environ.get("BILT_TORCH_THREADS", _THREADS)))

# Lazy globals so import is cheap; models load on startup event.
_embedder = None
_reranker = None

app = FastAPI(title="BiltIQ Local Embed+Rerank", version="1.0")


@app.on_event("startup")
def _load_models() -> None:
    global _embedder, _reranker
    from sentence_transformers import CrossEncoder, SentenceTransformer

    t0 = time.time()
    _embedder = SentenceTransformer(EMBED_MODEL, device="cpu")
    _reranker = CrossEncoder(RERANK_MODEL, device="cpu", max_length=512)
    print(
        f"[embed_server] loaded embed={EMBED_MODEL} rerank={RERANK_MODEL} "
        f"threads={torch.get_num_threads()} in {time.time()-t0:.1f}s",
        flush=True,
    )


# --- e5 prefix helper -------------------------------------------------------
def _prefix(texts: List[str], input_type: str) -> List[str]:
    """e5 wants 'query: ' on search queries and 'passage: ' on documents.
    Only applied to e5-family models; other models pass through untouched."""
    if "e5" not in EMBED_MODEL.lower():
        return texts
    tag = "query: " if input_type == "query" else "passage: "
    return [tag + t for t in texts]


# --- /v1/embeddings (OpenAI-compatible) ------------------------------------
class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: str = EMBED_MODEL
    encoding_format: Literal["float"] = "float"
    # Non-OpenAI extension: lets the RAG layer tag queries vs passages for e5.
    input_type: Literal["query", "passage"] = "passage"


@app.post("/v1/embeddings")
def embeddings(req: EmbeddingRequest):
    if _embedder is None:
        raise HTTPException(503, "model not loaded")
    texts = [req.input] if isinstance(req.input, str) else list(req.input)
    if not texts:
        raise HTTPException(400, "input is empty")
    vecs = _embedder.encode(
        _prefix(texts, req.input_type),
        normalize_embeddings=True,  # cosine == dot product downstream
        convert_to_numpy=True,
        batch_size=32,
    )
    data = [
        {"object": "embedding", "index": i, "embedding": v.tolist()}
        for i, v in enumerate(vecs)
    ]
    n_tokens = sum(len(t.split()) for t in texts)  # rough; we don't bill anyone
    return {
        "object": "list",
        "data": data,
        "model": req.model,
        "usage": {"prompt_tokens": n_tokens, "total_tokens": n_tokens},
    }


# --- /rerank (Cohere/Jina-style) -------------------------------------------
class RerankRequest(BaseModel):
    query: str
    documents: List[str]
    top_n: int = Field(default=0, description="0 = return all, ranked")
    return_documents: bool = True


@app.post("/rerank")
def rerank(req: RerankRequest):
    if _reranker is None:
        raise HTTPException(503, "model not loaded")
    if not req.documents:
        raise HTTPException(400, "documents is empty")
    pairs = [[req.query, d] for d in req.documents]
    scores = _reranker.predict(pairs, convert_to_numpy=True, batch_size=16)
    order = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)
    if req.top_n and req.top_n > 0:
        order = order[: req.top_n]
    results = []
    for rank, i in enumerate(order):
        item = {"index": i, "relevance_score": float(scores[i])}
        if req.return_documents:
            item["document"] = req.documents[i]
        results.append(item)
    return {"model": RERANK_MODEL, "results": results}


# --- health / models --------------------------------------------------------
@app.get("/health")
def health():
    ready = _embedder is not None and _reranker is not None
    return {"status": "ok" if ready else "loading"}


@app.get("/v1/models")
def models():
    return {
        "object": "list",
        "data": [
            {"id": EMBED_MODEL, "object": "model", "owned_by": "biltiq"},
            {"id": RERANK_MODEL, "object": "model", "owned_by": "biltiq"},
        ],
    }
