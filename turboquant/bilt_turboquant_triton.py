"""TurboQuant 3-bit KV compression for vLLM's TritonAttentionImpl (no nvcc).
================================================================================
BiltIQ AI · port of manthanquant/turboquant to the Triton attention backend.

Why
---
manthanquant hooks `FlashAttentionImpl` and serves compressed decode via a
COMPILED CUDA kernel (`_C.fused_attention`). Gemma-4 E-series is forced onto
`TritonAttentionImpl` (heterogeneous head dims 256 vs 512), and this laptop has
no CUDA toolkit (`_C` cannot be built). So manthanquant is a no-op here.

This module ports the same idea to Triton WITHOUT nvcc:
  * PREFILL: run the original Triton attention (real paged KV cache stays correct),
    then 3-bit Lloyd-Max compress that layer's KV into a CPU shadow cache (pure
    numpy — `manthanquant.cpu_quantize.tq_encode_numpy`).
  * DECODE:  reconstruct K/V from the shadow cache (`tq_decode_numpy`) and run
    attention in PURE PYTORCH (GQA-expand + sliding-window slice + optional
    logits soft-cap), replacing the CUDA kernel.

Correctness-first: we only take the compressed-decode path for the clean
single-token, single-sequence decoder case with no attention sinks; anything
unusual falls back to the original Triton kernel, so we never emit wrong output.

Scope note: on a DISCRETE GPU this proves compression + correctness + telemetry,
but does NOT by itself free VRAM — the original paged cache is still allocated
(kept for the fallback path). Real VRAM reduction needs vLLM's cache allocation
shrunk; that's a follow-on. On GB10 unified memory the shadow cache is ~free.

Activate: set BILT_TURBOQUANT=1 and call register() (done by sitecustomize).
Needs `manthanquant` on PYTHONPATH (for cpu_quantize).
"""
from __future__ import annotations

import os
import numpy as np
import torch

BITS = 3
_orig_forward = None
_orig_kv_update = None
_installed = False

# id(impl) -> per-layer state
_shadow: dict[int, "LayerCache"] = {}
_pending: dict[int, tuple[np.ndarray, np.ndarray]] = {}
_first_id = 0
_last_was_decode = False
_stats = {"prefill": 0, "decode_fused": 0, "decode_fallback": 0,
          "orig_bytes": 0, "comp_bytes": 0, "layers": 0}


class LayerCache:
    """Per-layer 3-bit-compressed shadow KV for ONE sequence (max_num_seqs=1)."""
    __slots__ = ["k_radii", "k_packed", "v_radii", "v_packed", "seq_len",
                 "orig_bytes", "comp_bytes"]

    def __init__(self):
        self.clear()

    def clear(self):
        self.k_radii, self.k_packed = [], []
        self.v_radii, self.v_packed = [], []
        self.seq_len = 0
        self.orig_bytes = 0
        self.comp_bytes = 0

    def append(self, key_np, value_np):
        """key_np/value_np: float32 [tokens, kv_heads, head_dim]."""
        from manthanquant.cpu_quantize import tq_encode_numpy
        t, kh, d = key_np.shape
        self.orig_bytes += 2 * t * kh * d * 2  # bf16 K+V
        kr, kp = tq_encode_numpy(key_np.reshape(-1, d).astype(np.float32), bits=BITS)
        vr, vp = tq_encode_numpy(value_np.reshape(-1, d).astype(np.float32), bits=BITS)
        self.comp_bytes += 2 * (kr.nbytes + kp.nbytes)
        self.k_radii.append(kr.reshape(t, kh))
        self.k_packed.append(kp.reshape(t, kh, -1))
        self.v_radii.append(vr.reshape(t, kh))
        self.v_packed.append(vp.reshape(t, kh, -1))
        self.seq_len += t

    def stacked(self):
        if self.seq_len == 0:
            return None
        return (np.concatenate(self.k_radii, 0), np.concatenate(self.k_packed, 0),
                np.concatenate(self.v_radii, 0), np.concatenate(self.v_packed, 0))


def _decompress(radii, packed, d, device, dtype):
    """radii [S,KH], packed [S,KH,W] -> torch [S,KH,d] on device."""
    from manthanquant.cpu_quantize import tq_decode_numpy
    s, kh = radii.shape
    vecs = tq_decode_numpy(radii.reshape(-1), packed.reshape(-1, packed.shape[-1]),
                           d, bits=BITS)  # [S*KH, d] float32
    return torch.from_numpy(vecs.reshape(s, kh, d)).to(device, dtype)


def _compressed_decode(self, query, output, n):
    """Pure-PyTorch attention over the 3-bit shadow cache. Returns output or None."""
    cache = _shadow.get(id(self))
    if cache is None:
        return None
    stk = cache.stacked()
    if stk is None:
        return None
    kr, kp, vr, vp = stk
    d = self.head_size
    dev, dt = query.device, query.dtype
    K = _decompress(kr, kp, d, dev, dt)          # [S, KH, d]
    V = _decompress(vr, vp, d, dev, dt)
    S = K.shape[0]

    # sliding window: decoder token attends to last (left+1) keys; (-1) = full
    left = self.sliding_window[0] if isinstance(self.sliding_window, (tuple, list)) \
        else self.sliding_window
    if left is not None and left >= 0:
        w = int(left) + 1
        if S > w:
            K, V, S = K[S - w:], V[S - w:], w

    rep = self.num_queries_per_kv
    if rep > 1:                                   # GQA: expand KV heads to query heads
        K = K.repeat_interleave(rep, dim=1)
        V = V.repeat_interleave(rep, dim=1)
    H = self.num_heads
    q = query[:n].to(dt)
    if q.dim() == 2:                              # vLLM passes [n, H*d] -> [n, H, d]
        q = q.view(n, H, K.shape[-1])
    qh = q.transpose(0, 1)                        # [H, n, d]
    kh = K.transpose(0, 1)                        # [H, S, d]
    vh = V.transpose(0, 1)                        # [H, S, d]
    scores = torch.matmul(qh, kh.transpose(1, 2)).float() * self.scale  # [H, n, S]
    cap = getattr(self, "logits_soft_cap", None)
    if cap:
        scores = cap * torch.tanh(scores / cap)
    attn = torch.softmax(scores, dim=-1).to(dt)
    out = torch.matmul(attn, vh)                  # [H, n, d]
    out = out.transpose(0, 1).reshape(n, H * K.shape[-1])  # [n, H*d]
    if output is not None:
        output[:n] = out.reshape(output[:n].shape).to(output.dtype)
        return output
    return out.to(dt)


def _flush_layer(self):
    """Compress this layer's queued KV (current step) into the shadow cache."""
    kv = _pending.pop(id(self), None)
    if kv is None:
        return
    k_np, v_np = kv
    cache = _shadow.get(id(self))
    if cache is None:
        cache = _shadow[id(self)] = LayerCache()
        _stats["layers"] = len(_shadow)
    cache.append(k_np, v_np)


def _maybe_new_request(self, is_decode):
    global _last_was_decode, _first_id
    if _first_id == 0:
        _first_id = id(self)
    if id(self) == _first_id:
        if (not is_decode) and _last_was_decode:
            for c in _shadow.values():
                c.clear()
            _last_was_decode = False
        if is_decode:
            _last_was_decode = True


def _patch_impl(T):
    """Patch one attention Impl class (Triton or FlashAttention). Each gets its OWN
    captured originals so both backends can be hooked simultaneously."""
    if getattr(T, "_bilt_turboquant", False):
        return False
    orig_forward = T.forward
    orig_kv_update = T.do_kv_cache_update

    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        r = orig_kv_update(self, layer, key, value, kv_cache, slot_mapping)
        try:
            n = slot_mapping.size(0)
            if 0 < n <= 8192:  # skip giant profiling batches
                _pending[id(self)] = (
                    key[:n].detach().float().cpu().numpy(),
                    value[:n].detach().float().cpu().numpy())
        except Exception:
            pass
        return r

    def forward(self, layer, query, key, value, kv_cache, attn_metadata,
                output=None, output_scale=None, output_block_scale=None):
        if attn_metadata is None:
            return orig_forward(self, layer, query, key, value, kv_cache,
                                attn_metadata, output, output_scale, output_block_scale)
        n = attn_metadata.num_actual_tokens
        is_decode = (attn_metadata.max_query_len == 1 and n <= 4)
        _maybe_new_request(self, is_decode)
        _flush_layer(self)  # compress current step's KV into shadow

        at = getattr(self, "attn_type", None)
        decoder = (at is None) or str(getattr(at, "value", at)).lower() == "decoder"
        # Layers without their own shadow cache (e.g. Gemma KV-sharing layers) just
        # fall back to the original kernel reading the real (shared) cache.
        can_fuse = (is_decode and n == 1 and decoder
                    and getattr(self, "sinks", None) is None)
        if can_fuse:
            try:
                out = _compressed_decode(self, query, output, n)
                if out is not None:
                    _stats["decode_fused"] += 1
                    if _stats["decode_fused"] % 256 == 0:
                        _dump_stats()
                    return out
            except Exception:
                pass
            _stats["decode_fallback"] += 1
        else:
            _stats["prefill"] += 1
        return orig_forward(self, layer, query, key, value, kv_cache,
                            attn_metadata, output, output_scale, output_block_scale)

    T.forward = forward
    T.do_kv_cache_update = do_kv_cache_update
    T._bilt_turboquant = True
    return True


def register():
    """Patch BOTH TritonAttentionImpl and FlashAttentionImpl for 3-bit KV compression.

    Triton backend => Gemma-4 E4B (heterogeneous head dims). FlashAttention backend
    => standard dense models (Qwen etc.). Hooking both means the compression engages
    regardless of which backend vLLM selects for the served model.
    """
    global _installed
    if _installed or os.environ.get("BILT_TURBOQUANT") != "1":
        return
    patched = []
    for modpath, clsname in (
        ("vllm.v1.attention.backends.triton_attn", "TritonAttentionImpl"),
        ("vllm.v1.attention.backends.flash_attn", "FlashAttentionImpl"),
    ):
        try:
            mod = __import__(modpath, fromlist=[clsname])
            cls = getattr(mod, clsname)
            if _patch_impl(cls):
                patched.append(clsname)
        except Exception:
            pass
    if not patched:
        return
    _installed = True
    try:
        flag = os.path.expanduser("~/logs/bilt_turboquant_active.flag")
        os.makedirs(os.path.dirname(flag), exist_ok=True)
        with open(flag, "a") as f:
            f.write(f"pid={os.getpid()} patched={'+'.join(patched)}\n")
    except Exception:
        pass


def _dump_stats():
    try:
        import json
        with open(os.path.expanduser("~/logs/bilt_turboquant_stats.json"), "w") as f:
            json.dump(stats(), f, indent=2)
    except Exception:
        pass


def stats():
    s = dict(_stats)
    s["ratio"] = (s["orig_bytes"] / s["comp_bytes"]) if s["comp_bytes"] else 0.0
    s["shadow_orig_bytes"] = sum(c.orig_bytes for c in _shadow.values())
    s["shadow_comp_bytes"] = sum(c.comp_bytes for c in _shadow.values())
    if s["shadow_comp_bytes"]:
        s["shadow_ratio"] = s["shadow_orig_bytes"] / s["shadow_comp_bytes"]
    return s
