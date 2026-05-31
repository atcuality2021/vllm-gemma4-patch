"""
CPUOffloadVocabEmbedding — keep large embedding tables in CPU RAM, gather on demand.
================================================================================
BiltIQ AI · vllm-gemma4-patch

Problem
-------
Gemma 4 E-series (E2B/E4B) stores its Per-Layer Embeddings (PLE) plus the 262K-vocab
token table in BF16 — ~7 GB — and vLLM keeps them GPU-resident
(`gemma3n.py: self.embed_tokens_per_layer = VocabParallelEmbedding(...)`). On a small
GPU (e.g. 8 GB RTX 5060) that alone OOMs, even though the NVFP4 transformer is only ~2 GB.

Key insight
-----------
An embedding is a GATHER, not a matmul. Each forward only needs the rows for the
*active tokens* (num_tokens x dim — a few MB). So we keep the whole table in pinned
CPU RAM and copy only the gathered rows to the GPU. Negligible PCIe traffic, zero GPU
compute on the critical path. This is the cheapest possible offload — unlike expert or
block offload, there is no CPU-side matmul, only an index_select.

This composes with: NVFP4 (transformer weights, GPU) + manthanquant (KV cache, GPU).
The embedding table is the one big BF16 chunk that does NOT need to be resident.

Status: DRAFT. Forward path + CPU-pinned weight are implemented for the TP=1
(single-GPU laptop) case. See TODOs for: TP>1 sharded gather, zero-transient-GPU
allocation, and the offload-selection policy (left for the dev — see bottom).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from vllm.model_executor.utils import set_weight_attrs


class CPUOffloadVocabEmbedding(VocabParallelEmbedding):
    """Drop-in `VocabParallelEmbedding` whose weight lives in pinned CPU memory.

    The weight is created by the parent (on GPU, via `quant_method.create_weights`)
    and then moved to pinned host RAM in `__init__`, freeing the GPU copy. The
    forward gathers active rows on the CPU and ships only those to the device.

    Constraints / current scope:
      * Unquantized embeddings only (Gemma PLE + vocab table are BF16 — they are in
        the model's `exclude_modules` for NVFP4, so they arrive unquantized).
      * TP=1 fast path implemented. TP>1 falls back to a masked gather (see TODO).
    """

    def __init__(self, *args, pin_memory: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self._pin = pin_memory
        # EMPIRICAL FINDING (2026-05-31, gemma-4-E4B-it-NVFP4 on 8GB RTX 5060):
        # The OOM is a single 5.25 GiB allocation of the PLE table during
        # create_weights() — it happens at CONSTRUCTION, before any post-init move
        # could run, and vLLM's built-in cpu_offload_gb does NOT cover embeddings
        # (same OOM with cpu_offload_gb=7). So move-after-init is too late; the param
        # must be BORN on CPU. See register()/_patch_create_weights() below — that is
        # the integration that actually prevents the alloc. This move() is kept only
        # as a fallback for the (rare) case the table already fits transiently.
        self._move_weight_to_cpu()

    def _move_weight_to_cpu(self) -> None:
        w = self.weight.data
        if w.device.type != "cpu":
            host = torch.empty(w.shape, dtype=w.dtype, device="cpu",
                               pin_memory=self._pin)
            host.copy_(w)
            new = torch.nn.Parameter(host, requires_grad=False)
            # Re-attach vLLM's loader metadata so checkpoint loading still targets us.
            # The safetensors shard is on CPU, so weight_loader's copy_ stays CPU->CPU
            # and never allocates on the GPU.
            set_weight_attrs(new, {"weight_loader": self.weight_loader})
            self.weight = new
            del w
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def forward(self, input_: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Gather active rows on CPU, return them on the input's device."""
        device = input_.device
        if self.tp_size == 1:
            # Fast path: full vocab on this rank, no masking / all-reduce.
            ids_cpu = input_.to("cpu", non_blocking=True).long()
            rows = F.embedding(ids_cpu, self.weight)          # CPU index_select
            return rows.to(device, non_blocking=True)         # few-MB H2D copy

        # TODO(TP>1): replicate parent's masking. Tokens outside this rank's vocab
        # shard must map to row 0 and be zeroed, then all-reduced across ranks:
        #   masked_input, mask = get_masked_input_and_mask(input_, ...)
        #   rows = F.embedding(masked_input.cpu().long(), self.weight).to(device)
        #   rows.masked_fill_(mask.unsqueeze(-1), 0.0)
        #   return tensor_model_parallel_all_reduce(rows)
        raise NotImplementedError(
            "CPUOffloadVocabEmbedding currently supports tensor_parallel_size=1; "
            "implement the masked gather above for TP>1."
        )


def swap_embeddings_to_cpu(model: torch.nn.Module, should_offload) -> int:
    """Replace selected `VocabParallelEmbedding` modules with CPU-offloaded ones.

    Walks the model, and for every `VocabParallelEmbedding` for which
    `should_offload(name, module)` returns True, swaps in a
    `CPUOffloadVocabEmbedding` carrying the same config + (moved) weight.

    Returns the count of modules swapped. Call AFTER model construction but, ideally,
    BEFORE checkpoint weights are loaded (so the big table never lands on the GPU).
    """
    swapped = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, VocabParallelEmbedding) and not isinstance(
            module, CPUOffloadVocabEmbedding
        ) and should_offload(name, module):
            parent_path, _, attr = name.rpartition(".")
            parent = model.get_submodule(parent_path) if parent_path else model
            new = CPUOffloadVocabEmbedding(
                num_embeddings=module.org_vocab_size
                if hasattr(module, "org_vocab_size") else module.num_embeddings,
                embedding_dim=module.embedding_dim,
                params_dtype=module.weight.dtype,
                prefix=name,
            )
            # carry over already-loaded weights, if any
            with torch.no_grad():
                new.weight.data.copy_(module.weight.data.to("cpu"))
            setattr(parent, attr, new)
            swapped += 1
    return swapped


# ──────────────────────────────────────────────────────────────────────────────
# vLLM plugin entry point — prevent the GPU allocation at CONSTRUCTION time.
#
# Empirically (gemma-4-E4B-it-NVFP4 on 8GB), the OOM is a 5.25 GiB embedding alloc
# inside UnquantizedEmbeddingMethod.create_weights — and vLLM's cpu_offload_gb does
# NOT cover it. So we patch create_weights to build large embedding tables under a
# CPU device context (born on CPU, never on GPU), then bind a gather-forward.
#
# Wire up via pyproject.toml:
#   [project.entry-points."vllm.general_plugins"]
#   cpu_offload = "cpu_offload.cpu_offload_embedding:register"
#
# STATUS: integration drafted; needs one live load-test on the 8GB box to confirm the
# CPU device-context overrides vLLM's build device (the policy + forward + unit tests
# are validated; this construction-time patch is the unvalidated piece).
# ──────────────────────────────────────────────────────────────────────────────

# ============================================================================
# ROOT CAUSE PINNED (2026-05-31, E4B-NVFP4 + FLASH_ATTN, no JIT masking):
# The 5.25 GiB OOM is NOT in create_weights. It is in vLLM's POST-LOAD move:
#   vllm/model_executor/model_loader/utils.py:144  device_loading_context
#     p.data = p.data.to(target_device)   # drags our CPU embedding back to GPU
# called from process_weights_after_loading(). So building on CPU is necessary
# but INSUFFICIENT — this global .to(cuda) sweep re-uploads it.
# REMAINING FIXES (all located):
#   1. (done) build large embedding on CPU in create_weights
#   2. patch device_loading_context to SKIP params tagged _cpu_offload=True
#   3. patch gemma4/gemma3n get_per_layer_input_embeddings() forward call site
#      to gather-on-CPU -> H2D (the weight is on CPU, forward must not index on GPU)
# Tag the param in create_weights: set_weight_attrs(weight, {"_cpu_offload": True})
# ============================================================================
_ORIG_CREATE_WEIGHTS = None
LARGE_EMBED_GB = 1.0  # tables bigger than this are built on CPU


def register() -> None:
    """vLLM general-plugin hook: install the create-on-CPU patch for big embeddings.

    IMPORTANT (validated 2026-05-31): vLLM v1 loads the model in a separate EngineCore
    subprocess, so an in-process register() call does NOT take effect — the patch must
    be loaded via the `vllm.general_plugins` entry point so vLLM runs it inside the
    worker. A manual register() in the parent process still OOMs (confirmed).
    """
    global _ORIG_CREATE_WEIGHTS
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        UnquantizedEmbeddingMethod,
    )
    if getattr(UnquantizedEmbeddingMethod, "_biltiq_cpu_offload", False):
        return
    _ORIG_CREATE_WEIGHTS = UnquantizedEmbeddingMethod.create_weights

    def create_weights(self, layer, input_size_per_partition, output_partition_sizes,
                       input_size, output_size, params_dtype, weight_loader=None, **kw):
        est_gb = (sum(output_partition_sizes) * input_size_per_partition
                  * torch.empty(0, dtype=params_dtype).element_size()) / 1024**3
        if est_gb >= LARGE_EMBED_GB:
            # Build the param DIRECTLY on CPU, bypassing the original (which allocates
            # on GPU with an explicit device= a context-manager cannot override).
            rows = sum(output_partition_sizes)
            host = torch.empty(rows, input_size_per_partition,
                               dtype=params_dtype, device="cpu")
            try:
                host = host.pin_memory()
            except Exception:
                pass
            weight = torch.nn.Parameter(host, requires_grad=False)
            set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0,
                                      "weight_loader": weight_loader,
                                      "_cpu_offload": True})
            layer.register_parameter("weight", weight)
            _bind_cpu_forward(layer)
        else:
            _ORIG_CREATE_WEIGHTS(self, layer, input_size_per_partition,
                                 output_partition_sizes, input_size, output_size,
                                 params_dtype, weight_loader=weight_loader, **kw)

    UnquantizedEmbeddingMethod.create_weights = create_weights
    UnquantizedEmbeddingMethod._biltiq_cpu_offload = True

    # Fix #2: stop process_weights_after_loading -> device_loading_context from
    # dragging our CPU embedding back to the GPU (utils.py:144 p.data.to(cuda)).
    from contextlib import contextmanager
    from vllm.model_executor.model_loader import utils as _ld
    if not getattr(_ld, "_biltiq_dlc_patched", False):
        @contextmanager
        def _dlc(module, target_device):
            if target_device.type == "cpu":
                yield module; return
            original = {}
            for name, p in module.named_parameters():
                if p.device.type == "cpu" and not getattr(p, "_cpu_offload", False):
                    original[name] = p.device
                    p.data = p.data.to(target_device)
            try:
                yield module
            finally:
                for name, p in module.named_parameters():
                    if name in original:
                        p.data = p.data.to(original[name])
        _ld.device_loading_context = _dlc
        _ld._biltiq_dlc_patched = True


def _bind_cpu_forward(layer) -> None:
    """Replace the layer's forward with a CPU-gather that returns GPU rows."""
    def forward(input_: torch.Tensor) -> torch.Tensor:
        ids_cpu = input_.to("cpu", non_blocking=True).long()
        rows = F.embedding(ids_cpu, layer.weight)
        return rows.to(input_.device, non_blocking=True)
    layer.forward = forward


# ──────────────────────────────────────────────────────────────────────────────
# DEV CONTRIBUTION REQUESTED — the offload-selection policy
# ──────────────────────────────────────────────────────────────────────────────
def should_offload_embedding(name: str, module: VocabParallelEmbedding,
                             vram_budget_gb: float) -> bool:
    """Decide whether THIS embedding table should be pushed to CPU.

    This is the real design decision, and it's a memory-vs-latency trade-off that
    depends on your deployment — that's why it's yours to set, not a fixed rule:

      * The PLE table (`embed_tokens_per_layer`) is the big win — ~7 GB, pure lookup,
        almost free to offload. Almost always YES on a small GPU.
      * The main token table (`embed_tokens`) is hit every step too; offloading it
        adds a tiny H2D per step. Offload it only if you still need the VRAM.
      * `lm_head` (if untied) is a MATMUL over the full vocab, NOT a gather —
        offloading it means a big GEMM with a CPU-resident weight (expensive). Usually
        keep on GPU. Decide based on whether it's tied to embed_tokens.

    Default policy (implemented below): offload PLE always, the main token table only
    under VRAM pressure (> 15% of the budget), and never the lm_head GEMM.
    """
    lname = name.lower()

    # lm_head is a matmul over the vocab, not a gather — offloading it would run a big
    # GEMM against CPU-resident weights every step. Keep it on the GPU.
    if "lm_head" in lname:
        return False

    size_gb = (module.weight.numel() * module.weight.element_size()) / (1024 ** 3)

    # Per-Layer Embeddings (Gemma E-series): the dominant BF16 table and a pure lookup.
    # Always the best win on a small GPU — offload unconditionally.
    if "per_layer" in lname:
        return True

    # Main token-embedding table: also a pure gather, but hit every step, so the small
    # H2D copy recurs. Offload only when it's a meaningful slice of the VRAM budget,
    # i.e. when keeping it resident actually costs us headroom we need.
    if "embed_tokens" in lname or "embed" in lname:
        return size_gb > 0.15 * max(vram_budget_gb, 1e-6)

    # Anything else (norms, projections, altup, etc.) stays resident.
    return False
