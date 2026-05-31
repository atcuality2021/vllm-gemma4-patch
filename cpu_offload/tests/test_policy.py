"""Unit tests for should_offload_embedding (pure policy logic, no GPU/vLLM model)."""
import types
import torch
import pytest

# import only the policy fn without triggering vLLM imports at module load
import importlib.util, pathlib
_src = pathlib.Path(__file__).resolve().parents[1] / "cpu_offload_embedding.py"


def _load_policy():
    # The module imports vllm at top level; if vllm is present this just works.
    spec = importlib.util.spec_from_file_location("_coe", _src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.should_offload_embedding


def _fake(numel, dtype=torch.bfloat16):
    w = types.SimpleNamespace(numel=lambda: numel,
                              element_size=lambda: torch.empty(0, dtype=dtype).element_size())
    return types.SimpleNamespace(weight=w)


should = _load_policy()


def test_ple_always_offloaded():
    # ~7GB per-layer table -> always True
    m = _fake(int(3.5e9))
    assert should("model.embed_tokens_per_layer", m, vram_budget_gb=8) is True


def test_lm_head_never_offloaded():
    m = _fake(int(0.67e9))
    assert should("lm_head", m, vram_budget_gb=8) is False
    assert should("model.lm_head", m, vram_budget_gb=2) is False


def test_main_table_offloaded_under_pressure():
    # 0.67B * 2 bytes = ~1.34GB. >15% of an 8GB budget (1.2GB) -> offload.
    m = _fake(int(0.67e9))
    assert should("model.embed_tokens", m, vram_budget_gb=8) is True


def test_main_table_kept_when_budget_large():
    # same table, but 24GB budget -> 15% = 3.6GB; 1.34GB < that -> keep resident.
    m = _fake(int(0.67e9))
    assert should("model.embed_tokens", m, vram_budget_gb=24) is False


def test_other_modules_resident():
    m = _fake(int(1e8))
    assert should("model.layers.0.altup_projections", m, vram_budget_gb=8) is False
