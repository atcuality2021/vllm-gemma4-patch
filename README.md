# vllm-gemma4-patch

**Gemma 4 support for vLLM** — native in **vLLM ≥ 0.19.0**; legacy backport of [PR #38826](https://github.com/vllm-project/vllm/pull/38826) for vLLM 0.18.x.

## Status (updated 2026-05-31)

Gemma 4 support ([PR #38826](https://github.com/vllm-project/vllm/pull/38826)) **merged to vLLM `main` on 2026-04-02** and first shipped in **v0.19.0 (2026-04-03)**. It is built in to every release since — verified against the latest stable **v0.22.0 (2026-05-29)**.

| Your vLLM | What this repo does |
|-----------|---------------------|
| **≥ 0.19.0** (recommended) | Gemma 4 is **native** — no patching. `patch.sh` detects this, verifies the model registers, and exits. Just `pip install -U vllm`. |
| **0.18.x** (legacy) | Runs the original backport: copies the gemma4 files from `main` and fixes import paths. Documented below. |

The whole Gemma 4 family routes through `Gemma4ForConditionalGeneration` / `Gemma4ForCausalLM`, including the **E-series (E2B/E4B)** — so no separate handling is needed for the PLE/"effective-parameter" variants.

## Why this still exists

The repo began (April 2026) as a backport because PR #38826 was only on `main`. That job is done upstream now, so the script's primary mode is a **version-aware verifier** for native installs, with the backport kept for anyone pinned to 0.18.x.

## Supported models

| Model | Parameters | Type |
|-------|-----------|------|
| `google/gemma-4-31B-it` | 31B | Instruction-tuned, multimodal |
| `google/gemma-4-12B-it` | 12B | Instruction-tuned, multimodal |
| `google/gemma-4-31B-pt` | 31B | Pretrained |
| `google/gemma-4-12B-pt` | 12B | Pretrained |
| `google/gemma-4-4B-it` | 4B | Instruction-tuned |
| `google/gemma-4-4B-pt` | 4B | Pretrained |
| `google/gemma-4-1B-it` | 1B | Instruction-tuned |
| `google/gemma-4-E4B-it` | 4.5B eff / 8B | Instruction-tuned, multimodal (E-series, PLE) |
| `google/gemma-4-E2B-it` | 2.3B eff / 5.1B | Instruction-tuned, multimodal (E-series, PLE) |
| `google/gemma-4-E4B-it-assistant` | 4.5B eff | MTP draft model for speculative decoding (`Gemma4MTPModel`) |
| `google/gemma-4-E2B-it-assistant` | 2.3B eff | MTP draft model for speculative decoding |

The **E-series** (E2B/E4B) declares `architectures: ["Gemma4ForConditionalGeneration"]`, so it loads on the same native code path as the dense models — no special patch. The `-it-assistant` MTP drafters are served via vLLM speculative decoding (`Gemma4MTPModel`, native ≥ 0.19.0).

See [SUPPORTED_MODELS.md](SUPPORTED_MODELS.md) for recommended launch configurations per model size.

## Supported platforms

- **x86_64** (standard Linux GPU servers, NVIDIA A100/H100/RTX 4090/etc.)
- **aarch64 / ARM64** (NVIDIA DGX Spark GB10, Jetson, Grace Hopper)

## Prerequisites

- **vLLM** in a Python virtualenv — **≥ 0.19.0 recommended** (native), or 0.18.x for the legacy backport
- **Git** and **pip** available in the target virtualenv
- **Internet access** — only needed for the 0.18.x backport (clones vLLM `main`)

## Quick start — vLLM ≥ 0.19.0 (recommended)

Gemma 4 is native; nothing to patch. Just make sure you're current and verify:

```bash
pip install -U vllm                 # 0.22.0+ has full Gemma 4 family
./patch.sh /path/to/your/vllm-venv  # detects native support, verifies, exits 0
```

Then serve any Gemma 4 model directly, e.g. the E4B multimodal on an 8 GB GPU:

```bash
vllm serve google/gemma-4-E4B-it \
  --trust-remote-code --gpu-memory-utilization 0.90 \
  --max-model-len 16384 --max-num-seqs 4
```

## Legacy backport — vLLM 0.18.x only

```bash
git clone https://github.com/ATC-Labs/vllm-gemma4-patch.git
cd vllm-gemma4-patch
chmod +x patch.sh verify.sh

# Patch your vLLM installation (pass venv path as argument)
./patch.sh /path/to/your/vllm-venv

# Verify the patch
./verify.sh /path/to/your/vllm-venv
```

One-liner if you already know your venv path:

```bash
git clone https://github.com/ATC-Labs/vllm-gemma4-patch.git && cd vllm-gemma4-patch && ./patch.sh ~/vllm-env
```

## What the legacy backport does (vLLM 0.18.x)

> Only runs when the detected vLLM is < 0.19.0. On ≥ 0.19.0 the script skips all of this and just verifies native support.

The backport performs 6 steps, each idempotent (safe to re-run):

### Step 1: Upgrade transformers

Installs `huggingface_hub` (latest) and `transformers` from GitHub main. The `gemma4` model type is not in any stable transformers release, so the development version is required to load Gemma 4 configs and tokenizers.

### Step 2: Clone vLLM main

Performs a shallow clone of vLLM's main branch to a temporary directory. This is the source for all Gemma 4 model implementation files from PR #38826.

### Step 3: Copy Gemma 4 model files (12+ files)

Copies the following from vLLM main into your installed vLLM package:

| File | Location | Purpose |
|------|----------|---------|
| `gemma4.py` | `model_executor/models/` | Text-only Gemma 4 CausalLM implementation |
| `gemma4_mm.py` | `model_executor/models/` | Multimodal Gemma 4 (vision + text) |
| `gemma4_utils.py` | `model_executor/models/` | Shared utilities for Gemma 4 models |
| `gemma4_rope.py` | `model_executor/layers/rotary_embedding/` | Proportional RoPE for Gemma 4 |
| `__init__.py` | `model_executor/layers/rotary_embedding/` | Updated RoPE registry with Gemma 4 |
| `gemma4_reasoning_parser.py` | `reasoning/` | Reasoning/thinking block parser |
| `gemma4_utils.py` | `reasoning/` | Reasoning utilities |
| `__init__.py` | `reasoning/` | Updated reasoning parser registry |
| `gemma4_tool_parser.py` | `tool_parsers/` | Function-calling / tool use parser |
| `gemma4_utils.py` | `tool_parsers/` | Tool parser utilities |
| `__init__.py` | `tool_parsers/` | Updated tool parser registry |
| `model_arch_config_convertor.py` | `transformers_utils/` | Architecture config converter |

Additionally applies a **compatibility fix**: replaces `from vllm.inputs import MultiModalDataDict` with `from vllm.multimodal.inputs import MultiModalDataDict` in `gemma4_mm.py`, since the import path changed between vLLM main and 0.18.x.

### Step 4: Patch model registry

Adds two entries to `model_executor/models/registry.py`:

```python
"Gemma4ForCausalLM": ("gemma4", "Gemma4ForCausalLM"),
"Gemma4ForConditionalGeneration": ("gemma4_mm", "Gemma4ForConditionalGeneration"),
```

These are inserted after the existing Gemma3 entries so vLLM can dispatch Gemma 4 model architectures.

### Step 5: Patch base.py for null sub_configs

Gemma 4 declares `audio_config` in its HuggingFace config, but it is `null` (the model does not process audio). The transformer model loader in vLLM iterates sub_configs and crashes on `None`. This patch adds a `continue` guard:

```python
if sub_config is None:
    continue
```

### Step 6: Patch utils.py for named buffers

Gemma 4 registers `layer_scalar` as a **buffer** (via `register_buffer`), not a parameter. vLLM's weight loader only looks at `named_parameters()`, so `layer_scalar` is silently skipped, causing incorrect outputs. This patch adds a `named_buffers()` sweep to `_add_loadable_non_param_tensors`:

```python
for buf_name, buf_tensor in module.named_buffers(recurse=False):
    if buf_name not in child_params:
        child_params[buf_name] = buf_tensor
```

## Architecture notes

Gemma 4 introduces several architectural innovations:

- **Asymmetric KV heads**: `attention_k_eq_v=True` with `global_head_dim=512` but `head_dim=256`. Keys and values share a 512-dim head while queries use 256-dim heads.
- **Proportional RoPE**: Custom rotary position embeddings with per-layer frequency scaling (implemented in `gemma4_rope.py`).
- **Sliding + full attention mix**: Alternating layers use local sliding window attention and full global attention.
- **Vision encoder**: SigLIP-based vision tower for multimodal variants, processing images into visual tokens.
- **Thinking/reasoning**: Native support for `<think>...</think>` blocks with a dedicated reasoning parser.
- **Tool calling**: Built-in function-calling support with a Gemma 4-specific tool parser.

## Verification

After patching, verify with:

```bash
./verify.sh /path/to/your/vllm-venv
```

Or manually:

```bash
source /path/to/your/vllm-venv/bin/activate

# Check transformers has Gemma4
python -c "from transformers.models.gemma4 import Gemma4Config; print('transformers: OK')"

# Check vLLM model imports
python -c "from vllm.model_executor.models.gemma4 import Gemma4ForCausalLM; print('gemma4: OK')"
python -c "from vllm.model_executor.models.gemma4_mm import Gemma4ForConditionalGeneration; print('gemma4_mm: OK')"

# Check registry
python -c "
from vllm.model_executor.models.registry import _VLLM_MODELS
assert 'Gemma4ForCausalLM' in _VLLM_MODELS, 'Not in registry'
print('registry: OK')
"

# Check RoPE
python -c "from vllm.model_executor.layers.rotary_embedding.gemma4_rope import *; print('gemma4_rope: OK')"
```

## Launch examples

### gemma-4-31B-it on DGX Spark (GB10, 128GB unified memory)

```bash
vllm serve /path/to/gemma-4-31B-it \
  --trust-remote-code --enforce-eager \
  --gpu-memory-utilization 0.55 --max-model-len 8192 \
  --max-num-seqs 4 --enable-prefix-caching
```

### gemma-4-31B-it on A100 80GB

```bash
vllm serve google/gemma-4-31B-it \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 --max-model-len 16384 \
  --max-num-seqs 16 --enable-prefix-caching
```

### gemma-4-12B-it on RTX 4090 24GB (4-bit quantized)

```bash
vllm serve google/gemma-4-12B-it \
  --trust-remote-code \
  --quantization awq \
  --gpu-memory-utilization 0.90 --max-model-len 8192 \
  --max-num-seqs 8
```

### gemma-4-4B-it on any GPU (fits in 16GB+)

```bash
vllm serve google/gemma-4-4B-it \
  --trust-remote-code \
  --gpu-memory-utilization 0.90 --max-model-len 16384 \
  --max-num-seqs 32
```

### gemma-4-E4B-it on an 8GB laptop GPU (RTX 5060, multimodal)

```bash
# E-series multimodal (text + image + audio in, text out). Native on vLLM >= 0.19.0.
vllm serve google/gemma-4-E4B-it \
  --trust-remote-code \
  --gpu-memory-utilization 0.90 --max-model-len 16384 \
  --max-num-seqs 4
# Optional ~3x decode via the MTP draft model (speculative decoding):
#   --speculative-config '{"model": "google/gemma-4-E4B-it-assistant", "num_speculative_tokens": 3}'
```

## Benchmark results

### GB10 (DGX Spark, 128GB unified memory, aarch64)

| Model | TTFT (8K ctx) | Decode throughput | Max context | Notes |
|-------|--------------|-------------------|-------------|-------|
| gemma-4-31B-it | ~4.2s | **~3.8 tok/s** | 8192 | `--enforce-eager`, 0.55 GPU util |

### A100 80GB (x86_64)

| Model | TTFT (8K ctx) | Decode throughput | Max context | Notes |
|-------|--------------|-------------------|-------------|-------|
| gemma-4-31B-it | ~1.1s | ~45 tok/s | 16384 | Single GPU, fp16 |
| gemma-4-12B-it | ~0.5s | ~85 tok/s | 32768 | Single GPU, fp16 |

## Known limitations

These apply to the **legacy 0.18.x backport** only; native installs (≥ 0.19.0) are unaffected:

- **Requires transformers from GitHub main**: On 0.18.x the `gemma4` model type was not in any stable transformers release. On vLLM ≥ 0.19.0 a compatible transformers ships as a normal dependency.
- **Audio**: The dense Gemma 4 config declares `audio_config: null` (dense models don't process audio). The E-series (E2B/E4B) *does* accept audio input — install `vllm[audio]` to enable it.
- **Sliding window attention**: On some platforms, sliding window + prefix caching interactions may require `--enforce-eager` for stability.
- **Shallow clone required (backport path)**: The 0.18.x path clones vLLM main at HEAD. If the Gemma 4 files are reorganized upstream, the copy paths may need updating. The native path has no such dependency.

## Upstream references

- **vLLM PR #38826**: [Add Gemma 4 support](https://github.com/vllm-project/vllm/pull/38826)
- **Google Gemma 4 announcement**: [blog.google](https://blog.google/technology/developers/gemma-4/)
- **HuggingFace model cards**: [google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it)

## License

Apache 2.0. See [LICENSE](LICENSE).
