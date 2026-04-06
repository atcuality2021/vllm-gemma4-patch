#!/bin/bash
# =============================================================================
# vLLM Gemma 4 Patch — Verification Script
# =============================================================================
# Verifies that the Gemma 4 patch was applied correctly to vLLM 0.18.x.
# Optionally runs a test inference if a model path is provided.
#
# Usage:
#   ./verify.sh /path/to/vllm-venv [/path/to/gemma4-model]
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/common.sh"

pass()  { echo -e "  ${GREEN}PASS${NC}  $1"; }
fail()  { echo -e "  ${RED}FAIL${NC}  $1"; FAILURES=$((FAILURES + 1)); }
skip()  { echo -e "  ${YELLOW}SKIP${NC}  $1"; }

FAILURES=0

# -- Args --
if [ $# -lt 1 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    echo "Usage: $0 <venv-path> [model-path]"
    echo ""
    echo "  venv-path     Path to vLLM virtualenv"
    echo "  model-path    (Optional) Path to a Gemma 4 model for test inference"
    exit 1
fi

VENV="$1"
MODEL_PATH="${2:-}"
PYTHON="$VENV/bin/python"

[ -f "$PYTHON" ] || { echo -e "${RED}ERROR: python not found at $PYTHON${NC}"; exit 1; }

# -- Auto-detect vLLM package path --
PYVER=$(detect_pyver "$VENV")
VLLM_PKG="$VENV/lib/python${PYVER}/site-packages/vllm"

if [ ! -d "$VLLM_PKG" ]; then
    PIP="$VENV/bin/pip"
    VLLM_LOCATION=$("$PIP" show vllm 2>/dev/null | grep "^Location:" | awk '{print $2}')
    if [ -n "$VLLM_LOCATION" ] && [ -d "$VLLM_LOCATION/vllm" ]; then
        VLLM_PKG="$VLLM_LOCATION/vllm"
    fi
fi

echo ""
echo -e "${BOLD}=============================================${NC}"
echo -e "${BOLD}  vLLM Gemma 4 Patch — Verification${NC}"
echo -e "  Python: ${BLUE}${PYVER}${NC}"
echo -e "  vLLM:   ${BLUE}${VLLM_PKG}${NC}"
echo -e "${BOLD}=============================================${NC}"
echo ""

# ==========================================================================
# Check 1: transformers has Gemma4Config
# ==========================================================================
echo -e "${BOLD}Checking transformers...${NC}"

if "$PYTHON" -c "from transformers.models.gemma4 import Gemma4Config" 2>/dev/null; then
    pass "transformers.models.gemma4.Gemma4Config importable"
else
    fail "transformers does not have gemma4 support (need dev version from GitHub)"
fi

# ==========================================================================
# Check 2: Gemma4 model files exist
# ==========================================================================
echo ""
echo -e "${BOLD}Checking model files...${NC}"

for f in gemma4.py gemma4_mm.py gemma4_utils.py; do
    if [ -f "$VLLM_PKG/model_executor/models/$f" ]; then
        pass "$f exists"
    else
        fail "$f missing from model_executor/models/"
    fi
done

# ==========================================================================
# Check 3: RoPE file exists
# ==========================================================================
echo ""
echo -e "${BOLD}Checking RoPE...${NC}"

if [ -f "$VLLM_PKG/model_executor/layers/rotary_embedding/gemma4_rope.py" ]; then
    pass "gemma4_rope.py exists"
else
    fail "gemma4_rope.py missing from rotary_embedding/"
fi

# ==========================================================================
# Check 4: Model imports work
# ==========================================================================
echo ""
echo -e "${BOLD}Checking Python imports...${NC}"

if "$PYTHON" -c "from vllm.model_executor.models.gemma4 import Gemma4ForCausalLM" 2>/dev/null; then
    pass "import Gemma4ForCausalLM"
else
    fail "cannot import Gemma4ForCausalLM"
fi

if "$PYTHON" -c "from vllm.model_executor.models.gemma4_mm import Gemma4ForConditionalGeneration" 2>/dev/null; then
    pass "import Gemma4ForConditionalGeneration"
else
    fail "cannot import Gemma4ForConditionalGeneration"
fi

if "$PYTHON" -c "from vllm.model_executor.layers.rotary_embedding.gemma4_rope import Gemma4RotaryEmbedding" 2>/dev/null; then
    pass "import Gemma4RotaryEmbedding"
else
    # Try wildcard import as fallback
    if "$PYTHON" -c "from vllm.model_executor.layers.rotary_embedding import gemma4_rope" 2>/dev/null; then
        pass "import gemma4_rope module"
    else
        fail "cannot import gemma4_rope"
    fi
fi

# ==========================================================================
# Check 5: Model registry
# ==========================================================================
echo ""
echo -e "${BOLD}Checking model registry...${NC}"

REGISTRY_CHECK=$("$PYTHON" -c "
try:
    from vllm.model_executor.models.registry import _VLLM_MODELS
    causal = 'Gemma4ForCausalLM' in _VLLM_MODELS
    cond = 'Gemma4ForConditionalGeneration' in _VLLM_MODELS
    print(f'causal={causal},cond={cond}')
except Exception as e:
    print(f'error={e}')
" 2>&1)

if echo "$REGISTRY_CHECK" | grep -q "causal=True"; then
    pass "Gemma4ForCausalLM in registry"
else
    fail "Gemma4ForCausalLM NOT in registry"
fi

if echo "$REGISTRY_CHECK" | grep -q "cond=True"; then
    pass "Gemma4ForConditionalGeneration in registry"
else
    fail "Gemma4ForConditionalGeneration NOT in registry"
fi

# ==========================================================================
# Check 6: base.py patch (null sub_config)
# ==========================================================================
echo ""
echo -e "${BOLD}Checking base.py patch...${NC}"

BASEPY="$VLLM_PKG/model_executor/models/transformers/base.py"
if [ -f "$BASEPY" ]; then
    if grep -q 'sub_config is None' "$BASEPY"; then
        pass "base.py has null sub_config guard"
    else
        fail "base.py missing null sub_config guard (audio_config=null will crash)"
    fi
else
    skip "base.py not found (may not apply to this vLLM version)"
fi

# ==========================================================================
# Check 7: utils.py patch (named buffers)
# ==========================================================================
echo ""
echo -e "${BOLD}Checking utils.py patch...${NC}"

UTILSPY="$VLLM_PKG/model_executor/models/utils.py"
if [ -f "$UTILSPY" ]; then
    if grep -q 'named_buffers' "$UTILSPY"; then
        pass "utils.py has named_buffers sweep"
    else
        fail "utils.py missing named_buffers sweep (layer_scalar will not load)"
    fi
else
    skip "utils.py not found"
fi

# ==========================================================================
# Check 8: Reasoning and tool parsers
# ==========================================================================
echo ""
echo -e "${BOLD}Checking parsers...${NC}"

for f in "reasoning/gemma4_reasoning_parser.py" "reasoning/gemma4_utils.py" \
         "tool_parsers/gemma4_tool_parser.py" "tool_parsers/gemma4_utils.py"; do
    if [ -f "$VLLM_PKG/$f" ]; then
        pass "$f exists"
    else
        fail "$f missing"
    fi
done

# ==========================================================================
# Optional: Test inference
# ==========================================================================
if [ -n "$MODEL_PATH" ]; then
    echo ""
    echo -e "${BOLD}Running test inference...${NC}"

    if [ ! -d "$MODEL_PATH" ]; then
        fail "Model path does not exist: $MODEL_PATH"
    else
        INFER_RESULT=$(GEMMA4_MODEL_PATH="$MODEL_PATH" "$PYTHON" -c "
import os
from vllm import LLM, SamplingParams
try:
    llm = LLM(
        model=os.environ['GEMMA4_MODEL_PATH'],
        trust_remote_code=True,
        enforce_eager=True,
        max_model_len=512,
        gpu_memory_utilization=0.5,
    )
    outputs = llm.generate(['Hello, world!'], SamplingParams(max_tokens=16, temperature=0.0))
    text = outputs[0].outputs[0].text.strip()
    print(f'OK: {text[:80]}')
except Exception as e:
    print(f'FAIL: {e}')
" 2>&1 | tail -1)

        if echo "$INFER_RESULT" | grep -q "^OK:"; then
            pass "Test inference succeeded: $INFER_RESULT"
        else
            fail "Test inference failed: $INFER_RESULT"
        fi
    fi
fi

# ==========================================================================
# Summary
# ==========================================================================
echo ""
echo -e "${BOLD}=============================================${NC}"
if [ "$FAILURES" -eq 0 ]; then
    echo -e "${GREEN}${BOLD}  All checks passed!${NC}"
else
    echo -e "${RED}${BOLD}  $FAILURES check(s) failed.${NC}"
fi
echo -e "${BOLD}=============================================${NC}"
echo ""

exit "$FAILURES"
