#!/bin/bash
# =============================================================================
# vLLM Gemma 4 Patch — Backport PR #38826 to vLLM 0.18.x
# =============================================================================
# Patches any vLLM 0.18.x installation to support Google Gemma 4 models.
# Works on x86_64 and aarch64 (ARM/GB10).
#
# Usage:
#   ./patch.sh /path/to/vllm-venv
#
# What it does:
#   1. Upgrades huggingface_hub + installs transformers from GitHub main
#   2. Clones vLLM main to get native gemma4 model files (PR #38826)
#   3. Copies 12+ gemma4 files (models, RoPE, parsers, registry)
#   4. Fixes import paths for 0.18.x compatibility
#   5. Patches base.py for null sub_configs (Gemma4 audio_config is null)
#   6. Patches utils.py to load named buffers (layer_scalar)
#   7. Patches model registry to register Gemma4 architectures
#
# On failure, rolls back all changes from a pre-patch backup.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/common.sh"

log()  { echo -e "${GREEN}[OK]${NC}    $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC}  $1"; }
info() { echo -e "${BLUE}[INFO]${NC}  $1"; }
err()  { echo -e "${RED}[ERROR]${NC} $1"; cleanup_on_failure; exit 1; }

# -- Globals --
TMPDIR=""
BACKUP_DIR=""
VLLM_PKG=""
ROLLBACK_NEEDED=false

# -- Cleanup --
cleanup_temp() {
    if [ -n "$TMPDIR" ] && [ -d "$TMPDIR" ]; then
        rm -rf "$TMPDIR"
    fi
}

cleanup_on_failure() {
    if [ "$ROLLBACK_NEEDED" = true ] && [ -n "$BACKUP_DIR" ] && [ -d "$BACKUP_DIR" ]; then
        echo ""
        warn "Rolling back changes from backup..."
        if [ -n "$VLLM_PKG" ] && [ -d "$VLLM_PKG" ]; then
            # Restore backed-up files
            for f in "$BACKUP_DIR"/*.backup; do
                [ -f "$f" ] || continue
                original=$(basename "$f" .backup)
                # Decode the path from the backup filename
                restore_path=$(head -1 "$f")
                tail -n +2 "$f" > "$restore_path" 2>/dev/null && \
                    info "Restored: $restore_path" || \
                    warn "Could not restore: $restore_path"
            done
        fi
        log "Rollback complete."
    fi
    cleanup_temp
}

trap cleanup_temp EXIT

# -- Usage --
usage() {
    echo "Usage: $0 <venv-path>"
    echo ""
    echo "  venv-path    Path to the Python virtualenv containing vLLM 0.18.x"
    echo ""
    echo "Examples:"
    echo "  $0 ~/vllm-env"
    echo "  $0 /opt/vllm/venv"
    exit 1
}

# -- Backup helper --
backup_file() {
    local filepath="$1"
    if [ -f "$filepath" ]; then
        local backup_name
        backup_name=$(echo "$filepath" | tr '/' '_')
        local backup_path="$BACKUP_DIR/${backup_name}.backup"
        # First line is the original path, rest is content
        echo "$filepath" > "$backup_path"
        cat "$filepath" >> "$backup_path"
    fi
}

# -- Args --
if [ $# -lt 1 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    usage
fi

VENV="$1"
PIP="$VENV/bin/pip"
PYTHON="$VENV/bin/python"

# -- Validate venv --
[ -d "$VENV" ]    || err "Virtualenv not found at: $VENV"
[ -f "$PIP" ]     || err "pip not found at: $PIP"
[ -f "$PYTHON" ]  || err "python not found at: $PYTHON"

# -- Auto-detect Python version and vLLM package path --
PYVER=$(detect_pyver "$VENV")
VLLM_PKG="$VENV/lib/python${PYVER}/site-packages/vllm"

if [ ! -d "$VLLM_PKG" ]; then
    # Try to find it via pip show
    VLLM_LOCATION=$("$PIP" show vllm 2>/dev/null | grep "^Location:" | awk '{print $2}')
    if [ -n "$VLLM_LOCATION" ] && [ -d "$VLLM_LOCATION/vllm" ]; then
        VLLM_PKG="$VLLM_LOCATION/vllm"
    else
        err "vLLM package not found. Is vLLM installed in $VENV?"
    fi
fi

[ -d "$VLLM_PKG" ] || err "vLLM not found at: $VLLM_PKG"

# -- Get vLLM version --
VLLM_VER=$("$PIP" show vllm 2>/dev/null | grep "^Version:" | awk '{print $2}')

# -- Verify 0.18.x --
if [[ ! "$VLLM_VER" =~ ^0\.18\. ]]; then
    warn "vLLM version is $VLLM_VER (expected 0.18.x). Patch may not apply cleanly."
    read -rp "Continue anyway? [y/N] " confirm
    if [[ ! "$confirm" =~ ^[yY]$ ]]; then
        echo "Aborted."
        exit 0
    fi
fi

echo ""
echo -e "${BOLD}=============================================${NC}"
echo -e "${BOLD}  vLLM Gemma 4 Patch${NC}"
echo -e "  vLLM version: ${BLUE}${VLLM_VER}${NC}"
echo -e "  Python:       ${BLUE}${PYVER}${NC}"
echo -e "  Platform:     ${BLUE}$(uname -m)${NC}"
echo -e "  venv:         ${BLUE}${VENV}${NC}"
echo -e "  vLLM path:    ${BLUE}${VLLM_PKG}${NC}"
echo -e "${BOLD}=============================================${NC}"
echo ""

# -- Check if already patched --
if grep -q 'Gemma4ForCausalLM' "$VLLM_PKG/model_executor/models/registry.py" 2>/dev/null; then
    warn "Gemma4 is already registered in the vLLM model registry."
    read -rp "Re-patch anyway? [y/N] " confirm
    if [[ ! "$confirm" =~ ^[yY]$ ]]; then
        echo "Aborted."
        exit 0
    fi
fi

# -- Create backup and temp dirs --
BACKUP_DIR=$(mktemp -d -t vllm-gemma4-backup.XXXXXXXX)
TMPDIR=$(mktemp -d -t vllm-gemma4-src.XXXXXXXX)
info "Backup dir: $BACKUP_DIR"
ROLLBACK_NEEDED=true

# ==========================================================================
# Step 1: Upgrade transformers (gemma4 model_type not in stable release)
# ==========================================================================
echo ""
echo -e "${BOLD}[1/6] Installing transformers with Gemma 4 support...${NC}"

"$PIP" install --upgrade huggingface_hub -q 2>&1 | tail -1 || true
info "huggingface_hub upgraded"

"$PIP" install git+https://github.com/huggingface/transformers.git --no-deps -q 2>&1 | tail -1 || \
    err "Failed to install transformers from GitHub main"

# Verify transformers has gemma4
"$PYTHON" -c "from transformers.models.gemma4 import Gemma4Config; print('  Gemma4Config: available')" || \
    err "transformers still does not have gemma4 support. Check your internet connection."
log "transformers upgraded with Gemma 4 support"

# ==========================================================================
# Step 2: Clone vLLM main (shallow)
# ==========================================================================
echo ""
echo -e "${BOLD}[2/6] Cloning vLLM main branch (shallow)...${NC}"

git clone --depth 1 https://github.com/vllm-project/vllm.git "$TMPDIR/vllm-src" 2>&1 | tail -3 || \
    err "Failed to clone vLLM repository. Check your internet connection."

SRC="$TMPDIR/vllm-src/vllm"

[ -f "$SRC/model_executor/models/gemma4.py" ] || \
    err "gemma4.py not found in vLLM main. PR #38826 may not be merged yet."
log "vLLM source cloned"

# ==========================================================================
# Step 3: Copy Gemma 4 model files
# ==========================================================================
echo ""
echo -e "${BOLD}[3/6] Copying Gemma 4 model files...${NC}"

MODELS="$VLLM_PKG/model_executor/models"
ROPE="$VLLM_PKG/model_executor/layers/rotary_embedding"

# Backup files that will be overwritten
backup_file "$ROPE/__init__.py"
backup_file "$VLLM_PKG/reasoning/__init__.py"
backup_file "$VLLM_PKG/tool_parsers/__init__.py"

# -- Model files --
cp "$SRC/model_executor/models/gemma4.py"       "$MODELS/"
cp "$SRC/model_executor/models/gemma4_mm.py"     "$MODELS/"
cp "$SRC/model_executor/models/gemma4_utils.py"  "$MODELS/"
info "Copied: gemma4.py, gemma4_mm.py, gemma4_utils.py"

# -- Fix import path for 0.18.x compatibility --
# In vLLM main: vllm.inputs.MultiModalDataDict
# In vLLM 0.18.x: vllm.multimodal.inputs.MultiModalDataDict
sed -i 's|from vllm\.inputs import MultiModalDataDict|from vllm.multimodal.inputs import MultiModalDataDict|' \
    "$MODELS/gemma4_mm.py"
info "Fixed import path in gemma4_mm.py for 0.18.x"

# -- RoPE --
cp "$SRC/model_executor/layers/rotary_embedding/gemma4_rope.py" "$ROPE/"
cp "$SRC/model_executor/layers/rotary_embedding/__init__.py"    "$ROPE/__init__.py"
info "Copied: gemma4_rope.py + rotary_embedding __init__.py"

# Copy telechat3 rope if present (dependency in main's __init__.py)
if [ -f "$SRC/model_executor/layers/rotary_embedding/telechat3_scaling_rope.py" ]; then
    cp "$SRC/model_executor/layers/rotary_embedding/telechat3_scaling_rope.py" "$ROPE/"
    info "Copied: telechat3_scaling_rope.py (new dependency)"
fi

# -- Reasoning parsers --
if [ -d "$SRC/reasoning" ]; then
    cp "$SRC/reasoning/gemma4_reasoning_parser.py"  "$VLLM_PKG/reasoning/"
    cp "$SRC/reasoning/gemma4_utils.py"             "$VLLM_PKG/reasoning/"
    cp "$SRC/reasoning/__init__.py"                 "$VLLM_PKG/reasoning/__init__.py"
    info "Copied: reasoning parsers"
fi

# -- Tool parsers --
if [ -d "$SRC/tool_parsers" ]; then
    cp "$SRC/tool_parsers/gemma4_utils.py"          "$VLLM_PKG/tool_parsers/"
    cp "$SRC/tool_parsers/gemma4_tool_parser.py"    "$VLLM_PKG/tool_parsers/"
    cp "$SRC/tool_parsers/__init__.py"              "$VLLM_PKG/tool_parsers/__init__.py"
    info "Copied: tool parsers"
fi

# -- Config convertor --
if [ -f "$SRC/transformers_utils/model_arch_config_convertor.py" ]; then
    backup_file "$VLLM_PKG/transformers_utils/model_arch_config_convertor.py"
    cp "$SRC/transformers_utils/model_arch_config_convertor.py" \
       "$VLLM_PKG/transformers_utils/model_arch_config_convertor.py"
    info "Copied: model_arch_config_convertor.py"
fi

log "All Gemma 4 files copied (12+ files)"

# ==========================================================================
# Step 4: Patch model registry
# ==========================================================================
echo ""
echo -e "${BOLD}[4/6] Patching model registry...${NC}"

REGISTRY="$VLLM_PKG/model_executor/models/registry.py"
backup_file "$REGISTRY"

if ! grep -q 'Gemma4ForCausalLM' "$REGISTRY"; then
    # Insert after Gemma3ForConditionalGeneration entry
    sed -i '/"Gemma3ForConditionalGeneration".*gemma3_mm/a\    "Gemma4ForCausalLM": ("gemma4", "Gemma4ForCausalLM"),\n    "Gemma4ForConditionalGeneration": ("gemma4_mm", "Gemma4ForConditionalGeneration"),  # noqa: E501' \
        "$REGISTRY"

    # Verify the patch applied
    if grep -q 'Gemma4ForCausalLM' "$REGISTRY"; then
        log "Registry patched: Gemma4ForCausalLM + Gemma4ForConditionalGeneration"
    else
        err "Failed to patch registry. The Gemma3 anchor line may have changed."
    fi
else
    log "Registry already has Gemma4 entries (skipped)"
fi

# ==========================================================================
# Step 5: Patch base.py (null sub_config guard)
# ==========================================================================
echo ""
echo -e "${BOLD}[5/6] Patching base.py for null sub_configs...${NC}"

BASEPY="$VLLM_PKG/model_executor/models/transformers/base.py"

if [ ! -f "$BASEPY" ]; then
    warn "base.py not found at expected path. Skipping (may not be needed for your vLLM version)."
else
    backup_file "$BASEPY"
    if ! grep -q 'sub_config is None' "$BASEPY"; then
        sed -i '/if sub_config.dtype != (dtype := self.config.dtype):/i\            if sub_config is None:\n                continue' "$BASEPY"

        if grep -q 'sub_config is None' "$BASEPY"; then
            log "base.py patched: null sub_config guard added"
        else
            warn "base.py patch may not have applied. Gemma4 audio_config=null may cause issues."
        fi
    else
        log "base.py already has null sub_config guard (skipped)"
    fi
fi

# ==========================================================================
# Step 6: Patch utils.py (named buffers for layer_scalar)
# ==========================================================================
echo ""
echo -e "${BOLD}[6/6] Patching utils.py for named buffer loading...${NC}"

UTILSPY="$VLLM_PKG/model_executor/models/utils.py"

if [ ! -f "$UTILSPY" ]; then
    warn "utils.py not found at expected path. Skipping."
else
    backup_file "$UTILSPY"
    if ! grep -q 'named_buffers' "$UTILSPY"; then
        "$PYTHON" - "$UTILSPY" << 'PYEOF'
import sys

path = sys.argv[1]
with open(path) as f:
    content = f.read()

old = '''                child_params[stat_name] = module_state_dict[stat_name]'''
new = '''                child_params[stat_name] = module_state_dict[stat_name]

        # Also include named buffers (e.g. layer_scalar in Gemma4)
        for buf_name, buf_tensor in module.named_buffers(recurse=False):
            if buf_name not in child_params:
                child_params[buf_name] = buf_tensor'''

# Replace the LAST occurrence (inside _add_loadable_non_param_tensors)
idx = content.rfind(old)
if idx != -1:
    content = content[:idx] + new + content[idx + len(old):]
    with open(path, 'w') as f:
        f.write(content)
    print("  Patched: named_buffers sweep added")
else:
    print("  WARNING: Could not find patch target in utils.py — may already be patched")
    sys.exit(1)
PYEOF

        if grep -q 'named_buffers' "$UTILSPY"; then
            log "utils.py patched: named buffer loading for layer_scalar"
        else
            warn "utils.py patch may not have applied. Gemma4 layer_scalar may not load correctly."
        fi
    else
        log "utils.py already has named_buffers support (skipped)"
    fi
fi

# ==========================================================================
# Cleanup
# ==========================================================================
cleanup_temp
TMPDIR=""
ROLLBACK_NEEDED=false

echo ""
echo -e "${BOLD}=============================================${NC}"
echo -e "${GREEN}${BOLD}  Gemma 4 patch applied successfully!${NC}"
echo -e "${BOLD}=============================================${NC}"
echo ""
echo -e "  ${BOLD}Verify:${NC}"
echo -e "    $PYTHON -c 'from vllm.model_executor.models.gemma4 import Gemma4ForCausalLM; print(\"OK\")'"
echo ""
echo -e "  ${BOLD}Launch (example for 31B):${NC}"
echo -e "    source $VENV/bin/activate"
echo -e "    vllm serve /path/to/gemma-4-31B-it \\"
echo -e "      --trust-remote-code --enforce-eager \\"
echo -e "      --gpu-memory-utilization 0.90 --max-model-len 16384 \\"
echo -e "      --max-num-seqs 16 --enable-prefix-caching"
echo ""
echo -e "  ${BOLD}Backup saved to:${NC} $BACKUP_DIR"
echo -e "  (Delete once you have verified the patch works.)"
echo ""
