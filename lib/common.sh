#!/usr/bin/env bash
# =============================================================================
# Shared helpers for vllm-gemma4-patch scripts
# =============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

# Detect Python minor version in a venv (e.g., "3.12")
detect_pyver() {
    local venv="$1"
    "$venv/bin/python" --version 2>&1 | sed -n 's/.*Python 3\.\([0-9]*\).*/3.\1/p'
}
