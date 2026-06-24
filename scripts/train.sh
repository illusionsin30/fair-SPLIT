#!/usr/bin/env bash
# ============================================================
# train.sh - Unified launcher for src.train
#
# Usage:
#   bash scripts/train.sh --model cart  --dataset compass
#   bash scripts/train.sh --model split --dataset iris
#   bash scripts/train.sh --model resplit --dataset diabetes
#   bash scripts/train.sh --model split --dataset compass \
#       --lookahead 3 --depth 6 --reg 0.005
#
# All arguments are forwarded to python -m src.train.
# ============================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT_DIR"

echo "Working directory: $ROOT_DIR"
echo "Command: python -m src.train $*"
echo "========================================"

python -m src.train "$@"
