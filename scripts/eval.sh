#!/usr/bin/env bash
# ============================================================
# eval.sh — Run 4 models × 12 datasets × 2 fairness modes
#
# Models: cart, split-greedy, split-optimal, resplit
# Modes:  nofair  (keep sensitive columns, no calibration)
#         fair    (drop sensitive columns + post-hoc calibration)
#
# Usage:
#   bash scripts/eval.sh                  # run all experiments
#   bash scripts/eval.sh 2>&1 | tee log   # with progress output
# ============================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RESUME_FILE="results/.done_experiments"

# -------------------------------------------------------------------
# All datasets (shared across models)
# -------------------------------------------------------------------

ALL_DATASETS=(
    adult bank compass german heloc spambase
    covertype thyroid communities heart lawschool acsincome
)

# -------------------------------------------------------------------
# Hyperparameters
# -------------------------------------------------------------------
CART_DEPTH=5
SPLIT_DEPTH=5
SPLIT_REG=0.0001
SPLIT_LOOKAHEAD=2
NUM_PREFIX=30
RASHOMON_BOUND=0.02
SEED=42

# -------------------------------------------------------------------
# Runner
#   $1 = display model name (used in done_key & log line)
#   $2 = --model argument to src.train
#   $3 = dataset
#   $4 = label        ("nofair" or "fair")
#   $@ = extra flags  (passed directly to src.train)
# -------------------------------------------------------------------

_run() {
    local display="$1"    # e.g. split-greedy
    local model="$2"      # e.g. split   (the --model arg)
    local dataset="$3"
    local label="$4"
    shift 4
    local extra_args=("$@")

    local done_key="${display}/${dataset}/${label}"
    if [[ -f "$RESUME_FILE" ]] && grep -qxF "$done_key" "$RESUME_FILE"; then
        echo "  [skip] $label / $display / $dataset"
        return
    fi

    echo ""
    echo "========================================"
    echo "  $label  |  $display / $dataset"
    echo "========================================"

    python -m src.train \
        --model "$model" --dataset "$dataset" \
        --random_state "$SEED" \
        "${extra_args[@]}"

    echo "$done_key" >> "$RESUME_FILE"
}

# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

echo "========================================"
echo "  fair-SPLIT — 4 models × 12 datasets × 2 modes"
echo "  Started: $(date)"
echo "========================================"
echo ""

for dataset in "${ALL_DATASETS[@]}"; do

    # ================================================================
    #  No-fairness mode — sensitive columns kept, no calibration
    # ================================================================

    # 1. CART (nofair)
    _run cart cart "$dataset" nofair \
        --max_depth "$CART_DEPTH"

    # 2. SPLIT greedy (nofair)
    _run split-greedy split "$dataset" nofair \
        --depth "$SPLIT_DEPTH" --reg "$SPLIT_REG" \
        --lookahead "$SPLIT_LOOKAHEAD" \
        --leaf_fill greedy --max_features 50

    # 3. SPLIT optimal / DP (nofair)
    _run split-optimal split "$dataset" nofair \
        --depth "$SPLIT_DEPTH" --reg "$SPLIT_REG" \
        --lookahead "$SPLIT_LOOKAHEAD" \
        --leaf_fill optimal

    # 4. ReSPLIT (nofair)
    _run resplit resplit "$dataset" nofair \
        --depth "$SPLIT_DEPTH" --reg "$SPLIT_REG" \
        --lookahead "$SPLIT_LOOKAHEAD" \
        --num_prefix "$NUM_PREFIX" --rashomon_bound "$RASHOMON_BOUND"

    # ================================================================
    #  Fairness mode — drop sensitive columns + post-hoc calibration
    # ================================================================

    # 1. CART (fair)
    _run cart cart "$dataset" fair \
        --max_depth "$CART_DEPTH" --fair

    # 2. SPLIT greedy (fair)
    _run split-greedy split "$dataset" fair \
        --depth "$SPLIT_DEPTH" --reg "$SPLIT_REG" \
        --lookahead "$SPLIT_LOOKAHEAD" \
        --leaf_fill greedy --max_features 50 --fair

    # 3. SPLIT optimal / DP (fair)
    _run split-optimal split "$dataset" fair \
        --depth "$SPLIT_DEPTH" --reg "$SPLIT_REG" \
        --lookahead "$SPLIT_LOOKAHEAD" \
        --leaf_fill optimal --fair

    # 4. ReSPLIT (fair)
    _run resplit resplit "$dataset" fair \
        --depth "$SPLIT_DEPTH" --reg "$SPLIT_REG" \
        --lookahead "$SPLIT_LOOKAHEAD" \
        --num_prefix "$NUM_PREFIX" --rashomon_bound "$RASHOMON_BOUND" --fair

done

echo ""
echo "========================================"
echo "  All experiments complete"
echo "  Finished: $(date)"
echo "========================================"
