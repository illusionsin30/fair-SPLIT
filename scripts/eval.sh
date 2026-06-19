#!/usr/bin/env bash
# Run the full fair-SPLIT evaluation grid:
#   - nofair baseline
#   - original fair sample-level calibration
#   - LPFR leaf-level recalibration
# across all 12 datasets.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

SEEDS="${SEEDS:-0,42,99,1234,10086}"
RUN_NAME="${RUN_NAME:-full-12datasets-nofair-fair-lpfr-5seeds}"
FAIR_METRIC="${FAIR_METRIC:-dp}"
FAIR_ACC_BUDGET="${FAIR_ACC_BUDGET:-0.03}"
FAIR_LAMBDA="${FAIR_LAMBDA:-1.0}"
JOBS="${JOBS:-32}"
DATASET_JOBS="${DATASET_JOBS:-3}"
HEAVY_DATASETS="${HEAVY_DATASETS:-acsincome}"
HEAVY_DATASET_JOBS="${HEAVY_DATASET_JOBS:-1}"

DATASETS="adult,bank,compass,german,heloc,spambase,covertype,thyroid,communities,heart,lawschool,acsincome"

cd "${ROOT_DIR}"

python scripts/eval_multi_seed.py \
  --seeds "${SEEDS}" \
  --jobs "${JOBS}" \
  --dataset_jobs "${DATASET_JOBS}" \
  --heavy_datasets "${HEAVY_DATASETS}" \
  --heavy_dataset_jobs "${HEAVY_DATASET_JOBS}" \
  --datasets "${DATASETS}" \
  --fair_posts sample,leaf_pareto \
  --fair_metric "${FAIR_METRIC}" \
  --fair_lambda "${FAIR_LAMBDA}" \
  --fair_acc_budget "${FAIR_ACC_BUDGET}" \
  --run_name "${RUN_NAME}" \
  --fail_fast
