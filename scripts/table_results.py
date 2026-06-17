#!/usr/bin/env python3
"""Aggregate experiment log files into a structured JSON results file.

Reads ``../results/.done_experiments`` to discover completed experiments,
parses each corresponding ``*.log`` file, and writes a single
``../results/all_results.json`` organised by model → dataset → mode.

Log filename convention (set by ``src/train.py:_make_filename``)::

    cart-adult-d5-raw-fair.log
    cart-adult-d5-raw-nofair.log
    split-greedy-bank-d5-bin-fair.log
    split-optimal-bank-d5-bin-fair.log
    resplit-compass-d5-bin-fair.log

Output structure (one JSON file per mode)::

    results/all_results_nofair.json
    results/all_results_fair.json

Each file has the same internal structure::

    {
      "cart": {
        "adult": {
          "test_accuracy": 0.8450,
          "majority_baseline": 0.7521,
          "training_time_s": 6.8,
          "fairness": {               // only when dataset has sensitive attrs
            "sex": {
              "sp_diff": 0.1347,
              "di_ratio": 0.3682,
              "eo_diff": 0.0648,
              "calibrated_acc": 0.8489,       // fair-mode only
              "calibrated_sp_diff": 0.0001,   // fair-mode only
              "calibrated_di_ratio": 0.9994,  // fair-mode only
              "calibrated_eo_diff": 0.0069    // fair-mode only
            },
            ...
          }
        },
        ...
      },
      "split-greedy": { ... },
      "split-optimal": { ... },
      "resplit": { ... }
    }
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
DONE_FILE = os.path.join(RESULTS_DIR, ".done_experiments")
OUT_FILE_NOFAIR = os.path.join(RESULTS_DIR, "all_results_nofair.json")
OUT_FILE_FAIR = os.path.join(RESULTS_DIR, "all_results_fair.json")

# ---------------------------------------------------------------------------
# Sensitive-attribute map — kept in sync with src/train.py
# ---------------------------------------------------------------------------
SENSITIVE_ATTRS = {
    "adult":      ["sex", "race"],
    "compass":    ["sex", "race"],
    "german":     ["personal_status", "age"],
    "bank":       ["marital", "age"],
    "lawschool":  ["race", "gender"],
    "acsincome":  ["SEX", "RAC1P"],
}

DATASET_ORDER = [
    "adult", "bank", "compass", "german", "heloc", "spambase",
    "covertype", "thyroid", "communities", "heart",
    "lawschool", "acsincome",
]

MODEL_ORDER = ["cart", "split-greedy", "split-optimal", "resplit"]


# ---------------------------------------------------------------------------
# Single-log parser — scans lines once and extracts everything.
# ---------------------------------------------------------------------------

# Match a line like:  "Fairness — sex:"  or  "Fairness — sex (calibrated):"
_FAIR_BLOCK_HEADER = re.compile(
    r"^Fairness — (?P<attr>.+?)(?: \(calibrated\))?:$"
)

# Metric patterns — use .search() so leading whitespace doesn't matter.
_RE_TEST_ACC = re.compile(r"Test accuracy:\s+([\d.]+)")
_RE_MAJ_BL = re.compile(r"Majority-class baseline:\s+([\d.]+)")
_RE_TIME = re.compile(r"Training time:\s+([\d.]+)s")
_RE_SP_DIFF = re.compile(r"Statistical parity diff:\s+([\d.]+)")
_RE_DI_RATIO = re.compile(r"Disparate impact ratio:\s+([\d.]+)")
_RE_EO_DIFF = re.compile(r"Equal opportunity diff:\s+([\d.]+)")


def _parse_calibrated_acc(line):
    """Return calibrated_acc float from a line like
    '  Calibrated accuracy: 0.8489 (was 0.8442, Δ=+0.0046)'
    Returns None if the line doesn't match.
    """
    m = re.search(
        r"Calibrated accuracy:\s+([\d.]+)\s+\(was\s+[\d.]+\s*,\s*Δ=.+\)",
        line,
    )
    return float(m.group(1)) if m else None


def parse_log(filepath, dataset):
    """Parse one log file into a dict.

    Returns:
        (result, is_fair)
        ``is_fair`` is True if --fair was enabled (preprocess + calibration).
    """
    with open(filepath, "r") as f:
        lines = f.readlines()

    result = {}
    is_fair = False

    # --- Phase 1: top-level metrics (accuracy, baseline, time) ---
    for line in lines:
        if (m := _RE_TEST_ACC.search(line)):
            result["test_accuracy"] = float(m.group(1))
        elif (m := _RE_MAJ_BL.search(line)):
            result["majority_baseline"] = float(m.group(1))
        elif (m := _RE_TIME.search(line)):
            result["training_time_s"] = float(m.group(1))

    # Detect fair mode — look for "FairPreprocess" or "FairCalibrate"
    for line in lines:
        if "FairPreprocess" in line or "FairCalibrate" in line:
            is_fair = True
            break

    # --- Phase 2: fairness blocks ---
    if dataset not in SENSITIVE_ATTRS:
        return result, is_fair

    fairness = {}
    current_attr = None        # e.g. "sex"
    current_block = {}         # metrics being collected
    is_calibrated = False

    for line in lines:
        # Detect fairness block header
        fm = _FAIR_BLOCK_HEADER.search(line)
        if fm:
            # Save previous block
            if current_attr is not None and current_block:
                key = current_attr
                if key not in fairness:
                    fairness[key] = {}
                fairness[key].update(current_block)
            # Start new block
            current_attr = fm.group("attr").strip()
            current_block = {}
            is_calibrated = "calibrated" in line
            continue

        if current_attr is None:
            continue

        # Metric lines within a fairness block
        if (m := _RE_SP_DIFF.search(line)):
            val = float(m.group(1))
            key = "calibrated_sp_diff" if is_calibrated else "sp_diff"
            current_block[key] = val
        elif (m := _RE_DI_RATIO.search(line)):
            val = float(m.group(1))
            key = "calibrated_di_ratio" if is_calibrated else "di_ratio"
            current_block[key] = val
        elif (m := _RE_EO_DIFF.search(line)):
            val = float(m.group(1))
            key = "calibrated_eo_diff" if is_calibrated else "eo_diff"
            current_block[key] = val
        elif (cal_acc := _parse_calibrated_acc(line)) is not None:
            current_block["calibrated_acc"] = cal_acc

    # Save last block
    if current_attr is not None and current_block:
        key = current_attr
        if key not in fairness:
            fairness[key] = {}
        fairness[key].update(current_block)

    if fairness:
        result["fairness"] = fairness

    return result, is_fair


# ---------------------------------------------------------------------------
# Log filename matching
# ---------------------------------------------------------------------------

_LOG_PATTERN_CACHE = {}


def _find_log_file(model, dataset, mode):
    """Find the log file for ``model`` × ``dataset`` × ``mode``."""
    cache_key = (model, dataset, mode)
    if cache_key in _LOG_PATTERN_CACHE:
        pat = _LOG_PATTERN_CACHE[cache_key]
    else:
        pat = re.compile(
            r"^" + re.escape(model) + r"-" + re.escape(dataset) + r"-"
            r"d\d+-(?:raw|bin)-" + re.escape(mode) + r"\.log$"
        )
        _LOG_PATTERN_CACHE[cache_key] = pat

    for fname in os.listdir(RESULTS_DIR):
        if pat.match(fname):
            return os.path.join(RESULTS_DIR, fname)
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _round(obj):
    if isinstance(obj, float):
        return round(obj, 6)
    if isinstance(obj, dict):
        return {k: _round(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round(v) for v in obj]
    return obj


def _order_output(raw):
    """Sort by MODEL_ORDER × DATASET_ORDER."""
    ordered = {}
    for model in MODEL_ORDER:
        if model in raw:
            ordered[model] = {}
            for ds in DATASET_ORDER:
                if ds in raw[model]:
                    ordered[model][ds] = raw[model][ds]
    return _round(ordered)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not os.path.isfile(DONE_FILE):
        print(f"[ERROR] .done_experiments not found at {DONE_FILE}")
        sys.exit(1)

    with open(DONE_FILE, "r") as fh:
        done = [line.strip() for line in fh if line.strip()]

    output_nofair = {}
    output_fair = {}
    for entry in done:
        parts = entry.split("/")
        if len(parts) != 3:
            print(f"  [WARN] malformed done entry: {entry}")
            continue
        model, dataset, mode = parts

        log_path = _find_log_file(model, dataset, mode)
        if log_path is None:
            print(f"  [WARN] no log for {model}/{dataset}/{mode}")
            continue

        try:
            parsed, is_fair = parse_log(log_path, dataset)
        except Exception as exc:
            print(f"  [WARN] parsing {log_path}: {exc}")
            continue

        target = output_fair if is_fair else output_nofair
        target.setdefault(model, {})[dataset] = parsed

    output_nofair = _order_output(output_nofair)
    output_fair = _order_output(output_fair)

    def _count(raw):
        return sum(len(raw[m][ds]) for m in raw for ds in raw[m])

    with open(OUT_FILE_NOFAIR, "w") as fh:
        json.dump(output_nofair, fh, indent=2, ensure_ascii=False)
    print(f"Wrote {OUT_FILE_NOFAIR}  ({_count(output_nofair)} experiments)")

    with open(OUT_FILE_FAIR, "w") as fh:
        json.dump(output_fair, fh, indent=2, ensure_ascii=False)
    print(f"Wrote {OUT_FILE_FAIR}  ({_count(output_fair)} experiments)")


if __name__ == "__main__":
    main()
