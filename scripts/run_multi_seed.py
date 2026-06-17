#!/usr/bin/env python3
"""Run one fair-SPLIT experiment across multiple seeds and aggregate metrics."""

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts import table_results  # pylint: disable=wrong-import-position

MetricDict = Dict[str, Any]


def parse_seeds(value: str) -> List[int]:
    """Parse a comma-separated seed list such as '0,1,2'."""
    seeds = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        seeds.append(int(item))
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def reject_owned_train_args(train_args: Sequence[str]) -> None:
    """Reject src.train args controlled by the multi-seed runner."""
    owned = {"--random_state", "--results_dir", "--log_suffix"}
    present = [arg for arg in train_args if arg in owned]
    if present:
        joined = ", ".join(sorted(set(present)))
        raise ValueError(f"Do not pass runner-owned train args: {joined}")


def infer_experiment_identity(train_args: Sequence[str]) -> Dict[str, str]:
    """Infer model, dataset, and mode from src.train CLI args."""
    model = ""
    dataset = ""
    leaf_fill = "greedy"
    for index, arg in enumerate(train_args):
        if arg == "--model" and index + 1 < len(train_args):
            model = train_args[index + 1]
        elif arg == "--dataset" and index + 1 < len(train_args):
            dataset = train_args[index + 1]
        elif arg == "--leaf_fill" and index + 1 < len(train_args):
            leaf_fill = train_args[index + 1]

    display_model = f"split-{leaf_fill}" if model == "split" else model
    return {
        "model": display_model,
        "dataset": dataset,
        "mode": "fair" if "--fair" in train_args else "nofair",
    }


def default_output_dir(run_name: str) -> str:
    """Return a default output directory under results/multiseed."""
    safe_name = run_name.strip().replace(os.sep, "_")
    if not safe_name:
        raise ValueError("run_name must not be empty when output_dir is omitted.")
    return os.path.join(ROOT, "results", "multiseed", safe_name)


def aggregate_seed_results(per_seed: Mapping[str, MetricDict]) -> MetricDict:
    """Aggregate recursive numeric metrics across seed result dictionaries."""
    return _aggregate_nodes(list(per_seed.values()))


def flatten_summary_rows(
    summary: Mapping[str, Any],
    model: str = "",
    dataset: str = "",
    mode: str = "",
) -> List[Dict[str, Any]]:
    """Flatten nested summary metrics into CSV rows."""
    rows = []

    def visit(prefix: str, node: Any) -> None:
        """Visit nested summary nodes."""
        if _is_summary_leaf(node):
            rows.append({
                "model": model,
                "dataset": dataset,
                "mode": mode,
                "metric": prefix,
                "mean": node["mean"],
                "std": node["std"],
                "n": node["n"],
            })
            return
        if isinstance(node, dict):
            for key in sorted(node):
                child_prefix = f"{prefix}.{key}" if prefix else str(key)
                visit(child_prefix, node[key])

    visit("", summary)
    return rows


def write_json(path: str, payload: Mapping[str, Any]) -> None:
    """Write JSON payload with stable formatting."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2, ensure_ascii=False)
        file_obj.write("\n")


def write_summary_csv(path: str, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write flattened summary rows to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = ["model", "dataset", "mode", "metric", "mean", "std", "n"]
    with open(path, "w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_one_seed(
    seed: int,
    train_args: Sequence[str],
    logs_dir: str,
) -> Dict[str, Any]:
    """Run one seed and parse its resulting log."""
    cmd = [
        sys.executable,
        "-m",
        "src.train",
        *train_args,
        "--random_state",
        str(seed),
        "--results_dir",
        logs_dir,
        "--log_suffix",
        f"seed{seed}",
    ]
    completed = subprocess.run(
        cmd,
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.returncode != 0:
        return {
            "seed": seed,
            "ok": False,
            "returncode": completed.returncode,
            "output": completed.stdout,
        }

    log_path = _extract_log_path(completed.stdout)
    if not log_path:
        return {
            "seed": seed,
            "ok": False,
            "returncode": 0,
            "output": completed.stdout,
            "error": "Could not find 'Log saved to:' line in training output.",
        }
    if not os.path.isabs(log_path):
        log_path = os.path.join(ROOT, log_path)

    identity = infer_experiment_identity(train_args)
    parsed, _ = table_results.parse_log(log_path, identity["dataset"])
    return {"seed": seed, "ok": True, "log_path": log_path, "result": parsed}


def build_parser() -> argparse.ArgumentParser:
    """Build the multi-seed runner parser."""
    parser = argparse.ArgumentParser(
        description="Run one fair-SPLIT experiment across multiple seeds.",
    )
    parser.add_argument("--seeds", required=True, help="Comma-separated seeds, e.g. 0,1,2")
    parser.add_argument("--output_dir", default="", help="Output directory for logs and summaries.")
    parser.add_argument(
        "--run_name",
        default="single",
        help="Name under results/multiseed when output_dir is omitted.",
    )
    parser.add_argument("--fail_fast", action="store_true", help="Stop on first failed seed.")
    parser.add_argument("train_args", nargs=argparse.REMAINDER, help="Arguments after -- passed to src.train.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    train_args = list(args.train_args)
    if train_args and train_args[0] == "--":
        train_args = train_args[1:]
    if not train_args:
        parser.error("Pass src.train arguments after --")
    try:
        reject_owned_train_args(train_args)
        seeds = parse_seeds(args.seeds)
    except ValueError as exc:
        parser.error(str(exc))

    output_dir = args.output_dir or default_output_dir(args.run_name)
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    per_seed = {}
    failures = []
    for seed in seeds:
        print(f"[multi-seed] running seed={seed}")
        outcome = run_one_seed(seed, train_args, logs_dir)
        if outcome["ok"]:
            per_seed[str(seed)] = outcome["result"]
        else:
            failures.append(outcome)
            print(f"[multi-seed] seed={seed} failed")
            if args.fail_fast:
                break

    identity = infer_experiment_identity(train_args)
    summary = aggregate_seed_results(per_seed) if per_seed else {}
    rows = flatten_summary_rows(
        summary,
        model=identity["model"],
        dataset=identity["dataset"],
        mode=identity["mode"],
    )

    write_json(os.path.join(output_dir, "per_seed.json"), {
        "metadata": {
            "seeds": seeds,
            "command_args": train_args,
            "n_completed": len(per_seed),
            "n_failed": len(failures),
        },
        "results": per_seed,
        "failures": failures,
    })
    write_json(os.path.join(output_dir, "summary.json"), summary)
    write_summary_csv(os.path.join(output_dir, "summary.csv"), rows)

    print(f"[multi-seed] wrote {output_dir}")
    return 1 if failures else 0


def _aggregate_nodes(nodes: Sequence[Any]) -> Any:
    """Aggregate matching numeric leaves across nested dictionaries."""
    numeric_values = [float(node) for node in nodes if _is_number(node)]
    if numeric_values and len(numeric_values) == len(nodes):
        return _mean_std(numeric_values)

    dict_nodes = [node for node in nodes if isinstance(node, dict)]
    if not dict_nodes:
        return None

    keys = sorted({key for node in dict_nodes for key in node.keys()})
    output = {}
    for key in keys:
        child_values = [node[key] for node in dict_nodes if key in node]
        aggregated = _aggregate_nodes(child_values)
        if aggregated is not None:
            output[key] = aggregated
    return output


def _is_number(value: Any) -> bool:
    """Return True for finite int/float metric values."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _mean_std(values: Sequence[float]) -> Dict[str, float]:
    """Return sample mean/std/n for metric values."""
    n = len(values)
    mean = sum(values) / n
    if n == 1:
        std = 0.0
    else:
        variance = sum((value - mean) ** 2 for value in values) / (n - 1)
        std = math.sqrt(variance)
    return {"mean": round(mean, 6), "std": round(std, 6), "n": n}


def _is_summary_leaf(node: Any) -> bool:
    """Return True if node is a {'mean', 'std', 'n'} summary leaf."""
    return (
        isinstance(node, dict)
        and set(node.keys()) == {"mean", "std", "n"}
        and _is_number(node["mean"])
        and _is_number(node["std"])
        and isinstance(node["n"], int)
    )


def _extract_log_path(output: str) -> Optional[str]:
    """Extract train.py's final log path from stdout."""
    for line in reversed(output.splitlines()):
        if line.startswith("Log saved to:"):
            return line.split(":", 1)[1].strip()
    return None


if __name__ == "__main__":
    raise SystemExit(main())
