#!/usr/bin/env python3
"""Run the fair-SPLIT evaluation grid across multiple seeds."""

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts import run_multi_seed  # pylint: disable=wrong-import-position

DATASETS = [
    "adult", "bank", "compass", "german", "heloc", "spambase",
    "covertype", "thyroid", "communities", "heart", "lawschool", "acsincome",
]
MODEL_ORDER = ["cart", "split-greedy", "split-optimal", "resplit"]
MODES = ["nofair", "fair"]


def build_experiments(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Return eval.sh-equivalent experiment definitions."""
    experiments = []
    for dataset in args.datasets:
        experiments.extend(_dataset_experiments(dataset, args, fair=False))
        experiments.extend(_dataset_experiments(dataset, args, fair=True))
    return experiments


def build_parser() -> argparse.ArgumentParser:
    """Build parser for full-grid multi-seed evaluation."""
    parser = argparse.ArgumentParser(description="Run eval.sh grid across multiple seeds.")
    parser.add_argument("--seeds", required=True, help="Comma-separated seeds, e.g. 0,1,2")
    parser.add_argument("--output_dir", default="", help="Output directory.")
    parser.add_argument("--run_name", default="full-eval", help="Name under results/multiseed.")
    parser.add_argument("--datasets", default=",".join(DATASETS), help="Comma-separated dataset subset.")
    parser.add_argument("--fail_fast", action="store_true", help="Stop on first failed experiment.")
    parser.add_argument("--cart_depth", type=int, default=5)
    parser.add_argument("--split_depth", type=int, default=5)
    parser.add_argument("--split_reg", type=float, default=0.0001)
    parser.add_argument("--split_lookahead", type=int, default=2)
    parser.add_argument("--split_greedy_max_features", type=int, default=50)
    parser.add_argument("--num_prefix", type=int, default=30)
    parser.add_argument("--rashomon_bound", type=float, default=0.02)
    return parser


def parse_dataset_subset(value: str) -> List[str]:
    """Parse and validate a comma-separated dataset list."""
    datasets = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(datasets) - set(DATASETS))
    if unknown:
        raise ValueError(f"Unknown datasets: {', '.join(unknown)}")
    if not datasets:
        raise ValueError("At least one dataset is required.")
    return datasets


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run full-grid multi-seed evaluation."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.datasets = parse_dataset_subset(args.datasets)
        seeds = run_multi_seed.parse_seeds(args.seeds)
    except ValueError as exc:
        parser.error(str(exc))

    output_dir = args.output_dir or run_multi_seed.default_output_dir(args.run_name)
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    experiments = build_experiments(args)
    per_seed = {}
    failures = []

    for experiment in experiments:
        display_model = experiment["display_model"]
        dataset = experiment["dataset"]
        mode = experiment["mode"]
        per_seed.setdefault(display_model, {}).setdefault(dataset, {}).setdefault(mode, {})
        print(f"[eval-multi-seed] {display_model}/{dataset}/{mode}")
        for seed in seeds:
            outcome = run_multi_seed.run_one_seed(seed, experiment["train_args"], logs_dir)
            if outcome["ok"]:
                per_seed[display_model][dataset][mode][str(seed)] = outcome["result"]
            else:
                failure = dict(outcome)
                failure.update({"model": display_model, "dataset": dataset, "mode": mode})
                failures.append(failure)
                print(f"[eval-multi-seed] failed {display_model}/{dataset}/{mode}/seed{seed}")
                if args.fail_fast:
                    break
        if failures and args.fail_fast:
            break

    summary = _aggregate_full_grid(per_seed)
    rows = _flatten_full_grid(summary)
    run_multi_seed.write_json(os.path.join(output_dir, "per_seed.json"), {
        "metadata": {
            "seeds": seeds,
            "datasets": args.datasets,
            "models": MODEL_ORDER,
            "modes": MODES,
            "n_failures": len(failures),
        },
        "results": per_seed,
        "failures": failures,
    })
    run_multi_seed.write_json(os.path.join(output_dir, "summary.json"), summary)
    run_multi_seed.write_summary_csv(os.path.join(output_dir, "summary.csv"), rows)
    print(f"[eval-multi-seed] wrote {output_dir}")
    return 1 if failures else 0


def _dataset_experiments(
    dataset: str,
    args: argparse.Namespace,
    fair: bool,
) -> List[Dict[str, Any]]:
    """Return all model experiments for one dataset and fairness mode."""
    mode = "fair" if fair else "nofair"
    fair_flag = ["--fair"] if fair else []
    return [
        {
            "display_model": "cart",
            "model": "cart",
            "dataset": dataset,
            "mode": mode,
            "train_args": [
                "--model", "cart", "--dataset", dataset,
                "--max_depth", str(args.cart_depth), *fair_flag,
            ],
        },
        {
            "display_model": "split-greedy",
            "model": "split",
            "dataset": dataset,
            "mode": mode,
            "train_args": [
                "--model", "split", "--dataset", dataset,
                "--depth", str(args.split_depth), "--reg", str(args.split_reg),
                "--lookahead", str(args.split_lookahead), "--leaf_fill", "greedy",
                "--max_features", str(args.split_greedy_max_features), *fair_flag,
            ],
        },
        {
            "display_model": "split-optimal",
            "model": "split",
            "dataset": dataset,
            "mode": mode,
            "train_args": [
                "--model", "split", "--dataset", dataset,
                "--depth", str(args.split_depth), "--reg", str(args.split_reg),
                "--lookahead", str(args.split_lookahead), "--leaf_fill", "optimal",
                *fair_flag,
            ],
        },
        {
            "display_model": "resplit",
            "model": "resplit",
            "dataset": dataset,
            "mode": mode,
            "train_args": [
                "--model", "resplit", "--dataset", dataset,
                "--depth", str(args.split_depth), "--reg", str(args.split_reg),
                "--lookahead", str(args.split_lookahead),
                "--num_prefix", str(args.num_prefix),
                "--rashomon_bound", str(args.rashomon_bound), *fair_flag,
            ],
        },
    ]


def _aggregate_full_grid(per_seed: Dict[str, Any]) -> Dict[str, Any]:
    """Aggregate model/dataset/mode seed dictionaries."""
    summary = {}
    for model, datasets in per_seed.items():
        summary[model] = {}
        for dataset, modes in datasets.items():
            summary[model][dataset] = {}
            for mode, seed_results in modes.items():
                summary[model][dataset][mode] = run_multi_seed.aggregate_seed_results(seed_results)
    return summary


def _flatten_full_grid(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten full-grid summary into CSV rows."""
    rows = []
    for model, datasets in summary.items():
        for dataset, modes in datasets.items():
            for mode, metrics in modes.items():
                rows.extend(run_multi_seed.flatten_summary_rows(metrics, model, dataset, mode))
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
