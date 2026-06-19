#!/usr/bin/env python3
"""Run the fair-SPLIT evaluation grid across multiple seeds."""

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
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
FAIR_POSTS = ["sample", "leaf_pareto"]
FAIR_POST_TO_MODE = {
    "sample": "fair",
    "leaf_pareto": "lpfr",
}
MODES = ["nofair", "fair", "lpfr"]


def build_experiments(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Return eval.sh-equivalent experiment definitions."""
    experiments = []
    for dataset in args.datasets:
        if not args.skip_nofair:
            experiments.extend(_dataset_experiments(dataset, args, mode="nofair"))
        for fair_post in args.fair_posts:
            experiments.extend(_dataset_experiments(
                dataset, args, mode=FAIR_POST_TO_MODE[fair_post],
                fair_post=fair_post,
            ))
    return experiments


def build_parser() -> argparse.ArgumentParser:
    """Build parser for full-grid multi-seed evaluation."""
    parser = argparse.ArgumentParser(description="Run eval.sh grid across multiple seeds.")
    parser.add_argument("--seeds", required=True, help="Comma-separated seeds, e.g. 0,1,2")
    parser.add_argument(
        "--jobs",
        type=int,
        default=_default_jobs(),
        help="Number of experiment processes to run concurrently.",
    )
    parser.add_argument(
        "--dataset_jobs",
        type=int,
        default=3,
        help="Max concurrent tasks for the same dataset.",
    )
    parser.add_argument(
        "--heavy_datasets",
        default="acsincome",
        help="Comma-separated datasets that should use heavy_dataset_jobs.",
    )
    parser.add_argument(
        "--heavy_dataset_jobs",
        type=int,
        default=1,
        help="Max concurrent tasks for each heavy dataset.",
    )
    parser.add_argument("--output_dir", default="", help="Output directory.")
    parser.add_argument("--run_name", default="full-eval", help="Name under results/multiseed.")
    parser.add_argument("--datasets", default=",".join(DATASETS), help="Comma-separated dataset subset.")
    parser.add_argument(
        "--fair_posts",
        default="sample,leaf_pareto",
        help="Comma-separated fair postprocessors to run: sample,leaf_pareto.",
    )
    parser.add_argument(
        "--skip_nofair",
        action="store_true",
        help="Only run fair post-processing modes, skipping the nofair baseline.",
    )
    parser.add_argument(
        "--fair_metric",
        type=str,
        default="dp",
        choices=["dp", "eo"],
        help="LPFR fairness objective.",
    )
    parser.add_argument(
        "--fair_lambda",
        type=float,
        default=1.0,
        help="LPFR fairness-gain weight.",
    )
    parser.add_argument(
        "--fair_acc_budget",
        type=float,
        default=0.02,
        help="LPFR maximum calibration accuracy drop.",
    )
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


def parse_fair_posts(value: str) -> List[str]:
    """Parse and validate fair post-processing modes."""
    fair_posts = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(fair_posts) - set(FAIR_POSTS))
    if unknown:
        raise ValueError(f"Unknown fair postprocessors: {', '.join(unknown)}")
    if not fair_posts:
        raise ValueError("At least one fair postprocessor is required.")
    return fair_posts


def _default_jobs() -> int:
    """Return default parallelism from EVAL_JOBS or CPU count."""
    env_jobs = os.environ.get("EVAL_JOBS")
    if env_jobs:
        return max(1, int(env_jobs))
    return 32


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run full-grid multi-seed evaluation."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.datasets = parse_dataset_subset(args.datasets)
        args.fair_posts = parse_fair_posts(args.fair_posts)
        seeds = run_multi_seed.parse_seeds(args.seeds)
        if args.jobs < 1:
            raise ValueError("--jobs must be >= 1")
        if args.dataset_jobs < 1:
            raise ValueError("--dataset_jobs must be >= 1")
        args.heavy_datasets = parse_dataset_subset(args.heavy_datasets)
        if args.heavy_dataset_jobs < 1:
            raise ValueError("--heavy_dataset_jobs must be >= 1")
    except ValueError as exc:
        parser.error(str(exc))

    output_dir = args.output_dir or run_multi_seed.default_output_dir(args.run_name)
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    experiments = build_experiments(args)
    per_seed = {}
    failures = []

    for experiment in experiments:
        _init_result_slot(per_seed, experiment)

    total_tasks = len(experiments) * len(seeds)
    print(f"[eval-multi-seed] running {total_tasks} tasks with jobs={args.jobs}")
    _run_tasks_parallel(
        experiments=experiments,
        seeds=seeds,
        logs_dir=logs_dir,
        jobs=args.jobs,
        dataset_jobs=args.dataset_jobs,
        heavy_datasets=args.heavy_datasets,
        heavy_dataset_jobs=args.heavy_dataset_jobs,
        fail_fast=args.fail_fast,
        per_seed=per_seed,
        failures=failures,
    )

    summary = _aggregate_full_grid(per_seed)
    rows = _flatten_full_grid(summary)
    run_multi_seed.write_json(os.path.join(output_dir, "per_seed.json"), {
        "metadata": {
            "seeds": seeds,
            "datasets": args.datasets,
            "models": MODEL_ORDER,
            "modes": _selected_modes(args),
            "fair_posts": args.fair_posts,
            "jobs": args.jobs,
            "dataset_jobs": args.dataset_jobs,
            "heavy_datasets": args.heavy_datasets,
            "heavy_dataset_jobs": args.heavy_dataset_jobs,
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
    mode: str,
    fair_post: str = "sample",
) -> List[Dict[str, Any]]:
    """Return all model experiments for one dataset and fairness mode."""
    fair_args = _fair_args(mode, fair_post, args)
    return [
        {
            "display_model": "cart",
            "model": "cart",
            "dataset": dataset,
            "mode": mode,
            "train_args": [
                "--model", "cart", "--dataset", dataset,
                "--max_depth", str(args.cart_depth), *fair_args,
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
                "--max_features", str(args.split_greedy_max_features), *fair_args,
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
                *fair_args,
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
                "--rashomon_bound", str(args.rashomon_bound), *fair_args,
            ],
        },
    ]


def _fair_args(mode: str, fair_post: str, args: argparse.Namespace) -> List[str]:
    """Return src.train fairness args for one experiment mode."""
    if mode == "nofair":
        return []
    fair_args = ["--fair", "--fair_post", fair_post]
    if fair_post == "leaf_pareto":
        fair_args.extend([
            "--fair_metric", args.fair_metric,
            "--fair_lambda", str(args.fair_lambda),
            "--fair_acc_budget", str(args.fair_acc_budget),
        ])
    return fair_args


def _selected_modes(args: argparse.Namespace) -> List[str]:
    """Return output modes selected by CLI options."""
    modes = [] if args.skip_nofair else ["nofair"]
    modes.extend(FAIR_POST_TO_MODE[fair_post] for fair_post in args.fair_posts)
    return modes


def _init_result_slot(per_seed: Dict[str, Any], experiment: Dict[str, Any]) -> None:
    """Ensure per_seed contains the nested keys for an experiment."""
    per_seed.setdefault(experiment["display_model"], {}).setdefault(
        experiment["dataset"], {}
    ).setdefault(experiment["mode"], {})


def _run_tasks_parallel(
    experiments: Sequence[Dict[str, Any]],
    seeds: Sequence[int],
    logs_dir: str,
    jobs: int,
    dataset_jobs: int,
    heavy_datasets: Sequence[str],
    heavy_dataset_jobs: int,
    fail_fast: bool,
    per_seed: Dict[str, Any],
    failures: List[Dict[str, Any]],
) -> None:
    """Run experiment/seed tasks concurrently and collect parsed results."""
    pending = [
        (experiment, seed)
        for seed in seeds
        for experiment in experiments
    ]
    active_by_dataset = {experiment["dataset"]: 0 for experiment in experiments}
    heavy_set = set(heavy_datasets)
    futures = {}
    completed = 0

    def dataset_limit(dataset: str) -> int:
        return heavy_dataset_jobs if dataset in heavy_set else dataset_jobs

    def submit_ready(executor: ThreadPoolExecutor) -> None:
        nonlocal pending
        index = 0
        while len(futures) < jobs and index < len(pending):
            experiment, seed = pending[index]
            dataset = experiment["dataset"]
            if active_by_dataset[dataset] >= dataset_limit(dataset):
                index += 1
                continue
            pending.pop(index)
            active_by_dataset[dataset] += 1
            future = executor.submit(_run_one_task, experiment, seed, logs_dir)
            futures[future] = (experiment, seed)

    try:
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            submit_ready(executor)
            total_tasks = len(pending) + len(futures)
            while futures:
                done, _running = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    experiment, seed = futures.pop(future)
                    active_by_dataset[experiment["dataset"]] -= 1
                    completed += 1
                    try:
                        outcome = future.result()
                    except Exception as exc:  # pragma: no cover - defensive guard
                        outcome = {
                            "seed": seed,
                            "ok": False,
                            "returncode": None,
                            "output": "",
                            "error": repr(exc),
                        }
                    _record_outcome(per_seed, failures, experiment, seed, outcome)
                    print(f"[eval-multi-seed] progress {completed}/{total_tasks}")
                if fail_fast and failures:
                    for future in futures:
                        future.cancel()
                    break
                submit_ready(executor)
    finally:
        pass


def _run_one_task(
    experiment: Dict[str, Any],
    seed: int,
    logs_dir: str,
) -> Dict[str, Any]:
    """Run one experiment/seed pair."""
    print(
        f"[eval-multi-seed] start "
        f"{experiment['display_model']}/{experiment['dataset']}/"
        f"{experiment['mode']}/seed{seed}"
    )
    return run_multi_seed.run_one_seed(seed, experiment["train_args"], logs_dir)


def _record_outcome(
    per_seed: Dict[str, Any],
    failures: List[Dict[str, Any]],
    experiment: Dict[str, Any],
    seed: int,
    outcome: Dict[str, Any],
) -> None:
    """Record one task outcome into success or failure collections."""
    display_model = experiment["display_model"]
    dataset = experiment["dataset"]
    mode = experiment["mode"]
    if outcome["ok"]:
        per_seed[display_model][dataset][mode][str(seed)] = outcome["result"]
        print(f"[eval-multi-seed] done {display_model}/{dataset}/{mode}/seed{seed}")
    else:
        failure = dict(outcome)
        failure.update({"model": display_model, "dataset": dataset, "mode": mode})
        failures.append(failure)
        print(f"[eval-multi-seed] failed {display_model}/{dataset}/{mode}/seed{seed}")


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
