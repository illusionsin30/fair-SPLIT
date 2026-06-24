"""Unified training entry point with CLI support.

All models (CART / SPLIT / ReSPLIT) share the same data pipeline:
  1. Load raw data
  2. Preprocess target to binary {0,1} for SPLIT compatibility
  3. Single train / test split
  4. Train and evaluate -> directly comparable results

Usage:
    python -m src.train --model cart  --dataset adult
    python -m src.train --model split --dataset bank
    python -m src.train --model resplit --dataset covertype
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .data import (
    load_adult_data, load_bike_data, load_spambase_data,
    load_bank_data, load_covertype_data, load_compass_data,
    load_heloc_data, load_thyroid_data,
    load_german_credit_data, load_communities_data,
    load_heart_data, load_law_school_data, load_acs_income_data,
    load_iris_data, load_diabetes_data,
)
from .data import prepare_for_split
from .evaluate import evaluate_classification, evaluate_fairness
from .utils.fairness import (
    apply_leaf_pareto_recalibration,
    fair_preprocess,
    fair_calibrate,
    fit_leaf_pareto_recalibrator,
)
from .utils.fairness import (
    apply_leaf_pareto_recalibration,
    fair_preprocess,
    fair_calibrate,
    fit_leaf_pareto_recalibrator,
)
from .tree import CART, SPLIT, ReSPLIT, LicketySPLIT
from .utils.helpers import used_split_features


MODEL_CHOICES = ["cart", "split", "licketysplit", "resplit"]
DATASET_CHOICES = [
    "adult", "bank", "compass", "german", "heloc", "spambase",
    "covertype", "thyroid", "communities", "heart",
    "lawschool", "acsincome",
]

SENSITIVE_ATTRS = {
    "adult": ["sex", "race"],
    "compass": ["sex", "race"],
    "german": ["personal_status", "age"],
    "bank": ["marital", "age"],
    "lawschool": ["race", "gender"],
    "acsincome": ["SEX", "RAC1P"],
    "communities": [],
    "spambase": [],
    "covertype": [],
    "thyroid": [],
    "heloc": [],
    "heart": [],
}


def _run_fairness_eval(
    y_test,
    y_pred,
    sensitive_test,
    args,
    label="",
    model=None,
    X_cal=None,
    y_cal=None,
    sensitive_cal=None,
    X_test=None,
):
def _run_fairness_eval(
    y_test,
    y_pred,
    sensitive_test,
    args,
    label="",
    model=None,
    X_cal=None,
    y_cal=None,
    sensitive_cal=None,
    X_test=None,
):
    """Evaluate fairness using pre-extracted sensitive values."""
    if not sensitive_test:
        return
    for col, vals in sensitive_test.items():
        if col is None or vals is None or col not in SENSITIVE_ATTRS.get(args.dataset, []):
            continue
        tag = f"{col}{' ' + label if label else ''}"
        evaluate_fairness(y_test, y_pred, vals, name=tag)
        if getattr(args, "fair", False):
        if getattr(args, "fair", False):
            from sklearn.metrics import accuracy_score
            acc_before = accuracy_score(y_test, y_pred)
            fair_post = getattr(args, "fair_post", "sample")
            if fair_post == "sample":
                y_calibrated = fair_calibrate(
                    y_pred,
                    y_test,
                    vals,
                    random_state=args.random_state,
                )
                evaluate_fairness(
                    y_test, y_calibrated, vals, name=f"{col} (calibrated)"
                )
                acc_after = accuracy_score(y_test, y_calibrated)
                print(f"  Calibrated accuracy: {acc_after:.4f}"
                      f" (was {acc_before:.4f}, delta={acc_after - acc_before:+.4f})")
            elif fair_post == "leaf_pareto":
                if (model is None or X_cal is None or y_cal is None or
                        sensitive_cal is None or X_test is None or
                        col not in sensitive_cal):
                    print(f"  [LeafPareto] Skipped '{col}': "
                          "calibration data unavailable")
                    continue
                recalibrator = fit_leaf_pareto_recalibrator(
                    model=model,
                    X_cal=X_cal,
                    y_cal=y_cal,
                    sensitive=sensitive_cal[col],
                    metric=getattr(args, "fair_metric", "dp"),
                    fair_lambda=getattr(args, "fair_lambda", 1.0),
                    acc_budget=getattr(args, "fair_acc_budget", 0.02),
                )
                y_calibrated = apply_leaf_pareto_recalibration(
                    model=model,
                    X=X_test,
                    base_pred=y_pred,
                    recalibrator=recalibrator,
                )
                evaluate_fairness(
                    y_test, y_calibrated, vals, name=f"{col} (LPFR)"
                )
                acc_after = accuracy_score(y_test, y_calibrated)
                print(f"  LPFR accuracy: {acc_after:.4f}"
                      f" (was {acc_before:.4f}, delta={acc_after - acc_before:+.4f})")


def get_parser():
    """Build an argparse parser for the training CLI."""
    parser = argparse.ArgumentParser(
        description="Train decision tree models (CART / SPLIT / ReSPLIT). "
        "All models use the same binary-classification data for "
        "direct comparison.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.train --model cart  --dataset adult
  python -m src.train --model split --dataset bank --lookahead 2 --depth 5
  python -m src.train --model resplit --dataset covertype --num_prefix 30
        """,
    )

    parser.add_argument(
        "--model", type=str, required=True, choices=MODEL_CHOICES,
        help="Tree model to train.",
    )
    parser.add_argument(
        "--dataset", type=str, required=True, choices=DATASET_CHOICES,
        help="Dataset to use.",
    )

    parser.add_argument(
        "--test_size", type=float, default=0.2,
        help="Test fraction (default: 0.2).",
    )
    parser.add_argument(
        "--random_state", type=int, default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--fair", action="store_true", default=False,
        help="Drop sensitive columns + per-group threshold calibration.",
    )
    parser.add_argument(
        "--fair_post", type=str, default="sample",
        choices=["sample", "leaf_pareto"],
        help="Fair post-processing method: sample or leaf_pareto.",
    )
    parser.add_argument(
        "--fair_metric", type=str, default="dp", choices=["dp", "eo"],
        help="Fairness metric for leaf_pareto: dp or eo.",
    )
    parser.add_argument(
        "--fair_lambda", type=float, default=1.0,
        help="Fairness gain weight for leaf_pareto post-processing.",
    )
    parser.add_argument(
        "--fair_acc_budget", type=float, default=0.02,
        help="Maximum calibration accuracy drop for leaf_pareto.",
    )
    parser.add_argument(
        "--fair_post", type=str, default="sample",
        choices=["sample", "leaf_pareto"],
        help="Fair post-processing method: sample or leaf_pareto.",
    )
    parser.add_argument(
        "--fair_metric", type=str, default="dp", choices=["dp", "eo"],
        help="Fairness metric for leaf_pareto: dp or eo.",
    )
    parser.add_argument(
        "--fair_lambda", type=float, default=1.0,
        help="Fairness gain weight for leaf_pareto post-processing.",
    )
    parser.add_argument(
        "--fair_acc_budget", type=float, default=0.02,
        help="Maximum calibration accuracy drop for leaf_pareto.",
    )
    parser.add_argument(
        "--results_dir", type=str, default="results",
        help="Directory for training logs (default: results).",
    )
    parser.add_argument(
        "--log_suffix", type=str, default="",
        help="Optional suffix inserted before .log, e.g. seed7.",
    )

    cart_group = parser.add_argument_group("CART")
    cart_group.add_argument("--binarize_cart", action="store_true",
                            help="Binarize features for CART (fair comparison "
                            "with SPLIT on the same feature space).")
    cart_group.add_argument("--max_depth", type=int, default=6)
    cart_group.add_argument("--min_samples_split", type=int, default=10)
    cart_group.add_argument("--min_samples_leaf", type=int, default=5)

    split_group = parser.add_argument_group("SPLIT / ReSPLIT")
    split_group.add_argument("--lookahead", type=int, default=2,
                             dest="lookahead_depth")
    split_group.add_argument("--depth", type=int, default=5,
                             dest="full_depth_budget")
    split_group.add_argument("--reg", type=float, default=0.0001,
                             help="Sparsity penalty lambda per leaf (default: 0.0001).")
    split_group.add_argument("--time_limit", type=int, default=60)
    split_group.add_argument("--binarizer", type=str, default="gbdt",
                             choices=["gbdt", "midpoint"])
    split_group.add_argument("--binarize", action="store_true", default=True)
    split_group.add_argument("--no-binarize", action="store_false",
                             dest="binarize")
    split_group.add_argument("--max_features", type=int, default=0,
                             help="Top-K features (0=all, default: 0). "
                             "Lower = faster, 0 = all features.")
    split_group.add_argument("--max_thresholds", type=int, default=50,
                             help="Max midpoints per numeric feature (default: 50).")
    split_group.add_argument("--leaf_fill", type=str, default="greedy",
                             choices=["greedy", "optimal"])

    resplit_group = parser.add_argument_group("ReSPLIT")
    resplit_group.add_argument("--rashomon_bound", type=float, default=0.01,
                               dest="rashomon_bound_multiplier")
    resplit_group.add_argument("--num_prefix", type=int, default=20,
                               dest="num_prefix_candidates")

    return parser


def load_dataset(name):
    """Load raw dataset."""
    mapping = {
        "adult": load_adult_data,
        "bike": load_bike_data,
        "spambase": load_spambase_data,
        "bank": load_bank_data,
        "covertype": load_covertype_data,
        "compass": load_compass_data,
        "heloc": load_heloc_data,
        "thyroid": load_thyroid_data,
        "german": load_german_credit_data,
        "communities": load_communities_data,
        "heart": load_heart_data,
        "lawschool": load_law_school_data,
        "acsincome": load_acs_income_data,
        "iris": load_iris_data,
        "diabetes": load_diabetes_data,
    }
    return mapping[name]()


def _safe_stratify(y):
    """Return y for stratification, or None when impossible."""
    if y.dtype.kind == "f":
        return None
    _, counts = np.unique(y, return_counts=True)
    if counts.min() < 2 or len(counts) > len(y) // 2:
        return None
    return y


def _train_test_split(X, y, test_size, random_state):
    """Split with automatic stratification fallback."""
    stratify = _safe_stratify(y)
    return train_test_split(
        X, y, test_size=test_size, stratify=stratify,
        random_state=random_state,
    )


def prepare_data(name, test_size, random_state, fair=False):
    """Load, preprocess target, split.

    Args:
        fair: If True, drop known sensitive columns before training.

    Returns:
        (X_train, X_test, y_train, y_test, num_feats, cat_feats)
        Same split for all models.
    """
    X, y, num_feats, cat_feats = load_dataset(name)
    X, y = prepare_for_split(X, y)

    sensitive_data = {}
    for col in SENSITIVE_ATTRS.get(name, []):
        if not col:
            continue
        if col in X.columns:
            vals = X[col].copy()
            if vals.dtype.kind in ("i", "f") and vals.nunique() > 10:
                _, cut_bins = pd.qcut(vals.unique(), q=4, retbins=True,
                                      duplicates="drop")
                n_bins = len(cut_bins) - 1
                labels = [f"{cut_bins[i]:.0f}-{cut_bins[i+1]:.0f}"
                          for i in range(n_bins)]
                vals = pd.cut(vals, bins=cut_bins, labels=labels,
                              include_lowest=True)
                print(f"  [Fairness] Discretized '{col}' into {n_bins} "
                      f"groups: {', '.join(labels)}")
            sensitive_data[col] = vals
        else:
            print(f"  [WARN] Fairness column '{col}' not found. "
                  f"Available: {list(X.columns)[:8]}...")

    if fair:
        sensitive_cols = list(sensitive_data.keys())
        X, _ = fair_preprocess(X, sensitive_cols)
        num_feats = [c for c in num_feats if c in X.columns]
        cat_feats = [c for c in cat_feats if c in X.columns]

    X_train, X_test, y_train, y_test = _train_test_split(
        X, y, test_size, random_state,
    )

    sensitive_train = {}
    sensitive_test = {}
    for col, vals in sensitive_data.items():
        sensitive_train[col] = vals.iloc[X_train.index].values
        sensitive_train[col] = vals.iloc[X_train.index].values
        sensitive_test[col] = vals.iloc[X_test.index].values

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        num_feats,
        cat_feats,
        sensitive_train,
        sensitive_test,
    )


def _unpack_prepared_data(prepared):
    """Unpack current or legacy prepare_data return values."""
    if len(prepared) == 8:
        return prepared
    if len(prepared) == 7:
        X_train, X_test, y_train, y_test, num_feats, cat_feats, sensitive_test = prepared
        return (
            X_train,
            X_test,
            y_train,
            y_test,
            num_feats,
            cat_feats,
            {},
            sensitive_test,
        )
    raise ValueError(f"prepare_data returned {len(prepared)} values; expected 7 or 8")


def train_cart(args):
    """Train the CART model (classification)."""
    prepared = prepare_data(
        args.dataset, args.test_size, args.random_state, fair=args.fair,
    )
    prepared = prepare_data(
        args.dataset, args.test_size, args.random_state, fair=args.fair,
    )
    (X_train, X_test, y_train, y_test,
     num_feats, cat_feats, sensitive_train, sensitive_test) = (
        _unpack_prepared_data(prepared)
     num_feats, cat_feats, sensitive_train, sensitive_test) = (
        _unpack_prepared_data(prepared)
    )

    if args.binarize_cart:
        from .utils.binarizer import NumericBinarizer
        enc = NumericBinarizer(max_thresholds=50)
        X_train_bin = pd.DataFrame(enc.fit_transform(X_train),
                                   columns=enc.get_feature_names_out())
        X_test_bin = pd.DataFrame(enc.transform(X_test),
                                  columns=enc.get_feature_names_out())
        X_train, X_test = X_train_bin, X_test_bin
        num_feats, cat_feats = list(X_train.columns), []
        print(f"  [CART] Binarized: {X_train.shape[1]} features")

    header("CART", args)
    t0 = time.perf_counter()

    model = CART(
        task="classification",
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        numeric_features=num_feats,
        categorical_features=cat_feats,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    elapsed = time.perf_counter() - t0

    evaluate_classification(y_test, y_pred, y_train)
    _show_used_features(model, X_test.columns.tolist() if hasattr(X_test, "columns") else None)
    _run_fairness_eval(
        y_test, y_pred, sensitive_test, args,
        model=model,
        X_cal=X_train,
        y_cal=y_train,
        sensitive_cal=sensitive_train,
        X_test=X_test,
    )
    _run_fairness_eval(
        y_test, y_pred, sensitive_test, args,
        model=model,
        X_cal=X_train,
        y_cal=y_train,
        sensitive_cal=sensitive_train,
        X_test=X_test,
    )
    print(f"Training time: {elapsed:.3f}s")


def train_split(args):
    """Train the SPLIT model."""
    prepared = prepare_data(
    prepared = prepare_data(
        args.dataset, args.test_size, args.random_state, fair=args.fair,
    )
    X_train, X_test, y_train, y_test, _, _, sensitive_train, sensitive_test = (
        _unpack_prepared_data(prepared)
    )
    X_train, X_test, y_train, y_test, _, _, sensitive_train, sensitive_test = (
        _unpack_prepared_data(prepared)
    )

    header("SPLIT", args)
    t0 = time.perf_counter()

    model = SPLIT(
        lookahead_depth=args.lookahead_depth,
        full_depth_budget=args.full_depth_budget,
        reg=args.reg,
        time_limit=args.time_limit,
        binarize=args.binarize,
        binarizer=args.binarizer,
        leaf_fill=args.leaf_fill,
        max_features=args.max_features,
        max_thresholds=args.max_thresholds,
        random_state=args.random_state,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    elapsed = time.perf_counter() - t0

    evaluate_classification(y_test, y_pred, y_train)
    print(f"Number of leaves: {model.num_leaves()}")
    _show_used_features(model)
    _run_fairness_eval(
        y_test, y_pred, sensitive_test, args,
        model=model,
        X_cal=X_train,
        y_cal=y_train,
        sensitive_cal=sensitive_train,
        X_test=X_test,
    )
    _run_fairness_eval(
        y_test, y_pred, sensitive_test, args,
        model=model,
        X_cal=X_train,
        y_cal=y_train,
        sensitive_cal=sensitive_train,
        X_test=X_test,
    )
    print(f"Training time: {elapsed:.3f}s")
    print(model.tree)


def train_resplit(args):
    """Train the ReSPLIT model."""
    prepared = prepare_data(
    prepared = prepare_data(
        args.dataset, args.test_size, args.random_state, fair=args.fair,
    )
    X_train, X_test, y_train, y_test, _, _, sensitive_train, sensitive_test = (
        _unpack_prepared_data(prepared)
    )
    X_train, X_test, y_train, y_test, _, _, sensitive_train, sensitive_test = (
        _unpack_prepared_data(prepared)
    )

    header("ReSPLIT", args)
    t0 = time.perf_counter()

    model = ReSPLIT(
        lookahead_depth=args.lookahead_depth,
        full_depth_budget=args.full_depth_budget,
        reg=args.reg,
        rashomon_bound_multiplier=args.rashomon_bound_multiplier,
        num_prefix_candidates=args.num_prefix_candidates,
        time_limit=args.time_limit,
        binarize=args.binarize,
        binarizer=args.binarizer,
        max_features=args.max_features,
        max_thresholds=args.max_thresholds,
        random_state=args.random_state,
    )
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0

    n_trees = len(model)
    print(f"candidate tree set size: {n_trees} trees")
    for i in range(min(3, n_trees)):
        y_pred = model.predict(X_test, idx=i)
        obj = model.get_objective(i)
        acc = np.mean(y_pred == y_test)
        print(f"  Tree {i}: objective={obj:.6f}, test_accuracy={acc:.4f}")

    print()
    y_best = model.predict(X_test, idx=0)
    evaluate_classification(y_test, y_best, y_train)
    _show_used_features(model)
    _run_fairness_eval(
        y_test, y_best, sensitive_test, args,
        model=model,
        X_cal=X_train,
        y_cal=y_train,
        sensitive_cal=sensitive_train,
        X_test=X_test,
    )
    _run_fairness_eval(
        y_test, y_best, sensitive_test, args,
        model=model,
        X_cal=X_train,
        y_cal=y_train,
        sensitive_cal=sensitive_train,
        X_test=X_test,
    )
    print(f"Training time: {elapsed:.3f}s")


def _show_used_features(model, feature_names=None):
    """Print the set of features actually used for splits in the tree.

    For SPLIT/ReSPLIT/LicketySPLIT (which binarize internally), the model
    stores binarized feature names in ``feature_names_``.  For CART the tree
    nodes already hold human-readable column names or binarized threshold
    strings pulled from the training DataFrame.
    """
    tree = None
    if hasattr(model, "root") and model.root is not None:
        tree = model.root
    elif hasattr(model, "tree") and model.tree is not None:
        tree = model.tree
    elif hasattr(model, "models") and len(model.models) > 0:
        tree = model.models[0][0]
        if hasattr(tree, "tree"):
            tree = tree.tree
    if tree is None:
        return
    binarized_names = getattr(model, "feature_names_", None)
    if binarized_names is not None:
        feats = used_split_features(tree, binarized_names)
    else:
        feats = used_split_features(tree, feature_names)
    if feats:
        print(f"Split features ({len(feats)}): {', '.join(str(f) for f in feats)}")
    else:
        print("Split features: (none - tree is a single leaf)")


def header(model_name, args):
    print("=" * 60)
    print(f"Model: {model_name}  |  Dataset: {args.dataset}")
    print(f"Params: {_param_summary(model_name, args)}")
    print("=" * 60)


def _param_summary(model_name, args):
    if model_name == "CART":
        return (f"max_depth={args.max_depth}, "
                f"min_samples_split={args.min_samples_split}, "
                f"min_samples_leaf={args.min_samples_leaf}")
    elif model_name in ("SPLIT",):
        return (f"lookahead={args.lookahead_depth}, "
                f"depth={args.full_depth_budget}, "
                f"reg={args.reg}, leaf_fill={args.leaf_fill}, "
                f"max_features={args.max_features or 'all'}")
    elif model_name == "LicketySPLIT":
        return (f"depth={args.full_depth_budget}, "
                f"reg={args.reg}, "
                f"max_features={args.max_features or 'all'}")
    else:
        return (f"lookahead={args.lookahead_depth}, "
                f"depth={args.full_depth_budget}, "
                f"reg={args.reg}, "
                f"rashomon_eps={args.rashomon_bound_multiplier}, "
                f"num_prefix={args.num_prefix_candidates}")


def train_licketysplit(args):
    """Train the LicketySPLIT model."""
    prepared = prepare_data(
    prepared = prepare_data(
        args.dataset, args.test_size, args.random_state, fair=args.fair,
    )
    X_train, X_test, y_train, y_test, _, _, sensitive_train, sensitive_test = (
        _unpack_prepared_data(prepared)
    )
    X_train, X_test, y_train, y_test, _, _, sensitive_train, sensitive_test = (
        _unpack_prepared_data(prepared)
    )

    header("LicketySPLIT", args)
    t0 = time.perf_counter()

    model = LicketySPLIT(
        full_depth_budget=args.full_depth_budget,
        reg=args.reg,
        verbose=True,
        binarize=args.binarize,
        binarizer=args.binarizer,
        max_features=args.max_features,
        max_thresholds=args.max_thresholds,
        gbdt_n_est=100,
        gbdt_max_depth=1,
        random_state=args.random_state,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    elapsed = time.perf_counter() - t0

    evaluate_classification(y_test, y_pred, y_train)
    print(f"Number of leaves: {model.num_leaves()}")
    _show_used_features(model)
    _run_fairness_eval(
        y_test, y_pred, sensitive_test, args,
        model=model,
        X_cal=X_train,
        y_cal=y_train,
        sensitive_cal=sensitive_train,
        X_test=X_test,
    )
    _run_fairness_eval(
        y_test, y_pred, sensitive_test, args,
        model=model,
        X_cal=X_train,
        y_cal=y_train,
        sensitive_cal=sensitive_train,
        X_test=X_test,
    )
    print(f"Training time: {elapsed:.3f}s")
    print(model.tree)


DISPATCH = {
    "cart": train_cart,
    "split": train_split,
    "licketysplit": train_licketysplit,
    "resplit": train_resplit,
}


class _Tee:
    """Write to both stdout and a log file."""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def _make_filename(args):
    """Build log filename."""
    model = args.model
    dataset = args.dataset
    if model == "cart":
        depth = getattr(args, "max_depth", 6)
    else:
        depth = getattr(args, "full_depth_budget", 5)
    if model == "cart":
        binarize = "bin" if getattr(args, "binarize_cart", False) else "raw"
    else:
        binarize = "raw" if getattr(args, "binarize", True) is False else "bin"
    if args.fair and getattr(args, "fair_post", "sample") == "leaf_pareto":
        mode = "lpfr"
    elif args.fair:
        mode = "fair"
    else:
        mode = "nofair"
    if model == "split":
        leaf = getattr(args, "leaf_fill", "greedy")
        display_model = f"split-{leaf}"
    else:
        display_model = model

    filename = f"{display_model}-{dataset}-d{depth}-{binarize}-{mode}"
    filename = f"{display_model}-{dataset}-d{depth}-{binarize}-{mode}"
    if getattr(args, "log_suffix", ""):
        safe_suffix = str(args.log_suffix).strip().replace(os.sep, "_")
        filename = f"{filename}-{safe_suffix}"
    return os.path.join(getattr(args, "results_dir", "results"), f"{filename}.log")


def main(argv=None):
    parser = get_parser()
    args = parser.parse_args(argv)

    log_path = _make_filename(args)
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    tee = _Tee(log_path)
    sys.stdout = tee

    try:
        DISPATCH[args.model](args)
    finally:
        sys.stdout = tee.terminal
        tee.log.close()
        print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    main()
