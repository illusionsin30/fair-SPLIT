"""Unified training entry point with CLI support.

All models (CART / SPLIT / ReSPLIT) share the same data pipeline:
  1. Load raw data
  2. Preprocess target to binary {0,1} for SPLIT compatibility
  3. Single train / test split
  4. Train and evaluate → directly comparable results

Usage:
    python -m src.train --model cart  --dataset adult
    python -m src.train --model split --dataset bank
    python -m src.train --model resplit --dataset covertype
"""

import argparse
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
from .evaluate import evaluate_classification
from .tree import CART, SPLIT, ReSPLIT, LicketySPLIT
from .utils.helpers import used_split_features


MODEL_CHOICES = ["cart", "split", "licketysplit", "resplit"]
DATASET_CHOICES = [
    # SPLIT paper benchmarks
    "adult", "bike", "spambase", "bank", "covertype",
    "compass", "heloc", "thyroid",
    # Fairness / traditional benchmarks
    "german", "communities", "heart",
    "lawschool", "acsincome",
    # sklearn
    "iris", "diabetes",
]


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

    # --- required ---
    parser.add_argument(
        "--model", type=str, required=True, choices=MODEL_CHOICES,
        help="Tree model to train.",
    )
    parser.add_argument(
        "--dataset", type=str, required=True, choices=DATASET_CHOICES,
        help="Dataset to use.",
    )

    # --- data ---
    parser.add_argument(
        "--test_size", type=float, default=0.2,
        help="Test fraction (default: 0.2).",
    )
    parser.add_argument(
        "--random_state", type=int, default=42,
        help="Random seed (default: 42).",
    )

    # --- CART ---
    cart_group = parser.add_argument_group("CART")
    cart_group.add_argument("--binarize_cart", action="store_true",
                            help="Binarize features for CART (fair comparison "
                            "with SPLIT on the same feature space).")
    cart_group.add_argument("--max_depth", type=int, default=6)
    cart_group.add_argument("--min_samples_split", type=int, default=10)
    cart_group.add_argument("--min_samples_leaf", type=int, default=5)

    # --- SPLIT / ReSPLIT shared ---
    split_group = parser.add_argument_group("SPLIT / ReSPLIT")
    split_group.add_argument("--lookahead", type=int, default=2,
                             dest="lookahead_depth")
    split_group.add_argument("--depth", type=int, default=5,
                             dest="full_depth_budget")
    split_group.add_argument("--reg", type=float, default=0.0001,
                             help="Sparsity penalty λ per leaf (default: 0.0001).")
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

    # --- ReSPLIT ---
    resplit_group = parser.add_argument_group("ReSPLIT")
    resplit_group.add_argument("--rashomon_bound", type=float, default=0.01,
                               dest="rashomon_bound_multiplier")
    resplit_group.add_argument("--num_prefix", type=int, default=20,
                               dest="num_prefix_candidates")

    return parser


# ------------------------------------------------------------------
# Shared data pipeline
# ------------------------------------------------------------------

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


def prepare_data(name, test_size, random_state):
    """Load, preprocess target, split.

    Returns:
        (X_train, X_test, y_train, y_test, num_feats, cat_feats)
        Same split for all models.  Feature-type lists are only consumed
        by CART; SPLIT/ReSPLIT binarize everything internally.
    """
    X, y, num_feats, cat_feats = load_dataset(name)
    X, y = prepare_for_split(X, y)
    X_train, X_test, y_train, y_test = _train_test_split(
        X, y, test_size, random_state,
    )
    return X_train, X_test, y_train, y_test, num_feats, cat_feats


# ------------------------------------------------------------------
# Training dispatchers
# ------------------------------------------------------------------

def train_cart(args):
    """Train the CART model (classification)."""
    (X_train, X_test, y_train, y_test,
     num_feats, cat_feats) = prepare_data(
        args.dataset, args.test_size, args.random_state,
    )

    # If --binarize_cart is set, binarize features before passing to CART
    # so SPLIT vs CART comparisons use the same feature space.
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
    _show_used_features(model)
    print(f"Training time: {elapsed:.3f}s")


def train_split(args):
    """Train the SPLIT model."""
    X_train, X_test, y_train, y_test, _, _ = prepare_data(
        args.dataset, args.test_size, args.random_state,
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
    print(f"Training time: {elapsed:.3f}s")
    print(model.tree)


def train_resplit(args):
    """Train the ReSPLIT model."""
    X_train, X_test, y_train, y_test, _, _ = prepare_data(
        args.dataset, args.test_size, args.random_state,
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
    print(f"Rashomon set size: {n_trees} trees")
    for i in range(min(3, n_trees)):
        y_pred = model.predict(X_test, idx=i)
        obj = model.get_objective(i)
        acc = np.mean(y_pred == y_test)
        print(f"  Tree {i}: objective={obj:.6f}, test_accuracy={acc:.4f}")

    # Best tree
    print()
    y_best = model.predict(X_test, idx=0)
    evaluate_classification(y_test, y_best, y_train)
    _show_used_features(model)
    print(f"Training time: {elapsed:.3f}s")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _show_used_features(model, feature_names=None):
    """Print the set of features actually used for splits in the tree."""
    tree = None
    if hasattr(model, "root") and model.root is not None:
        tree = model.root                     # CART
    elif hasattr(model, "tree") and model.tree is not None:
        tree = model.tree                     # SPLIT / LicketySPLIT
    elif hasattr(model, "models") and len(model.models) > 0:
        tree = model.models[0][0]             # ReSPLIT (best tree from Rashomon set)
        if hasattr(tree, "tree"):
            tree = tree.tree  # might be wrapped
    if tree is None:
        return
    if feature_names is None:
        feature_names = getattr(model, "feature_names_", None)
    feats = used_split_features(tree, feature_names)
    if feats:
        print(f"Split features ({len(feats)}): {', '.join(str(f) for f in feats)}")
    else:
        print("Split features: (none — tree is a single leaf)")


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
    elif model_name in ("SPLIT", "LicketySPLIT"):
        return (f"lookahead={args.lookahead_depth}, "
                f"depth={args.full_depth_budget}, "
                f"reg={args.reg}, leaf_fill={args.leaf_fill}, "
                f"max_features={args.max_features or 'all'}")
    else:  # ReSPLIT
        return (f"lookahead={args.lookahead_depth}, "
                f"depth={args.full_depth_budget}, "
                f"reg={args.reg}, "
                f"rashomon_eps={args.rashomon_bound_multiplier}, "
                f"num_prefix={args.num_prefix_candidates}")


# ------------------------------------------------------------------
def train_licketysplit(args):
    """Train the LicketySPLIT model."""
    X_train, X_test, y_train, y_test, _, _ = prepare_data(
        args.dataset, args.test_size, args.random_state,
    )

    header("LicketySPLIT", args)
    t0 = time.perf_counter()

    model = LicketySPLIT(
        full_depth_budget=args.full_depth_budget,
        lookahead_range=args.lookahead_depth,
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
    print(f"Training time: {elapsed:.3f}s")
    print(model.tree)


# ------------------------------------------------------------------
# Dispatch
# ------------------------------------------------------------------

DISPATCH = {
    "cart": train_cart,
    "split": train_split,
    "licketysplit": train_licketysplit,
    "resplit": train_resplit,
}


def main(argv=None):
    parser = get_parser()
    args = parser.parse_args(argv)
    DISPATCH[args.model](args)


if __name__ == "__main__":
    main()
