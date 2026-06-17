# ============================================================
# Unified CART (classification + regression) — from scratch
# ============================================================
# A single tree class parameterised by `task`:
#   - "classification": Gini impurity, majority-vote leaves, supports
#                       binary AND multiclass targets.
#   - "regression":     variance impurity (MSE), mean-value leaves.
#
# Categorical splits use Breiman's classical shortcut:
#   * Binary classification: sort categories by P(y=1 | category)
#   * Regression:            sort categories by E[y | category]
#   * Multiclass:            no closed-form prefix shortcut, so we either
#                            brute-force subsets when K is small, or skip.

from itertools import combinations

import numpy as np
import pandas as pd


class Node:
    """A single tree node — either internal (with a split) or a leaf."""

    __slots__ = (
        "feature",
        "is_numeric",
        "threshold",
        "left_categories",
        "left",
        "right",
        "prediction",
    )

    def __init__(self):
        self.feature = None
        self.is_numeric = None
        self.threshold = None
        self.left_categories = None
        self.left = None
        self.right = None
        self.prediction = None  # class label OR mean target value


class CART:
    """CART for classification (binary/multiclass) or regression."""

    def __init__(
        self,
        task="classification",
        max_depth=6,
        min_samples_split=20,
        min_samples_leaf=10,
        numeric_features=None,
        categorical_features=None,
        max_cat_brute_force=10,
    ):
        assert task in ("classification", "regression")
        self.task = task
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.numeric_features = list(numeric_features or [])
        self.categorical_features = list(categorical_features or [])
        # If a categorical feature has more distinct levels than this and the
        # problem is multiclass, we skip it rather than enumerate 2^K subsets.
        self.max_cat_brute_force = max_cat_brute_force
        self.root = None
        self._classes = None  # populated in fit() for classification

    # ---------- Impurity ----------
    def _impurity(self, y):
        """Gini for classification, variance for regression."""
        if len(y) == 0:
            return 0.0
        if self.task == "classification":
            _, counts = np.unique(y, return_counts=True)
            p = counts / counts.sum()
            return 1.0 - np.sum(p * p)
        # regression: population variance acts as MSE around the mean
        return float(np.var(y))

    def _weighted_impurity(self, y_left, y_right):
        n = len(y_left) + len(y_right)
        return (len(y_left) / n) * self._impurity(y_left) + (
            len(y_right) / n
        ) * self._impurity(y_right)

    # ---------- Leaf prediction ----------
    def _leaf_value(self, y):
        if self.task == "classification":
            vals, counts = np.unique(y, return_counts=True)
            if len(counts) == 0:
                return 0  # fallback for empty node
            return vals[np.argmax(counts)]
        return float(np.mean(y))

    # ---------- Numeric split ----------
    def _best_numeric_split(self, column, y):
        """Sweep candidate thresholds in O(N log N).

        Classification uses cumulative class counts; regression uses
        cumulative sums / sums-of-squares so variance at every cut is
        constant-time.
        """
        values = np.asarray(column.values)
        order = np.argsort(values, kind="mergesort")
        v_sorted = values[order]
        y_sorted = np.asarray(y)[order]
        n = len(y_sorted)
        if n < 2 * self.min_samples_leaf:
            return 0.0, None

        parent_imp = self._impurity(y_sorted)
        best_gain = 0.0
        best_threshold = None

        if self.task == "classification":
            classes = self._classes
            class_to_idx = {c: i for i, c in enumerate(classes)}
            y_idx = np.array([class_to_idx[v] for v in y_sorted])
            cum = np.zeros((n + 1, len(classes)), dtype=np.int64)
            for i in range(n):
                cum[i + 1] = cum[i]
                cum[i + 1, y_idx[i]] += 1
            total = cum[-1]

            for i in range(self.min_samples_leaf, n - self.min_samples_leaf + 1):
                if v_sorted[i] == v_sorted[i - 1]:
                    continue
                left_counts = cum[i]
                right_counts = total - left_counts
                n_left, n_right = i, n - i
                p_left = left_counts / n_left
                p_right = right_counts / n_right
                g_left = 1.0 - np.sum(p_left * p_left)
                g_right = 1.0 - np.sum(p_right * p_right)
                w_imp = (n_left / n) * g_left + (n_right / n) * g_right
                gain = parent_imp - w_imp
                if gain > best_gain:
                    best_gain = gain
                    best_threshold = (v_sorted[i - 1] + v_sorted[i]) / 2.0
        else:
            # Regression: variance via cumulative sum and sum-of-squares.
            cs = np.concatenate([[0.0], np.cumsum(y_sorted)])
            css = np.concatenate([[0.0], np.cumsum(y_sorted * y_sorted)])
            total_s, total_ss = cs[-1], css[-1]

            for i in range(self.min_samples_leaf, n - self.min_samples_leaf + 1):
                if v_sorted[i] == v_sorted[i - 1]:
                    continue
                n_left, n_right = i, n - i
                s_left, ss_left = cs[i], css[i]
                s_right, ss_right = total_s - s_left, total_ss - ss_left
                var_left = ss_left / n_left - (s_left / n_left) ** 2
                var_right = ss_right / n_right - (s_right / n_right) ** 2
                w_imp = (n_left / n) * var_left + (n_right / n) * var_right
                gain = parent_imp - w_imp
                if gain > best_gain:
                    best_gain = gain
                    best_threshold = (v_sorted[i - 1] + v_sorted[i]) / 2.0
        return best_gain, best_threshold

    # ---------- Categorical split ----------
    def _ordered_categories(self, values, y, categories):
        """Order categories so the optimal split is some prefix.

        - Binary classification: order by P(y=1 | category)
        - Regression:            order by E[y | category]
        Both are Breiman's provably-optimal orderings.
        """
        keys = []
        for c in categories:
            mask = values == c
            keys.append((c, y[mask].mean() if mask.sum() else 0.0))
        keys.sort(key=lambda t: t[1])
        return [c for c, _ in keys]

    def _best_categorical_split(self, column, y):
        values = np.asarray(column.values)
        categories = np.unique(values)
        if len(categories) <= 1:
            return 0.0, None

        is_binary_clf = (
            self.task == "classification" and len(self._classes) == 2
        )
        is_regression = self.task == "regression"

        # Candidate "left" subsets to try.
        if is_binary_clf or is_regression:
            sorted_cats = self._ordered_categories(values, y, categories)
            # Optimal split is a prefix of the sorted order — O(K log K).
            candidate_left_sets = [
                set(sorted_cats[:k]) for k in range(1, len(sorted_cats))
            ]
        else:
            # Multiclass: brute-force all non-trivial subsets when small,
            # otherwise skip to keep training tractable.
            K = len(categories)
            if K > self.max_cat_brute_force:
                return 0.0, None
            cat_list = list(categories)
            candidate_left_sets = []
            # Enumerate halves once (a subset and its complement give the
            # same partition, so we stop at K/2).
            for r in range(1, K // 2 + 1):
                for combo in combinations(cat_list, r):
                    candidate_left_sets.append(set(combo))

        parent_imp = self._impurity(y)
        best_gain = 0.0
        best_left = None
        n = len(y)

        for left_set in candidate_left_sets:
            left_mask = np.array([v in left_set for v in values])
            n_left = left_mask.sum()
            n_right = n - n_left
            if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                continue
            w_imp = self._weighted_impurity(y[left_mask], y[~left_mask])
            gain = parent_imp - w_imp
            if gain > best_gain:
                best_gain = gain
                best_left = left_set
        return best_gain, best_left

    def _best_split(self, X, y):
        best = {"gain": 0.0}
        for feat in self.numeric_features:
            gain, thr = self._best_numeric_split(X[feat], y)
            if gain > best["gain"] and thr is not None:
                best = {
                    "gain": gain,
                    "feature": feat,
                    "is_numeric": True,
                    "threshold": thr,
                }
        for feat in self.categorical_features:
            gain, left_set = self._best_categorical_split(X[feat], y)
            if gain > best["gain"] and left_set is not None:
                best = {
                    "gain": gain,
                    "feature": feat,
                    "is_numeric": False,
                    "left_categories": left_set,
                }
        return best if best["gain"] > 0 else None

    # ---------- Recursive build ----------
    def _build(self, X, y, depth):
        node = Node()
        # Stop if pure (classification) / zero-variance (regression),
        # at depth limit, or below the minimum split size.
        if (
            depth >= self.max_depth
            or len(y) < self.min_samples_split
            or self._impurity(y) == 0.0
        ):
            node.prediction = self._leaf_value(y)
            return node

        split = self._best_split(X, y)
        if split is None:
            node.prediction = self._leaf_value(y)
            return node

        node.feature = split["feature"]
        node.is_numeric = split["is_numeric"]
        if node.is_numeric:
            node.threshold = split["threshold"]
            left_mask = X[node.feature].values <= node.threshold
        else:
            node.left_categories = split["left_categories"]
            left_mask = np.array(
                [v in node.left_categories for v in X[node.feature].values]
            )

        X_left = X[left_mask].reset_index(drop=True)
        X_right = X[~left_mask].reset_index(drop=True)
        y_left = y[left_mask]
        y_right = y[~left_mask]

        node.left = self._build(X_left, y_left, depth + 1)
        node.right = self._build(X_right, y_right, depth + 1)
        return node

    # ---------- Public API ----------
    def fit(self, X, y):
        X = X.reset_index(drop=True)
        y = np.asarray(y)
        if self.task == "classification":
            self._classes = np.unique(y)
        else:
            y = y.astype(float)
        self.root = self._build(X, y, depth=0)
        return self

    def _predict_row(self, row, node):
        # Leaves are the only nodes with no children.
        if node.left is None:
            return node.prediction
        if node.is_numeric:
            go_left = row[node.feature] <= node.threshold
        else:
            go_left = row[node.feature] in node.left_categories
        return self._predict_row(row, node.left if go_left else node.right)

    def predict(self, X):
        return np.array(
            [self._predict_row(row, self.root) for _, row in X.iterrows()]
        )


# Backwards-compatible alias for convenience.
def CARTClassifier(**kw):
    return CART(task="classification", **kw)


# ============================================================
# SPLIT — SParse Lookahead for Interpretable Trees
# ============================================================
# Implements Algorithm 2 from the SPLIT paper:
#
#   Phase 1: ModifiedGOSDT (our OptimalTreeSolver) with
#            get_bounds (Algorithm 1/12) finds the optimal prefix
#            tree of depth dl, where the boundary condition at
#            d' == dl sets lb = ub = greedy completion loss.
#   Phase 2: (optional post-processing)  For each leaf of the
#            prefix, replace the greedy subtree with an optimal
#            GOSDT subtree (our solver without lookahead boundary)
#            using renormalised λ.
#
# Reference:
#   "Fast Sparse Decision Tree Optimization via Reference Ensembles"
#   (https://doi.org/10.1609/aaai.v36i9.21194)   — AAAI 2022
#   "Near-Optimal Decision Trees in a SPLIT Second"
#   — ICML 2025

from .solver import OptimalTreeSolver
from .builder import GreedyTreeBuilder
from .utils.nodes import SPLITLeaf, SPLITNode
from .utils.helpers import predict_sample, predict_batch, tree_to_dict, num_leaves
from .utils.binarizer import NumericBinarizer, ThresholdGuessBinarizer


class SPLIT:
    """SPLIT: optimal shallow prefix + filled leaves (Algorithm 2).

    Parameters
    ----------
    lookahead_depth : int
        Depth of the optimal prefix tree (Phase 1, dl in paper).
        Must be >= 1.  dl=1 means root-level optimisation only.
    full_depth_budget : int
        Total depth budget d for the final tree.  0 = no limit.
    reg : float
        Regularisation penalty λ per leaf (sparsity penalty).
    time_limit : float
        Wall-clock time limit (seconds) for the optimal prefix solver.
    verbose : bool
        Print progress information.
    binarize : bool
        Binarize continuous features before training.
    binarizer : str
        "midpoint" (lossless) or "gbdt" (compact).
    gbdt_n_est : int
        Number of GBDT trees (only for "gbdt" binarizer).
    gbdt_max_depth : int
        Max depth of each GBDT tree.
    leaf_fill : str
        Strategy for filling prefix leaves —
        "greedy" (fast, Algorithm 4) or "optimal" (slower, GOSDT).
    max_features : int
        Top-K candidate features (0 = all).
    max_thresholds : int
        Max midpoints per numeric feature.
    random_state : int
        Random seed.
    """

    def __init__(
        self,
        lookahead_depth=2,
        full_depth_budget=5,
        reg=0.0001,
        time_limit=60,
        verbose=False,
        binarize=True,
        binarizer="gbdt",
        gbdt_n_est=100,
        gbdt_max_depth=1,
        leaf_fill="optimal",
        max_features=0,
        max_thresholds=50,
        random_state=42,
    ):
        if lookahead_depth < 1:
            raise ValueError(
                f"lookahead_depth must be >= 1, got {lookahead_depth}. "
                "A value of 0 means no lookahead (use a plain greedy tree instead)."
            )

        self.lookahead_depth = lookahead_depth          # dl
        self.full_depth_budget = full_depth_budget       # d
        self.reg = reg                                   # λ
        self.time_limit = time_limit
        self.verbose = verbose
        self.binarize_flag = binarize
        self.binarizer_type = binarizer
        self.gbdt_n_est = gbdt_n_est
        self.gbdt_max_depth = gbdt_max_depth
        self.leaf_fill = leaf_fill
        self.max_features = max_features
        self.max_thresholds = max_thresholds
        self.random_state = random_state

        # Derived
        self._has_no_depth_limit = full_depth_budget == 0
        if self._has_no_depth_limit:
            self._remaining_depth = 0   # unlimited in practice
        else:
            self._remaining_depth = full_depth_budget - lookahead_depth

        self.classes_ = None
        self.tree = None
        self._enc = None
        self._n_total = 0    # global N for loss normalisation

    # ------------------------------------------------------------------
    # fit / predict
    # ------------------------------------------------------------------

    def fit(self, X, y):
        """Train SPLIT (Algorithm 2).

        Parameters
        ----------
        X : DataFrame or ndarray of features.
        y : array-like of binary labels {0, 1}.
        """
        y = np.asarray(y, dtype=np.int64)
        self.classes_ = np.unique(y).tolist()
        self._n_total = len(y)          # global N

        # Binarization
        if self.binarize_flag:
            X_bin = self._binarize_fit_transform(X, y)
            if self._enc is not None:
                self.feature_names_ = list(self._enc.get_feature_names_out())
            else:
                self.feature_names_ = [f"f{i}" for i in range(X_bin.shape[1])]
        else:
            X_bin = np.asarray(X, dtype=bool)
            self.feature_names_ = [f"f{i}" for i in range(X_bin.shape[1])]

        # ----------------------------------------------------------------
        # Phase 1: optimal prefix via DP with greedy boundary
        # (Algorithm 2, lines 1-2)
        #
        # ModifiedGOSDT with get_bounds: at lookahead depth dl,
        # bounds are set to greedy completion loss.
        # ----------------------------------------------------------------
        prefix_solver = OptimalTreeSolver(
            depth_budget=self.lookahead_depth,
            reg=self.reg,
            time_limit=self.time_limit,
            max_features=self.max_features,
            remaining_depth=self._remaining_depth,
            global_N=self._n_total,
        )

        # Greedy upper bound for DP pruning (optional but speeds things up)
        greedy_tree, greedy_loss_sub = GreedyTreeBuilder(
            depth_budget=self.full_depth_budget,
            reg=self.reg,
            max_features=self.max_features,
            global_N=self._n_total,
        ).build(X_bin, y)
        greedy_leaves = num_leaves(greedy_tree)
        greedy_loss_global = self._compute_loss(greedy_tree, X_bin, y)
        print(f"[SPLIT] Binarized features: {X_bin.shape[1]}, "
              f"greedy leaves: {greedy_leaves}, "
              f"greedy loss (global): {greedy_loss_global:.6f}")

        prefix_tree, prefix_loss = prefix_solver.fit(
            X_bin, y, upper_bound_tree=(greedy_tree, greedy_loss_global),
        )
        prefix_leaves = num_leaves(prefix_tree)
        print(f"[SPLIT] Optimal prefix loss: {prefix_loss:.6f}, "
              f"leaves: {prefix_leaves}")

        # ----------------------------------------------------------------
        # Phase 2: post-processing (Algorithm 2, lines 3-9)
        #
        # For each leaf of the prefix, replace with optimal subtree.
        # Depth is tracked from the root so each leaf gets
        # remaining_depth = full_depth_budget - current_depth.
        # ----------------------------------------------------------------
        if (self._remaining_depth > 0 or self._has_no_depth_limit):
            if self.verbose:
                print(
                    f"[SPLIT] Phase 2: filling leaves "
                    f"(method={self.leaf_fill})..."
                )
            self.tree = self._fill_leaves(
                prefix_tree, X_bin, y, current_depth=0)
        else:
            self.tree = prefix_tree

        return self

    def predict(self, X):
        """Predict class labels for X."""
        if self.tree is None:
            raise RuntimeError("Model not fitted yet. Call fit() first.")

        if self.binarize_flag and self._enc is not None:
            X_bin = np.asarray(self._enc.transform(X), dtype=bool)
        else:
            X_bin = np.asarray(X, dtype=bool)

        classes = np.array(self.classes_)
        return np.array(
            [predict_sample(X_bin[i], self.tree, classes)
             for i in range(len(X_bin))]
        )

    # ------------------------------------------------------------------
    # Tree utilities
    # ------------------------------------------------------------------

    def tree_to_dict(self):
        """Export the fitted tree as a nested dict."""
        if self.tree is None:
            raise RuntimeError("Model not fitted yet.")
        return tree_to_dict(self.tree, self.classes_)

    def num_leaves(self):
        """Return the number of leaves in the fitted tree."""
        if self.tree is None:
            raise RuntimeError("Model not fitted yet.")
        return num_leaves(self.tree)

    def __str__(self):
        if self.tree is None:
            return "SPLIT(unfitted)"
        return str(self.tree)

    # ------------------------------------------------------------------
    # Internal: binarization
    # ------------------------------------------------------------------

    def _binarize_fit_transform(self, X, y):
        """Fit a binarizer and return the binary feature matrix."""
        if self.binarizer_type == "gbdt":
            self._enc = ThresholdGuessBinarizer(
                n_estimators=self.gbdt_n_est,
                max_depth=self.gbdt_max_depth,
                random_state=self.random_state,
            )
        else:
            self._enc = NumericBinarizer(max_thresholds=self.max_thresholds)
        return np.asarray(self._enc.fit_transform(X, y), dtype=bool)

    # ------------------------------------------------------------------
    # Internal: leaf filling (Algorithm 2, lines 3-9)
    # ------------------------------------------------------------------

    def _fill_leaves(self, node, X, y, current_depth):
        """Recursively replace every leaf with a completed subtree.

        Algorithm 2 lines 3-9: for each prefix leaf at depth du,
        find an optimal subtree of depth d-du using global-N
        normalisation with the original λ.
        """
        if isinstance(node, SPLITLeaf):
            n_local = len(y)
            if n_local == 0:
                return node

            # Remaining depth budget for this leaf's subtree.
            remaining_depth = self.full_depth_budget - current_depth
            if self._has_no_depth_limit:
                remaining_depth = max(remaining_depth, 1)

            if self.leaf_fill == "optimal" and remaining_depth > 0:
                sub_solver = OptimalTreeSolver(
                    depth_budget=remaining_depth,
                    reg=self.reg,
                    time_limit=self.time_limit,
                    max_features=self.max_features,
                    remaining_depth=0,
                    global_N=self._n_total,
                )
                subtree, _ = sub_solver.fit(X, y)
                return self._remap_tree(subtree, self.classes_)
            else:
                # Greedy leaf fill — Algorithm 4 with global-N normalisation
                builder = GreedyTreeBuilder(
                    depth_budget=remaining_depth,
                    reg=self.reg,
                    max_features=self.max_features,
                    global_N=self._n_total,
                )
                subtree, _ = builder.build(X, y)
                return subtree
        else:
            # Internal node: recurse on children, incrementing depth.
            left_mask = X[:, node.feature]
            right_mask = ~left_mask
            node.left_child = self._fill_leaves(
                node.left_child, X[left_mask], y[left_mask],
                current_depth + 1,
            )
            node.right_child = self._fill_leaves(
                node.right_child, X[right_mask], y[right_mask],
                current_depth + 1,
            )
            return node

    @staticmethod
    def _remap_tree(node, target_classes):
        """Ensure leaf predictions use indices into target_classes."""
        if isinstance(node, SPLITLeaf):
            return SPLITLeaf(
                prediction=int(node.prediction),
                loss=node.loss,
            )
        return SPLITNode(
            feature=node.feature,
            left_child=SPLIT._remap_tree(node.left_child, target_classes),
            right_child=SPLIT._remap_tree(node.right_child, target_classes),
        )

    def _compute_loss(self, tree, X, y):
        """Compute L(T, D, λ) = (1/N) Σ errors + λ · num_leaves."""
        preds = predict_batch(X, tree, np.array(self.classes_))
        n_err = int(np.sum(preds != y))
        return n_err / self._n_total + self.reg * num_leaves(tree)


# ============================================================
# ReSPLIT — Rashomon set via SPLIT prefixes
# ============================================================
# Approximation of Algorithm 5 (RESPLIT) from the paper.
# Instead of ModifiedTreeFARMS (which requires C++ GOSDT), we:
#   1. Generate diverse prefix candidates via our DP solver with
#      a relaxed optimality bound (emulating a subset of TreeFARMS
#      output — the prefixes whose best greedy completion is
#      within ε of optimal).
#   2. For each leaf, use Greedy (Alg 4) to set an upper bound Lg,
#      then call our DP solver to enumerate multiple near-optimal
#      subtrees (those with loss ≤ Lg).
#   3. Filter the cross-product to the Rashomon set.
#
# This trades exact TreeFARMS enumeration for our pure-Python BnB,
# faithfully following the paper's Algorithm 5 structure but with
# our own solver backend.


class ReSPLIT:
    """Approximate Rashomon set per Algorithm 5 (RESPLIT).

    Parameters
    ----------
    lookahead_depth : int
        Depth of each prefix tree (dl in paper).
    full_depth_budget : int
        Total depth budget d.
    reg : float
        Regularisation penalty λ per leaf.
    rashomon_bound_multiplier : float
        ε — trees with objective ≤ (1+ε)·best_objective are kept.
    num_prefix_candidates : int
        Number of diverse prefixes to generate.
    time_limit : float
        Per-prefix solver time limit (seconds).
    verbose : bool
        Print progress.
    binarize : bool
        Binarize continuous features.
    binarizer : str
        "midpoint" or "gbdt".
    max_features : int
        Top-K candidate features (0 = all).
    max_thresholds : int
        Max midpoints per numeric feature.
    random_state : int
        Random seed.
    """

    def __init__(
        self,
        lookahead_depth=3,
        full_depth_budget=5,
        reg=0.0001,
        rashomon_bound_multiplier=0.01,
        num_prefix_candidates=20,
        time_limit=10,
        verbose=False,
        binarize=True,
        binarizer="gbdt",
        max_features=0,
        max_thresholds=50,
        random_state=42,
    ):
        self.lookahead_depth = lookahead_depth
        self.full_depth_budget = full_depth_budget
        self.reg = reg
        self.rashomon_bound_multiplier = rashomon_bound_multiplier
        self.num_prefix_candidates = num_prefix_candidates
        self.time_limit = time_limit
        self.verbose = verbose
        self.binarize_flag = binarize
        self.binarizer_type = binarizer
        self.max_features = max_features
        self.max_thresholds = max_thresholds
        self.random_state = random_state

        self._remaining_depth = full_depth_budget - lookahead_depth
        self._has_no_depth_limit = full_depth_budget == 0

        self.classes_ = None
        self.models = []          # list of (tree, objective) tuples
        self._enc = None
        self._n_total = 0
        self._rng = np.random.RandomState(random_state)

    # ------------------------------------------------------------------
    # fit / access
    # ------------------------------------------------------------------

    def fit(self, X, y):
        """Build the Rashomon set."""
        y = np.asarray(y, dtype=np.int64)
        self.classes_ = np.unique(y).tolist()
        self._n_total = len(y)

        if self.binarize_flag:
            X_bin = self._binarize_fit_transform(X, y)
            if self._enc is not None:
                self.feature_names_ = list(self._enc.get_feature_names_out())
            else:
                self.feature_names_ = [f"f{i}" for i in range(X_bin.shape[1])]
        else:
            X_bin = np.asarray(X, dtype=bool)
            self.feature_names_ = [f"f{i}" for i in range(X_bin.shape[1])]

        all_features = list(range(X_bin.shape[1]))
        candidates = []

        if self.verbose:
            print(
                f"[ReSPLIT] Generating {self.num_prefix_candidates} "
                f"prefix candidates..."
            )

        # Generate diverse prefixes by shuffling feature order
        # (emulates ModifiedTreeFARMS's diverse prefix enumeration)
        for i in range(self.num_prefix_candidates):
            seed = self._rng.randint(0, 2**31 - 1)
            local_rng = np.random.RandomState(seed)

            # Shuffle feature order for diversity
            shuffled_features = all_features.copy()
            local_rng.shuffle(shuffled_features)

            # Build a greedy prefix using the shuffled order
            prefix_tree, _ = self._greedy_prefix(
                X_bin, y, shuffled_features, self.lookahead_depth
            )

            # For each leaf, get Greedy upper bound Lg (Algorithm 5, line 7)
            # then enumerate subtree(s) with loss ≤ Lg (Algorithm 5, line 8)
            if self._remaining_depth > 0 or self._has_no_depth_limit:
                fill_depth = self._remaining_depth
                full_trees = self._enumerate_completions(
                    prefix_tree, X_bin, y, fill_depth)
            else:
                full_trees = [(prefix_tree, self._compute_objective(
                    prefix_tree, X_bin, y))]

            for full_tree, obj in full_trees:
                if not self._is_duplicate(full_tree, candidates):
                    candidates.append((full_tree, obj))

            if self.verbose and (i + 1) % 10 == 0:
                print(f"[ReSPLIT]   ... {i + 1}/{self.num_prefix_candidates} done")

        if not candidates:
            raise RuntimeError("No valid trees generated.")

        # Filter to Rashomon set (ε-threshold)
        best_obj = min(obj for _, obj in candidates)
        threshold = best_obj * (1.0 + self.rashomon_bound_multiplier)
        self.models = [
            (tree, obj) for tree, obj in candidates if obj <= threshold
        ]
        # Sort by objective (best first)
        self.models.sort(key=lambda x: x[1])

        if self.verbose:
            print(
                f"[ReSPLIT] Rashomon set: {len(self.models)} trees "
                f"(best objective: {best_obj:.6f}, threshold: {threshold:.6f})"
            )

        return self

    def __len__(self):
        return len(self.models)

    def __getitem__(self, idx):
        """Return the idx-th tree in the Rashomon set (sorted by quality)."""
        return self.models[idx][0]

    def predict(self, X, idx=0):
        """Predict using the idx-th tree in the Rashomon set."""
        if idx >= len(self.models):
            raise IndexError(
                f"Tree index {idx} out of range (have {len(self.models)} trees)."
            )
        tree = self.models[idx][0]

        if self.binarize_flag and self._enc is not None:
            X_bin = np.asarray(self._enc.transform(X), dtype=bool)
        else:
            X_bin = np.asarray(X, dtype=bool)

        classes = np.array(self.classes_)
        return np.array(
            [predict_sample(X_bin[i], tree, classes)
             for i in range(len(X_bin))]
        )

    def get_objective(self, idx=0):
        """Return the objective value of the idx-th tree."""
        return self.models[idx][1]

    # ------------------------------------------------------------------
    # Internal: binarization
    # ------------------------------------------------------------------

    def _binarize_fit_transform(self, X, y):
        if self.binarizer_type == "gbdt":
            self._enc = ThresholdGuessBinarizer(
                n_estimators=100,
                max_depth=1,
                random_state=self.random_state,
            )
        else:
            self._enc = NumericBinarizer(max_thresholds=self.max_thresholds)
        return np.asarray(self._enc.fit_transform(X, y), dtype=bool)

    # ------------------------------------------------------------------
    # Internal: greedy prefix (for diversity)
    # ------------------------------------------------------------------

    def _greedy_prefix(self, X, y, feature_order, depth):
        """Build a greedy prefix with a specific feature evaluation order."""
        reorder = np.array(feature_order, dtype=int)
        builder = GreedyTreeBuilder(
            depth_budget=depth,
            reg=self.reg,
            max_features=self.max_features,
            feature_order=reorder,
            pick_first=True,    # diversity: use shuffled order, not best-entropy
            global_N=self._n_total,
        )
        tree, loss = builder.build(X, y)
        return tree, loss

    # ------------------------------------------------------------------
    # Internal: leaf completion with subtree enumeration
    # ------------------------------------------------------------------

    def _enumerate_completions(self, prefix_tree, X, y, remaining_depth):
        """For each leaf, enumerate 1-2 near-optimal completions.

        Follows Algorithm 5: greedy sets upper bound Lg, then solver
        finds subtrees with loss ≤ Lg.
        """
        leaves_data = self._collect_leaves(prefix_tree, X, y)
        all_trees = []

        # For each leaf, generate subtree options
        leaf_options = []
        for leaf_mask, y_leaf in leaves_data:
            options = []
            n_local = len(y_leaf)
            if n_local == 0:
                options = [(SPLITLeaf(prediction=0, loss=0.0),)]
                leaf_options.append(options)
                continue

            # Algorithm 5 line 7: Greedy upper bound Lg
            # Use global-N normalisation with original λ (not scaled_reg).
            gbuilder = GreedyTreeBuilder(
                depth_budget=remaining_depth,
                reg=self.reg,
                max_features=self.max_features,
                global_N=self._n_total,
            )
            greedy_st, greedy_lb_sub = gbuilder.build(
                X[leaf_mask], y_leaf)
            greedy_global = self._compute_objective(
                greedy_st, X[leaf_mask], y_leaf)

            # Option 1: greedy completion
            options.append(greedy_st)

            # Algorithm 5 line 8: find subtrees with loss ≤ Lg
            # Use our solver with the greedy loss as upper bound
            if remaining_depth > 0:
                try:
                    sub_solver = OptimalTreeSolver(
                        depth_budget=remaining_depth,
                        reg=self.reg,
                        time_limit=self.time_limit,
                        max_features=self.max_features,
                        remaining_depth=0,
                        global_N=self._n_total,
                    )
                    opt_st, opt_loss = sub_solver.fit(
                        X[leaf_mask], y_leaf,
                        upper_bound_tree=(greedy_st, greedy_global),
                    )
                    # Only add if structurally different from greedy
                    if not self._trees_equal(opt_st, greedy_st):
                        options.append(opt_st)
                except Exception:
                    pass  # solver timed out, just use greedy

            leaf_options.append(options)

        # Cross-product: one tree per combination of leaf options
        # (prefer the first option = greedy for each leaf)
        for combo_idx in range(max(len(opts) for opts in leaf_options)):
            tree = self._build_combination(prefix_tree, X, y, leaf_options, combo_idx)
            if tree is not None:
                obj = self._compute_objective(tree, X, y)
                all_trees.append((tree, obj))

        return all_trees if all_trees else [(prefix_tree, self._compute_objective(prefix_tree, X, y))]

    def _collect_leaves(self, node, X, y):
        """Collect (mask, y) tuples for all leaf subproblems."""
        result = []
        self._collect_leaves_impl(node, X, y, np.ones(len(y), dtype=bool), result)
        return result

    def _collect_leaves_impl(self, node, X, y, mask, result):
        if isinstance(node, SPLITLeaf):
            result.append((mask, y[mask]))
        else:
            left_mask = mask & X[:, node.feature]
            right_mask = mask & ~X[:, node.feature]
            self._collect_leaves_impl(node.left_child, X, y, left_mask, result)
            self._collect_leaves_impl(node.right_child, X, y, right_mask, result)

    def _build_combination(self, node, X, y, leaf_options, idx):
        """Build a tree by selecting option idx (mod len) for each leaf."""
        opt_idx = [0]
        return self._build_combo_impl(node, X, y, leaf_options, idx, opt_idx)

    def _build_combo_impl(self, node, X, y, leaf_options, target_idx, counter):
        if isinstance(node, SPLITLeaf):
            options = leaf_options[counter[0]]
            chosen = options[target_idx % len(options)]
            counter[0] += 1
            return chosen
        else:
            left_mask = X[:, node.feature]
            right_mask = ~left_mask
            left_child = self._build_combo_impl(
                node.left_child, X[left_mask], y[left_mask],
                leaf_options, target_idx, counter)
            right_child = self._build_combo_impl(
                node.right_child, X[right_mask], y[right_mask],
                leaf_options, target_idx, counter)
            return SPLITNode(
                feature=node.feature,
                left_child=left_child,
                right_child=right_child,
            )

    # ------------------------------------------------------------------
    # Internal: evaluation & dedup
    # ------------------------------------------------------------------

    def _compute_objective(self, tree, X, y):
        """Compute regularized objective: misclass/N + λ·num_leaves."""
        preds = predict_batch(X, tree, np.array(self.classes_))
        n_err = int(np.sum(preds != y))
        return n_err / self._n_total + self.reg * num_leaves(tree)

    @staticmethod
    def _is_duplicate(tree, candidates):
        """Check if tree is structurally identical to any existing candidate."""
        for existing, _ in candidates:
            if ReSPLIT._trees_equal(tree, existing):
                return True
        return False

    @staticmethod
    def _trees_equal(t1, t2):
        from .utils.helpers import are_trees_equal
        return are_trees_equal(t1, t2)


# ============================================================
# LicketySPLIT — Polynomial-time recursive SPLIT
# ============================================================
# Algorithm 3 from the SPLIT paper:
#
#   t_lookahead = SPLIT(ℓ, D, λ, 1, d, 0)   // dl=1, no postprocessing
#   for each child u of t_lookahead:
#       λu = λ * |D| / |D(u)|
#       tu = LicketySPLIT(ℓ, D(u), λu, d-1)
#       Replace u with tu
#
# SPLIT with dl=1 evaluates the optimal root split by computing
# GreedyTree(D(f), d-1) + GreedyTree(D(¬f), d-1) for every feature f.
# The recursion then replaces each greedy child with the same logic.
#
# Runtime: O(n · k² · d²) (polynomial — Theorem 6.4 in paper)


class LicketySPLIT:
    """LicketySPLIT: polynomial-time recursive SPLIT (Algorithm 3).

    Parameters
    ----------
    full_depth_budget : int
        Total depth budget d for the final tree.
    reg : float
        Sparsity penalty λ per leaf.
    verbose : bool
        Print progress.
    binarize : bool
        Binarize continuous features.
    binarizer : str
        "gbdt" or "midpoint".
    max_features : int
        Top-K candidate features (0 = all).
    max_thresholds : int
        Max midpoints per numeric feature.
    gbdt_n_est : int
        Number of GBDT trees for threshold guessing.
    gbdt_max_depth : int
        Max depth of each GBDT tree.
    random_state : int
        Random seed.
    """

    def __init__(
        self,
        full_depth_budget=5,
        reg=0.0001,
        verbose=False,
        binarize=True,
        binarizer="gbdt",
        max_features=0,
        max_thresholds=50,
        gbdt_n_est=100,
        gbdt_max_depth=1,
        random_state=42,
    ):
        self.full_depth_budget = full_depth_budget
        self.reg = reg
        self.verbose = verbose
        self.binarize_flag = binarize
        self.binarizer_type = binarizer
        self.max_features = max_features
        self.max_thresholds = max_thresholds
        self.gbdt_n_est = gbdt_n_est
        self.gbdt_max_depth = gbdt_max_depth
        self.random_state = random_state

        self.classes_ = None
        self.tree = None
        self.feature_names_ = None
        self._enc = None
        self._n_total = 0
        self._has_no_depth_limit = full_depth_budget == 0

    # ------------------------------------------------------------------
    # fit / predict
    # ------------------------------------------------------------------

    def fit(self, X, y):
        """Train LicketySPLIT (Algorithm 3)."""
        y = np.asarray(y, dtype=np.int64)
        self.classes_ = np.unique(y).tolist()
        self._n_total = len(y)

        if self.binarize_flag:
            X_bin = self._binarize_fit_transform(X, y)
        else:
            X_bin = np.asarray(X, dtype=bool)
        self.feature_names_ = [f"f{i}" for i in range(X_bin.shape[1])]

        if self.verbose:
            print(f"[LicketySPLIT] {X_bin.shape[1]} binarized features, "
                  f"depth={self.full_depth_budget}, reg={self.reg}")

        self.tree, _ = self._recurse(X_bin, y, self.full_depth_budget)
        return self

    def predict(self, X):
        """Predict class labels."""
        if self.tree is None:
            raise RuntimeError("Model not fitted yet.")
        if self.binarize_flag and self._enc is not None:
            X_bin = np.asarray(self._enc.transform(X), dtype=bool)
        else:
            X_bin = np.asarray(X, dtype=bool)
        classes = np.array(self.classes_)
        return np.array(
            [predict_sample(X_bin[i], self.tree, classes)
             for i in range(len(X_bin))]
        )

    def num_leaves(self):
        return num_leaves(self.tree) if self.tree is not None else 0

    def __str__(self):
        if self.tree is None:
            return "LicketySPLIT(unfitted)"
        return str(self.tree)

    # ------------------------------------------------------------------
    # Core recursion — Algorithm 3
    #
    # Equivalent to SPLIT(dl=1, postprocess=False):
    #   For every feature f, evaluate Greedy(D(f), depth-1, λu)
    #   + Greedy(D(¬f), depth-1, λu) using renormalised λ.
    #   Pick the best feature, then recursively apply LicketySPLIT
    #   on each child (Algorithm 3, lines 2-8).
    # ------------------------------------------------------------------

    def _recurse(self, X, y, depth):
        """Recursive LicketySPLIT step.

        Parameters
        ----------
        X : ndarray[bool]
            Binary feature matrix for this subproblem.
        y : ndarray[int64]
            Labels for this subproblem.
        depth : int
            Remaining depth budget.

        Returns
        -------
        (tree, loss) — loss is on global scale (normalised by
        self._n_total).
        """
        n = len(y)
        if n == 0:
            leaf = SPLITLeaf(prediction=0, loss=0.0)
            return leaf, 0.0

        # --- Leaf baseline (on global scale) ---
        classes, counts = np.unique(y, return_counts=True)
        y_pred = int(classes[np.argmax(counts)])
        n_wrong = int(n - counts.max())
        leaf_loss = n_wrong / self._n_total + self.reg

        if depth < 1 or n < 2:
            leaf = SPLITLeaf(prediction=y_pred, loss=n_wrong / self._n_total)
            return leaf, leaf_loss

        # --- Evaluate each feature by greedy completion (SPLIT dl=1) ---
        best_loss = leaf_loss
        best_feat = -1

        # Feature ranking by entropy for candidate pruning
        candidates = self._rank_features(X, y)
        if self.verbose:
            print(f"  [LicketySPLIT] depth={depth}, n={n}, "
                  f"λ={self.reg:.6f}, leaf={leaf_loss:.6f}, "
                  f"candidates={len(candidates)}")

        # Child depth budget for greedy evaluation
        child_depth = max(depth - 1, 0)

        for feat in candidates:
            left_mask = X[:, feat]
            right_mask = ~left_mask
            nl = left_mask.sum()
            nr = n - nl
            if nl == 0 or nr == 0:
                continue

            # Greedy evaluation of full subtree for each child
            # using global-N normalisation with original λ.
            lb = GreedyTreeBuilder(
                depth_budget=child_depth,
                reg=self.reg,
                max_features=self.max_features,
                global_N=self._n_total,
            )
            left_tree, left_loss = lb.build(X[left_mask], y[left_mask])

            rb = GreedyTreeBuilder(
                depth_budget=child_depth,
                reg=self.reg,
                max_features=self.max_features,
                global_N=self._n_total,
            )
            right_tree, right_loss = rb.build(X[right_mask], y[right_mask])

            # Losses are already on global scale (global-N normalised)
            total = left_loss + right_loss
            if total < best_loss:
                best_loss = total
                best_feat = feat

        if best_feat >= 0:
            # --- Split and recurse (Algorithm 3 lines 2-7) ---
            left_mask = X[:, best_feat]
            right_mask = ~left_mask

            # Algorithm 3 line 5: recursive LicketySPLIT on children
            left_child, _ = self._recurse(
                X[left_mask], y[left_mask], depth - 1)
            right_child, _ = self._recurse(
                X[right_mask], y[right_mask], depth - 1)

            if self.verbose:
                print(f"    → split feat={best_feat}, "
                      f"loss={best_loss:.6f} (leaf={leaf_loss:.6f})")
            return (
                SPLITNode(feature=int(best_feat),
                          left_child=left_child,
                          right_child=right_child),
                best_loss,
            )

        # No split improves leaf → return leaf
        return (
            SPLITLeaf(prediction=y_pred, loss=n_wrong / self._n_total),
            leaf_loss,
        )

    @staticmethod
    def _count_leaves(node):
        if isinstance(node, SPLITLeaf):
            return 1
        return (LicketySPLIT._count_leaves(node.left_child)
                + LicketySPLIT._count_leaves(node.right_child))

    # ------------------------------------------------------------------
    # Feature ranking (entropy gain)
    # ------------------------------------------------------------------

    def _rank_features(self, X, y):
        m = X.shape[1]
        if self.max_features <= 0 or self.max_features >= m:
            return np.arange(m, dtype=int)

        n = len(y)
        _, counts = np.unique(y, return_counts=True)
        p = counts.astype(np.float64) / n
        p = p[p > 0]
        ent_parent = -np.sum(p * np.log2(p)) if len(p) > 1 else 0.0

        gains = np.zeros(m, dtype=np.float64)
        for feat in range(m):
            left = X[:, feat]
            nl = left.sum()
            nr = n - nl
            if nl == 0 or nr == 0:
                gains[feat] = -1.0
                continue
            ent_l = self._entropy(y[left])
            ent_r = self._entropy(y[~left])
            gains[feat] = ent_parent - ((nl / n) * ent_l + (nr / n) * ent_r)

        order = np.argsort(-gains, kind="mergesort")
        return order[:self.max_features].astype(int)

    @staticmethod
    def _entropy(labels):
        _, cnt = np.unique(labels, return_counts=True)
        cnt = cnt.astype(np.float64)
        cnt = cnt[cnt > 0]
        p = cnt / cnt.sum()
        if len(p) <= 1:
            return 0.0
        return float(-np.sum(p * np.log2(p)))

    # ------------------------------------------------------------------
    # Binarization
    # ------------------------------------------------------------------

    def _binarize_fit_transform(self, X, y):
        if self.binarizer_type == "gbdt":
            self._enc = ThresholdGuessBinarizer(
                n_estimators=self.gbdt_n_est,
                max_depth=self.gbdt_max_depth,
                random_state=self.random_state,
            )
        else:
            self._enc = NumericBinarizer(max_thresholds=self.max_thresholds)
        return np.asarray(self._enc.fit_transform(X, y), dtype=bool)
