"""CART and SPLIT-family tree implementations."""

from itertools import combinations

import numpy as np
import pandas as pd


class Node:
    """Tree node."""

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
        self.prediction = None


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
        self.max_cat_brute_force = max_cat_brute_force
        self.root = None
        self._classes = None

    def _impurity(self, y):
        """Gini for classification, variance for regression."""
        if len(y) == 0:
            return 0.0
        if self.task == "classification":
            _, counts = np.unique(y, return_counts=True)
            p = counts / counts.sum()
            return 1.0 - np.sum(p * p)
        return float(np.var(y))

    def _weighted_impurity(self, y_left, y_right):
        n = len(y_left) + len(y_right)
        return (len(y_left) / n) * self._impurity(y_left) + (
            len(y_right) / n
        ) * self._impurity(y_right)

    def _leaf_value(self, y):
        if self.task == "classification":
            vals, counts = np.unique(y, return_counts=True)
            if len(counts) == 0:
                return 0
            return vals[np.argmax(counts)]
        return float(np.mean(y))

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

    def _ordered_categories(self, values, y, categories):
        """Order categories by target mean."""
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

        if is_binary_clf or is_regression:
            sorted_cats = self._ordered_categories(values, y, categories)
            candidate_left_sets = [
                set(sorted_cats[:k]) for k in range(1, len(sorted_cats))
            ]
        else:
            K = len(categories)
            if K > self.max_cat_brute_force:
                return 0.0, None
            cat_list = list(categories)
            candidate_left_sets = []
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

    def _build(self, X, y, depth):
        node = Node()
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


def CARTClassifier(**kw):
    return CART(task="classification", **kw)


from .solver import OptimalTreeSolver
from .builder import GreedyTreeBuilder
from .utils.nodes import SPLITLeaf, SPLITNode
from .utils.helpers import predict_sample, predict_batch, tree_to_dict, num_leaves
from .utils.binarizer import NumericBinarizer, ThresholdGuessBinarizer


class SPLIT:
    """SPLIT tree builder."""

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
                "Use a positive lookahead depth."
            )

        self.lookahead_depth = lookahead_depth
        self.full_depth_budget = full_depth_budget
        self.reg = reg
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

        self._has_no_depth_limit = full_depth_budget == 0
        if self._has_no_depth_limit:
            self._remaining_depth = 0
        else:
            self._remaining_depth = full_depth_budget - lookahead_depth

        self.classes_ = None
        self.tree = None
        self._enc = None
        self._n_total = 0

    def fit(self, X, y):
        """Train SPLIT."""
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

        prefix_solver = OptimalTreeSolver(
            depth_budget=self.lookahead_depth,
            reg=self.reg,
            time_limit=self.time_limit,
            max_features=self.max_features,
            remaining_depth=self._remaining_depth,
            global_N=self._n_total,
        )

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

    def _fill_leaves(self, node, X, y, current_depth):
        """Fill leaf subtrees."""
        if isinstance(node, SPLITLeaf):
            n_local = len(y)
            if n_local == 0:
                return node

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
                builder = GreedyTreeBuilder(
                    depth_budget=remaining_depth,
                    reg=self.reg,
                    max_features=self.max_features,
                    global_N=self._n_total,
                )
                subtree, _ = builder.build(X, y)
                return subtree
        else:
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
        """Remap leaf predictions."""
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
        """Compute regularized loss."""
        preds = predict_batch(X, tree, np.array(self.classes_))
        n_err = int(np.sum(preds != y))
        return n_err / self._n_total + self.reg * num_leaves(tree)


class ReSPLIT:
    """Maintain a candidate tree set."""

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
        self.models = []
        self._enc = None
        self._n_total = 0
        self._rng = np.random.RandomState(random_state)

    def fit(self, X, y):
        """Build the candidate tree set."""
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

        for i in range(self.num_prefix_candidates):
            seed = self._rng.randint(0, 2**31 - 1)
            local_rng = np.random.RandomState(seed)

            shuffled_features = all_features.copy()
            local_rng.shuffle(shuffled_features)

            prefix_tree, _ = self._greedy_prefix(
                X_bin, y, shuffled_features, self.lookahead_depth
            )

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

        best_obj = min(obj for _, obj in candidates)
        threshold = best_obj * (1.0 + self.rashomon_bound_multiplier)
        self.models = [
            (tree, obj) for tree, obj in candidates if obj <= threshold
        ]
        self.models.sort(key=lambda x: x[1])

        if self.verbose:
            print(
                f"[ReSPLIT] candidate tree set: {len(self.models)} trees "
                f"(best objective: {best_obj:.6f}, threshold: {threshold:.6f})"
            )

        return self

    def __len__(self):
        return len(self.models)

    def __getitem__(self, idx):
        """Return the idx-th tree."""
        return self.models[idx][0]

    def predict(self, X, idx=0):
        """Predict using the idx-th tree."""
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

    def _greedy_prefix(self, X, y, feature_order, depth):
        """Build a greedy prefix."""
        reorder = np.array(feature_order, dtype=int)
        builder = GreedyTreeBuilder(
            depth_budget=depth,
            reg=self.reg,
            max_features=self.max_features,
            feature_order=reorder,
            pick_first=True,
            global_N=self._n_total,
        )
        tree, loss = builder.build(X, y)
        return tree, loss

    def _enumerate_completions(self, prefix_tree, X, y, remaining_depth):
        """Enumerate leaf completions."""
        leaves_data = self._collect_leaves(prefix_tree, X, y)
        all_trees = []

        leaf_options = []
        for leaf_mask, y_leaf in leaves_data:
            options = []
            n_local = len(y_leaf)
            if n_local == 0:
                options = [(SPLITLeaf(prediction=0, loss=0.0),)]
                leaf_options.append(options)
                continue

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

            options.append(greedy_st)

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
                    if not self._trees_equal(opt_st, greedy_st):
                        options.append(opt_st)
                except Exception:
                    pass

            leaf_options.append(options)

        for combo_idx in range(max(len(opts) for opts in leaf_options)):
            tree = self._build_combination(prefix_tree, X, y, leaf_options, combo_idx)
            if tree is not None:
                obj = self._compute_objective(tree, X, y)
                all_trees.append((tree, obj))

        return all_trees if all_trees else [(prefix_tree, self._compute_objective(prefix_tree, X, y))]

    def _collect_leaves(self, node, X, y):
        """Collect leaf subproblems."""
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
        """Build a tree from leaf options."""
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

    def _compute_objective(self, tree, X, y):
        """Compute regularized objective."""
        preds = predict_batch(X, tree, np.array(self.classes_))
        n_err = int(np.sum(preds != y))
        return n_err / self._n_total + self.reg * num_leaves(tree)

    @staticmethod
    def _is_duplicate(tree, candidates):
        """Check for structural duplicates."""
        for existing, _ in candidates:
            if ReSPLIT._trees_equal(tree, existing):
                return True
        return False

    @staticmethod
    def _trees_equal(t1, t2):
        from .utils.helpers import are_trees_equal
        return are_trees_equal(t1, t2)


class LicketySPLIT:
    """LicketySPLIT tree builder."""

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

    def fit(self, X, y):
        """Train LicketySPLIT."""
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

    def _recurse(self, X, y, depth):
        """Recursive LicketySPLIT step."""
        n = len(y)
        if n == 0:
            leaf = SPLITLeaf(prediction=0, loss=0.0)
            return leaf, 0.0

        classes, counts = np.unique(y, return_counts=True)
        y_pred = int(classes[np.argmax(counts)])
        n_wrong = int(n - counts.max())
        leaf_loss = n_wrong / self._n_total + self.reg

        if depth < 1 or n < 2:
            leaf = SPLITLeaf(prediction=y_pred, loss=n_wrong / self._n_total)
            return leaf, leaf_loss

        best_loss = leaf_loss
        best_feat = -1

        candidates = self._rank_features(X, y)
        if self.verbose:
            print(f"  [LicketySPLIT] depth={depth}, n={n}, "
                  f"reg={self.reg:.6f}, leaf={leaf_loss:.6f}, "
                  f"candidates={len(candidates)}")

        child_depth = max(depth - 1, 0)

        for feat in candidates:
            left_mask = X[:, feat]
            right_mask = ~left_mask
            nl = left_mask.sum()
            nr = n - nl
            if nl == 0 or nr == 0:
                continue

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

            total = left_loss + right_loss
            if total < best_loss:
                best_loss = total
                best_feat = feat

        if best_feat >= 0:
            left_mask = X[:, best_feat]
            right_mask = ~left_mask

            left_child, _ = self._recurse(
                X[left_mask], y[left_mask], depth - 1)
            right_child, _ = self._recurse(
                X[right_mask], y[right_mask], depth - 1)

            if self.verbose:
                print(f"    split feat={best_feat}, "
                      f"loss={best_loss:.6f} (leaf={leaf_loss:.6f})")
            return (
                SPLITNode(feature=int(best_feat),
                          left_child=left_child,
                          right_child=right_child),
                best_loss,
            )

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
