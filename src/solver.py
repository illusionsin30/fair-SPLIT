"""Pure-Python optimal decision tree solver using DP + memoization.

Replaces the C++ GOSDT solver for the lookahead phase of SPLIT.
Supports binary AND multiclass classification with pre-binarized (0/1)
features.

Objective:  minimize  (misclassification_loss + λ * num_leaves)

Splits are unconditional if depth > 0 (like CART): the best entropy-gain
feature is always used.  The λ penalty is accumulated in the returned
loss value but does NOT gate splits — this avoids deadlock on imbalanced
datasets where no single binary feature can reduce 0-1 misclassification.

Speed optimisations:
  - Top-K feature pre-filtering by entropy gain.
  - Greedy upper bound for early stopping once bound is matched.
  - Level-dependent feature budget.
"""

import time
import numpy as np

from .utils.nodes import SPLITLeaf, SPLITNode


class OptimalTreeSolver:
    """DP optimal tree solver with unconditional splits.

    Args:
        depth_budget: Maximum tree depth.
        reg: Regularization penalty per leaf (added to loss).
        time_limit: Maximum wall-clock time in seconds.
        max_features: Top-K candidate features (0 = all).
    """

    def __init__(self, depth_budget=3, reg=0.0001, time_limit=60,
                 max_features=25, remaining_depth=0):
        if depth_budget < 0:
            raise ValueError("depth_budget must be non-negative")
        self.depth_budget = depth_budget
        self.reg = reg
        self.time_limit = time_limit
        self.max_features = max_features
        self.remaining_depth = remaining_depth  # for greedy completion at boundary

        self._memo = {}
        self._n_total = 0
        self._start_time = None
        self._timed_out = False
        self._upper_bound = np.inf
        self._feature_order = None
        self._greedy_builder = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X, y, upper_bound_tree=None):
        """Find an optimal tree.

        Args:
            X: (n_samples, n_features) binary ndarray.
            y: (n_samples,) integer class labels.
            upper_bound_tree: Optional (tree, loss) greedy upper bound.

        Returns:
            (tree, loss).
        """
        X = np.asarray(X, dtype=bool)
        y = np.asarray(y, dtype=np.int64)
        self._n_total = len(y)
        self._memo = {}
        self._start_time = time.perf_counter()
        self._timed_out = False

        if upper_bound_tree is not None:
            _, greedy_loss = upper_bound_tree
            self._upper_bound = greedy_loss
        else:
            self._upper_bound = np.inf

        self._feature_order = self._rank_features(X, y)

        # Diagnostic
        fo = self._feature_order if self._feature_order is not None else np.array([], dtype=int)
        top_n = min(5, len(fo))
        top_info = [f"{fo[i]}(nl={X[:, fo[i]].sum()})" for i in range(top_n)]
        print(f"[Solver] {X.shape[1]} feats, top-{len(fo)} "
              f"candidates, leaf={self._leaf_loss(y):.6f}, "
              f"ub={self._upper_bound:.6f}")
        print(f"[Solver]   top: {', '.join(top_info)}")

        tree, loss = self._build(X, y, self.depth_budget)
        return tree, loss

    def clear_memo(self):
        self._memo = {}

    # ------------------------------------------------------------------
    # Feature ranking (entropy gain)
    # ------------------------------------------------------------------

    def _rank_features(self, X, y):
        """Sort features by decreasing entropy gain."""
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
            right = ~left
            nl = left.sum()
            nr = n - nl
            if nl == 0 or nr == 0:
                gains[feat] = -1.0
                continue
            ent_l = self._entropy(y[left])
            ent_r = self._entropy(y[right])
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

    def _candidates_for_level(self, depth):
        fo = self._feature_order
        if fo is None or len(fo) == 0:
            return np.array([], dtype=int)
        if depth >= self.depth_budget:
            return fo
        fraction = depth / max(self.depth_budget, 1)
        k = max(5, int(len(fo) * fraction))
        return fo[:k]

    # ------------------------------------------------------------------
    # DP core  (unconditional splits, λ in loss only)
    # ------------------------------------------------------------------

    def _build(self, X, y, depth):
        """Recursively build the optimal tree.

        At depth>0: exhaustive split search (optimal).
        At depth==0 (lookahead boundary): return GREEDY completion
          for self.remaining_depth — Algorithm 1 of the paper.
        """
        n = len(y)
        if n == 0:
            return SPLITLeaf(prediction=0, loss=0.0), 0.0

        # --- lookahead boundary: greedy completion (paper Eq.4 branch ②) ---
        if depth == 0 and self.remaining_depth > 0:
            from .builder import GreedyTreeBuilder
            if self._greedy_builder is None:
                self._greedy_builder = GreedyTreeBuilder(
                    depth_budget=self.remaining_depth,
                    reg=self.reg,
                    max_features=self.max_features,
                )
            tree, loss = self._greedy_builder.build(X, y, n_total=self._n_total)
            return tree, loss

        # time-limit check
        if (self._start_time is not None
                and time.perf_counter() - self._start_time > self.time_limit):
            self._timed_out = True
            return self._make_leaf(y), self._leaf_loss(y)

        # leaf baseline
        leaf = self._make_leaf(y)
        leaf_loss = self._leaf_loss(y)

        # pruning: leaf already worse than known upper bound
        if leaf_loss > self._upper_bound:
            return leaf, leaf_loss

        best_node = leaf
        best_loss = leaf_loss

        # try splits if depth budget remains
        if depth > 0 and n >= 2:
            key = self._memo_key(X, y, depth)
            if key in self._memo:
                return self._memo[key]

            candidates = self._candidates_for_level(depth)

            for feat in candidates:
                left_mask = X[:, feat]
                right_mask = ~left_mask
                nl = left_mask.sum()
                nr = n - nl
                if nl == 0 or nr == 0:
                    continue

                left_child, left_loss = self._build(
                    X[left_mask], y[left_mask], depth - 1)
                if self._timed_out:
                    break
                right_child, right_loss = self._build(
                    X[right_mask], y[right_mask], depth - 1)
                if self._timed_out:
                    break

                total_loss = left_loss + right_loss
                if total_loss < best_loss:
                    best_loss = total_loss
                    best_node = SPLITNode(
                        feature=int(feat),
                        left_child=left_child,
                        right_child=right_child,
                    )
                    # Early exit: matched/beaten the greedy upper bound
                    if best_loss <= self._upper_bound:
                        break

            self._memo[key] = (best_node, best_loss)

        return best_node, best_loss

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_leaf(self, y):
        classes, counts = np.unique(y, return_counts=True)
        y_pred = int(classes[np.argmax(counts)])
        raw_loss = np.sum(y != y_pred) / self._n_total
        return SPLITLeaf(prediction=y_pred, loss=raw_loss)

    def _leaf_loss(self, y):
        classes, counts = np.unique(y, return_counts=True)
        y_pred = int(classes[np.argmax(counts)])
        return np.sum(y != y_pred) / self._n_total + self.reg

    @staticmethod
    def _memo_key(X, y, depth):
        return hash((X.tobytes(), y.tobytes(), depth))
