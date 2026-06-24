"""Optimal decision tree solver."""

import time
import numpy as np

from .builder import GreedyTreeBuilder
from .utils.nodes import SPLITLeaf, SPLITNode


class OptimalTreeSolver:
    """Optimal tree solver with memoization."""

    def __init__(self, depth_budget=3, reg=0.0001, time_limit=60,
                 max_features=0, remaining_depth=0, global_N=None):
        if depth_budget < 0:
            raise ValueError("depth_budget must be non-negative")
        self.depth_budget = depth_budget
        self.reg = reg
        self.time_limit = time_limit
        self.max_features = max_features
        self.remaining_depth = remaining_depth
        self._global_N = global_N

        self._memo = {}
        self._n_global = 0
        self._start_time = None
        self._timed_out = False
        self._feature_order = None

    def fit(self, X, y, upper_bound_tree=None):
        """Find the optimal tree under the regularized objective."""
        X = np.asarray(X, dtype=bool)
        y = np.asarray(y, dtype=np.int64)

        if self._global_N is not None:
            self._n_global = self._global_N
        else:
            self._n_global = len(y)

        self._memo = {}
        self._start_time = time.perf_counter()
        self._timed_out = False
        self._global_classes = np.unique(y)

        if upper_bound_tree is not None:
            _, ub_loss = upper_bound_tree
            self._upper_bound = ub_loss
        else:
            self._upper_bound = np.inf

        self._feature_order = self._rank_features(X, y)

        fo = self._feature_order
        top_n = min(5, len(fo))
        top_info = [f"{fo[i]}(nl={X[:, fo[i]].sum()})" for i in range(top_n)]
        print(f"[Solver] {X.shape[1]} feats, top-{len(fo)} "
              f"candidates, N={self._n_global}, "
              f"leaf={self._leaf_loss(y):.6f}, "
              f"ub={self._upper_bound:.6f}")
        print(f"[Solver]   top: {', '.join(top_info)}")

        tree, loss = self._build(X, y, self.depth_budget)
        return tree, loss

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

    def _build(self, X, y, depth):
        """Recursive optimal tree search."""
        n = len(y)
        if n == 0:
            return SPLITLeaf(prediction=0, loss=0.0), 0.0

        if depth == 0 and self.remaining_depth > 0:
            return self._greedy_completion(X, y)

        if (self._start_time is not None
                and time.perf_counter() - self._start_time > self.time_limit):
            self._timed_out = True
            leaf = self._make_leaf(y)
            return leaf, self._leaf_loss(y)

        leaf = self._make_leaf(y)
        leaf_loss = self._leaf_loss(y)
        if depth > 0 and n >= 2:
            memo_key = self._memo_key(y, depth)
            if memo_key in self._memo:
                return self._memo[memo_key]

            best_node = leaf
            best_loss = leaf_loss
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
                    if best_loss <= self._upper_bound:
                        break

            self._memo[memo_key] = (best_node, best_loss)
            return best_node, best_loss

        return leaf, leaf_loss

    def _greedy_completion(self, X, y):
        """Return a greedy subtree at the boundary."""
        builder = GreedyTreeBuilder(
            depth_budget=self.remaining_depth,
            reg=self.reg,
            max_features=self.max_features,
            global_N=self._n_global,
        )
        tree, _loss_sub = builder.build(X, y)
        loss_global = self._compute_tree_loss(tree, X, y)
        return tree, loss_global

    def _compute_tree_loss(self, tree, X, y):
        """Compute the regularized loss using global normalization."""
        from .utils.helpers import predict_batch
        classes = self._global_classes
        if len(classes) == 0:
            return 0.0
        preds = predict_batch(X, tree, classes)
        n_err = int(np.sum(preds != y))
        n_leaves = self._count_leaves(tree)
        return n_err / self._n_global + self.reg * n_leaves

    @staticmethod
    def _count_leaves(node):
        if isinstance(node, SPLITLeaf):
            return 1
        return (OptimalTreeSolver._count_leaves(node.left_child)
                + OptimalTreeSolver._count_leaves(node.right_child))

    def _make_leaf(self, y):
        """Create a leaf node predicting the majority class."""
        classes, counts = np.unique(y, return_counts=True)
        y_pred = int(classes[np.argmax(counts)])
        n_wrong = int(len(y) - counts.max())
        raw_loss = n_wrong / self._n_global
        return SPLITLeaf(prediction=y_pred, loss=raw_loss)

    def _leaf_loss(self, y):
        """Leaf loss on a subproblem."""
        classes, counts = np.unique(y, return_counts=True)
        n_wrong = int(len(y) - counts.max())
        return n_wrong / self._n_global + self.reg

    def _candidates_for_level(self, depth):
        fo = self._feature_order
        if fo is None or len(fo) == 0:
            return np.array([], dtype=int)
        if depth >= self.depth_budget:
            return fo
        fraction = depth / max(self.depth_budget, 1)
        k = max(5, int(len(fo) * fraction))
        return fo[:k]

    @staticmethod
    def _memo_key(y, depth):
        """Build a memoization key from y and depth."""
        return hash((y.tobytes(), depth))
