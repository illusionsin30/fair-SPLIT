"""Greedy tree builder."""

import numpy as np

from .utils.nodes import SPLITLeaf, SPLITNode


class GreedyTreeBuilder:
    """Top-down greedy tree builder."""

    def __init__(self, depth_budget=3, reg=0.0001, max_features=0,
                 feature_order=None, pick_first=False, global_N=None):
        """Initialize the builder."""
        self.depth_budget = depth_budget
        self.reg = reg
        self.max_features = max_features
        self._feature_order = feature_order
        self._pick_first = pick_first
        self._global_N = global_N

    def build(self, X, y):
        """Build a greedy tree."""
        X = np.asarray(X, dtype=bool)
        y = np.asarray(y, dtype=np.int64)
        self._n_norm = self._global_N if self._global_N is not None else len(y)
        if self._feature_order is None:
            self._feature_order = self._rank_features(X, y)
        return self._greedy_impl(X, y, self.depth_budget)

    def _greedy_impl(self, X, y, depth):
        """Recursive greedy tree induction."""
        n = len(y)

        classes, counts = np.unique(y, return_counts=True)
        y_pred = int(classes[np.argmax(counts)])
        n_wrong = int(n - counts.max())
        lb = self.reg + (n_wrong / self._n_norm if self._n_norm > 0 else 0.0)
        if depth > 1 and n >= 2:
            best_feat = self._best_feature(X, y, depth)
            if best_feat >= 0:
                left_mask = X[:, best_feat]
                right_mask = ~left_mask
                nl = left_mask.sum()
                nr = n - nl

                if nl > 0 and nr > 0:
                    tleft, lbleft = self._greedy_impl(
                        X[left_mask], y[left_mask], depth - 1)
                    tright, lbright = self._greedy_impl(
                        X[right_mask], y[right_mask], depth - 1)

                    if lbleft + lbright < lb:
                        return (
                            SPLITNode(
                                feature=int(best_feat),
                                left_child=tleft,
                                right_child=tright
                            ),
                            lbleft + lbright,
                        )

        leaf_loss_without_reg = (n_wrong / self._n_norm) if self._n_norm > 0 else 0.0
        return (
            SPLITLeaf(prediction=y_pred, loss=leaf_loss_without_reg),
            lb,
        )

    def _best_feature(self, X, y, depth):
        """Return the index of the feature with the best information gain."""
        n = len(y)
        if n == 0:
            return -1

        _, counts = np.unique(y, return_counts=True)
        p = counts.astype(np.float64) / n
        p = p[p > 0]
        ent_parent = -np.sum(p * np.log2(p)) if len(p) > 1 else 0.0

        candidates = self._candidates_for_level(depth)
        best_gain = -1.0
        best_feat = -1

        for feat in candidates:
            left_mask = X[:, feat]
            right_mask = ~left_mask
            nl = left_mask.sum()
            nr = n - nl
            if nl == 0 or nr == 0:
                continue

            ent_l = self._entropy(y[left_mask])
            ent_r = self._entropy(y[right_mask])
            gain = ent_parent - ((nl / n) * ent_l + (nr / n) * ent_r)

            if self._pick_first and gain > 0:
                return int(feat)
            if gain > best_gain:
                best_gain = gain
                best_feat = int(feat)

        return best_feat

    def _rank_features(self, X, y):
        """
        Sort features by decreasing entropy gain on the full dataset.
        """
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

    def _candidates_for_level(self, depth):
        if self._feature_order is None or len(self._feature_order) == 0:
            return np.array([], dtype=int)
        if depth >= self.depth_budget:
            return self._feature_order
        fraction = depth / max(self.depth_budget, 1)
        k = max(5, int(len(self._feature_order) * fraction))
        return self._feature_order[:k]

    @staticmethod
    def _entropy(labels):
        _, cnt = np.unique(labels, return_counts=True)
        cnt = cnt.astype(np.float64)
        cnt = cnt[cnt > 0]
        p = cnt / cnt.sum()
        if len(p) <= 1:
            return 0.0
        return float(-np.sum(p * np.log2(p)))
