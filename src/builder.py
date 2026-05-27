"""Greedy tree builder using entropy-based information gain.

Implements the paper's sparse-greedy Algorithm 4: splits are chosen by
entropy gain and unconditionally accepted while depth budget remains.
The λ penalty is added to the leaf loss to compute the regularized
objective, but does NOT block splits — this avoids the pathological
case where no single binary feature reduces 0-1 misclassification
(e.g. imbalanced datasets like Adult 76/24).

The λ penalty is correctly accumulated in the returned loss value,
so the SPLIT objective (Equation 1) is still accurately evaluated.
"""

import numpy as np

from .utils.nodes import SPLITLeaf, SPLITNode


class GreedyTreeBuilder:
    """Top-down greedy tree builder.

    Args:
        depth_budget: Maximum tree depth (0 = leaf only).
        reg: Regularization penalty per leaf (added to loss, does NOT
             gate split acceptance).
        max_features: Top-K candidate features at the root.
            0 = unlimited.
    """

    def __init__(self, depth_budget=3, reg=0.0001, max_features=25,
                 feature_order=None, pick_first=False):
        self.depth_budget = depth_budget
        self.reg = reg
        self.max_features = max_features
        self._n_total = 0
        self._feature_order = feature_order  # pre-set order (for ReSPLIT diversity)
        self._pick_first = pick_first  # pick first with gain>0 instead of best

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, X, y, n_total=None):
        """Build a greedy tree.

        Args:
            n_total: Global N for loss normalization.  If None, uses len(y).

        Returns:
            (tree, loss) where loss = misclass_rate + λ·num_leaves.
        """
        X = np.asarray(X, dtype=bool)
        y = np.asarray(y, dtype=np.int64)
        self._n_total = n_total if n_total is not None else len(y)
        if self._feature_order is None:
            self._feature_order = self._rank_features(X, y)
        return self._train(X, y, self.depth_budget, self.depth_budget)

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
            gains[feat] = ent_parent - (
                (nl / n) * ent_l + (nr / n) * ent_r
            )
        order = np.argsort(-gains, kind="mergesort")
        return order[:self.max_features].astype(int)

    def _candidates_for_level(self, depth):
        if self._feature_order is None:
            return range(0)
        if depth >= self.depth_budget:
            return self._feature_order
        fraction = depth / max(self.depth_budget, 1)
        k = max(5, int(len(self._feature_order) * fraction))
        return self._feature_order[:k]

    # ------------------------------------------------------------------
    # Recursive training
    # ------------------------------------------------------------------

    def _train(self, X, y, depth, max_depth_orig):
        """Recursively build a greedy subtree.

        Splits are always accepted if depth > 0 and a feature with
        positive entropy gain exists (CART-like).  The λ penalty is
        added to leaf losses so the returned value is the correct
        regularized objective.
        """
        n = len(y)

        # Leaf prediction
        if n > 0:
            classes, counts = np.unique(y, return_counts=True)
            y_pred = int(classes[np.argmax(counts)])
        else:
            y_pred = 0

        # Try split
        if depth > 0 and n >= 2:
            best_feat, best_gain = self._best_feature(X, y, depth)
            if best_gain > 0 and best_feat >= 0:
                left_mask = X[:, best_feat]
                right_mask = ~left_mask
                X_left, y_left = X[left_mask], y[left_mask]
                X_right, y_right = X[right_mask], y[right_mask]

                if len(y_left) > 0 and len(y_right) > 0:
                    left_child, left_loss = self._train(
                        X_left, y_left, depth - 1, max_depth_orig)
                    right_child, right_loss = self._train(
                        X_right, y_right, depth - 1, max_depth_orig)
                    return (
                        SPLITNode(feature=best_feat,
                                  left_child=left_child,
                                  right_child=right_child),
                        left_loss + right_loss,
                    )

        # Leaf
        leaf_loss = (np.sum(y != y_pred) / self._n_total
                     if n > 0 else 0.0) + self.reg
        return SPLITLeaf(prediction=y_pred, loss=leaf_loss - self.reg), leaf_loss

    # ------------------------------------------------------------------
    # Split selection (entropy gain)
    # ------------------------------------------------------------------

    def _best_feature(self, X, y, depth):
        """Return (feature_index, entropy_gain)."""
        n = len(y)
        if n == 0:
            return -1, -1.0

        _, counts = np.unique(y, return_counts=True)
        p = counts.astype(np.float64) / n
        p = p[p > 0]
        ent_parent = -np.sum(p * np.log2(p)) if len(p) > 1 else 0.0

        best_gain = -1.0
        best_feat = -1
        candidates = self._candidates_for_level(depth)

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
                return feat, gain
            if gain > best_gain:
                best_gain = gain
                best_feat = feat

        return best_feat, best_gain

    @staticmethod
    def _entropy(labels):
        """Multi-class entropy."""
        _, cnt = np.unique(labels, return_counts=True)
        cnt = cnt.astype(np.float64)
        cnt = cnt[cnt > 0]
        p = cnt / cnt.sum()
        if len(p) <= 1:
            return 0.0
        return float(-np.sum(p * np.log2(p)))
