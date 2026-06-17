"""Sparse Greedy tree builder — exactly Algorithm 4 from the SPLIT paper.

Algorithm 4 Greedy(D, d, λ) → (tgreedy, lb):
  1: tgreedy ← (Leaf predicting the majority label in D)
  2: lb ← λ + (proportion of D that does not have the majority label)
  3: if d > 1 then
  4:    let f be the information gain maximizing split with respect to D
  5:    tleft, lbleft ← Greedy(D(f), d − 1, λ)
  6:    tright, lbright ← Greedy(D(f̄), d − 1, λ)
  7:    if lbleft + lbright < lb then
  8:        lb ← lbleft + lbright
  9:        tgreedy ← tree corresponding to: if f is True then tleft, else tright
 10:    end if
 11: end if
 12: return tgreedy, lb

The λ penalty gates split acceptance: a split is taken only when the
regularised loss of the children (each including its own λ per leaf)
is strictly less than the leaf loss.  This implements sparse greedy
induction where λ directly controls tree complexity.
"""

import numpy as np

from .utils.nodes import SPLITLeaf, SPLITNode


class GreedyTreeBuilder:
    """Top-down sparse-greedy tree builder per Algorithm 4.

    Loss convention (per-subproblem):
        lb = λ · num_leaves + (1/|D|) · Σ 1[yi ≠ prediction]

    The split is λ-gated: children are accepted only when
        lbleft + lbright < lbleaf
    which is equivalent to
        λ + (err_left/|D_left| + err_right/|D_right|) < err_leaf/|D|

    Because going from 1 leaf to ≥2 leaves costs at least one extra λ.
    """

    def __init__(self, depth_budget=3, reg=0.0001, max_features=0,
                 feature_order=None, pick_first=False, global_N=None):
        """Args:
            depth_budget: Maximum depth for the greedy tree (d in Alg 4).
                d=1 means leaf only (depth budget of 1 → no splits possible).
            reg: Regularisation penalty λ per leaf.
            max_features: Top-K candidate features ranked by entropy gain.
                0 = unlimited (all features).
            feature_order: Pre-computed feature ordering (int array).
                When provided, skips the internal entropy ranking.
            pick_first: If True, pick the first feature with positive gain
                instead of the best gain.  Used by ReSPLIT for diversity.
            global_N: Global dataset size N for loss normalisation.
                When provided, errors are normalised by global_N instead of
                the local subproblem size, making the greedy builder optimise
                the same global objective as Equation 1 in the paper.
                When None, falls back to per-subproblem normalisation (legacy).
        """
        self.depth_budget = depth_budget
        self.reg = reg
        self.max_features = max_features
        self._feature_order = feature_order
        self._pick_first = pick_first
        self._global_N = global_N

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, X, y):
        """Build a sparse-greedy tree per Algorithm 4.

        Args:
            X: (n_samples, n_features) binary ndarray (bool).
            y: (n_samples,) integer class labels.

        Returns:
            (tree, lb) where
                tree  — SPLITNode / SPLITLeaf root,
                lb    — regularised loss using global-N normalisation
                        (if global_N was provided) or per-subproblem
                        normalisation (legacy fallback).
        """
        X = np.asarray(X, dtype=bool)
        y = np.asarray(y, dtype=np.int64)
        # Use global_N if provided; otherwise fall back to local N.
        self._n_norm = self._global_N if self._global_N is not None else len(y)
        if self._feature_order is None:
            self._feature_order = self._rank_features(X, y)
        return self._greedy_impl(X, y, self.depth_budget)

    # ------------------------------------------------------------------
    # Algorithm 4 core recursion
    # ------------------------------------------------------------------

    def _greedy_impl(self, X, y, depth):
        """Recursive greedy tree induction — Algorithm 4.

        Returns (tree, lb) where lb uses global-N normalisation
        (if global_N was provided) or per-subproblem normalisation
        (legacy fallback).
        """
        n = len(y)

        # --- Lines 1-2: leaf baseline ---
        classes, counts = np.unique(y, return_counts=True)
        y_pred = int(classes[np.argmax(counts)])
        n_wrong = int(n - counts.max())
        # Global-N normalisation: error/N + λ (matches Equation 1).
        # Legacy fallback: error/|D| + λ (per-subproblem).
        lb = self.reg + (n_wrong / self._n_norm if self._n_norm > 0 else 0.0)

        # --- Line 3: try to split if depth budget remains ---
        if depth > 1 and n >= 2:
            best_feat = self._best_feature(X, y, depth)
            if best_feat >= 0:
                # --- Lines 5-6: recurse on children ---
                left_mask = X[:, best_feat]
                right_mask = ~left_mask
                nl = left_mask.sum()
                nr = n - nl

                if nl > 0 and nr > 0:
                    tleft, lbleft = self._greedy_impl(
                        X[left_mask], y[left_mask], depth - 1)
                    tright, lbright = self._greedy_impl(
                        X[right_mask], y[right_mask], depth - 1)

                    # --- Line 7: λ-gated split acceptance ---
                    if lbleft + lbright < lb:
                        # --- Lines 8-9: accept the split ---
                        return (
                            SPLITNode(feature=int(best_feat),
                                      left_child=tleft,
                                      right_child=tright),
                            lbleft + lbright,
                        )

        # --- Line 12: return leaf (split not accepted or not possible) ---
        # Leaf loss stored WITHOUT λ so the parent can sum correctly.
        # (The λ was already added in `lb` at line 2.)
        leaf_loss_without_reg = (n_wrong / self._n_norm) if self._n_norm > 0 else 0.0
        return (
            SPLITLeaf(prediction=y_pred, loss=leaf_loss_without_reg),
            lb,
        )

    # ------------------------------------------------------------------
    # Feature selection (information gain) — Line 4
    # ------------------------------------------------------------------

    def _best_feature(self, X, y, depth):
        """Return the index of the feature maximising information gain.

        Corresponds to Algorithm 4, Line 4:
            "let f be the information gain maximizing split w.r.t. D"
        """
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

    # ------------------------------------------------------------------
    # Feature ranking (entropy gain — used as candidate pre-filter)
    # ------------------------------------------------------------------

    def _rank_features(self, X, y):
        """Sort features by decreasing entropy gain on the full dataset."""
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
