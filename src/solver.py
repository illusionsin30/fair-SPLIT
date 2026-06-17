r"""Pure-Python optimal decision tree solver via DP + branch-and-bound.

Implements the recursive equation (Equation 4 / Equation 8) from the
SPLIT paper, replacing the C++ GOSDT used in the original codebase.

Objective (per Equation 1):
    L(T, D, \lambda) = (1/N) \sum_{i=1}^N 1[y_i \neq T(x_i)] + \lambda \cdot S(T)
where N is the GLOBAL dataset size and S(T) is the number of leaves.

Branch-and-bound guts
----------------------
At the lookahead boundary (remaining_depth == 0) we call Algorithm 4
(Greedy) to get an upper-bound tree and set lb = ub = greedy loss
(see Algorithm 12 / get_bounds line 6-7).

At other depths we compute standard DP: for each feature candidate,
recurse on both children; the best total loss min_f(L(D(f), d'-1, \lambda)
+ L(D(\bar f), d'-1, \lambda)) is compared against the leaf loss.  If it
is strictly better we split, otherwise we return the leaf.

The \lambda penalty is accumulated in the returned loss value and is the
sole split-gating mechanism (no unconditional splitting).
"""

import time
import numpy as np

from .builder import GreedyTreeBuilder
from .utils.nodes import SPLITLeaf, SPLITNode


class OptimalTreeSolver:
    """DP optimal tree solver with branch-and-bound per Equation 4 / Eq. 8.

    Parameters
    ----------
    depth_budget : int
        Maximum depth for this solver call.
    reg : float
        Regularisation penalty \lambda per leaf.
    time_limit : float
        Maximum wall-clock time in seconds.
    max_features : int
        Top-K candidate features (0 = all).
    remaining_depth : int
        Depth budget for greedy completion at the lookahead boundary.
        When the solver reaches depth 0 in its recursion with
        remaining_depth > 0, it delegates to Algorithm 4 (Greedy)
        for the remaining levels (Algorithm 12 / get_bounds).
    global_N : int or None
        Global dataset size for loss normalisation.  If None, uses
        len(y) of the current subproblem (root call only).
    """

    def __init__(self, depth_budget=3, reg=0.0001, time_limit=60,
                 max_features=0, remaining_depth=0, global_N=None):
        if depth_budget < 0:
            raise ValueError("depth_budget must be non-negative")
        self.depth_budget = depth_budget
        self.reg = reg
        self.time_limit = time_limit
        self.max_features = max_features
        self.remaining_depth = remaining_depth  # greedy depth at boundary
        self._global_N = global_N

        # Internal state (reset per fit() call)
        self._memo = {}
        self._n_global = 0          # N used for loss normalisation
        self._start_time = None
        self._timed_out = False
        self._feature_order = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X, y, upper_bound_tree=None):
        """Find the optimal tree according to Eq. 1.

        Parameters
        ----------
        X : (n_samples, n_features) binary ndarray, dtype=bool.
        y : (n_samples,) integer class labels.
        upper_bound_tree : optional (tree, loss) tuple used as a
            global upper bound for pruning.

        Returns
        -------
        (tree, loss) — the optimal tree and its regularised loss
        (normalised by global N).
        """
        X = np.asarray(X, dtype=bool)
        y = np.asarray(y, dtype=np.int64)

        if self._global_N is not None:
            self._n_global = self._global_N
        else:
            self._n_global = len(y)

        self._memo = {}
        self._start_time = time.perf_counter()
        self._timed_out = False
        # Store global class labels for loss computation (subproblems may
        # contain only a subset of classes).
        self._global_classes = np.unique(y)

        if upper_bound_tree is not None:
            _, ub_loss = upper_bound_tree
            self._upper_bound = ub_loss
        else:
            self._upper_bound = np.inf

        # Feature ranking (entropy gain on current data)
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

    # ------------------------------------------------------------------
    # Feature ranking
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
    # DP core — Equation 4 / Equation 8
    # ------------------------------------------------------------------

    def _build(self, X, y, depth):
        """Recursive optimal tree DP.

        Parameters
        ----------
        X : ndarray[bool]
        y : ndarray[int64]
        depth : int
            Remaining depth budget for THIS solver call.  When depth
            reaches 0 and remaining_depth > 0 we hit the lookahead
            boundary and delegate to Algorithm 4.

        Returns
        -------
        (tree, loss) — loss normalised by global N.
        """
        n = len(y)
        if n == 0:
            return SPLITLeaf(prediction=0, loss=0.0), 0.0

        # --- lookahead boundary: greedy completion (Algorithm 12 lines 6-7) ---
        if depth == 0 and self.remaining_depth > 0:
            return self._greedy_completion(X, y)

        # --- time-limit check ---
        if (self._start_time is not None
                and time.perf_counter() - self._start_time > self.time_limit):
            self._timed_out = True
            leaf = self._make_leaf(y)
            return leaf, self._leaf_loss(y)

        # --- leaf baseline ---
        leaf = self._make_leaf(y)
        leaf_loss = self._leaf_loss(y)

        # NOTE: We do NOT prune when leaf_loss > upper_bound.  The leaf
        # loss is an UPPER BOUND on the best tree at this node (splitting
        # can only improve it), not a lower bound.  Pruning on it would
        # prevent the solver from ever finding splits at the root when
        # the greedy upper bound is better than a single leaf — which is
        # exactly when splitting is most needed.  The correct early-exit
        # is at the feature loop below (best_loss <= upper_bound).

        # --- try splits if depth budget remains ---
        if depth > 0 and n >= 2:
            # memoization lookup
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

                # λ-gated split: children must strictly improve
                # the regularised objective vs a leaf
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

            self._memo[memo_key] = (best_node, best_loss)
            return best_node, best_loss

        # Fallback: no splits possible (depth exhausted or too few samples)
        return leaf, leaf_loss

    def _greedy_completion(self, X, y):
        """Return a greedy subtree at the lookahead boundary.

        Calls Algorithm 4 (Greedy) for the remaining depth budget.
        The greedy loss is returned directly — we treat it as if
        lb = ub = greedy_loss (per Algorithm 12, line 6-7).
        """
        builder = GreedyTreeBuilder(
            depth_budget=self.remaining_depth,
            reg=self.reg,
            max_features=self.max_features,
            global_N=self._n_global,
        )
        # builder.build() returns per-subproblem normalised loss.
        # We need to convert to global N scale:
        #   lb_sub = λ·L + err_sub / |D_sub|
        #   lb_global = λ·L + err_sub / N
        # The conversion is:
        #   lb_global = lb_sub + err_sub·(1/N - 1/|D_sub|)
        #   ... which equals lb_sub + (err/|D|)·(|D|/N - 1)
        # But it is simpler to take the subtree, recompute its loss
        # on the global scale using predict + count.
        tree, _loss_sub = builder.build(X, y)
        loss_global = self._compute_tree_loss(tree, X, y)
        return tree, loss_global

    def _compute_tree_loss(self, tree, X, y):
        """Compute L(T, D, λ) using global N normalisation."""
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

    # ------------------------------------------------------------------
    # Leaf helpers (global N normalised loss)
    # ------------------------------------------------------------------

    def _make_leaf(self, y):
        """Create a leaf node predicting the majority class."""
        classes, counts = np.unique(y, return_counts=True)
        y_pred = int(classes[np.argmax(counts)])
        n_wrong = int(len(y) - counts.max())
        # store raw error count — caller adds reg
        raw_loss = n_wrong / self._n_global
        return SPLITLeaf(prediction=y_pred, loss=raw_loss)

    def _leaf_loss(self, y):
        """L(c, y) + λ for a leaf on subproblem y (global N scale)."""
        classes, counts = np.unique(y, return_counts=True)
        n_wrong = int(len(y) - counts.max())
        return n_wrong / self._n_global + self.reg

    # ------------------------------------------------------------------
    # Candidate pruning
    # ------------------------------------------------------------------

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
    # Memoization
    # ------------------------------------------------------------------

    @staticmethod
    def _memo_key(y, depth):
        """Build a compact memoization key from y and depth.

        We use the hash of y.tobytes() + depth — fast and unique enough
        for node-level caching in a single fit() call.
        """
        return hash((y.tobytes(), depth))
