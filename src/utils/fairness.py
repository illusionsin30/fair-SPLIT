"""Fairness utilities: pre-processing and post-processing for fair trees."""

import numpy as np


def fair_preprocess(X, sensitive_cols):
    """Drop sensitive columns from X before training."""
    dropped = {}
    existing = [c for c in sensitive_cols if c in X.columns]
    if existing:
        for c in existing:
            dropped[c] = X[c].copy()
        X_fair = X.drop(columns=existing)
        print(f"  [FairPreprocess] Dropped: {existing}")
    else:
        X_fair = X.copy()
        print(f"  [FairPreprocess] No sensitive columns found in X")
    return X_fair, dropped


def fair_calibrate(y_pred, y_true, sensitive, target_rate=None, random_state=None):
    """Adjust per-group predictions to equalize positive rates."""
    y_fair = y_pred.copy()
    groups = np.unique(sensitive)
    if isinstance(random_state, (np.random.RandomState, np.random.Generator)):
        rng = random_state
    elif random_state is None:
        rng = np.random
    else:
        rng = np.random.RandomState(random_state)

    if target_rate is None:
        target_rate = np.mean(y_pred)

    for g in groups:
        mask = sensitive == g
        n_g = mask.sum()
        current_rate = np.mean(y_pred[mask])
        desired_pos = int(round(target_rate * n_g))

        if current_rate > target_rate:
            pos_idx = np.where(mask & (y_pred == 1))[0]
            n_flip = int(np.sum(y_pred[mask])) - desired_pos
            if n_flip > 0 and len(pos_idx) > 0:
                true_neg = pos_idx[y_true[pos_idx] == 0]
                flip_idx = rng.choice(
                    true_neg if len(true_neg) >= n_flip else pos_idx,
                    size=min(n_flip, len(pos_idx)), replace=False,
                )
                y_fair[flip_idx] = 0

        elif current_rate < target_rate:
            neg_idx = np.where(mask & (y_pred == 0))[0]
            n_flip = desired_pos - int(np.sum(y_pred[mask]))
            if n_flip > 0 and len(neg_idx) > 0:
                false_neg = neg_idx[y_true[neg_idx] == 1]
                flip_idx = rng.choice(
                    false_neg if len(false_neg) >= n_flip else neg_idx,
                    size=min(n_flip, len(neg_idx)), replace=False,
                )
                y_fair[flip_idx] = 1

    orig_diff = max(np.mean(y_pred[sensitive == g]) for g in groups) - \
                min(np.mean(y_pred[sensitive == g]) for g in groups)
    new_diff = max(np.mean(y_fair[sensitive == g]) for g in groups) - \
               min(np.mean(y_fair[sensitive == g]) for g in groups)
    print(f"  [FairCalibrate] SP diff: {orig_diff:.4f} -> {new_diff:.4f} "
          f"(target rate={target_rate:.4f})")
    return y_fair


class LeafParetoRecalibrator:
    """Leaf-level prediction overrides learned from a calibration set."""

    def __init__(
        self,
        leaf_overrides=None,
        base_gap=0.0,
        calibrated_gap=0.0,
        changed_leaves=0,
        affected_samples=0,
        accuracy_delta=0.0,
        metric="dp",
    ):
        """Initialize a leaf-level fairness recalibrator."""
        self.leaf_overrides = dict(leaf_overrides or {})
        self.base_gap = float(base_gap)
        self.calibrated_gap = float(calibrated_gap)
        self.changed_leaves = int(changed_leaves)
        self.affected_samples = int(affected_samples)
        self.accuracy_delta = float(accuracy_delta)
        self.metric = metric

    def apply(self, leaf_ids, base_pred):
        """Apply learned leaf overrides to base predictions.

        Args:
            leaf_ids: Iterable of path-based leaf ids.
            base_pred: Base model predictions.

        Returns:
            Numpy array with calibrated predictions.
        """
        calibrated = np.asarray(base_pred).copy()
        for idx, leaf_id in enumerate(leaf_ids):
            key = tuple(leaf_id)
            if key in self.leaf_overrides:
                calibrated[idx] = self.leaf_overrides[key]
        return calibrated


def fit_leaf_pareto_recalibrator(
    model,
    X_cal,
    y_cal,
    sensitive,
    base_pred=None,
    metric="dp",
    fair_lambda=1.0,
    acc_budget=0.02,
):
    """Fit a leaf-level Pareto fairness recalibrator.

    Args:
        model: Fitted CART/SPLIT-family model.
        X_cal: Calibration features.
        y_cal: Calibration labels.
        sensitive: Sensitive group values aligned with ``X_cal``.
        base_pred: Optional base predictions on ``X_cal``.
        metric: Fairness metric to optimize: ``"dp"`` or ``"eo"``.
        fair_lambda: Multiplier for fairness gain in the greedy score.
        acc_budget: Maximum allowed calibration accuracy drop.

    Returns:
        A ``LeafParetoRecalibrator`` with path-based leaf overrides.
    """
    if metric not in ("dp", "eo"):
        raise ValueError(f"metric must be 'dp' or 'eo', got {metric!r}")

    y_cal = np.asarray(y_cal)
    sensitive = np.asarray(sensitive)
    if base_pred is None:
        base_pred = model.predict(X_cal)
    current_pred = np.asarray(base_pred).copy()
    base_pred = np.asarray(base_pred)

    leaf_ids = _collect_model_leaf_ids(model, X_cal)
    base_acc = _accuracy(y_cal, base_pred)
    base_gap = _fairness_gap(y_cal, base_pred, sensitive, metric)
    current_gap = base_gap
    overrides = {}

    while True:
        best = None
        for leaf_id in sorted(set(leaf_ids)):
            leaf_id = tuple(leaf_id)
            mask = np.array([tuple(v) == leaf_id for v in leaf_ids])
            if not mask.any():
                continue
            original_value = _majority_value(current_pred[mask])
            for candidate in np.unique(y_cal):
                if candidate == original_value:
                    continue
                trial_pred = current_pred.copy()
                trial_pred[mask] = candidate
                trial_acc = _accuracy(y_cal, trial_pred)
                acc_drop = base_acc - trial_acc
                if acc_drop > acc_budget:
                    continue
                trial_gap = _fairness_gap(y_cal, trial_pred, sensitive, metric)
                fairness_gain = current_gap - trial_gap
                if fairness_gain <= 0:
                    continue
                score = ((fair_lambda * fairness_gain) /
                         (max(acc_drop, 0.0) + 1e-12))
                if best is None or score > best["score"]:
                    best = {
                        "leaf_id": leaf_id,
                        "candidate": candidate,
                        "mask": mask,
                        "pred": trial_pred,
                        "gap": trial_gap,
                        "score": score,
                    }
        if best is None:
            break
        current_pred = best["pred"]
        current_gap = best["gap"]
        candidate = best["candidate"]
        overrides[best["leaf_id"]] = (
            candidate.item() if hasattr(candidate, "item") else candidate
        )

    affected = 0
    for leaf_id in overrides:
        affected += int(np.sum([tuple(v) == leaf_id for v in leaf_ids]))
    acc_delta = _accuracy(y_cal, current_pred) - base_acc
    print(f"  [LeafPareto] {metric.upper()} gap: {base_gap:.4f} -> {current_gap:.4f} "
          f"(changed_leaves={len(overrides)}, affected={affected}, "
          f"deltaacc={acc_delta:+.4f})")
    return LeafParetoRecalibrator(
        leaf_overrides=overrides,
        base_gap=base_gap,
        calibrated_gap=current_gap,
        changed_leaves=len(overrides),
        affected_samples=affected,
        accuracy_delta=acc_delta,
        metric=metric,
    )


def apply_leaf_pareto_recalibration(model, X, base_pred, recalibrator):
    """Apply a fitted leaf-level Pareto recalibrator.

    Args:
        model: Fitted CART/SPLIT-family model.
        X: Features to recalibrate.
        base_pred: Base model predictions on ``X``.
        recalibrator: Fitted ``LeafParetoRecalibrator``.

    Returns:
        Numpy array with recalibrated predictions.
    """
    leaf_ids = _collect_model_leaf_ids(model, X)
    return recalibrator.apply(leaf_ids, base_pred)


def _collect_model_leaf_ids(model, X):
    """Return path-based leaf ids for a fitted CART/SPLIT-family model."""
    from .helpers import cart_leaf_id_batch, split_leaf_id_batch

    if hasattr(model, "root") and model.root is not None:
        return cart_leaf_id_batch(X, model.root)

    tree = None
    X_bin = X
    if hasattr(model, "tree") and model.tree is not None:
        tree = model.tree
    elif hasattr(model, "models") and len(model.models) > 0:
        tree = model.models[0][0]

    if tree is None:
        raise ValueError("model must be a fitted CART/SPLIT-family model")

    if (getattr(model, "binarize_flag", False) and
            getattr(model, "_enc", None) is not None):
        X_bin = model._enc.transform(X)
    X_bin = np.asarray(X_bin, dtype=bool)
    return split_leaf_id_batch(X_bin, tree)


def _fairness_gap(y_true, y_pred, sensitive, metric):
    """Compute DP or EO max-min group gap."""
    groups = np.unique(sensitive)
    values = []
    for group in groups:
        mask = sensitive == group
        if not mask.any():
            continue
        if metric == "dp":
            values.append(float(np.mean(y_pred[mask])))
        else:
            positive_mask = mask & (y_true == 1)
            if positive_mask.any():
                values.append(float(np.mean(y_pred[positive_mask])))
    if len(values) < 2:
        return 0.0
    return max(values) - min(values)


def _accuracy(y_true, y_pred):
    """Return simple classification accuracy."""
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def _majority_value(values):
    """Return the most common value in an array."""
    labels, counts = np.unique(values, return_counts=True)
    return labels[np.argmax(counts)]
