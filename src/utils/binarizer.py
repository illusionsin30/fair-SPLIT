"""Binarization utilities."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_X_y, check_is_fitted


def _halfway_points(values):
    """Return midpoints between consecutive sorted unique values."""
    return [(values[i] + values[i + 1]) / 2.0 for i in range(len(values) - 1)]


class NumericBinarizer(BaseEstimator, TransformerMixin):
    """Binarize mixed-type DataFrames to a single binary matrix.

    - Numeric columns: midpoint-threshold binarization.
      Feature [1, 3, 7] -> two binary cols "x <= 2.0", "x <= 5.0".
    - String / categorical columns: one-hot encoding.
      Feature ["A","B","A"] -> two binary cols "x=A", "x=B".

    This is a lossless transform for numeric features; the original values
    can be recovered via inverse_transform (numeric columns only).

    Attributes:
        n_features_in_: Number of input features.
        n_features_out_: Number of output binary features.
        feature_names_in_: Names of input features.
        _numeric_cols_: Column indices of numeric features.
        _string_cols_: Column indices of string/categorical features.
        column_values_: Unique sorted values per numeric feature.
        _string_categories_: Unique categories per string feature.
    """

    def __init__(self, max_thresholds=50):
        """Max midpoints per numeric feature."""
        self.max_thresholds = max_thresholds

    def fit(self, X, y=None, columns=None):
        """Fit the binarizer."""
        self.feature_names_in_ = columns
        if hasattr(X, "columns"):
            self.feature_names_in_ = list(X.columns)
        if self.feature_names_in_ is None:
            if hasattr(X, "shape"):
                self.feature_names_in_ = [f"x{i}" for i in range(X.shape[1])]
            else:
                self.feature_names_in_ = [f"x0"]

        if hasattr(X, "iloc"):
            X_df = X
        else:
            X_df = pd.DataFrame(X, columns=self.feature_names_in_)

        self._numeric_cols_ = []
        self._string_cols_ = []
        self.column_values_ = []
        self._string_categories_ = []

        for i, col_name in enumerate(self.feature_names_in_):
            col = X_df.iloc[:, i]
            if pd.api.types.is_numeric_dtype(col):
                self._numeric_cols_.append(i)
                unique_vals = np.unique(col.dropna().values.astype(np.float64))
                if self.max_thresholds and len(unique_vals) > self.max_thresholds + 1:
                    indices = np.linspace(
                        0, len(unique_vals) - 1,
                        self.max_thresholds + 1, dtype=int,
                    )
                    unique_vals = unique_vals[indices]
                self.column_values_.append(unique_vals)
            else:
                self._string_cols_.append(i)
                cats = sorted(col.dropna().unique())
                self._string_categories_.append(cats)

        self.n_features_in_ = len(self.feature_names_in_)
        num_width = sum(max(len(vals) - 1, 0) for vals in self.column_values_)
        cat_width = sum(len(cats) for cats in self._string_categories_)
        self.n_features_out_ = num_width + cat_width
        return self

    def get_feature_names_out(self, *args, **params):
        """Generate transformed column names."""
        check_is_fitted(self, ["n_features_in_", "_numeric_cols_", "_string_cols_",
                               "column_values_", "_string_categories_"])
        names = []
        for idx, vals in zip(self._numeric_cols_, self.column_values_):
            base = self.feature_names_in_[idx]
            if len(vals) <= 1:
                continue
            for hp in _halfway_points(vals):
                names.append(f"{base} <= {hp}")
        for idx, cats in zip(self._string_cols_, self._string_categories_):
            base = self.feature_names_in_[idx]
            for cat in cats:
                names.append(f"{base}={cat}")
        return np.array(names)

    def transform(self, X):
        """Transform X to binary features.

        Args:
            X: array-like of shape (n_samples, n_features_in_).

        Returns:
            X_bin: ndarray of shape (n_samples, n_features_out_), {0, 1}.
        """
        check_is_fitted(self, ["n_features_in_", "_numeric_cols_", "_string_cols_",
                               "column_values_", "_string_categories_"])

        if hasattr(X, "iloc"):
            X_df = X
        else:
            X_df = pd.DataFrame(X, columns=self.feature_names_in_)

        blocks = []

        for idx, vals in zip(self._numeric_cols_, self.column_values_):
            col = X_df.iloc[:, idx].values.astype(np.float64)
            if len(vals) <= 1:
                continue
            for hp in _halfway_points(vals):
                blocks.append((col <= hp).astype(np.float64))

        for idx, cats in zip(self._string_cols_, self._string_categories_):
            col = X_df.iloc[:, idx].values.astype(str)
            for cat in cats:
                blocks.append((col == cat).astype(np.float64))

        if not blocks:
            return np.zeros((X_df.shape[0], 0), dtype=np.float64)
        return np.column_stack(blocks)

    def inverse_transform(self, Xt):
        """Recover approximate original numeric values."""
        check_is_fitted(self, ["_numeric_cols_", "column_values_"])
        Xt_arr = np.asarray(Xt, dtype=np.float64)

        num_width = sum(max(len(v) - 1, 0) for v in self.column_values_)
        Xt_num = Xt_arr[:, :num_width]

        X = np.empty((Xt_arr.shape[0], len(self._numeric_cols_)))
        col_idx = 0
        for i, feature_values in enumerate(self.column_values_):
            n_unique = len(feature_values)
            n_mid = n_unique - 1
            if n_unique <= 1:
                X[:, i] = feature_values[0]
                continue
            block = Xt_num[:, col_idx : col_idx + n_mid]
            indices = np.argmax(block, axis=1)
            indices = np.where(np.any(block, axis=1), indices, n_mid)
            X[:, i] = feature_values[np.clip(indices, 0, len(feature_values) - 1)]
            col_idx += n_mid
        return X

    def feature_map(self):
        """Map original feature index to binarized columns."""
        check_is_fitted(self, ["n_features_in_", "_numeric_cols_", "_string_cols_",
                               "column_values_", "_string_categories_"])
        ret = {}
        idx = 0
        for col_i, vals in zip(self._numeric_cols_, self.column_values_):
            n_mid = max(len(vals) - 1, 0)
            if n_mid > 0:
                ret[col_i] = list(range(idx, idx + n_mid))
            idx += n_mid
        for col_i, cats in zip(self._string_cols_, self._string_categories_):
            n_cat = len(cats)
            if n_cat > 0:
                ret[col_i] = list(range(idx, idx + n_cat))
            idx += n_cat
        return ret


class ThresholdGuessBinarizer(BaseEstimator, TransformerMixin):
    """Binarize numeric features using thresholds from a GBDT ensemble."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 1,
        learning_rate: float = 0.1,
        random_state: int = 42,
        column_elimination: bool = False,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.column_elimination = column_elimination

    def fit(self, X, y, columns=None):
        """Fit GBDT stumps and extract thresholds."""
        from sklearn.ensemble import GradientBoostingClassifier

        self.feature_names_in_ = columns
        if hasattr(X, "columns"):
            self.feature_names_in_ = list(X.columns)

        if hasattr(X, "iloc"):
            X_df = X
        else:
            X_df = pd.DataFrame(X, columns=self.feature_names_in_)

        if self.feature_names_in_ is None:
            self.feature_names_in_ = [f"x{i}" for i in range(X_df.shape[1])]

        num_cols = []
        str_cols = []
        for i, col_name in enumerate(self.feature_names_in_):
            col = X_df.iloc[:, i]
            if pd.api.types.is_numeric_dtype(col):
                num_cols.append(i)
            else:
                str_cols.append(i)
        print(f"  [Binarizer] {len(num_cols)} numeric, {len(str_cols)}"
              f" categorical (of {len(self.feature_names_in_)} total)")
        if str_cols:
            print(f"    categorical: {[self.feature_names_in_[i] for i in str_cols]}")

        self._str_cols_ = str_cols
        self._str_categories_ = []
        for i in str_cols:
            cats = sorted(X_df.iloc[:, i].dropna().unique())
            self._str_categories_.append(cats)

        if num_cols:
            X_num = X_df.iloc[:, num_cols].values.astype(np.float64)
        else:
            X_num = np.zeros((X_df.shape[0], 0))

        gbdt = GradientBoostingClassifier(
            loss="log_loss",
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )
        gbdt.fit(X_num, y)

        thresholds = []
        for local_j, global_j in enumerate(num_cols):
            th_j = []
            for est in gbdt.estimators_.ravel():
                tree = est.tree_
                th_j.append(tree.threshold[tree.feature == local_j])
            th_j = np.unique(np.concatenate(th_j)) if th_j else np.array([])
            for th in th_j:
                thresholds.append((global_j, th))

        global_to_local = {global_j: local_j
                           for local_j, global_j in enumerate(num_cols)}

        if self.column_elimination and thresholds:
            self.thresholds_ = self._column_elimination(
                X_num, y, thresholds, gbdt, global_to_local,
            )
        else:
            self.thresholds_ = thresholds

        self.n_features_out_ = (len(self.thresholds_)
                                + sum(len(c) for c in self._str_categories_))
        return self

    def transform(self, X):
        """Apply threshold binarization and one-hot encoding."""
        check_is_fitted(self, ["thresholds_", "_str_cols_", "_str_categories_"])

        if hasattr(X, "iloc"):
            X_df = X
        else:
            X_df = pd.DataFrame(X, columns=self.feature_names_in_)

        blocks = []
        for j, th in self.thresholds_:
            col_vals = X_df.iloc[:, j].values.astype(np.float64)
            blocks.append((col_vals <= th).astype(np.float64))

        for idx, cats in zip(self._str_cols_, self._str_categories_):
            col_str = X_df.iloc[:, idx].values.astype(str)
            for cat in cats:
                blocks.append((col_str == cat).astype(np.float64))

        if not blocks:
            return np.zeros((X_df.shape[0], 0), dtype=np.float64)
        return np.column_stack(blocks)

    def get_feature_names_out(self, *args, **params):
        """Generate transformed column names."""
        check_is_fitted(self, ["feature_names_in_", "thresholds_",
                               "_str_cols_", "_str_categories_"])
        names = []
        for j, th in self.thresholds_:
            names.append(f"{self.feature_names_in_[j]} <= {th:.6f}")
        for idx, cats in zip(self._str_cols_, self._str_categories_):
            base = self.feature_names_in_[idx]
            for cat in cats:
                names.append(f"{base}={cat}")
        return np.array(names)

    def feature_map(self):
        """Map original feature index to binarized columns."""
        check_is_fitted(self, ["thresholds_", "_str_cols_", "_str_categories_"])
        ret = {}
        idx = 0
        for j, _ in self.thresholds_:
            ret.setdefault(j, []).append(idx)
            idx += 1
        for col_i, cats in zip(self._str_cols_, self._str_categories_):
            ret.setdefault(col_i, []).extend(range(idx, idx + len(cats)))
            idx += len(cats)
        return ret

    @staticmethod
    def _threshold(estimator, feature):
        """Extract thresholds used for a feature."""
        f = estimator.tree_.feature
        t = estimator.tree_.threshold
        return t[f == feature]

    def _column_elimination(self, X_num, y, thresholds, gbdt, global_to_local):
        """Eliminate thresholds by importance."""
        th = list(thresholds)
        if len(th) <= 1:
            return th

        X_thr = np.column_stack([
            (X_num[:, global_to_local[j]] <= val).astype(np.float64)
            for j, val in th
        ])

        gbdt.fit(X_thr, y)
        base_score = gbdt.score(X_thr, y)

        curr_score = np.inf
        last_dropped = None
        n_iter = 0
        max_iter = len(th) - 1
        while (curr_score >= base_score and n_iter < max_iter
               and gbdt.feature_importances_.size > 0):
            least = int(np.argmin(gbdt.feature_importances_))
            last_dropped = (X_thr[:, least].copy(), th[least])
            X_thr = np.delete(X_thr, least, axis=1)
            th.pop(least)
            gbdt.fit(X_thr, y)
            curr_score = gbdt.score(X_thr, y)
            n_iter += 1

        if last_dropped is not None:
            th.append(last_dropped[1])

        print(f"  [ThresholdGuess] {len(thresholds)} thresholds"
              f" -> {len(th)} kept ({n_iter} eliminated)")
        return th
