"""Dataset loading utilities."""
import os
import re
import subprocess

import numpy as np
import pandas as pd


_OPENML_CACHE = {}

def _fetch_openml(name, version="active", target_col=None, drop_cols=None):
    """Load a dataset from OpenML via sklearn, with caching.

    Args:
        name: OpenML dataset name (e.g. "adult").
        version: Dataset version (default "active" = latest).
        target_col: Column name to use as target.  If None, auto-detected
                    from the returned frame's target attribute.
        drop_cols: Columns to drop from X (e.g. identifiers / leaky cols).

    Returns:
        X (DataFrame), y (ndarray), num_feats (list), cat_feats (list).
    """
    from sklearn.datasets import fetch_openml

    cache_key = (name, version)
    if cache_key not in _OPENML_CACHE:
        print(f"  [OpenML] Loading {name} (version={version}) ...")
        bunch = fetch_openml(name=name, version=version, parser="auto",
                             as_frame=True)
        _OPENML_CACHE[cache_key] = bunch

    bunch = _OPENML_CACHE[cache_key]
    X = bunch.data.copy()
    y_raw = bunch.target

    if target_col is not None and target_col in X.columns:
        y_raw = X.pop(target_col)
    elif isinstance(y_raw, pd.DataFrame):
        y_raw = y_raw.iloc[:, 0]
    elif y_raw.name is not None and y_raw.name in X.columns:
        X = X.drop(columns=[y_raw.name])
    y = np.asarray(y_raw)

    if drop_cols:
        X = X.drop(columns=[c for c in drop_cols if c in X.columns],
                   errors="ignore")

    num_feats = list(X.select_dtypes(include=[np.number]).columns)
    cat_feats = list(X.select_dtypes(exclude=[np.number]).columns)

    n_before = len(X)
    valid = X.notna().all(axis=1) & pd.notna(y)
    X = X.loc[valid].reset_index(drop=True)
    y = y[valid]
    if len(X) < n_before:
        print(f"  Dropped {n_before - len(X)} rows with missing values.")

    print(f"  Shape: {X.shape}  |  Classes: {sorted(np.unique(y))}")
    return X, y, num_feats, cat_feats


_KAGGLEHUB_AVAILABLE = None

def _check_kagglehub():
    global _KAGGLEHUB_AVAILABLE
    if _KAGGLEHUB_AVAILABLE is None:
        try:
            import kagglehub
            _KAGGLEHUB_AVAILABLE = True
        except ImportError:
            _KAGGLEHUB_AVAILABLE = False
    return _KAGGLEHUB_AVAILABLE


def _download_kaggle_dataset(dataset_slug, filename, local_dir, env_var):
    """Download a Kaggle dataset, with local-cache and fallback.

    Supports fuzzy filename matching to handle Kaggle's duplicate-file
    renaming (e.g. "file.csv" -> "file (1).csv").

    Returns:
        Absolute path to the CSV file.
    """
    env_path = os.environ.get(env_var)
    if env_path and os.path.isfile(env_path):
        print(f"Found local data via ${env_var}: {env_path}")
        return env_path

    base_no_ext = os.path.splitext(filename)[0]

    cache_base = os.path.expanduser("~/.cache/kagglehub/datasets")
    dataset_cache = os.path.join(
        cache_base, dataset_slug.replace("/", os.sep))
    if os.path.isdir(dataset_cache):
        for root, _dirs, files in os.walk(dataset_cache):
            for f in files:
                if f == filename or f.startswith(base_no_ext):
                    found = os.path.join(root, f)
                    print(f"Found in kagglehub cache: {found}")
                    return found

    local_dir_path = os.path.join("data", local_dir)
    if os.path.isdir(local_dir_path):
        for f in os.listdir(local_dir_path):
            if f == filename or f.startswith(base_no_ext):
                found = os.path.join(local_dir_path, f)
                if os.path.isfile(found):
                    print(f"Found local data: {found}")
                    return found

    if _check_kagglehub():
        try:
            import kagglehub
            path = kagglehub.dataset_download(dataset_slug)
            print(f"kagglehub downloaded to: {path}")
            for root, _dirs, files in os.walk(path):
                for f in files:
                    if f == filename or f.startswith(base_no_ext):
                        print(f"  Found {f} at {root}")
                        return os.path.join(root, f)
        except Exception as exc:
            print(f"[WARN] kagglehub download failed: {exc}")

    target = os.path.join("data", local_dir)
    try:
        os.makedirs(target, exist_ok=True)
        subprocess.run(
            ["kaggle", "datasets", "download",
             dataset_slug, "-p", target, "--unzip"],
            check=True,
        )
        for root, _dirs, files in os.walk(target):
            for f in files:
                if f == filename or f.startswith(base_no_ext):
                    print(f"Kaggle CLI downloaded to: {root}")
                    return os.path.join(root, f)
    except Exception as exc:
        print(f"[WARN] kaggle CLI download failed: {exc}")

    raise RuntimeError(
        f"Cannot obtain {filename} from {dataset_slug}.\n"
        f"Options:\n"
        f"  1. pip install kagglehub\n"
        f"  2. pip install kaggle && kaggle datasets download {dataset_slug}\n"
        f"  3. Download {filename} manually and set {env_var}=/path/to/file"
    )


def load_adult_data():
    """Adult / Census Income - binary classification (>50K/yr)."""
    print("[Adult] Loading via sklearn (OpenML ID 1590) ...")
    X, y, num, cat = _fetch_openml("adult", version=2)
    if y.dtype.kind in ("U", "S", "O"):
        y = np.array([str(v).strip().rstrip(".") for v in y])
        y = (y == ">50K").astype(int)
    print("  Class balance:\n", pd.Series(y).value_counts(normalize=True))
    return X, y, num, cat


def load_bike_data():
    """Bike Sharing - regression (cnt)."""
    print("[Bike] Loading via sklearn (OpenML ID 42712) ...")
    X, y, num, cat = _fetch_openml("Bike_Sharing_Demand", version=2,
                                   target_col="cnt",
                                   drop_cols=["dteday", "Unnamed: 0"])
    print(f"  Target (cnt): mean={y.mean():.1f}, std={y.std():.1f}")
    return X, y, num, cat


def load_spambase_data():
    """Spambase - binary classification."""
    print("[Spambase] Loading via sklearn (OpenML ID 44) ...")
    X, y, num, cat = _fetch_openml("spambase", version=1)
    y = y.astype(int)
    print("  Class balance:\n", pd.Series(y).value_counts(normalize=True))
    return X, y, num, cat


def load_bank_data():
    """Bank Marketing - binary classification (term deposit)."""
    print("[Bank] Loading via sklearn (OpenML ID 1461) ...")
    X, y, num, cat = _fetch_openml("bank-marketing", version=2)
    y = (y.astype(int) - 1).astype(int)
    _bank_names = [
        "age", "job", "marital", "education", "default", "balance",
        "housing", "loan", "contact", "day", "month", "duration",
        "campaign", "pdays", "previous", "poutcome",
    ]
    if all(re.fullmatch(r"V\d+", col) for col in X.columns) and len(X.columns) == len(_bank_names):
        rename = dict(zip([f"V{i}" for i in range(1, len(_bank_names) + 1)], _bank_names))
        X = X.rename(columns=rename)
        num = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
        cat = ["job", "marital", "education", "default", "housing", "loan",
               "contact", "month", "poutcome"]
    print("  Class balance:\n", pd.Series(y).value_counts(normalize=True))
    return X, y, num, cat


def load_covertype_data():
    """Covertype - 7-class classification."""
    print("[Covertype] Loading via sklearn (OpenML ID 1596) ...")
    X, y, num, cat = _fetch_openml("covertype", version=3)
    y = y.astype(int) - 1
    print("  Class balance:\n", pd.Series(y).value_counts(normalize=True))
    return X, y, num, cat


def load_thyroid_data():
    """Thyroid Disease - binary classification (sick vs negative).

    OpenML data_id=40701 returns labels '0' (healthy) / '1' (sick).
    """
    print("[Thyroid] Loading via sklearn (OpenML ID 40701) ...")
    from sklearn.datasets import fetch_openml
    data = fetch_openml(data_id=40701, parser="auto", as_frame=True)
    X = data.data
    y = data.target.astype(int).values
    num_feats = list(X.select_dtypes(include=[np.number]).columns)
    cat_feats = list(X.select_dtypes(exclude=[np.number]).columns)
    print(f"  Shape: {X.shape}  |  Classes: {sorted(np.unique(y))}")
    print("  Class balance:\n", pd.Series(y).value_counts(normalize=True))
    return X, y, num_feats, cat_feats


def load_german_credit_data():
    """German Credit (Statlog) - binary classification.

    Fairness attributes: personal_status, age (openml col names).
    """
    print("[German Credit] Loading via sklearn (OpenML ID 31) ...")
    X, y, num, cat = _fetch_openml("credit-g", version=2)
    y = (y == "bad").astype(int)
    print("  Class balance:\n", pd.Series(y).value_counts(normalize=True))
    return X, y, num, cat


def load_communities_data():
    """Communities and Crime - binary (high/low violent crime).

    OpenML v1 returns numeric column names (0-146).  The last column is
    ViolentCrimesPerPop (continuous target).  First 2 columns are state/county.
    """
    print("[Communities & Crime] Loading via sklearn (OpenML name='communities-and-crime', v=1) ...")
    from sklearn.datasets import fetch_openml
    data = fetch_openml(name="communities-and-crime", version=1,
                        parser="auto", as_frame=True)
    X = data.data
    X = X.iloc[:, 2:]
    for col in X.columns:
        if X[col].dtype.kind == "O":
            X[col] = pd.to_numeric(X[col], errors="coerce")
    y = X.pop(X.columns[-1]).values
    X.columns = [f"feat_{i}" for i in range(X.shape[1])]

    valid = ~np.isnan(y)
    X = X.loc[valid].reset_index(drop=True)
    y = y[valid]

    nan_frac = X.isna().mean()
    drop_cols = list(nan_frac[nan_frac > 0.5].index)
    if drop_cols:
        X = X.drop(columns=drop_cols)
        print(f"  Dropped {len(drop_cols)} features with >50% missing values.")
    fill_cols = X.isna().any()
    for col in X.columns[fill_cols]:
        med = X[col].median()
        X[col] = X[col].fillna(med)
    if fill_cols.any():
        print(f"  Median-filled {fill_cols.sum()} features with <50% missing.")

    threshold = float(np.median(y))
    print(f"  Binarizing violent-crime rate at median = {threshold:.4f}")
    y = (y >= threshold).astype(int)

    non_num = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_num:
        X = X.drop(columns=non_num)
        print(f"  Dropped non-numeric columns: {non_num}")
    num = list(X.columns)
    cat = []
    print(f"  Shape: {X.shape}")
    print("  Class balance:\n", pd.Series(y).value_counts(normalize=True))
    return X, y, num, cat


def load_heart_data():
    """Heart Disease (Cleveland) - binary classification.

    OpenML v1 embeds the target column ('target') inside X.
    """
    print("[Heart Disease] Loading via sklearn (OpenML name='heart-disease', v=1) ...")
    from sklearn.datasets import fetch_openml
    data = fetch_openml(name="heart-disease", version=1, parser="auto",
                        as_frame=True)
    X = data.data
    y = X.pop("target").values
    y = (y.astype(int) > 0).astype(int)
    num = list(X.select_dtypes(include=[np.number]).columns)
    cat = list(X.select_dtypes(exclude=[np.number]).columns)
    print(f"  Shape: {X.shape}  |  Classes: {sorted(np.unique(y))}")
    print("  Class balance:\n", pd.Series(y).value_counts(normalize=True))
    return X, y, num, cat


COMPAS_NUMERIC_FEATURES = [
    "age", "priors_count", "juv_fel_count", "juv_misd_count",
    "juv_other_count",
]
COMPAS_CATEGORICAL_FEATURES = ["sex", "race", "c_charge_degree"]
COMPAS_TARGET = "is_recid"


def load_compass_data():
    """COMPAS recidivism - binary classification."""
    csv_path = _download_kaggle_dataset(
        "danofer/compass", "cox-violent-parsed_filt.csv",
        "compas", "COMPAS_DATA_PATH",
    )
    df = pd.read_csv(csv_path)
    print("Raw shape:", df.shape)

    cols = COMPAS_NUMERIC_FEATURES + COMPAS_CATEGORICAL_FEATURES + [COMPAS_TARGET]
    data = df[cols].copy()
    data = data[data[COMPAS_TARGET].isin([0, 1])]
    data = data.dropna().reset_index(drop=True)
    print("Cleaned shape:", data.shape)
    print("Class balance:\n", data[COMPAS_TARGET].value_counts(normalize=True))

    X = data[COMPAS_NUMERIC_FEATURES + COMPAS_CATEGORICAL_FEATURES]
    y = data[COMPAS_TARGET].astype(int).values
    return X, y, list(COMPAS_NUMERIC_FEATURES), list(COMPAS_CATEGORICAL_FEATURES)


def load_heloc_data():
    """HELOC - binary classification (RiskPerformance Good/Bad)."""
    csv_path = _download_kaggle_dataset(
        "averkiyoliabev/home-equity-line-of-creditheloc",
        "heloc_dataset_v1.csv", "heloc", "HELOC_DATA_PATH",
    )
    df = pd.read_csv(csv_path)
    print("Raw shape:", df.shape)

    target = "RiskPerformance"
    y = (df[target] == "Bad").astype(int).values
    X = df.drop(columns=[target])

    num_feats = list(X.columns)
    cat_feats = []

    X = X.replace(-9, np.nan).replace(-8, np.nan).replace(-7, np.nan)
    valid = X.notna().all(axis=1)
    X = X.loc[valid].reset_index(drop=True)
    y = y[valid.values]
    print("Cleaned shape:", X.shape)
    print("Class balance:\n", pd.Series(y).value_counts(normalize=True))
    return X, y, num_feats, cat_feats


_LAW_SCHOOL_FILENAME = "bar_pass_prediction.csv"
_LAW_SCHOOL_DATASET = "danofer/law-school-admissions-bar-passage"


def load_law_school_data():
    """Law School Admissions - binary classification (bar passage).
    Sensitive: race, gender.
    """
    print("[Law School] Loading from local CSV...")
    csv_path = _download_kaggle_dataset(
        _LAW_SCHOOL_DATASET, _LAW_SCHOOL_FILENAME,
        "law_school", "LAW_SCHOOL_DATA_PATH",
    )
    df = pd.read_csv(csv_path)
    print(f"  Raw shape: {df.shape}")

    target = "pass_bar"
    drop_cols = ["bar1", "bar1_yr", "bar2", "bar2_yr", "bar_passed", "bar",
                 "ID", "dnn_bar_pass_prediction"]

    y = df[target].astype(int).values
    X = df.drop(columns=[target] + [c for c in drop_cols if c in df.columns],
                errors="ignore")

    if "race" in X.columns:
        race_map = {1: "white", 2: "black", 3: "hispanic", 4: "asian",
                    5: "other", 6: "other", 7: "other"}
        X["race"] = X["race"].map(race_map).fillna("other")

    valid = X.notna().all(axis=1)
    X = X.loc[valid].reset_index(drop=True)
    y = y[valid.values]
    print(f"  Cleaned shape: {X.shape}")
    print("  Class balance:\n", pd.Series(y).value_counts(normalize=True))
    print(f"  Sensitive attributes: race, gender")

    num_feats = list(X.select_dtypes(include=[np.number]).columns)
    cat_feats = list(X.select_dtypes(exclude=[np.number]).columns)
    return X, y, num_feats, cat_feats


def load_acs_income_data():
    """ACS Income (2018 CA) - binary classification (>$50K).

    Requires: pip install folktables
    Sensitive: SEX, RAC1P.
    """
    import importlib
    spec = importlib.util.find_spec("folktables")
    if spec is None:
        raise ImportError(
            "folktables is required for ACS Income. "
            "Install with:  pip install folktables"
        )

    from folktables import ACSDataSource, ACSIncome

    print("[ACS Income] Downloading 2018 ACS data (California)...")
    data_source = ACSDataSource(survey_year="2018", horizon="1-Year",
                                survey="person")
    acs_data = data_source.get_data(states=["CA"], download=True)

    features, labels, _group = ACSIncome.df_to_numpy(acs_data)
    feature_names = ACSIncome.features
    X = pd.DataFrame(features, columns=feature_names)
    y = labels.astype(int)

    num_feats = list(X.columns)
    cat_feats = []

    valid = X.notna().all(axis=1)
    X = X.loc[valid].reset_index(drop=True)
    y = y[valid.values]
    print(f"  Shape: {X.shape}")
    print("  Class balance:\n", pd.Series(y).value_counts(normalize=True))
    return X, y, num_feats, cat_feats


def load_iris_data():
    """Iris - 3-class classification."""
    from sklearn.datasets import load_iris
    iris = load_iris(as_frame=True)
    return iris.data, iris.target.values, list(iris.data.columns), []


def load_diabetes_data():
    """Diabetes - regression."""
    from sklearn.datasets import load_diabetes
    diab = load_diabetes(as_frame=True)
    return diab.data, diab.target.values, list(diab.data.columns), []
