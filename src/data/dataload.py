# ============================================================
# Dataset loading — SPLIT paper (ICML 2025 Oral) benchmarks
# ============================================================
#
# UCI datasets are loaded via ucimlrepo.  Kaggle datasets use
# kagglehub with the same fallback strategy as COMPAS.
#
# Each loader returns:  X, y, numeric_features, categorical_features
# ============================================================
import os
import subprocess
import sys

import numpy as np
import pandas as pd


# ====================================================================
# UCI helper
# ====================================================================

def _fetch_uci(dataset_id, num_feats=None, cat_feats=None):
    """Load a dataset from the UCI ML Repository via ucimlrepo.

    Args:
        dataset_id: UCI dataset ID (e.g. 2 for Adult).
        num_feats: Explicit list of numeric feature names.  If None,
                   auto-detected via dtypes.
        cat_feats: Explicit list of categorical feature names.

    Returns:
        X (DataFrame), y (ndarray), num_feats (list), cat_feats (list).

    Raises
    ------
    ImportError if ucimlrepo is not installed.
    """
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError:
        raise ImportError(
            "ucimlrepo is required for UCI datasets. "
            "Install with:  pip install ucimlrepo"
        )

    dataset = fetch_ucirepo(id=dataset_id)
    X = dataset.data.features.copy()
    y = dataset.data.targets

    # Targets may be a DataFrame or Series
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]

    # Determine feature type lists
    if num_feats is None and cat_feats is None:
        num_feats = list(X.select_dtypes(include=[np.number]).columns)
        cat_feats = list(X.select_dtypes(exclude=[np.number]).columns)
    else:
        num_matched = [c for c in (num_feats or []) if c in X.columns]
        cat_matched = [c for c in (cat_feats or []) if c in X.columns]
        num_missed = [c for c in (num_feats or []) if c not in X.columns]
        cat_missed = [c for c in (cat_feats or []) if c not in X.columns]
        if num_missed or cat_missed:
            print(f"  WARNING: feature name mismatch!")
            if num_missed:
                print(f"    Numeric not found: {num_missed}")
            if cat_missed:
                print(f"    Categorical not found: {cat_missed}")
            print(f"    Available columns: {list(X.columns)}")
        num_feats = num_matched
        cat_feats = cat_matched
        # Any remaining columns become numeric by default
        used = set(num_feats) | set(cat_feats)
        extra = [c for c in X.columns if c not in used]
        if extra:
            print(f"  Unmatched columns → treated as numeric: {extra}")
        num_feats = num_feats + extra

    # Drop rows with missing values
    n_before = len(X)
    y = y.loc[X.index]
    valid = X.notna().all(axis=1) & y.notna()
    X = X.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)
    if len(X) < n_before:
        print(f"  Dropped {n_before - len(X)} rows with missing values.")

    print(f"  Shape: {X.shape}  |  Classes: {sorted(np.unique(y))}")
    return X, y.values, num_feats, cat_feats


# ====================================================================
# Kaggle helper  (shared with COMPAS)
# ====================================================================

_KAGGLEHUB_AVAILABLE = None

def _check_kagglehub():
    global _KAGGLEHUB_AVAILABLE
    if _KAGGLEHUB_AVAILABLE is None:
        try:
            import kagglehub  # noqa: F401
            _KAGGLEHUB_AVAILABLE = True
        except ImportError:
            _KAGGLEHUB_AVAILABLE = False
    return _KAGGLEHUB_AVAILABLE


def _download_kaggle_dataset(dataset_slug, filename, local_dir, env_var):
    """Download a Kaggle dataset, with local-cache and fallback.

    Args:
        dataset_slug: e.g. "danofer/compass".
        filename: Name of the CSV file to locate.
        local_dir: Subdirectory under data/ for the CLI fallback.
        env_var: Environment variable for a manual local path override.

    Returns:
        Absolute path to the CSV file.
    """
    # 1. Environment variable
    env_path = os.environ.get(env_var)
    if env_path and os.path.isfile(env_path):
        print(f"Found local data via ${env_var}: {env_path}")
        return env_path

    # 2. Kagglehub cache
    cache_base = os.path.expanduser("~/.cache/kagglehub/datasets")
    dataset_cache = os.path.join(
        cache_base, dataset_slug.replace("/", os.sep)
    )
    if os.path.isdir(dataset_cache):
        for root, _dirs, files in os.walk(dataset_cache):
            if filename in files:
                return os.path.join(root, filename)

    # 3. Local project directory
    local = os.path.join("data", local_dir, filename)
    if os.path.isfile(local):
        print(f"Found local data: {local}")
        return local

    # 4. kagglehub download
    if _check_kagglehub():
        try:
            import kagglehub
            path = kagglehub.dataset_download(dataset_slug)
            print(f"Downloaded to: {path}")
            return os.path.join(path, filename)
        except Exception as exc:
            print(f"[WARN] kagglehub download failed: {exc}")

    # 5. Kaggle CLI fallback
    target = os.path.join("data", local_dir)
    try:
        os.makedirs(target, exist_ok=True)
        subprocess.run(
            [sys.executable, "-m", "kaggle", "datasets", "download",
             dataset_slug, "-p", target, "--unzip"],
            check=True,
        )
        return os.path.join(target, filename)
    except Exception as exc:
        print(f"[WARN] kaggle CLI download failed: {exc}")

    raise RuntimeError(
        f"Cannot obtain {filename} from {dataset_slug}.\n"
        f"Options:\n"
        f"  1. pip install kagglehub\n"
        f"  2. pip install kaggle && kaggle datasets download {dataset_slug}\n"
        f"  3. Download {filename} manually and set {env_var}=/path/to/file"
    )


# ====================================================================
# Individual dataset loaders
# ====================================================================

# ---------- Adult (UCI #2) ----------
ADULT_NUMERIC = [
    "age", "fnlwgt", "education-num", "capital-gain", "capital-loss",
    "hours-per-week",
]
ADULT_CATEGORICAL = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country",
]


def load_adult_data():
    """Adult / Census Income dataset — binary classification.

    Predict whether income exceeds $50K/yr.
    """
    print("[Adult] Loading from UCI repository...")
    X, y, num, cat = _fetch_uci(2, num_feats=ADULT_NUMERIC,
                                cat_feats=ADULT_CATEGORICAL)

    # Clean target: strip trailing dots and whitespace (UCI Adult quirk)
    if y.dtype.kind in ("U", "S", "O"):
        y = np.array([str(v).strip().rstrip(".") for v in y])
        # Binarize: <=50K → 0, >50K → 1
        y = (y == ">50K").astype(int)
    print("  Class balance:\n", pd.Series(y).value_counts(normalize=True))

    return X, y, num, cat


# ---------- Bike (UCI #275) ----------
BIKE_NUMERIC = ["temp", "atemp", "hum", "windspeed", "casual", "registered"]
BIKE_CATEGORICAL = [
    "season", "yr", "mnth", "hr", "holiday", "weekday",
    "workingday", "weathersit",
]
BIKE_DROP = ["instant", "dteday"]


def load_bike_data():
    """Bike Sharing dataset — regression (cnt).

    Predict hourly bike rental count.
    """
    print("[Bike] Loading from UCI repository...")
    X, y, num, cat = _fetch_uci(275, num_feats=BIKE_NUMERIC,
                                cat_feats=BIKE_CATEGORICAL)

    # ucimlrepo returns targets as a 3-col DataFrame (casual, registered, cnt).
    # Pick 'cnt' as the target.
    if hasattr(y, 'ndim') and y.ndim == 2:
        y = y[:, 2] if y.shape[1] == 3 else y[:, -1]

    # Drop 'instant' and 'dteday' (non-predictive identifiers).
    drop_cols = [c for c in BIKE_DROP if c in X.columns]
    X = X.drop(columns=drop_cols, errors="ignore")
    num = [c for c in num if c in X.columns]
    cat = [c for c in cat if c in X.columns]

    print(f"  Target (cnt): mean={y.mean():.1f}, std={y.std():.1f}")
    return X, y, num, cat


# ---------- Spambase (UCI #94) ----------
SPAMBASE_NUMERIC = [f"word_freq_{i}" for i in range(48)] + \
                   [f"char_freq_{i}" for i in range(6)] + \
                   ["capital_run_length_average",
                    "capital_run_length_longest",
                    "capital_run_length_total"]


def load_spambase_data():
    """Spambase — binary classification (spam / not-spam).

    All 57 features are numeric (word/char frequencies).
    """
    print("[Spambase] Loading from UCI repository...")
    X, y, num, cat = _fetch_uci(94)

    # All features are numeric; target is already 0/1
    y = y.astype(int)
    print("  Class balance:\n", pd.Series(y).value_counts(normalize=True))
    return X, y, num, cat


# ---------- Bank Marketing (UCI #222) ----------
BANK_NUMERIC = ["age", "balance", "day", "duration", "campaign",
                "pdays", "previous"]
BANK_CATEGORICAL = ["job", "marital", "education", "default", "housing",
                    "loan", "contact", "month", "poutcome"]


def load_bank_data():
    """Bank Marketing — binary classification.

    Predict whether a client subscribes a term deposit.
    """
    print("[Bank] Loading from UCI repository...")
    X, y, num, cat = _fetch_uci(222, num_feats=BANK_NUMERIC,
                                cat_feats=BANK_CATEGORICAL)

    # Target 'y' → 0/1
    if y.dtype.kind in ("U", "S", "O"):
        y = (y == "yes").astype(int)
    print("  Class balance:\n", pd.Series(y).value_counts(normalize=True))
    return X, y, num, cat


# ---------- Covertype (UCI #31) ----------
COVERTYPE_NUMERIC = [
    "Elevation", "Aspect", "Slope",
    "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
]
# Wilderness areas and soil types are binary indicators —
# we keep them as numeric since they're already 0/1.
COVERTYPE_TARGET = "Cover_Type"


def load_covertype_data():
    """Covertype — 7-class classification.

    Predict forest cover type from cartographic variables.
    The 44 wilderness/soil-type binary columns are treated as numeric.
    """
    print("[Covertype] Loading from UCI repository...")
    X, y, num, cat = _fetch_uci(31)

    # Remap cover type from 1..7 → 0..6
    y = y.astype(int) - 1
    print("  Class balance:\n", pd.Series(y).value_counts(normalize=True))
    return X, y, num, cat


# ---------- Thyroid Disease (UCI #102) ----------

def load_thyroid_data():
    """Thyroid Disease — 3-class (or more) classification.

    The UCI thyroid dataset has been widely used in decision-tree papers.
    We select the 'ann-train' variant (the full 21-feature set).
    """
    print("[Thyroid] Loading from UCI repository...")
    X, y, num, cat = _fetch_uci(102)

    if y.dtype.kind in ("U", "S", "O"):
        # Map string labels to 0..K-1
        unique_labels = sorted(np.unique(y))
        mapping = {lab: i for i, lab in enumerate(unique_labels)}
        print(f"  Label mapping: {mapping}")
        y = np.array([mapping[v] for v in y])

    y = y.astype(int)
    print("  Class balance:\n", pd.Series(y).value_counts(normalize=True))
    return X, y, num, cat


# ---------- COMPAS (Kaggle) ----------
COMPAS_NUMERIC_FEATURES = [
    "age", "priors_count", "juv_fel_count", "juv_misd_count",
    "juv_other_count",
]
COMPAS_CATEGORICAL_FEATURES = ["sex", "race", "c_charge_degree"]
COMPAS_TARGET = "is_recid"


def load_compass_data():
    """COMPAS recidivism — binary classification."""
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


# ---------- HELOC (Kaggle) ----------

def load_heloc_data():
    """HELOC (Home Equity Line of Credit) — binary classification.

    Predict RiskPerformance (Good/Bad).
    """
    csv_path = _download_kaggle_dataset(
        "averkiyoliabev/home-equity-line-of-creditheloc",
        "heloc_dataset_v1.csv", "heloc", "HELOC_DATA_PATH",
    )
    df = pd.read_csv(csv_path)
    print("Raw shape:", df.shape)

    target = "RiskPerformance"
    y = (df[target] == "Bad").astype(int).values
    X = df.drop(columns=[target])

    # All features are numeric in HELOC
    num_feats = list(X.columns)
    cat_feats = []

    # Drop rows with special negative values (HEOC encoding for missing)
    X = X.replace(-9, np.nan).replace(-8, np.nan).replace(-7, np.nan)
    valid = X.notna().all(axis=1)
    X = X.loc[valid].reset_index(drop=True)
    y = y[valid.values]
    print("Cleaned shape:", X.shape)
    print("Class balance:\n", pd.Series(y).value_counts(normalize=True))
    return X, y, num_feats, cat_feats


# ---------- German Credit (UCI #144) ----------
GERMAN_NUMERIC = [
    "Duration", "Credit amount", "Installment rate",
    "Present residence since", "Age", "Number of existing credits",
    "Number of people being liable to provide maintenance for",
]
GERMAN_CATEGORICAL = [
    "Status of existing checking account", "Credit history",
    "Purpose", "Savings account/bonds",
    "Present employment since", "Personal status and sex",
    "Other debtors / guarantors", "Property",
    "Other installment plans", "Housing",
    "Job", "Telephone", "foreign worker",
]
GERMAN_TARGET = None  # last column = credit risk


def load_german_credit_data():
    """German Credit (Statlog) — binary classification.

    Predict credit risk (good / bad).  Widely used in fairness research
    with 'Personal status and sex' (gender) and 'Age' as sensitive attributes.
    """
    print("[German Credit] Loading from UCI repository...")
    X, y, num, cat = _fetch_uci(144, num_feats=GERMAN_NUMERIC,
                                cat_feats=GERMAN_CATEGORICAL)
    y = y.astype(int) - 1  # labels are 1/2 → 0/1
    print("  Class balance:\n", pd.Series(y).value_counts(normalize=True))
    return X, y, num, cat


# ---------- Communities and Crime (UCI #183) ----------
_COMMUNITIES_DROP = [
    "state", "county", "community", "communityname", "fold",
]


def load_communities_data():
    """Communities and Crime — binary classification (high / low violent crime).

    Predict whether the per-capita violent crime rate is above the median.
    Commonly used in fairness research with 'racepctblack' as sensitive attribute.
    """
    print("[Communities & Crime] Loading from UCI repository...")
    X, y, num, cat = _fetch_uci(183)

    # The target column from ucimlrepo may be 'ViolentCrimesPerPop'.
    # Binarize at median.
    threshold = float(np.median(y))
    print(f"  Binarizing violent-crime rate at median = {threshold:.4f}")
    y = (y >= threshold).astype(int)

    # Drop non-predictive string columns
    drop = [c for c in _COMMUNITIES_DROP if c in X.columns]
    X = X.drop(columns=drop, errors="ignore")
    num = [c for c in num if c in X.columns]
    cat = [c for c in cat if c in X.columns]

    # Drop any remaining non-numeric columns that couldn't be parsed
    non_num = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_num:
        X = X.drop(columns=non_num)
        cat = [c for c in cat if c in X.columns]
        print(f"  Dropped non-numeric columns: {non_num}")

    print(f"  Shape: {X.shape}  |  Class balance:")
    print(pd.Series(y).value_counts(normalize=True))
    return X, y, num, cat


# ---------- Heart Disease (UCI #45) ----------

def load_heart_data():
    """Heart Disease (Cleveland) — binary classification.

    Predict presence of heart disease.  Features include age, sex, chest
    pain type, cholesterol, etc.
    """
    print("[Heart Disease] Loading from UCI repository...")
    X, y, num, cat = _fetch_uci(45)

    # Target values > 0 indicate disease → binary
    y = (y > 0).astype(int)
    print("  Class balance:\n", pd.Series(y).value_counts(normalize=True))
    return X, y, num, cat


# ---------- Law School Admissions (folktables / fairlearn) ----------
_LAW_SCHOOL_URL = (
    "https://raw.githubusercontent.com/"
    "propublica/compas-analysis/master/"
    "compas-scores-two-years.csv"
)

# Law School dataset is available via fairlearn or a direct CSV.
# We use a lightweight bundled approach — look for it locally first.
_LAW_SCHOOL_FILENAME = "law_school_clean.csv"


def load_law_school_data():
    """Law School Admissions — binary classification (bar passage).

    Predict whether a law student passes the bar exam.
    Sensitive attributes: race, gender.

    Requires the dataset to be downloaded manually (not redistributable).
    Set LAW_SCHOOL_DATA_PATH to the CSV location, or place
    'law_school_clean.csv' under data/law_school/.

    Source: https://github.com/fairlearn/fairlearn/tree/main/test/unit/data
    """
    import urllib.request

    print("[Law School] Checking for dataset...")

    # 1. Environment variable
    env_path = os.environ.get("LAW_SCHOOL_DATA_PATH")
    if env_path and os.path.isfile(env_path):
        csv_path = env_path

    # 2. Local data directory
    local = os.path.join("data", "law_school", _LAW_SCHOOL_FILENAME)
    if not (env_path and os.path.isfile(env_path)):
        if os.path.isfile(local):
            csv_path = local

    # 3. Attempt download from fairlearn repository mirror
    url = (
        "https://raw.githubusercontent.com/fairlearn/fairlearn/"
        "main/test/unit/data/law_school_clean.csv"
    )
    try:
        os.makedirs(os.path.dirname(local), exist_ok=True)
        urllib.request.urlretrieve(url, local)
        csv_path = local
        print(f"  Downloaded to {local}")
    except Exception:
        raise FileNotFoundError(
            "Law School dataset not found.\n\n"
            "Options:\n"
            f"  1. Place {_LAW_SCHOOL_FILENAME} under data/law_school/\n"
            f"  2. Set LAW_SCHOOL_DATA_PATH=/path/to/{_LAW_SCHOOL_FILENAME}\n"
            f"  3. Download from: {url}\n"
        )

    df = pd.read_csv(csv_path)
    print(f"  Raw shape: {df.shape}")

    target = "pass_bar"
    sensitive = ["race", "gender"]
    drop_cols = ["admit", "bar1", "bar2", "bar_passed"]  # leaky columns

    y = df[target].astype(int).values
    X = df.drop(columns=[target] + [c for c in drop_cols if c in df.columns],
                errors="ignore")

    num_feats = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_feats = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # Drop rows with missing values
    valid = X.notna().all(axis=1)
    X = X.loc[valid].reset_index(drop=True)
    y = y[valid.values]
    print(f"  Cleaned shape: {X.shape}")
    print("  Class balance:\n", pd.Series(y).value_counts(normalize=True))
    print(f"  Sensitive attributes: {[s for s in sensitive if s in X.columns]}")
    return X, y, num_feats, cat_feats


# ---------- ACS Income (folktables) ----------

def load_acs_income_data():
    """ACS Income — binary classification (income > $50K).

    American Community Survey data, a modern and larger alternative to Adult.
    Requires: pip install folktables

    Sensitive attributes: race, sex, age.
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

    # Use the ACSIncome task definition from folktables
    features, labels, _group = ACSIncome.df_to_numpy(acs_data)

    # Build a clean DataFrame
    feature_names = ACSIncome.features
    X = pd.DataFrame(features, columns=feature_names)
    y = labels.astype(int)

    # All ACSIncome features are numeric
    num_feats = list(X.columns)
    cat_feats = []

    valid = X.notna().all(axis=1)
    X = X.loc[valid].reset_index(drop=True)
    y = y[valid.values]
    print(f"  Shape: {X.shape}")
    print("  Class balance:\n", pd.Series(y).value_counts(normalize=True))
    return X, y, num_feats, cat_feats


# ---------- Iris (sklearn) ----------
def load_iris_data():
    """Iris — 3-class classification."""
    from sklearn.datasets import load_iris

    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target.values
    return X, y, list(X.columns), []


# ---------- Diabetes (sklearn) ----------
def load_diabetes_data():
    """Diabetes — regression."""
    from sklearn.datasets import load_diabetes

    diab = load_diabetes(as_frame=True)
    X = diab.data
    y = diab.target.values
    return X, y, list(X.columns), []
