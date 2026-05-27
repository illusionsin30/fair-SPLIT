# fair-SPLIT

Final project of THU 2026-spring course *Statistical Machine Learning*.
Pure-Python replication of the
[SPLIT](https://github.com/VarunBabbar/SPLIT-ICML) family of decision tree
algorithms (ICML 2025 Oral).

## Models

| Model | Paper | Description |
|---|---|---|
| **CART** | — | Greedy tree with Gini impurity. Continuous features with per-node optimal thresholds. Multiclass & regression. |
| **SPLIT** | Algorithm 2 | Optimal shallow prefix (DP + greedy-boundary completion) + leaf filling. |
| **LicketySPLIT** | Algorithm 3 | Polynomial-time recursive SPLIT. Evaluates optimal root split by greedy-completion of subtrees, recurses. |
| **ReSPLIT** | §5.3 | Rashomon set of near-optimal trees via randomised diverse prefixes + greedy fill. |

All models are **pure Python** — no C++ compilation, no GOSDT dependency.

## Project structure

```
fair-SPLIT/
├── src/
│   ├── train.py              # CLI entry point (python -m src.train)
│   ├── tree.py               # CART, SPLIT, LicketySPLIT, ReSPLIT
│   ├── solver.py             # DP optimal prefix + greedy-boundary completion
│   ├── builder.py            # Entropy-greedy tree builder
│   ├── evaluate.py           # Accuracy evaluation
│   ├── utils/
│   │   ├── nodes.py          # SPLITLeaf / SPLITNode
│   │   ├── binarizer.py      # NumericBinarizer + ThresholdGuessBinarizer
│   │   └── helpers.py        # Prediction, tree export, feature listing
│   └── data/
│       ├── dataload.py       # 15 dataset loaders
│       └── process.py        # Target preprocessing
├── scripts/
│   └── train.sh              # Convenience launcher
├── requirements.txt
└── README.md
```

## Installation

```bash
conda create -n split python=3.10 -y
conda activate split
pip install -r requirements.txt

# Optional
pip install folktables          # for ACS Income dataset
pip install kaggle               # for COMPAS/HELOC (kagglehub fallback)
```

## Quick start

```bash
conda activate split

# CART — raw continuous features, Gini impurity
bash scripts/train.sh --model cart --dataset adult

# SPLIT — GBDT-threshold binarization, DP prefix + greedy fill
bash scripts/train.sh --model split --dataset bank --depth 5 --reg 0.001

# LicketySPLIT — polynomial-time recursive SPLIT
bash scripts/train.sh --model licketysplit --dataset adult --depth 5 --reg 0.001

# ReSPLIT — Rashomon set with diverse randomised prefixes
bash scripts/train.sh --model resplit --dataset compass --num_prefix 30
```

## Model-specific usage

### CART

```bash
bash scripts/train.sh --model cart --dataset adult \
    --max_depth 6 --min_samples_split 10 --min_samples_leaf 5

# Fair comparison with SPLIT: binarize CART's features too
bash scripts/train.sh --model cart --dataset adult --binarize_cart
```

**CART parameters**

| Flag | Default | Description |
|---|---|---|
| `--max_depth` | 6 | Maximum tree depth |
| `--min_samples_split` | 10 | Minimum samples to consider a split |
| `--min_samples_leaf` | 5 | Minimum samples per leaf |
| `--binarize_cart` | off | Binarize features like SPLIT (fair comparison) |

CART works on **raw continuous and categorical features** with per-node
optimal threshold search.  Use `--binarize_cart` to force it onto the same
feature space as SPLIT for a fair accuracy comparison.

### SPLIT  (Algorithm 2)

```bash
bash scripts/train.sh --model split --dataset adult \
    --depth 5 --lookahead 2 --reg 0.001

# Deeper lookahead = more optimal search (slower)
bash scripts/train.sh --model split --dataset bank \
    --depth 5 --lookahead 3 --reg 0.0001

# Optimal leaf filling (Phase 2 with DP, not greedy)
bash scripts/train.sh --model split --dataset spambase \
    --depth 4 --lookahead 2 --leaf_fill optimal
```

**SPLIT parameters**

| Flag | Default | Description |
|---|---|---|
| `--depth` | 5 | Total depth budget *d* |
| `--lookahead` | 2 | Lookahead depth *dl* for optimal prefix |
| `--reg` | 0.0001 | Sparsity penalty λ per leaf |
| `--leaf_fill` | greedy | Leaf completion: `greedy` or `optimal` |
| `--max_features` | 0 | Top‑*K* candidate features in DP (0 = all) |
| `--max_thresholds` | 50 | Max midpoints per numeric feature |
| `--binarizer` | gbdt | Feature binarization: `gbdt` (stump thresholds) or `midpoint` |
| `--time_limit` | 60 | DP solver time limit (seconds) |

**How SPLIT differs from the paper**

Our implementation replaces GOSDT (C++ branch-and-bound) with a pure-Python
DP solver.  At the lookahead boundary (`depth == 0`), the DP uses greedy
completion to evaluate subproblems, matching the paper's Equation 4 branch ②
and Algorithm 1.  Phase 2 fills leaves with either greedy or optimal (DP)
trees.  The paper uses GOSDT for both phases — our DP is algorithmically
equivalent but searches fewer subproblems per second.

### LicketySPLIT  (Algorithm 3)

```bash
bash scripts/train.sh --model licketysplit --dataset adult \
    --depth 5 --reg 0.001

# On a smaller, faster dataset
bash scripts/train.sh --model licketysplit --dataset german \
    --depth 4 --reg 0.001
```

**LicketySPLIT parameters**

| Flag | Default | Description |
|---|---|---|
| `--depth` | 5 | Total depth budget |
| `--lookahead` | 2 | Lookahead range (≥ 2) |
| `--reg` | 0.0001 | Sparsity penalty λ per leaf |

LicketySPLIT is **polynomial-time** — O(n·k²·d²) per Theorem 6.4.  At each
recursion step it evaluates every candidate feature by building a greedy
subtree at depth *d*‑1 for both children, then picks the best.  It's often
the best trade-off between speed and accuracy among our four models.

### ReSPLIT  (§5.3)

```bash
bash scripts/train.sh --model resplit --dataset compass \
    --num_prefix 30 --rashomon_bound 0.02

# Larger Rashomon set
bash scripts/train.sh --model resplit --dataset bank \
    --num_prefix 50 --rashomon_bound 0.05 --depth 5 --lookahead 3
```

**ReSPLIT parameters**

| Flag | Default | Description |
|---|---|---|
| `--num_prefix` | 20 | Number of randomised prefix candidates |
| `--rashomon_bound` | 0.01 | ε threshold: keep trees with loss ≤ (1+ε)·L* |
| `--depth` | 5 | Total depth budget |
| `--lookahead` | 2 | Prefix depth |
| `--reg` | 0.0001 | Sparsity penalty λ per leaf |

Diversity comes from **shuffled feature evaluation order** with
`pick_first` mode: each prefix candidate picks the first feature with
positive entropy gain in its shuffled order, producing structurally
different trees.  Deduplication removes identical trees; the Rashomon
threshold filters out those whose regularized loss is above the bound.

**ReSPLIT differs from the paper** in that we enumerate prefixes via
randomised greedy induction rather than TreeFARMS' exact Rashomon-set
prefix enumeration.  This trades runtime and completeness for simplicity.

## CLI reference

### Shared data parameters

```
--dataset {adult,bike,spambase,bank,covertype,compass,heloc,thyroid,
           german,communities,heart,lawschool,acsincome,iris,diabetes}
--test_size FLOAT        Test fraction (default: 0.2)
--random_state INT       Random seed (default: 42)
```

### Shared SPLIT / ReSPLIT / LicketySPLIT parameters

```
--binarizer {gbdt,midpoint}   Binarization method (default: gbdt)
--binarize / --no-binarize    Binarize features (default: --binarize)
--max_features INT            Top-K features in DP/builder (0=all, default: 0)
--max_thresholds INT          Max midpoints per numeric feature (default: 50)
--time_limit INT              DP timeout in seconds (default: 60)
```

## Datasets

### SPLIT paper benchmarks (ICML 2025)

| CLI name | Dataset | Source | Samples | Task |
|---|---|---|---|---|
| `adult` | Adult / Census Income | UCI #2 | 48K | binary |
| `bike` | Bike Sharing | UCI #275 | 17K | regression |
| `spambase` | Spambase | UCI #94 | 4.6K | binary |
| `bank` | Bank Marketing | UCI #222 | 45K | binary |
| `covertype` | Covertype | UCI #31 | 581K | 7-class |
| `compass` | COMPAS Recidivism | Kaggle | 18K | binary |
| `heloc` | HELOC | Kaggle | 10K | binary |
| `thyroid` | Thyroid Disease | UCI #102 | 7K | multiclass |

### Fairness benchmarks

| CLI name | Dataset | Source | Samples | Sensitive attributes |
|---|---|---|---|---|
| `german` | German Credit | UCI #144 | 1K | gender, age |
| `communities` | Communities & Crime | UCI #183 | 2K | race |
| `lawschool` | Law School Admissions | fairlearn | 20K | race, gender |
| `acsincome` | ACS Income (2018 CA) | folktables | 195K | race, sex |

### Classic benchmarks

| CLI name | Dataset | Source | Samples | Task |
|---|---|---|---|---|
| `heart` | Heart Disease | UCI #45 | 303 | binary |
| `iris` | Iris | sklearn | 150 | 3-class |
| `diabetes` | Diabetes | sklearn | 442 | regression |

### Dataset prerequisites

| Datasets | Package | Notes |
|---|---|---|
| UCI datasets (adult, bike, …) | `ucimlrepo` | Auto-download |
| compass, heloc | `kagglehub` or `kaggle` CLI | Auto-download from Kaggle |
| lawschool | none | Auto-download from fairlearn mirror; override with `$LAW_SCHOOL_DATA_PATH` |
| acsincome | `folktables` | ~200 MB download |
| iris, diabetes | `scikit-learn` | Bundled |

## Examples

### Quick comparisons

```bash
# CART (raw features) vs SPLIT (binarized features)
bash scripts/train.sh --model cart  --dataset adult
bash scripts/train.sh --model split --dataset adult --depth 5 --reg 0.001

# Fair comparison: both on binarized features
bash scripts/train.sh --model cart  --dataset adult --binarize_cart
bash scripts/train.sh --model split --dataset adult --depth 5 --reg 0.001

# SPLIT vs LicketySPLIT
bash scripts/train.sh --model split --dataset bank --depth 5 --lookahead 2
bash scripts/train.sh --model licketysplit --dataset bank --depth 5
```

### Rashomon set exploration

```bash
# Small Rashomon set
bash scripts/train.sh --model resplit --dataset compass \
    --num_prefix 20 --rashomon_bound 0.01

# Large and diverse
bash scripts/train.sh --model resplit --dataset bank \
    --num_prefix 50 --rashomon_bound 0.05 --depth 5 --lookahead 3
```

### Sparsity tuning

```bash
# Very sparse (high λ)
bash scripts/train.sh --model split --dataset adult --reg 0.01 --depth 3

# Dense / accurate (low λ)  
bash scripts/train.sh --model split --dataset adult --reg 0.0001 --depth 5
```

### Python API

```bash
python -m src.train --model cart --dataset adult
python -m src.train --model split --dataset bank --depth 5 --reg 0.001
python -m src.train --model licketysplit --dataset adult --depth 5
python -m src.train --model resplit --dataset compass --num_prefix 30
```

## Data preprocessing

All models share the **same train/test split** for direct comparison.

- **CART** (default): Receives raw features (numeric + categorical) and
  handles them natively via Gini-based threshold and category-subset splits.
- **CART** (`--binarize_cart`): Same binarized features as SPLIT for fair
  comparison.
- **SPLIT / LicketySPLIT / ReSPLIT**: Features are binarized via GBDT
  stump threshold guessing (`--binarizer gbdt`, default) or midpoint
  binarization (`--binarizer midpoint`).  Multiclass targets are kept
  as-is; regression targets are binarized at the median.

## Differences from the original paper

| Aspect | Paper (C++ GOSDT) | Our implementation |
|---|---|---|
| Optimal prefix search | GOSDT branch-and-bound + similar-support bounds + caching | Pure-Python DP with greedy-boundary completion |
| Leaf filling (Phase 2) | GOSDT optimal subtree | Greedy or DP (set by `--leaf_fill`) |
| Binarization | GBDT stump thresholds + column elimination | GBDT stump thresholds (column elimination disabled by default) |
| ReSPLIT prefix enumeration | TreeFARMS exact Rashomon-set enumeration | Randomised greedy prefixes with `pick_first` diversity |
| CART baseline | Greedy on binarized features | Greedy on raw continuous features (use `--binarize_cart` for parity) |

These trade-offs make our implementation **slower at optimal search**
but **algorithmically faithful** to the paper's objective function
(Equation 4 with greedy-completion at the lookahead boundary).
