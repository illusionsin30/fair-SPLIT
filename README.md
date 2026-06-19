# fair-SPLIT

Pure-Python decision tree library implementing
**CART**, **SPLIT**, **LicketySPLIT**, and **ReSPLIT** —
a family of sparse, interpretable tree algorithms with built-in
fairness evaluation and calibration.

## Models

| Model | Description | Speed |
|---|---|---|
| **CART** | Greedy tree with Gini impurity, continuous & categorical features, per-node optimal thresholds | Fast |
| **SPLIT** | Optimal lookahead prefix (DP) + greedy leaf completion | Fast |
| **LicketySPLIT** | Polynomial-time recursive SPLIT — optimal root split evaluated by greedy-subtree completion | Fast |
| **ReSPLIT** | Rashomon set of near-optimal trees via randomised diverse prefixes | Moderate |

## Installation

```bash
conda create -n split python=3.10 -y
conda activate split
pip install -r requirements.txt

# Optional
pip install folktables          # ACS Income dataset
pip install kaggle               # Kaggle fallback (COMPAS, HELOC, Law School)
```

## Quick start

```bash
conda activate split

# CART on Adult
bash scripts/train.sh --model cart --dataset adult

# SPLIT on Bank Marketing
bash scripts/train.sh --model split --dataset bank --depth 5 --reg 0.001

# LicketySPLIT on German Credit
bash scripts/train.sh --model licketysplit --dataset german --depth 5

# ReSPLIT Rashomon set on COMPAS
bash scripts/train.sh --model resplit --dataset compass --num_prefix 30

# Fairness-aware training on ACS Income
bash scripts/train.sh --model cart --dataset acsincome --fair
bash scripts/train.sh --model split --dataset acsincome --depth 5 --reg 0.001 --fair
```

## Usage

### Common arguments

```
--dataset NAME            Dataset (required, see list below)
--model {cart,split,licketysplit,resplit}
--test_size FLOAT         Test fraction (default: 0.2)
--random_state INT        Random seed (default: 42)
--fair                    Enable fairness preprocessing + calibration
--fair_post {sample,leaf_pareto}
                          Fair postprocessing method (default: sample)
--fair_metric {dp,eo}     LPFR objective: demographic parity or equal opportunity
--fair_lambda FLOAT       LPFR fairness-gain weight (default: 1.0)
--fair_acc_budget FLOAT   Max calibration accuracy drop for LPFR (default: 0.02)
```

### CART

```bash
bash scripts/train.sh --model cart --dataset adult
bash scripts/train.sh --model cart --dataset adult --max_depth 8
bash scripts/train.sh --model cart --dataset adult --binarize_cart   # binarized features for fair comparison with SPLIT
bash scripts/train.sh --model cart --dataset acsincome --fair
```

| Flag | Default | Description |
|---|---|---|
| `--max_depth` | 6 | Maximum tree depth |
| `--min_samples_split` | 10 | Min samples to split |
| `--min_samples_leaf` | 5 | Min samples per leaf |
| `--binarize_cart` | off | Use binarized features (same space as SPLIT) |

### SPLIT

```bash
bash scripts/train.sh --model split --dataset bank --depth 5 --reg 0.001
bash scripts/train.sh --model split --dataset adult --depth 5 --lookahead 3 --reg 0.0001
bash scripts/train.sh --model split --dataset adult --depth 3 --reg 0.01       # sparser tree
bash scripts/train.sh --model split --dataset spambase --leaf_fill optimal
```

| Flag | Default | Description |
|---|---|---|
| `--depth` | 5 | Total depth budget |
| `--lookahead` | 2 | Optimal prefix depth |
| `--reg` | 0.0001 | Sparsity penalty λ per leaf |
| `--leaf_fill` | greedy | `greedy` or `optimal` |
| `--max_features` | 0 | Top-K candidate features (0 = all) |
| `--max_thresholds` | 50 | Max midpoints per numeric feature |
| `--binarizer` | gbdt | `gbdt` (stump thresholds) or `midpoint` |
| `--time_limit` | 60 | DP solver timeout (seconds) |
| `--binarize / --no-binarize` | binarize | Toggle feature binarization |

### LicketySPLIT

```bash
bash scripts/train.sh --model licketysplit --dataset adult --depth 5 --reg 0.001
bash scripts/train.sh --model licketysplit --dataset german --depth 4
```

| Flag | Default | Description |
|---|---|---|
| `--depth` | 5 | Total depth budget |
| `--lookahead` | 2 | Lookahead range (≥ 2) |

### ReSPLIT

```bash
bash scripts/train.sh --model resplit --dataset compass --num_prefix 20
bash scripts/train.sh --model resplit --dataset bank --num_prefix 50 --rashomon_bound 0.05
bash scripts/train.sh --model resplit --dataset adult --num_prefix 30 --depth 5 --lookahead 3
```

| Flag | Default | Description |
|---|---|---|
| `--num_prefix` | 20 | Randomised prefix candidates |
| `--rashomon_bound` | 0.01 | ε: keep trees with loss ≤ (1+ε)·best |
| `--depth` | 5 | Total depth budget |
| `--lookahead` | 2 | Prefix depth |

### Fairness evaluation

```bash
# Preprocessing + postprocessing
bash scripts/train.sh --model cart --dataset acsincome --fair
bash scripts/train.sh --model split --dataset adult --depth 5 --fair

# Leaf-Pareto Fair Recalibration (LPFR)
bash scripts/train.sh --model split --dataset adult --depth 5 --fair \
  --fair_post leaf_pareto --fair_metric dp --fair_acc_budget 0.03

# Only evaluate (no preprocessing)
bash scripts/train.sh --model split --dataset adult --depth 5
```

When `--fair` is enabled:
1. **Preprocessing**: sensitive columns (sex, race, etc.) are dropped from features before training.
2. **Postprocessing**: the default `--fair_post sample` method calibrates per-group hard predictions to equalize positive prediction rates (demographic parity).

`--fair_post leaf_pareto` enables **Leaf-Pareto Fair Recalibration (LPFR)**.
LPFR freezes the trained tree structure and learns leaf-level prediction overrides
on the training split used as a calibration set. It optimizes a Pareto trade-off
between fairness gain (`--fair_metric dp` or `eo`) and calibration accuracy loss
(`--fair_acc_budget`), so the deployed model remains a tree: each test sample only
needs its leaf path and does not require `y_true` or random sample-level flips.

Output includes per-group accuracy, positive prediction rate, statistical parity difference,
disparate impact ratio, and equal opportunity difference — before and after calibration.
Groups with fewer than 50 test samples are flagged `(!)` and excluded from aggregate metrics.

## Datasets

### SPLIT paper benchmarks

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
| `adult` | Adult | UCI #2 | 48K | sex, race |
| `compass` | COMPAS | Kaggle | 18K | sex, race |
| `german` | German Credit | UCI #144 | 1K | sex+status, age |
| `communities` | Communities & Crime | UCI #183 | 2K | race |
| `lawschool` | Law School Admissions | Kaggle | 20K | race, gender |
| `acsincome` | ACS Income (2018 CA) | folktables | 196K | sex, race |
| `bank` | Bank Marketing | UCI #222 | 45K | marital, age |

### Classic benchmarks

| CLI name | Dataset | Source | Samples | Task |
|---|---|---|---|---|
| `heart` | Heart Disease | UCI #45 | 303 | binary |
| `iris` | Iris | sklearn | 150 | 3-class |
| `diabetes` | Diabetes | sklearn | 442 | regression |

### Dataset prerequisites

| Datasets | Required package | First run |
|---|---|---|
| UCI (adult, bike, spambase, bank, covertype, thyroid, german, communities, heart) | `ucimlrepo` | Auto-download |
| Kaggle (compass, heloc, lawschool) | `kagglehub` or `kaggle` CLI | Auto-download |
| acsincome | `folktables` | ~200 MB download |
| iris, diabetes | `scikit-learn` | Bundled |

Sensitive attributes are auto-detected per dataset.  If a column name does
not match (e.g. ucimlrepo returns `Attribute9` instead of a descriptive name),
the fairness panel simply reports under that name — functionality is unaffected.

## Output

Training logs are saved to `results/` with the naming scheme:

```
results/{model}-{dataset}-d{depth}-{bin/raw}-{nofair/fair/lpfr}.log
```

Each log contains the model parameters, test accuracy, majority baseline,
the list of features used for splits, per-group fairness metrics, and the
full tree structure.

## Multi-seed experiments

For one experiment across several seeds:

```bash
python scripts/run_multi_seed.py \
  --seeds 0,1,2,3,4 \
  --run_name cart-adult-d5 \
  -- \
  --model cart --dataset adult --max_depth 5
```

For the full evaluation grid across several seeds:

```bash
python scripts/eval_multi_seed.py \
  --seeds 0,1,2,3,4 \
  --run_name full-eval-5seeds
```

By default, the full grid includes the `nofair` baseline, original `--fair`
sample-level calibration (`mode=fair`), and LPFR (`mode=lpfr`). To compare only
the two fair postprocessors:

```bash
python scripts/eval_multi_seed.py \
  --seeds 0,1,2,3,4 \
  --jobs 8 \
  --dataset_jobs 3 \
  --heavy_datasets acsincome \
  --heavy_dataset_jobs 1 \
  --datasets adult,bank,compass \
  --skip_nofair \
  --fair_posts sample,leaf_pareto \
  --fair_metric dp \
  --fair_acc_budget 0.03 \
  --run_name fair-vs-lpfr-5seeds
```

`--jobs` controls how many experiment/seed training processes run concurrently.
`--dataset_jobs` limits concurrent tasks for the same dataset, which avoids
multiple ACSIncome/folktables jobs downloading or loading the same large data at
once. The convenience shell script uses 32 global workers, `DATASET_JOBS=3` for
ordinary datasets, and `HEAVY_DATASET_JOBS=1` for `acsincome` by default.
Override with `JOBS=16 DATASET_JOBS=2 HEAVY_DATASET_JOBS=1 bash scripts/eval.sh`
when memory, I/O, or CPU contention is too high.

Each multi-seed run writes:

- `per_seed.json`: parsed metrics for every completed seed.
- `summary.json`: recursive mean/std/n for every numeric metric.
- `summary.csv`: flattened metrics with columns `model,dataset,mode,metric,mean,std,n`.
- `logs/`: seed-specific training logs.

Standard deviation is the sample standard deviation when at least two seeds
complete, and `0.0` for a single completed seed.

## Project structure

```
fair-SPLIT/
├── src/
│   ├── train.py              # CLI entry (python -m src.train)
│   ├── tree.py               # CART, SPLIT, LicketySPLIT, ReSPLIT
│   ├── solver.py             # DP optimal prefix solver
│   ├── builder.py            # Entropy-greedy tree builder
│   ├── evaluate.py           # Accuracy & fairness evaluation
│   ├── utils/
│   │   ├── nodes.py          # SPLITLeaf / SPLITNode
│   │   ├── binarizer.py      # NumericBinarizer + ThresholdGuessBinarizer
│   │   ├── helpers.py        # Prediction, tree export, feature listing
│   │   └── fairness.py       # Fair preprocessing + postprocessing calibration
│   └── data/
│       ├── dataload.py       # 15 dataset loaders
│       ├── process.py        # Target preprocessing
│       └── __init__.py
├── scripts/
│   └── train.sh
├── results/                  # Training logs
├── requirements.txt
├── LICENSE
└── README.md
```
