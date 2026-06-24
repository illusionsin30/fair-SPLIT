# fair-SPLIT

Pure-Python decision tree library for **CART**, **SPLIT**, **LicketySPLIT**, and **ReSPLIT**.
The project focuses on sparse, interpretable trees with built-in fairness evaluation and calibration.

## Overview

Decision trees are easy to inspect because every prediction follows a concrete path of feature tests.
This repository provides:

- `CART`: classic greedy decision trees for classification and regression
- `SPLIT`: optimal shallow prefixes plus greedy or optimal leaf filling
- `LicketySPLIT`: polynomial-time recursive SPLIT
- `ReSPLIT`: a diverse set of candidate trees built from multiple prefixes

## What Is a Decision Tree?

A decision tree has two kinds of nodes:

- Internal nodes store a split test such as `age <= 30`.
- Leaf nodes store the final prediction.

Two properties matter most for interpretability:

- Sparsity: fewer leaves usually make the tree easier to read.
- Depth: deeper trees can fit more complex boundaries, but are harder to understand.

## SPLIT Family

### Core Idea

SPLIT only searches optimally near the root. Lower levels are completed greedily or with another optimal pass.
This keeps the model compact while avoiding the full cost of exact tree search everywhere.

### Objective

For binary classification, the regularized objective is:

```text
loss(T) = error_rate(T) + lambda * num_leaves(T)
```

Here `lambda` controls the trade-off between accuracy and sparsity.

### Binarization

SPLIT works on binary features.

- Numeric features are converted into threshold tests.
- Categorical features are one-hot encoded.

The repository supports both `midpoint` thresholds and `gbdt` stump threshold guessing.

### Execution Flow

- Phase 1: build an optimal prefix with dynamic programming.
- Phase 2: fill the remaining leaves with either greedy subtrees or optimal subtrees.

### ReSPLIT

ReSPLIT builds a diverse set of candidate trees by generating multiple prefixes and keeping the trees that stay within a small score range of the best objective value.

## Fairness Evaluation and Calibration

The project supports fairness-aware preprocessing and postprocessing.

### Metrics

| Metric | Meaning | Ideal |
|---|---|---|
| Statistical Parity Difference | Difference between the highest and lowest positive prediction rate across groups | Close to 0 |
| Disparate Impact Ratio | Lowest positive prediction rate divided by the highest | Close to 1 |
| Equal Opportunity Difference | Difference between the highest and lowest TPR across groups | Close to 0 |

### Fairness Workflow

Enable fairness with `--fair`.

1. Preprocessing drops sensitive columns before training.
2. Postprocessing calibrates group-level predictions.

`--fair_post sample` performs sample-level calibration.
`--fair_post leaf_pareto` enables Leaf-Pareto Fair Recalibration (LPFR), which keeps the tree structure fixed and adjusts leaf-level outputs.

## CART Baseline

CART is included as a baseline for comparison.

- It works directly on raw numeric and categorical features.
- It uses Gini impurity for classification and variance for regression.
- `--binarize_cart` lets CART operate in the same binary feature space as SPLIT.

## Data Pipeline

The full pipeline is:

```text
raw dataset -> preprocessing -> train/test split -> model training -> evaluation
```

The data loaders cover UCI, Kaggle, folktables, and sklearn datasets.
Sensitive attributes are detected per dataset and passed to the fairness evaluation code.

## Model Comparison

| Model | Feature space | Search strategy | Output |
|---|---|---|---|
| CART | Raw features | Pure greedy | One tree |
| SPLIT | Binary features | Optimal prefix + completion | One tree |
| LicketySPLIT | Binary features | Recursive optimal root split | One tree |
| ReSPLIT | Binary features | Diverse prefixes + completion | A set of trees |

Typical starting points:

- Fast baseline: `--model cart`
- Balanced trade-off: `--model split --depth 5 --lookahead 2 --reg 0.001`
- Sparse tree: `--model split --depth 3 --lookahead 2 --reg 0.01`
- Higher accuracy: `--model split --depth 6 --lookahead 3 --reg 0.0001`
- Diversity study: `--model resplit --num_prefix 30`

## Project Structure

```text
fair-SPLIT/
src/
  train.py
  tree.py
  solver.py
  builder.py
  evaluate.py
  utils/
    nodes.py
    binarizer.py
    helpers.py
    fairness.py
  data/
    dataload.py
    process.py
    __init__.py
scripts/
  train.sh
  eval.sh
  run_multi_seed.py
  eval_multi_seed.py
results/
requirements.txt
LICENSE
README.md
README_zh.md
```

## Supported Datasets

### Core Benchmarks

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

### Fairness Benchmarks

| CLI name | Dataset | Source | Samples | Sensitive attributes |
|---|---|---|---|---|
| `adult` | Adult | UCI #2 | 48K | sex, race |
| `compass` | COMPAS | Kaggle | 18K | sex, race |
| `german` | German Credit | UCI #144 | 1K | sex+status, age |
| `communities` | Communities & Crime | UCI #183 | 2K | race |
| `lawschool` | Law School Admissions | Kaggle | 20K | race, gender |
| `acsincome` | ACS Income (2018 CA) | folktables | 196K | sex, race |
| `bank` | Bank Marketing | UCI #222 | 45K | marital, age |

### Classic Benchmarks

| CLI name | Dataset | Source | Samples | Task |
|---|---|---|---|---|
| `heart` | Heart Disease | UCI #45 | 303 | binary |
| `iris` | Iris | sklearn | 150 | 3-class |
| `diabetes` | Diabetes | sklearn | 442 | regression |

### Dataset Requirements

| Datasets | Required package | First run |
|---|---|---|
| UCI datasets | `ucimlrepo` | Auto-download |
| Kaggle datasets | `kagglehub` or `kaggle` CLI | Auto-download |
| `acsincome` | `folktables` | ~200 MB download |
| `iris`, `diabetes` | `scikit-learn` | Bundled |

## Install and Run

### Environment Setup

```bash
conda create -n split python=3.10 -y
conda activate split
pip install -r requirements.txt
```

Optional dependencies:

```bash
pip install folktables
pip install kaggle
```

### Basic Usage

All training goes through `bash scripts/train.sh`:

```bash
bash scripts/train.sh --model <model> --dataset <dataset> [options]
```

### Example Runs

```bash
bash scripts/train.sh --model cart --dataset adult
bash scripts/train.sh --model split --dataset bank --depth 5 --reg 0.001
bash scripts/train.sh --model licketysplit --dataset german --depth 5
bash scripts/train.sh --model resplit --dataset compass --num_prefix 30
bash scripts/train.sh --model cart --dataset acsincome --fair
bash scripts/train.sh --model split --dataset acsincome --depth 5 --reg 0.001 --fair
```

Common CLI options:

| Flag | Default | Description |
|---|---|---|
| `--test_size` | 0.2 | Test split fraction |
| `--random_state` | 42 | Random seed |
| `--fair` | off | Enable fairness preprocessing and calibration |
| `--fair_post` | sample | `sample` or `leaf_pareto` |
| `--fair_metric` | dp | Fairness metric for LPFR |
| `--fair_lambda` | 1.0 | Fairness weight for LPFR |
| `--fair_acc_budget` | 0.02 | Max accuracy drop for LPFR |
| `--results_dir` | `results` | Output directory for logs |
| `--log_suffix` | empty | Optional suffix for log filenames |

### Model-Specific Options

#### CART

| Flag | Default | Description |
|---|---|---|
| `--max_depth` | 6 | Maximum depth |
| `--min_samples_split` | 10 | Minimum samples to split |
| `--min_samples_leaf` | 5 | Minimum samples per leaf |
| `--binarize_cart` | off | Use the binary feature space |

#### SPLIT

| Flag | Default | Description |
|---|---|---|
| `--lookahead` | 2 | Optimal prefix depth |
| `--depth` | 5 | Total depth budget |
| `--reg` | 0.0001 | Sparsity penalty per leaf |
| `--leaf_fill` | optimal | `greedy` or `optimal` |
| `--max_features` | 0 | Top-K candidate features |
| `--max_thresholds` | 50 | Max thresholds per numeric feature |
| `--binarizer` | gbdt | `gbdt` or `midpoint` |
| `--time_limit` | 60 | DP solver timeout in seconds |
| `--binarize / --no-binarize` | binarize | Toggle feature binarization |

#### LicketySPLIT

| Flag | Default | Description |
|---|---|---|
| `--lookahead` | 2 | Lookahead range |
| `--depth` | 5 | Total depth budget |

#### ReSPLIT

| Flag | Default | Description |
|---|---|---|
| `--num_prefix` | 20 | Number of randomized prefixes |
| `--rashomon_bound` | 0.01 | Keep trees within `(1+eps) * best` |
| `--depth` | 5 | Total depth budget |
| `--lookahead` | 2 | Prefix depth |

## Output

Training logs are written to `results/`.

Typical filenames look like:

```text
results/{model}-{dataset}-d{depth}-{bin/raw}-{nofair/fair/lpfr}.log
```

Each log contains model parameters, accuracy, baseline, feature usage, fairness metrics, and the full tree structure.

## Multi-Seed Experiments

Single experiment across multiple seeds:

```bash
python scripts/run_multi_seed.py \
  --seeds 0,1,2,3,4 \
  --run_name cart-adult-d5 \
  -- \
  --model cart --dataset adult --max_depth 5
```

Full evaluation grid across multiple seeds:

```bash
python scripts/eval_multi_seed.py \
  --seeds 0,1,2,3,4 \
  --run_name full-eval-5seeds
```

By default the grid includes `nofair`, sample-level fairness calibration (`mode=fair`), and LPFR (`mode=lpfr`).

Each run writes:

- `per_seed.json`: metrics for every completed seed
- `summary.json`: mean/std/n for numeric metrics
- `summary.csv`: flattened summary table
- `logs/`: per-seed training logs

## Notes

This repository focuses on practical tree training, evaluation, and fairness workflows.
