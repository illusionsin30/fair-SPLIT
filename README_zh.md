# fair-SPLIT

一个纯 Python 的决策树库，支持 **CART**、**SPLIT**、**LicketySPLIT** 和 **ReSPLIT**。
项目重点是稀疏、可解释的树模型，并内置公平性评估与校准。

## 一、项目简介

决策树之所以易于理解，是因为每个预测都对应一条明确的特征判断路径。
本仓库提供：

- `CART`：用于分类和回归的经典贪心决策树
- `SPLIT`：先求解浅层最优前缀，再补全剩余叶子
- `LicketySPLIT`：多项式时间的递归版 SPLIT
- `ReSPLIT`：通过多样化前缀构造多候选树集合

## 二、什么是决策树？

决策树包含两类节点：

- 内部节点：保存类似 `age <= 30` 的判断条件。
- 叶子节点：保存最终预测结果。

可解释性通常主要看两个方面：

- 稀疏性：叶子越少，树通常越容易阅读。
- 深度：树越深，表达能力越强，但也越难理解。

## 三、SPLIT 系列算法

### 3.1 核心思想

SPLIT 只在靠近根部的层做最优搜索，底层则用贪心或再次最优补全。
这样可以在保持模型简洁的同时，避免全树精确搜索的高开销。

### 3.2 优化目标

对于二分类问题，正则化目标可以写成：

```text
loss(T) = error_rate(T) + lambda * num_leaves(T)
```

其中 `lambda` 控制准确率与稀疏性的权衡。

### 3.3 特征二值化

SPLIT 使用二值特征。

- 数值特征会被转换为阈值判断。
- 类别特征会被 one-hot 编码。

仓库支持 `midpoint` 阈值和 `gbdt` stump 阈值猜测两种方式。

### 3.4 运行流程

- 阶段一：使用动态规划构造最优前缀。
- 阶段二：用贪心子树或最优子树填充剩余叶子。

### 3.5 ReSPLIT

ReSPLIT 通过生成多样化前缀，并保留分数接近最优值的树，构造多候选树集合。

## 四、公平性评估与校准

项目支持公平性预处理和后处理。

### 4.1 公平性指标

| 指标 | 含义 | 理想值 |
|---|---|---|
| Statistical Parity Difference | 各组正预测率的最大值减最小值 | 接近 0 |
| Disparate Impact Ratio | 最小正预测率除以最大正预测率 | 接近 1 |
| Equal Opportunity Difference | 各组 TPR 的最大差异 | 接近 0 |

### 4.2 公平性流程

启用 `--fair` 后：

1. 训练前删除敏感属性列。
2. 训练后对分组预测做校准。

`--fair_post sample` 表示样本级校准。
`--fair_post leaf_pareto` 表示 Leaf-Pareto Fair Recalibration（LPFR），保持树结构不变，只调整叶子输出。

## 五、CART 基准

CART 作为对照基线保留在仓库中。

- 它直接使用原始数值和类别特征。
- 分类任务使用 Gini 不纯度，回归任务使用方差。
- `--binarize_cart` 可以让 CART 和 SPLIT 处于同一个二值特征空间中进行比较。

## 六、数据流水线

完整流程如下：

```text
原始数据集 -> 预处理 -> 划分训练/测试集 -> 模型训练 -> 评估
```

数据加载器覆盖 UCI、Kaggle、folktables 和 sklearn 数据集。
每个数据集都会单独识别敏感属性，并传给公平性评估模块。

## 七、算法之间的对比

| 模型 | 特征空间 | 搜索策略 | 输出 |
|---|---|---|---|
| CART | 原始特征 | 纯贪心 | 一棵树 |
| SPLIT | 二值特征 | 最优前缀 + 补全 | 一棵树 |
| LicketySPLIT | 二值特征 | 递归最优根切分 | 一棵树 |
| ReSPLIT | 二值特征 | 多样化前缀 + 补全 | 一组树 |

常见起点：

- 快速基线：`--model cart`
- 平衡速度和质量：`--model split --depth 5 --lookahead 2 --reg 0.001`
- 更稀疏的树：`--model split --depth 3 --lookahead 2 --reg 0.01`
- 更高准确率：`--model split --depth 6 --lookahead 3 --reg 0.0001`
- 多样性分析：`--model resplit --num_prefix 30`

## 八、项目结构

```text
fair-SPLIT/
├── src/
│   ├── train.py
│   ├── tree.py
│   ├── solver.py
│   ├── builder.py
│   ├── evaluate.py
│   ├── utils/
│   │   ├── nodes.py
│   │   ├── binarizer.py
│   │   ├── helpers.py
│   │   └── fairness.py
│   └── data/
│       ├── dataload.py
│       ├── process.py
│       └── __init__.py
├── scripts/
│   ├── train.sh
│   ├── eval.sh
│   ├── run_multi_seed.py
│   └── eval_multi_seed.py
├── results/
├── requirements.txt
├── LICENSE
├── README.md
└── README_zh.md
```

## 九、支持的数据集

### 核心基准

| 命令行名称 | 数据集 | 来源 | 样本量 | 任务类型 |
|---|---|---|---|---|
| `adult` | Adult / Census Income | UCI #2 | 48K | 二分类 |
| `bike` | Bike Sharing | UCI #275 | 17K | 回归 |
| `spambase` | Spambase | UCI #94 | 4.6K | 二分类 |
| `bank` | Bank Marketing | UCI #222 | 45K | 二分类 |
| `covertype` | Covertype | UCI #31 | 581K | 7 分类 |
| `compass` | COMPAS Recidivism | Kaggle | 18K | 二分类 |
| `heloc` | HELOC | Kaggle | 10K | 二分类 |
| `thyroid` | Thyroid Disease | UCI #102 | 7K | 多分类 |

### 公平性基准

| 命令行名称 | 数据集 | 来源 | 样本量 | 敏感属性 |
|---|---|---|---|---|
| `adult` | Adult | UCI #2 | 48K | sex, race |
| `compass` | COMPAS | Kaggle | 18K | sex, race |
| `german` | German Credit | UCI #144 | 1K | sex+status, age |
| `communities` | Communities & Crime | UCI #183 | 2K | race |
| `lawschool` | Law School Admissions | Kaggle | 20K | race, gender |
| `acsincome` | ACS Income (2018 CA) | folktables | 196K | sex, race |
| `bank` | Bank Marketing | UCI #222 | 45K | marital, age |

### 经典基准

| 命令行名称 | 数据集 | 来源 | 样本量 | 任务类型 |
|---|---|---|---|---|
| `heart` | Heart Disease | UCI #45 | 303 | 二分类 |
| `iris` | Iris | sklearn | 150 | 3 分类 |
| `diabetes` | Diabetes | sklearn | 442 | 回归 |

### 数据集依赖

| 数据集 | 所需包 | 首次运行 |
|---|---|---|
| UCI 数据集 | `ucimlrepo` | 自动下载 |
| Kaggle 数据集 | `kagglehub` 或 `kaggle` CLI | 自动下载 |
| `acsincome` | `folktables` | 约 200MB 下载 |
| `iris`、`diabetes` | `scikit-learn` | 随包提供 |

## 十、安装与运行

### 环境配置

```bash
conda create -n split python=3.10 -y
conda activate split
pip install -r requirements.txt
```

可选依赖：

```bash
pip install folktables
pip install kaggle
```

### 基本运行

所有训练都通过 `bash scripts/train.sh` 启动：

```bash
bash scripts/train.sh --model <模型> --dataset <数据集> [参数...]
```

### 运行示例

```bash
bash scripts/train.sh --model cart --dataset adult
bash scripts/train.sh --model split --dataset bank --depth 5 --reg 0.001
bash scripts/train.sh --model licketysplit --dataset german --depth 5
bash scripts/train.sh --model resplit --dataset compass --num_prefix 30
bash scripts/train.sh --model cart --dataset acsincome --fair
bash scripts/train.sh --model split --dataset acsincome --depth 5 --reg 0.001 --fair
```

常用 CLI 参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--test_size` | 0.2 | 测试集比例 |
| `--random_state` | 42 | 随机种子 |
| `--fair` | 关闭 | 启用公平性预处理和校准 |
| `--fair_post` | sample | `sample` 或 `leaf_pareto` |
| `--fair_metric` | dp | LPFR 使用的公平性指标 |
| `--fair_lambda` | 1.0 | LPFR 的公平性权重 |
| `--fair_acc_budget` | 0.02 | LPFR 允许的最大准确率下降 |
| `--results_dir` | `results` | 日志输出目录 |
| `--log_suffix` | 空 | 日志文件后缀 |

### 各模型参数

#### CART

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--max_depth` | 6 | 最大深度 |
| `--min_samples_split` | 10 | 最小分裂样本数 |
| `--min_samples_leaf` | 5 | 每个叶子的最小样本数 |
| `--binarize_cart` | 关闭 | 使用与 SPLIT 相同的二值特征空间 |

#### SPLIT

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--lookahead` | 2 | 最优前缀深度 |
| `--depth` | 5 | 总深度预算 |
| `--reg` | 0.0001 | 每个叶子的稀疏惩罚 |
| `--leaf_fill` | optimal | `greedy` 或 `optimal` |
| `--max_features` | 0 | 候选特征 Top-K |
| `--max_thresholds` | 50 | 每个数值特征的最大阈值数 |
| `--binarizer` | gbdt | `gbdt` 或 `midpoint` |
| `--time_limit` | 60 | DP 求解超时时间（秒） |
| `--binarize / --no-binarize` | binarize | 是否做特征二值化 |

#### LicketySPLIT

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--lookahead` | 2 | 前瞻范围 |
| `--depth` | 5 | 总深度预算 |

#### ReSPLIT

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--num_prefix` | 20 | 随机前缀数量 |
| `--rashomon_bound` | 0.01 | 保留 `(1+eps) * best` 范围内的树 |
| `--depth` | 5 | 总深度预算 |
| `--lookahead` | 2 | 前缀深度 |

## 十一、输出说明

训练日志会写入 `results/`。

常见文件名格式：

```text
results/{model}-{dataset}-d{depth}-{bin/raw}-{nofair/fair/lpfr}.log
```

每个日志包含模型参数、准确率、基线、使用到的特征、公平性指标和完整树结构。

## 十二、多种子实验

单个实验跑多个随机种子：

```bash
python scripts/run_multi_seed.py \
  --seeds 0,1,2,3,4 \
  --run_name cart-adult-d5 \
  -- \
  --model cart --dataset adult --max_depth 5
```

全量网格跑多个随机种子：

```bash
python scripts/eval_multi_seed.py \
  --seeds 0,1,2,3,4 \
  --run_name full-eval-5seeds
```

默认网格包含 `nofair`、样本级公平校准（`mode=fair`）和 LPFR（`mode=lpfr`）。

每次运行会生成：

- `per_seed.json`：每个完成种子的指标
- `summary.json`：数值指标的 mean/std/n
- `summary.csv`：扁平化汇总表
- `logs/`：每个种子的训练日志

## 十三、说明

本仓库聚焦于树模型训练、评估和公平性工作流。
