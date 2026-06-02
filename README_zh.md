# fair-SPLIT：可解释决策树与公平性评估工具

## 一、项目简介

**fair-SPLIT** 是一个纯 Python 实现的决策树算法库，融合了经典贪心方法（CART）与现代最优树搜索策略（SPLIT 系列算法），并内置了公平性评估与校准工具。

决策树是机器学习中最经典的可解释模型之一。一棵决策树的推理过程可以直观地展示为一系列"如果…那么…"的规则，任何人都能理解模型是如何作出判断的。然而，传统的贪心建树方法（如 CART、C4.5）虽然快速，却无法保证找到全局最优的树结构，可能导致模型在准确率和稀疏性上都不尽如人意。

本项目的核心目标是在**训练速度**、**模型准确率**、**稀疏可解释性**以及**公平性**之间寻求平衡，提供一套实用的工具链。

## 二、什么是决策树？

### 2.1 基本结构

一棵决策树由**内部节点**（Internal Node）和**叶子节点**（Leaf Node）组成：

- **内部节点**：存储一个"是/否"的判断条件，例如"年龄 ≤ 30？"。根据判断结果，样本被路由到左子节点或右子节点。
- **叶子节点**：存储一个预测值。对于分类任务，叶子存储类别标签；对于回归任务，叶子存储一个连续值。

=== 示意图：决策树结构 ===

```
                  [年龄 ≤ 30?]          ← 根节点（内部节点）
                  /          \
               Yes            No
              /                \
    [学历 ≥ 本科?]          [收入 > 50K]   ← 内部节点
       /      \              /      \
     Yes      No           Yes      No
     /          \           /        \
  [通过]       [不通过]   [通过]    [不通过]   ← 叶子节点
```

实际应用中，决策树的可解释性主要体现在两个方面：

1. **稀疏性**（Sparsity）：叶子数量越少，整棵树的逻辑越简洁，用户越容易理解。
2. **深度**（Depth）：从根到叶的最长路径长度。深度越大，树能表达越复杂的决策边界，但也越难理解。

研究表明，用户对决策树的理解程度与叶子数量呈强负相关——**叶子越少，越容易理解**。同时，在叶子数量相同的情况下，较深的稀疏树比浅而密的树更容易理解[^1]。

[^1]: Piltaver, R., et al. "What makes classification trees comprehensible?" Expert Systems with Applications, 2016.

### 2.2 经典建树方法：CART

**CART**（Classification And Regression Tree）是最经典的贪心建树算法。其核心思想是：

1. 从根节点开始，在所有特征中搜索**最优切分点**
2. 对数值型特征，遍历所有可能的分割阈值，选择使**Gini不纯度**最低的阈值
3. 对类别型特征，用Breiman的最优类别子集方法
4. 递归地对左右子节点重复此过程，直到满足停止条件

Gini不纯度的定义为：

$$Gini(D) = 1 - \sum_{c=1}^{C} p_c^2$$

其中 $p_c$ 是类别 $c$ 在数据集 $D$ 中的比例。Gini值越小，表示数据越"纯"（即大部分样本属于同一类别）。CART在每步贪心地选择能使Gini下降最多的特征和阈值。

### 2.3 贪心方法的局限性

贪心方法快速且易于实现，但有一个根本问题：**每一步的最优不等于全局的最优**。

考虑一个简化的例子：假设有两个特征 A 和 B。单独看，A 是最好的切分特征。但沿着 A 切分后，子节点中 B 的表现很差。而如果第一步选择 B（尽管单独看不如 A），后续再用 A 切分时，整体效果反而更好。

贪心方法每次只看一步，无法预见到这种"第二步的收益超过第一步的损失"的情况。在最坏情形下，贪心树和最优树之间的准确率差距可以达到 **10 个百分点**[^2]。

[^2]: Demirović, E., et al. "MurTree: Optimal Decision Trees via Dynamic Programming and Search." JMLR, 2022.

## 三、SPLIT 系列算法

### 3.1 核心思想

SPLIT（SParse Lookahead for Interpretable Trees）算法族的核心理念是：**并非树中所有深度都需要做全局最优搜索**。

直觉上，根节点附近的切分决策最为关键——它们在树的顶端，影响所有后续的分支，一个不好的根切分会让后面所有努力都大打折扣。而靠近叶子的决策则不那么重要——因为已经经过了好几层筛选，剩下的样本数量少，能做的切分也有限，贪心就足够了。

这一直觉在实验中得到了验证：在全局最优树的集合中，**越接近叶子的节点，越倾向于采用贪心切分**。这意味着我们可以只在靠近根部的几层做最优搜索，叶子附近用贪心填充，从而在准确率和速度之间找到最佳平衡。

### 3.2 优化目标

所有 SPLIT 算法都求解同一个正则化优化问题。给定训练数据集 $D = \{(x_i, y_i)\}_{i=1}^N$，最大深度约束 $d$，以及稀疏性惩罚参数 $\lambda$：

$$L^*(D, d, \lambda) = \min_{T \in \mathcal{T}} \; \frac{1}{N}\sum_{i=1}^{N} \mathbb{1}[T(x_i) \neq y_i] + \lambda \cdot S(T)$$

其中：

- $T(x_i)$ 是树对样本 $x_i$ 的预测
- $\mathbb{1}[\cdot]$ 是指示函数（预测错误为 1，正确为 0）
- $S(T)$ 是树 $T$ 的叶子数量
- $\lambda$ 是正则化参数，控制稀疏性：$\lambda$ 越大，树越稀疏（叶子越少）

这个目标函数的含义是：我们希望一棵树既准确（第一项——错误率低），又简洁（第二项——叶子少）。$\lambda$ 决定了我们在准确率和简洁性之间如何取舍。

### 3.3 特征二值化

SPLIT 算法要求输入特征是二值的（0 或 1）。这是因为最优搜索的搜索空间随特征取值数量呈指数增长，二值化可以大幅压缩搜索空间。

现实世界的数据集通常包含连续型特征（如年龄、收入）和多值类别特征（如职业、学历）。需要用两种策略将其转化为二值特征：

**数值型特征**：对每个连续特征收集一组阈值，将"$x_j \leq t$"这类条件作为二值特征。例如，对年龄特征选取阈值 30、45、60，则产生三个二值特征：

- `age ≤ 30`：True/False
- `age ≤ 45`：True/False
- `age ≤ 60`：True/False

如何选取阈值是关键。本项目提供两种方式：

1. **等间距中点法**（midpoint）：对特征的所有唯一取值排序，取相邻值的中点作为阈值。这种方式是无损的（不会丢失信息），但会产生大量二值特征（一个具有 1000 个不同值的特征会产生 999 个二值特征）。

2. **GBDT Stump 猜测法**（threshold guessing，默认）：先用梯度提升树（GBDT）训练一组深度为 1 的决策树桩（stump），收集所有 stump 中出现的阈值。这些阈值是 GBDT 认为"有区分力"的切分点，远比等间距法高效。

**类别型特征**：对每个类别做独热编码（one-hot encoding）。例如，工作类型有"私企、国企、外企"三个类别，则产生三个二值特征：`workclass=私企`、`workclass=国企`、`workclass=外企`。

### 3.4 算法流程

#### SPLIT（Algorithm 2）

SPLIT 分两阶段进行：

**阶段一：最优前缀搜索**

在树的顶部 $d_l$ 层（lookahead depth，默认 2 层）做最优搜索。搜索采用动态规划（DP），枚举所有可能的特征组合：

$$L(D, d', \lambda) = \min \begin{cases} \lambda + \frac{\text{少数类数量}}{N} & \text{(做叶子)} \\ \min_f L(D_f, d'-1, \lambda) + L(D_{\bar{f}}, d'-1, \lambda) & \text{(切分并递归)} \end{cases}$$

其中 $D_f$ 是特征 $f$ 为 1 的样本子集，$D_{\bar{f}}$ 是特征 $f$ 为 0 的样本子集。

在 lookahead 边界处（即 DP 搜索到的最深层），不再继续最优递归，而是用**贪心完成**（Greedy Completion）来计算子问题的代价。这是 SPLIT 论文的核心创新——第 $d_l$ 层以下的搜索空间被贪心树取代，构成一个紧的上界。

**阶段二：叶子填充**

阶段一产出的前缀树的每一片叶子，都可以进一步扩展。根据配置，可以用贪心树（`--leaf_fill greedy`）或再次调用 DP 最优树（`--leaf_fill optimal`）来填充到剩余的深度预算。

#### LicketySPLIT（Algorithm 3）

LicketySPLIT 是 SPLIT 的多项式时间变体，其核心思想是**在每个节点只确定最优的根切分**，然后递归地对子节点重复这一过程。

对于根节点，LicketySPLIT 对每个候选特征 $f$，计算：

$$cost(f) = L_{greedy}(D_f, d-1, \lambda) + L_{greedy}(D_{\bar{f}}, d-1, \lambda)$$

即：如果根切分选择 $f$，左子树和右子树都用深度为 $d-1$ 的**贪心树**完成，总代价是多少。选择使 $cost(f)$ 最小的特征作为根切分，然后对左右子节点递归调用 LicketySPLIT。

这一算法的理论时间复杂度为 $O(n \cdot k^2 \cdot d^2)$（$n$ 为样本数，$k$ 为特征数，$d$ 为深度），是多项式级别的——比指数级的最优搜索快了几个数量级。

#### ReSPLIT

ReSPLIT（Rashomon set Estimation with SPLIT）用于估计决策树的**Rashomon集合**——即在最优目标值 $\epsilon$ 范围内的所有近似最优树：

$$R(D, \lambda, \epsilon, d) = \{T : L(T, D, \lambda) \leq L^*(D, d, \lambda) + \epsilon,\; depth(T) \leq d\}$$

Rashomon集合有什么意义？它告诉我们"除了最优的那棵树之外，还有哪些树也差不多好"。在实际应用中，我们可以从集合中挑选一棵**既准确又公平**的树。

我们的 ReSPLIT 通过随机化特征评估顺序来生成多个结构不同的前缀树（通过 shuffled feature order + `pick_first` 策略实现多样性），然后用贪心树填充叶子，保留目标值在 $(1+\epsilon)L^*$ 内的树。

## 四、公平性评估与校准

### 4.1 公平性为什么重要？

机器学习模型在实际应用中可能对某些群体产生系统性偏见。例如，一个贷款审批模型可能在同样还款能力的情况下，对不同种族或性别的申请人给出不同的预测结果。这种偏见可能源于：

- **训练数据的偏差**：历史数据本身反映了社会的不平等
- **特征选择的问题**：模型使用了与敏感属性高度相关的代理变量（如邮政编码可以代理种族信息）

决策树作为可解释模型，特别适合进行公平性分析——我们可以**直接看到哪些特征参与了决策**，也可以按敏感属性分组查看模型在各群体上的表现。

### 4.2 公平性指标

我们实现了三个标准的公平性指标，对每个敏感属性（如性别、种族）的不同取值组分别计算：

| 指标 | 公式 | 含义 | 理想值 |
|---|---|---|---|
| **Statistical Parity Difference** | $\max_g P(\hat{y}=1 \mid g) - \min_g P(\hat{y}=1 \mid g)$ | 最大和最小正预测率之差 | 接近 0 |
| **Disparate Impact Ratio** | $\frac{\min_g P(\hat{y}=1 \mid g)}{\max_g P(\hat{y}=1 \mid g)}$ | 最小与最大正预测率之比 | > 0.8 |
| **Equal Opportunity Difference** | $\max_g TPR_g - \min_g TPR_g$ | 不同组真阳性率（TPR）的最大差异 | 接近 0 |

其中 $P(\hat{y}=1 \mid g)$ 表示敏感属性为 $g$ 的群体中被预测为正类的比例。$TPR_g$ 表示群体 $g$ 中真实正例被正确预测的比例。

### 4.3 公平性方法

使用 `--fair` 参数开启公平性增强，包含两步：

**预处理（Pre-processing）**：在训练前，从特征矩阵中**删除敏感属性列**（如性别、种族），防止模型直接使用这些属性做决策。这是公平机器学习中最常用的基线方法。

**后处理校准（Post-processing）**：训练完成后，对预测结果进行**按组阈值校准**。具体地：

1. 计算全局目标正预测率 $\bar{p} = \frac{1}{N}\sum_i \hat{y}_i$
2. 对每个敏感组 $g$，如果该组的正预测率高于 $\bar{p}$，则将部分预测从 1 翻转为 0（优先翻转那些实际标签就是 0 的样本，以减少精度损失）
3. 如果该组的正预测率低于 $\bar{p}$，则将部分预测从 0 翻转为 1（优先翻转实际标签为 1 的样本）

校准后的各组正预测率被对齐到全局均值，Statistical Parity Difference 大幅缩小，同时在多数情况下准确率变化很小。

### 4.4 间接歧视与公平性局限

删除敏感属性并做后处理校准，并不能完全消除模型偏见。因为特征矩阵中可能存在与敏感属性**高度相关的代理变量**（proxy variables）。例如：

- 邮政编码（ZIP code）在某些地区与种族高度相关
- 职业类型（occupation）可能与性别相关
- 教育水平（education）在历史上与社会经济地位相关

决策树即使不直接使用"种族"做切分，仍可能通过邮编、职业等变量间接产生针对特定群体的差异化预测。这是公平机器学习领域一个活跃的研究方向，目前没有完美的解决方案。

## 五、经典决策树基准：CART

为了提供对比基准，本项目也实现了经典的 CART 算法。CART 与 SPLIT 系列算法有两个关键区别：

1. **特征处理**：CART 直接在原始连续特征上工作，在每个节点搜索最优切分阈值——这是一个 O(n log n) 的操作，但能保证在当前节点找到局部最优的连续切分点。SPLIT 则在预处理的二值化特征上工作，特征搜索更快（O(1) 特征访问），但阈值是固定的。

2. **优化方式**：CART 是纯贪心的——每步选取当前最优切分，不回溯、不前瞻。SPLIT 在前 $d_l$ 层做最优化，只在深层才退化为贪心。

使用 `--binarize_cart` 可以让 CART 也在二值化特征上运行，此时与 SPLIT 的比较是公平的（相同的特征空间）。

## 六、数据结构：从特征到二值化矩阵

整个数据流水线可以概括为：

```
原始数据集 (DataFrame)
  ├── 数值列 (age, income, ...)
  │     └── GBDT Stump 阈值猜测 → 二值列 (age≤28.5, income≤50K, ...)
  ├── 类别列 (sex, race, ...)
  │     └── One-hot 编码 → 二值列 (sex=Female, race=White, ...)
  └── 目标列 (label)
        └── 多分类 → 保留K类 (SPLIT原生支持)
            回归   → 中位数二分
```

最终得到的二值化矩阵 X 形状为 `(样本数, 二值特征数)`。对于 Adult 数据集，14 个原始特征会产生约 130 个二值特征。对于 Bank 数据集，16 个原始特征会产生约 80 个二值特征。

## 七、算法之间的对比

| 维度 | CART | SPLIT | LicketySPLIT | ReSPLIT |
|---|---|---|---|---|
| **建树方式** | 纯贪心 | lookahead 最优 + 贪心填充 | 递归最优根切分 | 随机前缀 × N + 贪心填充 |
| **特征** | 连续/类别（原始） | 二值化 | 二值化 | 二值化 |
| **时间复杂度** | O(d·k·n log n) | O(n·k^{d_l+1} + n·k^{d-d_l}) | O(n·k²·d²) 多项式 | O(P·SPLIT) |
| **最优性保证** | 无 | 前 d_l 层最优 | 每层根切分最优 | 近似 Rashomon 集 |
| **产出** | 1 棵树 | 1 棵树 | 1 棵树 | P 棵树（集合） |

**参数选择指南**：

- 快速探索：`--model cart` 或 `--model licketysplit`
- 平衡速度与质量：`--model split --depth 5 --lookahead 2 --reg 0.001`
- 追求稀疏：`--model split --depth 3 --lookahead 2 --reg 0.01`
- 追求准确：`--model split --depth 6 --lookahead 3 --reg 0.0001`
- 多模型比较：`--model resplit --num_prefix 30`

## 八、项目结构

```
fair-SPLIT/
├── src/
│   ├── train.py              # 命令行入口 (python -m src.train)
│   ├── tree.py               # CART, SPLIT, LicketySPLIT, ReSPLIT 四个模型类
│   ├── solver.py             # 动态规划最优前缀求解器
│   ├── builder.py            # 基于熵增益的贪心树构造器
│   ├── evaluate.py           # 准确率评估 + 公平性指标计算
│   ├── utils/
│   │   ├── nodes.py          # 树节点数据结构 (SPLITLeaf / SPLITNode)
│   │   ├── binarizer.py      # 特征二值化器 (NumericBinarizer + ThresholdGuessBinarizer)
│   │   ├── helpers.py        # 预测、树导出、分裂特征遍历
│   │   └── fairness.py       # 公平性预处理 + 后处理校准
│   └── data/
│       ├── dataload.py       # 15 个数据集的自动下载与加载
│       ├── process.py        # 目标变量预处理（多分类→保留，回归→二值化）
│       └── __init__.py
├── scripts/
│   └── train.sh              # 便捷启动脚本
├── results/                  # 训练日志输出目录
├── requirements.txt
├── LICENSE
├── README.md                 # 英文快速参考
└── README_zh.md              # 本文档（中文完整说明）
```

## 九、支持的 15 个数据集

| 命令行名称 | 数据集 | 来源 | 样本量 | 任务类型 |
|---|---|---|---|---|
| `adult` | Adult / Census Income | UCI #2 | ~48K | 二分类 |
| `bike` | Bike Sharing | UCI #275 | ~17K | 回归 |
| `spambase` | Spambase | UCI #94 | ~4.6K | 二分类 |
| `bank` | Bank Marketing | UCI #222 | ~45K | 二分类 |
| `covertype` | Covertype | UCI #31 | ~581K | 7分类 |
| `compass` | COMPAS Recidivism | Kaggle | ~18K | 二分类 |
| `heloc` | HELOC | Kaggle | ~10K | 二分类 |
| `thyroid` | Thyroid Disease | UCI #102 | ~7K | 多分类 |
| `german` | German Credit | UCI #144 | ~1K | 二分类 |
| `communities` | Communities & Crime | UCI #183 | ~2K | 二分类 |
| `heart` | Heart Disease | UCI #45 | ~303 | 二分类 |
| `lawschool` | Law School Admissions | Kaggle | ~20K | 二分类 |
| `acsincome` | ACS Income (2018 CA) | folktables | ~196K | 二分类 |
| `iris` | Iris | sklearn | ~150 | 3分类 |
| `diabetes` | Diabetes | sklearn | ~442 | 回归 |

### 数据集自动下载

- **UCI 数据集**：通过 `ucimlrepo` 自动获取
- **Kaggle 数据集**：通过 `kagglehub` 自动获取，失败时回退到 `kaggle` 命令行工具
- **ACS Income**：通过 `folktables` 自动下载美国人口普查数据（约 200MB）
- **Iris/Diabetes**：随 scikit-learn 内置

## 十、安装与运行

### 环境配置

```bash
conda create -n split python=3.10 -y
conda activate split
pip install -r requirements.txt
```

可选依赖：

```bash
pip install folktables   # ACS Income 数据集
pip install kaggle        # Kaggle 数据集的命令行回退方案
```

### 基本运行

所有模型通过 `bash scripts/train.sh` 启动，参数直接传给 `python -m src.train`：

```bash
bash scripts/train.sh --model <模型名> --dataset <数据集名> [参数...]
```

### 运行示例

```bash
# 基准测试：CART on Adult
bash scripts/train.sh --model cart --dataset adult

# SPLIT 不同稀疏度
bash scripts/train.sh --model split --dataset bank --depth 3 --reg 0.01
bash scripts/train.sh --model split --dataset bank --depth 5 --reg 0.001
bash scripts/train.sh --model split --dataset bank --depth 6 --reg 0.0001

# 多项式时间方案：LicketySPLIT
bash scripts/train.sh --model licketysplit --dataset adult --depth 5

# Rashomon 集合：ReSPLIT
bash scripts/train.sh --model resplit --dataset compass --num_prefix 30 --rashomon_bound 0.02

# 公平性对比
bash scripts/train.sh --model cart --dataset acsincome
bash scripts/train.sh --model cart --dataset acsincome --fair
```

详细参数说明请参见 [README.md](README.md) 中的参数表格。

## 十一、输出解读

每次训练结果保存在 `results/` 目录下，文件名为：

```
{模型}-{数据集}-d{深度}-{bin/raw}-{fair/nofair}.log
```

日志包含：

1. **数据信息**：数据集名称、样本量、类别分布
2. **模型参数**：所有超参数的值
3. **测试准确率**和**多数类基线**（如果模型总是预测样本最多的类别，能达到的准确率）
4. **分裂特征列表**：按树结构层级顺序列出实际参与决策的特征
5. **公平性面板**：各敏感属性组的样本量、准确率、正预测率，以及三项公平性指标。若 `--fair` 开启，还包含校准后的对比。
6. **完整树结构**：嵌套字典形式的树，便于直接查看每条决策路径。

训练准确率高于基线 → 模型学到了有效模式。准确率等于基线 → 模型可能只预测了多数类（未分裂或分裂无效）。SP diff（Statistical Parity Difference）越接近 0，DI（Disparate Impact Ratio）越接近 1，模型在各群体间越公平。

## 十二、参考文献

SPLIT 算法族源自以下论文：

> Babbar, V., McTavish, H., Rudin, C., & Seltzer, M. (2025).
> *Near-Optimal Decision Trees in a SPLIT Second.*
> Proceedings of the 42nd International Conference on Machine Learning (ICML), Vancouver, Canada. PMLR 267.
> arXiv: 2502.15988

本项目的 CART 实现参考：

> Breiman, L., Friedman, J., Olshen, R., & Stone, C. (1984).
> *Classification and Regression Trees.* Wadsworth.

阈值猜测（Threshold Guessing）方法参考：

> McTavish, H., Zhong, C., Achermann, R., Karimi, A., Chen, J., Rudin, C., & Seltzer, M. (2022).
> *Fast Sparse Decision Tree Optimization via Reference Ensembles.*
> Proceedings of the AAAI Conference on Artificial Intelligence, 36(9), 9604-9613.

公平性评估框架参考：

> Donnelly, J., Katta, S., Rudin, C., & Browne, E. (2023).
> *The Rashomon Importance Distribution: Getting RID of Unstable, Single Model-based Variable Importance.*
> NeurIPS 2023.
