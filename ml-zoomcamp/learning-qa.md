# ML Zoomcamp 学习问答笔记

本文沉淀本地学习 ML Zoomcamp 过程中遇到的问题和核心理解点。当前项目以 `ml-zoomcamp/` 为 uv 项目目录，Chapter 2 代码以 `02-regression/notebook.ipynb` 为准。

## 02-regression

### 1. EDA 与目标值变换

#### Q10: 为什么要做 log price？

价格通常是长尾分布：

```text
多数车集中在普通价格区间
少数豪车价格极高
```

如果直接用原始价格训练，极高价格样本会强烈影响平方误差。课程使用：

```python
y = np.log1p(df.msrp)
```

即：

```text
log(price + 1)
```

作用：

```text
压缩高价长尾
减少极端价格对模型的拉扯
让模型更关注相对误差
让线性模型更容易拟合
```

预测后转回美元价格：

```python
price_pred = np.expm1(y_pred)
```

#### Q11: log price 算不算特征工程？

广义上是建模前的数据处理；严格说，它是目标变量变换，不是输入特征工程。

区别：

```text
feature engineering      处理 X，例如 year -> age
target transformation    处理 y，例如 msrp -> log1p(msrp)
```

### 2. 特征工程与 One-Hot Encoding

#### Q12: One-Hot Encoding 是什么意思？

One-Hot Encoding 是把类别变量拆成多个 0/1 数字列。

原始字段：

```text
make = ford
make = toyota
make = bmw
```

转换后：

```text
is_make_ford  is_make_toyota  is_make_bmw
1             0               0
0             1               0
0             0               1
```

课程中的手写 one-hot：

```python
for v in ['chevrolet', 'ford', 'volkswagen', 'toyota', 'dodge']:
    feature = 'is_make_%s' % v
    df[feature] = (df['make'] == v).astype(int)
```

#### Q13: One-Hot 的优点和缺点是什么？

优点：

```text
简单直观
适合线性模型
不会错误引入类别顺序
每个类别可以有自己的权重
```

缺点：

```text
类别多时特征膨胀
数据变稀疏
训练时没见过的新类别不好处理
不能表达类别之间的相似性
完整 one-hot 加 intercept 可能导致共线性
```

例如：

```text
is_size_compact + is_size_midsize + is_size_large = 1
```

如果模型里还有一列 intercept `ones`，就会出现线性相关，导致矩阵求逆不稳定。

### 3. 线性回归与矩阵求解

#### Q17: `train_linear_regression(X, y)` 做了什么？

函数核心：

```python
def train_linear_regression(X, y):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)

    return w_full[0], w_full[1:]
```

它训练模型：

```text
y_pred = w0 + Xw
```

其中：

```text
w0      截距项
w       每个特征的权重
```

#### Q18: 为什么 `(X^T X)^-1 X^T y` 是线性回归的解？

线性回归最小化平方误差：

```text
min ||Xw - y||^2
```

令：

```text
L(w) = (Xw - y)^T (Xw - y)
```

展开并对 `w` 求导，令导数为 0：

```text
X^T X w = X^T y
```

如果 `X^T X` 可逆：

```text
w = (X^T X)^-1 X^T y
```

这就是 normal equation。

#### Q19: 为什么会出现 `LinAlgError: Singular matrix`？

因为：

```python
np.linalg.inv(XTX)
```

要求 `XTX` 可逆。如果特征列之间有重复或线性相关，矩阵就不可逆。

常见原因：

```text
X 已经手动加过 ones，函数内部又加了一次
完整 one-hot 和 intercept 共线
某些 one-hot 列全是 0
类别清洗失败导致 one-hot 全 0
特征重复或高度相关
```

检查方法：

```python
X_with_ones = np.column_stack([np.ones(X.shape[0]), X])

print(X_with_ones.shape)
print(np.linalg.matrix_rank(X_with_ones))
```

如果输出：

```text
(9, 5)
4
```

表示有 5 列，但秩只有 4，列之间线性相关。

### 4. 正则化

#### Q21: `reg = r * np.eye(XTX.shape[0]); XTX = XTX + reg` 是什么意思？

这是 Ridge/L2 正则化。

```python
np.eye(XTX.shape[0])
```

生成单位矩阵。乘以 `r` 后加到 `XTX` 上：

```text
XTX + rI
```

代码：

```python
reg = r * np.eye(XTX.shape[0])
XTX = XTX + reg
```

对应数学解：

```text
w = (X^T X + rI)^-1 X^T y
```

作用：

```text
让矩阵更容易求逆
限制权重不要过大
缓解共线性和过拟合
提升泛化稳定性
```

#### Q22: 为什么 regularization works？

普通线性回归最小化训练误差：

```text
min ||Xw - y||^2
```

正则化后最小化：

```text
min ||Xw - y||^2 + r||w||^2
```

它同时考虑：

```text
预测误差要小
权重不要太大
```

对目标函数求导并令导数为 0：

```text
(X^T X + rI)w = X^T y
```

所以：

```text
w = (X^T X + rI)^-1 X^T y
```

直觉上，如果 `X^T X` 在某些方向上非常小，直接求逆会出现：

```text
1 / 很小的数
```

导致权重爆炸。加上 `r` 后变成：

```text
1 / (很小的数 + r)
```

数值更稳定。

#### Q23: 加了正则化后，还是原来的最小 RMSE 吗？

不是。加正则化后，目标函数已经改变。

普通模型：

```text
只追求训练误差最小
```

正则化模型：

```text
在训练误差和权重大小之间做折中
```

所以它可能牺牲一点训练集 RMSE，但换来更好的 validation/test RMSE。

`r` 的含义：

```text
r = 0       不正则化
r 很小      轻微稳定
r 适中      通常泛化更好
r 太大      权重被压得太小，欠拟合
```

真实选择应该看 validation RMSE，而不是只看 train RMSE。

### 5. 线性模型、神经网络与 LLM

#### Q27: 这个汽车价格任务如果用神经网络可以吗？

可以，但不一定更划算。

神经网络流程大概是：

```text
数值特征 -> normalization
类别特征 -> one-hot 或 embedding
目标值 -> log1p(msrp)
模型 -> MLP 回归
loss -> MSE on log price
```

优点：

```text
能学习非线性关系
能学习特征交互
embedding 可表达类别相似性
```

缺点：

```text
数据量可能不够
更容易过拟合
调参成本更高
tabular 数据上不一定强于树模型
可解释性更弱
```

当前数据约 11914 行。对这个规模的结构化表格数据，常见强基线通常是：

```text
Ridge regression
Random Forest
XGBoost / LightGBM / CatBoost
TabPFN / tabular foundation model
```

#### Q28: LLM 能直接用提示词或智能体取代线性模型吗？

不建议。

普通 LLM prompt 并没有真正基于训练集优化 RMSE。它可能根据常识猜一个看似合理的价格，但存在问题：

```text
不稳定
不校准
没有明确优化验证集 RMSE
批量预测成本高
难审计
难复现
```

线性模型或树模型做的是明确的监督学习：

```text
用训练标签拟合参数
用 validation/test RMSE 评估泛化
```

LLM 更适合做建模助手：

```text
理解字段含义
建议特征工程
发现数据清洗问题
生成 pandas/sklearn 代码
解释模型和误差
整理实验记录
```

它可以增强建模流程，但不应直接替代监督学习模型本身。

#### Q29: 当前任务的推荐建模优先级是什么？

对这个标准结构化表格回归任务：

```text
1. Ridge regression：教学清晰，可解释性好
2. CatBoost / LightGBM：实际效果大概率更强
3. TabPFN / tabular foundation model：可以实验
4. NN：可尝试，但需要更多调参
5. 普通 LLM prompt 直接预测：不推荐作为主模型
```

## 6. Homework 2025 细节

#### Q30: `engine_displacement`、`model_year` 这类数值特征需要额外处理吗？

对 2025 cohort 的 02-regression homework，按题目要求不需要额外缩放、中心化或转成 `age`。题目明确要求使用这些列：

```text
engine_displacement
horsepower
vehicle_weight
model_year
fuel_efficiency_mpg
```

所以 baseline 应该直接使用：

```python
base = ['engine_displacement', 'horsepower', 'vehicle_weight', 'model_year']
```

例如 `model_year -> age = 2023 - model_year` 在直觉上没问题，但这属于额外 feature engineering。对普通无正则线性回归，在有 intercept 的情况下，`model_year` 和 `age` 这种线性平移通常预测近似等价；但对 ridge regularization，特征平移会改变权重惩罚方式，从而影响结果。

为了对齐 homework 选项，应先按题意使用原始 `model_year`。真实项目里可以再实验：

```text
model_year -> age
数值标准化
非线性变换
交互特征
```

但这些实验都应该通过 validation RMSE 判断是否有效。

#### Q31: 如果用 mean 填充缺失值，validation/test 应该用各自的 mean 吗？

不应该。只要填充值是从数据中统计出来的，就必须只从 training set 计算。

正确写法：

```python
mean_hp = df_train['horsepower'].mean()

X_train = prepare_X(df_train, mean_hp)
X_val = prepare_X(df_val, mean_hp)
X_test = prepare_X(df_test, mean_hp)
```

错误写法：

```python
X_val = prepare_X(df_val, df_val['horsepower'].mean())
X_test = prepare_X(df_test, df_test['horsepower'].mean())
```

原因是后者偷看了 validation/test 的整体统计信息，属于 data leakage。真实预测时，新数据的整体均值通常是未知的，预处理参数必须从训练集学到，再应用到 validation/test。

如果填 0：

```python
X_val = prepare_X(df_val, 0)
X_test = prepare_X(df_test, 0)
```

则没有这个问题，因为 `0` 是固定规则，不是从 validation/test 统计出来的。

#### Q32: 为什么最终 test 前要合并 train 和 validation？

因为 validation set 的职责是帮助做选择，例如：

```text
选择缺失值填 0 还是填 mean
选择 regularization 参数 r
比较不同特征工程方案
```

当方案已经确定后，validation set 的调参任务完成。此时可以把 train 和 validation 合并，用更多数据重新训练最终模型：

```python
df_full_train = pd.concat([df_train, df_val])
df_full_train = df_full_train.reset_index(drop=True)

y_full_train = np.concatenate([y_train, y_val])

X_full_train = prepare_X(df_full_train, 0)
w0, w = train_linear_regression_reg(X_full_train, y_full_train, r=0.001)
```

然后只在 test set 上做最终评估：

```python
X_test = prepare_X(df_test, 0)
y_pred = w0 + X_test.dot(w)

rmse(y_test, y_pred)
```

不能把 test 也合并进训练，因为 test 的作用是模拟从未见过的新数据，做最终验收。如果反复用 test 结果改模型，test 就退化成新的 validation set，最终评估会偏乐观。

一句话：

```text
validation 用来选择方案；方案确定后可以加入训练。
test 只用于最终评估，不能参与训练或调参。
```

### 7. 学习主线总结

这个 Chapter 2 的核心主线是：

```text
1. 明确预测目标：MSRP，不是二手车价格
2. 观察目标分布：价格长尾
3. 对目标做 log1p 变换
4. 切分 train/validation/test
5. 用线性回归建立 baseline
6. 做特征工程：age、one-hot、缺失值处理
7. 发现特征增多导致矩阵不稳定
8. 用 regularization 稳定模型
9. 用 validation RMSE 选择参数
10. 最后在 test set 上评估
```

一句话：

```text
这章不是只在学线性回归公式，而是在学一个完整的 tabular regression 工作流。
```

## 03-classification

### 1. DictVectorizer：fit vs transform

```python
dv = DictVectorizer(sparse=False)

# train：fit + transform，决定列名和顺序
X_train = dv.fit_transform(df_train[...].to_dict(orient='records'))

# val / test：只用 transform，用同一套列
X_val   = dv.transform(df_val[...].to_dict(orient='records'))
X_test  = dv.transform(df_test[...].to_dict(orient='records'))
```

`fit` = 扫描数据，记住所有 category=value 组合（45 列固定不再变）。  
`transform` = 按这些列逐行填值。  
val/test 如果 fit 了，列数可能不同，模型直接报错。

`to_dict(orient='records')` 把每行转成一个 dict：`{'contract': 'two_year', 'tenure': 72, ...}`，DictVectorizer 逐 key-value 编码。

### 2. 编码规则

| 值的类型 | 处理 |
|---|---|
| categorical（`contract='two_year'`） | one-hot：该列=1，同类其他=0 |
| numerical（`tenure=72`） | 原样保留 |

DictVectorizer 不丢参考类别（和 `pd.get_dummies(drop_first=True)` 不同）。每个 category 值都有单独一列，信息冗余但对有正则化的线性模型不是大问题。

### 3. 截距和权重

```python
w0 = model.intercept_[0]   # 截距，shape (1,) 的数组取第一个
w  = model.coef_[0]        # 权重，shape (1, n) 取第一行
```

二分类只有一个决策边界，但 sklearn 为了 API 统一始终返回数组。

### 4. Mutual Information vs Correlation

| | Correlation | Mutual Information |
|---|---|---|
| 捕获的关系 | 仅线性 | 任意（线性、非线性） |
| 引用函数 | `.corr()` / `.corrwith()` | `mutual_info_score`（离散）/ `mutual_info_classif`（连续） |
| 适用特征 | 数值型 | 数值型 + 类别型均可 |

MI 核心公式：`I(X;Y) = H(Y) - H(Y|X)` — 知道 X 后 Y 的不确定性减少了多少。知道 contract 减少的 uncertainty 比知道 gender 多得多。

contract MI 最高的原因：MI 综合了**效应量 × 覆盖人群**——electronic_check 风险比虽高但只覆盖 33% 客户，contract 覆盖全量且各类别 churn rate 差异大。

### 5. 特征尺度和 solver 的坑

这是 homework Q4-Q6 的核心坑。

`annual_income`（~40000-120000）和其他特征（0-10）差 4-5 个数量级。

```python
# liblinear + 未缩放：acc = 0.70，调 C 完全没用
model = LogisticRegression(solver='liblinear', C=1.0, random_state=42)

# lbfgs + 未缩放：acc = 0.85，数值稳定性更好
model = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)
```

不同 C 全给出相同结果的原因：特征尺度差距太大，大值特征的梯度信号 ~1e-5，优化器根本不动它。正则化在尺度瓶颈面前无效。

**解法**：one-hot 之前对 numerical 做 StandardScaler。

### 6. StandardScaler 的正确用法

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_train[numerical] = scaler.fit_transform(df_train[numerical])
df_val[numerical]   = scaler.transform(df_val[numerical])
df_test[numerical]  = scaler.transform(df_test[numerical])

# 然后再做 DictVectorizer
X_train = dv.fit_transform(df_train[...].to_dict(orient='records'))
```

只标准化 numerical 列，categorical 的 one-hot（0/1）不需要。  
和 DictVectorizer 一样：**fit on train only，transform on everything**。

### 7. 小模型近似大模型

contract + tenure + monthlycharges（3 特征 → 5 one-hot 列）精度 79.6%，全量 45 特征的精度 80.6%。差距不到 1pp。

```text
可解释性 > 特征堆砌。能用三五个变量解释清楚的模型，
比 45 维黑箱更有价值。
```

### 8. 模型上线用法

```python
customer = dicts_test[-1]                      # 取一个客户
X_small = dv.transform([customer])             # 包一层 [list]，用同一套 dv 编码
prob = model.predict_proba(X_small)[0, 1]     # predict_proba 返回 (n, 2)，第 1 列是 P(churn)
```

### 9. 逻辑回归 = 线性回归 + sigmoid

看代码：

```python
def linear_regression(xi):
    result = w0
    for j in range(len(w)):
        result = result + xi[j] * w[j]
    return result                    # 任意实数

def logistic_regression(xi):
    score = w0
    for j in range(len(w)):
        score = score + xi[j] * w[j]
    result = sigmoid(score)
    return result                    # 压缩到 [0, 1]，概率
```

唯一的区别是套了一层 sigmoid。逻辑回归名字里的 "regression" 没叫错——底层做的确实就是线性回归，sigmoid 只负责把实数映射成概率。

这就是为什么 02 章先学线性回归再学分类：模型结构没变，变的只是输出层的解释方式。线性回归的 score 直接当预测值用，逻辑回归把 score 送进 sigmoid 变成 P(churn)。

### 学习主线

```text
1. 数据清洗（列名统一、缺失值、类型转换）
2. EDA（churn rate、risk ratio、mutual info、correlation）
3. 切分 train/val/test
4. DictVectorizer 做 one-hot
5. Logistic regression 训练
6. 模型解释（系数含义、小模型）
7. 最终在 test set 评估
8. 对单条数据做预测（生产环境用法）
9. 理解逻辑回归本质 = 线性回归 + sigmoid
```

## 04-evaluation 核心问答

### Q1: Accuracy / Precision / Recall 分别看什么？

```text
accuracy  = (TP + TN) / total         # 总体预测对的比例
precision = TP / (TP + FP)            # 预测为正的人里，多少真为正
recall    = TP / (TP + FN)            # 真实为正的人里，多少被抓到
FPR       = FP / (FP + TN)            # 真实为负的人里，多少被误报为正
TPR       = recall
```

类别不均衡时，accuracy 可能误导。比如全预测负类也能有不低准确率。

### Q2: `Counter(y_pred >= 1.0)` 和 `1 - y_val.mean()` 是什么？

`y_pred >= 1.0` 基本全是 `False`，等价于 dummy model：所有人都预测为负类。

若 `y_val` 是 0/1：

```python
y_val.mean()      # 正类比例
1 - y_val.mean()  # 负类比例，也就是全预测负类的 accuracy
```

### Q3: AUC 是什么面积？

AUC 是 ROC 曲线下面积，不是两条曲线之间的“豆夹形状”。

```text
x 轴 = FPR
y 轴 = TPR
AUC = ∫ TPR d(FPR)
```

直觉：随机抽一个正样本和一个负样本，模型给正样本更高分的概率。

```text
AUC = 1.0 完美排序
AUC = 0.5 随机排序
AUC < 0.5 方向反了
```

### Q4: ROC/AUC 只适合 Logistic Regression 吗？

不是。只要是二分类模型，并能输出 score/probability，就能算：

```text
ROC/AUC
precision
recall
accuracy
F1
confusion matrix
```

适用于 logistic regression、decision tree、random forest、boosting、NN 等。

AUC 看整体排序能力；precision/recall/accuracy 看某个 threshold 下的决策效果。实际项目通常都要看。

### Q5: 2025 HW04 的 Q1 “use numerical variable as score” 是什么意思？

数据是 `course_lead_scoring.csv`，目标是：

```text
converted
```

意思是：不训练模型，直接把每个 numerical column 当成预测分数，和 `converted` 算 AUC。

```python
for col in numerical:
    auc = roc_auc_score(y_train, df_train[col])
    if auc < 0.5:
        auc = roc_auc_score(y_train, -df_train[col])
    print(col, auc)
```

这里 `df_train[col]` 是 score，不一定是概率，也不需要归一化。AUC 只看排序。

### Q6: HW04 Q2 训练 LogisticRegression 要不要 scale？

这份 homework 实测需要 scale numerical，才能得到选项里的约 `0.92`。

```text
未 scale AUC ≈ 0.817
scale 后 AUC ≈ 0.921
```

原因：`annual_income` 和其他数值特征尺度差很多，LogisticRegression + 正则化对尺度敏感。

正确方式：

```python
scaler = StandardScaler()
df_train[numerical] = scaler.fit_transform(df_train[numerical])
df_val[numerical] = scaler.transform(df_val[numerical])
df_test[numerical] = scaler.transform(df_test[numerical])
```

只在 train 上 `fit`，val/test 只能 `transform`。

### Q8: KFold 双层循环在做什么？

外层遍历不同 `C`，内层做 5-fold CV。

```python
for C in [...]:
    kfold = KFold(n_splits=5, shuffle=True, random_state=1)

    for train_idx, val_idx in kfold.split(df_full_train):
        ...
```

要点：

```text
KFold(...) 只是定义切分规则
split(df_full_train) 才生成 5 组 train/val 索引
random_state 固定，所以每个 C 用同一套 5 折，比较公平
输出 mean AUC +- std，看平均效果和稳定性
```

## 06-trees 核心问答

### Q1: Gini impurity 数学概念是什么？如何用它建立一个树节点？

Gini impurity 衡量一个节点里的类别有多混杂：

```text
Gini = 1 - sum(p_i^2)
```

二分类时：

```text
Gini = 1 - p_ok^2 - p_default^2
```

直觉：

```text
Gini = 0   最纯，全是同一类
Gini = 0.5 二分类最混杂，ok/default 各一半
```

简化贷款数据：

| id | debt | assets | status |
|---:|---:|---:|---|
| 1 | 500 | 9000 | ok |
| 2 | 800 | 7000 | ok |
| 3 | 1200 | 5000 | ok |
| 4 | 1500 | 4500 | default |
| 5 | 2200 | 3000 | default |
| 6 | 3000 | 2000 | default |
| 7 | 900 | 1000 | default |
| 8 | 2600 | 6000 | ok |

根节点：

```text
ok = 4
default = 4
Gini_root = 1 - (4/8)^2 - (4/8)^2 = 0.5
```

候选切分 1：

```text
assets <= 3750
```

```text
left:  ok = 0, default = 3
Gini_left = 0

right: ok = 4, default = 1
Gini_right = 1 - (4/5)^2 - (1/5)^2 = 0.32

weighted_gini = 3/8 * 0 + 5/8 * 0.32 = 0.20
```

候选切分 2：

```text
debt <= 1750
```

```text
left:  ok = 3, default = 2
Gini_left = 1 - (3/5)^2 - (2/5)^2 = 0.48

right: ok = 1, default = 2
Gini_right = 1 - (1/3)^2 - (2/3)^2 = 0.444

weighted_gini = 5/8 * 0.48 + 3/8 * 0.444 = 0.467
```

比较后：

```text
assets <= 3750    weighted_gini = 0.20
debt <= 1750      weighted_gini = 0.467
```

所以当前节点会选择 `assets <= 3750`，因为它让左右分支更纯。

### Q2: 建节点时是不是“候选特征 × 候选阈值”两层筛选？

对。当前节点里有一批训练样本，算法会做局部搜索：

```text
for feature in candidate_features:
    collect thresholds for this feature

    for threshold in thresholds:
        split into left/right
        compute weighted_gini

choose feature + threshold with minimum weighted_gini
```

数值特征的候选阈值通常来自排序后的相邻取值中点；类别特征一般会先 one-hot，或按类别分组处理。

选出来的不是“当前节点”，而是**当前节点的切分规则**：

```text
当前节点:
    if assets <= 3750:
        go left
    else:
        go right
```

左右子节点继续重复同样过程。决策树是贪心算法：每个节点只选当前最好的 split，不保证整棵树全局最优。

### Q4: 数据大时真的逐个 threshold 测试吗？

精确树概念上会试候选阈值，但实现上不是每次重新扫全表。常见优化：

```text
1. 对特征排序
2. 从左到右扫描
3. 增量维护 left/right 类别计数
4. 快速更新 weighted_gini
```

更大数据里，XGBoost/LightGBM 常用 histogram/binning，只在桶边界找 split。

### Q5: Gini 很小或为 0 就停止切分吗？

通常会停止，尤其 `Gini = 0` 表示节点已经全是同一类。但实际停止还受这些条件控制：

```text
max_depth
min_samples_leaf
min_samples_split
min_impurity_decrease
```

只追求训练集叶子纯度，容易过拟合。

### Q6: `min_samples_split` 和 `min_samples_leaf` 的关键区别？

```text
min_samples_split:
    父节点至少有多少样本，才允许尝试继续切

min_samples_leaf:
    切完后，每个子节点至少要有多少样本
```

前者控制“能不能尝试切”，后者控制“切出来是否合法”。后者更直接防止生成很小的叶子。

### Q7: 叶子节点最后如何预测？

分类树训练完成后，每个叶子在**硬分类逻辑上**可以等效成一个标签节点。新样本走到这个叶子后，`predict()` 输出该叶子的多数类。

但模型内部不会只存一个标签，通常还会保留训练样本落到该叶子后的类别计数/分布：

```text
ok: 18
default: 2
```

所以：

```text
predict()       -> ok
predict_proba() -> P(ok)=0.9, P(default)=0.1
```

这些分布会用于概率预测、AUC 计算、模型解释和判断这个叶子的预测是否稳定。也就是说：

```text
对最终硬分类:
    叶子等效为一个标签

对模型内部和概率输出:
    叶子保留训练真实分布
```

### Q8: 为什么真实数据往往要切好几层？

单个特征和阈值通常只能粗分。多层路径组合后，才形成多因素规则：

```text
debt 高 + assets 低 + records=yes -> default 风险高
```

树天然能表达非线性和特征交互；但越深越容易过拟合，要靠 validation 和复杂度参数控制。

### Q9: DT、RF、线性模型、NN 怎么定位？

```text
线性模型:
    一个整体线性公式，可解释，但表达能力有限

决策树:
    if-else 规则，能表达非线性和特征交互，单棵树容易过拟合

随机森林:
    多棵树投票/平均，比单棵树稳定，可解释性略下降

神经网络:
    表达能力更强，需要更多数据和调参，可解释性弱
```

DT 不是简单线性算法，也没到 NN 那么复杂；RF 是多棵 DT 的集成。

### Q10: Random Forest 每棵树是如何建立的？随机性在哪里？

RF 是很多棵 decision tree 的集成。每棵树仍然按 Gini/Entropy 找 split，但训练时引入两类随机性：

```text
1. 行随机:
   每棵树从训练集 bootstrap sample

2. 特征随机:
   每个节点只随机看一部分特征，再在这部分特征里找最优 split
```

scikit-learn 当前默认大致是：

```text
bootstrap = True
max_features = sqrt
criterion = gini
```

如果 one-hot 后有 100 个特征，`max_features=sqrt` 表示每个节点大约随机看 10 个特征。注意这是**每个节点重新抽特征**，不是一棵树固定一批特征。

### Q11: bootstrap 行采样会去重吗？

不会。bootstrap 是有放回抽样，重复行会保留。

```text
原始行: [A, B, C, D, E]

tree_1 sample:
[A, A, C, E, E]
```

重复行等效于在这棵树里权重更高。没被某棵树抽到的行叫 OOB samples，可以用于 out-of-bag 评估，但它不是传统 train/validation split。

### Q12: 每个节点只随机看一部分特征，会不会错过最优 split？

会。单棵树确实可能因为没看到最强特征，选不到当前节点的全局最优 split。

RF 故意这么做，因为它追求的不是每棵树最优，而是让树之间差异更大：

```text
每棵树都看全部特征:
    强特征反复被选中，树长得很像，错误也相似

每个节点随机看部分特征:
    单树弱一点，但树之间相关性更低，集成后更稳
```

所以 RF 是“随机候选集 + 候选集内最优”，不是随便切。

### Q13: bias 和 variance 分别是什么？RF 主要解决哪个？

```text
bias 高:
    模型太简单，训练集都学不好，欠拟合

variance 高:
    模型太敏感，训练集很好，验证集差，过拟合
```

树模型里：

```text
浅树:
    bias 高，variance 低

深树:
    bias 低，variance 高

Random Forest:
    通过 bootstrap + 特征随机 + 多树平均，主要降低 variance
```

RF 牺牲一点单棵树的局部最优性，换整体模型更稳定的泛化表现。

### Q14: Gradient Boosting 的核心直觉是什么？

Gradient Boosting 将多棵树串行相加：

```text
F_m(x) = F_(m-1)(x) + eta * tree_m(x)
```

每棵新树负责修正当前整体模型仍然存在的错误：

```text
当前整体模型 -> 计算当前错误 -> 新树学习修正 -> 加入新树 -> 重复
```

与 RF 的关键区别：

```text
Random Forest:
    树独立训练，最后平均，主要降低 variance

Gradient Boosting:
    树顺序训练，后一棵依赖当前整体模型，逐轮降低 loss
```

### Q15: 每轮新树修正的是上一棵树吗？

不是。它修正的是之前所有树组合后的整体模型：

```text
F_(m-1)(x)
= F_0(x)
  + eta * tree_1(x)
  + ...
  + eta * tree_(m-1)(x)
```

平方误差下，第 `m` 棵树的训练目标是当前整体模型留下的残差：

```text
r_i = y_i - F_(m-1)(x_i)
tree_m(x_i) ≈ r_i
```

例如真实值为 `80`，当前整体预测为 `100`：

```text
r = 80 - 100 = -20
```

因此新树应提供负修正，让整体预测下降。

### Q16: 初始模型、残差和平方误差之间是什么关系？

`F_0(x)` 是加入第一棵树前的初始预测。平方误差回归里，它通常是训练目标均值，对所有样本先预测同一个值。

对样本 `i`：

```text
当前预测 = F(x_i)
残差 r_i = y_i - F(x_i)
平方误差 = [y_i - F(x_i)]^2 = r_i^2
```

所以残差表示当前模型还未解释的部分：

```text
r_i > 0: 当前预测低了，需要向上修正
r_i < 0: 当前预测高了，需要向下修正
```

### Q17: 为什么平方误差下，残差就是 loss 的负梯度？

使用带 `1/2` 的平方误差：

```text
L_i(F) = 1/2 * [y_i - F(x_i)]^2
```

对当前预测 `F(x_i)` 求导，因为模型能改变的是预测值，真实值 `y_i` 是固定的：

```text
∂L_i / ∂F(x_i)
= 1/2 * 2[y_i - F(x_i)] * (-1)
= F(x_i) - y_i
```

因此负梯度为：

```text
-∂L_i / ∂F(x_i)
= y_i - F(x_i)
= r_i
```

所以在平方误差下，新树“学习残差”等价于“学习当前 loss 的负梯度”。如果不写 `1/2`，负梯度是 `2r_i`，方向相同。

### Q18: 为什么沿负梯度方向修改预测能降低 loss？

对一个很小的预测改变量 `Delta F`，一阶近似为：

```text
L(F + Delta F) ≈ L(F) + L'(F) * Delta F
```

选择：

```text
Delta F = -eta * L'(F), eta > 0
```

则 loss 的近似变化为：

```text
L'(F) * Delta F
= -eta * [L'(F)]^2
<= 0
```

因此负梯度是局部下降方向。Gradient Boosting 训练新树去近似这个方向：

```text
tree_m(x_i) ≈ -∂L_i / ∂F(x_i)
```

平方误差下它恰好等于残差；其他 loss 下通常不再是普通残差。`eta` 需要较小，因为步长过大时，一阶近似可能不再准确并越过最低点。

### Q19: 06 章有哪些值得记录的踩坑？

`99999999` 这种极大值通常是缺失值占位符，会污染 mean/std；`0` 不一定异常，要看业务含义。

新版本 XGBoost 的 `feature_names` 需要 string list，不接受 NumPy ndarray：

```python
features = list(dv.get_feature_names_out())

dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)
```
