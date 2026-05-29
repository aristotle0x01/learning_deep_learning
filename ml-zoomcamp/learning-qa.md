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

