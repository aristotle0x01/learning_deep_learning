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

### 6. 学习主线总结

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
