# learning_deep_learning（NN/LLM 主线与传统 ML 支线学习策略）

## 1. 核心结论与执行顺序

| 问题         | 结论                                                  | 置信度 |
| ------------ | ----------------------------------------------------- | ------ |
| 总目标       | 以 NN/LLM 理解为主，顺带建立传统 ML 快速适配能力      | 高     |
| 学习方法     | 代码实操优先，数学按卡点反向补                        | 高     |
| 深度学习主线 | PyTorch + Zero to Hero + Raschka                      | 高     |
| 传统 ML 定位 | 工具箱/适配层：知道何时不用 LLM/NN，能快速做 baseline | 高     |
| LLM 应用主线 | LLM Zoomcamp RAG/Eval，在模型直觉建立之后推进         | 中     |

第一原则：NN/LLM 是主线；传统 ML 是必要但压缩的支线。传统 ML 的价值在于杀鸡不用牛刀。
第二原则：这是业余时间的长期目标，以年为单位推进，不赶进度、不断线即可。

### 执行顺序

传统 ML 全部前置，NN 主线一气呵成。

| 顺序 | 内容                                   | 目的                                            | 完成标准                                                   |
| ---- | -------------------------------------- | ----------------------------------------------- | ---------------------------------------------------------- |
| 1    | 3Blue1Brown 前 4 节                    | 建立 NN、loss、gradient、backprop 几何直觉      | 能用白话解释反向传播在做什么                               |
| 2    | ML Zoomcamp Module 1                   | Python、pandas、numpy 环境准备                  | 环境就绪，能跑 notebook                                    |
| 3    | ML Zoomcamp Module 2 + HOML Ch2/Ch4    | 回归、特征、train/test、RMSE、pipeline          | 产出回归 notebook                                          |
| 4    | ML Zoomcamp Module 3 + HOML Ch3        | 分类、混淆矩阵、precision/recall、F1            | 产出二分类 notebook                                        |
| 5    | ML Zoomcamp Module 4                   | threshold、ROC/AUC、交叉验证                    | 产出一份评估指标笔记                                       |
| 6    | ML Zoomcamp Module 6 + HOML Ch6/Ch7    | 决策树、随机森林、GBDT、XGBoost                 | 产出树模型 notebook                                        |
| 7    | ML Zoomcamp Module 5 + Midterm Project | 模型保存、API、端到端项目                       | 一个完整项目：清洗→特征→训练→验证→保存→推理                |
| 8    | Zero to Hero MicroGrad                 | 手写 autograd，理解计算图和反向传播             | 能独立写出 Value 类和 backward()，不查代码                 |
| 9    | Zero to Hero Makemore 前几讲           | 理解 embedding、MLP、tensor shape、训练循环     | 跑通 MLP 语言模型，能解释每个 tensor 维度的含义            |
| 10   | fast.ai 前 2-4 课                      | 跑通真实 DL 任务                                | 独立跑通一个图像分类和一个文本分类                         |
| 11   | Raschka Ch2-Ch5                        | 系统理解 tokenizer、attention、GPT、pretraining | 能画出 GPT 架构图，解释 Q/K/V 和 self-attention 的计算过程 |
| 12   | Raschka Ch7 / LoRA 按需                | instruction tuning / PEFT                       | 理解微调和全量训练的差异和适用场景                         |
| 13   | LLM Zoomcamp RAG + Eval                | 工作场景应用原型                                | 跑通一个 RAG 原型，含检索、生成、评估                      |

## 2. 资源评估

| 资源                                | 最适合承担的角色                | 可参考性 | 可复制动手性 | 原理领会效率 | 判断                       |
| ----------------------------------- | ------------------------------- | -------: | -----------: | -----------: | -------------------------- |
| Karpathy Zero to Hero               | NN 原理实践主轴                 |      3/5 |          4/5 |          5/5 | 先试跑 MicroGrad           |
| Sebastian Raschka LLMs from Scratch | LLM 内部结构系统参考            |      5/5 |          4/5 |          4/5 | 放在 MicroGrad/Makemore 后 |
| HOML                                | 传统 ML workflow 与评估体系参考 |      5/5 |          4/5 |          4/5 | 传统 ML 的主参考书         |
| ML Zoomcamp                         | 传统 ML 作业和项目闭环          |      3/5 |          5/5 |          3/5 | 传统 ML 的短周期实操支线   |
| fast.ai                             | 深度学习应用层入口              |      3/5 |          4/5 |          3/5 | 用于快速跑通 DL 项目       |
| LLM Zoomcamp                        | RAG/LLM 工程应用                |      4/5 |          5/5 |          3/5 | 用于工作场景原型           |

## 3. 传统 ML 支线

结论：`ML Zoomcamp 70% + HOML 30%`。

目的不是深入传统 ML，而是掌握一组够用的判断和实现能力：特征、标签、训练/验证、评估、过拟合、树模型、部署。以后工作里遇到结构化数据、标签稳定、规则/模型可解释性要求高的场景，可以快速适配，不必默认上 LLM。

| 阶段 | ML Zoomcamp                    | HOML 补充           | 目标                                       |
| ---- | ------------------------------ | ------------------- | ------------------------------------------ |
| 0    | Module 1                       | Ch1 粗读            | Python、pandas、numpy、基本线代            |
| 1    | Module 2 Linear Regression     | Ch2 + Ch4 相关部分  | 回归、特征、train/test、RMSE/MAE、pipeline |
| 2    | Module 3 Binary Classification | Ch3                 | 分类、混淆矩阵、precision/recall、F1       |
| 3    | Module 4 Evaluation            | Ch3 再读            | threshold、ROC/AUC、交叉验证               |
| 4    | Module 6 Tree-Based Models     | Ch6 + Ch7           | 决策树、随机森林、GBDT、XGBoost            |
| 5    | Module 5 Deployment            | HOML Ch2 末尾按需查 | 模型保存、API、Docker、推理服务            |
| 6    | Midterm Project                | 按卡点查 HOML       | 端到端传统 ML 项目                         |

不完整刷 HOML。重点是 Ch2、Ch3、Ch6、Ch7。
不完整刷 ML Zoomcamp。重点是前 6 个模块和一个项目。
不把传统 ML 拉成长线。完成一轮项目闭环后，回到 NN/LLM 主线。

传统 ML 最低产出：

1. 一个回归 notebook。
2. 一个二分类 notebook。
3. 一个树模型/XGBoost notebook。
4. 一份评估指标笔记。
5. 一个端到端项目：清洗、特征、训练、验证、保存模型、API 或脚本推理。

## 4. DL / LLM 主线

Zero to Hero 是 NN 原理手术台。
Raschka 是 LLM 内部结构参考书。
fast.ai 是应用层快速启动器。
LLM Zoomcamp 是工程应用和 RAG/Eval 主线。

| 顺序 | 内容                                | 目的                                            |
| ---- | ----------------------------------- | ----------------------------------------------- |
| 1    | 3Blue1Brown Neural Networks 前 4 节 | 建立神经网络、loss、gradient、backprop 直觉     |
| 2    | Zero to Hero MicroGrad              | 手写 autograd，理解计算图和反向传播             |
| 3    | Zero to Hero Makemore 前几讲        | 理解 embedding、MLP、tensor shape、训练循环     |
| 4    | fast.ai 前 2-4 课                   | 跑通一个真实 DL 任务                            |
| 5    | Raschka Ch2-Ch5                     | 系统理解 tokenizer、attention、GPT、pretraining |
| 6    | Raschka Ch7 / LoRA 按需             | 理解 instruction tuning / PEFT                  |
| 7    | LLM Zoomcamp RAG + Eval             | 做工作场景应用原型                              |

## 5. 数学反向补充

| 卡点             | 回查内容                                  |
| ---------------- | ----------------------------------------- |
| tensor shape 错  | 矩阵乘法、维度变换、broadcasting          |
| loss 不下降      | 导数、梯度下降、学习率、optimizer         |
| 分类指标混乱     | 概率、混淆矩阵、precision/recall、ROC/AUC |
| 过拟合           | 正则化、交叉验证、偏差-方差               |
| embedding 相似度 | 点积、余弦相似度、向量归一化              |
| attention 不清楚 | Q/K/V、softmax、mask、矩阵乘法            |
| RAG 不稳定       | 召回率、排序、重排、离线评估              |

数学不是前置门槛，是调试工具。

## 6. 工作场景迁移

主线场景：物料审核 / 去重 / 合规审查。

| 层级         | 方案                             | 产出                         |
| ------------ | -------------------------------- | ---------------------------- |
| 规则层       | 黑白名单、字段校验、阈值、规则树 | 可解释审核基线               |
| 传统 ML 层   | Logistic Regression / XGBoost    | 异常、错误分类、重复风险评分 |
| Embedding 层 | 文本向量化、相似度检索           | 去重候选集                   |
| RAG 层       | 检索制度、历史案例、标准文档     | 可追溯辅助判断               |
| 人工复核层   | 高风险结果进入人工队列           | 控制误杀和漏判               |
| 审计层       | 保存输入、规则、分数、证据、结论 | 可复盘、可治理               |

## 7. 不做

| 不做                            | 原因                       |
| ------------------------------- | -------------------------- |
| 不从头重刷概率和线代            | 低反馈                     |
| 不同时完整刷所有资源            | 会制造虚假勤奋             |
| 不把传统 ML 拉成长主线          | 当前主目标是 NN/LLM        |
| 不把课程完成率当目标            | 目标是产物和判断力         |
| 不把 LLM API 调用等同于 AI 能力 | 必须补评估、检索、模型直觉 |
| 不一开始训练大模型              | 当前收益低、成本高         |

## 来源

| 资源                      | 链接                                                        |
| ------------------------- | ----------------------------------------------------------- |
| ML Zoomcamp               | https://datatalks.club/docs/courses/ml-zoomcamp/curriculum/ |
| HOML 3rd                  | https://github.com/ageron/handson-ml3                       |
| Zero to Hero              | https://github.com/karpathy/nn-zero-to-hero                 |
| MicroGrad                 | https://github.com/karpathy/micrograd                       |
| Raschka LLMs from Scratch | https://github.com/rasbt/LLMs-from-scratch                  |
| fast.ai                   | https://course.fast.ai/index.html                           |
| LLM Zoomcamp              | https://datatalks.club/blog/llm-zoomcamp.html               |
