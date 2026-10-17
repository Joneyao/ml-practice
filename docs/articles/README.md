# 配套文章同步目录

本目录存放《机器学习实战》系列文章的 markdown 同步副本（与公众号 `easy_ai` 双发）。
文章写作走 writing-articles skill，完成后同步进本目录。当前仅为占位说明，尚无
正文文件。

## 14 篇规划表

| 编号 | 标题 | 代码模块 | 数据集 | 状态 |
|------|------|----------|--------|------|
| A1 | 别急着建模：一份真实数据能有多脏 | `data_cleaning/explore.py` | Titanic | 代码就绪，待写文章 |
| A2 | 缺失值不是删掉就完事：填补策略的代价 | `data_cleaning/imputation.py` | Titanic | 文章已写（见 `article_02_imputation-strategy-cost.md`） |
| A3 | 异常值：噪声，还是真实的极端情况？ | `data_cleaning/outliers.py` | House Prices (Ames) | 代码就绪，待写文章 |
| B1 | 从一列数据里，能榨出几个特征？ | `feature_engineering/`（待建） | Titanic | 规划中 |
| B2 | 类别变量编码：One-Hot 之外的世界 | `feature_engineering/`（待建） | Telco Customer Churn | 规划中 |
| B3 | 用 Pipeline 把特征工程和模型拧成一股绳 | `feature_engineering/`（待建） | Telco Customer Churn | 规划中 |
| C1 | 回归问题：从线性回归到正则化（Ridge/Lasso） | `modeling/`（待建） | House Prices (Ames) | 规划中 |
| C2 | 分类问题：选算法不是看榜，是看场景 | `modeling/`（待建） | Telco Customer Churn | 规划中 |
| C3 | 不均衡数据：当"坏人"只占 2% | `modeling/`（待建） | Credit Card Fraud Detection | 规划中 |
| D1 | 交叉验证不是跑一遍就完事：K-Fold 的正确姿势 | `tuning_evaluation/`（待建） | House Prices (Ames) | 规划中 |
| D2 | 调参的性价比：GridSearch、RandomSearch 还是贝叶斯优化 | `tuning_evaluation/`（待建） | Telco Customer Churn | 规划中 |
| E1 | 训好的模型怎么用起来：从 pickle 到一个能调用的接口 | `deployment/`（待建） | Telco Customer Churn | 规划中 |
| F1 | 从 sklearn 到 PyTorch：换个框架，同样的损失和梯度下降 | `deep_learning/`（待建） | MNIST | 规划中 |
| F2 | 图像分类实战：CNN 入门与一个真实的小项目 | `deep_learning/`（待建） | CIFAR-10 / Fashion-MNIST | 规划中 |

详细大纲和每篇的揭示性时刻，参见设计文档
`docs/superpowers/specs/2026-10-16-ml-practice-series-design.md`（主仓库）。
