# 参考资料

本仓库的代码只借鉴经典教材、官方文档和公开课的思路与方法论，所有函数均从原理
重新实现，不摘抄任何教程或开源项目源码。

## 经典书籍/课程（工程方法论的深度锚点）

| 书 / 课 | 一句话定位 | 主要参考章节 |
|---------|-----------|-------------|
| Aurélien Géron《Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow》(3rd ed.) | 完整工程流程的标准模板，Ames 房价数据集也是该书案例之一 | 第 2 章 End-to-End Machine Learning Project——本批 A1-A3 对应"数据探查与清洗"部分 |
| Abhishek Thakur《Approaching (Almost) Any Machine Learning Problem》（开源免费书，[GitHub](https://github.com/abhishekkrthakur/approachingalmost)） | 偏工程实战，专讲交叉验证策略和特征工程坑 | 对应后续阶段 B（特征工程）/D（调参评估） |
| 吴恩达 CS229 / Machine Learning Specialization | 延续《从零开始学机器学习》入门系列的理论锚点 | 按需查阅 |

## 官方文档（代码实现的权威依据）

**原则：只借鉴思路和接口设计，代码自己从原理重新实现，不摘抄文档或任何教程中的
示例代码。** 与 `ml-from-scratch` 的开发约定一致。

| 文档 | 用途 |
|------|------|
| scikit-learn [Pipelines and composite estimators](https://scikit-learn.org/stable/modules/compose.html) | Pipeline/ColumnTransformer 的标准用法，后续阶段 B3 的核心依据 |
| scikit-learn [`sklearn.impute`](https://scikit-learn.org/stable/modules/impute.html)（`SimpleImputer`/`KNNImputer`） | A2 缺失值填补策略对比的接口参考 |
| pandas 官方文档 [Working with missing data](https://pandas.pydata.org/docs/user_guide/missing_data.html) | A1/A2 缺失值探查与处理的方法论参考 |

## 数据集出处

| 数据集 | 来源 | 获取方式 | 用在哪 |
|--------|------|----------|--------|
| Titanic | [Titanic - Machine Learning from Disaster](https://www.kaggle.com/c/titanic/)（Kaggle 公开数据集） | `seaborn.load_dataset("titanic")` 内置镜像，无需 Kaggle API 凭据 | A1 数据探查、A2 缺失值填补 |
| House Prices (Ames) | Ames housing dataset（Kaggle House Prices 竞赛同源数据） | `sklearn.datasets.fetch_openml(data_id=42165, as_frame=True)` | A3 异常值检测与处理 |


## 第二批补充（特征工程，阶段 B）

**延续本仓库一贯原则：只借鉴思路和接口设计，代码自己从原理重新实现，不
摘抄任何教程或开源项目源码。**

| 文档 | 用途 |
|------|------|
| scikit-learn [Pipelines and composite estimators](https://scikit-learn.org/stable/modules/compose.html) | B3 `ColumnTransformer`/`Pipeline` 组合方式的核心依据 |
| scikit-learn [Encoding categorical features](https://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features) | B2 `OneHotEncoder`/`LabelEncoder` 的接口参考 |
| Abhishek Thakur《Approaching (Almost) Any Machine Learning Problem》 | 交叉验证与特征工程防泄漏思路，B3 揭示性时刻的方法论来源 |

新增数据集出处：

| 数据集 | 来源 | 获取方式 | 用在哪 |
|--------|------|----------|--------|
| Telco Customer Churn | [Kaggle: blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) | `sklearn.datasets.fetch_openml(data_id=42178, as_frame=True)`（OpenML 同源镜像，无需 Kaggle 凭据） | B2 类别编码、B3 Pipeline 数据泄漏复现 |
| Titanic（原始版，含 name/cabin 文本列） | 同 [Titanic - Machine Learning from Disaster](https://www.kaggle.com/c/titanic/) | `sklearn.datasets.fetch_openml(data_id=40945, as_frame=True)` | B1 从文本列构造特征 |


## 第三批补充（建模与算法选择，阶段 C）

**延续本仓库一贯原则：只借鉴思路和接口设计，代码自己从原理重新实现。**

| 文档 | 用途 |
|------|------|
| scikit-learn [Linear Models（Ridge/Lasso）](https://scikit-learn.org/stable/modules/linear_model.html) | C1 正则化回归依据 |
| scikit-learn [Ensemble methods](https://scikit-learn.org/stable/modules/ensemble.html) | C2 RandomForest/GBDT 接口参考 |
| imbalanced-learn [SMOTE 官方文档](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html) | C3 过采样依据 |
| scikit-learn [Metrics: precision/recall/ROC](https://scikit-learn.org/stable/modules/model_evaluation.html) | C3 不均衡评估指标依据 |

新增数据集出处：

| 数据集 | 来源 | 获取方式 | 用在哪 |
|--------|------|----------|--------|
| Credit Card Fraud Detection | [Kaggle: mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) | `sklearn.datasets.fetch_openml(data_id=1597, as_frame=True)`（OpenML 同源镜像，无需凭据） | C3 不均衡分类 |
