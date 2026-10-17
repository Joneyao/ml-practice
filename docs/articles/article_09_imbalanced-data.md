> 作者：轻松学AI
> 日期：2026-10-16
> 风格：通俗科普 + 实操教程
> 状态：待发布

# 机器学习实战 #09：不均衡数据，当"坏人"只占 0.17%

给你一个信用卡欺诈检测模型，准确率 99.83%。听起来是不是很厉害？

我现在就能写出这个模型，一行真正的学习都不需要：**永远预测"这笔交易不是欺诈"。**

真实的信用卡欺诈数据集，284807 笔交易里只有 492 笔是欺诈，占 0.173%。所以一个无脑预测"都不是欺诈"的模型，天然就能答对 99.827% 的样本——因为 99.827% 的样本本来就不是欺诈。它的准确率高得惊人，却一笔欺诈都抓不到，召回率是 0。

这就是不均衡数据最毒的陷阱：**当一类样本极端稀少时，准确率这个最直觉的指标会彻底骗你。** 这一篇讲怎么识破这个陷阱，以及当"坏人"只占千分之一点七时，到底该用什么指标、什么方法。

## 准确率的骗局

先把这个骗局用代码坐实。配套仓库里用 sklearn 的 `DummyClassifier` 复现了这个"什么都不学"的模型：

```python
def naive_accuracy_trap(X, y, seed=0):
    """永远预测多数类：准确率虚高，recall=0。"""
    from sklearn.model_selection import cross_val_predict
    dummy = DummyClassifier(strategy="most_frequent")
    pred = cross_val_predict(dummy, X, y, cv=5)
    from sklearn.metrics import accuracy_score, recall_score, precision_score
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
    }
```

`DummyClassifier(strategy="most_frequent")` 的策略就是永远预测出现最多的那一类。在欺诈数据上跑出来的结果：

![准确率99.8却召回率为0的陷阱](accuracy_trap.png)

准确率 0.998，召回率 0.000。一个零召回的模型，准确率却逼近满分。如果你的评估只看准确率，这个废物模型会光荣"达标"，然后在上线第一天放过所有欺诈交易。

问题的根源在于，准确率把"抓住欺诈"和"放过正常交易"这两件完全不同重要性的事，按样本数量平等地加权平均了。但在欺诈场景里，抓住那 0.173% 的欺诈，才是任务的全部意义；答对那 99.827% 的正常交易，毫无难度也毫无价值。

## 换一把尺子：precision、recall、F1、AUC

既然准确率废了，得换指标。不均衡问题里真正有用的是这几个，先搞清楚它们各自在量什么：

**召回率（recall）**：所有真实的欺诈里，你抓到了多少。漏掉一笔欺诈，召回率就掉一点。对欺诈检测、疾病筛查这种"宁可错杀不可放过"的场景，召回率是命根子。

**精确率（precision）**：你标记为欺诈的交易里，有多少真的是欺诈。误报一笔正常交易为欺诈，精确率就掉一点。对"误报代价高"的场景（比如每次报警都要人工复核、或者直接冻结用户的卡）重要。

**F1**：召回率和精确率的调和平均，一个综合分。两者都高 F1 才高，任何一个塌了 F1 都会被拖下来。

**AUC**：衡量模型"把欺诈排在正常交易前面"的整体能力，不依赖某个具体的判定阈值。它对不均衡相对不敏感，是比准确率公允得多的总体指标。

一句话记忆：**召回率问"漏了多少坏人"，精确率问"冤枉了多少好人"，F1 要两者兼顾，AUC 看整体排序能力。** 准确率在极端不均衡下，一个都不如。

## 两种应对不均衡的办法

识破陷阱之后，怎么让模型真正重视那稀少的一类？常见两条路。

**第一条：调整类别权重（class_weight）。** 不改数据，改模型的"在意程度"。告诉模型：错判一个欺诈样本的代价，远高于错判一个正常样本。sklearn 里一个参数就能开——`class_weight="balanced"`，它会按类别频率的倒数自动加权，稀少的欺诈类获得更高权重。

**第二条：过采样少数类（SMOTE）。** 不改模型，改数据。SMOTE 的思路不是简单复制欺诈样本（那样容易过拟合），而是在已有欺诈样本之间"插值"合成新的欺诈样本，把少数类的数量人为补上来，让两类趋于平衡。

这里有个极其重要、也极其容易踩错的工程细节：**SMOTE 只能在训练折上做，绝不能碰验证折。** 如果先对全量数据过采样再切分验证，合成的样本会同时出现在训练和验证里，造成数据泄漏——这正是上一篇讲过的坑。配套仓库里用 imbalanced-learn 的 `Pipeline`（注意不是 sklearn 的 Pipeline）来保证这一点：

```python
def compare_imbalance_strategies(X, y, seed=0):
    """三种策略对比（不看 accuracy，看 PR/F1/AUC）。SMOTE 只在训练折过采样。"""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    baseline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    class_weight = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    smote = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=seed)),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    rows = [
        {"strategy": "baseline", **_scored(baseline, X, y, cv)},
        {"strategy": "class_weight", **_scored(class_weight, X, y, cv)},
        {"strategy": "smote", **_scored(smote, X, y, cv)},
    ]
    return pd.DataFrame(rows)
```

用 `imblearn.pipeline.Pipeline` 的关键原因：它的 SMOTE 步骤只在每一折的训练阶段触发，验证阶段自动跳过。交叉验证的每一折里，过采样只发生在训练部分，验证部分保持原始的真实分布——这样算出来的分数才如实反映模型在真实不均衡数据上的表现。

## 结果比"越处理越好"要复杂

跑完三种策略，真实结果给我上了一课——它不是"处理不均衡 = 全面变好"那么简单：

| 策略 | 精确率 | 召回率 | F1 | AUC |
|------|--------|--------|-----|-----|
| 原始（baseline） | 0.970 | 0.825 | 0.891 | 0.980 |
| class_weight | 0.438 | 0.915 | 0.592 | 0.980 |
| SMOTE | 0.413 | 0.911 | 0.568 | 0.978 |

![三种不均衡策略的召回率F1和AUC对比](imbalance_strategies.png)

仔细读这组数字，故事比想象的有意思。

先看召回率：baseline 0.825，class_weight 升到 0.915，SMOTE 0.911。处理不均衡确实让模型抓到了更多欺诈——召回率提升了约 9 个百分点，这是我们想要的。

但再看精确率：baseline 高达 0.970，class_weight 却暴跌到 0.438，SMOTE 0.413。代价出现了——为了多抓欺诈，模型变得"草木皆兵"，把大量正常交易也误报成欺诈。精确率 0.438 意味着：它标记为欺诈的交易里，超过一半其实是冤枉的。

于是 F1 反而下降了：baseline 0.891 是三者里最高的，加了平衡策略后 F1 掉到 0.59 左右。

这说明什么？**不均衡处理不是免费的午餐，它是在召回率和精确率之间做交易。** class_weight 和 SMOTE 用大幅牺牲精确率，换来了召回率的小幅提升。这笔交易划不划算，取决于你的业务：如果漏掉一笔欺诈的损失（比如几万块盗刷）远大于误报一笔的成本（比如给用户发条短信确认），那牺牲精确率保召回率是对的；反过来，如果每次误报都要冻结用户的卡、引发大量投诉，那 baseline 的高精确率可能更合适。

值得单独留意的是 AUC：三种策略的 AUC 都稳定在 0.98 左右，几乎不动。这不是巧合——AUC 衡量的是模型给样本排序的能力（把欺诈排在正常前面），而 class_weight 和 SMOTE 主要是在挪动"判定为欺诈"的那条阈值线，没有从根本上改变排序能力。换句话说，这三个模型"分辨欺诈的本事"差不多，区别只在于把阈值卡在哪、因此在精确率和召回率之间落在哪个点上。

这给了一个更深的实战启示：**与其纠结用 class_weight 还是 SMOTE，不如先看 AUC 确认模型的排序能力够不够，再根据业务对精确率/召回率的偏好，去调那条判定阈值。** 阈值是比"用哪种重采样"更直接的旋钮。

![判定阈值左右移动时召回率和精确率此消彼长](recall_precision_tradeoff.png)

这张图把阈值的作用画了出来。模型给每笔交易打一个"欺诈评分"，正常交易（蓝）评分普遍低、欺诈（红）评分普遍高，但两堆有重叠。判定阈值就是那条竖线：往左挪（激进），线右边框进更多红点（召回率升），但也框进更多蓝点（误报增、精确率降）；往右挪（保守）正好相反。class_weight 和 SMOTE 做的事，本质就是帮你把这条线往左推了一把。

## 回到那个 99.8% 的模型

现在再看开头那个准确率 99.83% 的模型，你应该能一眼看穿它了：召回率 0，AUC 0.5（和瞎猜一样），F1 为 0。换上正确的指标，它的废物本质无所遁形。

这一篇的技术组件都是现成的——`DummyClassifier`、`class_weight` 参数、`imblearn` 的 `SMOTE`，调用都是几行。真正要带走的是三件事：

第一，**极端不均衡下，准确率是假指标**，必须换成 recall/precision/F1/AUC。

第二，**处理不均衡是一场交易**，不是免费提升。class_weight 和 SMOTE 通常用精确率换召回率，划不划算由业务决定。

第三，**SMOTE 必须在 Pipeline 里、只作用于训练折**，否则就是数据泄漏。

欺诈检测、疾病筛查、设备故障预警——现实里最有价值的预测问题，往往都是这种"要找的东西极其稀少"的不均衡问题。看懂了准确率的陷阱，你才算真正摸到了这类问题的门。

这组建模三篇到此结束。下一组我们回到评估本身——交叉验证到底该怎么做，调参的性价比怎么算。

---

本文的准确率陷阱复现与三种不均衡策略对比代码，以及对应图表的真实数据，都在配套仓库里：[ml-practice](https://github.com/Joneyao/ml-practice)。

- [Kaggle: Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- [imbalanced-learn: SMOTE 官方文档](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)
- [scikit-learn: Metrics and scoring](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [不均衡数据处理方法综述（机器之心）](https://www.jiqizhixin.com/articles/2017-03-20-8)
- [为什么准确率在不均衡数据上会骗人（知乎）](https://zhuanlan.zhihu.com/p/56662485)
