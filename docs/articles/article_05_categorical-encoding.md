> 作者：轻松学AI
> 日期：2026-10-16
> 风格：通俗科普 + 实操教程
> 状态：待发布

# 机器学习实战 #05：类别变量编码，One-Hot 之外的世界

一个电信客户流失数据集，7043 行。里面有 16 个描述客户服务的类别列——用不用光纤、开不开在线备份、签的是月付还是年付。

把这 16 列拼成一个"客户画像"组合列，然后 One-Hot 编码，会发生什么？

维度从个位数炸到 **4423**。而这 4423 个画像里，有 **3555 个在整个数据集里只出现过一次**，占比 80.3%。换句话说，你造出了 4423 个几乎全是 0 的新列，其中八成列只对应一个客户。

这就是类别编码的核心矛盾：**编码方式本身不难，难的是判断什么时候哪种编码会把你带进坑里。** 这一篇讲四种常见编码，以及高基数类别那个最容易踩的雷。

## 四种编码，先认清它们各自在做什么

类别变量编码，本质是把"男/女""月付/年付"这种文本，转成模型能算的数字。但转法不止一种，每种隐含的假设不同。

![同一列合同类型的三种编码方式对比](encoding_ways.png)

**One-Hot 编码**：每个类别变成一个独立的 0/1 列。三个合同类型就变三列，是哪个类型哪列就是 1。好处是不引入任何虚假的大小关系，坏处是类别一多，列数就爆炸。

```python
def encode_onehot(df, col):
    """One-Hot 编码单列，返回每个类别一列的 0/1 DataFrame。"""
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoded = encoder.fit_transform(df[[col]])
    columns = encoder.get_feature_names_out([col])
    return pd.DataFrame(encoded, columns=columns, index=df.index)
```

**Label 编码**：每个类别映射成一个整数，按字母序排。`Bank transfer=0, Credit card=1, Electronic check=2, Mailed check=3`。问题来了——这个顺序纯粹是字母序，没有任何业务含义，但模型会把它当成"2 比 1 大"。对线性模型是灾难，对树模型倒无所谓（树只关心分裂点，不关心数值大小的含义）。

**Ordinal 编码**：和 Label 像，但顺序是你按业务含义指定的。合同类型 `Month-to-month < One year < Two year` 是真实有序的，映射成 0/1/2 就合理。

```python
def encode_ordinal(df, col, order):
    """Ordinal 编码：按业务定义的顺序映射到 0..n-1。"""
    mapping = {category: rank for rank, category in enumerate(order)}
    return df[col].map(mapping).rename(f"{col}_ordinal")
```

**Target 编码**：每个类别映射成"这个类别下目标变量的平均值"。比如"月付客户的平均流失率"。信息量最大，但也最危险——它直接用到了标签，一不小心就会数据泄漏。下一篇会专门讲它怎么把人坑惨。

说白了，Label 和 Ordinal 长得一样，区别只在顺序有没有业务含义；One-Hot 最安全但最占地方；Target 最聪明但最容易出事。

## 低基数时，选哪种编码几乎不影响结果

先说一个让很多人意外的事实：当类别少、而且有限时，纠结选哪种编码，基本是浪费时间。

拿电信数据里的合同类型 `Contract` 举例，它只有 3 个值：月付、一年、两年。分别用 One-Hot、业务顺序的 Ordinal、安全使用的 Target 编码，各自训练逻辑回归做交叉验证，三者的准确率分别是 0.7876、0.7874、0.7873——差异完全在噪声范围内。

为什么？因为 3 个类别，不管怎么编码，信息量就那么多。One-Hot 用 3 列表达，Ordinal 用 1 列表达，但它们承载的是同一件事：这个客户签的哪种合同。编码方式只是换个容器装同样的信息。

所以别在低基数类别的编码选择上反复纠结。真正会出问题的，是另一种情况——类别特别多的时候。

## 高基数类别：维度爆炸的真实现场

把那 16 个服务类别列拼成一个组合列 `customer_profile`，每个客户的"完整画像"就是一个独一无二的字符串。跑一下基数报告：

```python
def cardinality_report(df, col):
    """报告某列的基数、One-Hot 维度、只出现一次的类别占比。"""
    n_unique = df[col].nunique()
    n_rows = len(df)
    value_counts = df[col].value_counts()
    singleton_ratio = float((value_counts == 1).sum() / n_unique) if n_unique > 0 else 0.0
    return {
        "column": col, "n_rows": n_rows, "n_unique": n_unique,
        "onehot_dims": n_unique,
        "avg_rows_per_category": n_rows / n_unique if n_unique > 0 else 0.0,
        "singleton_ratio": singleton_ratio,
    }
```

对比低基数列和这个组合列，维度差距一目了然：

![低基数列与高基数组合列的One-Hot维度对比](b2_encoding_dimension.png)

`Contract` 是 3 维，`PaymentMethod` 是 4 维，而 `customer_profile` 是 **4423 维**。注意这张图用的是对数坐标，否则前两根柱子在 4423 面前根本看不见。

更要命的是这 4423 个类别的分布：

![高基数组合列中只出现一次的类别占比](b2_singleton_ratio.png)

4423 个画像里，80.3% 只出现过一次。这意味着如果你 One-Hot 编码这个组合列，造出来的 4423 列中，绝大多数列只有一个客户是 1、其余 7042 个客户全是 0。

这种特征对模型几乎是毒药。模型会"记住"每个只出现一次的画像对应的那个客户的标签——这不是学习规律，是背答案。在训练集上表现好得异常，一到新数据就全线崩溃。这正是高基数类别最典型的过拟合陷阱。

遇到高基数类别，出路通常有三条：要么像前面说的用 Target 编码（但要小心泄漏），要么把稀有类别合并成一个"其他"类，要么干脆别拼组合列、保留原始的低基数列分别编码。具体选哪条，取决于这些类别对预测到底有没有用——如果 80% 都只出现一次，它们大概率没有可泛化的信息。

## Label 编码的隐藏陷阱：虚假的大小关系

最后单独说一下 Label 编码，因为它是新手最容易误用的一个。

`PaymentMethod` 有四种：银行转账、信用卡、电子支票、邮寄支票。用 `LabelEncoder` 一编码，它们按字母序变成 0、1、2、3。

```python
def encode_label(df, col):
    """Label 编码：类别 -> 任意整数（按字母序），不隐含真实大小关系。
    只适合树模型（分裂点不依赖数值大小含义），不建议喂给线性模型。
    """
    encoder = LabelEncoder()
    return pd.Series(encoder.fit_transform(df[col]), index=df.index, name=f"{col}_label")
```

问题在于：银行转账是 0、邮寄支票是 3，但这四种支付方式之间**根本没有大小顺序**。字母序纯属巧合。可一旦喂给线性模型，模型就会认为"邮寄支票(3) 是银行转账(0) 的三倍"——这个关系是编码器凭空造出来的，数据里压根不存在。

这就是为什么 Label 编码的 docstring 里写明了"只适合树模型"。树模型做分裂时只问"这个值大于还是小于某个阈值能不能把样本分开"，不在乎数值本身代表什么量级，所以虚假顺序不影响它。但线性模型会把数值直接乘进权重，虚假的大小关系会实打实地污染结果。

记住一个简单规则：**类别有真实顺序（如合同时长、学历、满意度等级），用 Ordinal 手动指定顺序；类别无顺序，线性模型用 One-Hot，树模型可以用 Label 图省事。** 不要无脑 `LabelEncoder` 一把梭。

这一篇的技术都不复杂，`OneHotEncoder`、`LabelEncoder` 都是 sklearn 里的标准组件，几行就能调用。难的是判断力：低基数时随便选，高基数时警惕维度爆炸和过拟合，Label 编码只喂给树模型。

下一篇会把这些编码器放进一个真正的工作流里，顺便复现一个让模型"离线 92 分、上线 76 分"的经典事故——数据泄漏。

---

本文的四种编码实现与基数报告代码，以及维度对比图、稀有类别占比图的真实数据，都在配套仓库里：[ml-practice](https://github.com/Joneyao/ml-practice)。

- [Kaggle: Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- [scikit-learn: Encoding categorical features](https://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features)
- [Abhishek Thakur, Approaching (Almost) Any Machine Learning Problem](https://github.com/abhishekkrthakur/approachingalmost)
- [类别特征编码方法详解（掘金）](https://juejin.cn/post/7051539531508613128)
