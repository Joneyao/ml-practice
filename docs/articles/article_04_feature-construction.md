> 作者：轻松学AI
> 日期：2026-10-16
> 风格：通俗科普 + 实操教程
> 状态：待发布

# 机器学习实战 #04：从一列数据里，能榨出几个特征？

泰坦尼克数据里有 263 名乘客没填年龄。

按常规套路，面对缺失的年龄，你只有两个选择：要么用中位数把它填上，要么干脆别用这一列。两条路都有代价——填补是在猜，不用是丢信息。

但有第三条路，而且效果出人意料地好：别管年龄了，从乘客的**姓名**里把称谓抠出来当特征。姓名这一列一个都不缺，`Mr`、`Miss`、`Mrs`、`Master` 这些称谓本身就编码了性别、婚姻状态、甚至年龄段的信息。

跑一遍交叉验证，结果是这样的：用"船舱等级 + 年龄 + 票价 + 性别"（年龄走中位数填补）训练，5 折平均准确率 0.7838；换成"船舱等级 + 票价 + 称谓"，完全不碰那一列缺失的年龄，准确率 0.7876——反而略高。

这就是特征工程最反直觉的地方：**一个永不缺失的文本列，有时候比一个要费劲填补的数值列更有用。** 这一篇就讲，怎么从已有的列里"榨"出新特征。

## 模型只认数字，但数字藏在文本里

入门系列里我们反复说过一句话：模型只看得懂数字。性别是"男/女"，模型看不懂，得先编码成 0/1。

但"把文本转成数字"只是最低要求。特征工程真正的价值，是从一列原始数据里，提炼出比它本身更有区分度的新列。

![从一列姓名榨出称谓性别年龄段等多个特征](squeeze_features.png)

拿姓名举例。`"Allen, Miss. Elisabeth Walton"` 这个字符串，直接喂给模型毫无意义——每个人的姓名都不一样，等于没信息。但姓名里夹着一个称谓 `Miss`，这个称谓是有规律的：`Master` 是对男孩的称呼，`Mrs` 基本是已婚女性，`Miss` 多是未婚女性或女孩。一个称谓同时暗示了性别、大致年龄、社会身份。

配套仓库里用一行正则把它抠出来：

```python
_TITLE_PATTERN = re.compile(r",\s*([^.]+)\.")


def extract_title(name: str) -> str:
    """从 "姓, 称谓. 名" 格式的姓名提取称谓，合并稀有/等价称谓。"""
    match = _TITLE_PATTERN.search(name)
    title = match.group(1).strip() if match else "Unknown"
    if title in _RARE_TITLES:
        return "Rare"
    if title in ("Mlle", "Ms"):
        return "Miss"
    if title == "Mme":
        return "Mrs"
    return title
```

姓名的格式是"姓, 称谓. 名"，逗号和点之间夹着的就是称谓。抠出来之后还做了一步合并：`Dr`、`Rev`、`Col` 这类人数很少的称谓统一归成 `Rare`，否则每个只有一两个样本的称谓反而会干扰模型。`Mlle`（法语小姐）并进 `Miss`，`Mme`（法语夫人）并进 `Mrs`。

同样的思路还能用在船舱号上。`"C85"` 的首字母 `C` 是甲板编号，不同甲板在船上的位置不同，离救生艇的远近也不同。缺失的船舱号就归成 `Unknown`——注意，这里的"缺失"本身也是信息：买得起带编号船舱的乘客，和没有船舱记录的乘客，生还率本来就不一样。

## 怎么判断一个新特征有没有用

构造特征只是第一步。造出来的特征到底有没有区分度，得有个量尺。

一个简单好用的指标：按这个特征分组，看每组的生还率差多少。差得越开，说明这个特征越能把"生还"和"遇难"分开。配套仓库里把它叫 `survival_rate_spread`——组间生还率的最大值减最小值。

```python
def survival_rate_spread(df, col, target_col="survived"):
    """按 col 分组算 target_col 均值，组间最大值减最小值，衡量该特征的区分度。"""
    rates = df.groupby(col)[target_col].mean()
    return float(rates.max() - rates.min())
```

拿泰坦尼克数据跑一遍，把原始列和构造出来的特征放一起比：

![构造特征与原始列的生还率区分度对比](b1_survival_spread.png)

数字很说明问题。原始的船舱等级 `pclass`，三个等级之间生还率差 0.364；原始的性别 `sex`，男女生还率差也就是性别本身的区分度。但从姓名构造的 `title`，5 个称谓之间生还率差到了 0.625；从船舱号构造的 `deck`，9 个甲板之间差 0.732。

区分度最高的是交叉特征。把"船舱等级"和"性别"拼成一个新列——`1_female`、`3_male` 这种——6 个组合之间的生还率差达到 0.819，比任何单一原始列都高。

```python
def make_cross_feature(df, col_a, col_b):
    """两列值用下划线拼接成交叉特征，如 pclass=1 + sex=female -> "1_female"。"""
    return df[col_a].astype(str) + "_" + df[col_b].astype(str)
```

为什么交叉特征这么强？因为"一等舱的女性"和"三等舱的男性"的生还率天差地别，但这个差异拆开看"等级"或"性别"单独一个变量，都表达不充分。交叉之后，模型能直接看到"等级和性别的组合"这个更精细的信号。

说白了，特征工程不是把原始列翻译成数字就完事，是把藏在数据里的规律，用模型能直接用上的形式重新组织一遍。

## 从永不缺失的列里榨特征，连填补都省了

回到开头那个反直觉的结论。

263 个缺失的年龄，是泰坦尼克数据里最棘手的一块。填中位数？但 10 岁小孩和 40 岁大人的生还率完全不同，用一个中位数糊上去，等于把这部分信息抹平。不用年龄？又丢掉了一个明显和生还相关的变量。

称谓给了第三条路。`Master` 几乎都是小男孩，`Miss` 偏年轻——称谓本身就粗粒度地编码了年龄信息，而且它来自姓名，一个都不缺。

配套仓库里做了个对比实验。baseline 特征集是"船舱等级 + 年龄 + 票价 + 性别"，其中年龄用中位数填补；engineered 特征集是"船舱等级 + 票价 + 称谓"，完全不碰年龄列。两套都用逻辑回归做 5 折交叉验证：

```python
def compare_feature_sets(df, target_col="survived"):
    working = df.copy()
    working["title"] = working["name"].apply(extract_title)
    y = working[target_col]

    baseline_cols = ["pclass", "age", "fare"]
    baseline_score = _cv_score(
        working[baseline_cols + ["sex"]], y, baseline_cols, ["sex"],
    )
    engineered_cols = ["pclass", "fare"]
    engineered_score = _cv_score(
        working[engineered_cols + ["title"]], y, engineered_cols, ["title"],
    )
    return pd.DataFrame([
        {"feature_set": "baseline", "cv_score": baseline_score},
        {"feature_set": "engineered", "cv_score": engineered_score},
    ])
```

结果：

| 特征集 | 包含的特征 | 5 折交叉验证平均分 |
|--------|-----------|-------------------|
| baseline | 等级 + 年龄（填补）+ 票价 + 性别 | 0.7838 |
| engineered | 等级 + 票价 + 称谓 | 0.7876 |

![两套特征集的交叉验证分数对比](b1_feature_set_comparison.png)

差距不大，但方向很关键：从姓名构造的称谓，替代了"性别 + 填补过的年龄"这个组合，不仅省掉了填补缺失值的麻烦，效果还略好一点。

这不是说填补缺失值没用——下一篇会专门讲填补策略。这里要说的是一个优先级：**遇到一列棘手的缺失数据时，先别急着想怎么填，先看看有没有别的列能构造出等价甚至更好的特征。** 有时候绕过问题，比硬解问题更聪明。

称谓这个例子还有个额外的好处：它永远不会缺失。姓名是每个乘客都有的，从它派生的特征天然没有缺失值问题。用这种"稳定来源"构造的特征，上线之后也更可靠——你不用担心某天突然来一批没有年龄的新数据。

特征工程没有魔法，`extract_title` 不过是一行正则，`make_cross_feature` 不过是两列字符串拼接。难的不是技术，是那个意识——拿到一列数据，多问一句："从它身上，还能派生出什么模型用得上的信号？"

下一篇我们换个数据集，看看当类别变量不是 3 个而是几千个的时候，编码会出什么乱子。

---

本文的特征构造与效果对比代码，以及生还率区分度图、特征集对比图的真实数据，都在配套仓库里：[ml-practice](https://github.com/Joneyao/ml-practice)。

- [Kaggle: Titanic - Machine Learning from Disaster](https://www.kaggle.com/c/titanic/)
- [Aurélien Géron, Hands-On Machine Learning (3rd ed.), Chapter 2](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/ch02.html)
- [scikit-learn: Encoding categorical features](https://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features)
- [Kaggle Titanic 特征工程实战（知乎专栏）](https://zhuanlan.zhihu.com/p/31743196)
