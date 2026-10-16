# Review: Task 6 — export_charts.py（A1-A3 真实数据配图导出脚本）

## 范围

对照 `docs/superpowers/plans/2026-10-16-ml-practice-batch1-data-cleaning.md`
的 Task 6（export_charts）定义，核对实现文件
`third-party/ml-practice/data_cleaning/export_charts.py`，commit
`e14afd1 feat(ml-practice): A1-A3 真实数据配图导出脚本`。

## (a) 脚本确实被执行，且无字体缺字警告；真实 PNG+CSV 产物存在

从该任务的 coder 子会话（`sess_359db560-73ef-4595-bef9-5af2a9b3f991`
messages.jsonl）回放工具调用记录，确认：

- `python -m data_cleaning.export_charts 2>&1 | grep -i "glyph\|missing" || echo "NO FONT WARNINGS"`
  第一次运行输出里带文件名（`missing_heatmap.png`/`missing_by_group.png`
  字面含 "missing"），导致该 grep 检查产生假阳性噪音；coder 发现后修改
  `main()` 不再回显文件名，只打印一行汇总（"导出完成：共 N 张图表，输出
  目录 ..."），重新运行后 grep 结果为干净的 `NO FONT WARNINGS`。
- `ls -la .../charts_out/` 的记录输出（工具结果 idx 53）与当前磁盘状态
  完全一致：

  ```
  imputation_comparison.csv   149 B
  imputation_comparison.png 36609 B
  missing_by_group.png      31010 B
  missing_heatmap.png       39231 B
  naive_vs_honest.png       52821 B
  outlier_scatter.png      110605 B
  strategy_comparison.csv    175 B
  strategy_comparison.png   28409 B
  ```

  本次复核直接重新 `ls -la` 了磁盘上的 `charts_out/` 目录，文件名、大小
  与会话记录完全吻合，文件 mtime（2026-10-16 23:21:22~23:21:25）落在
  commit 时间（23:21:58）之前几十秒内，时序自洽。
- 两张图（`missing_heatmap.png`、`outlier_scatter.png`）在会话中被
  `read_file` 实际读取查看（图像 base64 返回，Size 字段与磁盘文件大小一致
  36609/110605 字节等），证明不是摆放的空文件或占位图。

结论：脚本被真实执行过，字体检查通过，产物真实存在。**PASS**

## (b) 所有图表使用真实模块输出，无硬编码数字

逐个检查 6 个 `export_xxx()` 函数：

- `export_a1_missing_heatmap`：直接对 `load_titanic()` 的 `df.isna()` 结果
  画热力图，无硬编码。
- `export_a1_naive_vs_honest`：标题里的 `naive_score` 来自
  `explore.naive_cv_score(df, target_col="survived")` 的真实返回值；三条
  柱子（全量/非缺失/缺失子集）的存活率都是对 `df["survived"]` 做
  `.mean()` 现场计算，无硬编码。
- `export_a2_imputation_comparison` / `export_a2_missing_by_group`：
  分别调用 `imputation.compare_strategies()` 和
  `imputation.classify_missingness()`，绘图数据和标题里的 MAR/MCAR 标签
  都来自函数返回值。
- `export_a3_outlier_scatter`：散点图数据和标题里的异常值计数
  `int(mask.sum())` 来自 `outliers.detect_iqr()` 的真实掩码。
- `export_a3_strategy_comparison`：调用 `outliers.compare_outlier_strategies()`，
  柱状图数据直接取自返回的 DataFrame。

CSV 产物内容（`imputation_comparison.csv`、`strategy_comparison.csv`）
与会话记录中 `cat` 命令的输出完全一致（dropna 714 样本 / mean、
grouped_median、knn 均为 891 样本；keep/drop/winsorize 的 R² 分别为
0.502/0.485/0.509），与脚本计算逻辑吻合，非拍脑袋数字。

脚本中仅有的常量是绘图样式参数（`figsize`、颜色 hex 码、`dpi`），不是数据
本身。**PASS**

## (c) 全部走 `setup_cjk_font()` + `save_fig`；图表集合覆盖 A1/A2/A3 映射

`main()` 中只调用一次 `setup_cjk_font()`（符合全局设置一次的惯例），
6 个 `export_xxx()` 函数无一例外都以 `return save_fig(fig, OUT, "xxx.png")`
结尾，没有任何函数绕过 `save_fig` 自行 `fig.savefig(...)`。

对照 Task 6 文档里"图与文章的映射"要求的 6 张图：

| 文档要求 | 脚本函数 | 产物 |
|---|---|---|
| A1 missing_heatmap.png | `export_a1_missing_heatmap` | ✓ |
| A1 naive_vs_honest.png | `export_a1_naive_vs_honest` | ✓ |
| A2 imputation_comparison.png | `export_a2_imputation_comparison` | ✓ |
| A2 missing_by_group.png | `export_a2_missing_by_group` | ✓ |
| A3 outlier_scatter.png | `export_a3_outlier_scatter` | ✓ |
| A3 strategy_comparison.png | `export_a3_strategy_comparison` | ✓ |

六张图全部覆盖，无遗漏。**PASS**

## (d) 无 TODO/占位

对 `export_charts.py` 做 `TODO|FIXME|placeholder|占位` 全文检索，零命中。
每个函数都是完整实现，没有 `pass`/`...` 占位函数体。**PASS**

## (e) 调用 explore/imputation/outliers 的签名与 Task 3-5 完全一致

逐一核对实际模块签名（已读取 `explore.py`/`imputation.py`/`outliers.py`
源码）：

- `explore.naive_cv_score(df: pd.DataFrame, target_col: str, seed: int = 0)`
  ← 调用 `explore.naive_cv_score(df, target_col="survived")`，匹配。
- `imputation.compare_strategies(df, target_col, age_col="age", group_col="pclass")`
  ← 调用 `imputation.compare_strategies(df, target_col="survived")`，匹配
  （用默认 age_col/group_col）。
- `imputation.classify_missingness(df, col, group_col, threshold=0.1)`
  ← 调用 `imputation.classify_missingness(df, col="age", group_col="pclass")`，匹配。
- `outliers.detect_iqr(series, k=1.5)`
  ← 调用 `outliers.detect_iqr(df["GrLivArea"])`，匹配。
- `outliers.compare_outlier_strategies(df, col, target_col)`
  ← 调用 `outliers.compare_outlier_strategies(df, col="GrLivArea", target_col="SalePrice")`，
  匹配。

全部签名一致，无参数名或参数顺序错位。**PASS**

## (f) 原创代码，非抄录

`export_charts.py` 的六个导出函数都是围绕 Task 3-5 模块接口新写的组合逻辑
（取数据 → 调函数 → 画图 → 存盘），不是从 `ml-from-scratch` 或任何外部
教程直接搬运的代码块。唯一与 `third-party/ml-from-scratch/common/plotting.py`
完全一致的部分是 `common/plotting.py` 本身（`setup_cjk_font`/`save_fig`），
但这是 Task 2 就已确立、且计划文档明确要求"照搬 ml-from-scratch 的实现"
的公共基础设施，不属于 Task 6 的交付范围，Task 6 本身没有抄录行为。
**PASS**

## (g) commit 存在

```
e14afd1 (HEAD -> master, origin/master, origin/HEAD) feat(ml-practice): A1-A3 真实数据配图导出脚本
1d52867 feat(ml-practice): A3 异常值检测与处理策略对比
b16e356 feat(ml-practice): A2 缺失值填补策略对比
9ed8eb2 feat(ml-practice): A1 数据探查与假高分复现
ee79eba feat(ml-practice): common 绘图字体与数据集加载工具
98a3896 feat(ml-practice): 初始化 ml-practice 仓库骨架
a2d985d docs: 机器学习实战第一批（数据清洗）实现计划
8e87901 docs: 机器学习实战系列设计文档
d061b1f feat: add per-section charts to MJPEG/H.264 #09-17 and publish to WeChat draft
```

commit message 与计划文档要求的文案完全一致，diff 只新增
`export_charts.py` 一个文件（126 行）。**PASS**

## 附带观察（非阻断项）

coder 在完成报告中提到："仓库有 post-commit hook 在提交后自动推送到
origin/master，尽管 step instruction 说 'No push'"。这是本次 Task 6
step 指令与仓库级 `git-push` 规则（commit 后必须 push）之间的冲突，
coder 已如实披露且未隐瞒，推送内容也就是已验证通过的同一个 commit，
不影响本任务的正确性判定，仅作信息记录。

## 结论

(a)-(g) 全部核实通过，没有发现伪造执行证据、硬编码数据、签名不匹配、
占位代码或抄录行为。

**Verdict: APPROVED**
