# Task 7 审查报告（docs + GitHub 仓库 + submodule + push）

## (a) 全量回归 `python -m pytest -v` 是否真的跑过，输出是否为 "14 passed"

coder 会话记录（`messages.jsonl` line 35-36）显示执行了
`cd third-party/ml-practice && python -m pytest -v 2>&1`，真实工具调用结果：

```
collected 14 items
common/test_common.py::test_setup_font_returns_name PASSED   [  7%]
common/test_common.py::test_load_titanic_from_cache PASSED   [ 14%]
common/test_common.py::test_load_house_prices_from_cache PASSED [ 21%]
data_cleaning/test_explore.py::test_missing_summary_counts_nulls PASSED [ 28%]
data_cleaning/test_explore.py::test_dtype_summary_has_all_columns PASSED [ 35%]
data_cleaning/test_explore.py::test_naive_cv_score_returns_float_in_range PASSED [ 42%]
data_cleaning/test_imputation.py::test_classify_missingness_detects_mar PASSED [ 50%]
data_cleaning/test_imputation.py::test_impute_mean_fills_all_na PASSED [ 57%]
data_cleaning/test_imputation.py::test_impute_grouped_median_fills_all_na PASSED [ 64%]
data_cleaning/test_imputation.py::test_impute_knn_fills_all_na PASSED [ 71%]
data_cleaning/test_imputation.py::test_compare_strategies_returns_all_rows PASSED [ 78%]
data_cleaning/test_outliers.py::test_detect_iqr_flags_large_areas PASSED [ 85%]
... 14 passed in 2.77s ...
```

coder 紧接着自评 "14 passed, matches expectation"，随后 `send_message` 里
引用 "14 passed in 2.77s"，与真实工具结果一致（不是凭空声称）。
按模块拆分：common 3 + A1(explore) 3 + A2(imputation) 5 + A3(outliers) 3 = 14，
与此前 Task 2-5 各自独立审查认定的测试数一致。**(a) 通过。**

## (b) `docs/references.md` 三块内容 + "只借鉴思路，重新实现" 原则声明；
`docs/articles/README.md` 14 篇表格，A1-A3 标"代码就绪，待写文章"，无实际文章

实读 `third-party/ml-practice/docs/references.md`：

1. **经典书籍/课程锚点**：Hands-On ML（Géron，第 2 章 End-to-End ML Project）、
   Approaching (Almost) Any ML Problem（含 GitHub 链接）、吴恩达 CS229/ML
   Specialization——三项齐全。
2. **官方文档清单**：scikit-learn Pipelines 文档、`sklearn.impute` 文档、
   pandas 缺失值处理文档，并明确写出"只借鉴思路和接口设计，代码自己从
   原理重新实现，不摘抄文档或任何教程中的示例代码"，与 `ml-from-scratch`
   的措辞风格和原则一致。
3. **数据集出处**：Titanic（Kaggle 链接 + seaborn 获取方式）、House Prices
   (Ames，OpenML data_id=42165) 均列出来源和获取方式。

三块内容齐全，且开头总述句已声明"只借鉴思路...不摘抄任何教程或开源项目
源码"的核心原则。

实读 `third-party/ml-practice/docs/articles/README.md`：14 篇表格完整，
A1/A2/A3 三行状态均为"代码就绪，待写文章"，B1-F2 共 11 行状态均为
"规划中"。全文检索 `*.py`/`*.md` 的 TODO/FIXME/XXX/placeholder，仅命中
该 README 表格里的"（待建）"字样（描述未来模块路径，非代码占位，是
对真实未完成状态的准确标注）和既有的 task1-6 审查文档里的"零命中"说明，
没有"占位性"文章正文。`docs/articles/` 目录下只有这一个 README 文件，
没有任何 `article*.md`。主仓库 `articles/easy_ai/` 下最新日期仍是
2026-09-26（上一个系列），没有机器学习实战系列的文章文件。**(b) 通过。**

## (c) 无 TODO/占位

对 `third-party/ml-practice` 全库做 `TODO|FIXME|XXX|placeholder|待建` 检索，
命中的全部是：
- 既有 task1-6 审查文档里"零命中"的说明文字本身；
- `docs/articles/README.md` 表格中 11 行"规划中"模块路径用 `（待建）`
  准确标注"尚未创建的目录"，这是文档对未来工作的如实描述，不是代码里的
  占位逻辑或半成品实现。

未发现任何 `.py` 源码文件中的 TODO/FIXME/NotImplementedError 占位。**(c) 通过。**

## (d) 四个 commit 存在，消息逐字匹配

`git -C WeChatAutoPress log --oneline -12`：

```
a883c7c (HEAD -> master, origin/master, origin/HEAD) chore: 将 ml-practice 登记为 submodule
58d2fcb docs(ml-practice): 参考资料清单与文章同步目录占位
e14afd1 feat(ml-practice): A1-A3 真实数据配图导出脚本
1d52867 feat(ml-practice): A3 异常值检测与处理策略对比
b16e356 feat(ml-practice): A2 缺失值填补策略对比
9ed8eb2 feat(ml-practice): A1 数据探查与假高分复现
ee79eba feat(ml-practice): common 绘图字体与数据集加载工具
98a3896 feat(ml-practice): 初始化 ml-practice 仓库骨架
a2d985d docs: 机器学习实战第一批（数据清洗）实现计划
8e87901 docs: 机器学习实战系列设计文档
```

Task 7 相关的三个主仓库 commit 消息与计划逐字一致：
`docs(ml-practice): 参考资料清单与文章同步目录占位`（docs commit）、
`chore: 将 ml-practice 登记为 submodule`（submodule commit）。
第四个要求的 commit 是 `ml-practice` 子仓库自身的 init commit：
`git -C third-party/ml-practice log --oneline` 显示
`cfde447 (HEAD -> main, origin/main) init: ml-practice 第一批数据清洗模块`，
消息同样逐字匹配计划 Step 5 的 `git commit -m "init: ml-practice 第一批
数据清洗模块"`。四个 commit（docs、ml-practice init、submodule chore，
外加之前 Task 1-6 的 6 个 feat commit）均存在且消息准确。**(d) 通过。**

## (e) submodule 是否真正注册（非裸目录）

- `.gitmodules` 含：
  ```
  [submodule "third-party/ml-practice"]
      path = third-party/ml-practice
      url = git@github.com:Joneyao/ml-practice.git
  ```
- `git -C WeChatAutoPress submodule status` 输出中包含：
  ```
  cfde44730a04cf2bbee475e984e2b61ea35d5273 third-party/ml-practice (heads/main)
  ```
  这是标准的 "已注册 submodule" 格式（commit hash + 路径 + 分支名），
  不是普通嵌套目录的空白/`-` 前缀格式。
- coder 的真实命令序列（会话记录 line 46-61）显示先 `git rm -r --cached
  third-party/ml-practice` 清除普通文件追踪，再 `git submodule add
  git@github.com:Joneyao/ml-practice.git third-party/ml-practice`，
  最终 commit `a883c7c` 的 diff 摘要显示 `create mode 160000
  third-party/ml-practice`（gitlink 模式 160000，是 submodule 的标志
  性文件模式）并同时 `delete mode 100644` 掉之前作为普通文件追踪的
  所有 ml-practice 源码文件——这是从"普通嵌套目录"正确迁移为
  "submodule 指针"的实际 diff 证据，不是声称。**(e) 通过。**

## (f) GitHub 仓库 `Joneyao/ml-practice` 创建 + 主仓库 push 成功

会话记录真实工具结果：
- `gh repo create Joneyao/ml-practice --public ...` 返回
  `✓ Created repository Joneyao/ml-practice on github.com`
  `https://github.com/Joneyao/ml-practice`（line 47，真实 gh CLI 输出，
  非声称）。
- ml-practice 子仓库 `git push -u origin main` 有真实的 "Counting
  objects..." 推送过程输出（line 54），`git -C third-party/ml-practice
  remote -v` 确认 `origin git@github.com:Joneyao/ml-practice.git`，
  且 `git -C third-party/ml-practice log origin/main..main` 空输出，
  证明本地 `main` 与远程 `origin/main` 完全同步，没有未推送的提交。
- 主仓库侧：会话记录 line 61 显示 submodule commit 触发了仓库自带的
  "自动 push 钩子"（`🔄 Auto-pushing to origin/master...`），line 65
  的显式 `git push` 返回 `Everything up-to-date`，且同一命令输出的
  `local vs origin HEAD` 两行哈希完全相同（`a883c7c3c3987...`）。
  现场复核 `git -C WeChatAutoPress log origin/master..master` 为空，
  `git status --short` 无未提交变更，与会话记录结论一致。**(f) 通过。**

## (g) SCOPE：确认未写 WeChat 文章，未启动 batch 2-5

- `third-party/ml-practice/docs/articles/` 目录下只有一个 `README.md`
  占位说明文件，没有任何 `article*.md` 正文。
- `third-party/ml-practice` 顶层目录仅有 `common/`、`data_cleaning/`、
  `docs/`、`data/`（下载缓存目录，被 `.gitignore` 排除）——没有计划里
  属于后续批次的 `feature_engineering/`、`modeling/`、
  `tuning_evaluation/`、`deployment/`、`deep_learning/` 任何目录。
- 主仓库 `articles/easy_ai/` 下最新文章日期仍是 2026-09-26（上一个
  "从零开始学机器学习"系列第 18/19 篇），没有任何《机器学习实战》
  系列的文章文件，也没有新建 2026-10 的 easy_ai 文章目录。
- 本次改动范围严格限定在 Task 7 要求的两个文档文件
  （`docs/references.md`、`docs/articles/README.md`）+ GitHub 仓库创建
  + submodule 注册 + 两次 push，没有越界触碰 batch 2-5 的代码或文章。
  **(g) 通过。**

## 综合结论

七项核实全部通过：真实测试证据链完整（命令+真实输出一致，14 passed）、
两份文档内容齐全且声明了"只借鉴思路"原则、文章目录仅占位无正文、
无代码占位、四个 commit 消息逐字匹配、submodule 以 gitlink 模式正确
注册（非裸目录）、GitHub 仓库真实创建且两级仓库（ml-practice 子仓库 +
WeChatAutoPress 主仓库）均已推送到位、且严格限定在 batch 1 范围内未
越界写文章或启动后续批次。批次 1（Task 1-7）全部完成，无阻塞性问题。

**Verdict: APPROVED**
