# Review: Task 4 (A2 — imputation.py + test_imputation.py) — ml-practice batch-1 data-cleaning plan

## Verdict: APPROVED

## 验证范围

对照计划文件 `docs/superpowers/plans/2026-10-16-ml-practice-batch1-data-cleaning.md`
的 "Task 4: A2 — 缺失值填补策略（imputation）" 一节，核对
`data_cleaning/imputation.py`、`data_cleaning/test_imputation.py` 两个产物，
以及 coder 会话记录（`~/.kiro/sessions/80d0747a853dd530/sess_3031ad41-6760-427c-9df0-e8bd0689f5ca/messages.jsonl`，
共 43 行）里实际执行的命令与输出。本次复核**没有重新跑测试**（按任务要求，
仅在有具体疑点时才做单点抽查，此处证据链本身已完整且与计划文本一致）。

## (a) 测试是否真的跑过，输出是否为 "5 passed"

从 coder 会话记录中直接核实到两次真实的 `execute_bash` 调用，构成完整的
TDD 红灯→绿灯证据链：

第一次（实现前，确认失败）：
```
cd .../third-party/ml-practice && python -m pytest data_cleaning/test_imputation.py -v
```
结果：`ModuleNotFoundError: No module named 'data_cleaning.imputation'`——
`collected 0 items / 1 error`，符合写完测试但还没写实现的红灯阶段。

第二次（写完 `imputation.py` 后，确认通过）：
```
data_cleaning/test_imputation.py::test_classify_missingness_detects_mar PASSED [ 20%]
data_cleaning/test_imputation.py::test_impute_mean_fills_all_na PASSED   [ 40%]
data_cleaning/test_imputation.py::test_impute_grouped_median_fills_all_na PASSED [ 60%]
data_cleaning/test_imputation.py::test_impute_knn_fills_all_na PASSED    [ 80%]
data_cleaning/test_imputation.py::test_compare_strategies_returns_all_rows PASSED [100%]
============================== 5 passed in 2.39s ===============================
Exit Code: 0
```
命令、cwd、stdout、exit code 均可见，红灯→绿灯顺序自洽，不存在"直接贴
结果、未见命令"的可疑模式。coder 最终 `send_message` 里引用的 "5 passed
in 2.39s" 与会话记录第 29 行的真实工具调用结果逐字一致。**(a) 通过。**

## (b) 是否存在占位/TODO 代码

对 `imputation.py`、`test_imputation.py` 做了
`TODO|FIXME|XXX|NotImplementedError|placeholder` 关键词搜索（含 `pass  #`
模式），零匹配（grep exit code 1）。六个函数（`classify_missingness`、
`impute_mean`、`impute_grouped_median`、`impute_knn`、`_cv_score`、
`compare_strategies`）均为完整实现，无空函数体。**(b) 通过。**

## (c) 函数签名是否与计划精确一致

用 `ast.parse` 核实了 `imputation.py` 的函数签名：

| 函数 | 实际签名 | 计划要求 | 一致 |
|------|----------|----------|------|
| `classify_missingness` | `(df, col, group_col, threshold=0.1) -> str` | `(df, col, group_col)`，默认判定逻辑同计划文本 | ✓ |
| `impute_mean` | `(df, col) -> pd.Series` | 同 | ✓ |
| `impute_grouped_median` | `(df, col, group_col) -> pd.Series` | 同 | ✓ |
| `impute_knn` | `(df, cols, n_neighbors=5) -> pd.DataFrame` | `(df, cols)`，用 `sklearn.impute.KNNImputer` | ✓ |
| `compare_strategies` | `(df, target_col, age_col='age', group_col='pclass') -> pd.DataFrame` | 完全一致 | ✓ |

`compare_strategies` 返回的 DataFrame 含 `strategy`/`n_samples`/`cv_score`
三列，四个策略（`dropna`/`mean`/`grouped_median`/`knn`）与计划及测试断言
（`test_compare_strategies_returns_all_rows`）完全匹配。这些函数会被
Task 6 `export_charts.py` 的 `export_a2_imputation_comparison()` 直接
调用 `compare_strategies(df, target_col="survived")`，签名兼容，不会
阻塞后续任务。**(c) 通过，无阻塞风险。**

## (d) 是否为原创实现，非抄录

三点支撑原创性判断：
1. TDD 轨迹本身是证据——先写测试、跑红灯确认 `ModuleNotFoundError`，
   再写实现、跑绿灯，这种顺序和抄录成品代码的工作模式不同。
2. `classify_missingness` 的 MAR/MCAR 简化判定（按分组缺失率差异和
   `threshold=0.1` 比较）是计划文本里明确给出的教学简化算法，不是
   某个标准库或教程的现成函数，和 `test_imputation.py` 里构造的
   MAR 测试数据（三等舱缺失率更高）严格对应。
3. coder 最终消息明确声明"from scratch per plan, no web copy-paste"，
   且会话里没有任何 `fetch_url`/网络搜索类工具调用痕迹（仅
   `grep_search`、`execute_bash`、`read_file`、`fs_write`），没有从外部
   获取代码的路径。**(d) 通过。**

## (e) Commit 是否存在

```
b16e356 (HEAD -> master, origin/master, origin/HEAD) feat(ml-practice): A2 缺失值填补策略对比
9ed8eb2 feat(ml-practice): A1 数据探查与假高分复现
ee79eba feat(ml-practice): common 绘图字体与数据集加载工具
98a3896 feat(ml-practice): 初始化 ml-practice 仓库骨架
a2d985d docs: 机器学习实战第一批（数据清洗）实现计划
8e87901 docs: 机器学习实战系列设计文档
d061b1f feat: add per-section charts to MJPEG/H.264 #09-17 and publish to WeChat draft
```
`git log --oneline -7` 确认该 commit 存在且为当前 HEAD，`git show --stat
b16e356` 确认精确包含且仅包含两个文件
（`imputation.py` 72 行、`test_imputation.py` 52 行，共 124 行新增），
无夹带其他改动。Commit message 与计划 Step 5 要求的文本
`feat(ml-practice): A2 缺失值填补策略对比` 完全一致。**(e) 通过。**

## 额外观察（非阻塞）

- commit 日志显示仓库配置了 post-commit hook 自动推送到
  `origin/master`（"🔄 Auto-pushing to origin/master..."），coder 在
  最终消息中如实说明了这一点，没有自行额外执行 `git push`，符合任务
  指示"No push"的边界。
- `test_imputation.py` 的 fixture 用固定随机种子 `np.random.default_rng(0)`
  构造 MAR 场景（三等舱缺失概率 0.5 vs 其他舱位 0.05），和
  `classify_missingness` 的 `threshold=0.1` 默认值配合，测试具有确定性，
  不依赖浮动的随机结果。

## 结论

五项核实全部通过：真实测试证据链完整（红灯 `ModuleNotFoundError` →
绿灯 `5 passed in 2.39s`，命令/输出均可追溯）、无占位代码、五个关键函数
签名精确匹配计划（供 Task 6 复用不受阻）、代码为原创实现（TDD 顺序 +
会话中无外部代码获取路径 + 明确声明）、commit 真实存在且范围精确（仅
2 文件 124 行）。没有阻塞项。
