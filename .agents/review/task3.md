# Review: Task 3 (A1 — 数据探查 explore.py + test_explore.py) — ml-practice batch-1 data-cleaning plan

## Verdict: APPROVED

## 验证范围

对照计划文件 `docs/superpowers/plans/2026-10-16-ml-practice-batch1-data-cleaning.md`
的 "Task 3: A1 — 数据探查（explore）" 一节，核对 `data_cleaning/explore.py`、
`data_cleaning/test_explore.py` 两个产物，以及 coder 会话记录
（`sess_690fd96e-78f0-4a74-941c-b74cf549f0b1/messages.jsonl`，共 118 行）里
实际执行的命令与输出。本次复核未重新跑测试（按任务要求，仅在有具体疑点
时才做单点抽查，此处证据链本身已完整且一致）。

## (a) 测试是否真的跑过，输出是否为 "3 passed"

从 coder 会话记录里直接核实到完整的 TDD 执行轨迹，比预期更丰富——不是
一次性"写完就通过"，而是经历了红灯→环境排障→首次绿灯失败→修正→真正
绿灯的完整过程：

1. 第一次运行（`cd .../ml-practice && python -m pytest data_cleaning/test_explore.py -v`）：
   `collected 0 items / 1 error`（explore 模块还不存在，红灯符合预期）。
2. 写完 `explore.py` 后第二次运行：仍然报错——这里 coder 遇到了环境问题
   （执行 pytest 用的 venv 缺 scikit-learn/seaborn），花了十几轮命令
   （`which python`、`pip show`、`uv pip install scikit-learn`、排查
   `VIRTUAL_ENV`、确认最终用 `/data/work/.../agentflow/.venv/bin/python`
   这个解释器）才打通依赖环境。这段排障过程本身就是该命令被真实执行的
   强证据——如果是编造的测试输出，不会有这么多条真实的环境报错和修复记录。
3. 依赖打通后第三次运行：2 passed，1 failed——
   `test_naive_cv_score_returns_float_in_range` 报
   `ValueError: n_splits=5 cannot be greater than the number of members in
   each class`。这是计划里给出的示范代码在小样本 fixture（8 行、dropna
   后剩 5 行，且目标列最少类别只有 2 个样本）下的真实缺陷，与计划文本
   相符（`cross_val_score(clf, X, y, cv=5)` 固定用 5 折）。
4. coder 据此修正实现，把固定 `cv=5` 改为
   `n_splits = min(5, y.value_counts().min())` 配合 `StratifiedKFold`，
   第四次运行拿到真实输出：
   ```
   data_cleaning/test_explore.py::test_missing_summary_counts_nulls PASSED [ 33%]
   data_cleaning/test_explore.py::test_dtype_summary_has_all_columns PASSED [ 66%]
   data_cleaning/test_explore.py::test_naive_cv_score_returns_float_in_range PASSED [100%]
   ======================== 3 passed in 1.31s =========================
   Exit Code: 0
   ```

证据链完整：命令、cwd、stdout、exit code 均可见，红灯→中间失败→定位
根因→修正→绿灯的顺序和内容完全自洽，不存在"直接贴结果、未见命令"的
可疑模式。coder 完成信息里给出的 pytest 输出与会话记录里第 91 行的
真实工具调用结果逐字一致。**(a) 通过。**

## (b) 是否存在占位/TODO 代码

对 `explore.py`、`test_explore.py` 做了
`TODO|FIXME|XXX|NotImplementedError|placeholder` 关键词搜索，无匹配
（grep 返回 exit code 1，即零匹配）。三个函数（`missing_summary`、
`dtype_summary`、`naive_cv_score`）均为完整实现，没有 `pass` 或空函数体。
**(b) 通过。**

## (c) 函数签名是否与计划精确一致

用 `ast.parse` 核实了 `explore.py` 的函数签名：

- `missing_summary(df)`：单参数 `df`，无默认值。✅ 与计划
  `missing_summary(df) -> pd.DataFrame` 一致，返回 DataFrame 含
  `column` / `missing_count` / `missing_rate` 三列，经测试
  `test_missing_summary_counts_nulls` 验证字段名和数值正确
  （age 列 missing_count=3，missing_rate=3/8）。
- `dtype_summary(df)`：单参数 `df`，无默认值。✅ 与计划
  `dtype_summary(df) -> pd.DataFrame` 一致，返回 `column` / `dtype` /
  `n_unique` 三列，经测试验证覆盖全部输入列。
- `naive_cv_score(df, target_col, seed=0)`：三参数，`seed` 默认值 0。✅
  与计划 `naive_cv_score(df, target_col) -> float` 一致（`seed` 是计划
  文本里也提到的可选参数，默认值匹配），返回类型为 `float`，经测试验证
  落在 `[0.0, 1.0]` 区间。

这三个签名是计划里明确标注"供 A2 和 export_charts 复用"的接口，核实结果
精确匹配，没有参数名、顺序或默认值的偏差。**(c) 通过，非阻塞。**

## (d) 代码是否原创，而非直接抄录

- `missing_summary` / `dtype_summary` 是标准的 pandas 聚合写法
  （`isna().sum()`、`dtype`/`nunique()` 逐列收集成 DataFrame），没有
  特有的变量命名、注释风格或出处字符串指向某个具体教程/开源项目。
- `naive_cv_score` 的最终实现（用 `StratifiedKFold` + 动态计算
  `n_splits = min(5, y.value_counts().min())`）**没有出现在计划文本的
  示范代码里**——计划给的示范代码是固定 `cross_val_score(clf, X, y, cv=5)`。
  这个修正是 coder 在真实运行遇到 `ValueError` 后自己诊断、自己改的，
  会话记录里完整保留了报错堆栈和修正后的 diff。这是原创调试过程的直接
  证据，不是抄录——抄录不会产生"运行失败→自己定位样本量不足→换用
  StratifiedKFold 动态算折数"这种和计划文本不同的修正路径。
- 未发现与某个具体开源仓库逐段一致的特征。coder 在完成信息里也明确声明
  "原创实现确认：explore.py 全部函数…为按计划从零实现，未从网络或任何
  开源项目复制代码"。

**(d) 通过。**

## (e) Commit 是否存在

```
9ed8eb2 (HEAD -> master, origin/master, origin/HEAD) feat(ml-practice): A1 数据探查与假高分复现
ee79eba feat(ml-practice): common 绘图字体与数据集加载工具
98a3896 feat(ml-practice): 初始化 ml-practice 仓库骨架
```

`git log --oneline -6` 确认该 commit 存在且为当前 HEAD。`git show --stat`
确认精确包含且仅包含两个文件（`data_cleaning/explore.py` 41 行、
`data_cleaning/test_explore.py` 32 行，共 73 行新增），无夹带其他改动。
Commit message 与计划 Step 5 要求的文本
`feat(ml-practice): A1 数据探查与假高分复现` 完全一致。`git status` 显示
工作区干净（除未跟踪的 `.agents/` 复核产物目录外无残留改动）。
**(e) 通过。**

## 额外观察（非阻塞）

- `test_explore.py` 文件内容与计划文本给出的测试代码逐字节一致（已用
  diff 核对），没有偏离。
- `naive_cv_score` 对固定 `cv=5` 的修正是对计划示范代码的合理工程修正
  （小样本教学场景下的真实边界条件），不影响对外签名，也不影响后续
  Task 4/6 对这三个函数的调用方式。

## 结论

五项核实全部通过：真实测试证据链完整且包含完整的环境排障与根因修正
过程（比单纯"一次性绿灯"更有力地证明了测试被真实执行）、无占位代码、
三个关键函数签名精确匹配计划、代码为原创实现（且修正路径本身就是原创性
的证据）、commit 真实存在且范围精确。没有阻塞项。
