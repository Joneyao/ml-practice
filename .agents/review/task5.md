# Review: Task 5 (A3 — outliers.py + test_outliers.py) — ml-practice batch-1 data-cleaning plan

## Verdict: APPROVED

## 验证范围

对照计划文件 `docs/superpowers/plans/2026-10-16-ml-practice-batch1-data-cleaning.md`
的 "Task 5: A3 — 异常值检测与处理（outliers）" 一节，核对
`data_cleaning/outliers.py`、`data_cleaning/test_outliers.py` 两个产物，
以及 coder 会话记录
（`~/.kiro/sessions/80d0747a853dd530/sess_0f2a041b-5e99-4958-bc90-17dd535557e3/messages.jsonl`，
共 42 行）里实际执行的命令与输出。未重新跑测试——证据链本身完整且与计划
文本逐字一致，没有出现需要单点抽查的疑点。

## (a) 测试是否真的跑过，输出是否为 "3 passed"

从会话记录中核实到两次真实的 `execute_bash` 调用，构成完整的 TDD
红灯→绿灯证据链：

第一次（写完测试、尚未写实现，确认失败，行 20-21）：
```
cd .../third-party/ml-practice && python -m pytest data_cleaning/test_outliers.py -v
```
结果：
```
ImportError ... ModuleNotFoundError: No module named 'data_cleaning.outliers'
collected 0 items / 1 error
Exit Code: 2
```
符合红灯阶段。

第二次（写完 `outliers.py` 后，确认通过，行 27-28）：
```
data_cleaning/test_outliers.py::test_detect_iqr_flags_large_areas PASSED [ 33%]
data_cleaning/test_outliers.py::test_detect_zscore_flags_large_areas PASSED [ 66%]
data_cleaning/test_outliers.py::test_compare_outlier_strategies_returns_all_rows PASSED [100%]
======================== 3 passed in 1.14s =========================
Exit Code: 0
```
命令、cwd、stdout、exit code 均完整可见，三条测试名与
`test_outliers.py` 文件内实际的三个测试函数（`test_detect_iqr_flags_large_areas`、
`test_detect_zscore_flags_large_areas`、`test_compare_outlier_strategies_returns_all_rows`）
逐一对应，无遗漏无多余。coder 最终 `send_message`（行 35）中引用的
"3 passed" 与该真实工具调用结果一致。**(a) 通过。**

## (b) 是否存在占位/TODO 代码

对 `outliers.py`、`test_outliers.py` 做了
`TODO|FIXME|XXX|NotImplementedError|placeholder` 关键词搜索（含大小写
不敏感），零匹配（grep exit code 1）。`detect_iqr`、`detect_zscore`、
`_winsorize`、`_fit_and_score`、`compare_outlier_strategies` 五个函数
均为完整实现，无空函数体、无 `pass` 占位。**(b) 通过。**

## (c) 函数签名是否与计划精确一致

实际读取 `outliers.py` 核实：

| 函数 | 实际签名 | 计划要求 | 一致 |
|------|----------|----------|------|
| `detect_iqr` | `(series: pd.Series, k: float = 1.5) -> pd.Series`（布尔掩码） | `detect_iqr(series, k=1.5) -> pd.Series` | ✓ |
| `detect_zscore` | `(series: pd.Series, threshold: float = 3.0) -> pd.Series`（布尔掩码） | `detect_zscore(series, threshold=3.0) -> pd.Series` | ✓ |
| `compare_outlier_strategies` | `(df: pd.DataFrame, col: str, target_col: str) -> pd.DataFrame` | `compare_outlier_strategies(df, col, target_col) -> pd.DataFrame` | ✓ |

`compare_outlier_strategies` 返回的 DataFrame 含 `strategy`/`n_samples`/
`r2`/`rmse` 四列，三个策略（`keep`/`drop`/`winsorize`）与计划及测试断言
（`test_compare_outlier_strategies_returns_all_rows` 断言
`set(result["strategy"]) == {"keep", "drop", "winsorize"}` 且
`result["r2"].notna()` 全真）完全匹配。该函数会被 Task 6
`export_charts.py` 的 `export_a3_strategy_comparison()` 直接调用
`compare_outlier_strategies(df, col="GrLivArea", target_col="SalePrice")`，
再取 `result["r2"]` 画柱状图——签名和列名均兼容，不会阻塞 Task 6。
**(c) 通过，无阻塞风险。**

## (d) 是否为原创实现，非抄录

- TDD 轨迹本身是证据：会话行 17-18 先写测试文件，行 20-21 跑红灯确认
  `ModuleNotFoundError`，行 24-25 才写实现，行 27-28 跑绿灯——这是
  先写验证标准、再写实现的顺序，不是先有现成代码再凑测试的模式。
- 会话行 13-14 显示 coder 在动手实现前先用 `python3 -c` 探测本机
  `sklearn.__version__`（1.9.1）和 `root_mean_squared_error` 是否存在，
  确认后才决定不需要 `sqrt(mse)` 兜底——这是针对本地环境做的具体判断，
  不是照抄教程代码的痕迹。
- 会话中仅出现 `read_file`、`execute_bash`、`fs_write`、`send_message`
  四类工具调用，没有任何 `fetch_url`/网络搜索类调用，没有从外部获取
  代码的路径。**(d) 通过。**

## (e) Commit 是否存在

```
1d52867 (HEAD -> master, origin/master, origin/HEAD) feat(ml-practice): A3 异常值检测与处理策略对比
b16e356 feat(ml-practice): A2 缺失值填补策略对比
9ed8eb2 feat(ml-practice): A1 数据探查与假高分复现
ee79eba feat(ml-practice): common 绘图字体与数据集加载工具
98a3896 feat(ml-practice): 初始化 ml-practice 仓库骨架
a2d985d docs: 机器学习实战第一批（数据清洗）实现计划
8e87901 docs: 机器学习实战系列设计文档
d061b1f feat: add per-section charts to MJPEG/H.264 #09-17 and publish to WeChat draft
```
`git log --oneline -8` 确认该 commit 存在、为当前 HEAD，且与
`origin/master` 同步（post-commit hook 自动推送，会话行 32/33 中
coder 如实说明，未自行额外执行 `git push`）。`git show --stat 1d52867`
确认精确包含且仅包含两个文件（`outliers.py` 54 行、`test_outliers.py`
35 行，共 89 行新增），无夹带其他改动。Commit message 与计划 Step 5
要求的文本 `feat(ml-practice): A3 异常值检测与处理策略对比` 完全一致。
**(e) 通过。**

## (f) RMSE 实现方式与列名

计划预留了 fallback 分支（若 `sklearn.metrics.root_mean_squared_error`
不可用则用 `sqrt(mean_squared_error)` 代替，但仍保持 `r2`/`rmse` 两列
不变）。实际会话行 13-14 已核实本机 `sklearn==1.9.1` 原生支持
`root_mean_squared_error`（>=1.4 引入），coder 直接导入使用，未触发
fallback 分支，无需额外验证 sqrt 等价性。`_fit_and_score` 返回的字典
键严格为 `{"r2": ..., "rmse": ...}`，与计划要求及 Task 6 export_charts
里 `result["r2"]` 的取用方式一致。**(f) 通过，fallback 未触发，列名
符合要求。**

## 结论

六项核实全部通过：真实测试证据链完整（红灯 `ModuleNotFoundError` →
绿灯 `3 passed in 1.14s`，命令/输出均可追溯）、无占位代码、三个关键
函数签名精确匹配计划（供 Task 6 复用不受阻）、代码为原创实现（TDD 顺序
+ 针对本机 sklearn 版本的具体探测 + 会话中无外部代码获取路径）、
commit 真实存在且范围精确（仅 2 文件 89 行）、RMSE 使用原生
`root_mean_squared_error`（未触发 fallback），列名 `r2`/`rmse` 符合
要求。没有阻塞项。
