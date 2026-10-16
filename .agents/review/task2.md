# Review: Task 2 (公共工具：绘图 + 数据加载) — ml-practice batch-1 data-cleaning plan

## Verdict: APPROVED

## 验证范围

对照计划文件 `docs/superpowers/plans/2026-10-16-ml-practice-batch1-data-cleaning.md`
的 "Task 2: 公共工具（绘图 + 数据加载）" 一节，核实 `common/plotting.py`、
`common/data_loader.py`、`common/test_common.py` 三个产物，以及 coder 会话
记录（`sess_ab5126f1-a1e9-4dd2-92b0-6c5d632a3715/messages.jsonl`）里实际执行
的命令与输出。

## (a) 测试是否真的跑过，输出是否为 "3 passed"

从 coder 会话的 `messages.jsonl` 中直接核实到两次真实的 `execute_bash` 调用：

第一次（实现前，确认失败）：
```
cd .../third-party/ml-practice && python -m pytest common/test_common.py -v
```
结果：`ImportError: cannot import name 'plotting' from 'common'` ——
`collected 0 items / 1 error`，符合 TDD 红灯阶段的预期。

第二次（实现后，确认通过）：
```
common/test_common.py::test_setup_font_returns_name PASSED               [ 33%]
common/test_common.py::test_load_titanic_from_cache PASSED               [ 66%]
common/test_common.py::test_load_house_prices_from_cache PASSED          [100%]
============================== 3 passed in 0.67s ===============================
Exit Code: 0
```

本次复核**没有重新跑测试**（按任务要求，复核者应信任已记录的真实输出，
仅在有具体疑点时才做单点抽查）。这里直接采信会话记录里的 "3 passed"，
因为证据链完整：命令、cwd、stdout、exit code 均可见，且红灯→绿灯的顺序
符合 TDD 流程，不存在"直接贴结果、未见命令"的可疑模式。

## (b) 是否存在占位/TODO 代码

对 `plotting.py`、`data_loader.py`、`test_common.py` 三个文件做了
`TODO|FIXME|XXX|NotImplementedError|placeholder` 关键词搜索，无匹配。
`load_titanic` / `load_house_prices` 的下载分支（`seaborn.load_dataset` /
`fetch_openml`）虽然测试里没有触发（测试用缓存路径），但分支代码本身是
完整实现，不是占位符或 `pass`。

## (c) 函数签名是否与计划精确一致

用 `ast.parse` 核实了三个产物函数的签名，结果：

- `setup_cjk_font() -> str`：无参数，返回类型标注 `str`。✅ 与计划一致。
- `save_fig(fig, out_dir, filename) -> str`：三个位置参数，返回类型标注
  `str`。✅ 与计划一致（计划文本本身没写返回类型标注，但实现加了更严格的
  `-> str`，不是破坏性变更，属于合理加强）。
- `load_titanic(cache_dir="data", force_download=True)`：两个参数名、
  默认值均与计划一致。✅
- `load_house_prices(cache_dir="data", force_download=True)`：同上。✅

这四个签名是后续任务（A1/A2/A3）会直接 import 调用的接口，核实结果是
精确匹配，没有参数名、顺序或默认值的偏差，不存在阻塞性接口风险。

## (d) 代码是否原创，而非直接抄录

- `plotting.py` 与 sibling 仓库 `third-party/ml-from-scratch/common/plotting.py`
  的 `setup_cjk_font` / `save_fig` 两个函数逐字节 diff **完全相同**。这不算
  违规抄袭——计划文本第 147 行明确写了"照搬 `ml-from-scratch/common/plotting.py`
  的实现（已验证可用）"，即这是计划主动要求的复用自身项目内已验证代码，
  不是从外部开源项目抄录，符合用户在原始需求第 10/11 条确认的"只借鉴思路，
  不抄开源源码"原则（这里复用的是仓库维护者自己此前验证过的实现，不涉及
  第三方版权）。
- `data_loader.py` 是本任务新增内容，`ml-from-scratch` 没有对应模块可比对。
  实现是标准的"本地缓存优先，否则调用标准库下载"模式（`os.path.exists`
  判断 + `pd.read_csv` / `sns.load_dataset` / `fetch_openml`），结构是惯用
  写法，没有发现与某个具体开源仓库逐段一致的特征（无特有变量命名、无独特
  注释风格、无出处字符串），判断为原创实现。
- coder 在完成信息里也明确声明"Code is original — not copied from any web
  source"。

## (e) Commit 是否存在

```
ee79eba (HEAD -> master, origin/master, origin/HEAD) feat(ml-practice): common 绘图字体与数据集加载工具
```

`git log --oneline -5` 确认该 commit 存在且为当前 HEAD。`git show --stat`
确认精确包含且仅包含三个文件（`common/data_loader.py`、`common/plotting.py`、
`common/test_common.py`，共 100 行新增），无夹带其他改动。Commit message
与计划 Step 6 要求的文本完全一致。

## (f) plotting.py 是否自包含，不依赖本项目的 backend font_helper

对 `plotting.py` 和 `data_loader.py` 做了 `font_helper|backend` 关键词
搜索，无匹配。`plotting.py` 的 import 列表仅有 `os`、`matplotlib`、
`matplotlib.pyplot`、`matplotlib.font_manager`，均为第三方/标准库，没有
任何指向本项目 `backend/src/utils/font_helper.py` 的引用。字体选择逻辑
是仓库自带的 `_CJK_CANDIDATES` 列表 + `fm.fontManager.ttflist` 检测，完全
自洽，符合计划"零外部依赖"的要求。

## 结论

六项核实全部通过：真实测试证据链完整（红灯→绿灯，"3 passed"）、无占位
代码、四个关键函数签名精确匹配计划、代码来源可解释且不违反版权约定、
commit 真实存在且范围精确、`plotting.py` 自包含不依赖主项目 font_helper。
没有阻塞项。
