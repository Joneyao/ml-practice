# Review: Task 1 (仓库骨架) — ml-practice batch-1 data-cleaning plan

## Verdict: APPROVED

## 验证范围

对照计划文件 `docs/superpowers/plans/2026-10-16-ml-practice-batch1-data-cleaning.md`
的 "Task 1: 仓库骨架" 一节（Step 1-7），逐项核实 `third-party/ml-practice/` 下
的实际产物与计划描述、以及 sibling 仓库 `third-party/ml-from-scratch` 的结构
风格做对比。

## (a) 六个产物是否存在且内容匹配计划

全部六个文件存在，内容逐一核对如下：

- `requirements.txt`：六行依赖（pandas/numpy/scikit-learn/matplotlib/
  seaborn/pytest），版本下限与计划文本逐字一致。
- `.gitignore`：`__pycache__/`、`*.pyc`、`.pytest_cache/`、`*.egg-info/`，
  以及两条带中文注释的缓存排除规则（`data/*.csv`、`**/charts_out/`），
  与计划文本逐字一致。
- `LICENSE`：标准 MIT 模板，版权行为 `Copyright (c) 2026 轻松学AI`，与计划
  "Copyright (c) 2026 轻松学AI" 的要求一致。
- `README.md`：见下方 (b) 的专项核对。
- `common/__init__.py`、`data_cleaning/__init__.py`：均为 0 字节空文件，
  符合"包标识"要求。

六项均通过。

## (b) README 风格镶嵌与"参考思路不抄源码"声明

逐段对照 `third-party/ml-from-scratch/README.md`：

- 结构完全镶嵌：标题 → 一句话定位 → "这个仓库是干什么的" → 目录结构代码块
  → "快速开始" → "参考资料" → "开发约定" → License，七个区块顺序和命名
  与 sibling 仓库一一对应。
- "快速开始"命令改写得体：`pytest -v` / `python -m data_cleaning.export_charts`
  对应本仓库实际模块名（而非照搬 sibling 的 `foundations.export_charts`）。
- "参考资料"段落明确写出"**只借鉴思路，不抄源码**"，呼应用户在原始需求
  中第 10 条确认的原则（"好的开源代码可以直接抄录……本地验证，尽可能自己
  实现"最终被用户收敛为"只借鉴思路"，见第 11 条）；并点名 Hands-On ML、
  Approaching (Almost) Any ML Problem、scikit-learn 官方文档三个参考源，
  与计划 Step 4 的要求一致。
- "开发约定"额外加入了 sibling 没有的一条 "superpowers 工作流" 说明
  （brainstorming → writing-plans → executing-plans），这是对计划上下文
  的合理补充，不违反"镶嵌 sibling 风格"的要求。
- 未发现 TODO / TBD / placeholder 占位文本（已用 grep 核实，无匹配）。

唯一需要注意但不构成阻塞项的点：README 中引用了 `docs/references.md` 和
`docs/articles/`，这两个路径在当前 Task 1 范围内尚不存在——但这是预期的，
因为它们属于 Task 7（docs + submodule 挂载）的产物，Task 1 的文件清单本身
不包含 docs/ 目录。README 提前"声明"了后续任务要建的结构，这是正常的仓库
介绍写法，不是本任务的缺陷。

## (c) .gitignore 与 requirements.txt 精确匹配

两个文件均与计划 Step 1/Step 2 给出的代码块逐字节比对一致（包括注释文字、
空行位置），无遗漏也无额外内容。

## (d) Commit 是否真实存在

```
98a3896 (HEAD -> master, origin/master, origin/HEAD) feat(ml-practice): 初始化 ml-practice 仓库骨架
```

`git log --oneline -5` 确认该 commit 存在，且为 HEAD（最新提交）。
`git show --stat` 确认该 commit 精确包含且仅包含计划 Task 1 列出的六个
文件（.gitignore / LICENSE / README.md / common/__init__.py /
data_cleaning/__init__.py / requirements.txt），无夹带其他改动。
Commit message 与计划 Step 7 要求的文本
`feat(ml-practice): 初始化 ml-practice 仓库骨架` 完全一致。
`git status --short third-party/ml-practice` 输出为空，说明没有未提交的
残留改动。

## (e) 是否有提前进行 GitHub 仓库创建 / submodule 挂载 / push

- `.gitmodules` 中没有 `ml-practice` 条目，`git submodule status` 的输出
  中也没有 ml-practice，确认尚未被注册为 submodule。
- `third-party/ml-practice/.git` 不存在，说明这里仍是主仓库的普通跟踪
  目录，没有被初始化成独立 git 仓库或被替换成 submodule 链接。
- commit 98a3896 已经是 `origin/master` 的一部分（`HEAD -> master,
  origin/master, origin/HEAD` 都指向同一个 commit），即当前所在的主仓库
  （WeChatAutoPress）分支已经推送过——但这是 WeChatAutoPress 自身仓库的
  日常推送节奏（符合用户规则"每次 git commit 之后必须立即 git push"），
  与"ml-practice 作为独立仓库推送到 GitHub"是两件不同的事。本任务范围内
  没有看到为 ml-practice 单独创建远程仓库、添加 remote 或执行提前的
  独立推送动作，Task 1 的"仅建骨架、不碰 GitHub/submodule"边界遵守良好。

## 结论

六个产物、内容、commit message、提交范围全部符合计划 Task 1 的要求，没有
占位文本，没有提前越界进行 GitHub/submodule 操作。没有阻塞项。
