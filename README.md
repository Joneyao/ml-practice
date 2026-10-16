# ml-practice

机器学习实战 —— 用 pandas / scikit-learn 把经典算法跑在真实数据上的完整工程流程，配套《机器学习实战》系列文章，承接《从零开始学机器学习》。

## 这个仓库是干什么的

《从零开始学机器学习》讲的是算法本身怎么用 NumPy 从零实现。这个仓库接着往下走：**把一份真实的脏数据，一步步带到能交给模型训练的状态**，中间每个决策（怎么判断缺失机制、要不要删异常值、用什么策略填补）都写成可跑的代码。

每个模块都自包含：核心实现 + 单元测试 + 导出真实运行数据的绘图脚本（文章里的每张图都来自代码真实输出，不是画的）。

## 目录结构

```
ml-practice/
├── common/           # 公共工具：数据集加载与缓存、绘图字体辅助
├── data_cleaning/    # 数据清洗：数据探查、缺失值填补、异常值检测与处理
└── docs/
    ├── references.md # 经典书籍锚点 + 官方文档 + 数据集出处清单
    └── articles/     # 配套系列文章 markdown（与公众号双发）
```

## 快速开始

```bash
pip install -r requirements.txt
pytest -v                                 # 跑全部测试
python -m data_cleaning.export_charts     # 导出数据清洗篇的真实数据配图
```

## 参考资料

代码根据经典教材和官方文档的原理从零重新实现，**只借鉴思路，不抄源码**。参考清单见 [docs/references.md](docs/references.md)，主要包括：Aurélien Géron《Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow》、Abhishek Thakur《Approaching (Almost) Any Machine Learning Problem》，以及 scikit-learn 官方文档（工业对照）。

## 开发约定

- **代码先行 + 真实数据配图：** 每个模块先实现、跑通、导出真实中间数据，这些数据直接作为配套文章的配图，可追溯到代码输出。
- **参考不摘抄：** 好的开源思路可以借鉴，但代码必须自己从原理重新实现，并在本地验证正确性。
- **测试驱动：** 每个模块先写测试，再写实现。
- **superpowers 工作流：** 按 brainstorming → writing-plans → executing-plans 的流程推进，每个 Task 独立验证、独立提交。

## License

MIT License，见 [LICENSE](LICENSE)。
