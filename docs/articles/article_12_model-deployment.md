> 作者：轻松学AI
> 日期：2026-10-16
> 风格：通俗科普 + 实操教程
> 状态：待发布

# 机器学习实战 #12：训好的模型怎么用起来，从 pickle 到一个能调用的接口

前面十一篇，我们在 Jupyter notebook 里做了很多事——清洗数据、造特征、训模型、调参。但这些成果都困在 notebook 里。一个在你电脑上跑出 AUC 0.85 的模型，别人没法用，产品也集成不了。

从"notebook 里跑出一个好模型"到"别人能用 HTTP 调用它"，中间隔着几道坎：怎么把训好的模型存下来、怎么重新加载、怎么包成一个服务、怎么保证线上线下处理数据的方式一致。这一篇就把这几道坎走一遍，做出一个最小但完整的预测接口。

**先把话说清楚：这一篇做的是一个本地演示接口，没有鉴权、没有限流、没有 HTTPS。** 它的目的是打通"模型 → 可调用"的主干，不是一个能直接上生产的服务。生产还差一整套工程，文末会讲清楚差在哪。

![从notebook模型到可调用接口的三步](deploy_journey.png)

## 第一道坎：把模型存下来

模型训练完，它活在内存里——一堆学好的权重和预处理器的状态。程序一退出，这些就没了。要让模型能被复用，得先把它序列化（serialize）存到硬盘。

Python 生态里最常用的是 `joblib`，它专门优化了对 NumPy 数组的序列化，比标准库的 `pickle` 更适合存机器学习模型。配套仓库里的做法是：训练一个完整的 Pipeline（标准化 + 逻辑回归），然后一行 `joblib.dump` 存成 `.pkl` 文件。

```python
FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]


def build_pipeline():
    """标准化 + 逻辑回归的最简 pipeline。"""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])


def train_and_save(model_path):
    """用 Telco 数据训练并 joblib.dump 到 model_path。"""
    X, y = _prepare(load_telco_churn())
    pipe = build_pipeline().fit(X, y)
    joblib.dump(pipe, model_path)
    return pipe
```

这里有个关键决定：存的是**整个 Pipeline**，不是光秃秃的模型。为什么重要？因为 Pipeline 里打包了预处理步骤（标准化）。如果只存模型、不存预处理器，那加载模型之后你还得记得"当初是怎么标准化的、用的哪些均值和方差"，手动复现一遍——极易出错。把预处理和模型一起存进 Pipeline，加载进来就是一个完整的"输入原始特征、输出预测"的黑盒，不用操心中间步骤。这就是前面讲 Pipeline 时埋下的伏笔，到部署这一步才完全兑现它的价值。

## 第二道坎：包成一个能调用的接口

模型存成文件之后，怎么让别人用？最通用的方式是包成一个 HTTP 接口——别人发一个请求过来，附上客户数据，接口返回预测结果。这样任何语言、任何系统都能调用，不用关心模型是 Python 写的还是别的。

Python 里做这件事，FastAPI 是当下最顺手的选择：声明式、自带数据校验、几十行就能起一个服务。配套仓库里的接口长这样：

```python
class ChurnRequest(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float


class ChurnResponse(BaseModel):
    churn_probability: float
    churn_prediction: int


def create_app(model_path):
    """加载 joblib 模型，暴露 POST /predict。"""
    app = FastAPI(title="Telco Churn 预测（演示）")
    model = joblib.load(model_path)

    @app.post("/predict", response_model=ChurnResponse)
    def predict(req: ChurnRequest):
        X = pd.DataFrame([{
            "tenure": req.tenure,
            "MonthlyCharges": req.MonthlyCharges,
            "TotalCharges": req.TotalCharges,
        }])
        proba = float(model.predict_proba(X)[0, 1])
        return ChurnResponse(churn_probability=proba, churn_prediction=int(proba >= 0.5))

    return app
```

几个值得说的设计点。

`ChurnRequest` 和 `ChurnResponse` 是 pydantic 模型，它们定义了接口的"契约"——请求必须带这三个字段、类型得对，否则 FastAPI 自动返回 422 错误，根本不会跑到模型那一步。这省掉了大量手写的输入校验，也让接口的输入输出一目了然。

模型在 `create_app` 里加载**一次**，而不是每次请求都重新从硬盘读。加载模型是慢操作，放在请求处理函数里会让每个请求都卡一下。启动时加载好、常驻内存，请求来了直接用——这是服务化的基本功。

整个请求-响应的流程是这样的：

```mermaid
---
title: 一次预测请求的流转
---
flowchart LR
    A["客户端 POST /predict<br/>带 3 个特征"] --> B["pydantic 校验字段<br/>类型不对直接拒绝"]
    B --> C["组装成 DataFrame<br/>喂给加载好的 Pipeline"]
    C --> D["Pipeline 内部<br/>标准化 + 逻辑回归"]
    D --> E["返回流失概率<br/>和 0/1 预测"]
```

注意组装 DataFrame 那一步：请求进来的三个字段，被拼成一个和训练时列名完全一致的 DataFrame，再喂给 Pipeline。这是"线上线下特征一致性"最朴素的保证——列名、顺序、含义都得和训练时对齐，错一个模型就会给出莫名其妙的结果。真实项目里，这种一致性是部署最容易出 bug 的地方。

验证接口能用，不必真的起一个服务器。FastAPI 自带的 `TestClient` 可以在代码里直接发请求：

```python
def test_predict_endpoint_returns_probability(tmp_path):
    model_path = str(tmp_path / "model.pkl")
    train_and_save(model_path)
    app = create_app(model_path)
    client = TestClient(app)
    resp = client.post("/predict", json={
        "tenure": 2, "MonthlyCharges": 85.0, "TotalCharges": 170.0,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
```

跑通这个测试，就证明了完整链路打通：训练 → 存盘 → 加载 → 起接口 → 发请求 → 拿到一个合理的流失概率。真要对外服务时，再用 `uvicorn app:app` 把它跑成一个真实的 HTTP 服务即可。

## 这离生产还差多少

现在到了必须说清楚的部分。上面这个接口能跑、能预测，但它是个**演示**，不是生产服务。差距很大，而且每一条都不是可选项：

![演示接口与生产服务还差的几层工程](demo_vs_prod.png)

**鉴权**。现在任何人知道地址都能调。生产接口必须有身份验证——API Key、OAuth、或者至少放在内网网关后面。一个裸奔的预测接口，等于把你的模型和算力免费送给全世界。

**限流**。没有限流，一个客户端狂刷请求就能把服务打垮。生产需要按调用方限流，保护服务稳定。

**输入加固**。pydantic 保证了类型对，但没保证值合理——有人传 `tenure=-999` 或者 `MonthlyCharges=1e18`，模型会算出一个无意义的结果还一本正经返回。生产需要业务层面的边界校验。

**HTTPS**。演示用 HTTP 明文传输，生产必须 HTTPS，否则客户数据在网络上裸奔。

**模型版本管理**。模型会迭代。生产需要知道当前线上是哪个版本的模型、能快速回滚到上一版、能对比新旧版本的线上表现。一个写死路径加载单个 `.pkl` 的接口，做不到这些。

**监控**。模型上线后性能会随数据分布漂移而退化。生产需要持续监控预测分布、定期用新数据重新评估，发现退化及时重训。

这些不是"锦上添花"，是一个预测服务从"能跑"到"能依赖"之间的全部工程。把它们一次性讲完是另一个系列的量，但你至少得知道：**从 notebook 到生产，模型本身只是起点，外面还包着一整层工程。**

这一篇的技术组件——`joblib`、FastAPI、pydantic——都是现成的，拼起来也就几十行。但它串起了整个系列的收尾逻辑：Pipeline 让模型连同预处理一起打包，joblib 让它落盘复用，FastAPI 让它对外可调。从一份脏 CSV 开始，到一个能响应 HTTP 请求的预测接口，一条完整的实战链路到此走通。

工程线到这里告一段落。下一组是整个系列的彩蛋——从 scikit-learn 跨到 PyTorch，看看入门系列里我们手写的那个神经网络，用深度学习框架重写会是什么样。

---

本文的模型序列化与 FastAPI 接口代码都在配套仓库里：[ml-practice](https://github.com/Joneyao/ml-practice)。

- [Kaggle: Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- [scikit-learn: Model persistence](https://scikit-learn.org/stable/model_persistence.html)
- [FastAPI 官方文档](https://fastapi.tiangolo.com/)
- [joblib 官方文档](https://joblib.readthedocs.io/)
- [机器学习模型部署最佳实践（InfoQ 中文）](https://www.infoq.cn/article/ml-model-deployment)
