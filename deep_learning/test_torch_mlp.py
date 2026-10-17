import numpy as np
from deep_learning.torch_mlp import TorchMLP, load_digits_xy, train_torch_mlp


def test_torch_mlp_shape():
    import torch
    model = TorchMLP(hidden_size=32)
    out = model(torch.zeros(5, 64))
    assert out.shape == (5, 10)


def test_load_digits_xy_normalized():
    X, y = load_digits_xy()
    assert X.shape[1] == 64
    assert X.max() <= 1.0 + 1e-9 and X.min() >= 0.0
    assert set(np.unique(y)) <= set(range(10))


def test_train_converges_like_handwritten():
    """对照入门系列手写版：同数据同结构，PyTorch 版也应收敛到高准确率、
    loss 单调下降趋势。"""
    X, y = load_digits_xy()
    res = train_torch_mlp(X, y, hidden_size=64, lr=0.3, n_epochs=200, seed=0)
    # 手写版文章里说 ~95%，PyTorch 版充分训练后应达到相近高位
    assert res["acc_history"][-1] > 0.90
    # loss 整体下降：末尾显著低于开头
    assert res["loss_history"][-1] < res["loss_history"][0] * 0.5
