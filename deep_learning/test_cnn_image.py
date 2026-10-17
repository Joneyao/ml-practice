import pytest
from deep_learning.cnn_image import SmallCNN, load_fashion_mnist, train_cnn


def test_small_cnn_output_shape():
    import torch
    model = SmallCNN()
    out = model(torch.zeros(4, 1, 28, 28))  # Fashion-MNIST: 1 通道 28×28
    assert out.shape == (4, 10)


@pytest.mark.slow
def test_train_cnn_runs_on_subset():
    """极小子集 + 1 epoch，只验证训练循环能跑通并返回四条曲线。
    标记 slow 是因为首次需下载 Fashion-MNIST（约 30MB）。"""
    train_ds = load_fashion_mnist(train=True, subset=200)
    val_ds = load_fashion_mnist(train=False, subset=100)
    res = train_cnn(train_ds, val_ds, n_epochs=1, batch_size=32)
    assert len(res["train_loss"]) == 1
    assert len(res["val_acc"]) == 1
    assert 0.0 <= res["val_acc"][0] <= 1.0
