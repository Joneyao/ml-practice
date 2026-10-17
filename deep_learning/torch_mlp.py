"""F1：PyTorch 版 MLP——对齐入门系列手写版（64→hidden→10），验证框架与手写一致。

教学配置，非最优精度。用 sklearn 8×8 digits（与手写版同源），全批量梯度下降，
CrossEntropyLoss（内含 softmax）+ SGD，和手写版的前向/反向/更新一一对应。
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_digits


class TorchMLP(nn.Module):
    """单隐藏层 MLP：Linear(64→hidden) → ReLU → Linear(hidden→10)。
    结构与 ml-from-scratch 手写版完全一致；softmax 由 CrossEntropyLoss 隐含。
    """

    def __init__(self, hidden_size=64):
        super().__init__()
        self.fc1 = nn.Linear(64, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 10)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def load_digits_xy():
    """sklearn 8×8 digits，像素 0-16 除以 16 归一化（对齐手写版）。"""
    digits = load_digits()
    X = digits.data.astype(np.float32) / 16.0
    y = digits.target.astype(np.int64)
    return X, y


def train_torch_mlp(X, y, hidden_size=64, lr=0.1, n_epochs=100, seed=0):
    """全批量训练，记录每 epoch 的交叉熵损失和训练准确率。"""
    torch.manual_seed(seed)
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    model = TorchMLP(hidden_size=hidden_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    loss_history, acc_history = [], []
    for _ in range(n_epochs):
        optimizer.zero_grad()
        logits = model(Xt)                 # 前向
        loss = criterion(logits, yt)       # softmax + 交叉熵
        loss.backward()                    # autograd 自动反向
        optimizer.step()                   # 梯度下降更新
        with torch.no_grad():
            pred = logits.argmax(dim=1)
            acc = float((pred == yt).float().mean())
        loss_history.append(float(loss.item()))
        acc_history.append(acc)

    return {"model": model, "loss_history": loss_history, "acc_history": acc_history}
