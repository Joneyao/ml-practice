"""F2：小型 CNN 图像分类——Fashion-MNIST，训练/验证曲线诊断过拟合。

教学配置，非最优精度。2 个卷积块的小 CNN，默认少 epoch，CPU 可跑。

说明：原计划用 CIFAR-10，但其 170MB 的下载在本环境极慢；设计文档允许用
Fashion-MNIST 作为等价替代（同样是 10 类图像分类，适合讲卷积与过拟合曲线），
且体积小得多（约 30MB）。Fashion-MNIST 是 1 通道 28×28 灰度图。
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T


class SmallCNN(nn.Module):
    """2 卷积块 + 全连接（Fashion-MNIST 版，1 通道 28×28）：
    Conv(1→16)→ReLU→MaxPool → Conv(16→32)→ReLU→MaxPool → FC→10。
    28×28 经两次 2×2 池化 → 7×7。
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


def load_fashion_mnist(root="data/fashion_mnist", train=True, subset=None):
    """加载 Fashion-MNIST（首次自动下载）。subset 给定时只取前 N 张。"""
    transform = T.Compose([T.ToTensor()])
    ds = torchvision.datasets.FashionMNIST(root=root, train=train,
                                           download=True, transform=transform)
    if subset is not None:
        ds = Subset(ds, list(range(min(subset, len(ds)))))
    return ds


def _run_epoch(model, loader, criterion, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    for xb, yb in loader:
        if train:
            optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        if train:
            loss.backward()
            optimizer.step()
        total_loss += float(loss.item()) * len(yb)
        correct += int((logits.argmax(1) == yb).sum())
        n += len(yb)
    return total_loss / n, correct / n


def train_cnn(train_ds, val_ds, n_epochs=5, lr=0.001, batch_size=64, seed=0):
    """训练循环，每 epoch 记录 train/val 的 loss 和 acc。"""
    torch.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    model = SmallCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    for _ in range(n_epochs):
        tl, ta = _run_epoch(model, train_loader, criterion, optimizer)
        with torch.no_grad():
            vl, va = _run_epoch(model, val_loader, criterion, None)
        hist["train_loss"].append(tl)
        hist["train_acc"].append(ta)
        hist["val_loss"].append(vl)
        hist["val_acc"].append(va)
    hist["model"] = model
    return hist
