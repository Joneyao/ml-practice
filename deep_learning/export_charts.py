"""运行 F1/F2，导出真实训练曲线配图（PNG + CSV）到 charts_out/。"""
import os
import csv
import matplotlib.pyplot as plt
from common.plotting import setup_cjk_font, save_fig
from deep_learning import torch_mlp, cnn_image

OUT = os.path.join(os.path.dirname(__file__), "charts_out")


def export_f1_mlp_curve():
    X, y = torch_mlp.load_digits_xy()
    res = torch_mlp.train_torch_mlp(X, y, hidden_size=64, lr=0.3, n_epochs=200)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(res["loss_history"])
    ax1.set_title("PyTorch MLP 训练损失")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("交叉熵损失")
    ax2.plot(res["acc_history"], color="#55A868")
    ax2.set_title("PyTorch MLP 训练准确率")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("准确率")
    with open(os.path.join(OUT, "torch_mlp_curve.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "loss", "acc"])
        for i, (l, a) in enumerate(zip(res["loss_history"], res["acc_history"])):
            w.writerow([i, l, a])
    return save_fig(fig, OUT, "torch_mlp_curve.png")


def export_f2_cnn_curve():
    # 教学配置：稍大子集跑出像样曲线，仍控制 CPU 可接受
    train_ds = cnn_image.load_fashion_mnist(train=True, subset=5000)
    val_ds = cnn_image.load_fashion_mnist(train=False, subset=1000)
    hist = cnn_image.train_cnn(train_ds, val_ds, n_epochs=8, batch_size=64)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(hist["train_loss"], label="训练")
    ax1.plot(hist["val_loss"], label="验证")
    ax1.set_title("CNN 训练/验证损失")
    ax1.set_xlabel("epoch")
    ax1.legend()
    ax2.plot(hist["train_acc"], label="训练")
    ax2.plot(hist["val_acc"], label="验证")
    ax2.set_title("CNN 训练/验证准确率")
    ax2.set_xlabel("epoch")
    ax2.legend()
    with open(os.path.join(OUT, "cnn_curve.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_loss", "train_acc", "val_acc"])
        for i in range(len(hist["train_loss"])):
            w.writerow([i, hist["train_loss"][i], hist["val_loss"][i],
                        hist["train_acc"][i], hist["val_acc"][i]])
    return save_fig(fig, OUT, "cnn_train_val_curve.png")


def main():
    setup_cjk_font()
    os.makedirs(OUT, exist_ok=True)
    paths = [export_f1_mlp_curve(), export_f2_cnn_curve()]
    print("导出完成：")
    for p in paths:
        print("  ", p)


if __name__ == "__main__":
    main()
