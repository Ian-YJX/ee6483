import os
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score

# ========== 配置 ==========
DATA_DIR = "/home/ian/ee6483/data"
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform_val = None  # 全局变量供 predict_testset 使用


# ========== 模型 ==========
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ========== 训练与验证 ==========
def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for pixel, labels in dataloader:
        pixel, labels = pixel.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        logits = model(pixel)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for pixel, labels in dataloader:
            pixel, labels = pixel.to(DEVICE), labels.to(DEVICE)
            logits = model(pixel)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    acc = correct / total
    f1 = f1_score(y_true, y_pred, average="macro")
    return total_loss / total, acc, f1


def predict_testset(model, device=DEVICE):
    """预测测试集并生成 submission.csv"""
    global transform_val

    test_ds = datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True, transform=transform_val
    )
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4)

    model.eval()
    preds_all = []
    with torch.no_grad():
        for imgs, _ in test_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            preds = torch.argmax(logits, dim=1)
            preds_all.extend(preds.cpu().numpy())

    df = pd.DataFrame({"id": list(range(len(preds_all))), "label": preds_all})
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/submission_cnn.csv", index=False)
    print("submission_cnn.csv saved to results/")


# ========== 主函数 ==========
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint file (.pt)")
    args = parser.parse_args()

    global transform_val
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

    train_ds = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform_train)
    val_ds = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform_val)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = SimpleCNN(num_classes=10).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # ===== 加载 checkpoint =====
    best_val_acc = 0.0
    if args.resume and os.path.isfile(args.resume):
        print(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=DEVICE)

        # 新格式（dict中包含 model_state_dict）
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            best_val_acc = checkpoint.get("best_val_acc", 0.0)
            print(f"Loaded new-format checkpoint. best_val_acc={best_val_acc:.4f}")
        else:
            # 旧格式：纯 state_dict
            model.load_state_dict(checkpoint)
            print("Loaded old-format checkpoint (no best_val_acc info).")

    elif args.resume:
        print(f"Checkpoint not found: {args.resume}")

    # ===== 训练 =====
    train_losses, val_losses, train_accs, val_accs, val_f1s = [], [], [], [], []

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)

        print(f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
              f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f} | val_f1={val_f1:.4f}")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("checkpoints/cifar10", exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "best_val_acc": best_val_acc,
            }, "checkpoints/cifar10/best_cnn_model.pt")
            print(f"New best model saved (val_acc={val_acc:.4f})")

    print(f"Training completed. Best validation accuracy: {best_val_acc:.4f}")

    # ===== 绘图 =====
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label="Train Acc")
    plt.plot(epochs, val_accs, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_curves_cnn.png")
    plt.show()

    predict_testset(model)


if __name__ == "__main__":
    main()
