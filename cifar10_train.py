import os
import argparse
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTModel
import matplotlib.pyplot as plt


# ========== 配置 ==========
MODEL_NAME = "google/vit-base-patch16-224"
DATA_DIR = "/home/ian/ee6483/data"
BATCH_SIZE = 16
NUM_EPOCHS = 25
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ========== 模型 ==========
class ViTWithMLP(nn.Module):
    def __init__(self, model_name, num_classes=10, mlp_hidden=512, dropout=0.1):
        super().__init__()
        self.backbone = ViTModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, num_classes)
        )

    def forward(self, pixel_values):
        out = self.backbone(pixel_values=pixel_values)
        cls_feat = out.last_hidden_state[:, 0, :]
        logits = self.mlp(cls_feat)
        return logits



# ========== 训练与验证 ==========
def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for pixel, labels in dataloader:
        pixel, labels = pixel.to(DEVICE), labels.to(DEVICE)
        logits = model(pixel)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
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
    with torch.no_grad():
        for pixel, labels in dataloader:
            pixel, labels = pixel.to(DEVICE), labels.to(DEVICE)
            logits = model(pixel)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


# ========== 主函数 ==========
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint file (.pt) to resume training")
    args = parser.parse_args()

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    train_ds = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform)
    val_ds = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 初始化模型
    model = ViTWithMLP(MODEL_NAME, num_classes=10).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # 如果提供 checkpoint 路径则加载
    if args.resume and os.path.isfile(args.resume):
        print(f"Loading checkpoint from {args.resume}")
        state_dict = torch.load(args.resume, map_location=DEVICE)
        model.load_state_dict(state_dict)
        print("Checkpoint loaded successfully.")
    elif args.resume:
        print(f"Checkpoint not found: {args.resume}")

    # 记录指标
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    # 训练循环
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

    # 保存权重
    os.makedirs("checkpoints/cifar10", exist_ok=True)
    ckpt_path = f"checkpoints/cifar10/vit_mlp_cifar10_epoch{NUM_EPOCHS}.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Checkpoint saved to {ckpt_path}")

    # ===== 绘制曲线 =====
    epochs = range(1, NUM_EPOCHS + 1)
    plt.figure(figsize=(10, 4))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label="Train Acc")
    plt.plot(epochs, val_accs, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_curves.png")
    plt.show()

if __name__ == "__main__":
    main()
