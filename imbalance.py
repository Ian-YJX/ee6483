import os
import random
import argparse
from collections import defaultdict, Counter
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms
from transformers import ViTModel
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sns


# ================== Config ==================
DATA_DIR = "/home/ian/ee6483/data"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


# ================== Dataset Utils ==================
def make_imbalanced_dataset(dataset, imbalance_ratio=0.2, imbalanced_classes=[2, 3, 5]):
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)
    imbalanced_indices = []
    for cls, indices in class_indices.items():
        if cls in imbalanced_classes:
            keep_n = int(len(indices) * imbalance_ratio)
            indices = random.sample(indices, keep_n)
        imbalanced_indices.extend(indices)
    print(f"Imbalanced classes {imbalanced_classes} reduced to {imbalance_ratio*100}% samples.")
    return Subset(dataset, imbalanced_indices)


def get_class_weights(dataset, num_classes=10):
    labels = [dataset[i][1] for i in range(len(dataset))]
    counts = Counter(labels)
    total = sum(counts.values())
    weights = [total / (num_classes * counts[i]) for i in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float)


def make_weighted_sampler(dataset, num_classes=10):
    labels = [dataset[i][1] for i in range(len(dataset))]
    class_counts = np.bincount(labels, minlength=num_classes)
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = [class_weights[label] for label in labels]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


# ================== Models ==================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class ViTWithMLP(nn.Module):
    def __init__(self, model_name="google/vit-base-patch16-224", num_classes=10, hidden=512):
        super().__init__()
        self.backbone = ViTModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, pixel_values):
        out = self.backbone(pixel_values=pixel_values)
        cls_feat = out.last_hidden_state[:, 0, :]
        return self.mlp(cls_feat)


# ================== Training / Eval ==================
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * y.size(0)
            preds = logits.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    acc = correct / total
    f1 = f1_score(y_true, y_pred, average="macro")
    return total_loss / total, acc, f1, (y_true, y_pred)


# ================== Main ==================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["cnn", "vit"], default="cnn")
    parser.add_argument("--mode", choices=["baseline", "weighted_ce", "oversample"], default="baseline")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--imbalance", action="store_true", help="whether or not to impose imbalance issue")
    args = parser.parse_args()

    # ---------- Transforms ----------
    if args.model == "vit":
        size = 224
        tf_train = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])
        tf_val = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])
    else:  # CNN
        tf_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])
        tf_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

    # ---------- Dataset ----------
    train_ds = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=tf_train)
    val_ds = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=tf_val)
    if args.imbalance:
        train_ds = make_imbalanced_dataset(train_ds, 0.05, [2, 3, 5])

    # ---------- Sampler / Loader ----------
    if args.mode == "oversample":
        sampler = make_weighted_sampler(train_ds)
        train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler, num_workers=4)
    else:
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)

    # ---------- Model ----------
    model = SimpleCNN().to(DEVICE) if args.model == "cnn" else ViTWithMLP().to(DEVICE)

    # ---------- Loss ----------
    if args.mode == "weighted_ce":
        weights = get_class_weights(train_ds).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ---------- Training ----------
    train_losses, val_losses, train_accs, val_accs, val_f1s = [], [], [], [], []
    best_val_acc = 0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc, val_f1, (y_true, y_pred) = evaluate(model, val_loader, criterion)
        scheduler.step()

        train_losses.append(tr_loss); val_losses.append(val_loss)
        train_accs.append(tr_acc); val_accs.append(val_acc); val_f1s.append(val_f1)

        print(f"[{epoch:02d}/{args.epochs}] "
              f"TrainAcc={tr_acc:.4f}, ValAcc={val_acc:.4f}, F1={val_f1:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints/best_{args.model}_{args.mode}.pt")

    print(f"Best Validation Accuracy: {best_val_acc:.4f}")

    # ---------- Curves ----------
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss"); plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.title("Loss Curve")
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="Train Acc"); plt.plot(val_accs, label="Val Acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.title("Accuracy Curve")
    plt.tight_layout()
    plt.savefig(f"curve_{args.model}_{args.mode}.png")
    plt.show()

    # ---------- Confusion Matrix ----------
    # cm = confusion_matrix(y_true, y_pred)
    # plt.figure(figsize=(6, 5))
    # sns.heatmap(cm, cmap="Blues", cbar=False)
    # plt.title(f"Confusion Matrix ({args.model}-{args.mode})")
    # plt.xlabel("Predicted"); plt.ylabel("True")
    # plt.savefig(f"confmat_{args.model}_{args.mode}.png")
    # plt.show()
    # ---------- Confusion Matrix ----------
    cm = confusion_matrix(y_true, y_pred)
    np.set_printoptions(linewidth=150)
    print("\nConfusion Matrix (raw counts):")
    print(cm)

    np.savetxt(f"confmat_{args.model}_{args.mode}.csv", cm, delimiter=",", fmt="%d")

    print(f"Final F1 Score: {val_f1s[-1]:.4f}")


if __name__ == "__main__":
    main()
