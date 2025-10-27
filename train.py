import os
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import ViTImageProcessor, ViTModel, ViTConfig

# ========== 配置 ==========
MODEL_NAME = "google/vit-base-patch16-224"
DATA_DIR = "/home/ian/ee6483/data"  # 假设 data/train/cats, data/train/dogs, data/val/...
BATCH_SIZE = 16
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== 数据集 ==========
class CatDogDataset(Dataset):
    def __init__(self, root_dir, processor):
        """
        root_dir 应该有子目录 cats/ 和 dogs/
        """
        self.samples = []
        self.processor = processor
        cls2label = {"cats": 0, "dogs": 1}
        for cls_name, lbl in cls2label.items():
            cls_dir = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(cls_dir, fname), lbl))
        if len(self.samples) == 0:
            raise RuntimeError(f"No images found under {root_dir}")
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        path, lbl = self.samples[idx]
        img = Image.open(path).convert("RGB")
        enc = self.processor(images=img, return_tensors="pt")
        # enc["pixel_values"] 的形状是 [1, 3, H, W]
        pixel_values = enc["pixel_values"].squeeze(0)  # drop batch dim
        return pixel_values, lbl

def collate_fn(batch):
    xs, ys = zip(*batch)
    xs = torch.stack(xs, dim=0)
    ys = torch.tensor(ys, dtype=torch.long)
    return {"pixel_values": xs, "labels": ys}

# ========== 模型 ==========
class ViTWithMLP(nn.Module):
    def __init__(self, model_name, num_classes=2, mlp_hidden=512, dropout=0.1):
        super().__init__()
        # 用 ViTModel（不含分类头）作为 backbone
        self.backbone = ViTModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size  # 通常 768
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, num_classes)
        )
    def forward(self, pixel_values):
        out = self.backbone(pixel_values=pixel_values)
        # last_hidden_state 的形状 [B, seq_len, hidden_size]
        # 通常取 CLS token 的表示（第 0 个 token）
        cls_feat = out.last_hidden_state[:, 0, :]
        logits = self.mlp(cls_feat)
        return logits

# ========== 训练 & 验证 ==========
def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch in dataloader:
        pixel = batch["pixel_values"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
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
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            pixel = batch["pixel_values"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            logits = model(pixel)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total

def main():
    # 预处理器（负责 resize / normalize / 转 tensor）
    processor = ViTImageProcessor.from_pretrained(MODEL_NAME)

    train_ds = CatDogDataset(os.path.join(DATA_DIR, "train"), processor)
    val_ds   = CatDogDataset(os.path.join(DATA_DIR, "val"),   processor)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=4)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn, num_workers=4)

    model = ViTWithMLP(MODEL_NAME, num_classes=2).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

    # 保存模型权重 & 预处理器配置
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/vit_mlp_catdog.pt")
    processor.save_pretrained("checkpoints")

if __name__ == "__main__":
    main()