import torch
from PIL import Image
from transformers import ViTImageProcessor, ViTModel
import torch.nn as nn
import os

# ——— 配置区 ———
MODEL_DIR = "checkpoints"             # 存放 preprocessor_config.json 和 .pt 的目录
WEIGHT_FILE = "vit_mlp_catdog.pt"     # 你的模型权重文件名
IMAGE_PATH = "/usr1/home/s125mdg53_03/ee6483/data/datasets/test/491.jpg"   # 要测试的图片
NUM_CLASSES = 2
MLP_HIDDEN = 512

# ——— 模型定义（和训练时对应） ———
class ViTWithMLP(nn.Module):
    def __init__(self, base_model_name, num_classes=2, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.backbone = ViTModel.from_pretrained(base_model_name)
        hidden_size = self.backbone.config.hidden_size
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, pixel_values):
        out = self.backbone(pixel_values=pixel_values)
        # 取 CLS token 特征
        cls_feat = out.last_hidden_state[:, 0, :]
        logits = self.mlp(cls_feat)
        return logits

def resolve_model_path(model_dir, weight_file):
    wp = os.path.join(model_dir, weight_file)
    if os.path.isfile(wp):
        return wp
    else:
        raise FileNotFoundError(f"Model weight file not found: {wp}")

def main():
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载 image processor（预处理器）
    processor = ViTImageProcessor.from_pretrained(MODEL_DIR)

    # 构建模型，加载权重
    model = ViTWithMLP(base_model_name="google/vit-base-patch16-224", num_classes=NUM_CLASSES, hidden_dim=MLP_HIDDEN)
    weight_path = resolve_model_path(MODEL_DIR, WEIGHT_FILE)
    state = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # 读图 & 预处理
    img = Image.open(IMAGE_PATH).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")
    pixel = inputs["pixel_values"].to(device)

    # 推理
    with torch.no_grad():
        logits = model(pixel)
        pred = logits.argmax(dim=1).item()

    # 输出结果
    print("cat" if pred == 0 else "dog")

if __name__ == "__main__":
    main()