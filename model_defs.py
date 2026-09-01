"""Định nghĩa model EfficientNet-B4 hierarchical + tiền xử lý + hàm dự đoán cho web demo.
Tiền xử lý đồng bộ y nguyên với han-nom-classification_cp/src/services/sino_classification_service.py
(bản rút gọn: chỉ chạy 1 model B4 hierarchical, không ensemble).
Tách khỏi app.py để test được không cần Streamlit."""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
from pathlib import Path
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import efficientnet_b4
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent  # thư mục dự án (cha của web_demo/)

# Đường dẫn checkpoint (bộ triển khai slim)
MODELS_DIR = Path(__file__).resolve().parent / "models"   # web_demo/models/
B4_HIER = "EfficientNet-B4 hierarchical"

CKPTS = {
    B4_HIER: MODELS_DIR / "best_hierarchical_b4.pth",
}

KAGGLE_MODEL_IDS = {
    B4_HIER: "phuchoangnguyen/sinonomimg-eb4-hier/pyTorch/default",
}

def _local_checkpoint(name):
    return CKPTS[name]


@lru_cache(maxsize=None)
def _download_kaggle_model(model_name):
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError(
            "Thiếu package kagglehub. Hãy cài kagglehub để tải model từ Kaggle."
        ) from exc

    return Path(kagglehub.model_download(KAGGLE_MODEL_IDS[model_name]))


def _find_checkpoint(root, preferred_name):
    preferred = root / preferred_name
    if preferred.exists():
        return preferred

    matches = [p for p in root.rglob(preferred_name) if p.is_file()]
    if matches:
        return matches[0]

    stem_tokens = [token for token in Path(preferred_name).stem.lower().split("_") if len(token) > 2]
    if stem_tokens:
        token_matches = [
            p for p in root.rglob("*.pth") if all(token in p.name.lower() for token in stem_tokens)
        ]
        if token_matches:
            return token_matches[0]

    pth_files = [p for p in root.rglob("*.pth") if p.is_file()]
    if len(pth_files) == 1:
        return pth_files[0]
    if pth_files:
        return sorted(pth_files)[0]

    raise FileNotFoundError(f"Không tìm thấy checkpoint .pth trong {root}")


def _resolve_checkpoint(model_name):
    local = _local_checkpoint(model_name)
    if local.exists():
        return local
    kaggle_root = _download_kaggle_model(model_name)
    return _find_checkpoint(kaggle_root, local.name)

MAIN_CATEGORIES = {"SinoNom": 0, "NonSinoNom": 1}
DOC_TYPES = {"general": 0, "admin": 1, "scene": 2, "epitaph": 3}
TEXT_DIRECTIONS = {"vertical": 0, "horizontal": 1}
INV_MAIN = {v: k for k, v in MAIN_CATEGORIES.items()}
INV_DOC = {v: k for k, v in DOC_TYPES.items()}
INV_DIR = {v: k for k, v in TEXT_DIRECTIONS.items()}
L1_NAMES = ["SinoNom", "NonSinoNom"]
L2_NAMES = ["general", "admin", "scene", "epitaph"]
L3_NAMES = ["vertical (dọc)", "horizontal (ngang)"]

# Chạy inference trên CPU (theo yêu cầu — dễ so thời gian, không phụ thuộc GPU)
DEVICE = torch.device("cpu")
torch.set_num_threads(1)

IMAGENET_NORM = T.Compose([T.ToTensor(),
                           T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def cap_long_side(image, max_side=512):
    """Chuẩn hoá độ phân giải làm việc: thu nhỏ về cạnh dài tối đa max_side
    (giữ tỉ lệ, không phóng to ảnh nhỏ hơn). Phải khớp với pipeline train."""
    w, h = image.size
    m = max(w, h)
    if m <= max_side:
        return image
    scale = max_side / m
    return image.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BILINEAR)


def letterbox_resize(image, size, fill=(255, 255, 255)):
    w, h = image.size
    side = max(w, h)
    canvas = Image.new('RGB', (side, side), fill)
    canvas.paste(image, ((side - w) // 2, (side - h) // 2))
    return canvas.resize(size, Image.BILINEAR)


def to_rgb(image):
    if image.mode == 'P' and 'transparency' in image.info:
        image = image.convert('RGBA')
    if image.mode == 'RGBA':
        bg = Image.new('RGB', image.size, (255, 255, 255))
        bg.paste(image, mask=image.split()[-1])
        image = bg
    elif image.mode != 'RGB':
        image = image.convert('RGB')
    # Luôn chuẩn hoá độ phân giải làm việc trước khi vào model (23/08)
    return cap_long_side(image)


# ==================== KIẾN TRÚC ====================
class HierarchicalEfficientNetB4(nn.Module):
    """EfficientNet-B4 backbone + đầu phân loại phân cấp 3 tầng (2, 4, 2)."""

    def __init__(self, num_classes=(2, 4, 2)):
        super().__init__()
        base = efficientnet_b4(weights=None)
        self.features = base.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        d = base.classifier[1].in_features
        self.h1_layer = nn.Sequential(nn.Linear(d, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Dropout(0.5))
        self.h2_layer = nn.Sequential(nn.Linear(d + 512, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.4))
        self.h3_layer = nn.Sequential(nn.Linear(d + 512 + 256, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Dropout(0.3))
        self.classifier1 = nn.Linear(512, num_classes[0])
        self.classifier2 = nn.Linear(256, num_classes[1])
        self.classifier3 = nn.Linear(128, num_classes[2])

    def forward(self, x):
        f = torch.flatten(self.avgpool(self.features(x)), 1)
        h1 = self.h1_layer(f); o1 = self.classifier1(h1)
        h2 = self.h2_layer(torch.cat([f, h1], 1)); o2 = self.classifier2(h2)
        h3 = self.h3_layer(torch.cat([f, h1, h2], 1)); o3 = self.classifier3(h3)
        return [o1, o2, o3]


# ==================== LOAD ====================
def load_models():
    """Trả về dict {tên: (model, kind)} — kind: 'hier380'.
    Checkpoint thiếu sẽ bị bỏ qua."""
    models = {}
    p = _resolve_checkpoint(B4_HIER)
    if p.exists():
        m = HierarchicalEfficientNetB4()
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        m.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
        m.eval().to(DEVICE)
        models[B4_HIER] = (m, "hier380")
    return models


# ==================== DỰ ĐOÁN ====================
@torch.no_grad()
def predict_one(model, kind, image_rgb):
    """Trả về (s1[2], s2[4], s3[2]) — softmax 3 tầng.
    Tiền xử lý: letterbox 380 + ImageNet norm (khớp service)."""
    t = IMAGENET_NORM(letterbox_resize(image_rgb, (380, 380)))
    t = t.unsqueeze(0).to(DEVICE)
    out = model(t)
    return tuple(torch.softmax(p, 1)[0].cpu().numpy() for p in out)


def predict_all(models, image):
    """Chạy model B4 hierarchical (đo thời gian).
    Trả về dict {tên hệ: {"probs": (s1,s2,s3), "time": giây}}."""
    import time
    image_rgb = to_rgb(image)
    results = {}
    for name, (m, kind) in models.items():
        t0 = time.perf_counter()
        probs = predict_one(m, kind, image_rgb)
        results[name] = {"probs": probs, "time": time.perf_counter() - t0}
    return results


def decode(probs):
    """(s1,s2,s3) -> dict hiển thị theo đúng logic phân cấp."""
    s1, s2, s3 = probs
    i1, i2, i3 = int(s1.argmax()), int(s2.argmax()), int(s3.argmax())
    out = {"L1": (L1_NAMES[i1], float(s1[i1]), s1),
           "L2": None, "L3": None,
           "final": L1_NAMES[i1]}
    if i1 == 0:  # SinoNom
        out["L2"] = (L2_NAMES[i2], float(s2[i2]), s2)
        out["final"] = L2_NAMES[i2]
        if i2 == 0:  # general
            out["L3"] = (L3_NAMES[i3], float(s3[i3]), s3)
            out["final"] = f"general / {L3_NAMES[i3]}"
    return out
