"""Định nghĩa 3 model + tiền xử lý + hàm dự đoán cho web demo.
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
CKPTS = {
    "ResNet50+CBAM": MODELS_DIR / "best_hierarchical_model_slim.pth",
    "EfficientNet-B4 hierarchical": MODELS_DIR / "best_hierarchical_b4.pth",
    "EfficientNet-B4 flat": MODELS_DIR / "best_accuracy_b4_modified.pth",
}
KAGGLE_MODEL_IDS = {
    "ResNet50+CBAM": "phuchoangnguyen/sinonomimg-resnet50-hier/pyTorch/default",
    "EfficientNet-B4 hierarchical": "phuchoangnguyen/sinonomimg-eb4-hier/pyTorch/default",
    "EfficientNet-B4 flat": "phuchoangnguyen/sinonomimg-eb4-flat/pyTorch/default",
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
    try:
        kaggle_root = _download_kaggle_model(model_name)
        return _find_checkpoint(kaggle_root, local.name)
    except Exception:
        if local.exists():
            return local
        raise

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
torch.set_num_threads(max(1, os.cpu_count() - 2))

IMAGENET_NORM = T.Compose([T.ToTensor(),
                           T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
FLAT_TFM = T.Compose([T.Resize((380, 380), antialias=True), T.ToTensor(),
                      T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


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
class ChannelAttention(nn.Module):
    def __init__(self, channel_in, reduction_ratio=16, pool_types=('avg', 'max')):
        super().__init__()
        self.pool_types = pool_types
        self.shared_mlp = nn.Sequential(
            nn.Flatten(), nn.Linear(channel_in, channel_in // reduction_ratio),
            nn.ReLU(inplace=True), nn.Linear(channel_in // reduction_ratio, channel_in))

    def forward(self, x):
        atts = []
        for pt in self.pool_types:
            if pt == 'avg':
                p = nn.AvgPool2d((x.size(2), x.size(3)))(x)
            else:
                p = nn.MaxPool2d((x.size(2), x.size(3)))(x)
            atts.append(self.shared_mlp(p))
        s = torch.stack(atts, 0).sum(0)
        return x * torch.sigmoid(s).unsqueeze(2).unsqueeze(3).expand_as(x)


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.compress = ChannelPool()
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, 1, (kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(1, eps=1e-5, momentum=0.01, affine=True))

    def forward(self, x):
        return x * torch.sigmoid(self.spatial_attention(self.compress(x)))


class CBAM(nn.Module):
    def __init__(self, channel_in, reduction_ratio=16, spatial=True):
        super().__init__()
        self.spatial = spatial
        self.channel_attention = ChannelAttention(channel_in, reduction_ratio)
        if spatial:
            self.spatial_attention = SpatialAttention(7)

    def forward(self, x):
        x = self.channel_attention(x)
        if self.spatial:
            x = self.spatial_attention(x)
        return x


class BottleneckCBAM(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_cbam=True):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.cbam = CBAM(planes * 4) if use_cbam else None

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.cbam:
            out = self.cbam(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class HierarchicalResNet50(nn.Module):
    def __init__(self, num_classes=(2, 4, 2)):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        d = 2048
        self.h1_layer = nn.Sequential(nn.Linear(d, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Dropout(0.5))
        self.h2_layer = nn.Sequential(nn.Linear(d + 512, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.4))
        self.h3_layer = nn.Sequential(nn.Linear(d + 512 + 256, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Dropout(0.3))
        self.classifier1 = nn.Linear(512, num_classes[0])
        self.classifier2 = nn.Linear(256, num_classes[1])
        self.classifier3 = nn.Linear(128, num_classes[2])

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * 4, 1, stride, bias=False),
                nn.BatchNorm2d(planes * 4))
        layers = [BottleneckCBAM(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * 4
        for _ in range(1, blocks):
            layers.append(BottleneckCBAM(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        f = torch.flatten(self.avgpool(x), 1)
        h1 = self.h1_layer(f); o1 = self.classifier1(h1)
        h2 = self.h2_layer(torch.cat([f, h1], 1)); o2 = self.classifier2(h2)
        h3 = self.h3_layer(torch.cat([f, h1, h2], 1)); o3 = self.classifier3(h3)
        return [o1, o2, o3]


class HierarchicalEfficientNetB4(nn.Module):
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
    """Trả về dict {tên: (model, kind)} — kind: 'hier224' | 'hier380' | 'flat380'.
    Checkpoint thiếu sẽ bị bỏ qua."""
    models = {}
    p = _resolve_checkpoint("ResNet50+CBAM")
    if p.exists():
        m = HierarchicalResNet50()
        m.load_state_dict(torch.load(p, map_location="cpu", weights_only=False)["model_state_dict"])
        m.eval().to(DEVICE)
        models["ResNet50+CBAM"] = (m, "hier224")
    p = _resolve_checkpoint("EfficientNet-B4 hierarchical")
    if p.exists():
        m = HierarchicalEfficientNetB4()
        m.load_state_dict(torch.load(p, map_location="cpu", weights_only=False)["model_state_dict"])
        m.eval().to(DEVICE)
        models["EfficientNet-B4 hierarchical"] = (m, "hier380")
    p = _resolve_checkpoint("EfficientNet-B4 flat")
    if p.exists():
        sd = torch.load(p, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        m = efficientnet_b4(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 6)
        m.load_state_dict(sd)
        m.eval().to(DEVICE)
        models["EfficientNet-B4 flat"] = (m, "flat380")
    return models


# ==================== DỰ ĐOÁN ====================
def _flat_to_hier(p6):
    """[non_sino, admin, epitaph, scene, horizontal, vertical] -> (s1, s2, s3)"""
    s1 = np.array([1.0 - p6[0], p6[0]])
    s2 = np.array([p6[4] + p6[5], p6[1], p6[3], p6[2]])
    s2 = s2 / max(1e-9, s2.sum())
    s3 = np.array([p6[5], p6[4]])
    s3 = s3 / max(1e-9, s3.sum())
    return s1, s2, s3


@torch.no_grad()
def predict_one(model, kind, image_rgb):
    """Trả về (s1[2], s2[4], s3[2]) — softmax 3 tầng."""
    if kind == "hier224":
        t = IMAGENET_NORM(letterbox_resize(image_rgb, (224, 224)))
    elif kind == "hier380":
        t = IMAGENET_NORM(letterbox_resize(image_rgb, (380, 380)))
    else:
        t = FLAT_TFM(image_rgb)
    t = t.unsqueeze(0).to(DEVICE)
    out = model(t)
    if kind == "flat380":
        p6 = torch.softmax(out, 1)[0].cpu().numpy()
        return _flat_to_hier(p6)
    return tuple(torch.softmax(p, 1)[0].cpu().numpy() for p in out)


def predict_all(models, image):
    """Chạy mọi model (đo thời gian) + các ensemble.
    Trả về dict {tên hệ: {"probs": (s1,s2,s3), "time": giây}}."""
    import time
    image_rgb = to_rgb(image)
    results = {}
    for name, (m, kind) in models.items():
        t0 = time.perf_counter()
        probs = predict_one(m, kind, image_rgb)
        results[name] = {"probs": probs, "time": time.perf_counter() - t0}

    def add_ens(label, member_names):
        ms = [results[n] for n in member_names if n in results]
        if len(ms) != len(member_names):
            return
        probs = tuple(np.mean([r["probs"][i] for r in ms], axis=0) for i in range(3))
        results[label] = {"probs": probs, "time": sum(r["time"] for r in ms)}

    R, BH, BF = "ResNet50+CBAM", "EfficientNet-B4 hierarchical", "EfficientNet-B4 flat"
    add_ens("Ens: B4 hier + B4 flat", [BH, BF])
    add_ens("Ens: ResNet + B4 hier", [R, BH])
    add_ens("Ens: ResNet + B4 flat", [R, BF])
    add_ens("⭐ Ensemble 3 model", [R, BH, BF])
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
