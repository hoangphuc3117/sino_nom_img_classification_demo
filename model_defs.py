"""Định nghĩa model + pipeline cho web demo (bản 24/09/2026, dùng bộ models_best):
  Tầng 1  nhánh chữ Hán Nôm (EfficientNet-B4, lưới 24×24 @768, toàn khung + 4 tile) — điểm > T (0.50) → có Hán Nôm.
  Tầng 2  4 nhóm = trung bình 0.5/0.5 của DHC (đầu tầng 2) và flat (8 nhãn con gộp về 6 lớp, khử label smoothing) — y logic service.
  Tầng 3  dọc/ngang chỉ khi nhóm = general: một model nói dọc → lấy vector model đó (OR-dọc).
  Chiều   PP-LCNet 5 lớp (0/90/180/270/mirror) chỉ cho general; nếu lệch và tin ≥ 0.8 → sửa ảnh, phân loại lại (luồng đã áp dụng ở hn_classification).
Tách khỏi app.py để test được không cần Streamlit."""
import os, time
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
from pathlib import Path
import numpy as np, torch, torch.nn as nn
import torchvision.transforms as T
from torchvision.models import efficientnet_b4
from PIL import Image, ImageOps
from pplcnet_torch import PPLCNetDocOrientation

MODELS_DIR = Path(__file__).resolve().parent / "models"
CKPTS = {
    "text":   MODELS_DIR / "text_branch_r6.pth",       # nhánh chữ vòng 6
    "flat":   MODELS_DIR / "flat_b4_scratch_ep9.pth",  # flat 8 nhãn con, train từ đầu, EMA epoch 9
    "dhc":    MODELS_DIR / "dhc_b4_scratch.pth",       # hierarchical 2/4/2, train từ đầu
    "orient": MODELS_DIR / "orient5_pplcnet.pth",      # chiều ảnh 5 lớp
}
TEXT_THRESHOLD = 0.50      # ngưỡng tầng 1 (đo với tile trên test_full: tầng 1 96.2%, sót 4/1162, nhận nhầm 45/116)
ORIENT_MIN_CONF = 0.80     # chỉ sửa ảnh khi độ tin chiều ≥ ngưỡng
W_FLAT_TIER2 = 0.5         # trọng số flat ở tầng 2 (0.5 = giống service đang chạy)

DEVICE = torch.device("cpu")
torch.set_num_threads(max(4, (os.cpu_count() or 4) // 2))

FLAT8_CLASSES = ["non_sino_nom", "admin", "epitaph", "scene_img", "scene_obj", "scene_txt", "vertical", "horizontal"]  # thứ tự lúc train
SERVICE6 = ["non_sino_nom", "admin", "epitaph", "scene", "horizontal", "vertical"]
MAP8 = [SERVICE6.index("scene" if c.startswith("scene") else c) for c in FLAT8_CLASSES]
L1_NAMES = ["SinoNom", "NonSinoNom"]
L2_NAMES = ["general", "admin", "scene", "epitaph"]
L3_NAMES = ["vertical (dọc)", "horizontal (ngang)"]
ORIENT_CLASSES = ["0", "90", "180", "270", "mirror"]

# ==================== KIẾN TRÚC ====================
class TextBranch(nn.Module):
    def __init__(self):
        super().__init__(); b = efficientnet_b4(weights=None); self.features = b.features
        self.score = nn.Sequential(nn.Conv2d(1792, 256, 1), nn.ReLU(inplace=True), nn.Dropout2d(0.1), nn.Conv2d(256, 1, 1))
    def forward(self, x): return self.score(self.features(x)).squeeze(1)

class HierarchicalEfficientNetB4(nn.Module):
    def __init__(self, num_classes=(2, 4, 2)):
        super().__init__(); base = efficientnet_b4(weights=None); self.features = base.features; self.avgpool = nn.AdaptiveAvgPool2d((1, 1)); d = base.classifier[1].in_features
        self.h1_layer = nn.Sequential(nn.Linear(d, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Dropout(0.5))
        self.h2_layer = nn.Sequential(nn.Linear(d + 512, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.4))
        self.h3_layer = nn.Sequential(nn.Linear(d + 512 + 256, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Dropout(0.3))
        self.classifier1 = nn.Linear(512, num_classes[0]); self.classifier2 = nn.Linear(256, num_classes[1]); self.classifier3 = nn.Linear(128, num_classes[2])
    def forward(self, x):
        f = torch.flatten(self.avgpool(self.features(x)), 1); h1 = self.h1_layer(f); h2 = self.h2_layer(torch.cat([f, h1], 1)); h3 = self.h3_layer(torch.cat([f, h1, h2], 1))
        return [self.classifier1(h1), self.classifier2(h2), self.classifier3(h3)]

def build_flat8():
    m = efficientnet_b4(weights=None); m.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(1792, 8)); return m

# ==================== TIỀN XỬ LÝ ====================
IMAGENET_NORM = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
FLAT_TFM = T.Compose([T.Resize((380, 380), antialias=True), T.ToTensor(), T.Normalize((0.5,) * 3, (0.5,) * 3)])
TEXT_TFM = T.Compose([T.Resize((768, 768), antialias=True), T.ToTensor(), T.Normalize((0.5,) * 3, (0.5,) * 3)])
ORIENT_TFM = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def normalize_pil_image(image):
    image = ImageOps.exif_transpose(image)
    if image.mode == "P" and "transparency" in image.info: image = image.convert("RGBA")
    if image.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", image.size, (255, 255, 255)); bg.paste(image, mask=image.split()[-1]); return bg
    return image.convert("RGB")

def cap_long_side(image, max_side=512):
    w, h = image.size; m = max(w, h)
    if m <= max_side: return image
    s = max_side / m; return image.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)

def letterbox_resize(image, size, fill=(255, 255, 255)):
    w, h = image.size; side = max(w, h); c = Image.new("RGB", (side, side), fill); c.paste(image, ((side - w) // 2, (side - h) // 2)); return c.resize(size, Image.BILINEAR)

TILE_MODES = {"0": "chỉ toàn khung", "1": "toàn khung + 1 tile giữa (56 % mỗi chiều)", "4": "toàn khung + 4 tile 2×2 (chồng 12 %)"}
def text_views(image, tiles="1"):
    """Các khung đưa vào nhánh chữ. tiles: "0" = chỉ toàn khung; "1" = thêm 1 tile giữa cỡ 56 %×56 % (cùng cỡ với tile góc);
    "4" = thêm 4 tile 2×2 chồng 12 % (cấu hình đo ngưỡng 0.50 ban đầu). True/False cũ ánh xạ sang "4"/"0"."""
    if tiles is True: tiles = "4"
    if tiles is False or tiles in ("0", 0): return [image]
    w, h = image.size; ov = 0.12; tw, th = int(w * (0.5 + ov / 2)), int(h * (0.5 + ov / 2)); vs = [image]
    if str(tiles) == "1":
        x0, y0 = (w - tw) // 2, (h - th) // 2; vs.append(image.crop((x0, y0, x0 + tw, y0 + th))); return vs
    for x0 in (0, w - tw):
        for y0 in (0, h - th): vs.append(image.crop((x0, y0, x0 + tw, y0 + th)))
    return vs

def fix_orientation(image, orientation):
    """Đưa ảnh về thẳng. 90 = ảnh đã bị xoay 90° thuận chiều kim đồng hồ → xoay ngược lại; mirror = lật ngang lại."""
    return {"0": image, "90": image.transpose(Image.ROTATE_90), "180": image.transpose(Image.ROTATE_180),
            "270": image.transpose(Image.ROTATE_270), "mirror": ImageOps.mirror(image)}[orientation]

# ==================== LOAD ====================
def load_models():
    """Trả về dict {'text','flat','dhc','orient'} → model đã eval trên CPU. Thiếu file nào thì bỏ qua (báo trong app)."""
    models = {}
    p = CKPTS["text"]
    if p.exists():
        m = TextBranch(); sd = torch.load(p, map_location="cpu", weights_only=False); sd = sd.get("state_dict", sd)
        m.load_state_dict({k: v for k, v in sd.items() if k.startswith(("features.", "score."))}); models["text"] = m.eval().to(DEVICE)
    p = CKPTS["flat"]
    if p.exists():
        m = build_flat8(); sd = torch.load(p, map_location="cpu", weights_only=False); m.load_state_dict(sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd); models["flat"] = m.eval().to(DEVICE)
    p = CKPTS["dhc"]
    if p.exists():
        m = HierarchicalEfficientNetB4(); ck = torch.load(p, map_location="cpu", weights_only=False); m.load_state_dict(ck["model_state_dict"] if "model_state_dict" in ck else ck); models["dhc"] = m.eval().to(DEVICE)
    p = CKPTS["orient"]
    if p.exists():
        ck = torch.load(p, map_location="cpu", weights_only=False); m = PPLCNetDocOrientation(3, ck.get("scale", 1.0), 5); m.load_state_dict(ck["state_dict"]); models["orient"] = m.eval().to(DEVICE)
    return models

# ==================== DỰ ĐOÁN ====================
@torch.no_grad()
def text_score(model, image_rgb, tiles="1"):
    x = torch.stack([TEXT_TFM(v) for v in text_views(image_rgb, tiles)]).to(DEVICE)
    s = torch.sigmoid(model(x))                      # [views, 24, 24]
    per_view = s.flatten(1).max(1).values.cpu().numpy()
    return float(per_view.max()), per_view, s[0].cpu().numpy()   # điểm ảnh, điểm từng khung, lưới toàn khung (để vẽ heatmap)

def _flat_to_hier(p6):
    p6 = np.clip(p6 - 0.05 / 6, 0.0, None); p6 = p6 / max(1e-9, p6.sum())     # khử label smoothing như service
    s1 = np.array([1.0 - p6[0], p6[0]]); s2 = np.array([p6[4] + p6[5], p6[1], p6[3], p6[2]]); s2 = s2 / max(1e-9, s2.sum())
    s3 = np.array([p6[5], p6[4]]); s3 = s3 / max(1e-9, s3.sum()); return s1, s2, s3

@torch.no_grad()
def predict_flat(model, image_rgb):
    p8 = torch.softmax(model(FLAT_TFM(image_rgb).unsqueeze(0).to(DEVICE)), 1)[0].cpu().numpy(); p6 = np.zeros(6)
    for k, j in enumerate(MAP8): p6[j] += float(p8[k])
    return _flat_to_hier(p6), p8

@torch.no_grad()
def predict_dhc(model, image_rgb):
    t = IMAGENET_NORM(letterbox_resize(cap_long_side(image_rgb), (380, 380))).unsqueeze(0).to(DEVICE)
    return tuple(torch.softmax(p, 1)[0].cpu().numpy() for p in model(t))

@torch.no_grad()
def predict_orientation(model, image_rgb):
    p = torch.softmax(model(ORIENT_TFM(image_rgb).unsqueeze(0).to(DEVICE)), 1)[0].cpu().numpy(); i = int(p.argmax())
    return ORIENT_CLASSES[i], float(p[i]), p

def ensemble_23(hier_probs, flat_probs, w_flat=W_FLAT_TIER2):
    """Tầng 2 trung bình có trọng số; tầng 3 OR-dọc — y logic service đang chạy."""
    s2 = w_flat * flat_probs[1] + (1 - w_flat) * hier_probs[1]
    h3, f3 = hier_probs[2], flat_probs[2]; hv = int(h3.argmax()) == 0; fv = int(f3.argmax()) == 0
    s3 = np.mean([h3, f3], 0) if (hv and fv) else h3 if hv else f3 if fv else np.mean([h3, f3], 0)
    return s2, s3

def classify(models, image, tiles="1", threshold=TEXT_THRESHOLD, w_flat=W_FLAT_TIER2, reuse_tier1=None):
    """Một lượt phân loại đủ 3 tầng + chiều ảnh. Trả dict có xác suất từng model, kết quả gộp, thời gian từng bước.
    reuse_tier1: kết quả lượt trước để dùng lại tầng 1 (có/không có Hán Nôm không đổi khi xoay/lật ảnh) → lượt 2 chỉ chạy flat + DHC + chiều."""
    img = normalize_pil_image(image); r = {"size": img.size, "times": {}}
    if reuse_tier1 is not None:
        r.update(p_text=reuse_tier1["p_text"], per_view=reuse_tier1["per_view"], grid=reuse_tier1["grid"], is_sino=reuse_tier1["is_sino"], tier1_reused=True); r["times"]["tầng 1 (dùng lại lượt 1)"] = 0.0
    else:
        t0 = time.perf_counter(); p_text, per_view, grid = text_score(models["text"], img, tiles); r["times"]["tầng 1 (nhánh chữ)"] = time.perf_counter() - t0
        r.update(p_text=p_text, per_view=per_view, grid=grid, is_sino=p_text > threshold)
    if not r["is_sino"]:
        r["final"] = "NonSinoNom"; r["times"]["tổng"] = sum(r["times"].values()); return r
    t0 = time.perf_counter(); flat_h, p8 = predict_flat(models["flat"], img); r["times"]["flat B4 @380"] = time.perf_counter() - t0
    t0 = time.perf_counter(); dhc_h = predict_dhc(models["dhc"], img); r["times"]["DHC B4 @380"] = time.perf_counter() - t0
    s2, s3 = ensemble_23(dhc_h, flat_h, w_flat); i2 = int(s2.argmax()); i3 = int(s3.argmax())
    r.update(flat=flat_h, flat_p8=p8, dhc=dhc_h, s2=s2, s3=s3, doc_type=L2_NAMES[i2], doc_conf=float(s2[i2]))
    r["final"] = L2_NAMES[i2]
    if i2 == 2:  # scene → nhãn con theo flat
        r["scene_sublabel"] = FLAT8_CLASSES[3 + int(np.argmax(p8[3:6]))]; r["final"] = f"scene / {r['scene_sublabel']}"
    if i2 == 0:  # general → hướng chữ + chiều ảnh
        r["direction"] = L3_NAMES[i3]; r["dir_conf"] = float(s3[i3]); r["final"] = f"general / {L3_NAMES[i3]}"
        if "orient" in models:
            t0 = time.perf_counter(); ori, conf, pvec = predict_orientation(models["orient"], img); r["times"]["chiều ảnh (PP-LCNet)"] = time.perf_counter() - t0
            r.update(orientation=ori, orient_conf=conf, orient_probs=pvec)
    r["times"]["tổng"] = sum(r["times"].values()); return r

def classify_with_fix(models, image, tiles="1", threshold=TEXT_THRESHOLD, w_flat=W_FLAT_TIER2, min_conf=ORIENT_MIN_CONF):
    """Luồng đầy đủ: phân loại → nếu general và chiều ≠ 0 với độ tin ≥ min_conf → sửa ảnh → phân loại lại → dùng kết quả lần 2.
    Trả (kết quả lần 1, kết quả lần 2 hoặc None, ảnh đã sửa hoặc None)."""
    r1 = classify(models, image, tiles, threshold, w_flat)
    if r1.get("orientation", "0") != "0" and r1.get("orient_conf", 0.0) >= min_conf:
        fixed = fix_orientation(normalize_pil_image(image), r1["orientation"]); r2 = classify(models, fixed, tiles, threshold, w_flat, reuse_tier1=r1)
        return r1, r2, fixed
    return r1, None, None
