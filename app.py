"""Web demo phân loại ảnh tài liệu Hán Nôm — bộ model tốt nhất (24/09/2026) + xác định chiều ảnh.
Chạy:  streamlit run app.py
"""
import numpy as np
import streamlit as st
from PIL import Image

from model_defs import (load_models, classify, classify_with_fix, fix_orientation, normalize_pil_image,
                        CKPTS, DEVICE, TEXT_THRESHOLD, ORIENT_MIN_CONF, W_FLAT_TIER2,
                        L1_NAMES, L2_NAMES, L3_NAMES, ORIENT_CLASSES, FLAT8_CLASSES)

st.set_page_config(page_title="Hán Nôm Classifier", page_icon="🏮", layout="wide")
st.title("🏮 Phân loại ảnh tài liệu Hán Nôm")
st.caption("Tầng 1 nhánh chữ @768 → tầng 2 nhóm (DHC + flat) → tầng 3 dọc/ngang → chiều ảnh (0/90/180/270/mirror) → sửa ảnh & phân loại lại · "
           f"Inference **CPU** ({DEVICE})")


@st.cache_resource(show_spinner="Đang load 4 model (lần đầu ~10 s)...")
def get_models():
    return load_models()


models = get_models()

with st.sidebar:
    st.header("Model")
    labels = {"text": "Nhánh chữ Hán Nôm (tầng 1)", "flat": "EfficientNet-B4 flat 6 lớp", "dhc": "EfficientNet-B4 hierarchical (DHC rotswap)", "orient": "PP-LCNet chiều ảnh 5 lớp"}
    for key, p in CKPTS.items():
        st.markdown(f"{'✅' if key in models else '❌'} **{labels[key]}**"); st.caption(p.name)
    st.divider(); st.header("Tuỳ chọn")
    from model_defs import TILE_MODES
    use_tiles = st.radio("Tầng 1: số khung đưa vào nhánh chữ", options=list(TILE_MODES), index=0, format_func=lambda k: TILE_MODES[k],
                         help="Bậc thang: ~90% ảnh chỉ chạy toàn khung (≈0,55 s); ảnh chưa đạt ngưỡng mới chấm thêm 2 tile (≈+1 s). Test: T=0,5 sót 4, nhầm 34.")
    threshold = st.slider("Ngưỡng tầng 1 (điểm nhánh chữ)", 0.10, 0.90, TEXT_THRESHOLD, 0.05)
    w_flat = st.slider("Trọng số flat ở tầng 2 (DHC = 1 − w)", 0.0, 1.0, W_FLAT_TIER2, 0.1)
    auto_fix = st.toggle("Tự sửa chiều ảnh rồi phân loại lại", value=True)
    min_conf = st.slider("Độ tin chiều tối thiểu để sửa", 0.5, 0.99, ORIENT_MIN_CONF, 0.01)
    st.divider()
    st.caption("Đo trên test_full 1278 ảnh (23/09/2026, tile, T=0,50): tầng 1 96,2 %, bỏ sót 4/1162, nhận nhầm 45/116; "
               "tầng 2–3 (tầng 1 đúng) 96,0 %; 6 lớp 92,5 %. Chiều ảnh: val 99,5 %, test 97,7 %. "
               "CPU M4 Pro: ≈2,3 s/ảnh có tile, ≈0,9 s không tile, chiều ảnh +0,05 s.")

missing = [k for k in ("text", "flat", "dhc") if k not in models]
if missing:
    st.error(f"Thiếu checkpoint: {missing} — kiểm tra thư mục models/"); st.stop()

uploaded = st.file_uploader("Tải ảnh tài liệu lên (jpg/png/webp...)", type=["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"])
if uploaded is None:
    st.info("⬆️ Tải một ảnh lên để phân loại."); st.stop()

image = Image.open(uploaded)
with st.spinner("Đang phân loại..."):
    if auto_fix:
        r1, r2, fixed = classify_with_fix(models, image, use_tiles, threshold, w_flat, min_conf)
    else:
        r1, r2, fixed = classify(models, image, use_tiles, threshold, w_flat), None, None
final = r2 if r2 is not None else r1


def show_result(r, title):
    with st.container(border=True):
        h1, h2, h3 = st.columns([3, 1, 2])
        with h1: st.subheader(title)
        with h2: st.metric("⏱ tổng (s)", f"{r['times']['tổng']:.2f}")
        with h3: (st.success if r["is_sino"] else st.warning)(f"**{r['final']}**")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"**Tầng 1:** {'SinoNom' if r['is_sino'] else 'NonSinoNom'} — điểm chữ {r['p_text']:.2f} (ngưỡng {threshold:.2f})")
            st.progress(min(1.0, r["p_text"]))
            with st.expander("điểm từng khung"):
                names = ["toàn khung", "tile vùng cao 1", "tile vùng cao 2"] if use_tiles == "cascade" else (["toàn khung", "tile điểm cao nhất" if use_tiles == "1" else "tile giữa"]) if len(r["per_view"]) == 2 else ["toàn khung", "tile trên-trái", "tile trên-phải", "tile dưới-trái", "tile dưới-phải"]
                for n, v in zip(names, r["per_view"]): st.caption(f"{n}: {v:.2f}")
        with c2:
            if "s2" not in r: st.markdown("**Tầng 2:** —")
            else:
                st.markdown(f"**Tầng 2:** {r['doc_type']} — {100*r['doc_conf']:.1f}%"); st.progress(min(1.0, r["doc_conf"]))
                with st.expander("phân bố (gộp · DHC · flat)"):
                    for i, n in enumerate(L2_NAMES): st.caption(f"{n}: {100*r['s2'][i]:.1f}% · {100*r['dhc'][1][i]:.1f}% · {100*r['flat'][1][i]:.1f}%")
                    if "scene_sublabel" in r: st.caption(f"nhãn con scene (flat): {r['scene_sublabel']}")
        with c3:
            if "direction" not in r: st.markdown("**Tầng 3:** —")
            else:
                st.markdown(f"**Tầng 3:** {r['direction']} — {100*r['dir_conf']:.1f}%"); st.progress(min(1.0, r["dir_conf"]))
                with st.expander("P(dọc) từng model"): st.caption(f"DHC {r['dhc'][2][0]:.2f} · flat {r['flat'][2][0]:.2f} · quy tắc OR-dọc")
        with c4:
            if "orientation" not in r: st.markdown("**Chiều ảnh:** — (chỉ chạy cho general)")
            else:
                ok = r["orientation"] == "0"
                st.markdown(f"**Chiều ảnh:** {'thẳng' if ok else r['orientation']} — {100*r['orient_conf']:.1f}%"); st.progress(min(1.0, r["orient_conf"]))
                with st.expander("phân bố"):
                    for n, v in zip(ORIENT_CLASSES, r["orient_probs"]): st.caption(f"{n}: {100*v:.1f}%")
        with st.expander("thời gian từng bước"):
            for k, v in r["times"].items(): st.caption(f"{k}: {v*1000:.0f} ms")


col_img, col_res = st.columns([1, 2], gap="large")
with col_img:
    st.image(image, caption=f"{uploaded.name} · {image.size[0]}×{image.size[1]}px", use_container_width=True)
    if r1.get("tile_box") is not None:
        from PIL import ImageDraw as _D
        _im = normalize_pil_image(image).copy(); _d = _D.Draw(_im)
        for _k, _b in enumerate(r1["tile_box"]): _d.rectangle(_b, outline=[(0, 160, 255), (255, 140, 0)][_k % 2], width=max(3, _im.width // 150))
        st.image(_im, caption="Tile đã chấm (xanh: vùng cao nhất, cam: vùng cao thứ hai)", use_container_width=True)
    if fixed is not None:
        st.image(fixed, caption=f"Ảnh đã sửa chiều ({r1['orientation']} → thẳng)", use_container_width=True)
    if r1.get("grid") is not None and r1["is_sino"]:
        with st.expander("heatmap nhánh chữ (toàn khung)"):
            g = r1["grid"]; g8 = (np.clip(g, 0, 1) * 255).astype(np.uint8)
            st.image(Image.fromarray(g8).resize((240, 240), Image.NEAREST), caption="24×24 ô · sáng = có chữ Hán Nôm", width=240)

with col_res:
    if r2 is not None:
        st.info(f"Ảnh bị **{r1['orientation']}** (độ tin {r1['orient_conf']:.2f}) → đã xoay/lật về thẳng và phân loại lại. Kết quả cuối lấy theo lần 2.")
        show_result(r2, "⭐ Kết quả sau khi sửa chiều (lần 2)")
        show_result(r1, "Lần 1 (ảnh gốc)")
    else:
        show_result(r1, "⭐ Kết quả")
        if r1.get("orientation", "0") != "0":
            st.warning(f"Chiều ảnh dự đoán {r1['orientation']} nhưng độ tin {r1['orient_conf']:.2f} < {min_conf:.2f} nên không tự sửa.")

st.divider()
st.caption("Tầng 2 chỉ chạy khi tầng 1 = SinoNom; tầng 3 và chiều ảnh chỉ chạy khi tầng 2 = general. Model chiều 5 lớp không hỗ trợ ảnh vừa xoay vừa lật.")
