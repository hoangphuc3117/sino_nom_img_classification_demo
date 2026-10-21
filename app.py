"""Web demo phân loại ảnh tài liệu Hán Nôm — bộ model tốt nhất (24/09/2026) + xác định chiều ảnh.
Chạy:  streamlit run app.py
"""
import numpy as np
import streamlit as st
from PIL import Image

from model_defs import (load_models, classify, classify_with_fix, fix_orientation, normalize_pil_image, heatmap_overlay,
                        CKPTS, DEVICE, TEXT_THRESHOLD, TILE_THRESHOLD, ORIENT_MIN_CONF, W_FLAT_TIER2,
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
    labels = {"text": "Nhánh chữ Hán Nôm (tầng 1)", "flat": "EfficientNet-B4 flat 5 lớp (xoay đổi nhãn)", "dhc": "EfficientNet-B4 DHC 2 tầng (Kaggle, xoay đổi nhãn)", "orient": "PP-LCNet chiều ảnh 5 lớp"}
    for key, p in CKPTS.items():
        st.markdown(f"{'✅' if key in models else '❌'} **{labels[key]}**"); st.caption(p.name)
    st.divider(); st.header("Tuỳ chọn")
    from model_defs import TILE_MODES
    use_tiles = st.radio("Tầng 1: số khung đưa vào nhánh chữ", options=list(TILE_MODES), index=0, format_func=lambda k: TILE_MODES[k],
                         help="Bậc thang: phần lớn ảnh có Hán chỉ chạy toàn khung; ảnh chưa đạt ngưỡng toàn khung mới chấm thêm 2 tile (ảnh lớn: tile ~1400 px ảnh gốc), tile phải vượt ngưỡng tile riêng.")
    threshold = st.slider("Ngưỡng tầng 1 — toàn khung", 0.10, 0.995, TEXT_THRESHOLD, 0.005,
                          help="r24 ep10: 0,70/0,70 (mặc định) = sót test 8/1160, nhầm 12/116, câu đối Việt 19/187; 0,60/0,60 = sót 6, nhầm 13; 0,825/0,975 = recall val 99 % (sót 16, nhầm 4/116, câu đối Việt 13/187); 0,975/0,975 = sót 20, nhầm 3.")
    tile_threshold = st.slider("Ngưỡng tầng 1 — tile", 0.10, 0.995, TILE_THRESHOLD, 0.005)
    w_flat = st.slider("Trọng số flat ở tầng 2 (DHC = 1 − w)", 0.0, 1.0, W_FLAT_TIER2, 0.1)
    auto_fix = st.toggle("Tự sửa chiều ảnh rồi phân loại lại", value=True)
    min_conf = st.slider("Độ tin chiều tối thiểu để sửa", 0.5, 0.99, ORIENT_MIN_CONF, 0.01)
    show_passes = st.toggle("Hiện chi tiết lần 1 / lần 2 (khi có sửa chiều ảnh)", value=False,
                            help="Tắt: chỉ hiện kết quả cuối cùng. Bật: khi ảnh bị xoay/lật và đã được sửa, hiện riêng kết quả lần 1 (ảnh gốc) và lần 2 (ảnh đã xoay thẳng). Chỉ đổi cách hiển thị, không chạy lại model.")
    st.divider()
    st.caption("Tầng 1 = nhánh chữ vòng 24 ep10 (22/10/2026: train thuần trên nhãn v2, không neo r7), ngưỡng mặc định 0,70 toàn khung / 0,70 tile. "
               "Test_full: sót 8/1160, nhầm 12/116 (r21 @0,60: sót 4, nhầm 21). Tập giữ riêng @0,70: câu đối Việt 19/187, hoa văn 14/165, non_chu_viet 4/51, âm web 8/422 "
               "(r21 @0,60: 83 / 45 / 27 / 41). Service hiện vẫn chạy r21, T=0,60. Chiều ảnh: val 99,5 %, test 97,7 %. Thời gian CPU ≈ như r21 (cùng kiến trúc, cùng số khung).")

missing = [k for k in ("text", "flat", "dhc") if k not in models]
if missing:
    st.error(f"Thiếu checkpoint: {missing} — kiểm tra thư mục models/"); st.stop()

uploaded = st.file_uploader("Tải ảnh tài liệu lên (jpg/png/webp...)", type=["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"])
if uploaded is None:
    st.info("⬆️ Tải một ảnh lên để phân loại."); st.stop()

image = Image.open(uploaded)
# Giữ kết quả theo (ảnh, tham số): kéo thanh độ đậm heatmap không phải chạy lại model
run_key = (uploaded.name, uploaded.size, hash(uploaded.getvalue()), use_tiles, threshold, tile_threshold, w_flat, auto_fix, min_conf)
if st.session_state.get("run_key") != run_key:
    with st.spinner("Đang phân loại..."):
        if auto_fix:
            st.session_state["run"] = classify_with_fix(models, image, use_tiles, threshold, w_flat, min_conf, tile_threshold=tile_threshold)
        else:
            st.session_state["run"] = (classify(models, image, use_tiles, threshold, w_flat, tile_threshold=tile_threshold), None, None)
    st.session_state["run_key"] = run_key
r1, r2, fixed = st.session_state["run"]
final = r2 if r2 is not None else r1


def show_result(r, title):
    with st.container(border=True):
        h1, h2, h3 = st.columns([3, 1, 2])
        with h1: st.subheader(title)
        with h2: st.metric("⏱ tổng (s)", f"{r['times']['tổng']:.2f}")
        with h3: (st.success if r["is_sino"] else st.warning)(f"**{r['final']}**")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"**Tầng 1:** {'SinoNom' if r['is_sino'] else 'NonSinoNom'} — toàn khung {r['per_view'][0]:.2f} (ngưỡng {threshold:.3f})"
                        + (f" · tile {max(r['per_view'][1:]):.2f} (ngưỡng {tile_threshold:.3f})" if len(r['per_view']) > 1 else ""))
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
                with st.expander("P(dọc) từng model"): st.caption(f"DHC {r['dhc'][2][0]:.2f}" + (f" (xoay 90°: {r['dhc_rot90'][2][0]:.2f} → nhất quán {0.5*(r['dhc'][2][0] + r['dhc_rot90'][2][1]):.2f})" if "dhc_rot90" in r else "") + f" · flat {r['flat'][2][0]:.2f} · quy tắc: DHC kiểm tra xoay 90°, > 0.95 thì lấy DHC, còn lại trung bình")
        with c4:
            if "orientation" not in r: st.markdown("**Chiều ảnh:** — (chỉ chạy cho general)")
            else:
                ok = r["orientation"] == "0"
                fixed_note = " → đã xoay thẳng" if (not ok and r["orient_conf"] >= min_conf and auto_fix) else ""
                st.markdown(f"**Chiều ảnh:** {'thẳng' if ok else r['orientation']}{fixed_note} — {100*r['orient_conf']:.1f}%"); st.progress(min(1.0, r["orient_conf"]))
                with st.expander("phân bố"):
                    for n, v in zip(ORIENT_CLASSES, r["orient_probs"]): st.caption(f"{n}: {100*v:.1f}%")
        with st.expander("thời gian từng bước"):
            for k, v in r["times"].items(): st.caption(f"{k}: {v*1000:.0f} ms")


def final_view(r1, r2):
    """Kết quả cuối để hiển thị gộp: tầng 2-3 lấy theo lần 2 (ảnh đã xoay thẳng), chiều ảnh lấy theo ảnh GỐC (lần 1),
    thời gian = tổng hai lần."""
    r = dict(r2)
    for k in ("orientation", "orient_conf", "orient_probs"):
        if k in r1: r[k] = r1[k]
        else: r.pop(k, None)
    r["times"] = {**{f"lần 1 · {k}": v for k, v in r1["times"].items() if k != "tổng"},
                  **{f"lần 2 · {k}": v for k, v in r2["times"].items() if k != "tổng"}}
    r["times"]["tổng"] = r1["times"]["tổng"] + r2["times"]["tổng"]
    return r


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
    if r1.get("grid") is not None:
        g = r1["grid"]
        heat_alpha = st.slider("Độ đậm heatmap", 0.0, 1.0, 0.55, 0.05, key="heat_alpha")
        st.image(heatmap_overlay(image, g, heat_alpha), use_container_width=True,
                 caption=f"Heatmap nhánh chữ trên ảnh gốc · xanh = không có chữ, đỏ = chắc chắn có chữ Hán Nôm · ô cao nhất {g.max():.2f}")
        with st.expander("lưới 24×24 gốc (trắng đen)"):
            g8 = (np.clip(g, 0, 1) * 255).astype(np.uint8)
            st.image(Image.fromarray(g8).resize((240, 240), Image.NEAREST), caption="24×24 ô · sáng = có chữ Hán Nôm", width=240)

with col_res:
    if r2 is not None and show_passes:
        st.info(f"Ảnh bị **{r1['orientation']}** (độ tin {r1['orient_conf']:.2f}) → đã xoay/lật về thẳng và phân loại lại. Kết quả cuối lấy theo lần 2.")
        show_result(r2, "⭐ Kết quả sau khi sửa chiều (lần 2)")
        show_result(r1, "Lần 1 (ảnh gốc)")
    elif r2 is not None:
        st.info(f"Ảnh bị **{r1['orientation']}** (độ tin {r1['orient_conf']:.2f}) → đã xoay/lật về thẳng trước khi phân loại.")
        show_result(final_view(r1, r2), "⭐ Kết quả")
    else:
        show_result(r1, "⭐ Kết quả")
        if r1.get("orientation", "0") != "0":
            st.warning(f"Chiều ảnh dự đoán {r1['orientation']} nhưng độ tin {r1['orient_conf']:.2f} < {min_conf:.2f} nên không tự sửa.")

st.divider()
st.caption("Tầng 2 chỉ chạy khi tầng 1 = SinoNom; tầng 3 và chiều ảnh chỉ chạy khi tầng 2 = general. Model chiều 5 lớp không hỗ trợ ảnh vừa xoay vừa lật.")
