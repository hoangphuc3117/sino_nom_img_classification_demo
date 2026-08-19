"""Web demo phân loại ảnh tài liệu Hán Nôm — 3 model + Ensemble.
Chạy:  streamlit run web_demo/app.py
"""
import streamlit as st
from PIL import Image

from model_defs import (load_models, predict_all, decode, DEVICE,
                        L1_NAMES, L2_NAMES, L3_NAMES, KAGGLE_MODEL_IDS)

st.set_page_config(page_title="Hán Nôm Classifier", page_icon="🏮", layout="wide")

st.title("🏮 Phân loại ảnh tài liệu Hán Nôm")
st.caption(f"3 tầng: SinoNom/NonSinoNom → loại tài liệu → hướng chữ · Inference: **CPU** ({DEVICE})")


@st.cache_resource(show_spinner="Đang load 3 model (lần đầu ~15s)...")
def get_models():
    return load_models()


models = get_models()

with st.sidebar:
    st.header("Model")
    st.caption("Model sẽ được tải tự động từ KaggleHub ở lần chạy đầu tiên và cache cục bộ cho các lần sau.")
    for name, kaggle_id in KAGGLE_MODEL_IDS.items():
        ok = name in models
        st.markdown(f"{'✅' if ok else '❌'} **{name}**")
        st.caption(kaggle_id)
    if len(models) >= 2:
        st.markdown("**Ensemble** = trung bình softmax; thời gian ensemble = tổng thời gian thành viên")
    st.divider()
    st.caption("Kết quả trên test 260 ảnh (18/08): Ensemble 3 model sai 6/260 · "
               "L2 99.1% · L3 100%")

if not models:
    st.error("Không load được model nào — kiểm tra đường dẫn checkpoint trong web_demo/model_defs.py")
    st.stop()

uploaded = st.file_uploader("Tải ảnh tài liệu lên (jpg/png/webp...)",
                            type=["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"])

if uploaded is None:
    st.info("⬆️ Tải một ảnh lên để phân loại. Ảnh sẽ được chạy qua từng model riêng lẻ và bản ensemble.")
    st.stop()

image = Image.open(uploaded)
col_img, col_res = st.columns([1, 2], gap="large")

with col_img:
    st.image(image, caption=f"{uploaded.name} · {image.size[0]}×{image.size[1]}px",
             use_container_width=True)

with st.spinner("Đang phân loại..."):
    results = predict_all(models, image)

with col_res:
    names = list(results.keys())
    # Thu tu: Ens 3 model -> 3 ensemble cap -> 3 model don
    def order(n):
        if n.startswith("⭐"): return 0
        if n.startswith("Ens:"): return 1
        return 2
    names.sort(key=order)
    for name in names:
        entry = results[name]
        d = decode(entry["probs"])
        is_ens = name.startswith("⭐")
        with st.container(border=True):
            head_l, head_m, head_r = st.columns([3, 1, 2])
            with head_l:
                st.subheader(name)
            with head_m:
                st.metric("⏱ giây", f"{entry['time']:.2f}")
            with head_r:
                if is_ens:
                    st.success(f"**{d['final']}**")
                else:
                    st.info(f"**{d['final']}**")
            c1, c2, c3 = st.columns(3)
            with c1:
                lbl, p, dist = d["L1"]
                st.markdown(f"**L1:** {lbl} — {100*p:.1f}%")
                st.progress(min(1.0, p))
                with st.expander("phân bố"):
                    for i, n in enumerate(L1_NAMES):
                        st.caption(f"{n}: {100*dist[i]:.1f}%")
            with c2:
                if d["L2"] is None:
                    st.markdown("**L2:** —")
                else:
                    lbl, p, dist = d["L2"]
                    st.markdown(f"**L2:** {lbl} — {100*p:.1f}%")
                    st.progress(min(1.0, p))
                    with st.expander("phân bố"):
                        for i, n in enumerate(L2_NAMES):
                            st.caption(f"{n}: {100*dist[i]:.1f}%")
            with c3:
                if d["L3"] is None:
                    st.markdown("**L3:** —")
                else:
                    lbl, p, dist = d["L3"]
                    st.markdown(f"**L3:** {lbl} — {100*p:.1f}%")
                    st.progress(min(1.0, p))
                    with st.expander("phân bố"):
                        for i, n in enumerate(L3_NAMES):
                            st.caption(f"{n}: {100*dist[i]:.1f}%")

st.divider()
st.caption("L2 chỉ hiển thị khi L1 = SinoNom; L3 chỉ hiển thị khi L2 = general (đúng logic phân cấp). "
           "Model flat 6 lớp được ánh xạ về 3 tầng để so cùng thước đo.")
