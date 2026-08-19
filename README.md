# Web demo — Phân loại ảnh tài liệu Hán Nôm

Demo Streamlit: tải ảnh lên → xem kết quả 3 tầng của từng model (ResNet50+CBAM,
EfficientNet-B4 hierarchical, EfficientNet-B4 flat) và bản Ensemble (trung bình softmax).

Từ bản này, 3 checkpoint được tải tự động qua KaggleHub:
- ResNet50+CBAM: `phuchoangnguyen/sinonomimg-resnet50-hier/pyTorch/default`
- EfficientNet-B4 hierarchical: `phuchoangnguyen/sinonomimg-eb4-hier/pyTorch/default`
- EfficientNet-B4 flat: `phuchoangnguyen/sinonomimg-eb4-flat/pyTorch/default`

Nếu chạy lần đầu trên máy mới hoặc Streamlit Cloud, hãy cài thêm `kagglehub`.

## Chạy

```bash
cd <thư mục dự án>
.venv/bin/streamlit run web_demo/app.py
```

Mở http://localhost:8501

## Checkpoint

Đường dẫn cấu hình trong `model_defs.py` sẽ ưu tiên tải từ KaggleHub rồi mới rơi về checkpoint local trong `models/` nếu có.

Checkpoint nào thiếu sẽ tự bỏ qua; Ensemble tính trên các model load được (≥2).
