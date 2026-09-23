# Web demo — Phân loại ảnh tài liệu Hán Nôm (bản 24/09/2026)

Streamlit: tải ảnh lên → phân loại 3 tầng bằng bộ model tốt nhất → xác định chiều ảnh → nếu ảnh bị xoay/lật thì tự sửa và phân loại lại.

## Pipeline
| Bước | Model | Ghi chú |
|---|---|---|
| Tầng 1: có Hán Nôm? | `models/text_branch_r6.pth` — nhánh chữ EfficientNet-B4, lưới 24×24 @768, toàn khung + 4 tile | có Hán Nôm nếu điểm ô lớn nhất > 0,50 |
| Tầng 2: nhóm tài liệu | `models/dhc_b4_scratch.pth` (hierarchical 2/4/2) + `models/flat_b4_scratch_ep9.pth` (flat 8 nhãn con → 6 lớp) | trung bình 0,5/0,5, y logic service đang chạy |
| Tầng 3: dọc/ngang | hai model trên | chỉ khi nhóm = general; một model nói dọc → dọc (OR-dọc) |
| Chiều ảnh | `models/orient5_pplcnet.pth` — PP-LCNet 5 lớp 0/90/180/270/mirror | chỉ khi nhóm = general; lệch với độ tin ≥ 0,8 → sửa ảnh → phân loại lại, kết quả cuối lấy lần 2 |

Mã: `model_defs.py` (model, tiền xử lý, `classify`, `classify_with_fix`), `pplcnet_torch.py` (PP-LCNet), `app.py` (giao diện). Bản cũ (2 model production, KaggleHub) lưu ở `*.bak_20260924`.

## Kết quả đo (test_full 1278 ảnh, 23/09/2026, tile, T = 0,50)
Tầng 1: 96,2 % (bỏ sót 4/1162 Hán Nôm, nhận nhầm 45/116 none) · Tầng 2–3 khi tầng 1 đúng: 96,0 % · 6 lớp: ≈92,5 % · Chiều ảnh: val 99,5 %, test 97,7 %.
So với 2 model đang chạy cùng logic: tầng 1 93,9 % (nhận nhầm 72/116), tầng 2–3 88,9 %, 6 lớp 83,9 %.

## Thời gian (CPU Apple M4 Pro, 7 luồng, batch 1)
≈2,4–2,7 s/ảnh có tile (tầng 1 ≈2,1 s, flat ≈0,2 s, DHC ≈0,15 s, chiều ảnh ≈0,015 s) · ≈0,55 s không tile (tắt trong sidebar; tầng 1 sót nhiều hơn). Ảnh bị xoay/lật: lượt 2 dùng lại tầng 1, chỉ chạy lại flat + DHC + chiều ≈0,35 s.

## Chạy
```bash
pip install -r requirements.txt
streamlit run app.py
```
Mở http://localhost:8501. Sidebar có: bật/tắt tile, ngưỡng tầng 1, trọng số flat ở tầng 2, bật/tắt tự sửa chiều và ngưỡng độ tin.

## Lưu ý
- 4 checkpoint (≈218 MB) nằm trong `models/`, không còn tải từ KaggleHub. Nếu deploy Streamlit Cloud cần đưa file lên Kaggle/Git LFS rồi thêm bước tải.
- Model chiều 5 lớp không hỗ trợ ảnh vừa xoay vừa lật (quyết định giữ 5 lớp, 24/09/2026).
- Thứ tự 8 nhãn con của flat mới: non_sino_nom, admin, epitaph, scene_img, scene_obj, scene_txt, vertical, horizontal (khác thứ tự 6 lớp của service; đã ánh xạ trong `MAP8`).
