# Web demo — Phân loại ảnh tài liệu Hán Nôm (bản 02/10/2026: tầng 1 nhánh chữ vòng 21)

Streamlit: tải ảnh lên → phân loại 3 tầng bằng bộ model tốt nhất → xác định chiều ảnh → nếu ảnh bị xoay/lật thì tự sửa và phân loại lại.

## Pipeline
| Bước | Model | Ghi chú |
|---|---|---|
| Tầng 1: có Hán Nôm? | `models/text_branch_r21.pth` (vòng 21, 02/10/2026 — nhãn ô mới, hàm xếp hạng vị trí, mô phỏng tile khi train; bản trước `text_branch_r7.pth`) — nhánh chữ EfficientNet-B4, lưới 24×24 @768, bậc thang (mặc định): toàn khung; nếu điểm ≤ ngưỡng thì chấm thêm 2 tile đặt tại 2 ô điểm cao nhất cách xa nhau, cỡ tile ~1400 px ảnh gốc (35–56 % cạnh; ảnh ≤ 2500 px giữ 56 %). Sidebar có thêm: chỉ toàn khung / 1 tile theo điểm / 1 tile giữa / 4 tile | có Hán Nôm nếu điểm ô lớn nhất > 0,60 |
| Tầng 2: nhóm tài liệu | `models/dhc_b4_2tang.pth` (DHC 2 tầng: loại tài liệu + hướng chữ, train Kaggle theo cặp, đổi nhãn dọc↔ngang khi xoay 90°/270°) + `models/flat_b4_5cls.pth` (flat 5 lớp, train theo cặp cùng nguyên tắc) — 26/09/2026; **05/10/2026: cả hai được fine-tune** chống nhận nhầm văn bia (chữ màu trên nền đá, trang viết tay chụp tại chỗ) — `outputs/finetune_tier2_bia/` (dhc_ep3 + flat_ep1), test_full 1278 ảnh 93,66 % (bản 26/09: 93,11 %), bia recall giữ 90,5 %; bản gốc tại `backups/tier2_20261005_truoc_finetune_bia/` | trung bình 0,5/0,5 |
| Tầng 3: dọc/ngang | hai model trên (+ DHC trên ảnh xoay 90°) | chỉ khi nhóm = general; **luật mới (05/10/2026):** (1) kiểm tra nhất quán: P(dọc) của DHC = (P(dọc) ảnh gốc + P(ngang) ảnh xoay 90°)/2 — DHC không đổi hướng khi xoay thì mất quyền ưu tiên; (2) luật lai: DHC P(dọc) > 0,95 thì lấy dọc theo DHC, còn lại trung bình 2 model. Trên 451 dọc + 37 ngang của test_full: OR-dọc cũ 450/34, luật mới 450/36; sửa được thơ chữ đỏ trên đá (DHC 0,98 ở 0° và 1,00 ở 90°, flat 0,13). Thêm ≈60–100 ms CPU |
| Chiều ảnh | `models/orient5_pplcnet.pth` — PP-LCNet 5 lớp 0/90/180/270/mirror | chỉ khi nhóm = general; lệch với độ tin ≥ 0,8 → sửa ảnh → phân loại lại, kết quả cuối lấy lần 2 |

Mã: `model_defs.py` (model, tiền xử lý, `classify`, `classify_with_fix`), `pplcnet_torch.py` (PP-LCNet), `app.py` (giao diện). Bản cũ (2 model production, KaggleHub) lưu ở `*.bak_20260924`.

## Kết quả đo tầng 1 vòng 21 (02/10/2026, chạy đúng hàm của service trên CPU 4 luồng)
| | r21, T = 0,60, tile mới (hiện tại) | r7, T = 0,50 (trước) |
|---|---|---|
| Ảnh Hán bị sót (test_full) | 4 / 1162 | 3 / 1162 |
| Nhận nhầm: test_full · folder giữ riêng · chữ Việt đình chùa · non_chu_viet · trang văn bản không Hán | 21 · 27 · 30 · 10 · 9 = 97 | 36 · 44 · 49 · 16 · 8 = 153 |
| Trang xoay nhận đúng | 255 / 255 | 254 / 255 |
| 6 lớp (test_full) | 93,11 % | 92,02 % |
| Thời gian service test_full: trung vị / TB / p95 | 0,64 / 0,73 / 1,19 s | 0,64 / 0,70 / 1,13 s |

Bản trước (r7) lưu ở `*.bak_20261002_r7`.

## Kết quả đo cũ (test_full 1278 ảnh, 23/09/2026, tile, T = 0,50, nhánh chữ vòng 6/7)
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
- Flat 6 lớp theo thứ tự service: non_sino_nom, admin, epitaph, scene, horizontal, vertical — không còn nhãn con scene. `model_defs.py` vẫn nhận checkpoint flat 8 nhãn con (tự nhận theo số đầu ra, ánh xạ bằng `MAP8`).
- Test_full 1278 ảnh với bộ model giống service (24/09/2026): 6 lớp 92,88 %, dọc→ngang 1/451 (test_ver_221); flat 8 nhãn cũ: 93,27 %, 0/451.

## Tầng 1 bậc thang (24/09/2026)
Test 1276 ảnh, T = 0,50: bỏ lọt 4/1162, nhận nhầm 34/114 (4 tile luôn: 4 / 45; chỉ toàn khung: 17 / 23). Chỉ khoảng 9 % ảnh phải chạy thêm 2 tile, nên thời gian CPU trung bình ≈0,6–0,7 s/ảnh cho tầng 1 (ảnh chạy thêm tile ≈1,6 s).
