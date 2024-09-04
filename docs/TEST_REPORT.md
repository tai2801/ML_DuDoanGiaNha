# Test Report — HomePredict RNN

Ngày chạy: 22/09/2024 · Môi trường: Windows 10, Python 3.11, TensorFlow, scikit-learn, Streamlit, FastAPI, pytest.

## 1. Pytest — 5/5 PASS
```
python -m pytest tests/ -q   ->   5 passed
```
| Test | Nội dung |
|---|---|
| `test_generator_shape_and_missing` | Generator sinh đúng 800 giao dịch + 1.008 dòng chuỗi; dữ liệu thô có thiếu có chủ đích |
| `test_cleaner_imputes_and_clips` | Sau Cleaner: 0 giá trị thiếu; log giá nằm trong bound IQR của train |
| `test_temporal_split_no_leakage` | max tháng train < min tháng val < min tháng test; thống kê điền khuyết đúng của train |
| `test_ood_detector` | Nhà bình thường: 0 cảnh báo; diện tích 900 m²: có cảnh báo; quận lạ: cảnh báo "không có trong dữ liệu" |
| `test_pipeline_end_to_end_tiny` | Train rút gọn (700 giao dịch, 3 epochs, RF 40 cây) trong thư mục tạm → nạp `HomePredictor` → dự đoán > 0, khoảng sai số chứa điểm dự đoán, nhà mẫu không OOD |

## 2. Huấn luyện đầy đủ
`python -m src.train`: split 1.998 / 581 / 1.021; LSTM 21 epochs, GRU 38 epochs (EarlyStopping); metrics đầy đủ trong `EXPERIMENT_RESULTS.md` và `outputs/metrics.json`; artifacts lưu trong `models/` (2 file `.keras`, RF nén gzip).

## 3. Kiểm thử tích hợp — PASS
- **CLI predict**: Quận 7, 85 m² → 11,924 tỷ ± 1,184 tỷ, giải thích đầy đủ, lịch sử được ghi vào SQLite. Ca OOD (Q1, 900 m²) trả cảnh báo đúng.
- **Streamlit**: `streamlit run app.py` → HTTP 200.
- **FastAPI**: `GET /health` → `{"status":"ok"}`; `POST /predict` trả cùng kết quả với CLI (11,924 tỷ).
