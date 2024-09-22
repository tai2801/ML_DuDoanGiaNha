# HomePredict RNN · Dự đoán giá nhà bằng mạng nơ-ron hồi quy

Hệ thống dự đoán giá nhà TP.HCM kết hợp **RNN (LSTM/GRU)** trên chuỗi giá theo
khu vực với **mô hình hồi quy đặc trưng** (Linear Regression / Random Forest):

```
Giá dự đoán = giá/m² dự báo của quận (GRU) × diện tích × hệ số căn nhà (Random Forest)
```

Mọi dự đoán đều kèm **khoảng sai số ±MAE**, **cảnh báo ngoài phân phối (OOD)**
và **giải thích yếu tố ảnh hưởng**. ⚠️ **Dữ liệu hoàn toàn là MÔ PHỎNG** (sinh
bằng công thức + nhiễu, seed 42) phục vụ học tập môn Học máy — không dùng để
định giá thật.

## Kết quả trên tập test (1.021 giao dịch)
| Mô hình | MAE (tỷ) | RMSE (tỷ) | R² |
|---|---|---|---|
| Linear Regression | 1,8131 | 4,3255 | 0,6444 |
| Random Forest | 2,7940 | 3,8659 | 0,7160 |
| **Hybrid GRU + RF** | **1,1844** | **1,9959** | **0,9243** |

Chuỗi giá/m²: GRU thắng LSTM trên validation (RMSE 1,17 vs 1,80 triệu/m²) nên
được chọn cho nhánh kết hợp. Chi tiết: `docs/EXPERIMENT_RESULTS.md`.

## Cài đặt
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # gồm tensorflow-cpu ~600MB
```

## Chạy
```bash
python -m src.generate_data   # sinh data/transactions.csv + district_monthly.csv (mô phỏng)
python -m src.train           # huấn luyện -> models/, outputs/metrics.json, biểu đồ
streamlit run app.py          # dashboard http://localhost:8501
uvicorn src.api:app --port 8000   # (tùy chọn) API: /health /metrics /predict
```

## Dự đoán bằng CLI
```bash
python -m src.predict --district "Quận 7" --area 85 --bedrooms 3 --bathrooms 2 \
  --floors 3 --year-built 2018 --condition "Tốt" --distance 7.2
# -> 11,924 tỷ ± 1,184 tỷ, kèm giải thích + cảnh báo OOD nếu có
```
Kết quả dự đoán tự ghi vào `data/prediction_history.db` (xem ở tab Lịch sử).

## Kiểm thử
```bash
python -m pytest tests/ -q    # 5 test: tiền xử lý, chống leakage, OOD, end-to-end
```

## Cấu trúc
```
config/config.yaml   Tham số tái lập thí nghiệm
src/   generate_data · preprocess · models_rnn · train · predict · api · ood
app.py               Dashboard Streamlit (4 tab)
data/                CSV mô phỏng + lịch sử dự đoán (SQLite)
models/              .keras (LSTM/GRU), .pkl (baseline, factor RF, scaler), JSON cấu hình/thống kê
outputs/             metrics.json, test_predictions.csv, charts/
docs/                14 tài liệu: overview, bài toán, use case, kiến trúc, dataset,
                     tiền xử lý, mô hình, cấu hình train, metrics, kết quả TN,
                     giới hạn, test report, hướng phát triển, tham khảo
tests/               pytest
```

## Tài liệu nhanh
- Kiến trúc kết hợp & vì sao chọn GRU: `docs/MODEL_ARCHITECTURE.md`
- Dữ liệu mô phỏng sinh thế nào: `docs/DATASET_DESCRIPTION.md`
- Giới hạn mô hình: `docs/LIMITATIONS.md` · Kiểm thử: `docs/TEST_REPORT.md`

## Giấy phép
MIT License (xem `LICENSE`).
