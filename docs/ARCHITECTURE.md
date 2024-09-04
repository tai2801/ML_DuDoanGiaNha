# Architecture

```
data/ (CSV mô phỏng) ──> src/generate_data.py (seed cố định)
        │
        ▼
src/preprocess.py  Cleaner (fit trên TRAIN) + temporal_split (theo tháng bán)
        │
        ├──> Baseline: Linear Regression / Random Forest (đặc trưng -> log giá)
        ├──> Factor RF: học factor = giá / (ppm2 thật x diện tích)
        └──> src/models_rnn.py: LSTM & GRU (chuỗi 12 tháng ppm2 quận -> tháng kế)
        │
        ▼
models/  *.keras, *.pkl (gzip cho RF), scaler, cleaner.json, train_stats.json,
         series_history.csv, train_config.json
outputs/ metrics.json, test_predictions.csv, charts/*.png
        │
        ▼
src/predict.py  HomePredictor (OOD check -> RNN forecast -> x hệ số RF
                -> giá + khoảng ±MAE + giải thích) ──> SQLite lịch sử
        ├── CLI: python -m src.predict
        ├── API: src/api.py (FastAPI: /health /metrics /predict)
        └── UI:  app.py (Streamlit 4 tab)
tests/  pytest: preprocessing, leakage guard, OOD, end-to-end train rút gọn
```

## Quyết định thiết kế chính
- **Chia tập theo thời gian** cho cả giao dịch (train < 2023-07 ≤ val < 2024-07 ≤ test)
  và chuỗi giá (RNN train trên tháng đích < 2024-01, val = 2024, test ≥ 2024-07),
  tránh rò rỉ dữ liệu từ tương lai vào quá khứ (data leakage) khi xáo trộn ngẫu nhiên.
- **Scaler chuỗi (MinMax) chỉ fit trên giai đoạn train** của chuỗi giá.
- **Dự đoán kết hợp (hybrid)** thay vì ép RNN dự đoán trực tiếp giá nhà:
  RNN giỏi xu hướng thời gian của khu vực; RF giỏi quan hệ phi tuyến của
  đặc trưng căn nhà. Xem MODEL_ARCHITECTURE.md.
