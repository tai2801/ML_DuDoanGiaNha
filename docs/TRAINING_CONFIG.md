# Training Configuration

Toàn bộ tham số nằm ở `config/config.yaml`, bản sao đã dùng được lưu tại
`models/train_config.json` sau khi train — chạy lại cùng config + seed cho
kết quả tương đương (sai khác nhỏ do tính song song của CPU).

| Hạng mục | Giá trị |
|---|---|
| Seed | 42 (generator, numpy, TensorFlow) |
| Giao dịch mô phỏng | 3.600 (12 quận × 84 tháng) |
| Split giao dịch | train < 2023-07 · val 2023-07→2024-06 · test ≥ 2024-07 |
| Split chuỗi RNN | train tháng đích < 2024-01 · val 2024 · test ≥ 2024-07 |
| RNN | window 12, units 32, Dense 16, Adam lr 0,005, MSE, batch 16, tối đa 80 epochs, EarlyStopping patience 10 (LSTM dừng ở 21, GRU ở 38 epochs) |
| Random Forest | 300 cây, max_depth 14, min_samples_leaf 2 |
| TensorFlow | tensorflow-cpu, Python 3.11 |
| Thời gian train toàn bộ | ~3–4 phút trên CPU |

Lệnh tái lập:
```bash
python -m src.generate_data      # sinh dữ liệu (hoặc train tự sinh nếu thiếu)
python -m src.train              # train + metrics + biểu đồ
python -m src.train --epochs 30  # ghi đè số epochs RNN nếu muốn chạy nhanh
```
