# Evaluation Metrics

Ba độ đo chuẩn cho hồi quy, tính trên tập **test** (chưa từng dùng để fit hay
chọn mô hình — việc chọn LSTM/GRU dựa trên validation):

- **MAE** (Mean Absolute Error): sai lệch tuyệt đối trung bình — dễ diễn giải
  ("sai trung bình ±X tỷ"). Đây là con số được hiển thị kèm mọi dự đoán.
- **RMSE** (Root Mean Squared Error): phạt nặng lỗi lớn; dùng để chọn mô hình RNN.
- **R²**: tỷ lệ phương sai giá được mô hình giải thích (1 = hoàn hảo, 0 = ngang
  dự đoán bằng giá trị trung bình).

Lưu ý thang đo: baseline và hybrid đo trên **giá nhà (tỷ VND)**; LSTM/GRU đo
trên **giá/m² (triệu VND/m²)** vì đó là đầu ra trực tiếp của chúng. So sánh
"ai thắng" ở mức giá nhà phải nhìn vào hàng Hybrid so với hai baseline.
Kết quả chi tiết: EXPERIMENT_RESULTS.md và `outputs/metrics.json`.
