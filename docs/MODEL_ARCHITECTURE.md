# Model Architecture

## Vì sao LSTM/GRU cho chuỗi giá khu vực?
Giá/m² của một quận theo tháng là **chuỗi thời gian có xu hướng + mùa vụ + nhiễu**:
giá tháng sau phụ thuộc mạnh vào các tháng trước. Mô hình hồi quy đặc trưng tĩnh
không nhìn thấy "đà tăng" của khu vực; RNN với cổng nhớ (LSTM/GRU) học được phụ
thuộc dài hạn trong cửa sổ 12 tháng và thường tốt hơn hẳn mô hình naive cho bài
toán one-step-ahead. GRU ít tham số hơn LSTM, train nhanh hơn — cả hai được huấn
luyện và chọn theo RMSE trên tập validation (kết quả: GRU thắng, xem EXPERIMENT_RESULTS).

## Kiến trúc đã cài đặt (đúng như code)
| Nhánh | Đầu vào | Mô hình | Đầu ra |
|---|---|---|---|
| Baseline Linear | Đặc trưng nhà + quận (one-hot, scaled) | LinearRegression trên log(giá) | Giá (tỷ) |
| Baseline RF | Như trên | RandomForest 300 cây, depth 14 | Giá (tỷ) |
| RNN LSTM | Chuỗi 12 tháng ppm2 của 1 quận (MinMax) | LSTM(32) → Dense(16, relu) → Dense(1) | ppm2 tháng kế (triệu/m²) |
| RNN GRU | Như trên | GRU(32) → Dense(16, relu) → Dense(1) | ppm2 tháng kế |
| Factor RF | Đặc trưng nhà + quận | RandomForest (cùng tham số) | Hệ số nhà factor |
| **Hybrid** | ppm2 dự báo (GRU) × diện tích × factor (RF) | Kết hợp | **Giá cuối (tỷ)** |

**Hệ số nhà (factor)** được định nghĩa từ dữ liệu train:
`factor = giá thật / (ppm2 thật của quận trong tháng bán × diện tích)` — tức
"mức đắt/rẻ của căn nhà so với mặt bằng quận tại thời điểm bán" (nhà mới, ít
tuổi → > 1; nhà cũ → < 1). Đánh giá hybrid trên tập test dùng dự báo
**one-step-ahead**: với mỗi giao dịch test, ppm2 của tháng bán được dự báo từ
12 tháng thật ngay trước đó — đúng cách một hệ thống thật vận hành.

**Dự báo tương lai** (tháng chưa có dữ liệu): dự báo đệ quy (recursive
multi-step) từ tháng cuối của chuỗi, số bước được báo rõ trong kết quả.

## Giải thích dự đoán (explainability)
- Độ quan trọng đặc trưng của Factor RF (top: diện tích 0,657 · khoảng cách
  trung tâm 0,208 · quận 1 0,037 · năm xây 0,020).
- Hệ số tuyến tính của baseline Linear (trong pipeline đã lưu).
- Phân rã minh bạch của hybrid: ppm2 quận hiện tại → ppm2 dự báo → × diện tích
  → × hệ số nhà. Người dùng thấy từng thành phần, không phải hộp đen.
