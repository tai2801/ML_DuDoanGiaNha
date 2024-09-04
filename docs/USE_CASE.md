# Use Case

## Actors
- **Người dùng cuối** (người mua/bán nhà, môi giới tập sự): tra cứu ước tính giá.
- **Nhà phân tích / sinh viên**: nạp dữ liệu, huấn luyện lại, so sánh mô hình.
- **Hệ thống**: sinh dữ liệu mô phỏng, huấn luyện, cảnh báo OOD, ghi lịch sử.

## Use case chính
| Mã | Tên | Luồng chính |
|---|---|---|
| UC1 | Xem dữ liệu | Mở dashboard → tab Dữ liệu → (tùy chọn) upload CSV → xem bảng, phân bố diện tích–giá, giá/m² theo quận |
| UC2 | Dự đoán giá nhà | Nhập đặc trưng nhà → hệ thống kiểm tra OOD → dự báo giá/m² quận bằng RNN → nhân hệ số căn nhà (RF) → trả giá + khoảng sai số + giải thích → ghi lịch sử |
| UC3 | So sánh mô hình | Tab Hiệu suất → xem MAE/RMSE/R² của Linear/RF/LSTM/GRU/Hybrid + biểu đồ dự đoán vs thực tế |
| UC4 | Huấn luyện lại | Sửa `config/config.yaml` → chạy `python -m src.train` → artifacts + metrics + biểu đồ được cập nhật |
| UC5 | Dự đoán qua API/CLI | Gọi `POST /predict` hoặc CLI → nhận JSON cùng cấu trúc với dashboard |

## Luồng ngoại lệ
- Input ngoài phân phối (diện tích 900m², quận lạ): vẫn trả kết quả nhưng kèm
  cảnh báo đỏ; quận không có trong train → API trả 400.
- Chưa huấn luyện: dashboard/API báo rõ cần chạy train trước.
- CSV upload sai định dạng: dashboard báo lỗi nạp và giữ dữ liệu mẫu.
