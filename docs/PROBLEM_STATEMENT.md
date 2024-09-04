# Problem Statement + Requirements

## Bài toán
Người mua/bán nhà và sinh viên học máy cần một công cụ ước tính giá nhà minh bạch:
không chỉ đưa một con số, mà cho biết **vì sao** ra con số đó, **sai số kỳ vọng**
bao nhiêu, và **khi nào không nên tin** kết quả (dữ liệu đầu vào lạ so với dữ
liệu đã học). Giá nhà chịu ảnh hưởng đồng thời của đặc trưng căn nhà và xu hướng
giá theo khu vực theo thời gian — do đó cần mô hình kết hợp cả hai.

## Functional Requirements
- FR1: Sinh/nạp dữ liệu giao dịch (CSV) và chuỗi giá tháng theo quận.
- FR2: Tiền xử lý: điền khuyết, xử lý outlier, mã hóa phân loại, chuẩn hóa —
  mọi thống kê chỉ học từ tập train.
- FR3: Huấn luyện baseline (Linear Regression, Random Forest) và RNN (LSTM, GRU).
- FR4: Dự đoán giá cho một căn nhà cụ thể, kèm khoảng sai số ±MAE và giải thích.
- FR5: Cảnh báo khi đầu vào nằm ngoài phân phối huấn luyện (OOD).
- FR6: Dashboard: xem dữ liệu, dự đoán, so sánh mô hình, lịch sử dự đoán.
- FR7: Lưu mô hình, scaler, encoder, cấu hình để tái lập và nạp lại dự đoán.
- FR8: API/CLI dự đoán dùng chung một dịch vụ dự đoán.

## Non-Functional Requirements
- NFR1: Tái lập — seed cố định, file config YAML, metrics ghi ra JSON.
- NFR2: Minh bạch — dữ liệu mô phỏng được ghi rõ; kèm khoảng sai số đo trên tập test.
- NFR3: Thời gian train toàn bộ < ~10 phút trên CPU thường (thực tế ~4 phút).
- NFR4: Chạy được offline sau khi cài requirements; không cần API key.
