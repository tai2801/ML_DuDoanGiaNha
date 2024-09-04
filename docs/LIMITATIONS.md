# Limitations

1. **Dữ liệu mô phỏng**: giá sinh từ công thức + nhiễu nên "dễ học" hơn thị
   trường thật rất nhiều (R² ppm2 ≈ 0,999 là dấu hiệu điển hình). Mô hình
   **không dùng được để định giá thật**; muốn dùng thật phải thay bằng dữ liệu
   giao dịch/thị trường có nguồn gốc rõ ràng và huấn luyện lại.
2. **Ngoại suy theo thời gian**: dự báo đệ quy nhiều tháng về tương lai tích lũy
   sai số; hệ thống báo rõ số bước dự báo trong kết quả.
3. **Ngoại suy theo đặc trưng**: nhà ngoài phân phối train (diện tích quá lớn,
   quận chưa có dữ liệu) chỉ được *cảnh báo*, không bị chặn — người dùng phải
   tự cân nhắc (trừ quận lạ: API từ chối vì không có chuỗi giá để dự báo).
4. **Thiếu yếu tố thị trường thật**: pháp lý, quy hoạch, lãi suất, thanh khoản,
   vị trí trong quận (mặt tiền/hẻm) chưa có trong đặc trưng.
5. **Độ lệch của RF baseline**: MAE của RF tệ hơn Linear dù R² cao hơn — đã
   trình bày minh bạch trong EXPERIMENT_RESULTS thay vì giấu.
6. Chưa có cơ chế cập nhật dữ liệu/train định kỳ (online learning) — xem
   FUTURE_IMPROVEMENTS.
