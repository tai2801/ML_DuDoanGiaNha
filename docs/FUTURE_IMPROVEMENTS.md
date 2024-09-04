# Future Improvements

- Thay dữ liệu mô phỏng bằng dữ liệu thị trường có nguồn gốc (cổng thông tin
  BĐS công khai, dữ liệu mở) và đánh giá lại — kỳ vọng metric sẽ thấp hơn nhiều.
- Thêm đặc trưng: mặt tiền/hẻm, pháp lý (sổ hồng), tiện ích xung quanh (trường,
  bệnh viện, metro) dạng khoảng cách/điểm tiện ích.
- Mô hình: thử Transformer/TCN cho chuỗi giá, và dự báo theo khoảng tin cậy
  (quantile regression / conformal prediction) thay cho ±MAE cố định.
- Cập nhật dữ liệu + train lại định kỳ; theo dõi drift của phân phối đầu vào.
- Bản đồ nhiệt giá theo quận trên dashboard; so sánh nhiều căn nhà cùng lúc.
- Đóng gói Docker cho cả train + serve.
