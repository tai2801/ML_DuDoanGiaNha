# Preprocessing

Mã nguồn: `src/preprocess.py`. Nguyên tắc xuyên suốt: **mọi thống kê chỉ fit
trên tập train**, tránh data leakage.

1. **Điền khuyết (`Cleaner.fit/transform`)**: cột số → trung vị của train;
   `condition` → mode của train. (Dữ liệu thô thiếu 111 year_built, 73 condition,
   57 bathrooms trên 3.600 dòng.)
2. **Outlier giá**: cắt (clip) log(price) theo khoảng IQR × 1,5 tính trên train —
   xử lý ~1% giao dịch giá bất thường do generator chèn.
3. **Đặc trưng dẫn xuất**: `age_years = năm bán − year_built` (clip ≥ 0).
4. **Mã hóa + chuẩn hóa** (trong sklearn Pipeline của từng mô hình):
   numeric → StandardScaler; `district`, `condition` → OneHotEncoder
   (`handle_unknown="ignore"` để gặp quận lạ không sập).
5. **Mục tiêu**: baseline học `log(price_ty)` rồi mũ hóa khi dự đoán (giảm lệch
   phân phối giá); metrics tính trên thang giá gốc (tỷ VND).
6. **Chuỗi giá cho RNN**: MinMaxScaler fit **chỉ trên các tháng < 2024-01**;
   cửa sổ trượt 12 tháng → dự báo tháng kế tiếp, theo từng quận.
7. **Chia tập theo thời gian** (không shuffle):
   giao dịch — train < 2023-07 (1.998), val 2023-07→2024-06 (581), test ≥ 2024-07 (1.021);
   chuỗi — train tháng đích < 2024-01, val = 2024 (72 mẫu), test ≥ 2024-07 (216 mẫu).
   Có assert + test tự động chống chồng lấn thời gian.
