# Dataset Description — DỮ LIỆU MÔ PHỎNG

> **Toàn bộ dữ liệu của dự án là GIẢ LẬP** bằng `src/generate_data.py` với
> seed = 42. Mức giá nền theo quận chỉ phản ánh *thứ hạng tương đối* thường
> thấy (Quận 1 cao nhất, Hóc Môn thấp nhất) — **không dùng để định giá thật**.

## 1. Bảng giao dịch `data/transactions.csv` — 3.600 dòng
| Cột | Ý nghĩa | Cách sinh |
|---|---|---|
| sale_month | Tháng bán (YYYY-MM) | Trọng số tăng dần theo năm |
| district | 12 quận/huyện TP.HCM | Theo trọng số dân cư minh họa |
| area_m2 | Diện tích 18–400 m² | Log-normal (trung vị ~75) |
| bedrooms / bathrooms / floors | Số phòng / tầng | Suy từ diện tích + Poisson/choice |
| year_built | Năm xây dựng | Trọng số thiên về nhà mới |
| condition | Mới / Tốt / Trung bình / Cũ | Xác suất 18/42/28/12% |
| distance_center_km | Khoảng cách tới trung tâm Q1 | Tâm quận + nhiễu |
| price_ty | Giá bán (tỷ VND) | Công thức dưới đây |

**Công thức giá:** `price = ppm2[quận, tháng] × area × house_factor / 1000`,
house_factor = hệ số tình trạng (Mới 1,12 · Tốt 1,03 · TB 0,95 · Cũ 0,85)
× khấu hao tuổi nhà (tối đa −22%) × tiện nghi (phòng tắm/tầng) × nhiễu log-normal σ=0,10.

**Nhiễu dữ liệu cố ý chèn** (để kiểm thử tiền xử lý): thiếu ~2,5% year_built,
~2% condition, ~1,5% bathrooms; ~1% outlier giá (×0,45 / ×1,8 / ×2,1).

## 2. Chuỗi giá khu vực `data/district_monthly.csv` — 1.008 dòng
12 quận × 84 tháng: giá/m² nền tháng 01/2019 (36–165 triệu), tăng trưởng
0,45–0,70%/tháng theo quận, mùa vụ ±2%, nhiễu bước ngẫu nhiên σ=0,6%.

## 3. Phân tích khám phá (từ dữ liệu đã sinh)
- Giá/m² kỳ cuối: Q1 276,4 · Q2 207,5 · Q3 193,9 · Phú Nhuận 172,4 ·
  Bình Thạnh 171,4 · Q10 145,4 · Q7 144,1 · Tân Bình 126,3 · Thủ Đức 110,5 ·
  Gò Vấp 103,0 · Bình Tân 83,7 · Hóc Môn 54,6 (triệu VND/m²) — vị trí là yếu tố
  chênh lệch giá lớn nhất (Q1 gấp ~5 lần Hóc Môn).
- Tương quan diện tích – giá: **r = 0,751**.
- Giá trung bình theo quận: Q1 17,74 tỷ → Hóc Môn 3,61 tỷ.
