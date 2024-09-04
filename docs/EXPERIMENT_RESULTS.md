# Experiment Results — Kết quả huấn luyện (22/09/2024)

Nguồn: `outputs/metrics.json` do `python -m src.train` sinh (seed 42).

## 1. Dự đoán giá nhà trên tập test (1.021 giao dịch) — đơn vị: tỷ VND
| Mô hình | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression (baseline) | 1,8131 | 4,3255 | 0,6444 |
| Random Forest (baseline) | 2,7940 | 3,8659 | 0,7160 |
| **Hybrid GRU + Factor RF** | **1,1844** | **1,9959** | **0,9243** |

Nhận xét:
- Mô hình Hybrid vượt cả hai baseline ở cả 3 độ đo: việc tách "xu hướng giá khu vực" (RNN) khỏi "đặc trưng căn nhà" (hệ số RF) hiệu quả hơn hồi quy trực tiếp.
- Random Forest có R² cao hơn Linear nhưng MAE cao hơn — RF (học trên log giá) hạn chế được các lỗi cực lớn (RMSE thấp hơn).

## 2. Dự đoán giá/m² tháng kế tiếp theo quận — đơn vị: triệu VND/m²
| Mô hình | Val (72 mẫu) MAE / RMSE / R² | Test (216 mẫu) MAE / RMSE / R² | Epochs chạy |
|---|---|---|---|
| LSTM | 1,4296 / 1,8000 / 0,9988 | 3,9678 / 4,8935 / 0,9922 | 21 |
| **GRU** | **0,9538 / 1,1712 / 0,9995** | **1,6878 / 2,0925 / 0,9986** | 38 |

GRU đạt sai số thấp hơn LSTM trên tập validation nên được chọn cho nhánh hybrid.

## 3. Phân tích ảnh hưởng
- **Vị trí**: giá/m² kỳ cuối chênh ~5 lần giữa Q1 (276,4 triệu) và Hóc Môn (54,6 triệu); giá nhà trung bình 17,74 tỷ (Q1) vs 3,61 tỷ (Hóc Môn).
- **Diện tích – giá**: tương quan r = 0,751; trong Factor RF, diện tích là đặc trưng quan trọng nhất (0,657), tiếp theo là khoảng cách trung tâm (0,208).
- Biểu đồ: `outputs/charts/district_price_trend.png`, `pred_vs_actual.png`, `model_compare.png`. Dự đoán từng dòng test: `outputs/test_predictions.csv`.

## 4. Ví dụ dự đoán qua CLI
Nhà Quận 7, 85 m², 3 PN, xây 2018, tình trạng Tốt → **11,924 tỷ ± 1,184 tỷ** (khoảng kỳ vọng 10,739–13,108 tỷ), ppm2 Q7 dự báo 146,06 triệu, hệ số nhà 0,96, không cảnh báo OOD. Nhà Quận 1 diện tích 900 m² → trả kết quả kèm cảnh báo ngoài phân phối huấn luyện (max train 341 m²).
