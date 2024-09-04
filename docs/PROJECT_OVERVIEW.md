# Project Overview — HomePredict RNN

**Môn học:** Học máy — HUIT

HomePredict là hệ thống dự đoán giá nhà ở TP.HCM kết hợp hai góc nhìn:

1. **Đặc trưng căn nhà + khu vực** tại thời điểm bán (diện tích, phòng, tuổi nhà, tình trạng, khoảng cách trung tâm, quận) — học bằng mô hình hồi quy truyền thống.
2. **Chuỗi giá lịch sử theo khu vực** (giá/m² trung bình tháng của từng quận) — học bằng mạng nơ-ron hồi quy (LSTM/GRU) để dự báo giá/m² tháng kế tiếp.

Sản phẩm gồm: pipeline huấn luyện tái lập (`src/train.py`), dịch vụ dự đoán (`src/predict.py` + CLI + FastAPI `src/api.py`), dashboard Streamlit (`app.py`) và bộ test pytest. **Toàn bộ dữ liệu là MÔ PHỎNG** (xem `DATASET_DESCRIPTION.md`).

Điểm nổi bật: có baseline so sánh, chia tập theo thời gian chống data leakage, cảnh báo ngoài phân phối (OOD), giải thích yếu tố ảnh hưởng và khoảng sai số đi kèm mọi dự đoán.
