"""HomePredict — Dashboard Streamlit.

Chạy: streamlit run app.py   (cần train trước: python -m src.train)
Dữ liệu là MÔ PHỎNG phục vụ học tập — không dùng để định giá thật.
"""
import json
import os
import sqlite3
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="HomePredict — Dự đoán giá nhà", layout="wide")
st.title("🏠 HomePredict — Dự đoán giá nhà TP.HCM")
st.caption("Dữ liệu MÔ PHỎNG (sinh bằng công thức + nhiễu, seed cố định) phục vụ học tập môn Học máy. "
           "Kết quả kèm khoảng sai số kỳ vọng — không phải định giá thật.")

tab_data, tab_predict, tab_models, tab_history = st.tabs(
    ["📊 Dữ liệu", "🔮 Dự đoán", "📈 Hiệu suất mô hình", "🕘 Lịch sử dự đoán"])


def load_metrics():
    p = os.path.join(ROOT, "outputs", "metrics.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


with tab_data:
    up = st.file_uploader("Upload CSV giao dịch (định dạng giống data/transactions.csv)", type="csv")
    default_path = os.path.join(ROOT, "data", "transactions.csv")
    if up is not None:
        df = pd.read_csv(up)
        st.success(f"Đã nạp file upload: {len(df)} dòng")
    elif os.path.exists(default_path):
        df = pd.read_csv(default_path)
        st.info(f"Đang dùng dữ liệu mẫu mô phỏng: {len(df)} giao dịch")
    else:
        df = None
        st.warning("Chưa có dữ liệu. Chạy: python -m src.generate_data")
    if df is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("Số giao dịch", len(df))
        c2.metric("Số quận", df["district"].nunique())
        c3.metric("Giá trung bình (tỷ)", round(df["price_ty"].mean(), 2))
        st.dataframe(df.head(50), use_container_width=True)
        st.subheader("Quan hệ diện tích – giá")
        st.scatter_chart(df.sample(min(800, len(df))), x="area_m2", y="price_ty")
        series_path = os.path.join(ROOT, "data", "district_monthly.csv")
        if os.path.exists(series_path):
            s = pd.read_csv(series_path)
            st.subheader("Giá/m² trung bình theo quận (tháng mới nhất)")
            latest = s[s["month"] == s["month"].max()].sort_values("ppm2", ascending=False)
            st.bar_chart(latest.set_index("district")["ppm2"])

with tab_predict:
    metrics = load_metrics()
    if metrics is None:
        st.warning("Chưa có mô hình. Chạy trước: python -m src.train")
    else:
        districts = sorted(pd.read_csv(os.path.join(ROOT, "data", "district_monthly.csv"))["district"].unique())
        with st.form("predict_form"):
            c1, c2 = st.columns(2)
            district = c1.selectbox("Quận", districts)
            area = c1.number_input("Diện tích (m²)", 15.0, 1000.0, 85.0)
            bedrooms = c1.number_input("Phòng ngủ", 1, 12, 3)
            bathrooms = c2.number_input("Phòng tắm", 1, 10, 2)
            floors = c2.number_input("Số tầng", 1, 12, 3)
            year_built = c2.number_input("Năm xây dựng", 1950, 2030, 2018)
            condition = c1.selectbox("Tình trạng", ["Mới", "Tốt", "Trung bình", "Cũ"])
            distance = c2.number_input("Khoảng cách tới trung tâm (km)", 0.1, 60.0, 7.0)
            month = st.text_input("Tháng dự đoán (YYYY-MM, trống = tháng kế tiếp sau dữ liệu)", "")
            submitted = st.form_submit_button("Dự đoán giá")
        if submitted:
            from src.predict import HomePredictor
            house = dict(district=district, area_m2=float(area), bedrooms=int(bedrooms),
                         bathrooms=int(bathrooms), floors=int(floors),
                         year_built=int(year_built), condition=condition,
                         distance_center_km=float(distance))
            result = HomePredictor().predict(house, month or None)
            HomePredictor().log_history(result)
            st.success(f"Giá ước tính: **{result['predicted_price_ty']:.2f} tỷ VND** "
                       f"(khoảng kỳ vọng {result['interval_ty'][0]:.2f} – {result['interval_ty'][1]:.2f} tỷ, "
                       f"±MAE {result['expected_error_plusminus_ty']} tỷ trên tập test)")
            for w in result["ood_warnings"]:
                st.error(f"⚠️ Ngoài phân phối huấn luyện: {w}")
            ex = result["explanation"]
            st.write(f"- Giá/m² quận {district}: mới nhất {ex['district_ppm2_latest_trieu']} triệu → "
                     f"dự báo {ex['district_ppm2_forecast_trieu']} triệu (tháng {result['target_month']})")
            st.write(f"- Hệ số căn nhà: {ex['house_factor']} (>1 tốt hơn mặt bằng mẫu, <1 thấp hơn)")
            st.write("- Đặc trưng ảnh hưởng nhất (mô hình hệ số):")
            st.json(ex["top_features"])
            st.caption(result["disclaimer"])

with tab_models:
    metrics = load_metrics()
    if metrics is None:
        st.warning("Chưa có kết quả huấn luyện.")
    else:
        rows = []
        for name, m in metrics["baseline_test_ty"].items():
            rows.append({"Mô hình": f"Baseline {name} (giá, tỷ)", **m})
        rows.append({"Mô hình": f"Hybrid {metrics['best_rnn'].upper()}+RF (giá, tỷ)", **metrics["hybrid_test_ty"]})
        for name, m in metrics["rnn_ppm2"].items():
            rows.append({"Mô hình": f"RNN {name.upper()} (giá/m², triệu)", **m["test_ppm2"]})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        for img, cap in [("district_price_trend.png", "Xu hướng giá theo quận"),
                         ("pred_vs_actual.png", "Dự đoán vs thực tế (hybrid, test)"),
                         ("model_compare.png", "So sánh MAE")]:
            p = os.path.join(ROOT, "outputs", "charts", img)
            if os.path.exists(p):
                st.image(p, caption=cap, use_container_width=True)

with tab_history:
    db = os.path.join(ROOT, "data", "prediction_history.db")
    if os.path.exists(db):
        with sqlite3.connect(db) as con:
            hist = pd.read_sql("SELECT * FROM predictions ORDER BY id DESC LIMIT 100", con)
        st.dataframe(hist, use_container_width=True)
    else:
        st.info("Chưa có lịch sử dự đoán. Hãy thực hiện một dự đoán ở tab Dự đoán.")
