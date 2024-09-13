"""Dịch vụ dự đoán giá nhà (hybrid RNN + hệ số RF) + CLI.

Kết quả luôn kèm:
- khoảng sai số kỳ vọng (± MAE của mô hình hybrid trên tập test) — KHÔNG phải
  con số "chính xác tuyệt đối";
- cảnh báo ngoài phân phối (OOD) nếu input lạ so với dữ liệu huấn luyện;
- giải thích yếu tố ảnh hưởng: giá/m2 dự báo của quận, hệ số căn nhà,
  độ quan trọng đặc trưng của mô hình hệ số.

CLI:  python -m src.predict --district "Quận 7" --area 85 --bedrooms 3 \
        --bathrooms 2 --floors 3 --year-built 2018 --condition Tốt \
        --distance 7.2 [--month 2024-11] [--no-log]
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sqlite3
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import models_rnn  # noqa: E402
from src.ood import check_ood  # noqa: E402
from src.preprocess import FEATURE_COLS  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class HomePredictor:
    def __init__(self, model_dir: str | None = None):
        import tensorflow as tf  # noqa: F401  (đảm bảo runtime sẵn sàng)
        from tensorflow import keras
        self.dir = model_dir or os.path.join(ROOT, "models")
        with open(os.path.join(self.dir, "train_config.json"), encoding="utf-8") as f:
            self.cfg = json.load(f)
        with open(os.path.join(ROOT, "outputs", "metrics.json"), encoding="utf-8") as f:
            self.metrics = json.load(f)
        self.best_rnn = self.metrics["best_rnn"]
        self.window = self.cfg["rnn"]["window"]
        self.model = keras.models.load_model(
            os.path.join(self.dir, f"rnn_{self.best_rnn}.keras"))
        with open(os.path.join(self.dir, "scaler_rnn.pkl"), "rb") as f:
            self.scaler = pickle.load(f)
        import gzip
        with gzip.open(os.path.join(self.dir, "factor_rf.pkl"), "rb") as f:
            self.factor_model = pickle.load(f)
        with open(os.path.join(self.dir, "train_stats.json"), encoding="utf-8") as f:
            self.stats = json.load(f)
        self.series = pd.read_csv(os.path.join(self.dir, "series_history.csv"),
                                  dtype={"month": str})
        self.series_by_district = models_rnn.build_windows(self.series)
        self.last_month = self.series["month"].max()

    def _forecast_ppm2(self, district: str, target_month: str) -> tuple[float, int]:
        months, values = self.series_by_district[district]
        if target_month in months:  # tháng đã có trong lịch sử: dự báo one-step từ quá khứ thật
            i = months.index(target_month)
            if i >= self.window:
                return float(models_rnn.forecast_next(
                    self.model, self.scaler, values[:i], self.window, 1)[0]), 1
        # Tháng tương lai: dự báo đệ quy từ tháng cuối cùng có dữ liệu
        last = pd.Period(self.last_month, freq="M")
        target = pd.Period(target_month, freq="M")
        steps = (target.year - last.year) * 12 + (target.month - last.month)
        if steps < 1:
            steps = 1
        preds = models_rnn.forecast_next(self.model, self.scaler, values, self.window, steps)
        return float(preds[-1]), steps

    def predict(self, house: dict, target_month: str | None = None) -> dict:
        target_month = target_month or str(pd.Period(self.last_month, freq="M") + 1)
        warnings = check_ood(house, self.stats)
        ppm2, steps = self._forecast_ppm2(house["district"], target_month)
        feat = {c: house.get(c) for c in FEATURE_COLS}
        feat["age_years"] = int(target_month[:4]) - int(house["year_built"])
        factor = float(self.factor_model.predict(pd.DataFrame([feat]))[0])
        price = ppm2 * float(house["area_m2"]) * factor / 1000.0
        mae = self.metrics["hybrid_test_ty"]["MAE"]
        latest_ppm2 = float(self.series[
            (self.series["district"] == house["district"])
            & (self.series["month"] == self.last_month)]["ppm2"].iloc[0])
        return dict(
            input=house, target_month=target_month,
            predicted_price_ty=round(price, 3),
            expected_error_plusminus_ty=mae,
            interval_ty=[round(max(price - mae, 0), 3), round(price + mae, 3)],
            model=f"Hybrid {self.best_rnn.upper()} (chuỗi giá quận) + RandomForest (hệ số căn nhà)",
            explanation=dict(
                district_ppm2_latest_trieu=round(latest_ppm2, 2),
                district_ppm2_forecast_trieu=round(ppm2, 2),
                forecast_steps_months=steps,
                house_factor=round(factor, 3),
                note=("Giá = giá/m² dự báo của quận × diện tích × hệ số căn nhà. "
                      "Hệ số > 1: nhà tốt hơn mặt bằng mẫu (mới, ít tuổi, nhiều tiện nghi); < 1: ngược lại."),
                top_features=self.metrics.get("rf_feature_importance", {})),
            ood_warnings=warnings,
            disclaimer=("Ước tính từ dữ liệu MÔ PHỎNG phục vụ học tập — không dùng "
                        "để định giá/mua bán thật. Sai số kỳ vọng ±MAE trên tập test."))

    def log_history(self, result: dict, db_path: str | None = None) -> None:
        db_path = db_path or os.path.join(ROOT, "data", "prediction_history.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        with sqlite3.connect(db_path) as con:
            con.execute("""CREATE TABLE IF NOT EXISTS predictions(
                id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, district TEXT,
                area_m2 REAL, target_month TEXT, predicted_price_ty REAL,
                ood_warnings TEXT)""")
            h = result["input"]
            con.execute("INSERT INTO predictions VALUES (NULL,?,?,?,?,?,?)",
                        (datetime.now().isoformat(timespec="seconds"), h["district"],
                         h["area_m2"], result["target_month"],
                         result["predicted_price_ty"],
                         json.dumps(result["ood_warnings"], ensure_ascii=False)))


def main():
    ap = argparse.ArgumentParser(description="Dự đoán giá nhà HomePredict")
    ap.add_argument("--district", required=True)
    ap.add_argument("--area", type=float, required=True, dest="area_m2")
    ap.add_argument("--bedrooms", type=int, default=3)
    ap.add_argument("--bathrooms", type=int, default=2)
    ap.add_argument("--floors", type=int, default=2)
    ap.add_argument("--year-built", type=int, default=2015, dest="year_built")
    ap.add_argument("--condition", default="Tốt",
                    choices=["Mới", "Tốt", "Trung bình", "Cũ"])
    ap.add_argument("--distance", type=float, default=5.0, dest="distance_center_km")
    ap.add_argument("--month", default=None)
    ap.add_argument("--no-log", action="store_true")
    a = ap.parse_args()
    house = dict(district=a.district, area_m2=a.area_m2, bedrooms=a.bedrooms,
                 bathrooms=a.bathrooms, floors=a.floors, year_built=a.year_built,
                 condition=a.condition, distance_center_km=a.distance_center_km)
    predictor = HomePredictor()
    result = predictor.predict(house, a.month)
    if not a.no_log:
        predictor.log_history(result)
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
