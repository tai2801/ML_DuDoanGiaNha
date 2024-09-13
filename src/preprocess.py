"""Tiền xử lý + chia tập THEO THỜI GIAN (chống data leakage).

Nguyên tắc chống leakage:
- Mọi thống kê điền khuyết / cắt outlier đều FIT trên tập TRAIN, áp cho val/test.
- Scaler/encoder của mô hình cũng chỉ fit trên train (trong pipeline sklearn).
- Chia tập theo mốc tháng bán: train < val < test, không xáo trộn ngẫu nhiên.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NUMERIC_COLS = ["area_m2", "bedrooms", "bathrooms", "floors", "year_built", "distance_center_km"]
FEATURE_COLS = NUMERIC_COLS + ["age_years", "district", "condition"]


class Cleaner:
    """Điền khuyết + cắt outlier giá theo thống kê của TRAIN."""

    def __init__(self):
        self.medians: dict = {}
        self.condition_mode: str = "Tốt"
        self.log_price_bounds: tuple[float, float] = (-np.inf, np.inf)

    def fit(self, df: pd.DataFrame) -> "Cleaner":
        for c in NUMERIC_COLS:
            self.medians[c] = float(pd.to_numeric(df[c], errors="coerce").median())
        mode = df["condition"].mode(dropna=True)
        self.condition_mode = str(mode.iloc[0]) if len(mode) else "Tốt"
        logp = np.log(pd.to_numeric(df["price_ty"], errors="coerce").dropna())
        q1, q3 = logp.quantile(0.25), logp.quantile(0.75)
        iqr = q3 - q1
        self.log_price_bounds = (float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr))
        return self

    def transform(self, df: pd.DataFrame, clip_price: bool = True) -> pd.DataFrame:
        out = df.copy()
        for c in NUMERIC_COLS:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(self.medians[c])
        out["condition"] = out["condition"].fillna(self.condition_mode)
        out["bathrooms"] = out["bathrooms"].round().astype(int)
        out["year_built"] = out["year_built"].round().astype(int)
        out["age_years"] = out["sale_month"].str[:4].astype(int) - out["year_built"]
        out["age_years"] = out["age_years"].clip(lower=0)
        if clip_price and "price_ty" in out:
            lo, hi = self.log_price_bounds
            out["price_ty"] = np.exp(np.log(out["price_ty"]).clip(lo, hi))
        return out

    def to_dict(self) -> dict:
        return dict(medians=self.medians, condition_mode=self.condition_mode,
                    log_price_bounds=list(self.log_price_bounds))

    @classmethod
    def from_dict(cls, d: dict) -> "Cleaner":
        c = cls(); c.medians = d["medians"]; c.condition_mode = d["condition_mode"]
        c.log_price_bounds = tuple(d["log_price_bounds"]); return c


def temporal_split(df: pd.DataFrame, val_start: str, test_start: str):
    """Chia theo sale_month (YYYY-MM dạng chuỗi so sánh được)."""
    train = df[df["sale_month"] < val_start]
    val = df[(df["sale_month"] >= val_start) & (df["sale_month"] < test_start)]
    test = df[df["sale_month"] >= test_start]
    assert len(train) and len(val) and len(test), "Mốc chia khiến một tập rỗng"
    assert train["sale_month"].max() < val["sale_month"].min(), "Leakage: train/val chồng thời gian"
    assert val["sale_month"].max() < test["sale_month"].min(), "Leakage: val/test chồng thời gian"
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def train_stats(df: pd.DataFrame) -> dict:
    """Khoảng phân phối của TRAIN — dùng cho cảnh báo ngoài phân phối (OOD)."""
    stats = {}
    for c in ["area_m2", "bedrooms", "bathrooms", "floors", "year_built", "distance_center_km"]:
        s = pd.to_numeric(df[c], errors="coerce")
        stats[c] = dict(min=float(s.min()), max=float(s.max()),
                        p01=float(s.quantile(0.01)), p99=float(s.quantile(0.99)))
    stats["districts"] = sorted(df["district"].unique().tolist())
    return stats
