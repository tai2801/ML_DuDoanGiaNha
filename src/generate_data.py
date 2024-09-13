"""Sinh DATASET MÔ PHỎNG giao dịch nhà ở TP.HCM.

Toàn bộ dữ liệu là GIẢ LẬP bằng công thức + nhiễu ngẫu nhiên có seed cố định,
KHÔNG phải dữ liệu giao dịch thật. Giá/m2 nền theo quận chỉ mang tính minh họa
theo thứ hạng tương đối giữa các quận, không dùng để định giá thật.

Công thức giá (minh họa):
    price_ty = ppm2[district, month] * area * house_factor / 1000   (tỷ VND)
    house_factor = condition_factor * age_factor * tiện nghi * exp(noise)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ppm2 nền tháng 01/2019 (triệu VND/m2) + tăng trưởng %/tháng + khoảng cách
# trung tâm (km, tới Bitexco Q1) — số liệu MINH HỌA.
DISTRICTS = {
    "Quận 1":     dict(base=165.0, growth=0.0062, center_km=0.8,  weight=0.10),
    "Quận 3":     dict(base=128.0, growth=0.0058, center_km=2.5,  weight=0.09),
    "Phú Nhuận":  dict(base=112.0, growth=0.0055, center_km=4.0,  weight=0.09),
    "Bình Thạnh": dict(base=102.0, growth=0.0058, center_km=4.5,  weight=0.11),
    "Quận 2":     dict(base=118.0, growth=0.0068, center_km=6.0,  weight=0.10),
    "Quận 10":    dict(base=98.0,  growth=0.0050, center_km=4.2,  weight=0.08),
    "Quận 7":     dict(base=92.0,  growth=0.0060, center_km=7.5,  weight=0.10),
    "Tân Bình":   dict(base=88.0,  growth=0.0052, center_km=6.5,  weight=0.09),
    "Gò Vấp":     dict(base=68.0,  growth=0.0055, center_km=8.5,  weight=0.09),
    "Thủ Đức":    dict(base=62.0,  growth=0.0070, center_km=13.0, weight=0.08),
    "Bình Tân":   dict(base=58.0,  growth=0.0050, center_km=11.0, weight=0.05),
    "Hóc Môn":    dict(base=36.0,  growth=0.0045, center_km=19.0, weight=0.02),
}
CONDITION_FACTOR = {"Mới": 1.12, "Tốt": 1.03, "Trung bình": 0.95, "Cũ": 0.85}


def build_district_series(seed: int = 42, start: str = "2019-01", end: str = "2025-12") -> pd.DataFrame:
    """Chuỗi giá trung bình tháng (triệu VND/m2) theo quận."""
    rng = np.random.default_rng(seed)
    months = pd.period_range(start, end, freq="M")
    rows = []
    for district, cfg in DISTRICTS.items():
        level = cfg["base"]
        for t, m in enumerate(months):
            seasonal = 1.0 + 0.02 * np.sin(2 * np.pi * (t % 12) / 12.0)
            level *= (1.0 + cfg["growth"]) * float(np.exp(rng.normal(0, 0.006)))
            rows.append(dict(month=str(m), district=district,
                             ppm2=round(level * seasonal, 3)))
    return pd.DataFrame(rows)


def build_transactions(seed: int = 42, n: int = 3600,
                       series: pd.DataFrame | None = None) -> pd.DataFrame:
    """Sinh n giao dịch; cố ý chèn giá trị thiếu (~2%) và outlier (~1%)."""
    rng = np.random.default_rng(seed + 1)
    if series is None:
        series = build_district_series(seed)
    ppm2_map = {(r.month, r.district): r.ppm2 for r in series.itertuples()}
    months = sorted(series["month"].unique())
    # Năm gần đây có nhiều giao dịch hơn
    year_w = np.array([1.0 + 0.18 * (int(m[:4]) - 2019) for m in months])
    month_idx = rng.choice(len(months), size=n, p=year_w / year_w.sum())
    districts = list(DISTRICTS)
    w = np.array([DISTRICTS[d]["weight"] for d in districts]); w /= w.sum()
    d_idx = rng.choice(len(districts), size=n, p=w)

    area = np.clip(rng.lognormal(np.log(75), 0.45, n), 18, 400)
    bedrooms = np.clip(1 + rng.poisson(np.maximum(area - 15, 5) / 32), 1, 8)
    bathrooms = np.clip(1 + rng.poisson(np.maximum(area - 20, 5) / 45), 1, 6)
    floors = rng.choice([1, 2, 3, 4, 5], n, p=[0.22, 0.34, 0.24, 0.13, 0.07])
    year_built = rng.choice(range(1985, 2026), n,
                            p=np.linspace(0.4, 1.6, 41) / np.linspace(0.4, 1.6, 41).sum())
    conditions = rng.choice(list(CONDITION_FACTOR), n, p=[0.18, 0.42, 0.28, 0.12])

    rows = []
    for i in range(n):
        month, district = months[month_idx[i]], districts[d_idx[i]]
        sale_year = int(month[:4])
        yb = int(min(year_built[i], sale_year))
        age = sale_year - yb
        distance = max(0.3, DISTRICTS[district]["center_km"] + abs(rng.normal(0, 1.6)))
        factor = (CONDITION_FACTOR[conditions[i]]
                  * max(0.78, 1 - 0.006 * age)
                  * (1 + 0.015 * (bathrooms[i] - 2))
                  * (1 + 0.008 * (floors[i] - 2))
                  * float(np.exp(rng.normal(0, 0.10))))
        price_ty = ppm2_map[(month, district)] * area[i] * factor / 1000.0
        rows.append(dict(
            sale_month=month, district=district, area_m2=round(float(area[i]), 1),
            bedrooms=int(bedrooms[i]), bathrooms=int(bathrooms[i]), floors=int(floors[i]),
            year_built=yb, condition=conditions[i], distance_center_km=round(float(distance), 2),
            price_ty=round(float(price_ty), 3)))
    df = pd.DataFrame(rows)
    # Chèn giá trị thiếu có chủ đích (để kiểm thử tiền xử lý)
    for col, frac in [("year_built", 0.025), ("condition", 0.02), ("bathrooms", 0.015)]:
        mask = rng.random(n) < frac
        df.loc[mask, col] = np.nan
    # Chèn outlier giá (~1%)
    mask = rng.random(n) < 0.01
    df.loc[mask, "price_ty"] = (df.loc[mask, "price_ty"]
                                * rng.choice([0.45, 1.8, 2.1], mask.sum())).round(3)
    df["year_built"] = df["year_built"].astype("Float64")
    df["bathrooms"] = df["bathrooms"].astype("Float64")
    return df.sort_values(["sale_month", "district"]).reset_index(drop=True)


def main(out_dir: str = "data", seed: int = 42, n: int = 3600) -> None:
    import os
    os.makedirs(out_dir, exist_ok=True)
    series = build_district_series(seed)
    tx = build_transactions(seed, n, series)
    series.to_csv(f"{out_dir}/district_monthly.csv", index=False)
    tx.to_csv(f"{out_dir}/transactions.csv", index=False)
    print(f"district_monthly: {len(series)} dòng | transactions: {len(tx)} dòng -> {out_dir}/")


if __name__ == "__main__":
    import sys
    main(*(sys.argv[1:] or []))
