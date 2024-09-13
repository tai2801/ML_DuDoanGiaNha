"""Phát hiện dữ liệu NGOÀI PHÂN PHỐI huấn luyện (Out-Of-Distribution).

Input nằm ngoài khoảng [min, max] (hoặc ngoài [p01, p99] -> cảnh báo nhẹ)
của tập train thì gắn cảnh báo. Dự đoán vẫn được trả kèm cảnh báo để người
dùng tự cân nhắc — hệ thống không giấu kết quả nhưng cũng không tỏ ra chắc chắn.
"""
from __future__ import annotations

FIELD_LABELS = {
    "area_m2": "Diện tích", "bedrooms": "Số phòng ngủ", "bathrooms": "Số phòng tắm",
    "floors": "Số tầng", "year_built": "Năm xây dựng",
    "distance_center_km": "Khoảng cách tới trung tâm",
}


def check_ood(house: dict, stats: dict) -> list[str]:
    warnings: list[str] = []
    if house.get("district") not in stats.get("districts", []):
        warnings.append(
            f"Quận '{house.get('district')}' KHÔNG có trong dữ liệu huấn luyện — "
            "kết quả không đáng tin, cần dữ liệu bổ sung.")
        return warnings
    for col, label in FIELD_LABELS.items():
        if col not in house or house[col] is None or col not in stats:
            continue
        v, s = float(house[col]), stats[col]
        if v < s["min"] or v > s["max"]:
            warnings.append(
                f"{label} = {v:g} nằm NGOÀI toàn bộ khoảng huấn luyện "
                f"[{s['min']:g}, {s['max']:g}] — ngoại suy, độ tin cậy thấp.")
        elif v < s["p01"] or v > s["p99"]:
            warnings.append(
                f"{label} = {v:g} ở vùng hiếm gặp trong dữ liệu huấn luyện "
                f"(ngoài phân vị 1–99%: [{s['p01']:g}, {s['p99']:g}]) — nên thận trọng.")
    return warnings
