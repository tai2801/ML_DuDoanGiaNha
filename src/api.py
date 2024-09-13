"""FastAPI cho HomePredict: GET /health, GET /metrics, POST /predict.

Chạy: uvicorn src.api:app --port 8000  (cần đã train trước: python -m src.train)
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from src.predict import HomePredictor  # noqa: E402

app = FastAPI(title="HomePredict API", version="1.0")
_predictor: HomePredictor | None = None


def get_predictor() -> HomePredictor:
    global _predictor
    if _predictor is None:
        _predictor = HomePredictor()
    return _predictor


class HouseIn(BaseModel):
    district: str
    area_m2: float
    bedrooms: int = 3
    bathrooms: int = 2
    floors: int = 2
    year_built: int = 2015
    condition: str = "Tốt"
    distance_center_km: float = 5.0
    target_month: str | None = None


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _predictor is not None}


@app.get("/metrics")
def metrics():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "outputs", "metrics.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@app.post("/predict")
def predict(house: HouseIn):
    p = get_predictor()
    data = house.model_dump()
    target = data.pop("target_month")
    if data["district"] not in p.stats["districts"]:
        raise HTTPException(400, f"Quận chưa có trong dữ liệu huấn luyện: {data['district']}")
    result = p.predict(data, target)
    p.log_history(result)
    return result
