"""Test HomePredict: tiền xử lý, chống leakage, OOD, pipeline end-to-end rút gọn."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.generate_data import build_district_series, build_transactions  # noqa: E402
from src.ood import check_ood  # noqa: E402
from src.preprocess import Cleaner, temporal_split, train_stats  # noqa: E402


@pytest.fixture(scope="module")
def data():
    series = build_district_series(42)
    return build_transactions(42, 800, series), series


def test_generator_shape_and_missing(data):
    tx, series = data
    assert len(tx) == 800 and len(series) == 12 * 84
    assert tx["price_ty"].gt(0).all()
    # Dữ liệu thô cố ý có giá trị thiếu để kiểm thử tiền xử lý
    assert tx["year_built"].isna().sum() > 0


def test_cleaner_imputes_and_clips(data):
    tx, _ = data
    cleaner = Cleaner().fit(tx[tx["sale_month"] < "2023-07"])
    out = cleaner.transform(tx)
    assert out[["area_m2", "year_built", "bathrooms"]].isna().sum().sum() == 0
    assert out["condition"].isna().sum() == 0
    logp = np.log(out["price_ty"])
    lo, hi = cleaner.log_price_bounds
    assert logp.max() <= hi + 1e-9 and logp.min() >= lo - 1e-9


def test_temporal_split_no_leakage(data):
    tx, _ = data
    train, val, test = temporal_split(tx, "2023-07", "2024-07")
    assert train["sale_month"].max() < val["sale_month"].min() < test["sale_month"].min()
    # Cleaner/scaler chỉ được fit trên train: trung vị train != trung vị toàn cục (kiểm tra cơ chế)
    cleaner = Cleaner().fit(train)
    assert cleaner.medians["area_m2"] == float(train["area_m2"].median())


def test_ood_detector(data):
    tx, _ = data
    stats = train_stats(Cleaner().fit(tx).transform(tx))
    normal = dict(district="Quận 7", area_m2=80, bedrooms=3, bathrooms=2,
                  floors=2, year_built=2015, distance_center_km=7.0)
    assert check_ood(normal, stats) == []
    weird = dict(normal, area_m2=900)  # ngoài max train (~400)
    assert any("Diện tích" in w for w in check_ood(weird, stats))
    alien = dict(normal, district="Quận Không Có")
    assert any("KHÔNG có trong dữ liệu" in w for w in check_ood(alien, stats))


def test_pipeline_end_to_end_tiny(tmp_path):
    """Train rút gọn (epochs=3, ít giao dịch) rồi dự đoán thật qua HomePredictor."""
    import shutil
    from src.train import run_training, load_config
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    work = tmp_path / "proj"
    shutil.copytree(root, work, ignore=shutil.ignore_patterns(
        "models", "outputs", "data", "__pycache__", ".venv"))
    (work / "data").mkdir(exist_ok=True)
    cfg = load_config(str(work / "config" / "config.yaml"))
    cfg["n_transactions"] = 700
    cfg["rnn"]["epochs"] = 3
    cfg["rf_params"]["n_estimators"] = 40
    results = run_training(cfg, workdir=str(work), log=lambda *a: None)
    for key in ("baseline_test_ty", "rnn_ppm2", "hybrid_test_ty"):
        assert key in results and results[key]
    assert results["hybrid_test_ty"]["MAE"] > 0
    sys.path.insert(0, str(work))
    for mod in list(sys.modules):
        if mod.startswith("src"):
            del sys.modules[mod]
    from src.predict import HomePredictor
    p = HomePredictor(model_dir=str(work / "models"))
    out = p.predict(dict(district="Quận 7", area_m2=85, bedrooms=3, bathrooms=2,
                         floors=3, year_built=2018, condition="Tốt",
                         distance_center_km=7.2))
    assert out["predicted_price_ty"] > 0
    assert out["interval_ty"][0] <= out["predicted_price_ty"] <= out["interval_ty"][1]
    assert out["ood_warnings"] == []
