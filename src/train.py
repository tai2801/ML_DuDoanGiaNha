"""Pipeline huấn luyện HomePredict — chạy: python -m src.train [--config ...]

Kiến trúc KẾT HỢP (hybrid):
  1) Baseline đặc trưng: Linear Regression + Random Forest dự đoán trực tiếp
     log(giá) từ đặc trưng căn nhà + quận (so sánh độc lập).
  2) RNN (LSTM & GRU): dự báo giá/m2 tháng kế tiếp của quận từ chuỗi 12 tháng.
  3) Mô hình HỆ SỐ (factor): Random Forest học factor = giá_thật /
     (ppm2_thật_của_quận_tháng_đó * diện_tích) từ đặc trưng căn nhà.
  Dự đoán kết hợp:  price = ppm2_RNN(dự báo) * area * factor_RF.
Mọi scaler/encoder/thống kê làm sạch chỉ fit trên TRAIN (chia theo thời gian).
"""
from __future__ import annotations

import gzip
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import generate_data, models_rnn  # noqa: E402
from src.preprocess import (FEATURE_COLS, NUMERIC_COLS, Cleaner,  # noqa: E402
                            temporal_split, train_stats)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return dict(MAE=round(float(mean_absolute_error(y_true, y_pred)), 4),
                RMSE=round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
                R2=round(float(r2_score(y_true, y_pred)), 4),
                n=int(len(y_true)))


def load_config(path: str | None = None, overrides: dict | None = None) -> dict:
    path = path or os.path.join(ROOT, "config", "config.yaml")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if overrides:
        for k, v in overrides.items():
            cfg[k] = v
    return cfg


def build_baseline(kind: str, rf_params: dict) -> Pipeline:
    pre = ColumnTransformer([
        ("num", StandardScaler(), [c for c in FEATURE_COLS if c not in ("district", "condition")]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["district", "condition"]),
    ])
    reg = LinearRegression() if kind == "linear" else RandomForestRegressor(**rf_params)
    return Pipeline([("pre", pre), ("reg", reg)])


def run_training(cfg: dict, workdir: str = ROOT, log=print) -> dict:
    data_dir, model_dir = os.path.join(workdir, "data"), os.path.join(workdir, "models")
    out_dir = os.path.join(workdir, "outputs")
    os.makedirs(model_dir, exist_ok=True); os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "charts"), exist_ok=True)

    tx_path, series_path = os.path.join(data_dir, "transactions.csv"), os.path.join(data_dir, "district_monthly.csv")
    if not os.path.exists(tx_path) or cfg.get("regenerate"):
        generate_data.main(data_dir, cfg["seed"], cfg["n_transactions"])
    tx = pd.read_csv(tx_path, dtype={"sale_month": str})
    series = pd.read_csv(series_path, dtype={"month": str})

    raw_train, _, _ = temporal_split(tx, cfg["val_start"], cfg["test_start"])
    cleaner = Cleaner().fit(raw_train)
    tx_clean = cleaner.transform(tx)
    train, val, test = temporal_split(tx_clean, cfg["val_start"], cfg["test_start"])
    log(f"Split theo thời gian: train={len(train)} val={len(val)} test={len(test)}")

    ppm2_map = {(r.month, r.district): r.ppm2 for r in series.itertuples()}
    results: dict = {"data": dict(train=len(train), val=len(val), test=len(test),
                                  districts=int(tx["district"].nunique()),
                                  months=int(series["month"].nunique()))}

    # --- 1) Baseline đặc trưng (mục tiêu log giá) ---
    baselines = {}
    for kind in ("linear", "random_forest"):
        pipe = build_baseline(kind, cfg["rf_params"])
        pipe.fit(train[FEATURE_COLS], np.log(train["price_ty"]))
        pred = np.exp(pipe.predict(test[FEATURE_COLS]))
        baselines[kind] = metrics(test["price_ty"], pred)
        # RandomForest nén gzip cho nhẹ (21MB -> ~6MB); Linear giữ pickle thường
        opener = gzip.open if kind == "random_forest" else open
        with opener(os.path.join(model_dir, f"baseline_{kind}.pkl"), "wb") as f:
            pickle.dump(pipe, f)
        if kind == "random_forest":
            names = pipe.named_steps["pre"].get_feature_names_out()
            imp = pipe.named_steps["reg"].feature_importances_
            results["rf_feature_importance"] = {
                str(n): round(float(v), 4) for n, v in
                sorted(zip(names, imp), key=lambda x: -x[1])[:10]}
    results["baseline_test_ty"] = baselines
    log(f"Baseline test: {baselines}")

    # --- 2) Mô hình hệ số (factor) cho nhánh kết hợp ---
    def factor_of(df):
        base = df.apply(lambda r: ppm2_map[(r["sale_month"], r["district"])] * r["area_m2"] / 1000.0, axis=1)
        return (df["price_ty"] / base).to_numpy()
    factor_model = build_baseline("random_forest", cfg["rf_params"])
    factor_model.fit(train[FEATURE_COLS], factor_of(train))
    with gzip.open(os.path.join(model_dir, "factor_rf.pkl"), "wb") as f:
        pickle.dump(factor_model, f)

    # --- 3) RNN trên chuỗi giá theo quận ---
    from sklearn.preprocessing import MinMaxScaler
    series_by_district = models_rnn.build_windows(series)
    months_sorted = sorted(series["month"].unique())
    train_mask_values = series[series["month"] < cfg["rnn_train_end"]]["ppm2"].to_numpy()
    scaler = MinMaxScaler().fit(train_mask_values.reshape(-1, 1))
    W = cfg["rnn"]["window"]
    Xtr, ytr, _ = models_rnn.make_sequences(series_by_district, months_sorted, W, scaler, target_to=cfg["rnn_train_end"])
    Xval, yval, _ = models_rnn.make_sequences(series_by_district, months_sorted, W, scaler, target_from=cfg["rnn_train_end"], target_to=cfg["test_start"])
    Xtest, ytest, meta_test = models_rnn.make_sequences(series_by_district, months_sorted, W, scaler, target_from=cfg["test_start"])
    yval_real = scaler.inverse_transform(yval.reshape(-1, 1)).ravel()
    ytest_real = scaler.inverse_transform(ytest.reshape(-1, 1)).ravel()

    rnn_results, trained = {}, {}
    for kind in ("lstm", "gru"):
        model, hist = models_rnn.train_rnn(kind, Xtr, ytr, Xval, yval, W,
                                           cfg["rnn"]["units"], cfg["rnn"]["epochs"],
                                           cfg["rnn"]["batch_size"], cfg["seed"])
        val_pred = scaler.inverse_transform(model.predict(Xval, verbose=0)).ravel()
        test_pred = scaler.inverse_transform(model.predict(Xtest, verbose=0)).ravel()
        rnn_results[kind] = dict(val_ppm2=metrics(yval_real, val_pred),
                                 test_ppm2=metrics(ytest_real, test_pred),
                                 epochs_run=len(hist["loss"]))
        trained[kind] = (model, test_pred)
        model.save(os.path.join(model_dir, f"rnn_{kind}.keras"))
        log(f"RNN {kind}: val={rnn_results[kind]['val_ppm2']} test={rnn_results[kind]['test_ppm2']}")
    results["rnn_ppm2"] = rnn_results
    best_rnn = min(rnn_results, key=lambda k: rnn_results[k]["val_ppm2"]["RMSE"])
    results["best_rnn"] = best_rnn

    # --- 4) Đánh giá KẾT HỢP trên giao dịch test (one-step-ahead theo tháng bán) ---
    best_model = trained[best_rnn][0]
    test_pred_map = {}
    for (district, month), pred in zip(meta_test, trained[best_rnn][1]):
        test_pred_map[(district, month)] = float(pred)
    # Tháng test đầu tiên có thể thiếu trong meta nếu cửa sổ không đủ — dự báo trực tiếp
    hybrid_pred, hybrid_true = [], []
    for r in test.itertuples():
        key = (r.district, r.sale_month)
        if key not in test_pred_map:
            months_d, values_d = series_by_district[r.district]
            i = months_d.index(r.sale_month)
            test_pred_map[key] = float(models_rnn.forecast_next(
                best_model, scaler, values_d[:i], W, steps=1)[0])
        factor = float(factor_model.predict(pd.DataFrame([{
            c: getattr(r, c) for c in FEATURE_COLS}]))[0])
        hybrid_pred.append(test_pred_map[key] * r.area_m2 * factor / 1000.0)
        hybrid_true.append(r.price_ty)
    results["hybrid_test_ty"] = metrics(hybrid_true, hybrid_pred)
    log(f"Hybrid ({best_rnn}+RF factor) test: {results['hybrid_test_ty']}")

    # --- Lưu artifact + thống kê OOD + chuỗi lịch sử ---
    with open(os.path.join(model_dir, "scaler_rnn.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(model_dir, "cleaner.json"), "w", encoding="utf-8") as f:
        json.dump(cleaner.to_dict(), f, ensure_ascii=False, indent=1)
    stats = train_stats(train)
    with open(os.path.join(model_dir, "train_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    series.to_csv(os.path.join(model_dir, "series_history.csv"), index=False)
    test_out = test[["sale_month", "district", "area_m2", "price_ty"]].copy()
    test_out["hybrid_pred_ty"] = np.round(hybrid_pred, 3)
    test_out.to_csv(os.path.join(out_dir, "test_predictions.csv"), index=False)
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    with open(os.path.join(model_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=1, default=str)

    _charts(series, test_out, results, os.path.join(out_dir, "charts"), log)
    return results


def _charts(series, test_out, results, chart_dir, log):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # 1) Giá theo quận theo thời gian
        fig, ax = plt.subplots(figsize=(10, 4.5))
        for d, g in series.groupby("district"):
            ax.plot(pd.PeriodIndex(g["month"], freq="M").to_timestamp(), g["ppm2"], label=d, lw=1.2)
        ax.set_title("Giá trung bình tháng theo quận (dữ liệu mô phỏng)"); ax.set_ylabel("triệu VND/m²")
        ax.legend(ncol=4, fontsize=7); fig.tight_layout(); fig.savefig(f"{chart_dir}/district_price_trend.png", dpi=110); plt.close(fig)
        # 2) Dự đoán vs thực tế (hybrid, tập test)
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        ax.scatter(test_out["price_ty"], test_out["hybrid_pred_ty"], s=8, alpha=0.5)
        lim = [0, max(test_out["price_ty"].max(), test_out["hybrid_pred_ty"].max()) * 1.05]
        ax.plot(lim, lim, "r--", lw=1); ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("Giá thật (tỷ)"); ax.set_ylabel("Giá dự đoán hybrid (tỷ)")
        ax.set_title("Hybrid: dự đoán vs thực tế (test)"); fig.tight_layout()
        fig.savefig(f"{chart_dir}/pred_vs_actual.png", dpi=110); plt.close(fig)
        # 3) So sánh MAE các mô hình
        labels = ["Linear", "RandomForest", f"Hybrid({results['best_rnn'].upper()}+RF)"]
        maes = [results["baseline_test_ty"]["linear"]["MAE"],
                results["baseline_test_ty"]["random_forest"]["MAE"],
                results["hybrid_test_ty"]["MAE"]]
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.bar(labels, maes, color=["#8a9a5b", "#4a6fa5", "#c07a3a"])
        ax.set_ylabel("MAE (tỷ VND)"); ax.set_title("So sánh MAE trên tập test")
        fig.tight_layout(); fig.savefig(f"{chart_dir}/model_compare.png", dpi=110); plt.close(fig)
        log("Đã tạo 3 biểu đồ trong outputs/charts/")
    except Exception as e:  # biểu đồ là phụ, không làm hỏng training
        log(f"Cảnh báo: không tạo được biểu đồ: {e}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    overrides = {}
    if args.epochs:
        overrides["rnn"] = None
    cfg = load_config(args.config)
    if args.epochs:
        cfg["rnn"]["epochs"] = args.epochs
    results = run_training(cfg)
    print(json.dumps(results, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
