"""Mô hình RNN (LSTM/GRU) trên CHUỖI GIÁ TRUNG BÌNH THÁNG THEO QUẬN.

Bài toán chuỗi: cho 12 tháng giá/m2 gần nhất của MỘT quận -> dự báo giá/m2
tháng kế tiếp của quận đó. Scaler (MinMax) chỉ fit trên giai đoạn train.
Đánh giá one-step-ahead trên giai đoạn test bằng cửa sổ lịch sử thật —
cách đánh giá chuẩn cho dự báo chuỗi thời gian.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def build_windows(series: pd.DataFrame, window: int = 12):
    """Trả về dict district -> mảng ppm2 theo tháng (đã sắp xếp)."""
    out = {}
    for district, g in series.groupby("district"):
        g = g.sort_values("month")
        out[district] = (g["month"].tolist(), g["ppm2"].to_numpy(dtype=float))
    return out


def make_sequences(series_by_district: dict, months_sorted: list[str],
                   window: int, scaler: MinMaxScaler,
                   target_from: str | None = None, target_to: str | None = None):
    """Tạo (X, y, meta) — mỗi mẫu: window tháng liên tiếp của 1 quận -> tháng kế.

    target_from/target_to: chỉ lấy mẫu có tháng đích trong [from, to).
    """
    X, y, meta = [], [], []
    for district, (months, values) in series_by_district.items():
        scaled = scaler.transform(values.reshape(-1, 1)).ravel()
        idx = {m: i for i, m in enumerate(months)}
        for j, m in enumerate(months_sorted):
            if m not in idx:
                continue
            ti = idx[m]
            if ti < window:
                continue
            if target_from and m < target_from:
                continue
            if target_to and m >= target_to:
                continue
            X.append(scaled[ti - window:ti])
            y.append(scaled[ti])
            meta.append((district, m))
    return (np.array(X, dtype=np.float32).reshape(-1, window, 1),
            np.array(y, dtype=np.float32), meta)


def build_keras_model(kind: str, window: int, units: int = 32, lr: float = 0.005):
    import tensorflow as tf
    layer = tf.keras.layers.LSTM if kind == "lstm" else tf.keras.layers.GRU
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(window, 1)),
        layer(units),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse")
    return model


def train_rnn(kind: str, Xtr, ytr, Xval, yval, window: int, units: int = 32,
              epochs: int = 80, batch_size: int = 16, seed: int = 42):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    model = build_keras_model(kind, window, units)
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10,
                                          restore_best_weights=True)
    hist = model.fit(Xtr, ytr, validation_data=(Xval, yval), epochs=epochs,
                     batch_size=batch_size, verbose=0, callbacks=[es])
    return model, hist.history


def forecast_next(model, scaler: MinMaxScaler, history_values: np.ndarray,
                  window: int, steps: int = 1) -> np.ndarray:
    """Dự báo đệ quy `steps` tháng kế tiếp từ chuỗi giá trị thật gần nhất."""
    hist = list(np.asarray(history_values, dtype=float))
    preds = []
    for _ in range(steps):
        x = scaler.transform(np.array(hist[-window:]).reshape(-1, 1)).ravel()
        p = model.predict(x.reshape(1, window, 1).astype(np.float32), verbose=0)[0, 0]
        val = float(scaler.inverse_transform([[p]])[0, 0])
        preds.append(val)
        hist.append(val)
    return np.array(preds)
