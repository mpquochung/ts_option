# dataset.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset

def load_and_prepare_data(cfg: Dict):
    """
    Load CSV, tạo feature, target t+2, split train/test theo tháng,
    và scale theo train.
    """
    ds_cfg = cfg["dataset"]
    csv_path = ds_cfg["csv_path"]
    time_col = ds_cfg["time_col"]
    target_col = ds_cfg["target_col"]
    horizon = ds_cfg["horizon"]

    df = pd.read_csv(csv_path)
    # parse time
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).reset_index(drop=True)

    # ==== merge external features nếu có ====
    if ds_cfg.get("external_features", {}).get("enabled", False):
        for ef in ds_cfg["external_features"]["files"]:
            ext = pd.read_csv(ef["path"])
            ext[ef["join_on"]] = pd.to_datetime(ext[ef["join_on"]])
            df = df.merge(ext[[ef["join_on"]] + ef["columns"]],
                          left_on=time_col, right_on=ef["join_on"],
                          how="left")
            df = df.drop(columns=[ef["join_on"]])

    # ==== chọn base feature cols ====
    base_feature_cols: List[str] = ds_cfg["base_feature_cols"].copy()

    # ==== engineered features ====
    eng_cfg = ds_cfg["engineered_features"]
    if eng_cfg.get("use_returns", False):
        df["return_stock"] = df["stock_price"].pct_change()
        df["return_option"] = df[target_col].pct_change()
        base_feature_cols += ["return_stock", "return_option"]

    if eng_cfg.get("use_spread_bs", False):
        df["spread_bs"] = df[target_col] - df["bs_price"]
        base_feature_cols += ["spread_bs"]

    if eng_cfg.get("use_rolling", False):
        w = eng_cfg.get("rolling_window", 5)
        df["vol_stock_rolling"] = (
            df["stock_price"].pct_change().rolling(w).std()
        )
        df["vol_option_rolling"] = (
            df[target_col].pct_change().rolling(w).std()
        )
        df["ma_stock"] = df["stock_price"].rolling(w).mean()
        df["ma_option"] = df[target_col].rolling(w).mean()
        base_feature_cols += [
            "vol_stock_rolling",
            "vol_option_rolling",
            "ma_stock",
            "ma_option",
        ]

    # ==== label: target t+2 ====
    df["target_tplus2"] = df[target_col].shift(-horizon)

    # drop các dòng có NaN do rolling/shift
    df = df.dropna(subset=base_feature_cols + ["target_tplus2"]).reset_index(drop=True)

    # ==== gán month index theo thời gian ====
    # unique Year-Month theo thứ tự thời gian
    df["year_month"] = df[time_col].dt.to_period("M").astype(str)
    unique_months = df["year_month"].drop_duplicates().tolist()

    train_months_n = ds_cfg["split"]["train_months"]
    test_month_idx = ds_cfg["split"]["test_month_index"]

    if len(unique_months) < test_month_idx:
        raise ValueError(
            f"Data chỉ có {len(unique_months)} tháng, nhưng config yêu cầu "
            f"test_month_index={test_month_idx}"
        )

    train_months = unique_months[:train_months_n]
    test_month = unique_months[test_month_idx - 1]  # index 1-based

    train_df = df[df["year_month"].isin(train_months)].copy()
    test_df = df[df["year_month"] == test_month].copy()

    # có thể thêm val tách từ train_df nếu muốn

    # ==== scale theo train ====
    X_train_raw = train_df[base_feature_cols].values
    y_train_raw = train_df["target_tplus2"].values.reshape(-1, 1)

    X_test_raw = test_df[base_feature_cols].values
    y_test_raw = test_df["target_tplus2"].values.reshape(-1, 1)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train = scaler_X.fit_transform(X_train_raw)
    y_train = scaler_y.fit_transform(y_train_raw)

    X_test = scaler_X.transform(X_test_raw)
    y_test = scaler_y.transform(y_test_raw)

    meta = {
        "feature_cols": base_feature_cols,
        "scaler_X": scaler_X,
        "scaler_y": scaler_y,
        "train_months": train_months,
        "test_month": test_month,
        "test_dates": test_df[time_col].values,
    }

    return (X_train, y_train, X_test, y_test, meta)


class TimeSeriesDataset(Dataset):
    """
    Dataset tạo sliding window:
    - Input: sequence length = window_size
    - Label: target tại end step (t), là giá option ở t+2 (do đã shift trong df)
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, window_size: int):
        assert len(X) == len(y)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.window_size = window_size

    def __len__(self):
        # số cửa sổ hợp lệ
        return len(self.X) - self.window_size + 1

    def __getitem__(self, idx):
        # window: [idx, idx+window_size)
        x_seq = self.X[idx:idx + self.window_size]  # (L, F)
        y_target = self.y[idx + self.window_size - 1]  # scalar (đã scaled)
        return x_seq, y_target
