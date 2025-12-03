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
    window_size = ds_cfg["window_size"]

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

    # ==== label: target ====
    df["target"] = df[target_col].shift(-horizon)

    # drop các dòng có NaN do rolling/shift
    df = df.dropna(subset=base_feature_cols + ["target"]).reset_index(drop=True)
    # ==== gán month index theo thời gian ====
    # unique Year-Month theo thứ tự thời gian
    df["year_month"] = df[time_col].dt.to_period("M").astype(str)
    unique_months = df["year_month"].drop_duplicates().tolist()

    df['sigma_N'] = (df['sigma'] / df['stock_price']) * np.sqrt(202) * (1 - df['maturity'])


    train_months_n = ds_cfg["split"]["train_months"]
    test_month_idx = ds_cfg["split"]["test_month_index"]

    if len(unique_months) < test_month_idx:
        raise ValueError(
            f"Data chỉ có {len(unique_months)} tháng, nhưng config yêu cầu "
            f"test_month_index={test_month_idx}"
        )

    train_months = unique_months[:train_months_n]
    test_month = unique_months[test_month_idx - 1]  # index 1-based

    print(f"Unique months: {unique_months}")
    print(f"Train months: {train_months}")

    train_df = df[df["year_month"].isin(train_months)].copy()
    
    # Test_df bao gồm tháng test và window_size dòng trước đó để tạo sequence
    test_month_start_idx = df[df["year_month"] == test_month].index[0]
    test_month_end_idx = df[df["year_month"] == test_month].index[-1]
    # Lấy thêm window_size dòng trước test month để có đủ context cho sequence đầu tiên
    start_idx = max(0, test_month_start_idx - window_size)
    test_df = df.iloc[start_idx:test_month_end_idx + 1].copy()

    # có thể thêm val tách từ train_df nếu muốn
    base_feature_cols = [col for col in base_feature_cols if col not in ['sigma_N']]

    # ==== scale theo train ====
    X_train_raw = train_df[base_feature_cols]
    y_train_raw = train_df["target"]
    y_train_raw = pd.DataFrame(y_train_raw.values, columns=["target"])

    X_test_raw = test_df[base_feature_cols]
    y_test_raw = test_df["target"]
    y_test_raw = pd.DataFrame(y_test_raw.values, columns=["target"])

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train_raw)
    X_train = pd.DataFrame(X_train_scaled, columns=base_feature_cols, index=X_train_raw.index)
    
    y_train_scaled = scaler_y.fit_transform(y_train_raw)
    y_train = pd.DataFrame(y_train_scaled, columns=["target"], index=y_train_raw.index)
    
    X_test_scaled = scaler_X.transform(X_test_raw)
    X_test = pd.DataFrame(X_test_scaled, columns=base_feature_cols, index=X_test_raw.index)
    
    y_test_scaled = scaler_y.transform(y_test_raw)
    y_test = pd.DataFrame(y_test_scaled, columns=["target"], index=y_test_raw.index)

    X_train['sigma_N'] = train_df['sigma_N'].values
    X_test['sigma_N'] = test_df['sigma_N'].values

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

    def __init__(self, X: pd.DataFrame, y: pd.DataFrame, input_size: int, output_size: int, random_noise: bool = True, train: bool = True):
        assert len(X) == len(y)
        self.X = X
        self.y = y
        self.input_size = input_size
        self.output_size = output_size
        self.maturity_idx = -1
        self.sigma_idx = -2
        self.random_noise = random_noise
        self.train = train

        print(len(X))

    def __len__(self):
        # số cửa sổ hợp lệ
        return len(self.X) - self.input_size - self.output_size + 1

    def __getitem__(self, idx):
        # window: [idx, idx+window_size)
        x_seq = self.X.iloc[idx:idx + self.input_size]  # (L, F)
        x_seq = x_seq.drop(columns=['sigma_N'])
        y_target = self.y.iloc[idx + self.input_size : idx + self.input_size  + self.output_size]  # scalar (đã scaled)
        
        x_seq = torch.tensor(x_seq.values, dtype=torch.float32)
        y_target = torch.tensor(y_target.values, dtype=torch.float32).squeeze(-1)  # (output_size,)

        # Thêm random noise vào feature sigma (return_stock) trong training
        # normalized sigma = sigma / stock_price
        if self.train and self.random_noise:
            x_sigma  = self.X.iloc[idx + self.input_size : idx + self.input_size  + self.output_size]['sigma_N']
                        
            sigma_value = torch.tensor(x_sigma.values, dtype=torch.float32)
            # N(1, lambda * sigma_value)
            lambda_ = 0.5
            noise = torch.normal(mean=1.0, std=lambda_ * sigma_value)
            y_target = y_target * noise


        return x_seq, y_target
