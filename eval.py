from typing import Dict, List, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from typing import Optional

def mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true, y_pred):
    return float(np.sqrt(mse(y_true, y_pred)))


METRIC_FUNCS = {
    "mse": mse,
    "mae": mae,
    "rmse": rmse,
}


def evaluate_model(
    cfg: Dict,
    model: torch.nn.Module,
    data_loader: DataLoader,
    scaler_y,
) -> Dict[str, float]:
    device = cfg["training"]["device"]
    model.eval()
    all_preds_scaled = []
    all_targets_scaled = []

    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred = model(x_batch)

            all_preds_scaled.append(y_pred.cpu().numpy())
            all_targets_scaled.append(y_batch.cpu().numpy())

    y_pred_scaled = np.concatenate(all_preds_scaled, axis=0)  # (N,1)
    y_true_scaled = np.concatenate(all_targets_scaled, axis=0)  # (N,1)

    # inverse scale về đơn vị gốc giá option
    y_pred = scaler_y.inverse_transform(y_pred_scaled).ravel()
    y_true = scaler_y.inverse_transform(y_true_scaled).ravel()

    metrics_to_use: List[str] = cfg["evaluation"]["metrics"]
    results = {}
    for m in metrics_to_use:
        func = METRIC_FUNCS[m]
        results[m] = func(y_true, y_pred)

    return results, (y_true, y_pred)


def rollout_predictions(
    cfg: Dict,
    model: torch.nn.Module,
    data_loader: DataLoader,
    scaler_y,
):
    """
    Chạy infer cho toàn bộ data_loader (thường là test_loader),
    trả về (y_true, y_pred) đã inverse scale về đơn vị gốc.
    """
    device = cfg["training"]["device"]
    model.eval()

    all_preds_scaled = []
    all_targets_scaled = []

    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = model(x_batch)

            all_preds_scaled.append(y_pred.cpu().numpy())
            all_targets_scaled.append(y_batch.cpu().numpy())

    y_pred_scaled = np.concatenate(all_preds_scaled, axis=0)   # (N, 1)
    y_true_scaled = np.concatenate(all_targets_scaled, axis=0) # (N, 1)

    y_pred = scaler_y.inverse_transform(y_pred_scaled).ravel()
    y_true = scaler_y.inverse_transform(y_true_scaled).ravel()

    return y_true, y_pred


def plot_time_series(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates: Optional[np.ndarray] = None,
    title: str = "Option price t+2: true vs predicted",
    save_path: Optional[str] = None,
):
    """
    Vẽ time series của giá thực và giá dự đoán.
    - nếu có dates: dùng làm trục x
    - nếu không: dùng index 0..N-1
    """
    assert len(y_true) == len(y_pred)

    if dates is None:
        x = np.arange(len(y_true))
    else:
        x = dates

    plt.figure(figsize=(12, 5))
    plt.plot(x, y_true, label="True", linewidth=1.5)
    plt.plot(x, y_pred, label="Predicted", linewidth=1.5, linestyle="--")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Option price (t+2)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # nếu x là datetime, xoay nhãn cho dễ nhìn
    if dates is not None:
        plt.xticks(rotation=45)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        print(f"Saved time series plot to {save_path}")
    else:
        plt.show()


def infer_last_window(
    cfg: Dict,
    model: torch.nn.Module,
    X_raw: np.ndarray,
    scaler_X,
    scaler_y,
    window_size: int,
) -> float:
    """
    Infer: lấy window_size điểm cuối từ X_raw (đã là feature ở domain gốc),
    scale rồi predict 1 giá trị t+2.
    """
    device = cfg["training"]["device"]

    # X_raw: (N, F)
    assert X_raw.shape[0] >= window_size
    last_window = X_raw[-window_size:, :]  # (L, F)

    X_scaled = scaler_X.transform(last_window)
    X_scaled = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0)  # (1, L, F)

    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X_scaled.to(device)).cpu().numpy()  # (1, 1)

    y_pred = scaler_y.inverse_transform(y_pred_scaled)[0, 0]
    return float(y_pred)

def load_model_from_checkpoint(
    model,
    checkpoint_path: str,
    device="cpu"
):
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    return model
