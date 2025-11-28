# eval.py
from typing import Dict, List, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader

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
