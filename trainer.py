from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader

from dataloader import TimeSeriesDataset, load_and_prepare_data
from models import BaseRNNModel, Seq2SeqModel

import os

def build_dataloaders(cfg: Dict):
    ds_cfg = cfg["dataset"]
    window_size = ds_cfg["window_size"]
    output_size = ds_cfg["output_window"]
    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["training"]["num_workers"]
    shuffle_train = cfg["training"]["shuffle_train"]

    X_train, y_train, X_test, y_test, meta = load_and_prepare_data(cfg)

    train_ds = TimeSeriesDataset(X_train, y_train, window_size, output_size)
    test_ds = TimeSeriesDataset(X_test, y_test, window_size, output_size)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    return train_loader, test_loader, meta


def build_model(cfg: Dict, input_dim: int) -> BaseRNNModel:
    m_cfg = cfg["model"]
    d_cfg = cfg["dataset"]

    input_size = d_cfg["window_size"]
    output_size = d_cfg["output_window"]

    if output_size == 1:
        model = BaseRNNModel(
            input_size=input_dim,
            hidden_size=m_cfg["hidden_size"],
            num_layers=m_cfg["num_layers"],
            dropout=m_cfg["dropout"],
            rnn_type=m_cfg["type"],
            bidirectional=m_cfg["bidirectional"],
        )
    else:
        model = Seq2SeqModel(
            input_size=input_dim,
            hidden_size=m_cfg["hidden_size"],
            num_layers=m_cfg["num_layers"],
            dropout=m_cfg["dropout"],
            rnn_type=m_cfg["type"],
            bidirectional=m_cfg["bidirectional"],
        )
    return model


def train_model(
    cfg: Dict,
    model: torch.nn.Module,
    train_loader: DataLoader,
) -> torch.nn.Module:
    device = cfg["training"]["device"]
    lr = cfg["training"]["learning_rate"]
    weight_decay = cfg["training"]["weight_decay"]
    num_epochs = cfg["training"]["num_epochs"]
    print_every = cfg["logging"]["print_every"]
    output_size = cfg["dataset"]["output_window"]

    ckpt_cfg = cfg.get("checkpoint", {})
    ckpt_enabled = ckpt_cfg.get("enabled", False)
    ckpt_dir = ckpt_cfg.get("dir", "checkpoints")
    ckpt_fname = ckpt_cfg.get("filename", "best_model.pt")
    save_best_only = ckpt_cfg.get("save_best_only", True)
    resume_path = ckpt_cfg.get("resume_from", None)

    model = model.to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    start_epoch = 1
    best_loss = float("inf")

    # ==== Resume nếu có ====
    if ckpt_enabled and resume_path is not None:
        print(f"Resuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_loss = checkpoint.get("best_loss", float("inf"))

    # ensure dir exists
    if ckpt_enabled:
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, ckpt_fname)

    model.train()
    for epoch in range(start_epoch, num_epochs + 1):
        epoch_loss = 0.0
        n_batches = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(x_batch, target_len=output_size)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(1, n_batches)

        if epoch % print_every == 0 or epoch == start_epoch or epoch == num_epochs:
            print(f"[Epoch {epoch}/{num_epochs}] Train MSE (scaled): {avg_loss:.6f}")

        # ==== Save checkpoint ====
        if ckpt_enabled:
            is_best = avg_loss < best_loss
            if is_best:
                best_loss = avg_loss

            if not save_best_only or is_best:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "best_loss": best_loss,
                        "config": cfg,
                    },
                    ckpt_path,
                )
                if is_best:
                    print(f"✅ Saved best checkpoint to {ckpt_path}")

    return model
