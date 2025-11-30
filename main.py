# main.py
import argparse
import os

import torch

from utils import load_config, set_seed
from trainer import build_dataloaders, build_model, train_model
from eval import evaluate_model, load_model_from_checkpoint, rollout_predictions, plot_time_series


def parse_args():
    parser = argparse.ArgumentParser(description="Option price t+2 forecasting")

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config YAML",
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="If set, plot time series of true vs predicted on test set",
    )

    # override 1 số hyperparam nếu muốn
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--hidden_size", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--model_type", type=str, default=None)  # lstm / rnn

    args = parser.parse_args()
    return args


def apply_overrides(cfg, args):
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        cfg["training"]["learning_rate"] = args.learning_rate
    if args.hidden_size is not None:
        cfg["model"]["hidden_size"] = args.hidden_size
    if args.num_epochs is not None:
        cfg["training"]["num_epochs"] = args.num_epochs
    if args.model_type is not None:
        cfg["model"]["type"] = args.model_type
    return cfg


def main():
    args = parse_args()
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args)

    set_seed(42)

    # nếu không có CUDA thì fallback CPU
    if cfg["training"]["device"] == "cuda" and not torch.cuda.is_available():
        print("CUDA không available, chuyển sang CPU.")
        cfg["training"]["device"] = "cpu"

    # build dataloader & meta (scalers, feature cols...)
    train_loader, test_loader, meta = build_dataloaders(cfg)
    feature_cols = meta["feature_cols"]
    scaler_y = meta["scaler_y"]

    print(f"Train months: {meta['train_months']}")
    print(f"Test month: {meta['test_month']}")
    print(f"Using {len(feature_cols)} features: {feature_cols}")

    # build model
    model = build_model(cfg, input_size=len(feature_cols))

    # train
    model = train_model(cfg, model, train_loader)

    # evaluate
    ckpt_cfg = cfg.get("checkpoint", {})
    if ckpt_cfg.get("enabled", False) and ckpt_cfg.get("filename", None):
        ckpt_path = os.path.join(
            ckpt_cfg.get("dir", "checkpoints"),
            ckpt_cfg.get("filename", "best_model.pt"),
        )
        if os.path.exists(ckpt_path):
            model = load_model_from_checkpoint(
                model,
                ckpt_path,
                device=cfg["training"]["device"],
            )

    results, (y_true, y_pred) = evaluate_model(cfg, model, test_loader, scaler_y)
    print("=== Test metrics (t+2 option_price) ===")
    for k, v in results.items():
        print(f"{k.upper()}: {v:.6f}")


    if args.plot:
        y_true_roll, y_pred_roll = rollout_predictions(cfg, model, test_loader, scaler_y)

        test_dates = meta.get("test_dates", None)
        window_size = cfg["dataset"]["window_size"]

        # Align dates với target: mỗi y_t là label của cửa sổ kết thúc ở index (window_size-1 + i)
        if test_dates is not None:
            # test_dates: length = N_test_raw
            # y_true_roll: length = N_seq = N_test_raw - window_size + 1
            test_dates_aligned = test_dates[window_size - 1:]
        else:
            test_dates_aligned = None

        title = f"Option price t+2 - Test month {meta['test_month']}"

        out_dir = cfg["logging"].get("output_dir", "outputs")
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, f"time_series_{meta['test_month']}.png")

        plot_time_series(
            y_true_roll,
            y_pred_roll,
            dates=test_dates_aligned,
            title=title,
            save_path=save_path,
        )


if __name__ == "__main__":
    main()
