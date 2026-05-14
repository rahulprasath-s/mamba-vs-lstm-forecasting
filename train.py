from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import torch

from src.config import DataConfig, LSTMConfig, MambaConfig, TrainingConfig
from src.data_utils import build_dataloaders
from src.lstm_model import LSTMForecaster
from src.mamba_model import MambaForecaster
from src.training import fit_model, get_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train LSTM and Mamba-style forecasters on Jena Climate data."
    )
    parser.add_argument("--data-path", type=Path, default=Path("data/jena_climate_2009_2016.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--model", choices=["lstm", "mamba", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--input-steps", type=int, default=168)
    parser.add_argument("--output-steps", type=int, default=24)
    parser.add_argument("--features", nargs="+", default=["T (degC)", "p (mbar)", "rh (%)"])
    parser.add_argument("--target", default="T (degC)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=min(2, os.cpu_count() or 0))

    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--no-amp", action="store_true")

    parser.add_argument("--lstm-hidden-size", type=int, default=128)
    parser.add_argument("--lstm-layers", type=int, default=2)
    parser.add_argument("--lstm-dropout", type=float, default=0.2)

    parser.add_argument("--mamba-d-model", type=int, default=256)
    parser.add_argument("--mamba-layers", type=int, default=3)
    parser.add_argument("--mamba-expand", type=int, default=2)
    parser.add_argument("--mamba-d-conv", type=int, default=3)
    parser.add_argument("--mamba-dropout", type=float, default=0.1)
    parser.add_argument("--no-checkpoint", action="store_true")
    return parser.parse_args()


def write_run_config(
    output_dir: Path,
    data_config: DataConfig,
    train_config: TrainingConfig,
    lstm_config: LSTMConfig,
    mamba_config: MambaConfig,
) -> None:
    payload = {
        "data": data_config.to_dict(),
        "training": train_config.to_dict(),
        "lstm": lstm_config.to_dict(),
        "mamba": mamba_config.to_dict(),
    }
    (output_dir / "run_config.json").write_text(json.dumps(payload, indent=2))


def main() -> None:
    args = parse_args()
    device = get_device()
    set_seed(args.seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    data_config = DataConfig(
        csv_path=args.data_path,
        features=tuple(args.features),
        target=args.target,
        input_steps=args.input_steps,
        output_steps=args.output_steps,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    train_config = TrainingConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_accum_steps=args.grad_accum_steps,
        amp=not args.no_amp,
        seed=args.seed,
        output_dir=output_dir,
    )
    lstm_config = LSTMConfig(
        hidden_size=args.lstm_hidden_size,
        num_layers=args.lstm_layers,
        dropout=args.lstm_dropout,
    )
    mamba_config = MambaConfig(
        d_model=args.mamba_d_model,
        num_layers=args.mamba_layers,
        expand=args.mamba_expand,
        d_conv=args.mamba_d_conv,
        dropout=args.mamba_dropout,
        use_checkpoint=not args.no_checkpoint,
    )
    write_run_config(output_dir, data_config, train_config, lstm_config, mamba_config)

    print(f"Device: {device}")
    bundle = build_dataloaders(data_config)
    print(
        f"Samples: train={bundle.train_size}, val={bundle.val_size}, test={bundle.test_size}"
    )
    print(f"Results directory: {output_dir}")

    results = []

    if args.model in {"lstm", "both"}:
        lstm = LSTMForecaster(
            input_size=len(data_config.features),
            hidden_size=lstm_config.hidden_size,
            num_layers=lstm_config.num_layers,
            output_steps=data_config.output_steps,
            dropout=lstm_config.dropout,
        )
        results.append(
            fit_model(
                lstm,
                "LSTM",
                bundle.train_loader,
                bundle.val_loader,
                bundle.test_loader,
                train_config,
                device,
            )
        )
        del lstm
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if args.model in {"mamba", "both"}:
        mamba = MambaForecaster(
            input_size=len(data_config.features),
            d_model=mamba_config.d_model,
            n_layers=mamba_config.num_layers,
            output_steps=data_config.output_steps,
            expand=mamba_config.expand,
            d_conv=mamba_config.d_conv,
            dropout=mamba_config.dropout,
            use_checkpoint=mamba_config.use_checkpoint,
        )
        results.append(
            fit_model(
                mamba,
                "Mamba",
                bundle.train_loader,
                bundle.val_loader,
                bundle.test_loader,
                train_config,
                device,
            )
        )

    from src.visualization import (
        plot_loss_curves,
        plot_sample_forecast,
        plot_test_comparison,
        save_metrics,
    )

    metrics = save_metrics(results, output_dir)
    plot_loss_curves(results, output_dir)
    plot_test_comparison(metrics, output_dir)

    for result in results:
        plot_sample_forecast(
            result["model"],
            bundle.test_loader,
            bundle.scaler,
            bundle.target_index,
            bundle.num_features,
            device,
            train_config.amp,
            output_dir,
            result["name"],
        )
        result.pop("model", None)

    print("\nFinal metrics")
    print(metrics.to_string(index=False))
    print(f"\nSaved checkpoints, metrics, plots, and config to {output_dir}")


if __name__ == "__main__":
    main()
