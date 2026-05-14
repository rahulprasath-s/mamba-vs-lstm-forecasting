from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path.cwd() / ".cache"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.training import autocast_context


def save_metrics(results: list[dict[str, Any]], output_dir: Path) -> pd.DataFrame:
    rows = []
    for result in results:
        rows.append(
            {
                "model": result["name"],
                "test_mse": result["test_mse"],
                "best_val_mse": result["best_val_mse"],
                "parameters": result["parameters"],
                "avg_epoch_time_sec": result["avg_epoch_time_sec"],
                "inference_ms": result["inference_ms"],
                "peak_vram_mb": result["peak_vram_mb"],
                "checkpoint_path": result["checkpoint_path"],
            }
        )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    return metrics


def plot_loss_curves(results: list[dict[str, Any]], output_dir: Path) -> None:
    for result in results:
        history = result["history"]
        epochs = range(1, len(history["train_loss"]) + 1)
        plt.figure(figsize=(8, 4.5))
        plt.plot(epochs, history["train_loss"], marker="o", label="Train")
        plt.plot(epochs, history["val_loss"], marker="o", label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.title(f"{result['name']} Loss Curve")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{result['name'].lower()}_loss_curve.png", dpi=160)
        plt.close()


def plot_test_comparison(metrics: pd.DataFrame, output_dir: Path) -> None:
    if metrics.empty:
        return
    plt.figure(figsize=(6, 4.5))
    bars = plt.bar(metrics["model"], metrics["test_mse"], color=["#4c78a8", "#f58518"])
    plt.ylabel("Test MSE")
    plt.title("Test Error Comparison")
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.6f}",
            ha="center",
            va="bottom",
        )
    plt.tight_layout()
    plt.savefig(output_dir / "test_mse_comparison.png", dpi=160)
    plt.close()


def inverse_target(
    scaler,
    values: np.ndarray,
    target_index: int,
    num_features: int,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    flat = values.reshape(-1)
    padded = np.zeros((flat.shape[0], num_features), dtype=np.float32)
    padded[:, target_index] = flat
    return scaler.inverse_transform(padded)[:, target_index].reshape(values.shape)


@torch.inference_mode()
def plot_sample_forecast(
    model: torch.nn.Module,
    loader,
    scaler,
    target_index: int,
    num_features: int,
    device: torch.device,
    amp: bool,
    output_dir: Path,
    model_name: str,
) -> None:
    model.eval()
    X_batch, y_batch = next(iter(loader))
    with autocast_context(device, amp):
        pred = model(X_batch.to(device, non_blocking=True)).float().cpu().numpy()

    actual = inverse_target(scaler, y_batch.numpy()[0], target_index, num_features)
    predicted = inverse_target(scaler, pred[0], target_index, num_features)

    plt.figure(figsize=(8, 4.5))
    plt.plot(actual, marker="o", label="Actual")
    plt.plot(predicted, marker="o", label=f"{model_name} prediction")
    plt.xlabel("Forecast step")
    plt.ylabel("Target value")
    plt.title(f"Sample {model_name} Forecast")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name.lower()}_sample_forecast.png", dpi=160)
    plt.close()
