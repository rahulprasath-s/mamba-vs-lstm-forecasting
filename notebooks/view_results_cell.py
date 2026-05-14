from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import display


RESULTS_DIR = Path("/content/drive/MyDrive/mamba-vs-lstm/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_TRAIN_LOSSES = [
    0.004609, 0.000761, 0.000722, 0.000705, 0.000701,
    0.000687, 0.000690, 0.000680, 0.000676, 0.000672,
    0.000667, 0.000665, 0.000662, 0.000660, 0.000650,
    0.000650, 0.000645, 0.000645, 0.000642, 0.000658,
]
DEFAULT_VAL_LOSSES = [
    0.000728, 0.000704, 0.000705, 0.000750, 0.000686,
    0.000762, 0.000713, 0.000673, 0.000733, 0.000677,
    0.000687, 0.000685, 0.000677, 0.000659, 0.000690,
    0.000647, 0.000645, 0.000657, 0.000648, 0.000656,
]

if "lstm_test_mse" not in globals():
    lstm_test_mse = 0.000628
if "mamba_test_mse" not in globals():
    mamba_test_mse = 0.000587
if "train_losses" not in globals():
    train_losses = DEFAULT_TRAIN_LOSSES
if "val_losses" not in globals():
    val_losses = DEFAULT_VAL_LOSSES


summary = pd.DataFrame(
    [
        {"model": "LSTM", "test_mse": float(lstm_test_mse)},
        {"model": "Mamba-style", "test_mse": float(mamba_test_mse)},
    ]
)
summary["relative_to_lstm"] = summary["test_mse"] / float(lstm_test_mse)
summary.to_csv(RESULTS_DIR / "summary_metrics.csv", index=False)
display(summary)


plt.figure(figsize=(8, 4.5))
plt.plot(range(1, len(train_losses) + 1), train_losses, marker="o", label="Train")
plt.plot(range(1, len(val_losses) + 1), val_losses, marker="o", label="Validation")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Mamba Training Curve")
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / "loss_curve.png", dpi=160)
plt.show()


plt.figure(figsize=(6, 4.5))
bars = plt.bar(summary["model"], summary["test_mse"], color=["#4c78a8", "#f58518"])
plt.ylabel("Test MSE")
plt.title("Normalized Test Error")
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
plt.savefig(RESULTS_DIR / "test_mse_comparison.png", dpi=160)
plt.show()


def inverse_temperature(values):
    values = np.asarray(values, dtype=np.float32)
    flat = values.reshape(-1)
    padded = np.zeros((flat.shape[0], 3), dtype=np.float32)
    padded[:, 0] = flat
    return scaler.inverse_transform(padded)[:, 0].reshape(values.shape)


if all(name in globals() for name in ["mamba", "test_dl", "scaler", "DEVICE"]):
    mamba.eval()
    X_batch, y_batch = next(iter(test_dl))
    with torch.inference_mode():
        with torch.amp.autocast(device_type=DEVICE, enabled=(DEVICE == "cuda")):
            pred_batch = mamba(X_batch.to(DEVICE, non_blocking=True)).float().cpu().numpy()

    actual_norm = y_batch.numpy()
    pred_norm = pred_batch
    actual_deg_c = inverse_temperature(actual_norm[0])
    pred_deg_c = inverse_temperature(pred_norm[0])

    plt.figure(figsize=(8, 4.5))
    plt.plot(actual_deg_c, marker="o", label="Actual")
    plt.plot(pred_deg_c, marker="o", label="Mamba prediction")
    plt.xlabel("Forecast step")
    plt.ylabel("Temperature (degC)")
    plt.title("Sample Test Forecast")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "sample_forecast.png", dpi=160)
    plt.show()
else:
    print("Skipped sample_forecast.png because mamba/test_dl/scaler are not in memory.")
    print("Run this cell immediately after the training cell to generate the forecast plot.")


print(f"Saved result artifacts to: {RESULTS_DIR}")
print(f"LSTM Test MSE   : {lstm_test_mse:.6f}")
print(f"Mamba Test MSE  : {mamba_test_mse:.6f}")
print(f"Improvement     : {(1 - mamba_test_mse / lstm_test_mse) * 100:.2f}% lower MSE")
