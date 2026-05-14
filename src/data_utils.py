from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

from src.config import DataConfig


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    scaler: MinMaxScaler
    target_index: int
    num_features: int
    train_size: int
    val_size: int
    test_size: int


def load_data(path: str | Path, features: tuple[str, ...] | list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [column for column in features if column not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in {path}: {missing}")

    df = df[list(features)].copy()
    df.replace(-9999.0, np.nan, inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"Loaded {len(df)} rows | Columns: {list(df.columns)}")
    return df


def normalize(df: pd.DataFrame):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df.values).astype(np.float32, copy=False)
    return scaled, scaler


def count_windows(data: np.ndarray, input_steps: int, output_steps: int) -> int:
    return max(0, len(data) - input_steps - output_steps + 1)


class ClimateWindowDataset(Dataset):
    """Lazy sliding-window dataset for sequence forecasting."""

    def __init__(
        self,
        data: np.ndarray,
        start_idx: int,
        end_idx: int,
        input_steps: int,
        output_steps: int,
        target_index: int,
    ):
        self.data = data
        self.start_idx = int(start_idx)
        self.end_idx = int(end_idx)
        self.input_steps = int(input_steps)
        self.output_steps = int(output_steps)
        self.target_index = int(target_index)

    def __len__(self):
        return max(0, self.end_idx - self.start_idx)

    def __getitem__(self, idx):
        i = self.start_idx + idx
        x = self.data[i : i + self.input_steps]
        y = self.data[
            i + self.input_steps : i + self.input_steps + self.output_steps,
            self.target_index,
        ]
        return (
            torch.from_numpy(np.ascontiguousarray(x)),
            torch.from_numpy(np.ascontiguousarray(y)),
        )


def create_windows(
    data: np.ndarray,
    input_steps: int,
    output_steps: int,
    target_index: int = 0,
):
    """Eager window creation kept for compatibility with older experiments."""
    X, y = [], []
    for i in range(count_windows(data, input_steps, output_steps)):
        X.append(data[i : i + input_steps])
        y.append(data[i + input_steps : i + input_steps + output_steps, target_index])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


def build_dataloaders(config: DataConfig) -> DataBundle:
    if config.target not in config.features:
        raise ValueError(f"Target {config.target!r} must be one of {config.features}")

    df = load_data(config.csv_path, config.features)
    scaled, scaler = normalize(df)
    n_windows = count_windows(scaled, config.input_steps, config.output_steps)
    if n_windows <= 0:
        raise ValueError("Not enough rows to create any input/output windows.")

    train_end = int(n_windows * config.train_split)
    val_end = int(n_windows * (config.train_split + config.val_split))
    target_index = list(config.features).index(config.target)

    train_ds = ClimateWindowDataset(
        scaled, 0, train_end, config.input_steps, config.output_steps, target_index
    )
    val_ds = ClimateWindowDataset(
        scaled, train_end, val_end, config.input_steps, config.output_steps, target_index
    )
    test_ds = ClimateWindowDataset(
        scaled, val_end, n_windows, config.input_steps, config.output_steps, target_index
    )

    loader_common = {
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory,
    }
    if config.num_workers > 0:
        loader_common.update(prefetch_factor=2, persistent_workers=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        **loader_common,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.eval_batch_size,
        shuffle=False,
        **loader_common,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.eval_batch_size,
        shuffle=False,
        **loader_common,
    )

    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        scaler=scaler,
        target_index=target_index,
        num_features=len(config.features),
        train_size=len(train_ds),
        val_size=len(val_ds),
        test_size=len(test_ds),
    )


def get_dataloaders(
    path: str,
    input_steps=168,
    output_steps=24,
    batch_size=64,
    train_split=0.7,
    val_split=0.15,
):
    """Backward-compatible helper used by the original quick test script."""
    config = DataConfig(
        csv_path=Path(path),
        input_steps=input_steps,
        output_steps=output_steps,
        batch_size=batch_size,
        eval_batch_size=batch_size,
        train_split=train_split,
        val_split=val_split,
        num_workers=0,
        pin_memory=False,
    )
    bundle = build_dataloaders(config)
    print(
        f"Train: {bundle.train_size} | Val: {bundle.val_size} | Test: {bundle.test_size}"
    )
    return bundle.train_loader, bundle.val_loader, bundle.test_loader, bundle.scaler
