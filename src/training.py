from __future__ import annotations

import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import TrainingConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def current_cuda_memory_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / 1024**2


def peak_cuda_memory_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024**2


def autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda")
    return nullcontext()


def make_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            pass
    return torch.cuda.amp.GradScaler(enabled=enabled)


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp: bool,
) -> float:
    model.eval()
    total_loss = 0.0
    total_seen = 0

    for X, y in loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with autocast_context(device, amp):
            loss = criterion(model(X), y)
        total_loss += loss.item() * X.size(0)
        total_seen += X.size(0)

    return total_loss / max(total_seen, 1)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    config: TrainingConfig,
    device: torch.device,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    total_seen = 0

    for batch_idx, (X, y) in enumerate(loader, start=1):
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        try:
            with autocast_context(device, config.amp):
                loss = criterion(model(X), y)
                scaled_loss = loss / config.grad_accum_steps

            scaler.scale(scaled_loss).backward()

            should_step = (
                batch_idx % config.grad_accum_steps == 0 or batch_idx == len(loader)
            )
            if should_step:
                scaler.unscale_(optimizer)
                if config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.max_grad_norm
                    )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        except torch.cuda.OutOfMemoryError as exc:
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            raise RuntimeError(
                "CUDA out of memory. Lower --batch-size or --mamba-d-model and rerun."
            ) from exc

        total_loss += loss.item() * X.size(0)
        total_seen += X.size(0)

    return total_loss / max(total_seen, 1)


@torch.inference_mode()
def benchmark_inference_ms(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    repeats: int = 100,
) -> float:
    model.eval()
    X = next(iter(loader))[0][:1].to(device, non_blocking=True)
    for _ in range(5):
        with autocast_context(device, amp):
            model(X)
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(repeats):
        with autocast_context(device, amp):
            model(X)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeats * 1000


def fit_model(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    config: TrainingConfig,
    device: torch.device,
) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"{model_name.lower()}_best.pt"

    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = nn.MSELoss()
    scaler = make_grad_scaler(enabled=config.amp and device.type == "cuda")

    best_val = float("inf")
    history = {
        "train_loss": [],
        "val_loss": [],
        "epoch_time_sec": [],
        "peak_vram_mb": [],
    }

    print(f"\nTraining {model_name}")
    print(f"Parameters: {count_parameters(model) / 1e6:.2f}M")

    for epoch in range(1, config.epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, config, device
        )
        val_loss = evaluate(model, val_loader, criterion, device, config.amp)
        epoch_time = time.perf_counter() - start
        peak_vram = peak_cuda_memory_mb()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["epoch_time_sec"].append(epoch_time)
        history["peak_vram_mb"].append(peak_vram)

        saved = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            saved = " saved"

        print(
            f"Epoch {epoch:02d}/{config.epochs} | "
            f"train={train_loss:.6f} | val={val_loss:.6f} | "
            f"time={epoch_time:.1f}s | peak_vram={peak_vram:.0f}MB{saved}"
        )

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    test_mse = evaluate(model, test_loader, criterion, device, config.amp)
    inference_ms = benchmark_inference_ms(
        model, test_loader, device, config.amp, repeats=100
    )

    return {
        "name": model_name,
        "checkpoint_path": str(checkpoint_path),
        "parameters": count_parameters(model),
        "best_val_mse": best_val,
        "test_mse": test_mse,
        "avg_epoch_time_sec": float(np.mean(history["epoch_time_sec"])),
        "inference_ms": inference_ms,
        "peak_vram_mb": max(history["peak_vram_mb"], default=0.0),
        "history": history,
        "model": model,
    }
