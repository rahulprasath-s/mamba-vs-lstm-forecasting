from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    csv_path: Path = Path("data/jena_climate_2009_2016.csv")
    features: tuple[str, ...] = ("T (degC)", "p (mbar)", "rh (%)")
    target: str = "T (degC)"
    input_steps: int = 168
    output_steps: int = 24
    train_split: float = 0.70
    val_split: float = 0.15
    batch_size: int = 64
    eval_batch_size: int = 128
    num_workers: int = 2
    pin_memory: bool = True

    def to_dict(self) -> dict:
        values = asdict(self)
        values["csv_path"] = str(self.csv_path)
        values["features"] = list(self.features)
        return values


@dataclass(frozen=True)
class LSTMConfig:
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class MambaConfig:
    d_model: int = 256
    num_layers: int = 3
    expand: int = 2
    d_conv: int = 3
    dropout: float = 0.1
    use_checkpoint: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    grad_accum_steps: int = 8
    max_grad_norm: float = 1.0
    amp: bool = True
    seed: int = 42
    output_dir: Path = Path("results")

    def to_dict(self) -> dict:
        values = asdict(self)
        values["output_dir"] = str(self.output_dir)
        return values

