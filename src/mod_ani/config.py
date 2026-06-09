"""Configuration objects for proposal experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal


DatasetName = Literal["TestData", "ANI1x", "ANI1ccx"]
ModelKind = Literal["baseline", "electron_radial"]


@dataclass(slots=True)
class ExperimentConfig:
    """Top-level knobs used by the notebook and scripts."""

    dataset: DatasetName = "TestData"
    lot: str = "wb97x-631gd"
    model_kind: ModelKind = "electron_radial"
    symbols: tuple[str, ...] = ("H", "C", "N", "O")
    batch_size: int = 256
    max_epochs: int = 3
    learning_rate: float = 5.0e-4
    weight_decay: float = 1.0e-6
    max_grad_norm: float = 10.0
    validation_fraction: float = 0.2
    divs_seed: int = 20260597
    batch_seed: int = 20260598
    force_training: bool = False
    force_coefficient: float = 0.1
    repulsion: bool = False
    max_abs_energy_hartree: float = 1.0e5
    cache_batches: bool = True
    num_workers: int = 0
    device: str = "auto"
    dtype: str = "float32"
    data_dir: Path = Path("data")
    run_dir: Path = Path("runs")
    refresh_batches: bool = False
    limit_train_batches: int | float = 1.0
    limit_valid_batches: int | float = 1.0
    notes: str = ""
    extra: dict[str, str | int | float | bool] = field(default_factory=dict)

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    def effective_learning_rate(self) -> float:
        if self.resolved_device() == "mps":
            return min(self.learning_rate, 1.0e-4)
        return self.learning_rate

    def as_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["data_dir"] = str(self.data_dir)
        data["run_dir"] = str(self.run_dir)
        return data


def quick_test_config(**overrides: object) -> ExperimentConfig:
    """Small default config intended to finish from a fresh notebook run."""

    config = ExperimentConfig(
        dataset="TestData",
        batch_size=128,
        max_epochs=2,
        limit_train_batches=16,
        limit_valid_batches=8,
        notes="Fast smoke run using TorchANI TestData.",
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def ani1x_config(**overrides: object) -> ExperimentConfig:
    """Larger proposal-style DFT energy run on ANI1x."""

    config = ExperimentConfig(
        dataset="ANI1x",
        batch_size=2560,
        max_epochs=20,
        cache_batches=False,
        notes="ANI1x DFT energy benchmark.",
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config
