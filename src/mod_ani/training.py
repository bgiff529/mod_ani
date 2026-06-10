"""Training, evaluation, and reporting utilities."""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from mod_ani.local_torchani import use_local_torchani

use_local_torchani()

from tqdm.auto import tqdm
from torchani.grad import forces_for_training
from torchani.units import hartree2kcalpermol

from mod_ani.config import ExperimentConfig
from mod_ani.models import build_model, count_parameters


def make_run_dir(config: ExperimentConfig) -> Path:
    """Create a timestamped run directory."""

    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(config.run_dir) / f"{stamp}-{config.dataset}-{config.model_kind}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _torch_dtype(config: ExperimentConfig) -> torch.dtype:
    if config.dtype == "float64":
        return torch.float64
    if config.dtype == "float32":
        return torch.float32
    raise ValueError("Only float32 and float64 are supported")


def move_batch(
    batch: dict[str, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Move a TorchANI batch to a device with GPU-safe floating dtypes."""

    moved: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if torch.is_floating_point(value):
            moved[key] = value.to(device=device, dtype=dtype, non_blocking=True)
        else:
            moved[key] = value.to(device=device, non_blocking=True)
    return moved


def filter_energy_outliers(
    batch: dict[str, torch.Tensor],
    max_abs_energy_hartree: float,
) -> dict[str, torch.Tensor] | None:
    """Drop conformers with non-finite or placeholder-scale target energies."""

    if max_abs_energy_hartree <= 0 or "energies" not in batch:
        return batch
    energies = batch["energies"]
    mask = torch.isfinite(energies) & (energies.abs() <= max_abs_energy_hartree)
    if bool(mask.all()):
        return batch
    if not bool(mask.any()):
        return None
    return {key: value[mask] if value.shape[:1] == mask.shape else value for key, value in batch.items()}


def evaluate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    max_abs_energy_hartree: float = 1.0e5,
    desc: str = "Validation",
    verbose: bool = True,
) -> dict[str, float]:
    """Evaluate energy prediction errors."""

    model.train(False)
    squared_error = 0.0
    absolute_error = 0.0
    count = 0
    skipped = 0
    with torch.no_grad():
        iterator = tqdm(
            dataloader,
            total=len(dataloader),
            desc=desc,
            unit="batch",
            disable=not verbose,
            leave=False,
        )
        for batch in iterator:
            batch = move_batch(batch, device, dtype)
            batch = filter_energy_outliers(batch, max_abs_energy_hartree)
            if batch is None:
                skipped += 1
                continue
            species = batch["species"]
            coordinates = batch["coordinates"]
            target_energies = batch["energies"]
            predicted_energies = model((species, coordinates)).energies
            errors = predicted_energies - target_energies
            squared_error += errors.pow(2).sum().item()
            absolute_error += errors.abs().sum().item()
            count += predicted_energies.shape[0]
            if verbose:
                iterator.set_postfix(conformers=count, skipped=skipped)
    model.train(True)
    rmse = math.sqrt(squared_error / max(count, 1))
    mae = absolute_error / max(count, 1)
    return {
        "rmse_hartree": rmse,
        "mae_hartree": mae,
        "rmse_kcal_mol": hartree2kcalpermol(rmse),
        "mae_kcal_mol": hartree2kcalpermol(mae),
        "num_conformers": float(count),
        "skipped_validation_batches": float(skipped),
    }


def train(
    config: ExperimentConfig,
    batched: dict[str, Any],
    run_dir: Path | None = None,
) -> tuple[torch.nn.Module, list[dict[str, Any]]]:
    """Train one configured model and write checkpoints/metrics."""

    if run_dir is None:
        run_dir = make_run_dir(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    if config.verbose:
        print("=" * 88)
        print(f"[train] Starting {config.model_kind} on {config.dataset} ({config.lot})")
        print(f"[train] Run directory: {run_dir}")
    (run_dir / "config.json").write_text(
        json.dumps(config.as_dict(), indent=2),
        encoding="utf-8",
    )

    device = torch.device(config.resolved_device())
    dtype = _torch_dtype(config)
    if device.type == "mps" and dtype == torch.float64:
        dtype = torch.float32
    if config.verbose:
        print(
            "[train] Device/dtype/lr: "
            f"{device.type}/{str(dtype).replace('torch.', '')}/"
            f"{config.effective_learning_rate():.3e}"
        )
        print(
            "[train] Options: "
            f"epochs={config.max_epochs}, force_training={config.force_training}, "
            f"repulsion={config.repulsion}, max_grad_norm={config.max_grad_norm}"
        )
        print("[train] Building model...")
    model = build_model(config).to(device=device, dtype=dtype)
    if config.verbose:
        print(f"[train] Trainable parameters: {count_parameters(model):,}")
    optimizer = torch.optim.AdamW(
        model.neural_networks.parameters(),
        lr=config.effective_learning_rate(),
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=10,
        threshold=0.0,
    )
    mse = torch.nn.MSELoss(reduction="none")

    training = batched["training"].as_dataloader(
        num_workers=config.num_workers,
        pin_memory=False,
    )
    validation = batched["validation"].as_dataloader(
        num_workers=config.num_workers,
        pin_memory=False,
        shuffle=False,
    )
    if config.verbose:
        print(f"[train] Training batches per epoch: {len(training)}")
        print(f"[train] Validation batches per epoch: {len(validation)}")

    history: list[dict[str, Any]] = []
    best_rmse = math.inf
    for epoch in range(1, config.max_epochs + 1):
        epoch_start = time.perf_counter()
        if config.verbose:
            print("-" * 88)
            print(f"[train] Epoch {epoch}/{config.max_epochs} started")
        model.train(True)
        epoch_loss = 0.0
        batches = 0
        skipped_train_batches = 0
        train_iterator = tqdm(
            training,
            total=len(training),
            desc=f"{config.model_kind} epoch {epoch}/{config.max_epochs}",
            unit="batch",
            disable=not config.verbose,
            leave=True,
        )
        for batch_idx, batch in enumerate(train_iterator):
            batch = move_batch(batch, device, dtype)
            batch = filter_energy_outliers(batch, config.max_abs_energy_hartree)
            if batch is None:
                skipped_train_batches += 1
                continue
            species = batch["species"]
            coordinates = batch["coordinates"].requires_grad_(config.force_training)
            target_energies = batch["energies"]
            num_atoms = (species >= 0).sum(dim=1, dtype=target_energies.dtype)
            try:
                predicted_energies = model((species, coordinates)).energies
            except RuntimeError as exc:
                shape_context = {
                    key: tuple(value.shape)
                    for key, value in batch.items()
                    if isinstance(value, torch.Tensor)
                }
                raise RuntimeError(
                    "Model forward failed "
                    f"(device={device.type}, model_kind={config.model_kind}, "
                    f"epoch={epoch}, batch_idx={batch_idx}, shapes={shape_context})"
                ) from exc
            energy_loss = (mse(predicted_energies, target_energies) / num_atoms.sqrt()).mean()
            if config.force_training and "forces" in batch:
                target_forces = batch["forces"]
                predicted_forces = forces_for_training(predicted_energies, coordinates)
                force_loss = (
                    mse(predicted_forces, target_forces).sum(dim=(1, 2)) / num_atoms
                ).mean()
                loss = energy_loss + config.force_coefficient * force_loss
            else:
                loss = energy_loss
            optimizer.zero_grad()
            loss.backward()
            if config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.neural_networks.parameters(),
                    max_norm=config.max_grad_norm,
                )
            optimizer.step()
            epoch_loss += loss.detach().item()
            batches += 1
            if config.verbose:
                train_iterator.set_postfix(
                    loss=f"{loss.detach().item():.4g}",
                    used=batches,
                    skipped=skipped_train_batches,
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                )

        if config.verbose:
            print(f"[train] Epoch {epoch}: validating...")
        metrics = evaluate(
            model,
            validation,
            device,
            dtype,
            config.max_abs_energy_hartree,
            desc=f"{config.model_kind} validation {epoch}/{config.max_epochs}",
            verbose=config.verbose,
        )
        epoch_seconds = time.perf_counter() - epoch_start
        metrics.update(
            {
                "epoch": float(epoch),
                "train_loss": epoch_loss / max(batches, 1),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "parameter_count": float(count_parameters(model)),
                "device": device.type,
                "dtype": str(dtype).replace("torch.", ""),
                "epoch_seconds": epoch_seconds,
                "training_batches": float(batches),
                "skipped_training_batches": float(skipped_train_batches),
            }
        )
        history.append(metrics)
        scheduler.step(metrics["rmse_kcal_mol"])
        if config.verbose:
            print(
                "[train] Epoch "
                f"{epoch}/{config.max_epochs} done in {epoch_seconds:.1f}s | "
                f"train_loss={metrics['train_loss']:.6g} | "
                f"val_rmse={metrics['rmse_kcal_mol']:.6g} kcal/mol | "
                f"val_mae={metrics['mae_kcal_mol']:.6g} kcal/mol | "
                f"val_conformers={int(metrics['num_conformers'])}"
            )

        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "config": config.as_dict(),
                "history": history,
            },
            run_dir / "latest_training_state.pt",
        )
        if config.verbose:
            print(f"[train] Saved latest checkpoint: {run_dir / 'latest_training_state.pt'}")
        if metrics["rmse_kcal_mol"] < best_rmse:
            best_rmse = metrics["rmse_kcal_mol"]
            torch.save(model.state_dict(), run_dir / "best_model_state.pt")
            if config.verbose:
                print(
                    "[train] New best model: "
                    f"{best_rmse:.6g} kcal/mol -> {run_dir / 'best_model_state.pt'}"
                )
        write_history(history, run_dir / "metrics.csv")
        (run_dir / "metrics.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if config.verbose:
            print(f"[train] Wrote metrics: {run_dir / 'metrics.csv'}")

    if config.verbose:
        print(f"[train] Finished {config.model_kind}. Best RMSE: {best_rmse:.6g} kcal/mol")
    return model, history


def write_history(history: list[dict[str, Any]], path: Path) -> None:
    if not history:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def train_pair(
    base_config: ExperimentConfig,
    batched: dict[str, Any],
    run_root: Path | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Train baseline and electron-radial models on the same batches."""

    if run_root is None:
        run_root = make_run_dir(base_config)
    run_root.mkdir(parents=True, exist_ok=True)
    results: dict[str, list[dict[str, Any]]] = {}
    for model_kind in ("baseline", "electron_radial"):
        config = ExperimentConfig(**base_config.as_dict())
        config.model_kind = model_kind
        model_dir = run_root / model_kind
        if config.verbose:
            print("#" * 88)
            print(f"[train_pair] Training model: {model_kind}")
        _, history = train(config, batched, run_dir=model_dir)
        results[model_kind] = history
        if config.verbose:
            print(f"[train_pair] Completed model: {model_kind}")
    return results
