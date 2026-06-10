"""Profiling helpers for TorchANI on Apple Silicon and other devices."""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from typing import Any

import torch

from mod_ani.config import ExperimentConfig
from mod_ani.local_torchani import use_local_torchani
from mod_ani.training import filter_energy_outliers, move_batch

use_local_torchani()

from torchani.neighbors import discard_outside_cutoff, neighbors_to_triples


def torch_dtype(config: ExperimentConfig) -> torch.dtype:
    """Return the usable dtype for a config/device pair."""

    if config.dtype == "float64":
        dtype = torch.float64
    elif config.dtype == "float32":
        dtype = torch.float32
    else:
        raise ValueError("Only float32 and float64 are supported")
    device_name = config.resolved_device()
    if device_name == "mps" and dtype == torch.float64:
        return torch.float32
    return dtype


def available_devices() -> list[str]:
    """Return devices worth comparing on this machine."""

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append("mps")
    return devices


def synchronize_device(device: torch.device | str) -> None:
    """Synchronize accelerator queues before/after timing."""

    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def first_batch(batched: dict[str, Any], split: str = "training") -> dict[str, torch.Tensor]:
    """Load the first batch from a TorchANI batched dataset split."""

    loader = batched[split].as_dataloader(num_workers=0, pin_memory=False, shuffle=False)
    return next(iter(loader))


def prepare_profile_batch(
    batch: dict[str, torch.Tensor],
    device_name: str,
    dtype: torch.dtype,
    max_abs_energy_hartree: float,
) -> dict[str, torch.Tensor]:
    """Move and filter one batch for profiling."""

    moved = move_batch(batch, torch.device(device_name), dtype)
    filtered = filter_energy_outliers(moved, max_abs_energy_hartree)
    if filtered is None:
        raise ValueError("The selected batch only contained filtered energy outliers")
    return filtered


def time_callable(
    operation: Callable[[], Any],
    device_name: str,
    warmup: int = 2,
    repeats: int = 5,
) -> dict[str, Any]:
    """Time one callable, synchronizing accelerators around every measurement."""

    for _ in range(warmup):
        operation()
    synchronize_device(device_name)

    samples: list[float] = []
    for _ in range(repeats):
        synchronize_device(device_name)
        start = time.perf_counter()
        operation()
        synchronize_device(device_name)
        samples.append(time.perf_counter() - start)
    return {
        "mean_ms": statistics.fmean(samples) * 1000.0,
        "median_ms": statistics.median(samples) * 1000.0,
        "min_ms": min(samples) * 1000.0,
        "max_ms": max(samples) * 1000.0,
        "samples_ms": [sample * 1000.0 for sample in samples],
    }


def _profile_row(
    operation: str,
    device_name: str,
    timing: dict[str, Any],
    batch: dict[str, torch.Tensor],
    pairs: int | None,
    triples: int | None,
) -> dict[str, Any]:
    species = batch["species"]
    coordinates = batch["coordinates"]
    row = {
        "operation": operation,
        "device": device_name,
        "batch_size": int(species.shape[0]),
        "atoms_per_conformer": int(species.shape[1]),
        "coordinate_dtype": str(coordinates.dtype).replace("torch.", ""),
        "mean_ms": timing["mean_ms"],
        "median_ms": timing["median_ms"],
        "min_ms": timing["min_ms"],
        "max_ms": timing["max_ms"],
        "pairs": pairs,
        "triples": triples,
    }
    return row


def profile_aev_and_training_step(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    config: ExperimentConfig,
    device_name: str,
    warmup: int = 2,
    repeats: int = 5,
) -> list[dict[str, Any]]:
    """Profile the main TorchANI operations on one device."""

    device = torch.device(device_name)
    dtype = torch.float32 if device.type == "mps" else torch_dtype(config)
    model = model.to(device=device, dtype=dtype)
    profile_batch = prepare_profile_batch(
        batch,
        device_name=device_name,
        dtype=dtype,
        max_abs_energy_hartree=config.max_abs_energy_hartree,
    )
    species = profile_batch["species"]
    coordinates = profile_batch["coordinates"]
    target_energies = profile_batch["energies"]
    elem_idxs = model.species_converter(
        species,
        nop=not getattr(model, "periodic_table_index", True),
    )
    aev = model.aev_computer
    mse = torch.nn.MSELoss(reduction="none")

    rows: list[dict[str, Any]] = []

    with torch.no_grad():
        neighbors = aev.neighborlist(aev.radial.cutoff, elem_idxs, coordinates, None, None)
        radial_terms = aev.radial(neighbors.distances)
        angular_neighbors = discard_outside_cutoff(neighbors, aev.angular.cutoff)
        triples = neighbors_to_triples(angular_neighbors)
        angular_terms = aev.angular(triples.distances, triples.diff_vectors)
        pairs = int(neighbors.indices.shape[1])
        triple_count = int(triples.central_idxs.shape[0])

        operations: list[tuple[str, Callable[[], Any], int | None, int | None]] = [
            (
                "neighborlist",
                lambda: aev.neighborlist(
                    aev.radial.cutoff,
                    elem_idxs,
                    coordinates,
                    None,
                    None,
                ),
                pairs,
                None,
            ),
            ("radial_terms", lambda: aev.radial(neighbors.distances), pairs, None),
            (
                "collect_radial",
                lambda: aev._collect_radial(elem_idxs, neighbors.indices, radial_terms),
                pairs,
                None,
            ),
            (
                "angular_cutoff",
                lambda: discard_outside_cutoff(neighbors, aev.angular.cutoff),
                pairs,
                None,
            ),
            ("neighbors_to_triples", lambda: neighbors_to_triples(angular_neighbors), pairs, triple_count),
            (
                "angular_terms",
                lambda: aev.angular(triples.distances, triples.diff_vectors),
                None,
                triple_count,
            ),
            (
                "collect_angular",
                lambda: aev._collect_angular(
                    elem_idxs,
                    angular_neighbors.indices,
                    triples.central_idxs,
                    triples.side_idxs,
                    triples.diff_signs,
                    angular_terms,
                ),
                None,
                triple_count,
            ),
            ("full_aev_forward", lambda: aev(elem_idxs, coordinates), pairs, triple_count),
            (
                "full_model_forward",
                lambda: model((species, coordinates)).energies,
                pairs,
                triple_count,
            ),
        ]

        for operation_name, operation, op_pairs, op_triples in operations:
            timing = time_callable(operation, device_name, warmup=warmup, repeats=repeats)
            rows.append(
                _profile_row(
                    operation_name,
                    device_name,
                    timing,
                    profile_batch,
                    op_pairs,
                    op_triples,
                )
            )

    optimizer = torch.optim.AdamW(
        model.neural_networks.parameters(),
        lr=config.effective_learning_rate(),
        weight_decay=config.weight_decay,
    )

    def train_step() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        predicted = model((species, coordinates)).energies
        num_atoms = (species >= 0).sum(dim=1, dtype=target_energies.dtype)
        loss = (mse(predicted, target_energies) / num_atoms.sqrt()).mean()
        loss.backward()
        if config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                model.neural_networks.parameters(),
                max_norm=config.max_grad_norm,
            )
        optimizer.step()
        return loss.detach()

    timing = time_callable(train_step, device_name, warmup=warmup, repeats=repeats)
    rows.append(
        _profile_row(
            "single_train_step",
            device_name,
            timing,
            profile_batch,
            pairs,
            triple_count,
        )
    )
    return rows
