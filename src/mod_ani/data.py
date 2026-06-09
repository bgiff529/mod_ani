"""Dataset download and batching helpers."""

from __future__ import annotations

import shutil
from pathlib import Path

from mod_ani.local_torchani import use_local_torchani

use_local_torchani()

import torchani
from torchani.datasets import ANIBatchedDataset, ANIDataset, BatchedDataset

from mod_ani.config import ExperimentConfig


def download_dataset(config: ExperimentConfig) -> ANIDataset:
    """Download/open a TorchANI built-in dataset."""

    dataset_factory = getattr(torchani.datasets, config.dataset)
    dataset = dataset_factory(lot=config.lot, download=True)
    if "energies" not in dataset.properties:
        raise ValueError(f"{config.dataset} does not expose an 'energies' property")
    return dataset


def describe_dataset(dataset: ANIDataset) -> dict[str, object]:
    """Return notebook-friendly dataset metadata."""

    return {
        "grouping": dataset.grouping,
        "properties": sorted(dataset.properties),
        "num_conformer_groups": dataset.num_conformer_groups,
        "num_conformers": dataset.num_conformers,
        "symbols": tuple(dataset.symbols) if dataset.grouping != "legacy" else (),
    }


def prepare_batched_dataset(
    dataset: ANIDataset,
    config: ExperimentConfig,
) -> dict[str, BatchedDataset]:
    """Create or reuse train/validation batches."""

    batched_dir = Path(config.data_dir) / "batched" / config.dataset / config.lot
    if config.refresh_batches and batched_dir.exists():
        shutil.rmtree(batched_dir)

    splits = {
        "training": 1.0 - config.validation_fraction,
        "validation": config.validation_fraction,
    }
    properties = tuple(
        prop
        for prop in ("species", "coordinates", "energies", "forces")
        if prop in dataset.properties
    )
    if not batched_dir.exists():
        torchani.datasets.create_batched_dataset(
            dataset,
            dest_path=batched_dir,
            batch_size=config.batch_size,
            splits=splits,
            properties=properties,
            divs_seed=config.divs_seed,
            batch_seed=config.batch_seed,
        )

    train_ds: BatchedDataset = ANIBatchedDataset(
        batched_dir,
        split="training",
        limit=config.limit_train_batches,
        properties=properties,
    )
    valid_ds: BatchedDataset = ANIBatchedDataset(
        batched_dir,
        split="validation",
        limit=config.limit_valid_batches,
        properties=properties,
    )

    if config.cache_batches:
        train_ds = train_ds.cache(pin_memory=False)
        valid_ds = valid_ds.cache(pin_memory=False)

    return {"training": train_ds, "validation": valid_ds}
