"""Dataset download and batching helpers."""

from __future__ import annotations

import shutil
from pathlib import Path

from mod_ani.local_torchani import use_local_torchani

use_local_torchani()

import torchani
from torchani.datasets import ANIBatchedDataset, ANIDataset, BatchedDataset

from mod_ani.config import ExperimentConfig


def _verbose(config: ExperimentConfig) -> bool:
    return bool(getattr(config, "verbose", True))


def download_dataset(config: ExperimentConfig) -> ANIDataset:
    """Download/open a TorchANI built-in dataset."""

    verbose = _verbose(config)
    if verbose:
        print(f"[data] Opening TorchANI dataset {config.dataset} ({config.lot})")
    dataset_factory = getattr(torchani.datasets, config.dataset)
    dataset = dataset_factory(lot=config.lot, download=True)
    if "energies" not in dataset.properties:
        raise ValueError(f"{config.dataset} does not expose an 'energies' property")
    if verbose:
        meta = describe_dataset(dataset)
        print(
            "[data] Ready: "
            f"{meta['num_conformers']} conformers in "
            f"{meta['num_conformer_groups']} groups; "
            f"properties={meta['properties']}"
        )
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
    verbose = _verbose(config)
    if config.refresh_batches and batched_dir.exists():
        if verbose:
            print(f"[data] Removing existing batches: {batched_dir}")
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
    if verbose:
        print(f"[data] Batch directory: {batched_dir}")
        print(f"[data] Batch properties: {properties}")
        print(f"[data] Splits: {splits}; batch_size={config.batch_size}")
    if not batched_dir.exists():
        if verbose:
            print("[data] Creating batches. This can take a while for ANI1x...")
        torchani.datasets.create_batched_dataset(
            dataset,
            dest_path=batched_dir,
            batch_size=config.batch_size,
            splits=splits,
            properties=properties,
            divs_seed=config.divs_seed,
            batch_seed=config.batch_seed,
            verbose=verbose,
        )
    elif verbose:
        print("[data] Reusing existing batches")

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
        if verbose:
            print("[data] Caching batches in RAM")
        train_ds = train_ds.cache(pin_memory=False)
        valid_ds = valid_ds.cache(pin_memory=False)

    if verbose:
        print(f"[data] Training batches: {len(train_ds)}")
        print(f"[data] Validation batches: {len(valid_ds)}")
    return {"training": train_ds, "validation": valid_ds}
