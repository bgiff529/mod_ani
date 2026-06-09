"""Command-line entry point mirroring the proposal notebook."""

from __future__ import annotations

import argparse
from pathlib import Path

from mod_ani.config import ExperimentConfig
from mod_ani.data import download_dataset, prepare_batched_dataset
from mod_ani.training import train_pair


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="TestData", choices=["TestData", "ANI1x", "ANI1ccx"])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-batches", type=int, default=16)
    parser.add_argument("--valid-batches", type=int, default=8)
    parser.add_argument("--run-dir", default="runs")
    args = parser.parse_args()

    config = ExperimentConfig(
        dataset=args.dataset,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        limit_train_batches=args.train_batches,
        limit_valid_batches=args.valid_batches,
        run_dir=Path(args.run_dir),
    )
    dataset = download_dataset(config)
    batched = prepare_batched_dataset(dataset, config)
    results = train_pair(config, batched)
    for kind, history in results.items():
        print(kind, history[-1])


if __name__ == "__main__":
    main()
