"""Model builders for baseline and electron-aware ANI experiments."""

from __future__ import annotations

import torch

from mod_ani.local_torchani import use_local_torchani

use_local_torchani()

import torchani
from torchani.arch import ANI, Assembler, simple_ani
from torchani.aev import ANIAngular
from torchani.nn import ANINetworks, parse_activation

from mod_ani.config import ExperimentConfig
from mod_ani.descriptors import HydrogenLikeRadial


def build_baseline_model(config: ExperimentConfig) -> ANI:
    """Standard ANI-like model with TorchANI's Gaussian radial AEVs."""

    return simple_ani(
        config.symbols,
        lot=config.lot,
        repulsion=config.repulsion,
        strategy="pyaev",
        periodic_table_index=True,
    )


def build_electron_radial_model(config: ExperimentConfig) -> ANI:
    """ANI model with hydrogen-like radial electron-density descriptors."""

    assembler = Assembler(periodic_table_index=True)
    assembler.set_symbols(config.symbols)
    assembler.set_global_cutoff_fn("smooth")
    assembler.set_aev_computer(
        radial=HydrogenLikeRadial.low_quantum_numbers(cutoff_fn="smooth"),
        angular=ANIAngular.like_2x(cutoff_fn="smooth"),
        strategy="pyaev",
    )
    assembler.set_atomic_networks(
        cls=ANINetworks,
        ctor="ani2x",
        kwargs={"bias": False, "activation": parse_activation("gelu")},
    )
    assembler.set_neighborlist("all_pairs")
    try:
        assembler.set_gsaes_as_self_energies(lot=config.lot)
    except KeyError:
        assembler.set_zeros_as_self_energies()
    if config.repulsion:
        assembler.add_potential(torchani.potentials.RepulsionXTB, name="repulsion")
    return assembler.assemble()


def build_model(config: ExperimentConfig) -> ANI:
    """Build the configured model."""

    if config.model_kind == "baseline":
        return build_baseline_model(config)
    if config.model_kind == "electron_radial":
        return build_electron_radial_model(config)
    raise ValueError(f"Unknown model_kind: {config.model_kind}")


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable model parameters."""

    return sum(p.numel() for p in model.parameters() if p.requires_grad)
