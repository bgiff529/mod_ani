"""Model builders for baseline and electron-aware ANI experiments."""

from __future__ import annotations

import torch

from mod_ani.local_torchani import use_local_torchani

use_local_torchani()

import torchani
from torchani.arch import ANI, Assembler, simple_ani
from torchani.aev import AEVComputer, ANIAngular
from torchani.nn import ANINetworks, SpeciesConverter, parse_activation
from torchani.sae import SelfEnergy
from torchani.tuples import SpeciesEnergies

from mod_ani.config import ExperimentConfig
from mod_ani.descriptors import HydrogenLikeRadial


class ElectronChannelNetworks(torch.nn.Module):
    """Electron/orbital-channel networks that replace atom-type networks.

    Each channel sees one hydrogen-like radial feature across neighbor species, the
    shared angular AEV block, and a one-hot central atom identity. Channel outputs
    are summed to produce one atomic neural-network energy.
    """

    def __init__(
        self,
        num_species: int,
        num_radial_features: int,
        angular_len: int,
        hidden_dims: tuple[int, ...] = (128, 64),
    ) -> None:
        super().__init__()
        self.num_species = num_species
        self.num_radial_features = num_radial_features
        self.angular_len = angular_len
        input_dim = num_species + angular_len + num_species
        self.channels = torch.nn.ModuleList(
            self._make_channel(input_dim, hidden_dims) for _ in range(num_radial_features)
        )

    @staticmethod
    def _make_channel(input_dim: int, hidden_dims: tuple[int, ...]) -> torch.nn.Sequential:
        layers: list[torch.nn.Module] = []
        previous = input_dim
        for hidden_dim in hidden_dims:
            layers.append(torch.nn.Linear(previous, hidden_dim, bias=False))
            layers.append(torch.nn.GELU())
            previous = hidden_dim
        output = torch.nn.Linear(previous, 1, bias=False)
        torch.nn.init.zeros_(output.weight)
        layers.append(output)
        return torch.nn.Sequential(*layers)

    def forward(self, elem_idxs: torch.Tensor, aevs: torch.Tensor) -> torch.Tensor:
        radial_len = self.num_species * self.num_radial_features
        radial = aevs[..., :radial_len].reshape(
            *aevs.shape[:-1], self.num_species, self.num_radial_features
        )
        angular = aevs[..., radial_len:]
        valid_atoms = elem_idxs >= 0
        safe_elem_idxs = elem_idxs.clamp(min=0)
        element_one_hot = torch.nn.functional.one_hot(
            safe_elem_idxs,
            num_classes=self.num_species,
        ).to(dtype=aevs.dtype)
        element_one_hot = element_one_hot * valid_atoms.unsqueeze(-1).to(dtype=aevs.dtype)

        channel_energies = []
        for channel_idx, channel in enumerate(self.channels):
            channel_radial = radial[..., :, channel_idx]
            channel_input = torch.cat([channel_radial, angular, element_one_hot], dim=-1)
            channel_energies.append(channel(channel_input).squeeze(-1))
        atomic_energies = torch.stack(channel_energies, dim=0).sum(dim=0)
        return atomic_energies.masked_fill(~valid_atoms, 0.0)


class ElectronChannelANI(torch.nn.Module):
    """ANI-style energy model with electron-channel networks."""

    periodic_table_index = True

    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        self.symbols = tuple(config.symbols)
        self.species_converter = SpeciesConverter(self.symbols)
        self.aev_computer = AEVComputer(
            radial=HydrogenLikeRadial.low_quantum_numbers(cutoff_fn="smooth"),
            angular=ANIAngular.like_2x(cutoff_fn="smooth"),
            num_species=len(self.symbols),
            strategy="pyaev",
            neighborlist="all_pairs",
        )
        self.neighborlist = self.aev_computer.neighborlist
        self.neural_networks = ElectronChannelNetworks(
            num_species=len(self.symbols),
            num_radial_features=self.aev_computer.radial.num_feats,
            angular_len=self.aev_computer.angular_len,
        )
        try:
            functional, basis_set = config.lot.split("-", 1)
            self.energy_shifter = SelfEnergy.with_gsaes(self.symbols, functional, basis_set)
        except KeyError:
            self.energy_shifter = SelfEnergy(self.symbols, [0.0] * len(self.symbols))

    def forward(self, species_coordinates: tuple[torch.Tensor, torch.Tensor]) -> SpeciesEnergies:
        species, coordinates = species_coordinates
        elem_idxs = self.species_converter(species, nop=False)
        aevs = self.aev_computer(elem_idxs, coordinates)
        atomic_energies = self.neural_networks(elem_idxs, aevs)
        energies = atomic_energies.sum(dim=-1) + self.energy_shifter(elem_idxs)
        return SpeciesEnergies(elem_idxs, energies)


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


def build_electron_channel_model(config: ExperimentConfig) -> ElectronChannelANI:
    """ANI-style model with electron/orbital-channel neural networks."""

    if config.repulsion:
        raise ValueError("electron_channels does not yet support repulsion=True")
    return ElectronChannelANI(config)


def build_model(config: ExperimentConfig) -> torch.nn.Module:
    """Build the configured model."""

    if config.model_kind == "baseline":
        return build_baseline_model(config)
    if config.model_kind == "electron_radial":
        return build_electron_radial_model(config)
    if config.model_kind == "electron_channels":
        return build_electron_channel_model(config)
    raise ValueError(f"Unknown model_kind: {config.model_kind}")


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable model parameters."""

    return sum(p.numel() for p in model.parameters() if p.requires_grad)
