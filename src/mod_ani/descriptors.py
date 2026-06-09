"""TorchANI-compatible electron-aware descriptor terms."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor
from torchani.aev import AEVComputer, ANIAngular, Radial
from torchani.cutoffs import CutoffArg


class HydrogenLikeRadial(Radial):
    """Screened hydrogen-like radial density features.

    This is the first-pass Aim 1 descriptor: a TorchANI radial term that can be
    dropped into an ``AEVComputer``. It uses a compact density proxy,
    ``(x**l * exp(-x / n))**2``, where ``x = z_eff * r``. The feature is not yet
    a full orbital basis with Laguerre nodes; it is intentionally small and
    stable for benchmarking against ANI radial Gaussians.
    """

    tensors = ["principal", "angular_momentum", "z_eff", "amplitudes"]

    def __init__(
        self,
        principal: Sequence[float],
        angular_momentum: Sequence[float],
        z_eff: Sequence[float],
        amplitudes: Sequence[float] | None = None,
        cutoff: float = 5.2,
        trainable: str | Sequence[str] = (),
        cutoff_fn: CutoffArg = "smooth",
    ) -> None:
        if not (
            len(principal) == len(angular_momentum) == len(z_eff)
            and (amplitudes is None or len(amplitudes) == len(principal))
        ):
            raise ValueError("All feature parameter sequences must have equal length")

        if amplitudes is None:
            amplitudes = [1.0] * len(principal)

        super().__init__(
            cutoff=cutoff,
            trainable=trainable,
            cutoff_fn=cutoff_fn,
            principal=principal,
            angular_momentum=angular_momentum,
            z_eff=z_eff,
            amplitudes=amplitudes,
        )

    def compute(self, distances: Tensor) -> Tensor:
        principal = torch.clamp(self.principal, min=1.0)
        angular_momentum = torch.clamp(self.angular_momentum, min=0.0)
        scaled_radius = torch.clamp(self.z_eff, min=1.0e-8) * distances
        radial_envelope = torch.pow(
            torch.clamp(scaled_radius, min=1.0e-12), angular_momentum
        ) * torch.exp(-scaled_radius / principal)
        return self.amplitudes * radial_envelope.square()

    @classmethod
    def low_quantum_numbers(
        cls,
        cutoff: float = 5.2,
        cutoff_fn: CutoffArg = "smooth",
    ) -> "HydrogenLikeRadial":
        """Initial H/C/N/O-friendly feature set for low-Z energy experiments."""

        return cls(
            principal=[1.0, 2.0, 2.0, 3.0, 3.0],
            angular_momentum=[0.0, 0.0, 1.0, 0.0, 1.0],
            z_eff=[1.0, 1.8, 1.8, 2.6, 2.6],
            cutoff=cutoff,
            cutoff_fn=cutoff_fn,
        )


def make_hydrogen_like_aev(
    num_species: int,
    radial_cutoff: float = 5.2,
    angular_cutoff: float = 3.5,
    cutoff_fn: CutoffArg = "smooth",
    strategy: str = "pyaev",
) -> AEVComputer:
    """Build an AEVComputer with H-like radial terms and standard ANI angular terms."""

    return AEVComputer(
        radial=HydrogenLikeRadial.low_quantum_numbers(
            cutoff=radial_cutoff,
            cutoff_fn=cutoff_fn,
        ),
        angular=ANIAngular.like_2x(cutoff=angular_cutoff, cutoff_fn=cutoff_fn),
        num_species=num_species,
        strategy=strategy,
    )
