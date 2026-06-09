# mod_ani

Electron-aware modifications to ANI-style neural network potentials.

This project starts with energy prediction by replacing ANI's radial
Behler-Parrinello-like descriptors with hydrogen-like radial electron-density
features, while keeping the rest of the TorchANI training path familiar.

## Setup

```bash
conda env update -f environment.yml
conda activate mod_ani
python -m pip install -e ".[dev]"
```

## Initial Direction

- Aim 1: benchmark hydrogen-like radial descriptors against standard ANI radial
  AEVs for molecular energies.
- Aim 2: add rotationally invariant angular/projector descriptors.
- Aim 3: split or augment atom-type networks toward electron-type transfer
  learning across larger atomic numbers.
