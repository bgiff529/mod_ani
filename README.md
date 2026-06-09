# mod_ani

Electron-aware modifications to ANI-style neural network potentials.

This project starts with energy prediction by replacing ANI's radial
Behler-Parrinello-like descriptors with hydrogen-like radial electron-density
features, while keeping the rest of the TorchANI training path familiar.

## Setup

```bash
conda env update -f environment.yml
conda activate mod_ani
python -m pip install -e vendor/torchani
python -m pip install -e ".[dev]" --no-deps
```

TorchANI is intentionally used from the local checkout at
`vendor/torchani`, so the code being executed is inspectable in this project
directory. The notebook prints `torchani.__file__` in its first executable
cell to confirm this.

## Notebook

Open [notebooks/proposal_workflow.ipynb](notebooks/proposal_workflow.ipynb)
and run the cells top to bottom. The default configuration uses TorchANI
`TestData` with a small batch limit so the full notebook is runnable as a smoke
benchmark. Set `RUN_FULL_ANI1X = True` in the final section for a larger ANI1x
DFT energy run.

## Initial Direction

- Aim 1: benchmark hydrogen-like radial descriptors against standard ANI radial
  AEVs for molecular energies.
- Aim 2: add rotationally invariant angular/projector descriptors.
- Aim 3: split or augment atom-type networks toward electron-type transfer
  learning across larger atomic numbers.
