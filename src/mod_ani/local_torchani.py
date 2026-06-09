"""Helpers for using the vendored TorchANI checkout during notebook runs."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def torchani_source_path() -> Path:
    return repo_root() / "vendor" / "torchani"


def use_local_torchani() -> Path:
    """Prepend the local TorchANI checkout to ``sys.path`` and reload if needed."""

    source = torchani_source_path()
    if not source.exists():
        raise FileNotFoundError(
            f"Expected TorchANI checkout at {source}. "
            "Run: git clone https://github.com/aiqm/torchani.git vendor/torchani"
        )
    source_str = str(source)
    if sys.path[0] != source_str:
        sys.path = [p for p in sys.path if p != source_str]
        sys.path.insert(0, source_str)

    existing = sys.modules.get("torchani")
    if existing is not None:
        module_file = Path(getattr(existing, "__file__", "")).resolve()
        if source not in module_file.parents:
            for name in list(sys.modules):
                if name == "torchani" or name.startswith("torchani."):
                    del sys.modules[name]
    importlib.import_module("torchani")
    return source
