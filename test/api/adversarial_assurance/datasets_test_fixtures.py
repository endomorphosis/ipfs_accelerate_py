"""Load datasets assurance fixture constructors from the imported datasets tree."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import ipfs_datasets_py


def _load(name: str):
    package_dir = Path(ipfs_datasets_py.__file__).resolve().parent
    path = (
        package_dir.parent
        / "tests"
        / "unit"
        / "logic"
        / "software_contracts"
        / "adversarial_assurance"
        / f"{name}.py"
    )
    if not path.is_file():
        raise ModuleNotFoundError(
            f"datasets adversarial-assurance fixtures {name} unavailable beside "
            f"imported ipfs_datasets_py ({path})"
        )
    spec = importlib.util.spec_from_file_location(
        f"pcce015.datasets_assurance_fixtures.{name}", path
    )
    if spec is None or spec.loader is None:
        raise ModuleNotFoundError(f"cannot load datasets fixture module {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


receipt_fixtures = _load("test_receipt_contracts")
