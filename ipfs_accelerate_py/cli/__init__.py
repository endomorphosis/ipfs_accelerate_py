"""CLI package: EAAEF-111 handoff argv plus the historical host ``cli.py``.

``ipfs_accelerate_py/cli.py`` remains the product host.  This package directory
would otherwise shadow that module, so host attributes are loaded lazily from
the sibling file.  ``from ipfs_accelerate_py.cli.supervisor_handoff import …``
does not import the host.
"""

from __future__ import annotations

from importlib import import_module
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from typing import Any

_HOST_MODULE: ModuleType | None = None
_HOST_PATH = Path(__file__).resolve().parent.parent / "cli.py"
_HOST_NAME = "ipfs_accelerate_py._cli_host"

__all__ = ["main", "supervisor_handoff"]


def _load_host() -> ModuleType:
    global _HOST_MODULE
    if _HOST_MODULE is not None:
        return _HOST_MODULE
    spec = spec_from_file_location(_HOST_NAME, _HOST_PATH)
    if spec is None or spec.loader is None:
        raise ImportError("historical ipfs_accelerate_py/cli.py is not importable")
    import sys

    module = module_from_spec(spec)
    sys.modules.setdefault(_HOST_NAME, module)
    spec.loader.exec_module(module)
    _HOST_MODULE = module
    return module


def __getattr__(name: str) -> Any:
    if name == "supervisor_handoff":
        return import_module(".supervisor_handoff", __name__)
    return getattr(_load_host(), name)


def __dir__() -> list[str]:
    names = {"main", "supervisor_handoff"}
    names.update(globals())
    try:
        names.update(name for name in dir(_load_host()) if not name.startswith("_"))
    except Exception:
        pass
    return sorted(names)
