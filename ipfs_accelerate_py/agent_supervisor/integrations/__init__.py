"""ASREF domain package: integrations.

Modules owned by move_map target package ``integrations`` live here.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "test_reuse_capabilities",
    "ipfs_datasets_test_certificate_provider",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        return import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
