"""Lazy installed-package capability discovery for PCCE v0.1."""

from __future__ import annotations

import importlib
import importlib.metadata
from dataclasses import dataclass
from typing import Any

from ipfs_accelerate_py.proof_context.compatibility import CompatibilityError

PORT_SCHEMA = "ipfs-accelerate.proof-context.v0.1"


class DependencyUnavailable(RuntimeError):
    """Required installed authority is absent or incompatible."""

    reason = "unavailable"


class DependencyInvalid(RuntimeError):
    """Installed authority failed identity or version checks."""

    reason = "invalid"


@dataclass(frozen=True)
class Capability:
    name: str
    distribution: str
    available: bool
    reason: str | None = None
    module: str | None = None
    version: str | None = None


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _try_import(module: str) -> Any | None:
    try:
        return importlib.import_module(module)
    except Exception:
        return None


def resolve_datasets() -> Capability:
    version = _distribution_version("ipfs_datasets_py")
    module = _try_import("ipfs_datasets_py.proof_context.provider")
    if module is None:
        return Capability(
            name="datasets",
            distribution="ipfs_datasets_py",
            available=False,
            reason="unavailable",
            version=version,
        )
    return Capability(
        name="datasets",
        distribution="ipfs_datasets_py",
        available=True,
        module="ipfs_datasets_py.proof_context.provider",
        version=version,
    )


def resolve_kit() -> Capability:
    version = _distribution_version("ipfs_kit_py")
    module = _try_import("ipfs_kit_py.proof_context.state_store")
    if module is None:
        return Capability(
            name="kit",
            distribution="ipfs_kit_py",
            available=False,
            reason="unavailable",
            version=version,
        )
    return Capability(
        name="kit",
        distribution="ipfs_kit_py",
        available=True,
        module="ipfs_kit_py.proof_context.state_store",
        version=version,
    )


def require_production_capability(cap: Capability) -> Capability:
    if not cap.available:
        raise DependencyUnavailable(
            f"{cap.name} v0.1 port is unavailable; this is not success"
        )
    return cap


def resolve_v01_surface() -> tuple[Capability, ...]:
    return (resolve_datasets(), resolve_kit())
