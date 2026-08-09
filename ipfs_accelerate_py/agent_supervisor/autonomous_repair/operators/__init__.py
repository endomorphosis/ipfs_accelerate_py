"""Finite reviewed repair operators for deterministic contract repair (DCR-040+).

The package is intentionally import-light.  Concrete operator modules register
into :mod:`registry`; this package root does not eagerly load them.
"""

from __future__ import annotations

from .registry import (
    REPAIR_OPERATOR_INTERFACE,
    REPAIR_OPERATOR_REGISTRY_INTERFACE,
    OperatorDescriptor,
    OperatorFamily,
    OperatorKind,
    OperatorRegistry,
    RepairOperatorRegistryError,
    build_default_operator_registry,
    default_operator_registry_id,
)

__all__ = (
    "REPAIR_OPERATOR_INTERFACE",
    "REPAIR_OPERATOR_REGISTRY_INTERFACE",
    "OperatorDescriptor",
    "OperatorFamily",
    "OperatorKind",
    "OperatorRegistry",
    "RepairOperatorRegistryError",
    "build_default_operator_registry",
    "default_operator_registry_id",
)
