"""Finite, non-executing deterministic repair-operator descriptors."""

from .registry import (
    OPERATOR_REGISTRY_INTERFACE,
    OPERATOR_REGISTRY_SCHEMA,
    OperatorDescriptor,
    OperatorRegistry,
    RepairOperatorRegistryError,
)

__all__ = [
    "OPERATOR_REGISTRY_INTERFACE",
    "OPERATOR_REGISTRY_SCHEMA",
    "OperatorDescriptor",
    "OperatorRegistry",
    "RepairOperatorRegistryError",
]
