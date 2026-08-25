"""Read-only project adapters. Generic inventory never admits mutation."""

from .base import (
    ADAPTER_ID,
    DEFAULT_MAX_DEPTH,
    DEFAULT_MAX_FILES,
    GenericProjectAdapter,
    INVENTORY_AUTHORIZES_MUTATION,
    InventoryBounds,
    InventorySignal,
    PROJECT_ADAPTER_INVENTORY_SCHEMA,
    ProjectAdapter,
    ProjectSupport,
    SignalKind,
    SupportOutcome,
    inspect_project,
)

__all__ = (
    "ADAPTER_ID",
    "DEFAULT_MAX_DEPTH",
    "DEFAULT_MAX_FILES",
    "GenericProjectAdapter",
    "INVENTORY_AUTHORIZES_MUTATION",
    "InventoryBounds",
    "InventorySignal",
    "PROJECT_ADAPTER_INVENTORY_SCHEMA",
    "ProjectAdapter",
    "ProjectSupport",
    "SignalKind",
    "SupportOutcome",
    "inspect_project",
)
