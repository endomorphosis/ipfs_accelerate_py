"""Highest-layer, cold-import-safe agent-supervisor composition boundary.

Only reviewed, provider-free contracts are eager today.  Convenience facades
are exposed lazily so importing this package never scans a repository, opens
DuckDB, resolves a provider/service, or starts work. Lower agent-supervisor
domain packages must never import ``entrypoints``.
"""

from __future__ import annotations

from typing import Any, Final

from . import contracts as _contracts

ENTRYPOINT_PACKAGE_NAME: Final = (
    "ipfs_accelerate_py.agent_supervisor.entrypoints"
)

# This is the reviewed eager public surface.  The source module owns the closed
# inventory; assigning the exact objects here preserves ``is`` identity across
# package and module imports.
ENTRYPOINT_CONTRACT_EXPORTS: Final[tuple[str, ...]] = tuple(_contracts.__all__)

# ASE3-009 production Python facade (lazy — never resolved at import time).
ENTRYPOINT_LAZY_FACADE_EXPORTS: Final[tuple[str, ...]] = (
    "Supervisor",
    "SupervisorRun",
    "SupervisorObservation",
    "SupervisorError",
    "SupervisorConfigurationError",
    "SupervisorAmbiguityError",
    "SupervisorUnavailableError",
    "ProductionServiceCompositionManifest",
    "ProductionServiceComposition",
    "resolve_production_composition",
    "build_production_composition_manifest",
    "ServiceCompositionError",
    "ActivationNotReadyError",
    "ConfigurationUnavailableError",
)

# Every listed package is below this composition layer.  It may be composed
# lazily by a facade, but none may import this package in return.
ENTRYPOINT_LOWER_DOMAIN_PACKAGES: Final[tuple[str, ...]] = (
    "core",
    "contracts",
    "control",
    "task_sources",
    "context",
    "analysis",
    "proof",
    "objectives",
    "planning",
    "validation",
    "prompt",
    "merge",
    "rescue",
    "runtime",
    "self_improvement",
    "todo_daemon",
    "integrations",
)

for _contract_name in ENTRYPOINT_CONTRACT_EXPORTS:
    globals()[_contract_name] = getattr(_contracts, _contract_name)

# Drop any previously cached lazy facade bindings (importlib.reload keeps the
# module dict; facades must stay unresolved until first attribute access).
for _lazy_name in (
    "Supervisor",
    "SupervisorRun",
    "SupervisorObservation",
    "SupervisorError",
    "SupervisorConfigurationError",
    "SupervisorAmbiguityError",
    "SupervisorUnavailableError",
    "ProductionServiceCompositionManifest",
    "ProductionServiceComposition",
    "resolve_production_composition",
    "build_production_composition_manifest",
    "ServiceCompositionError",
    "ActivationNotReadyError",
    "ConfigurationUnavailableError",
):
    globals().pop(_lazy_name, None)

_PACKAGE_METADATA_EXPORTS: Final[tuple[str, ...]] = (
    "ENTRYPOINT_PACKAGE_NAME",
    "ENTRYPOINT_CONTRACT_EXPORTS",
    "ENTRYPOINT_LAZY_FACADE_EXPORTS",
    "ENTRYPOINT_LOWER_DOMAIN_PACKAGES",
)

__all__: Final[tuple[str, ...]] = (
    *_PACKAGE_METADATA_EXPORTS,
    *ENTRYPOINT_CONTRACT_EXPORTS,
    *ENTRYPOINT_LAZY_FACADE_EXPORTS,
)

_FACADE_ATTRS: Final[frozenset[str]] = frozenset(
    {
        "Supervisor",
        "SupervisorRun",
        "SupervisorObservation",
        "SupervisorError",
        "SupervisorConfigurationError",
        "SupervisorAmbiguityError",
        "SupervisorUnavailableError",
    }
)
_FACTORY_ATTRS: Final[frozenset[str]] = frozenset(
    {
        "ProductionServiceCompositionManifest",
        "ProductionServiceComposition",
        "resolve_production_composition",
        "build_production_composition_manifest",
        "ServiceCompositionError",
        "ActivationNotReadyError",
        "ConfigurationUnavailableError",
    }
)


def __getattr__(name: str) -> Any:
    if name in _FACADE_ATTRS:
        from . import facade as _facade

        value = getattr(_facade, name)
        globals()[name] = value
        return value
    if name in _FACTORY_ATTRS:
        from . import service_factory as _service_factory

        value = getattr(_service_factory, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


del _contract_name
