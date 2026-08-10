"""Highest-layer, cold-import-safe agent-supervisor composition boundary.

Only reviewed, provider-free contracts are eager today.  Future convenience
facades belong here, but must be exposed lazily so importing this package never
scans a repository, opens DuckDB, resolves a provider/service, or starts work.
Lower agent-supervisor domain packages must never import ``entrypoints``.
"""

from __future__ import annotations

from typing import Final

from . import contracts as _contracts

ENTRYPOINT_PACKAGE_NAME: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor.entrypoints"
)

# This is the reviewed eager public surface.  The source module owns the closed
# inventory; assigning the exact objects here preserves ``is`` identity across
# package and module imports.
ENTRYPOINT_CONTRACT_EXPORTS: Final[tuple[str, ...]] = tuple(_contracts.__all__)

# Facades will be added by their implementation tasks through module-level lazy
# resolution.  Keeping the population explicit makes an accidental eager
# runtime/service export detectable.
ENTRYPOINT_LAZY_FACADE_EXPORTS: Final[tuple[str, ...]] = ()

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

del _contract_name
