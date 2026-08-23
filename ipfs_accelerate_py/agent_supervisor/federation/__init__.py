"""CausalAbstractionSupervisorFederation package.

The package import is deliberately cold: it opens no database, loads no DuckDB
extension, starts no worker, and contacts no provider.  Callers import the
closed contracts or a concrete state-owner component explicitly.
"""

from __future__ import annotations

from typing import Final

FEDERATION_PACKAGE_NAME: Final[str] = "ipfs_accelerate_py.agent_supervisor.federation"
FEDERATION_INTERFACE: Final[str] = "CausalAbstractionSupervisorFederation@1"
FEDERATION_PROGRAM_ID: Final[str] = "agent-supervisor-causal-event-federation-v1"
FEDERATION_OWNED_MODULES: Final[tuple[str, ...]] = (
    "agent_registry",
    "bootstrap_runtime",
    "budgets",
    "causal_abstraction",
    "causal_contracts",
    "causal_evidence",
    "causal_frontier",
    "causal_graph",
    "contracts",
    "deduplication",
    "durable_event_router",
    "event_router",
    "event_wait",
    "events",
    "fixed_point",
    "identity",
    "lifecycle",
    "merge",
    "outbox",
    "outbox_worker",
    "parallel_frontier",
    "policy",
    "proof_projection",
    "rebalancing",
    "recovery",
    "registry",
    "retrieval_projection",
    "scheduler",
    "semantic_projection",
    "sharding",
    "subscriptions",
    "supervisor_registry",
    "supervisor_runtime",
    "trigger",
    "work_stealing",
    "world_snapshot",
)

__all__: Final[tuple[str, ...]] = (
    "FEDERATION_INTERFACE",
    "FEDERATION_OWNED_MODULES",
    "FEDERATION_PACKAGE_NAME",
    "FEDERATION_PROGRAM_ID",
)
