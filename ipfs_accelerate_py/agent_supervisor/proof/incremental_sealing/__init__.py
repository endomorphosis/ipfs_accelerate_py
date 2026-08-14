"""Accelerate public facade for IncrementalProofSealer (IPS-043).

Exports the seven required public APIs lazily so a cold import of this package
performs no process spawn, network I/O, key generation, user-state access, or
optional backend probing.  Submodules remain independently importable.

Public APIs
-----------
``create_full_checkpoint``, ``create_incremental_plan``,
``execute_incremental_plan``, ``verify_seal``, ``explain_reuse``,
``explain_invalidation``, ``compare_full_and_incremental``.

Optional backends (ProveKit, recursive aggregation, IPFS) are reported as
typed availability results when probed; absence is never upgraded to success.
"""

from __future__ import annotations

from typing import Any

# Package-level evidence subsets for the public surface (IPS-043 / IPS-G090).
PUBLIC_API_SUBSET = "ips/public-api@1"
CLI_SUBSET = "ips/cli@1"
IMPORT_HERMETICITY_SUBSET = "ips/import-hermeticity@1"
PACKAGE_SCHEMA_VERSION = "1"

# Closed ordered public function freeze (plan §11).
PUBLIC_API_NAMES: tuple[str, ...] = (
    "create_full_checkpoint",
    "create_incremental_plan",
    "execute_incremental_plan",
    "verify_seal",
    "explain_reuse",
    "explain_invalidation",
    "compare_full_and_incremental",
)

# Focused zk-seal CLI operation freeze (IPS-043 effects; plan §11).
CLI_OPERATIONS: tuple[str, ...] = (
    "full",
    "incremental",
    "verify",
    "plan",
    "explain-reuse",
    "explain-invalidation",
    "benchmark",
    "cache-status",
    "force-full",
)

# Lazy export table: public name -> (relative submodule, attribute).
# Keep this closed and explicit; unknown attributes fail closed.
_EXPORTS: dict[str, tuple[str, str]] = {
    # Package metadata (resolved from this module).
    "PUBLIC_API_SUBSET": (__name__, "PUBLIC_API_SUBSET"),
    "CLI_SUBSET": (__name__, "CLI_SUBSET"),
    "IMPORT_HERMETICITY_SUBSET": (__name__, "IMPORT_HERMETICITY_SUBSET"),
    "PACKAGE_SCHEMA_VERSION": (__name__, "PACKAGE_SCHEMA_VERSION"),
    "PUBLIC_API_NAMES": (__name__, "PUBLIC_API_NAMES"),
    "CLI_OPERATIONS": (__name__, "CLI_OPERATIONS"),
    # Required public APIs (plan §11).
    "create_full_checkpoint": (".full_checkpoint", "create_full_checkpoint"),
    "create_incremental_plan": (".planner", "create_incremental_plan"),
    "execute_incremental_plan": (".executor", "execute_incremental_plan"),
    "verify_seal": (".verification", "verify_seal"),
    "explain_reuse": (".explanations", "explain_reuse"),
    "explain_invalidation": (".explanations", "explain_invalidation"),
    "compare_full_and_incremental": (
        ".explanations",
        "compare_full_and_incremental",
    ),
    # Related public surface used by CLI / callers.
    "compact_seal_chain": (".compaction", "compact_seal_chain"),
    "probe_backend_capability": (".backends", "probe_backend_capability"),
    "optional_capability_status": (__name__, "optional_capability_status"),
    # CLI entrypoints (lazy; cold package import does not load cli).
    "main": (".cli", "main"),
    "build_parser": (".cli", "build_parser"),
    "discovery_manifest": (".cli", "discovery_manifest"),
    # Result / request types commonly needed with the public APIs.
    "FullCheckpointSeal": (".full_checkpoint", "FullCheckpointSeal"),
    "RepositoryStateView": (".full_checkpoint", "RepositoryStateView"),
    "VerificationPolicyView": (".full_checkpoint", "VerificationPolicyView"),
    "RequiredUnitEvidence": (".full_checkpoint", "RequiredUnitEvidence"),
    "IncrementalProofPlan": (".planner", "IncrementalProofPlan"),
    "ParentSealContext": (".planner", "ParentSealContext"),
    "UnitPlanningInput": (".planner", "UnitPlanningInput"),
    "PlanMode": (".planner", "PlanMode"),
    "IncrementalProofResult": (".executor", "IncrementalProofResult"),
    "ResourcePolicy": (".executor", "ResourcePolicy"),
    "ExecutionOutcome": (".executor", "ExecutionOutcome"),
    "SealVerificationResult": (".verification", "SealVerificationResult"),
    "SealVerificationReason": (".verification", "SealVerificationReason"),
    "ProofReuseExplanation": (".explanations", "ProofReuseExplanation"),
    "ProofInvalidationExplanation": (
        ".explanations",
        "ProofInvalidationExplanation",
    ),
    "FullIncrementalComparison": (".explanations", "FullIncrementalComparison"),
    "ProofBackendCapability": (".backends", "ProofBackendCapability"),
    "BackendAvailabilityStatus": (".backends", "BackendAvailabilityStatus"),
    "CapabilityReasonCode": (".backends", "CapabilityReasonCode"),
    "RetentionPolicy": (".compaction", "RetentionPolicy"),
    "CompactionOutcome": (".compaction", "CompactionOutcome"),
}

__all__ = tuple(sorted(_EXPORTS))


def optional_capability_status(
    backend_id: str,
    *,
    availability_overrides: dict[str, bool] | None = None,
) -> dict[str, Any]:
    """Return a typed optional-capability report without auto-install.

    Missing optional backends (for example ProveKit) surface as closed
    availability statuses and reason codes.  This never installs tools,
    generates keys, or opens network connections.
    """

    from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.backends import (
        probe_backend_capability,
    )

    capability = probe_backend_capability(
        backend_id,
        allow_recursion_probe=False,
        availability_overrides=availability_overrides,
    )
    reason = capability.reason_code
    reason_code = str(getattr(reason, "value", reason) or "")
    status = capability.status
    status_value = str(getattr(status, "value", status))
    disposition = capability.aggregation_disposition
    disposition_value = str(getattr(disposition, "value", disposition))
    return {
        "schema": (
            "ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/"
            "optional-capability-status@1"
        ),
        "evidence_subset": PUBLIC_API_SUBSET,
        "backend_id": capability.backend_id,
        "status": status_value,
        "reason_code": reason_code,
        "message": capability.message,
        "production_seal_allowed": bool(capability.production_seal_allowed),
        "can_prove": bool(capability.can_prove),
        "can_verify": bool(capability.can_verify),
        "recursive_verification": bool(capability.recursive_verification),
        "aggregation_disposition": disposition_value,
        "auto_install": False,
        "network_accessed": False,
        "keys_generated": False,
        "user_state_mutated": False,
        "processes_started": False,
    }


def __getattr__(name: str) -> Any:
    """Resolve public contracts on demand without eager submodule imports."""

    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from exc
    if module_name == __name__:
        return globals()[attr_name]
    from importlib import import_module

    module = import_module(module_name, package=__name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
