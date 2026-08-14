"""Accelerate public facade for IncrementalProofSealer (IPS-043).

Exports the seven requested Python APIs lazily so a cold import of this package
performs no optional capability import, install, key generation, process spawn,
network I/O, or user-state access.  Submodules remain independently importable.

Production seals reject simulated evidence.  Missing optional capabilities are
typed via :func:`report_optional_capabilities` rather than raised as opaque
import failures.
"""

from __future__ import annotations

from typing import Any, Final

# Package-level evidence subsets for the public surface (IPS-043 / IPS-G090).
PUBLIC_API_SUBSET: Final[str] = "ips/public-api@1"
CLI_SUBSET: Final[str] = "ips/cli@1"
IMPORT_HERMETICITY_SUBSET: Final[str] = "ips/import-hermeticity@1"
PACKAGE_SCHEMA_VERSION: Final[str] = "1"

# Closed ordered public API freeze (plan §11 / IPS-043 interfaces).
PUBLIC_API_NAMES: Final[tuple[str, ...]] = (
    "create_full_checkpoint",
    "create_incremental_plan",
    "execute_incremental_plan",
    "verify_seal",
    "explain_reuse",
    "explain_invalidation",
    "compare_full_and_incremental",
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
    "report_optional_capabilities": (__name__, "report_optional_capabilities"),
    # Seven requested public APIs.
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
    # Supporting types commonly consumed with the public APIs.
    "FullCheckpointSeal": (".full_checkpoint", "FullCheckpointSeal"),
    "FullCheckpointError": (".full_checkpoint", "FullCheckpointError"),
    "FullCheckpointReason": (".full_checkpoint", "FullCheckpointReason"),
    "RepositoryStateView": (".full_checkpoint", "RepositoryStateView"),
    "RequiredUnitEvidence": (".full_checkpoint", "RequiredUnitEvidence"),
    "VerificationPolicyView": (".full_checkpoint", "VerificationPolicyView"),
    "IncrementalProofPlan": (".planner", "IncrementalProofPlan"),
    "ParentSealContext": (".planner", "ParentSealContext"),
    "PlanMode": (".planner", "PlanMode"),
    "UnitPlanningInput": (".planner", "UnitPlanningInput"),
    "PlannerError": (".planner", "PlannerError"),
    "IncrementalProofResult": (".executor", "IncrementalProofResult"),
    "ResourcePolicy": (".executor", "ResourcePolicy"),
    "ExecutionOutcome": (".executor", "ExecutionOutcome"),
    "ExecutorError": (".executor", "ExecutorError"),
    "SealVerificationResult": (".verification", "SealVerificationResult"),
    "SealVerificationReason": (".verification", "SealVerificationReason"),
    "VerificationError": (".verification", "VerificationError"),
    "ProofReuseExplanation": (".explanations", "ProofReuseExplanation"),
    "ProofInvalidationExplanation": (
        ".explanations",
        "ProofInvalidationExplanation",
    ),
    "FullIncrementalComparison": (".explanations", "FullIncrementalComparison"),
    "ExplanationError": (".explanations", "ExplanationError"),
    "ProofBackendCapability": (".backends", "ProofBackendCapability"),
    "BackendAvailabilityStatus": (".backends", "BackendAvailabilityStatus"),
    "probe_backend_capability": (".backends", "probe_backend_capability"),
}

__all__ = tuple(sorted(_EXPORTS))


def report_optional_capabilities() -> dict[str, Any]:
    """Return typed availability of optional kit/datasets/backend surfaces.

    Never installs tools, spawns processes, opens network sockets, generates
    keys, or mutates user state.  Missing optional capabilities are labeled
    with closed status strings rather than raised as import errors.
    """

    capabilities: dict[str, Any] = {
        "schema": (
            "ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/"
            "optional-capabilities@1"
        ),
        "evidence_subset": PUBLIC_API_SUBSET,
        "kit_store": {
            "status": "unavailable",
            "reason_code": "optional_kit_not_injected",
            "message": (
                "kit store adapter is optional and must be injected; "
                "cold public API does not open kit storage"
            ),
        },
        "datasets_semantic": {
            "status": "unavailable",
            "reason_code": "optional_datasets_not_loaded",
            "message": (
                "datasets semantic modules load only when a public API call "
                "requires them; cold import does not load datasets"
            ),
        },
        "backends": {},
    }

    # Probe only known hermetic backend identities without recursion material.
    try:
        from .backends import (
            BackendAvailabilityStatus,
            KNOWN_BACKEND_IDS,
            probe_backend_capability,
        )
    except Exception as exc:  # pragma: no cover - fail-closed typed gap
        capabilities["backends"] = {
            "status": "unavailable",
            "reason_code": "backend_probe_module_unavailable",
            "message": f"backend capability module unavailable: {exc}",
        }
        return capabilities

    backend_report: dict[str, Any] = {}
    for backend_id in sorted(KNOWN_BACKEND_IDS):
        try:
            capability = probe_backend_capability(
                backend_id,
                allow_recursion_probe=False,
            )
            backend_report[backend_id] = {
                "status": capability.status.value,
                "production_seal_allowed": capability.production_seal_allowed,
                "recursive_verification": capability.recursive_verification,
                "reason_code": capability.reason_code,
                "message": capability.message,
            }
        except Exception as exc:
            backend_report[backend_id] = {
                "status": BackendAvailabilityStatus.UNKNOWN.value,
                "production_seal_allowed": False,
                "recursive_verification": False,
                "reason_code": "probe_error",
                "message": str(exc),
            }
    capabilities["backends"] = backend_report
    return capabilities


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
