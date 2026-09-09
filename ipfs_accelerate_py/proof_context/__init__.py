"""Accelerator v0.1 proof-context runtime package.

Cold import performs no network, process, or filesystem mutation and does
not search sibling checkouts. Public names resolve lazily through
``__getattr__`` so importing this package does not bind datasets, kit,
lifecycle, recovery, or a model provider.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Final

SCHEMA: Final[str] = "ipfs-accelerate.proof-context.v0.1"

# name -> (relative module, attribute)
_LAZY_EXPORTS: Final[dict[str, tuple[str, str]]] = {
    "COMPATIBILITY_MATRIX_CONTENT_ID": (".facade", "COMPATIBILITY_MATRIX_CONTENT_ID"),
    "CONTRACT_SCHEMA_PREFIX": (".facade", "CONTRACT_SCHEMA_PREFIX"),
    "CONTRACT_VERSION": (".facade", "CONTRACT_VERSION"),
    "ENGINE_RECORD_SCHEMA": (".facade", "ENGINE_RECORD_SCHEMA"),
    "EPIC_A_GATE_CONTENT_ID": (".facade", "EPIC_A_GATE_CONTENT_ID"),
    "EPIC_A_GATE_TASK": (".facade", "EPIC_A_GATE_TASK"),
    "INTERFACE": (".facade", "INTERFACE"),
    "INSTANCE_OPERATIONS": (".facade", "INSTANCE_OPERATIONS"),
    "MODES": (".facade", "MODES"),
    "OPERATION_CONTRACTS": (".facade", "OPERATION_CONTRACTS"),
    "OPERATIONS": (".facade", "OPERATIONS"),
    "PCCE_006_CONTENT_ID": (".facade", "PCCE_006_CONTENT_ID"),
    "PROVIDER_BOUND": (".facade", "PROVIDER_BOUND"),
    "PROVENANCES": (".facade", "PROVENANCES"),
    "SIBLING_LAYOUT_REQUIRED": (".facade", "SIBLING_LAYOUT_REQUIRED"),
    "STATUSES": (".facade", "STATUSES"),
    "EngineIdentities": (".facade", "EngineIdentities"),
    "EnginePorts": (".facade", "EnginePorts"),
    "EngineRecord": (".facade", "EngineRecord"),
    "FacadeError": (".facade", "FacadeError"),
    "ProofCarryingContextEngine": (".facade", "ProofCarryingContextEngine"),
    "public_signature_snapshot": (".facade", "public_signature_snapshot"),
    "LIFECYCLE_CID": (".lifecycle", "LIFECYCLE_CID"),
    "LIFECYCLE_DESCRIPTOR": (".lifecycle", "LIFECYCLE_DESCRIPTOR"),
    "PatchLifecycle": (".lifecycle", "PatchLifecycle"),
    "LifecycleIdentities": (".lifecycle", "LifecycleIdentities"),
    "LifecyclePorts": (".lifecycle", "LifecyclePorts"),
    "LifecycleRecord": (".lifecycle", "LifecycleRecord"),
    "STAGES": (".lifecycle", "STAGES"),
    "POLICY_CID": (".policy", "POLICY_CID"),
    "POLICY_DESCRIPTOR": (".policy", "POLICY_DESCRIPTOR"),
    "LIVE_MODES": (".policy", "LIVE_MODES"),
    "FORBIDDEN_EVIDENCE": (".policy", "FORBIDDEN_EVIDENCE"),
    "admit_mode": (".policy", "admit_mode"),
    "RESULT_STATE_CID": (".results", "RESULT_STATE_CID"),
    "RESULT_DESCRIPTOR": (".results", "RESULT_DESCRIPTOR"),
    "ResultRecord": (".results", "ResultRecord"),
    "ProofContextError": (".errors", "ProofContextError"),
    "ERRORS": (".errors", "ERRORS"),
    "error_for": (".errors", "error_for"),
    "RECOVERY_CID": (".recovery", "RECOVERY_CID"),
    "RECOVERY_DESCRIPTOR": (".recovery", "RECOVERY_DESCRIPTOR"),
    "RecoveryCoordinator": (".recovery", "RecoveryCoordinator"),
    "AttemptIdentity": (".recovery", "AttemptIdentity"),
    "FencedCheckpointStore": (".recovery", "FencedCheckpointStore"),
    "Capability": (".dependencies", "Capability"),
    "DependencyUnavailable": (".dependencies", "DependencyUnavailable"),
    "resolve_datasets": (".dependencies", "resolve_datasets"),
    "resolve_kit": (".dependencies", "resolve_kit"),
    "resolve_v01_surface": (".dependencies", "resolve_v01_surface"),
    "RUNTIME_CID": (".bootstrap", "RUNTIME_CID"),
    "RUNTIME_DESCRIPTOR": (".bootstrap", "RUNTIME_DESCRIPTOR"),
    "RuntimeBundle": (".bootstrap", "RuntimeBundle"),
    "RuntimeOptions": (".bootstrap", "RuntimeOptions"),
    "open_engine": (".bootstrap", "open_engine"),
    "open_runtime": (".bootstrap", "open_runtime"),
    "runtime_descriptor": (".bootstrap", "runtime_descriptor"),
    "create_ordinary_python_repository": (
        ".bootstrap",
        "create_ordinary_python_repository",
    ),
}

for _lazy_name in tuple(_LAZY_EXPORTS):
    globals().pop(_lazy_name, None)
del _lazy_name

__all__: Final[tuple[str, ...]] = ("SCHEMA", *tuple(sorted(_LAZY_EXPORTS)))


def __getattr__(name: str) -> Any:
    """Resolve frozen public names on first access without import side effects."""

    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
