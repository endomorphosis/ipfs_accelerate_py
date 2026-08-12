"""Incremental-verification public package surface (IVP-019).

Importing this package is side-effect free.  Every name in ``__all__`` resolves
lazily through :func:`__getattr__` so cold import never loads planner, cache,
adapters, or other heavy collaborators until the attribute is first accessed.

Required production names (lazy):

* :func:`create_verification_plan` / :class:`IncrementalVerificationPlanner`
* :func:`choose_model_route` / :class:`ModelRoutePlanner`
* :func:`build_verification_commitment`
* :class:`VerificationReceiptCache`

Canonical contract types and orchestration helpers are also part of the frozen
public surface.  Private helpers, adapter internals, and optional backends stay
on their submodules and are not re-exported here.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Final

# name -> (relative module, attribute)
# Relative modules stay under this package; attribute is the symbol resolved
# on first access and cached into this module's globals.
_LAZY_EXPORTS: Final[dict[str, tuple[str, str]]] = {
    # --- contracts (light schemas / identities) ---
    "PROOF_OBLIGATION_NOT_APPLICABLE_CID": (
        ".contracts",
        "PROOF_OBLIGATION_NOT_APPLICABLE_CID",
    ),
    "TERMINAL_STATUS_PRECEDENCE": (".contracts", "TERMINAL_STATUS_PRECEDENCE"),
    "CacheReuseDecision": (".contracts", "CacheReuseDecision"),
    "CacheReuseDisposition": (".contracts", "CacheReuseDisposition"),
    "CounterexampleReceipt": (".contracts", "CounterexampleReceipt"),
    "DiagnosticValueState": (".contracts", "DiagnosticValueState"),
    "DirectExecutionObservation": (".contracts", "DirectExecutionObservation"),
    "ModelRoute": (".contracts", "ModelRoute"),
    "ModelRouteDecision": (".contracts", "ModelRouteDecision"),
    "ProofReceipt": (".contracts", "ProofReceipt"),
    "StaticAnalysisReceipt": (".contracts", "StaticAnalysisReceipt"),
    "TerminalStatus": (".contracts", "TerminalStatus"),
    "TestReceipt": (".contracts", "TestReceipt"),
    "TypeCheckReceipt": (".contracts", "TypeCheckReceipt"),
    "VerificationBoundsError": (".contracts", "VerificationBoundsError"),
    "VerificationBundle": (".contracts", "VerificationBundle"),
    "VerificationCommitment": (".contracts", "VerificationCommitment"),
    "VerificationContractError": (".contracts", "VerificationContractError"),
    "VerificationIdentityCompiler": (".contracts", "VerificationIdentityCompiler"),
    "VerificationIdentityError": (".contracts", "VerificationIdentityError"),
    "VerificationPlan": (".contracts", "VerificationPlan"),
    "VerificationReceipt": (".contracts", "VerificationReceipt"),
    "VerificationReceiptKey": (".contracts", "VerificationReceiptKey"),
    "VerificationReceiptKind": (".contracts", "VerificationReceiptKind"),
    "VerificationSummary": (".contracts", "VerificationSummary"),
    "aggregate_terminal_status": (".contracts", "aggregate_terminal_status"),
    "build_verification_commitment": (
        ".contracts",
        "build_verification_commitment",
    ),
    # --- planner (required public) ---
    "IncrementalVerificationPlanner": (
        ".planner",
        "IncrementalVerificationPlanner",
    ),
    "create_incremental_verification_planner": (
        ".planner",
        "create_incremental_verification_planner",
    ),
    "create_verification_plan": (".planner", "create_verification_plan"),
    # --- model route (required public) ---
    "ModelRoutePlanner": (".model_route", "ModelRoutePlanner"),
    "choose_model_route": (".model_route", "choose_model_route"),
    # --- receipt cache (required public) ---
    "VerificationReceiptCache": (".receipt_cache", "VerificationReceiptCache"),
    "production_eligible": (".receipt_cache", "production_eligible"),
    # --- bundle helpers ---
    "build_verification_bundle": (".bundle", "build_verification_bundle"),
    "build_verification_summary": (".bundle", "build_verification_summary"),
    # --- executor ---
    "VerificationExecutor": (".executor", "VerificationExecutor"),
    "execute_verification_plan": (".executor", "execute_verification_plan"),
    "create_verification_executor": (
        ".executor",
        "create_verification_executor",
    ),
    "compute_production_acceptance": (
        ".executor",
        "compute_production_acceptance",
    ),
}

# Drop any previously cached lazy bindings (importlib.reload must not leave
# resolved heavy modules attached until the next attribute access).
for _lazy_name in tuple(_LAZY_EXPORTS):
    globals().pop(_lazy_name, None)
del _lazy_name

__all__: Final[tuple[str, ...]] = tuple(sorted(_LAZY_EXPORTS))

# Stable evidence / interface labels for the frozen surface (not lazy).
PUBLIC_API_EVIDENCE: Final[str] = "ivp/public-api@1"
PUBLIC_API_INTERFACE: Final[str] = "IncrementalVerificationPublicApi@1"
REQUIRED_PUBLIC_NAMES: Final[tuple[str, ...]] = (
    "create_verification_plan",
    "choose_model_route",
    "build_verification_commitment",
    "VerificationReceiptCache",
    "IncrementalVerificationPlanner",
    "ModelRoutePlanner",
)


def __getattr__(name: str) -> Any:
    """Resolve frozen public names on first access (lazy, side-effect free)."""

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
