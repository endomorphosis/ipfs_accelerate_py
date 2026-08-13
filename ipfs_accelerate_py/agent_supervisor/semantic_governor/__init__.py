"""Semantic Compression Governor public package surface (SCG-036).

Importing this package is side-effect free: no I/O, process, network,
provider configuration, or optional installer is started. Every name in
``__all__`` resolves lazily through :func:`__getattr__`.

Required production names (lazy):

* :class:`SemanticCompressionGovernor` / :func:`create_semantic_compression_governor`
* The ten plan-required module-level APIs:
  ``evaluate_context_sufficiency``, ``create_shadow_plan``,
  ``compare_shadow_results``, ``diagnose_omission``,
  ``plan_context_expansion``, ``execute_expansion_loop``,
  ``update_calibration``, ``propose_rule_change``,
  ``evaluate_rule_candidate``, ``promote_compression_policy``

Leaf submodules (``runtime``, ``promotion``, ``sealing``, …) remain importable
directly and are not re-exported here.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Final

# name -> (relative module, attribute)
_LAZY_EXPORTS: Final[dict[str, tuple[str, str]]] = {
    # --- pins / helpers ---
    "SCG_PUBLIC_API_EVIDENCE": (".governor", "SCG_PUBLIC_API_EVIDENCE"),
    "SEMANTIC_COMPRESSION_GOVERNOR_INTERFACE": (
        ".governor",
        "SEMANTIC_COMPRESSION_GOVERNOR_INTERFACE",
    ),
    "SEMANTIC_COMPRESSION_GOVERNOR_SCHEMA": (
        ".governor",
        "SEMANTIC_COMPRESSION_GOVERNOR_SCHEMA",
    ),
    "SEMANTIC_COMPRESSION_GOVERNOR_PACKAGE_INTERFACE": (
        ".governor",
        "SEMANTIC_COMPRESSION_GOVERNOR_PACKAGE_INTERFACE",
    ),
    "REQUIRED_PUBLIC_APIS": (".governor", "REQUIRED_PUBLIC_APIS"),
    "REQUIRED_COMMANDS": (".governor", "REQUIRED_COMMANDS"),
    "ApiAvailability": (".governor", "ApiAvailability"),
    "GovernorApiUnavailableError": (".governor", "GovernorApiUnavailableError"),
    "GovernorApiUnavailableResult": (".governor", "GovernorApiUnavailableResult"),
    "GovernorPublicApiError": (".governor", "GovernorPublicApiError"),
    "UnknownCommandError": (".governor", "UnknownCommandError"),
    "UnknownFieldError": (".governor", "UnknownFieldError"),
    "api_interface_id": (".governor", "api_interface_id"),
    "api_interface_ids": (".governor", "api_interface_ids"),
    "create_semantic_compression_governor": (
        ".governor",
        "create_semantic_compression_governor",
    ),
    "governor_interface_id": (".governor", "governor_interface_id"),
    "invoke": (".governor", "invoke"),
    "invoke_envelope": (".governor", "invoke_envelope"),
    "public_api_evidence_id": (".governor", "public_api_evidence_id"),
    "public_api_interface_id": (".governor", "public_api_interface_id"),
    "public_api_schema": (".governor", "public_api_schema"),
    "required_commands": (".governor", "required_commands"),
    "required_public_apis": (".governor", "required_public_apis"),
    "resolve_public_api": (".governor", "resolve_public_api"),
    # --- composition class ---
    "SemanticCompressionGovernor": (".governor", "SemanticCompressionGovernor"),
    # --- ten required public APIs ---
    "evaluate_context_sufficiency": (".governor", "evaluate_context_sufficiency"),
    "create_shadow_plan": (".governor", "create_shadow_plan"),
    "compare_shadow_results": (".governor", "compare_shadow_results"),
    "diagnose_omission": (".governor", "diagnose_omission"),
    "plan_context_expansion": (".governor", "plan_context_expansion"),
    "execute_expansion_loop": (".governor", "execute_expansion_loop"),
    "update_calibration": (".governor", "update_calibration"),
    "propose_rule_change": (".governor", "propose_rule_change"),
    "evaluate_rule_candidate": (".governor", "evaluate_rule_candidate"),
    "promote_compression_policy": (".governor", "promote_compression_policy"),
}

# Drop any previously cached lazy bindings (importlib.reload safety).
for _lazy_name in tuple(_LAZY_EXPORTS):
    globals().pop(_lazy_name, None)
del _lazy_name

__all__: Final[tuple[str, ...]] = tuple(sorted(_LAZY_EXPORTS))

# Stable evidence / interface labels for the frozen surface (not lazy).
PUBLIC_API_EVIDENCE: Final[str] = "scg/public-api@1"
PUBLIC_API_INTERFACE: Final[str] = "SemanticCompressionGovernorPublicApi@1"
PUBLIC_API_SCHEMA: Final[str] = (
    "ipfs-accelerate.semantic-compression-governor-public-api@1"
)
REQUIRED_PUBLIC_NAMES: Final[tuple[str, ...]] = (
    "SemanticCompressionGovernor",
    "evaluate_context_sufficiency",
    "create_shadow_plan",
    "compare_shadow_results",
    "diagnose_omission",
    "plan_context_expansion",
    "execute_expansion_loop",
    "update_calibration",
    "propose_rule_change",
    "evaluate_rule_candidate",
    "promote_compression_policy",
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
