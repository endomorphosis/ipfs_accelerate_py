"""Adversarial Assurance Engine public package surface (AAE-048).

Importing this package is side-effect free: no I/O, process, network,
provider configuration, or optional installer is started. Every name in
``__all__`` resolves lazily through :func:`__getattr__`.

Required production names (lazy):

* :class:`AssuranceCampaignApi` / :func:`create_assurance_campaign_api`
* The twelve plan-required module-level APIs:
  ``create_assurance_manifest``, ``generate_mutation_candidates``,
  ``predict_detection_set``, ``execute_mutation``,
  ``classify_mutation_outcome``, ``diagnose_surviving_mutant``,
  ``analyze_vacuity``, ``propose_gap_remediation``,
  ``evaluate_remediation``, ``promote_assurance_policy``,
  ``plan_mutation_campaign``, ``execute_mutation_campaign``

Leaf submodules (``planning``, ``execution``, ``promotion``, …) remain
importable directly and are not re-exported here beyond the public facade.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Final

# name -> (relative module, attribute)
_LAZY_EXPORTS: Final[dict[str, tuple[str, str]]] = {
    # --- pins / helpers ---
    "AAE_PUBLIC_API_EVIDENCE": (".api", "AAE_PUBLIC_API_EVIDENCE"),
    "ADVERSARIAL_ASSURANCE_PUBLIC_API_INTERFACE": (
        ".api",
        "ADVERSARIAL_ASSURANCE_PUBLIC_API_INTERFACE",
    ),
    "ADVERSARIAL_ASSURANCE_PUBLIC_API_SCHEMA": (
        ".api",
        "ADVERSARIAL_ASSURANCE_PUBLIC_API_SCHEMA",
    ),
    "ASSURANCE_CAMPAIGN_API_INTERFACE": (".api", "ASSURANCE_CAMPAIGN_API_INTERFACE"),
    "REQUIRED_PUBLIC_APIS": (".api", "REQUIRED_PUBLIC_APIS"),
    "REQUIRED_COMMANDS": (".api", "REQUIRED_COMMANDS"),
    "ApiAvailability": (".api", "ApiAvailability"),
    "AssuranceApiUnavailableError": (".api", "AssuranceApiUnavailableError"),
    "AssuranceApiUnavailableResult": (".api", "AssuranceApiUnavailableResult"),
    "AssurancePublicApiError": (".api", "AssurancePublicApiError"),
    "PathExposureError": (".api", "PathExposureError"),
    "UnknownCommandError": (".api", "UnknownCommandError"),
    "UnknownFieldError": (".api", "UnknownFieldError"),
    "api_interface_id": (".api", "api_interface_id"),
    "api_interface_ids": (".api", "api_interface_ids"),
    "campaign_api_interface_id": (".api", "campaign_api_interface_id"),
    "create_assurance_campaign_api": (".api", "create_assurance_campaign_api"),
    "invoke": (".api", "invoke"),
    "invoke_envelope": (".api", "invoke_envelope"),
    "public_api_evidence_id": (".api", "public_api_evidence_id"),
    "public_api_interface_id": (".api", "public_api_interface_id"),
    "public_api_schema": (".api", "public_api_schema"),
    "required_commands": (".api", "required_commands"),
    "required_public_apis": (".api", "required_public_apis"),
    "resolve_public_api": (".api", "resolve_public_api"),
    # --- composition class / result types ---
    "AssuranceCampaignApi": (".api", "AssuranceCampaignApi"),
    "MutationCampaignExecutionResult": (".api", "MutationCampaignExecutionResult"),
    "VacuityCampaignAnalysisResult": (".api", "VacuityCampaignAnalysisResult"),
    # --- twelve required public APIs ---
    "create_assurance_manifest": (".api", "create_assurance_manifest"),
    "generate_mutation_candidates": (".api", "generate_mutation_candidates"),
    "predict_detection_set": (".api", "predict_detection_set"),
    "execute_mutation": (".api", "execute_mutation"),
    "classify_mutation_outcome": (".api", "classify_mutation_outcome"),
    "diagnose_surviving_mutant": (".api", "diagnose_surviving_mutant"),
    "analyze_vacuity": (".api", "analyze_vacuity"),
    "propose_gap_remediation": (".api", "propose_gap_remediation"),
    "evaluate_remediation": (".api", "evaluate_remediation"),
    "promote_assurance_policy": (".api", "promote_assurance_policy"),
    "plan_mutation_campaign": (".api", "plan_mutation_campaign"),
    "execute_mutation_campaign": (".api", "execute_mutation_campaign"),
}

# Drop any previously cached lazy bindings (importlib.reload safety).
for _lazy_name in tuple(_LAZY_EXPORTS):
    globals().pop(_lazy_name, None)
del _lazy_name

__all__: Final[tuple[str, ...]] = tuple(sorted(_LAZY_EXPORTS))

# Stable evidence / interface labels for the frozen surface (not lazy).
PUBLIC_API_EVIDENCE: Final[str] = "aae/public-api@1"
PUBLIC_API_INTERFACE: Final[str] = "AdversarialAssurancePublicApi@1"
PUBLIC_API_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/adversarial-assurance-public-api@1"
)
REQUIRED_PUBLIC_NAMES: Final[tuple[str, ...]] = (
    "AssuranceCampaignApi",
    "create_assurance_manifest",
    "generate_mutation_candidates",
    "predict_detection_set",
    "execute_mutation",
    "classify_mutation_outcome",
    "diagnose_surviving_mutant",
    "analyze_vacuity",
    "propose_gap_remediation",
    "evaluate_remediation",
    "promote_assurance_policy",
    "plan_mutation_campaign",
    "execute_mutation_campaign",
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
