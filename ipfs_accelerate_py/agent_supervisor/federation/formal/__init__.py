"""Bounded formal models for the causal supervisor federation.

The package exposes hermetic finite-state checks and optional TLC/Apalache
checks.  Every result is explicitly bounded; no result creates runtime
authority or supplies task-completion evidence by itself.
"""

from .models import (
    ADVERSARIAL_PROPERTY,
    CASF_EXTERNAL_CHECK_RECEIPT_SCHEMA,
    CASF_FORMAL_IDENTITY_SCHEMA,
    CASF_FORMAL_RECEIPT_SCHEMA,
    CASF_FORMAL_SUITE_SCHEMA,
    AdversarialMutation,
    ExternalCheckStatus,
    ExternalModelCheckReceipt,
    ExternalModelInvariant,
    FederationFormalError,
    FederationFormalIdentity,
    FederationFormalProperty,
    FederationFormalScenario,
    FederationFormalSuite,
    FederationModelState,
    FormalCounterexample,
    FormalTraceStep,
    HermeticCheckStatus,
    HermeticModelCheckReceipt,
    build_federation_formal_suite,
    check_federation_formal_suite,
    check_federation_scenario,
    run_external_model_checks,
)

__all__ = [
    "ADVERSARIAL_PROPERTY",
    "CASF_EXTERNAL_CHECK_RECEIPT_SCHEMA",
    "CASF_FORMAL_IDENTITY_SCHEMA",
    "CASF_FORMAL_RECEIPT_SCHEMA",
    "CASF_FORMAL_SUITE_SCHEMA",
    "AdversarialMutation",
    "ExternalCheckStatus",
    "ExternalModelCheckReceipt",
    "ExternalModelInvariant",
    "FederationFormalError",
    "FederationFormalIdentity",
    "FederationFormalProperty",
    "FederationFormalScenario",
    "FederationFormalSuite",
    "FederationModelState",
    "FormalCounterexample",
    "FormalTraceStep",
    "HermeticCheckStatus",
    "HermeticModelCheckReceipt",
    "build_federation_formal_suite",
    "check_federation_formal_suite",
    "check_federation_scenario",
    "run_external_model_checks",
]
