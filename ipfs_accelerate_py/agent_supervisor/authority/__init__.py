"""Caller identity and effect-bound capability decisions (EAAEF-G040)."""

from .external_principal import (
    CAPABILITY_DECISION_INTERFACE,
    CONTRACT_VERSION,
    EXTERNAL_PRINCIPAL_INTERFACE,
    CapabilityDecision,
    ExternalPrincipal,
    bind_capability,
)

__all__ = (
    "CAPABILITY_DECISION_INTERFACE",
    "CONTRACT_VERSION",
    "EXTERNAL_PRINCIPAL_INTERFACE",
    "CapabilityDecision",
    "ExternalPrincipal",
    "bind_capability",
)
