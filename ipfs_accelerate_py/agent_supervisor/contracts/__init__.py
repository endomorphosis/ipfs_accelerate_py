"""Side-effect-free shared supervisor contracts (ASE3-029)."""

from __future__ import annotations

from . import authority as _authority
from . import execution as _execution
from . import provider_capacity as _provider_capacity
from .authority import ProfileAuthorityService, VerifiedAuthorityBinding
from .provider_capacity import (
    NON_AUTHORITATIVE_CAPACITY_SCHEMA,
    NonAuthoritativeProviderCapacityObservation,
    PlanExecutionStore,
    ProviderAttemptStoreService,
)

for _name in dir(_execution):
    if _name.startswith("_"):
        continue
    globals()[_name] = getattr(_execution, _name)

__all__ = sorted(
    {
        "NON_AUTHORITATIVE_CAPACITY_SCHEMA",
        "NonAuthoritativeProviderCapacityObservation",
        "PlanExecutionStore",
        "ProfileAuthorityService",
        "ProviderAttemptStoreService",
        "VerifiedAuthorityBinding",
        *[name for name in dir(_execution) if not name.startswith("_")],
    }
)

del _name
del _authority
del _execution
del _provider_capacity
