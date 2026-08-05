"""Cold-safe public surface for proof-backed pytest reuse.

Importing this package deliberately imports only the standard-library
configuration model.  Candidate stores, certificate providers, ZK backends,
and pytest itself are loaded only by later, active parts of the integration.
"""

from .config import (
    PROOF_REUSE_MODE_ENV,
    PROOF_REUSE_MODES,
    PROOF_REUSE_REQUIRED_AUDIT_ENV,
    ProofReuseConfig,
    ProofReuseConfigurationError,
    ProofReuseMode,
)

__all__ = [
    "PROOF_REUSE_MODE_ENV",
    "PROOF_REUSE_MODES",
    "PROOF_REUSE_REQUIRED_AUDIT_ENV",
    "ProofReuseConfig",
    "ProofReuseConfigurationError",
    "ProofReuseMode",
]
