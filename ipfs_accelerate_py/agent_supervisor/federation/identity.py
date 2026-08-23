"""Canonical identity-contract exports for the causal event federation.

Identity validation remains implemented by :mod:`contracts`; this narrow
module provides the declared package surface without creating a second
identity authority.
"""

from __future__ import annotations

from .contracts import FederationIdentity, SubagentIdentity, SupervisorIdentity

__all__ = (
    "FederationIdentity",
    "SubagentIdentity",
    "SupervisorIdentity",
)
