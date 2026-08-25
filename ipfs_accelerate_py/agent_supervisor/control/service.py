"""Direct control-package entry point for the federation control service.

This module deliberately contains no CLI/MCP parsing and no alternate
implementation.  It gives future direct adapters one canonical import while
keeping federation command semantics in the federation package.
"""

from __future__ import annotations

from ..federation.control_service import (
    FederationControlAuditReceipt,
    FederationControlAuthorization,
    FederationControlCapability,
    FederationControlResponse,
    FederationControlService,
    FederationControlServiceError,
    qualified_federation_control_capability,
)
from ..federation.contracts import FederationCommand


def execute_federation_command(
    service: FederationControlService,
    command: FederationCommand,
) -> FederationControlResponse:
    """Call the canonical typed service without command-string adaptation."""

    if not isinstance(service, FederationControlService):
        raise FederationControlServiceError("service must be FederationControlService")
    return service.execute(command)


__all__ = [
    "FederationControlAuditReceipt",
    "FederationControlAuthorization",
    "FederationControlCapability",
    "FederationControlResponse",
    "FederationControlService",
    "FederationControlServiceError",
    "execute_federation_command",
    "qualified_federation_control_capability",
]
