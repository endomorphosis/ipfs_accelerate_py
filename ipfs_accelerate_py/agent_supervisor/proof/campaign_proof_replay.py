"""Proof-replay authority for the O2 operational campaign surface.

Campaign proof-replay reuses ``Operation.VALIDATION_REPLAY``.  It never
promotes a model, tactician, or prompt claim to kernel authority and never
treats timeout as falsehood.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from ..control.control_contracts import Operation, OperationAuthority
from .formal_verification_contracts import ContractValidationError


CAMPAIGN_PROOF_REPLAY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/campaign-proof-replay@1"
)
CAMPAIGN_PROOF_REPLAY_CONTROL_OPERATION: Final = Operation.VALIDATION_REPLAY
CAMPAIGN_PROOF_REPLAY_AUTHORITY: Final = Operation.VALIDATION_REPLAY.authority

_FORBIDDEN = frozenset(
    {
        "hidden_labels",
        "prompt_selected_authority",
        "prompt_authority",
        "self_promote",
        "self_promotion",
        "kernel_authority",
        "proof_authority",
    }
)


def assert_campaign_proof_replay_authority(
    requested_authority: OperationAuthority | str | None = None,
    *,
    parameters: Mapping[str, Any] | None = None,
) -> Operation:
    """Fail closed if proof-replay would raise authority or accept prompt proof."""

    payload = parameters or {}
    keys = {str(key).strip().lower().replace("-", "_") for key in payload}
    if keys.intersection(_FORBIDDEN):
        raise ContractValidationError(
            "campaign proof-replay cannot carry prompt-selected proof authority"
        )
    if requested_authority is None or requested_authority == "":
        return CAMPAIGN_PROOF_REPLAY_CONTROL_OPERATION
    if isinstance(requested_authority, OperationAuthority):
        authority = requested_authority
    else:
        authority = OperationAuthority(str(requested_authority))
    if authority.rank > CAMPAIGN_PROOF_REPLAY_AUTHORITY.rank:
        raise ContractValidationError(
            "campaign proof-replay cannot raise authority above validation_replay"
        )
    return CAMPAIGN_PROOF_REPLAY_CONTROL_OPERATION


def campaign_proof_replay_binding() -> dict[str, Any]:
    """Return the closed proof-replay publication record."""

    return {
        "schema": CAMPAIGN_PROOF_REPLAY_SCHEMA,
        "control_operation": CAMPAIGN_PROOF_REPLAY_CONTROL_OPERATION.value,
        "authority": CAMPAIGN_PROOF_REPLAY_AUTHORITY.value,
        "timeout_is_falsehood": False,
        "model_has_proof_authority": False,
        "prompt_has_proof_authority": False,
        "expands_control_catalog": False,
    }


__all__ = (
    "CAMPAIGN_PROOF_REPLAY_AUTHORITY",
    "CAMPAIGN_PROOF_REPLAY_CONTROL_OPERATION",
    "CAMPAIGN_PROOF_REPLAY_SCHEMA",
    "assert_campaign_proof_replay_authority",
    "campaign_proof_replay_binding",
)
