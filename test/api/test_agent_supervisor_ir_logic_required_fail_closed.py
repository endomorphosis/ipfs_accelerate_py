"""DCR-035 required IR logic gate fail-closed tests."""

from __future__ import annotations

from dataclasses import replace

import pytest
from ipfs_accelerate_py.agent_supervisor.planning.ir_logic_hooks import (
    evaluate_required_ir_logic_hook_gate,
)
from ipfs_accelerate_py.agent_supervisor.proof.ir_logic_application import (
    REQUIRED_IR_LOGIC_STAGES,
    IrLogicRequiredGateDisposition,
    IrLogicRequiredStageReceipt,
    evaluate_required_ir_logic_gate,
)


def _identities(*, include_late: bool = True) -> dict[str, str]:
    values = {
        "dcr030": "cid:dcr030",
        "dcr031": "cid:dcr031",
        "dcr032": "cid:dcr032",
    }
    if include_late:
        values.update({"dcr033": "cid:dcr033", "dcr034": "cid:dcr034"})
    return values


def _receipts(
    *, identities: dict[str, str] | None = None
) -> tuple[IrLogicRequiredStageReceipt, ...]:
    bound = identities or _identities()
    return tuple(
        IrLogicRequiredStageReceipt(
            stage=stage,
            identity_cids=bound,
            surface_cids=(f"surface:{stage}",),
        )
        for stage in REQUIRED_IR_LOGIC_STAGES
    )


def test_all_exact_stage_receipts_pass_but_never_authorize_execution_or_completion() -> None:
    result = evaluate_required_ir_logic_gate(
        _receipts(), required_identity_cids=_identities()
    )

    assert result.disposition is IrLogicRequiredGateDisposition.PASSING
    assert result.passing is True
    assert result.model_call_count == result.provider_call_count == 0
    assert result.execution_authorized is False
    assert result.completion_authorized is False


def test_missing_current_dcr033_dcr034_is_integration_pending_not_success() -> None:
    partial = _identities(include_late=False)
    result = evaluate_required_ir_logic_gate(
        _receipts(identities=partial), required_identity_cids=partial
    )
    via_hook = evaluate_required_ir_logic_hook_gate(
        {
            "dcr035_stage_receipts": _receipts(identities=partial),
            "dcr035_identity_cids": partial,
            "apply_ir_logic": True,
        }
    )

    for gate in (result, via_hook):
        assert gate.disposition is IrLogicRequiredGateDisposition.INTEGRATION_PENDING
        assert gate.passing is False
        assert gate.execution_authorized is False
        assert gate.completion_authorized is False
        assert any("dcr033" in reason for reason in gate.reason_codes)
        assert any("dcr034" in reason for reason in gate.reason_codes)


@pytest.mark.parametrize(
    "receipt",
    (
        replace(_receipts()[0], outcome="unsupported"),
        replace(_receipts()[0], outcome="unknown"),
        replace(_receipts()[0], outcome="error"),
        replace(_receipts()[0], bridge_only=True),
        replace(_receipts()[0], default_true=True),
        replace(_receipts()[0], swallowed_exception=True),
        replace(_receipts()[0], model_call_count=1),
        replace(_receipts()[0], surface_cids=()),
    ),
)
def test_nonpassing_stage_states_cannot_be_upgraded_by_defaults(
    receipt: IrLogicRequiredStageReceipt,
) -> None:
    receipts = (receipt, *_receipts()[1:])
    result = evaluate_required_ir_logic_gate(
        receipts, required_identity_cids=_identities()
    )

    assert result.disposition is IrLogicRequiredGateDisposition.REJECTED
    assert result.passing is False
    assert result.execution_authorized is False
    assert result.completion_authorized is False
    assert result.model_call_count == result.provider_call_count == 0


def test_omitted_or_malformed_stage_cannot_be_silently_swallowed() -> None:
    omitted = evaluate_required_ir_logic_gate(
        _receipts()[1:], required_identity_cids=_identities()
    )
    malformed = evaluate_required_ir_logic_gate(
        ({"stage": "diagnose", "identity_cids": _identities()},),
        required_identity_cids=_identities(),
    )

    for result in (omitted, malformed):
        assert result.disposition is IrLogicRequiredGateDisposition.REJECTED
        assert result.passing is False
        assert result.execution_authorized is False
        assert result.completion_authorized is False
