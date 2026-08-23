"""FACP-040: route migrated Accelerate transports through admission.

Acceptance:
- Direct handler call without token fails.
- All migrated transports make the same kernel call.
- Browser / model / peer inputs cannot select authority.
- Denied admission has zero handler invocations.
"""

from __future__ import annotations

from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.admission.formal_kernel import (
    KERNEL_ISSUER,
    AdmissionBindings,
    AdmissionErrorCode,
    AdmissionToken,
    AdmissionVerdict,
    OperationSpecView,
    binding_cid,
    default_kernel,
    derive_token_obligations,
    fresh_nonce,
)
from ipfs_accelerate_py.agent_supervisor.admission.transport_gate import (
    AUTHORITY_SELECTION_KEYS,
    BUNDLE,
    CLOSED_OUTCOMES,
    EVIDENCE_SUBSET,
    GOAL_ID,
    INVENTORIED_TRANSPORT_SEAMS,
    INTERFACE,
    KERNEL_CALL,
    MIGRATED_TRANSPORTS,
    SCHEMA,
    TASK_ID,
    UNSAFE_PROMOTION,
    UNTRUSTED_AUTHORITY_SOURCES,
    CommonTransportGate,
    HandlerNotUnlockedError,
    TransportKind,
    TransportRequest,
    argument_cid_for,
    default_transport_gate,
    reject_untrusted_authority_selection,
    same_kernel_call,
)


NOW_MS = 1_700_000_000_000
NOT_AFTER = NOW_MS + 60_000


def _spec(**overrides: Any) -> OperationSpecView:
    values: dict[str, Any] = {
        "operation_id": "accelerate.inference",
        "effect_class": "process",
        "authority_obligation": "capability_verified",
        "policy_obligation": "host_policy_required",
        "confirmation_obligation": "one_use_confirmation_required",
        "lease_obligation": "lease_required",
        "observation_obligation": "independent_observation_required",
        "idempotency_class": "at_most_once",
        "reversibility_class": "compensatable",
    }
    values.update(overrides)
    return OperationSpecView(**values)


def _allow_policy(*, policy_cid: str | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {
        "name": "allow-inference",
        "clauses": [
            {
                "clause_type": "permission",
                "actor": "*",
                "action": "accelerate.inference",
                "resource": "*",
            }
        ],
    }
    if policy_cid is not None:
        body["policy_cid"] = policy_cid
    return body


def _bindings(
    *,
    arguments: dict[str, Any] | None = None,
    nonce: str | None = None,
    not_after: int = NOT_AFTER,
    **overrides: Any,
) -> AdmissionBindings:
    args = arguments if arguments is not None else {"prompt": "hello", "model": "demo"}
    values: dict[str, Any] = {
        "actor_cid": binding_cid("actor", "operator-1"),
        "device_cid": binding_cid("device", "device-1"),
        "tenant_cid": binding_cid("tenant", "tenant-1"),
        "resource_cid": binding_cid("resource", "model:demo"),
        "operation_id": "accelerate.inference",
        "argument_cid": argument_cid_for(args),
        "contract_cid": binding_cid("contract", "facp/operation-spec@1"),
        "delegation_cid": binding_cid("delegation", "ucan:chain-1"),
        "policy_cid": binding_cid("policy", _allow_policy()),
        "confirmation_cid": binding_cid("confirmation", "confirm:1"),
        "lease_id": "lease:demo-1",
        "not_before": NOW_MS - 1_000,
        "not_after": not_after,
        "nonce": nonce or fresh_nonce(),
        "signature_cid": binding_cid("signature", "sig:1"),
        "revocation_id": "",
    }
    values.update(overrides)
    return AdmissionBindings(**values)


def _mint_token(
    gate: CommonTransportGate,
    *,
    spec: OperationSpecView | None = None,
    arguments: dict[str, Any] | None = None,
    bindings: AdmissionBindings | None = None,
) -> tuple[AdmissionToken, AdmissionBindings, OperationSpecView, dict[str, Any]]:
    spec = spec or _spec()
    args = arguments if arguments is not None else {"prompt": "hello", "model": "demo"}
    bindings = bindings or _bindings(arguments=args)
    policy = _allow_policy(policy_cid=bindings.policy_cid)
    token = gate.kernel.mint_token(spec, bindings, source_policy=policy)
    return token, bindings, spec, args


def _handler_factory(log: list[dict[str, Any]]):
    def _handler(arguments: Any) -> dict[str, Any]:
        entry = {"arguments": dict(arguments), "outcome": "Observed"}
        log.append(entry)
        return entry

    return _handler


def _ready_gate(
    *,
    handler_log: list[dict[str, Any]] | None = None,
) -> tuple[CommonTransportGate, list[dict[str, Any]]]:
    log: list[dict[str, Any]] = handler_log if handler_log is not None else []
    gate = default_transport_gate(now_ms=NOW_MS)
    gate.register_handler(_spec(), _handler_factory(log))
    return gate, log


# ---------------------------------------------------------------------------
# Module contract / evidence envelope
# ---------------------------------------------------------------------------


def test_facp_040_metadata_and_evidence_subset() -> None:
    assert TASK_ID == "FACP-040"
    assert GOAL_ID == "FACP-G320"
    assert BUNDLE == "facp/admission/transports"
    assert SCHEMA == "facp/common-transport-gate@1"
    assert INTERFACE == "CommonTransportGate@1"
    assert KERNEL_CALL == "effect_admission_kernel.unlock_handler"
    assert UNSAFE_PROMOTION is False
    assert set(MIGRATED_TRANSPORTS) == {"cli", "mcp", "mcp++", "python"}
    assert "Observed" in CLOSED_OUTCOMES
    assert "Rejected" in CLOSED_OUTCOMES
    required = {
        "same_token_decision_across_transports",
        "effect_class_match",
        "exact_args",
        "revocation",
        "denial",
        "typed_observation_outcome",
    }
    assert set(EVIDENCE_SUBSET) == required
    transports = {seam["transport"] for seam in INVENTORIED_TRANSPORT_SEAMS}
    assert {"cli", "mcp", "mcp++"}.issubset(transports)
    assert "browser" in UNTRUSTED_AUTHORITY_SOURCES
    assert "model" in UNTRUSTED_AUTHORITY_SOURCES
    assert "peer" in UNTRUSTED_AUTHORITY_SOURCES
    assert "tenant_cid" in AUTHORITY_SELECTION_KEYS
    assert "policy_cid" in AUTHORITY_SELECTION_KEYS
    assert "endpoint" in AUTHORITY_SELECTION_KEYS
    assert "path" in AUTHORITY_SELECTION_KEYS


# ---------------------------------------------------------------------------
# Direct handler call without token fails
# ---------------------------------------------------------------------------


def test_direct_handler_call_without_token_fails() -> None:
    gate, log = _ready_gate()
    gated = gate.get_gated_handler("accelerate.inference")
    with pytest.raises(HandlerNotUnlockedError) as exc:
        gated({"prompt": "hello", "model": "demo"})
    assert exc.value.code is AdmissionErrorCode.HANDLER_NOT_UNLOCKED
    assert log == []
    assert gate.handler_invocation_count() == 0


def test_direct_raw_handler_is_not_the_production_seam() -> None:
    """Production seams expose GatedHandler; raw call is not gate-admitted."""
    gate, log = _ready_gate()
    gated = gate.get_gated_handler("accelerate.inference")
    # Calling the gated wrapper without unlock still fails.
    with pytest.raises(HandlerNotUnlockedError):
        gated({"prompt": "x"})
    assert log == []
    assert gated.raw_handler is not None


# ---------------------------------------------------------------------------
# All migrated transports make the same kernel call
# ---------------------------------------------------------------------------


def test_all_migrated_transports_make_the_same_kernel_call() -> None:
    kernel_identities: list[dict[str, Any]] = []

    for transport in MIGRATED_TRANSPORTS:
        gate, log = _ready_gate()
        token, bindings, spec, args = _mint_token(gate)
        assert token.issuer == KERNEL_ISSUER
        assert token.argument_cid == bindings.argument_cid
        assert "argument_bound" in derive_token_obligations(spec)

        request = TransportRequest(
            operation_id=spec.operation_id,
            arguments=args,
            typestate="Reserved",
            token=token,
            authority_source="host",
        )
        result = gate.dispatch(transport, request)
        assert result.admitted is True
        assert result.handler_invoked is True
        assert result.outcome == "Observed"
        assert result.decision.verdict is AdmissionVerdict.ADMIT
        assert result.decision.unlocked is True
        assert result.kernel_call is not None
        assert result.kernel_call.method == KERNEL_CALL
        assert result.kernel_call.effect_class == "process"
        assert result.kernel_call.argument_cid == bindings.argument_cid
        assert result.argument_cid == bindings.argument_cid
        assert len(log) == 1
        kernel_identities.append(result.kernel_call.identity_without_transport())

    # Same unlock_handler call shape across every migrated transport.
    assert len({tuple(sorted(item.items())) for item in kernel_identities}) == 1

    # Adapter convenience methods share dispatch.
    gate, _ = _ready_gate()
    token, bindings, spec, args = _mint_token(gate)
    via_cli = gate.invoke_cli(
        TransportRequest(operation_id=spec.operation_id, arguments=args, token=token)
    )
    gate2, _ = _ready_gate()
    token2, bindings2, _, args2 = _mint_token(gate2, arguments=dict(args))
    via_mcp = gate2.invoke_mcp(
        TransportRequest(operation_id=spec.operation_id, arguments=args2, token=token2)
    )
    assert via_cli.kernel_call is not None
    assert via_mcp.kernel_call is not None
    assert via_cli.kernel_call.method == via_mcp.kernel_call.method == KERNEL_CALL
    assert via_cli.kernel_call.operation_id == via_mcp.kernel_call.operation_id
    assert via_cli.kernel_call.effect_class == via_mcp.kernel_call.effect_class
    assert via_cli.argument_cid == via_mcp.argument_cid == bindings.argument_cid
    assert bindings2.argument_cid == bindings.argument_cid
    assert same_kernel_call(via_cli.kernel_call, via_mcp.kernel_call)


def test_same_token_same_decision_across_transports() -> None:
    """One token yields the identical admission decision on every transport."""
    gate, log = _ready_gate()
    token, bindings, spec, args = _mint_token(gate)
    decisions = []
    identities = []
    for transport in MIGRATED_TRANSPORTS:
        result = gate.dispatch(
            transport,
            TransportRequest(
                operation_id=spec.operation_id,
                arguments=args,
                token=token,
                consume_token=False,
            ),
        )
        assert result.admitted is True
        assert result.handler_invoked is True
        assert result.kernel_call is not None
        assert result.kernel_call.token_id == token.token_id
        decisions.append(
            (
                result.decision.verdict,
                result.decision.unlocked,
                result.decision.code,
                result.outcome,
                result.argument_cid,
                result.decision.token_id,
            )
        )
        identities.append(result.kernel_call.identity_without_transport())
    assert len(set(decisions)) == 1
    assert decisions[0][0] is AdmissionVerdict.ADMIT
    assert decisions[0][4] == bindings.argument_cid
    assert decisions[0][5] == token.token_id
    assert len({tuple(sorted(i.items())) for i in identities}) == 1
    assert len(log) == len(MIGRATED_TRANSPORTS)


def test_invoke_all_migrated_shares_kernel_call_shape() -> None:
    identities = []
    decisions = []
    for transport in MIGRATED_TRANSPORTS:
        gate, _ = _ready_gate()
        token, bindings, spec, args = _mint_token(gate)
        result = gate.dispatch(
            transport,
            TransportRequest(
                operation_id=spec.operation_id,
                arguments=args,
                token=token,
            ),
        )
        assert result.kernel_call is not None
        identities.append(result.kernel_call.identity_without_transport())
        decisions.append(
            (
                result.decision.verdict,
                result.decision.unlocked,
                result.outcome,
                result.argument_cid,
            )
        )
    assert len({tuple(sorted(i.items())) for i in identities}) == 1
    assert len(set(decisions)) == 1
    assert decisions[0][0] is AdmissionVerdict.ADMIT
    assert decisions[0][3] == identities[0]["argument_cid"]
    assert decisions[0][3] == bindings.argument_cid


def test_same_kernel_call_helper() -> None:
    gate, _ = _ready_gate()
    token, _, spec, args = _mint_token(gate)
    r1 = gate.invoke_cli(
        TransportRequest(operation_id=spec.operation_id, arguments=args, token=token)
    )
    gate2, _ = _ready_gate()
    token2, _, _, args2 = _mint_token(gate2, arguments=dict(args))
    r2 = gate2.invoke_mcp_plus_plus(
        TransportRequest(operation_id=spec.operation_id, arguments=args2, token=token2)
    )
    assert r1.kernel_call is not None and r2.kernel_call is not None
    assert same_kernel_call(r1.kernel_call, r2.kernel_call)


# ---------------------------------------------------------------------------
# Browser / model / peer inputs cannot select authority
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("source", sorted(UNTRUSTED_AUTHORITY_SOURCES))
@pytest.mark.parametrize(
    "field",
    ["tenant_cid", "policy_cid", "endpoint", "path", "actor_cid", "issuer"],
)
def test_untrusted_inputs_cannot_select_authority(source: str, field: str) -> None:
    gate, log = _ready_gate()
    token, _, spec, args = _mint_token(gate)
    overlay = {field: "attacker-selected-value"}
    result = gate.dispatch(
        TransportKind.MCP,
        TransportRequest(
            operation_id=spec.operation_id,
            arguments=args,
            token=token,
            authority_source=source,
            transport_overlay=overlay,
        ),
    )
    assert result.admitted is False
    assert result.handler_invoked is False
    assert result.outcome == "Rejected"
    assert result.decision.verdict is AdmissionVerdict.DENY
    assert result.code in {
        AdmissionErrorCode.FORBIDDEN_ISSUER.value,
        AdmissionErrorCode.FREE_FORM_AUTHORITY.value,
        AdmissionErrorCode.NON_KERNEL_TOKEN_ISSUER.value,
    }
    assert log == []
    assert gate.handler_invocation_count() == 0


def test_reject_untrusted_authority_selection_helper() -> None:
    with pytest.raises(Exception) as exc:
        reject_untrusted_authority_selection(
            authority_source="browser",
            overlay={"tenant_cid": "t-evil", "policy_cid": "p-evil"},
        )
    assert exc.value.code in {
        AdmissionErrorCode.FORBIDDEN_ISSUER,
        AdmissionErrorCode.FREE_FORM_AUTHORITY,
    }
    # Host overlays still reject free-form authority keys.
    with pytest.raises(Exception) as exc2:
        reject_untrusted_authority_selection(
            authority_source="host",
            overlay={"consent": "yes", "allowed": True},
        )
    assert exc2.value.code is AdmissionErrorCode.FREE_FORM_AUTHORITY


def test_host_path_without_authority_overlay_admits() -> None:
    gate, log = _ready_gate()
    token, _, spec, args = _mint_token(gate)
    result = gate.invoke_python(
        TransportRequest(
            operation_id=spec.operation_id,
            arguments=args,
            token=token,
            authority_source="host",
            transport_overlay={"trace_id": "non-authority-metadata"},
        )
    )
    assert result.admitted is True
    assert result.handler_invoked is True
    assert len(log) == 1


# ---------------------------------------------------------------------------
# Denied admission has zero handler invocations
# ---------------------------------------------------------------------------


def test_denied_admission_zero_handler_invocations_missing_token() -> None:
    gate, log = _ready_gate()
    args = {"prompt": "hello", "model": "demo"}
    result = gate.invoke_cli(
        TransportRequest(
            operation_id="accelerate.inference",
            arguments=args,
            token=None,
        )
    )
    assert result.admitted is False
    assert result.handler_invoked is False
    assert result.outcome == "Rejected"
    assert result.decision.code is AdmissionErrorCode.HANDLER_NOT_UNLOCKED
    assert log == []
    assert gate.handler_invocation_count() == 0
    assert gate.handler_invocation_count("accelerate.inference") == 0


def test_denied_admission_zero_handler_invocations_revoked_token() -> None:
    gate, log = _ready_gate()
    token, _, spec, args = _mint_token(gate)
    gate.kernel.revoke(token)
    result = gate.invoke_mcp(
        TransportRequest(operation_id=spec.operation_id, arguments=args, token=token)
    )
    assert result.admitted is False
    assert result.handler_invoked is False
    assert result.decision.code is AdmissionErrorCode.REVOKED_TOKEN
    assert log == []
    assert gate.handler_invocation_count() == 0


def test_denied_admission_zero_handler_invocations_argument_mismatch() -> None:
    gate, log = _ready_gate()
    token, _, spec, _ = _mint_token(gate)
    tampered = {"prompt": "TAMPERED", "model": "demo"}
    result = gate.invoke_mcp_plus_plus(
        TransportRequest(
            operation_id=spec.operation_id,
            arguments=tampered,
            token=token,
        )
    )
    assert result.admitted is False
    assert result.handler_invoked is False
    assert result.decision.code is AdmissionErrorCode.ARGUMENT_MISMATCH
    assert result.argument_cid == argument_cid_for(tampered)
    assert log == []
    assert gate.handler_invocation_count() == 0


def test_denied_admission_zero_handler_invocations_expired_token() -> None:
    gate, log = _ready_gate()
    # Mint with short expiry, then advance clock via a new kernel binding.
    short = _bindings(not_after=NOW_MS + 5_000)
    args = {"prompt": "hello", "model": "demo"}
    token = gate.kernel.mint_token(
        _spec(),
        short,
        source_policy=_allow_policy(policy_cid=short.policy_cid),
    )
    # Rebind gate clock into the future by replacing kernel clock through verify path:
    # unlock_handler accepts now_ms via kernel's clock — construct a late gate sharing tokens.
    late = default_transport_gate(now_ms=NOW_MS + 10_000)
    late.kernel = gate.kernel  # share issued/revocation state
    # Force clock forward by monkeypatching clock_ms on the shared kernel.
    gate.kernel.clock_ms = lambda: NOW_MS + 10_000
    late.register_handler(_spec(), _handler_factory(log))
    result = late.invoke_cli(
        TransportRequest(operation_id="accelerate.inference", arguments=args, token=token)
    )
    assert result.admitted is False
    assert result.handler_invoked is False
    assert result.decision.code is AdmissionErrorCode.EXPIRED_TOKEN
    assert log == []
    assert late.handler_invocation_count() == 0


# ---------------------------------------------------------------------------
# Effect-class match + typed observation outcome
# ---------------------------------------------------------------------------


def test_effect_class_match_recorded_on_kernel_call() -> None:
    gate, _ = _ready_gate()
    token, bindings, spec, args = _mint_token(gate)
    result = gate.invoke_cli(
        TransportRequest(operation_id=spec.operation_id, arguments=args, token=token)
    )
    assert result.kernel_call is not None
    assert result.kernel_call.effect_class == spec.effect_class == "process"
    assert result.kernel_call.argument_cid == bindings.argument_cid


def test_typed_observation_outcome_rejects_success_boolean_promotion() -> None:
    gate = default_transport_gate(now_ms=NOW_MS)

    def bad_handler(_arguments: Any) -> dict[str, Any]:
        return {"success": True, "ok": True}

    gate.register_handler(_spec(), bad_handler)
    token, _, spec, args = _mint_token(gate)
    result = gate.invoke_mcp(
        TransportRequest(operation_id=spec.operation_id, arguments=args, token=token)
    )
    assert result.handler_invoked is True
    assert result.outcome == "Unknown"
    assert result.outcome in CLOSED_OUTCOMES


def test_unknown_transport_rejected() -> None:
    gate, _ = _ready_gate()
    with pytest.raises(Exception) as exc:
        gate.dispatch("http", TransportRequest(operation_id="accelerate.inference", arguments={}))
    assert "unknown migrated transport" in str(exc.value).lower() or exc.value.code is AdmissionErrorCode.UNKNOWN_ENUM


def test_transport_kind_enum_covers_migrated_set() -> None:
    assert {kind.value for kind in TransportKind} == set(MIGRATED_TRANSPORTS)
