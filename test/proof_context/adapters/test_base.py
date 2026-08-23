"""PCCE-030: provider-neutral CodingAgentAdapter contract tests."""

from __future__ import annotations

from typing import Any

import pytest

from ipfs_accelerate_py.proof_context.adapters.base import (
    ADAPTER_CONTRACT_CID,
    INTERFACE,
    AdapterResult,
    CancellationToken,
    CodingAgentAdapter,
    adapter_contract_cid,
    adapter_contract_descriptor,
    cancel_adapter,
    execute_propose,
    protocol_signature,
)
from ipfs_accelerate_py.proof_context.adapters.models import (
    CODING_AGENT_INVOCATION_SCHEMA,
    CONTEXT_PACK_SCHEMA,
    MODEL_ROUTE_DECISION_SCHEMA,
    PATCH_PROPOSAL_SCHEMA,
    TASK_SPECIFICATION_SCHEMA,
    CodingAgentInvocation,
    ContextPack,
    ModelRouteDecision,
    PatchProposal,
    TaskSpecification,
    parse_wire_record,
    wire_canonical_utf8,
)
from ipfs_accelerate_py.proof_context.errors import (
    BoundaryViolationError,
    ProofCancelledError,
    SimulatedPromotedError,
)

CID = "bafkreihxffr7ppivqrjwb3pafumcoh6mg7vyoglc6gpnmbtjgnq2pdajru"
CID_B = "bafkreihxffr7ppivqrjwb3pafumcoh6mg7vyoglc6gpnmbtjgnq2pdajrv"
CID_C = "bafkreihxffr7ppivqrjwb3pafumcoh6mg7vyoglc6gpnmbtjgnq2pdajrw"
CID_D = "bafkreihxffr7ppivqrjwb3pafumcoh6mg7vyoglc6gpnmbtjgnq2pdajrx"
CID_E = "bafkreihxffr7ppivqrjwb3pafumcoh6mg7vyoglc6gpnmbtjgnq2pdajry"


def _task(**overrides: Any) -> TaskSpecification:
    payload = {
        "schema": TASK_SPECIFICATION_SCHEMA,
        "task_id": "PCCE-030",
        "objective_id": "PCCE-G300",
        "repository_state_cid": CID,
        "owned_paths": ("src/demo/__init__.py",),
        "declared_files": ("src/demo/__init__.py",),
        "route_cid": CID_B,
        "provenance": "live",
    }
    payload.update(overrides)
    return TaskSpecification.from_mapping(payload)


def _pack(**overrides: Any) -> ContextPack:
    payload = {
        "schema": CONTEXT_PACK_SCHEMA,
        "pack_cid": CID_C,
        "repository_state_cid": CID,
        "sufficiency": "sufficient",
        "provenance": "live",
        "task_id": "PCCE-030",
        "capsule_cids": (CID_D,),
    }
    payload.update(overrides)
    return ContextPack.from_mapping(payload)


def _route(**overrides: Any) -> ModelRouteDecision:
    payload = {
        "schema": MODEL_ROUTE_DECISION_SCHEMA,
        "decision_cid": CID_B,
        "task_id": "PCCE-030",
        "tier": "medium",
        "provider": "grok",
        "model": "grok-4.6",
        "revision": "r1",
        "repository_state_cid": CID,
        "provenance": "live",
    }
    payload.update(overrides)
    return ModelRouteDecision.from_mapping(payload)


def _invocation(**overrides: Any) -> CodingAgentInvocation:
    payload = {
        "schema": CODING_AGENT_INVOCATION_SCHEMA,
        "invocation_cid": CID_D,
        "task_id": "PCCE-030",
        "repository_state_cid": CID,
        "route_cid": CID_B,
        "provider": "grok",
        "model": "grok-4.6",
        "revision": "r1",
        "tier": "medium",
        "token_count": 12,
        "cached_token_count": 4,
        "latency_ms": 9,
        "cost_micros": 3,
        "response_artifact_cid": CID_E,
        "provenance": "live",
    }
    payload.update(overrides)
    return CodingAgentInvocation.from_mapping(payload)


def _proposal(**overrides: Any) -> PatchProposal:
    payload = {
        "schema": PATCH_PROPOSAL_SCHEMA,
        "proposal_cid": CID_C,
        "task_id": "PCCE-030",
        "repository_state_cid": CID,
        "invocation_cid": CID_D,
        "patch_cid": CID_E,
        "declared_files": ("src/demo/__init__.py",),
        "provenance": "live",
    }
    payload.update(overrides)
    return PatchProposal.from_mapping(payload)


class _FakeAdapter:
    def __init__(self, result: AdapterResult) -> None:
        self.result = result
        self.cancelled = False

    def propose(
        self,
        task: TaskSpecification,
        context_pack: ContextPack,
        route: ModelRouteDecision,
        cancellation: CancellationToken | None = None,
    ) -> AdapterResult:
        if cancellation is not None:
            cancellation.check()
        return self.result

    def cancel(self, cancellation: CancellationToken) -> None:
        self.cancelled = True
        cancellation.cancel()


def test_wire_records_round_trip_byte_for_byte() -> None:
    records = (_task(), _invocation(), _proposal(), _pack(), _route())
    for record in records:
        encoded = record.to_canonical_utf8()
        restored = type(record).from_mapping(record.to_mapping())
        assert restored.to_canonical_utf8() == encoded
        assert parse_wire_record(dict(record.to_mapping())).to_canonical_utf8() == encoded
        assert encoded == wire_canonical_utf8(dict(record.to_mapping()))


def test_protocol_signature_and_contract_cid_are_stable() -> None:
    snapshot = protocol_signature()
    assert snapshot["interface"] == INTERFACE
    assert snapshot["approval_authority"] is False
    assert snapshot["canonical_branch_authority"] is False
    assert snapshot["provider_bound"] is False
    assert snapshot["propose"]["parameters"] == (
        "task",
        "context_pack",
        "route",
        "cancellation",
    )
    assert snapshot["cancel"]["parameters"] == ("cancellation",)
    cid = adapter_contract_cid()
    assert cid == ADAPTER_CONTRACT_CID
    assert cid.startswith("b")
    descriptor = adapter_contract_descriptor()
    assert descriptor["cid"] == cid
    assert descriptor["interface"] == INTERFACE


def test_execute_propose_returns_admitted_result() -> None:
    result = AdapterResult(proposal=_proposal(), invocation=_invocation(), patch_bytes=b"diff")
    admitted = execute_propose(_FakeAdapter(result), _task(), _pack(), _route())
    assert admitted.proposal.proposal_cid == result.proposal.proposal_cid
    assert admitted.invocation.has_live_evidence()
    assert admitted.accepted is False
    assert admitted.approved is False
    assert admitted.to_mapping()["approval_authority"] is False


def test_undeclared_files_are_rejected() -> None:
    proposal = _proposal(declared_files=("src/demo/secret.py",))
    result = AdapterResult(proposal=proposal, invocation=_invocation())
    with pytest.raises(BoundaryViolationError):
        execute_propose(_FakeAdapter(result), _task(), _pack(), _route())


def test_live_status_without_live_evidence_is_rejected() -> None:
    invocation = _invocation()
    object.__setattr__(invocation, "response_artifact_cid", None)
    result = AdapterResult(proposal=_proposal(), invocation=invocation)
    with pytest.raises(SimulatedPromotedError):
        execute_propose(_FakeAdapter(result), _task(), _pack(), _route())


def test_simulated_cannot_claim_live() -> None:
    invocation = _invocation(provenance="simulated")
    result = AdapterResult(proposal=_proposal(provenance="live"), invocation=invocation)
    with pytest.raises(SimulatedPromotedError):
        execute_propose(_FakeAdapter(result), _task(), _pack(), _route())


def test_self_approval_is_rejected() -> None:
    proposal = _proposal()
    with pytest.raises(BoundaryViolationError):
        object.__setattr__(proposal, "accepted", True)
    result = AdapterResult(proposal=proposal, invocation=_invocation())
    mapping = dict(result.to_mapping())
    mapping["self_approved"] = True
    with pytest.raises(BoundaryViolationError):
        from ipfs_accelerate_py.proof_context.adapters.base import _reject_self_approval

        _reject_self_approval(mapping)


def test_cancellation_cannot_claim_live() -> None:
    token = CancellationToken()
    token.cancel()
    result = AdapterResult(proposal=_proposal(), invocation=_invocation())
    with pytest.raises(ProofCancelledError):
        execute_propose(_FakeAdapter(result), _task(), _pack(), _route(), cancellation=token)
    with pytest.raises(BoundaryViolationError):
        AdapterResult(proposal=_proposal(), invocation=_invocation(), cancelled=True)
    adapter = _FakeAdapter(result)
    live = CancellationToken()
    cancel_adapter(adapter, live)
    assert adapter.cancelled is True
    assert live.cancelled is True


def test_insufficient_context_pack_cannot_propose() -> None:
    pack = _pack(sufficiency="insufficient")
    result = AdapterResult(proposal=_proposal(), invocation=_invocation())
    with pytest.raises(BoundaryViolationError):
        execute_propose(_FakeAdapter(result), _task(), pack, _route())


def test_coding_agent_adapter_is_a_protocol() -> None:
    result = AdapterResult(proposal=_proposal(), invocation=_invocation())
    adapter = _FakeAdapter(result)
    assert isinstance(adapter, CodingAgentAdapter)
