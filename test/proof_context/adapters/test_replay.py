"""PCCE-033 deterministic and fail-closed offline replay tests."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
from ipfs_accelerate_py.proof_context.adapters.base import (
    CancellationToken,
    execute_propose,
)
from ipfs_accelerate_py.proof_context.adapters.models import (
    CODING_AGENT_INVOCATION_SCHEMA,
    CONTEXT_PACK_SCHEMA,
    MAX_LOG_BYTES,
    MODEL_ROUTE_DECISION_SCHEMA,
    PATCH_PROPOSAL_SCHEMA,
    TASK_SPECIFICATION_SCHEMA,
    CodingAgentInvocation,
    ContextPack,
    ModelRouteDecision,
    PatchProposal,
    TaskSpecification,
    wire_canonical_utf8,
)
from ipfs_accelerate_py.proof_context.adapters.replay import (
    ADAPTER,
    REPLAY_BINDING_SCHEMA,
    ReplayAdapter,
    ReplayFixture,
    cid_for_bytes,
)
from ipfs_accelerate_py.proof_context.errors import (
    BoundaryViolationError,
    IdentityInconsistentError,
    MalformedError,
    ProofCancelledError,
)

REPOSITORY_CID = cid_for_bytes(b"repository")
PACK_CID = cid_for_bytes(b"context-pack")
ROUTE_CID = cid_for_bytes(b"route")
INVOCATION_CID = cid_for_bytes(b"original-invocation")
PROPOSAL_CID = cid_for_bytes(b"original-proposal")
OTHER_CID = cid_for_bytes(b"other")
PATCH = b"diff --git a/src/demo.py b/src/demo.py\n"
RECORDED_RESPONSE = b'{"recorded":true}'
RECORDED_LOG = b"recorded provider diagnostics"


def _records(
    *,
    provider: str = "offline",
    revision: str = "recorded-r1",
    pack_cid: str = PACK_CID,
    route_cid: str = ROUTE_CID,
) -> tuple[TaskSpecification, ContextPack, ModelRouteDecision]:
    task = TaskSpecification.from_mapping(
        {
            "schema": TASK_SPECIFICATION_SCHEMA,
            "task_id": "PCCE-033",
            "objective_id": "PCCE-G300",
            "repository_state_cid": REPOSITORY_CID,
            "owned_paths": ["src/demo.py"],
            "declared_files": ["src/demo.py"],
            "route_cid": route_cid,
            "provenance": "live",
        }
    )
    pack = ContextPack.from_mapping(
        {
            "schema": CONTEXT_PACK_SCHEMA,
            "pack_cid": pack_cid,
            "repository_state_cid": REPOSITORY_CID,
            "sufficiency": "sufficient",
            "task_id": "PCCE-033",
            "provenance": "live",
        }
    )
    route = ModelRouteDecision.from_mapping(
        {
            "schema": MODEL_ROUTE_DECISION_SCHEMA,
            "decision_cid": route_cid,
            "task_id": "PCCE-033",
            "repository_state_cid": REPOSITORY_CID,
            "provider": provider,
            "model": "recorded-model",
            "revision": revision,
            "tier": "medium",
            "provenance": "live",
        }
    )
    return task, pack, route


def _original_records(
    task: TaskSpecification,
    route: ModelRouteDecision,
    *,
    response: bytes = RECORDED_RESPONSE,
    invocation_cid: str = INVOCATION_CID,
    proposal_cid: str = PROPOSAL_CID,
    patch: bytes = PATCH,
) -> tuple[CodingAgentInvocation, PatchProposal]:
    invocation = CodingAgentInvocation.from_mapping(
        {
            "schema": CODING_AGENT_INVOCATION_SCHEMA,
            "invocation_cid": invocation_cid,
            "task_id": task.task_id,
            "repository_state_cid": task.repository_state_cid,
            "route_cid": route.decision_cid,
            "provider": route.provider,
            "model": route.model,
            "revision": route.revision,
            "tier": route.tier,
            "token_count": 12,
            "cached_token_count": 4,
            "latency_ms": 9,
            "cost_micros": 3,
            "response_artifact_cid": cid_for_bytes(response),
            "provenance": "live",
        }
    )
    proposal = PatchProposal.from_mapping(
        {
            "schema": PATCH_PROPOSAL_SCHEMA,
            "proposal_cid": proposal_cid,
            "task_id": task.task_id,
            "repository_state_cid": task.repository_state_cid,
            "invocation_cid": invocation.invocation_cid,
            "patch_cid": cid_for_bytes(patch),
            "declared_files": ["src/demo.py"],
            "provenance": "live",
        }
    )
    return invocation, proposal


def _fixture(
    *,
    response: bytes = RECORDED_RESPONSE,
    invocation_cid: str = INVOCATION_CID,
    proposal_cid: str = PROPOSAL_CID,
    patch: bytes = PATCH,
    log: bytes = RECORDED_LOG,
) -> tuple[
    ReplayFixture,
    TaskSpecification,
    ContextPack,
    ModelRouteDecision,
]:
    task, pack, route = _records()
    invocation, proposal = _original_records(
        task,
        route,
        response=response,
        invocation_cid=invocation_cid,
        proposal_cid=proposal_cid,
        patch=patch,
    )
    return (
        ReplayFixture.create(
            task=task,
            context_pack=pack,
            route=route,
            response_artifact=response,
            original_invocation=invocation,
            original_proposal=proposal,
            patch_bytes=patch,
            log_bytes=log,
        ),
        task,
        pack,
        route,
    )


def _adapter(
    selected: ReplayFixture,
    *fixtures: ReplayFixture,
) -> ReplayAdapter:
    return ReplayAdapter(
        (selected, *fixtures),
        selected_fixture_cid=selected.fixture_cid,
        selected_response_artifact_cid=selected.response_artifact_cid,
    )


def test_original_records_are_copied_and_permanently_labeled_replayed() -> None:
    fixture, task, pack, route = _fixture()
    result = execute_propose(_adapter(fixture), task, pack, route)

    assert fixture.original_invocation.provenance == "live"
    assert fixture.original_proposal.provenance == "live"
    assert result.invocation.provenance == "replayed"
    assert result.proposal.provenance == "replayed"
    assert result.invocation.invocation_cid == fixture.original_invocation.invocation_cid
    assert result.proposal.proposal_cid == fixture.original_proposal.proposal_cid
    assert result.proposal.invocation_cid == result.invocation.invocation_cid
    assert result.invocation.response_artifact_cid is None

    expected_invocation = dict(fixture.original_invocation.to_mapping())
    expected_invocation["provenance"] = "replayed"
    expected_invocation.pop("response_artifact_cid")
    expected_proposal = dict(fixture.original_proposal.to_mapping())
    expected_proposal["provenance"] = "replayed"
    assert dict(result.invocation.to_mapping()) == expected_invocation
    assert dict(result.proposal.to_mapping()) == expected_proposal


def test_recorded_usage_and_cost_are_preserved_exactly() -> None:
    fixture, task, pack, route = _fixture()
    result = execute_propose(_adapter(fixture), task, pack, route)

    fields = ("token_count", "cached_token_count", "latency_ms", "cost_micros")
    assert (
        tuple(getattr(result.invocation, name) for name in fields)
        == tuple(getattr(fixture.original_invocation, name) for name in fields)
        == (12, 4, 9, 3)
    )


def test_byte_identical_inputs_return_byte_identical_replayed_results() -> None:
    fixture, task, pack, route = _fixture()
    adapter = _adapter(fixture)
    first = execute_propose(adapter, task, pack, route)
    second = execute_propose(adapter, task, pack, route)

    assert first.proposal.to_canonical_bytes() == second.proposal.to_canonical_bytes()
    assert first.invocation.to_canonical_bytes() == second.invocation.to_canonical_bytes()
    assert first.patch_bytes == second.patch_bytes == PATCH
    assert first.log_bytes == second.log_bytes
    assert first.accepted is first.approved is False


def test_fixture_response_and_original_identity_verify_and_round_trip() -> None:
    fixture, *_ = _fixture()
    restored = ReplayFixture.from_mapping(fixture.to_mapping())

    assert restored.to_mapping() == fixture.to_mapping()
    assert restored.response_artifact_cid == cid_for_bytes(RECORDED_RESPONSE)
    assert restored.original_invocation.response_artifact_cid == restored.response_artifact_cid
    with pytest.raises(IdentityInconsistentError):
        replace(fixture, response_artifact=b"forged")
    with pytest.raises(IdentityInconsistentError):
        replace(
            fixture,
            original_invocation=replace(
                fixture.original_invocation,
                response_artifact_cid=OTHER_CID,
            ),
        )
    payload = dict(fixture.to_mapping())
    payload["fixture_cid"] = OTHER_CID
    with pytest.raises(IdentityInconsistentError):
        ReplayFixture.from_mapping(payload)


def test_unsourced_or_empty_required_evidence_is_rejected() -> None:
    fixture, task, pack, route = _fixture()
    invocation = replace(fixture.original_invocation, response_artifact_cid=None)
    with pytest.raises(MalformedError, match="exact bytes"):
        cid_for_bytes(None)
    with pytest.raises(MalformedError, match="original_invocation"):
        ReplayFixture.create(
            task=task,
            context_pack=pack,
            route=route,
            response_artifact=RECORDED_RESPONSE,
            original_invocation=None,
            original_proposal=fixture.original_proposal,
            patch_bytes=PATCH,
            log_bytes=RECORDED_LOG,
        )
    with pytest.raises(MalformedError, match="missing response artifact"):
        ReplayFixture.create(
            task=task,
            context_pack=pack,
            route=route,
            response_artifact=RECORDED_RESPONSE,
            original_invocation=invocation,
            original_proposal=fixture.original_proposal,
            patch_bytes=PATCH,
            log_bytes=RECORDED_LOG,
        )
    with pytest.raises(MalformedError, match="must not be empty"):
        ReplayFixture.create(
            task=task,
            context_pack=pack,
            route=route,
            response_artifact=b"",
            original_invocation=fixture.original_invocation,
            original_proposal=fixture.original_proposal,
            patch_bytes=PATCH,
            log_bytes=RECORDED_LOG,
        )
    with pytest.raises(MalformedError, match="must not be empty"):
        ReplayFixture.create(
            task=task,
            context_pack=pack,
            route=route,
            response_artifact=RECORDED_RESPONSE,
            original_invocation=fixture.original_invocation,
            original_proposal=fixture.original_proposal,
            patch_bytes=b"",
            log_bytes=RECORDED_LOG,
        )
    payload = dict(fixture.to_mapping())
    del payload["response_artifact_base64"]
    with pytest.raises(MalformedError, match="closed field set"):
        ReplayFixture.from_mapping(payload)


def test_source_records_cannot_arrive_prelabelled_as_replay() -> None:
    fixture, *_ = _fixture()
    replayed_invocation = replace(fixture.original_invocation, provenance="replayed")
    replayed_proposal = replace(fixture.original_proposal, provenance="replayed")

    with pytest.raises(BoundaryViolationError, match="exact live original"):
        replace(fixture, original_invocation=replayed_invocation)
    with pytest.raises(BoundaryViolationError, match="exact live original"):
        replace(fixture, original_proposal=replayed_proposal)


def test_selector_requires_exact_adapter_fixture_and_response_identities() -> None:
    fixture, *_ = _fixture()
    with pytest.raises(IdentityInconsistentError, match="exactly one"):
        ReplayAdapter(
            (fixture,),
            selected_fixture_cid=OTHER_CID,
            selected_response_artifact_cid=fixture.response_artifact_cid,
        )
    with pytest.raises(IdentityInconsistentError, match="exactly one"):
        ReplayAdapter(
            (fixture,),
            selected_fixture_cid=fixture.fixture_cid,
            selected_response_artifact_cid=OTHER_CID,
        )
    with pytest.raises(IdentityInconsistentError, match="adapter identity"):
        ReplayAdapter(
            (fixture,),
            selected_fixture_cid=fixture.fixture_cid,
            selected_response_artifact_cid=fixture.response_artifact_cid,
            adapter_id="OtherReplayAdapter@0.1",
        )


def test_distinct_responses_for_one_request_are_explicitly_disambiguated() -> None:
    first, task, pack, route = _fixture(
        response=b'{"recorded":"first"}',
        invocation_cid=cid_for_bytes(b"first invocation"),
        proposal_cid=cid_for_bytes(b"first proposal"),
    )
    second, *_ = _fixture(
        response=b'{"recorded":"second"}',
        invocation_cid=cid_for_bytes(b"second invocation"),
        proposal_cid=cid_for_bytes(b"second proposal"),
    )

    first_result = execute_propose(_adapter(first, second), task, pack, route)
    second_result = execute_propose(_adapter(second, first), task, pack, route)
    assert first_result.invocation.invocation_cid == first.original_invocation.invocation_cid
    assert second_result.invocation.invocation_cid == second.original_invocation.invocation_cid
    assert first_result.log_bytes != second_result.log_bytes
    assert first.response_artifact_cid.encode() in first_result.log_bytes
    assert second.response_artifact_cid.encode() in second_result.log_bytes

    with pytest.raises(IdentityInconsistentError, match="exactly one"):
        ReplayAdapter(
            (first, second),
            selected_fixture_cid=first.fixture_cid,
            selected_response_artifact_cid=second.response_artifact_cid,
        )


def test_result_binding_is_canonical_bounded_and_visible() -> None:
    fixture, task, pack, route = _fixture()
    result = execute_propose(_adapter(fixture), task, pack, route)
    binding_line, recorded_log = result.log_bytes.split(b"\n", 1)
    binding = json.loads(binding_line)

    assert binding_line == wire_canonical_utf8(binding).encode("utf-8")
    assert binding["schema"] == REPLAY_BINDING_SCHEMA
    assert binding["adapter_id"] == ADAPTER
    assert binding["fixture_cid"] == fixture.fixture_cid
    assert binding["response_artifact_cid"] == fixture.response_artifact_cid
    binding_body = dict(binding)
    binding_cid = binding_body.pop("binding_cid")
    assert binding_cid == cid_for_bytes(wire_canonical_utf8(binding_body).encode("utf-8"))
    assert recorded_log == RECORDED_LOG
    assert len(result.log_bytes) <= MAX_LOG_BYTES

    with pytest.raises(BoundaryViolationError, match="log exceeds"):
        _fixture(log=b"x" * MAX_LOG_BYTES)


@pytest.mark.parametrize(
    "kind",
    ["task", "pack", "route", "provider", "revision"],
)
def test_every_request_identity_mismatch_fails_closed(kind: str) -> None:
    fixture, task, pack, route = _fixture()
    adapter = _adapter(fixture)
    if kind == "task":
        task = TaskSpecification.from_mapping(
            {**dict(task.to_mapping()), "task_id": "PCCE-033-other"}
        )
        pack = ContextPack.from_mapping({**dict(pack.to_mapping()), "task_id": "PCCE-033-other"})
        route = ModelRouteDecision.from_mapping(
            {**dict(route.to_mapping()), "task_id": "PCCE-033-other"}
        )
    elif kind == "pack":
        pack = ContextPack.from_mapping({**dict(pack.to_mapping()), "pack_cid": OTHER_CID})
    elif kind == "route":
        route = ModelRouteDecision.from_mapping(
            {**dict(route.to_mapping()), "decision_cid": OTHER_CID}
        )
        task = TaskSpecification.from_mapping({**dict(task.to_mapping()), "route_cid": OTHER_CID})
    elif kind == "provider":
        route = ModelRouteDecision.from_mapping(
            {**dict(route.to_mapping()), "provider": "different-provider"}
        )
    else:
        route = ModelRouteDecision.from_mapping(
            {**dict(route.to_mapping()), "revision": "different-revision"}
        )
    with pytest.raises(IdentityInconsistentError):
        execute_propose(adapter, task, pack, route)


@pytest.mark.parametrize(
    ("field_name", "noncanonical"),
    [
        ("response_artifact_base64", "ZE=="),
        ("patch_base64", "ZE=="),
        ("log_base64", "ZE=="),
    ],
)
def test_base64_requires_decode_reencode_canonical_equality(
    field_name: str,
    noncanonical: str,
) -> None:
    fixture, *_ = _fixture()
    payload = dict(fixture.to_mapping())
    payload[field_name] = noncanonical
    with pytest.raises(MalformedError, match="canonical base64"):
        ReplayFixture.from_mapping(payload)


def test_adapter_has_no_external_effect_and_honours_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, task, pack, route = _fixture()

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("replay must not perform external effects")

    monkeypatch.setattr("subprocess.Popen", forbidden)
    monkeypatch.setattr("socket.socket", forbidden)
    result = execute_propose(_adapter(fixture), task, pack, route)
    assert result.patch_bytes == PATCH

    cancellation = CancellationToken()
    cancellation.cancel()
    with pytest.raises(ProofCancelledError):
        execute_propose(_adapter(fixture), task, pack, route, cancellation)
