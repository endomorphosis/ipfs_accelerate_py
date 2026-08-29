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
    _assert_patch_paths_match_declared,
    _parse_unified_diff_paths,
    adapter_contract_cid,
    adapter_contract_descriptor,
    cancel_adapter,
    execute_propose,
    protocol_signature,
)
from ipfs_accelerate_py.proof_context.adapters.models import (
    CODING_AGENT_INVOCATION_SCHEMA,
    CONTEXT_PACK_SCHEMA,
    MAX_PATCH_BYTES,
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
    MalformedError,
    ProofCancelledError,
    SimulatedPromotedError,
)

CID = "bafkreihxffr7ppivqrjwb3pafumcoh6mg7vyoglc6gpnmbtjgnq2pdajru"
CID_B = "bafkreihxffr7ppivqrjwb3pafumcoh6mg7vyoglc6gpnmbtjgnq2pdajrv"
CID_C = "bafkreihxffr7ppivqrjwb3pafumcoh6mg7vyoglc6gpnmbtjgnq2pdajrw"
CID_D = "bafkreihxffr7ppivqrjwb3pafumcoh6mg7vyoglc6gpnmbtjgnq2pdajrx"
CID_E = "bafkreihxffr7ppivqrjwb3pafumcoh6mg7vyoglc6gpnmbtjgnq2pdajry"
OWNED = "src/demo/__init__.py"
HEADER_ONLY_PATCH = f"diff --git a/{OWNED} b/{OWNED}\n".encode()
UNIFIED_PATCH = (
    f"diff --git a/{OWNED} b/{OWNED}\n"
    f"--- a/{OWNED}\n"
    f"+++ b/{OWNED}\n"
    "@@ -1 +1 @@\n"
    "-VALUE = 1\n"
    "+VALUE = 2\n"
).encode()
MODE_ONLY_PATCH = (f"diff --git a/{OWNED} b/{OWNED}\nold mode 100644\nnew mode 100755\n").encode()


def _task(**overrides: Any) -> TaskSpecification:
    payload = {
        "schema": TASK_SPECIFICATION_SCHEMA,
        "task_id": "PCCE-030",
        "objective_id": "PCCE-G300",
        "repository_state_cid": CID,
        "owned_paths": (OWNED,),
        "declared_files": (OWNED,),
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
        "declared_files": (OWNED,),
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


def _execute_patch(
    patch: bytes,
    *,
    declared_files: tuple[str, ...] = (OWNED,),
    owned_paths: tuple[str, ...] | None = None,
) -> AdapterResult:
    owned = owned_paths or declared_files
    task = _task(owned_paths=owned, declared_files=declared_files)
    result = AdapterResult(
        proposal=_proposal(declared_files=declared_files),
        invocation=_invocation(),
        patch_bytes=patch,
    )
    return execute_propose(_FakeAdapter(result), task, _pack(), _route())


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
    result = AdapterResult(
        proposal=_proposal(), invocation=_invocation(), patch_bytes=HEADER_ONLY_PATCH
    )
    admitted = execute_propose(_FakeAdapter(result), _task(), _pack(), _route())
    assert admitted.proposal.proposal_cid == result.proposal.proposal_cid
    assert admitted.invocation.has_live_evidence()
    assert admitted.accepted is False
    assert admitted.approved is False
    assert admitted.to_mapping()["approval_authority"] is False


@pytest.mark.parametrize("patch", (HEADER_ONLY_PATCH, MODE_ONLY_PATCH, UNIFIED_PATCH))
def test_scope_binding_preserves_header_only_mode_only_and_full_replay_evidence(
    patch: bytes,
) -> None:
    admitted = _execute_patch(patch)
    assert admitted.patch_bytes == patch
    assert _parse_unified_diff_paths(patch) == (OWNED,)


def test_scope_binding_covers_every_path_from_multiple_diff_headers() -> None:
    extra = "src/demo/extra.py"
    patch = HEADER_ONLY_PATCH + f"diff --git a/{extra} b/{extra}\n".encode()
    admitted = _execute_patch(
        patch,
        declared_files=(OWNED, extra),
        owned_paths=(OWNED, extra),
    )
    assert admitted.proposal.declared_files == (OWNED, extra)
    assert _parse_unified_diff_paths(patch) == (OWNED, extra)


@pytest.mark.parametrize("kind", ("rename", "copy"))
def test_scope_binding_covers_both_rename_or_copy_paths(kind: str) -> None:
    renamed = "src/demo/renamed.py"
    patch = (
        f"diff --git a/{OWNED} b/{renamed}\n"
        "similarity index 100%\n"
        f"{kind} from {OWNED}\n"
        f"{kind} to {renamed}\n"
    ).encode()
    admitted = _execute_patch(
        patch,
        declared_files=(OWNED, renamed),
        owned_paths=(OWNED, renamed),
    )
    assert admitted.proposal.declared_files == (OWNED, renamed)
    assert _parse_unified_diff_paths(patch) == (OWNED, renamed)


@pytest.mark.parametrize(
    "patch",
    (
        (
            f"diff --git a/{OWNED} b/{OWNED}\n"
            "--- /dev/null\n"
            f"+++ b/{OWNED}\n"
            "@@ -0,0 +1 @@\n"
            "+VALUE = 1\n"
        ).encode(),
        (
            f"diff --git a/{OWNED} b/{OWNED}\n"
            f"--- a/{OWNED}\n"
            "+++ /dev/null\n"
            "@@ -1 +0,0 @@\n"
            "-VALUE = 1\n"
        ).encode(),
    ),
)
def test_scope_binding_admits_paired_add_and_delete_markers(patch: bytes) -> None:
    assert _execute_patch(patch).patch_bytes == patch


@pytest.mark.parametrize(
    "old_path,new_path",
    (
        (f"a/{OWNED}", f"b/{OWNED}"),
        ("/dev/null", f"b/{OWNED}"),
        (f"a/{OWNED}", "/dev/null"),
    ),
)
def test_scope_binding_validates_git_binary_summary_paths(
    old_path: str,
    new_path: str,
) -> None:
    patch = (
        f"diff --git a/{OWNED} b/{OWNED}\nBinary files {old_path} and {new_path} differ\n"
    ).encode()
    assert _execute_patch(patch).patch_bytes == patch


@pytest.mark.parametrize(
    "summary",
    (
        "Binary files a/private/foreign.bin and b/private/foreign.bin differ\n",
        "Binary files /dev/null and b/private/foreign.bin differ\n",
        "Binary files a/private/foreign.bin and /dev/null differ\n",
        f"Binary files a/{OWNED} and b/{OWNED} and b/private/foreign.bin differ\n",
    ),
)
def test_scope_binding_rejects_foreign_or_ambiguous_binary_summaries(summary: str) -> None:
    with pytest.raises((BoundaryViolationError, MalformedError)):
        _execute_patch(HEADER_ONLY_PATCH + summary.encode())


def test_marker_looking_hunk_content_is_not_treated_as_a_file_marker() -> None:
    patch = (
        f"diff --git a/{OWNED} b/{OWNED}\n"
        f"--- a/{OWNED}\n"
        f"+++ b/{OWNED}\n"
        "@@ -1 +1 @@\n"
        "--- old source content\n"
        "+++ new source content\n"
    ).encode()
    assert _execute_patch(patch).patch_bytes == patch


def test_response_scope_lie_is_rejected_without_patch_path_leakage() -> None:
    foreign = "private/api_key=fixture-secret-value.txt"
    patch = f"diff --git a/{foreign} b/{foreign}\n".encode()
    with pytest.raises(BoundaryViolationError) as caught:
        _execute_patch(patch)
    rendered = str(caught.value)
    assert "fixture-secret-value" not in rendered
    assert foreign not in rendered
    assert dict(caught.value.details) == {
        "field": "declared_files",
        "reason": "scope",
    }


def test_non_utf8_rejection_does_not_retain_raw_provider_bytes() -> None:
    private_patch = HEADER_ONLY_PATCH + b"\xffapi_key=raw-fixture-secret"
    with pytest.raises(MalformedError) as caught:
        _execute_patch(private_patch)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert "raw-fixture-secret" not in str(caught.value)


def test_frozen_byte_bound_precedes_patch_scope_parsing() -> None:
    result = AdapterResult(
        proposal=_proposal(),
        invocation=_invocation(),
        patch_bytes=HEADER_ONLY_PATCH,
    )
    object.__setattr__(result, "patch_bytes", b"x" * (MAX_PATCH_BYTES + 1))
    with pytest.raises(BoundaryViolationError, match="frozen byte bound"):
        execute_propose(_FakeAdapter(result), _task(), _pack(), _route())


@pytest.mark.parametrize(
    "patch",
    (
        b"",
        b"opaque provider output\n",
        f"diff --git a/{OWNED} b/{OWNED}\n\x00".encode(),
        f"diff --git a/{OWNED} b/{OWNED}\n".encode() + b"\xff",
    ),
)
def test_scope_binding_rejects_empty_opaque_nul_and_non_utf8_patches(
    patch: bytes,
) -> None:
    with pytest.raises(MalformedError):
        _execute_patch(patch)


@pytest.mark.parametrize(
    "patch",
    (
        f"--- a/{OWNED}\n+++ b/{OWNED}\n".encode(),
        f"diff --git a/{OWNED} b/{OWNED}\n--- a/{OWNED}\n".encode(),
        f"diff --git a/{OWNED} b/{OWNED}\n+++ b/{OWNED}\n".encode(),
        (
            f"diff --git a/{OWNED} b/{OWNED}\n"
            f"--- a/{OWNED}\n"
            f"+++ b/{OWNED}\n"
            f"--- a/{OWNED}\n"
            f"+++ b/{OWNED}\n"
        ).encode(),
    ),
)
def test_scope_binding_rejects_foreign_unpaired_and_repeated_file_markers(
    patch: bytes,
) -> None:
    with pytest.raises(MalformedError):
        _execute_patch(patch)


@pytest.mark.parametrize(
    "old_marker,new_marker",
    (
        ("a/src/demo/foreign.py", f"b/{OWNED}"),
        (f"a/{OWNED}", "b/src/demo/foreign.py"),
        (f"b/{OWNED}", f"b/{OWNED}"),
        (f"a/{OWNED}", f"a/{OWNED}"),
    ),
)
def test_scope_binding_rejects_disagreeing_or_wrong_side_file_markers(
    old_marker: str,
    new_marker: str,
) -> None:
    patch = (f"diff --git a/{OWNED} b/{OWNED}\n--- {old_marker}\n+++ {new_marker}\n").encode()
    with pytest.raises((BoundaryViolationError, MalformedError)):
        _execute_patch(patch)


@pytest.mark.parametrize(
    "extended_headers",
    (
        f"rename from {OWNED}\nrename to src/demo/foreign.py\n",
        f"rename from src/demo/foreign.py\nrename to {OWNED}\n",
        f"copy from {OWNED}\ncopy to src/demo/foreign.py\n",
        f"copy from src/demo/foreign.py\ncopy to {OWNED}\n",
    ),
)
def test_scope_binding_rejects_extended_headers_that_disagree_with_diff_header(
    extended_headers: str,
) -> None:
    patch = HEADER_ONLY_PATCH + extended_headers.encode()
    with pytest.raises(BoundaryViolationError):
        _execute_patch(patch)


@pytest.mark.parametrize(
    "extended_headers",
    (
        f"rename from {OWNED}\n",
        f"rename to {OWNED}\n",
        f"copy from {OWNED}\n",
        f"copy to {OWNED}\n",
        f"rename from {OWNED}\ncopy to {OWNED}\n",
        f"rename from {OWNED}\nrename from {OWNED}\nrename to {OWNED}\n",
    ),
)
def test_scope_binding_rejects_unpaired_or_mixed_extended_headers(
    extended_headers: str,
) -> None:
    patch = HEADER_ONLY_PATCH + extended_headers.encode()
    with pytest.raises(MalformedError):
        _execute_patch(patch)


@pytest.mark.parametrize(
    "header",
    (
        "diff --git a/../outside.py b/../outside.py\n",
        f"diff --git /{OWNED} b/{OWNED}\n",
        f"diff --git a/{OWNED} c/{OWNED}\n",
        f"diff --git a/{OWNED} b/{OWNED} b/ambiguous.py\n",
        f"diff --git  a/{OWNED} b/{OWNED}\n",
        f'diff --git "a/{OWNED}" "b/{OWNED}"\n',
        "diff --git a/src\\demo.py b/src\\demo.py\n",
        f"diff --cc {OWNED}\n",
        f"diff --combined {OWNED}\n",
    ),
)
def test_scope_binding_rejects_escaping_or_ambiguous_diff_headers(header: str) -> None:
    with pytest.raises((BoundaryViolationError, MalformedError)):
        _execute_patch(header.encode())


@pytest.mark.parametrize(
    "hunk",
    (
        "@@ -1,2 +1 @@\n-old only\n",
        "@@ -1 +1,2 @@\n+new only\n",
        "@@ -1 +1 @@\n?invalid body\n",
        "@@ -1,x +1 @@\n",
        "@@ -1 +1,y @@\n",
    ),
)
def test_scope_binding_rejects_malformed_or_incomplete_hunk_counts(hunk: str) -> None:
    with pytest.raises(MalformedError):
        _execute_patch(HEADER_ONLY_PATCH + hunk.encode())


def test_scope_binding_requires_exact_path_set_in_both_directions() -> None:
    extra = "src/demo/extra.py"
    with pytest.raises(BoundaryViolationError):
        _execute_patch(
            HEADER_ONLY_PATCH,
            declared_files=(OWNED, extra),
            owned_paths=(OWNED, extra),
        )
    two_path_patch = HEADER_ONLY_PATCH + f"diff --git a/{extra} b/{extra}\n".encode()
    with pytest.raises(BoundaryViolationError):
        _execute_patch(two_path_patch)


def test_scope_binding_rejects_forged_empty_declared_path_population() -> None:
    proposal = _proposal()
    object.__setattr__(proposal, "declared_files", ())
    result = AdapterResult(
        proposal=proposal,
        invocation=_invocation(),
        patch_bytes=HEADER_ONLY_PATCH,
    )
    with pytest.raises(BoundaryViolationError, match="at least one"):
        execute_propose(_FakeAdapter(result), _task(), _pack(), _route())


def test_private_scope_validator_is_pure_and_exact() -> None:
    _assert_patch_paths_match_declared(UNIFIED_PATCH, (OWNED,))
    with pytest.raises(BoundaryViolationError):
        _assert_patch_paths_match_declared(UNIFIED_PATCH, ("src/demo/foreign.py",))


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
