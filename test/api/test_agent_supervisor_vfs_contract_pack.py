from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.vfs_contract_pack import (
    CONTRACT_PACK_AUTHORIZES_REPAIR,
    CONTRACT_PACK_IS_COMPLETION_EVIDENCE,
    CONTRACT_PACK_IS_CORRECTNESS_EVIDENCE,
    CanonicalVfsContractPack,
    ContractSourceKind,
    DataMode,
    ExecutionMode,
    ExpectationIssue,
    ExpectationState,
    FacadeCompatibility,
    IssueKind,
    OperationSupport,
    PublicSurface,
    SourceContract,
    VFS_CANONICAL_OPERATION_MATRIX_SCHEMA,
    VFS_CONTRACT_PACK_GOAL_ID,
    VFS_CONTRACT_PACK_SCHEMA,
    VFS_CONTRACT_PACK_VERSION,
    VfsContractPack,
    VfsContractPackError,
    VfsErrorCode,
    VfsInvariantKind,
    VfsOperation,
    assert_vfs_contract_pack_complete,
    build_vfs_contract_pack,
    canonical_vfs_contract_pack,
    publish_vfs_contract_pack,
)


def _canonical_digest(record: dict[str, object]) -> str:
    unsigned = dict(record)
    unsigned.pop("content_id")
    payload = json.dumps(
        unsigned,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def test_pack_identity_authority_and_serialization_are_deterministic() -> None:
    pack = build_vfs_contract_pack()
    record = pack.to_record()

    assert pack.schema == VFS_CONTRACT_PACK_SCHEMA
    assert pack.operation_matrix_schema == VFS_CANONICAL_OPERATION_MATRIX_SCHEMA
    assert pack.contract_version == VFS_CONTRACT_PACK_VERSION
    assert pack.goal_id == VFS_CONTRACT_PACK_GOAL_ID == "VFS-026"
    assert record["authority"] == {
        "completion_evidence": False,
        "correctness_evidence": False,
        "authorizes_repair": False,
    }
    assert not CONTRACT_PACK_IS_COMPLETION_EVIDENCE
    assert not CONTRACT_PACK_IS_CORRECTNESS_EVIDENCE
    assert not CONTRACT_PACK_AUTHORIZES_REPAIR
    assert pack.content_id == _canonical_digest(record)
    assert pack.to_record() == build_vfs_contract_pack().to_record()
    assert json.loads(pack.to_json()) == record
    assert isinstance(pack, CanonicalVfsContractPack)


def test_operation_and_invariant_vocabularies_are_complete() -> None:
    pack = build_vfs_contract_pack()

    assert {item.operation.value for item in pack.operations} == {
        "path.resolve",
        "mount",
        "read",
        "write",
        "open",
        "close",
        "seek",
        "stat",
        "list",
        "mkdir",
        "remove",
        "rename",
        "copy",
    }
    assert {item.kind.value for item in pack.invariants} == {
        "versioned_path",
        "unicode",
        "root",
        "traversal",
        "mount",
        "read_write",
        "handle_lifecycle",
        "seek",
        "stat_list",
        "directory_mutation",
        "namespace_mutation",
        "bytes_text",
        "sync_async",
        "error",
        "cid_size",
        "atomicity",
        "journal_replay",
        "versioning",
        "cache_pin_coherence",
        "backend_negotiation",
        "authorization",
        "resource",
        "degradation",
    }
    for operation in pack.operations:
        assert set(operation.execution_modes) == {
            ExecutionMode.SYNC,
            ExecutionMode.ASYNC,
        }
        assert operation.invariant_ids
        assert operation.error_codes
        assert type(operation.mutates) is bool
        assert operation.idempotent is None or type(operation.idempotent) is bool

    assert pack.operation_contract(VfsOperation.READ).output_modes == (DataMode.BYTES,)
    assert DataMode.BYTES in pack.operation_contract(VfsOperation.WRITE).input_modes
    assert pack.operation_contract(VfsOperation.OPEN).output_modes == (
        DataMode.HANDLE,
        DataMode.METADATA,
    )
    assert pack.operation_contract(VfsOperation.CLOSE).output_modes == (DataMode.NONE,)
    assert DataMode.HANDLE in pack.operation_contract(VfsOperation.SEEK).input_modes


def test_invariants_define_applicability_and_transport_neutral_errors() -> None:
    pack = build_vfs_contract_pack()
    operations = {item.operation: item for item in pack.operations}

    for invariant in pack.invariants:
        assert invariant.statement
        assert invariant.applies_to
        assert invariant.state is ExpectationState.RESOLVED
        for operation in invariant.applies_to:
            contract = operations[operation]
            assert invariant.invariant_id in contract.invariant_ids
            assert set(invariant.error_codes).issubset(contract.error_codes)

    traversal = pack.invariant_contract(VfsInvariantKind.TRAVERSAL)
    assert VfsOperation.PATH_RESOLVE in traversal.applies_to
    assert VfsErrorCode.TRAVERSAL_DENIED in traversal.error_codes
    atomicity = pack.invariant_contract(VfsInvariantKind.ATOMICITY)
    assert {
        VfsOperation.WRITE,
        VfsOperation.REMOVE,
        VfsOperation.RENAME,
        VfsOperation.COPY,
    }.issubset(atomicity.applies_to)
    authorization = pack.invariant_contract(VfsInvariantKind.AUTHORIZATION)
    assert set(authorization.applies_to) == set(VfsOperation)
    assert VfsErrorCode.PERMISSION_DENIED in authorization.error_codes


def test_every_public_surface_has_an_explicit_operation_mapping() -> None:
    pack = build_vfs_contract_pack()
    sources = {item.source_id: item for item in pack.sources}

    assert {item.surface for item in pack.surfaces} == set(PublicSurface)
    for surface in pack.surfaces:
        assert {item.operation for item in surface.operations} == set(VfsOperation)
        assert surface.execution_modes
        assert surface.transport_error_mapping_required
        for binding in surface.operations:
            assert binding.support in {
                OperationSupport.SUPPORTED,
                OperationSupport.UNSUPPORTED,
            }
            assert binding.source_contract_ids
            assert all(
                sources[source_id].expectation_authority
                for source_id in binding.source_contract_ids
            )

    handle_operations = {
        VfsOperation.OPEN,
        VfsOperation.CLOSE,
        VfsOperation.SEEK,
    }
    for surface_kind in (PublicSurface.CLI, PublicSurface.HTTP):
        surface = pack.surface_contract(surface_kind)
        assert {
            item.operation
            for item in surface.operations
            if item.support is OperationSupport.UNSUPPORTED
        } == handle_operations
    for surface_kind in (
        PublicSurface.PYTHON,
        PublicSurface.MCP,
        PublicSurface.MCP_PLUS_PLUS,
        PublicSurface.LIBP2P,
    ):
        assert set(pack.surface_contract(surface_kind).supported_operations) == set(
            VfsOperation
        )


def test_sources_are_reviewed_and_missing_expectations_remain_unresolved() -> None:
    pack = build_vfs_contract_pack()
    sources = {item.source_id: item for item in pack.sources}

    authoritative = [
        source for source in pack.sources if source.expectation_authority
    ]
    assert authoritative
    assert all(source.available and source.reviewed for source in authoritative)
    assert all(source.kind.may_define_expectation for source in authoritative)

    assert len(pack.unresolved_expectations) == 1
    issue = pack.unresolved_expectations[0]
    assert issue.kind is IssueKind.MISSING
    assert issue.state is ExpectationState.UNRESOLVED
    assert issue.resolution is None
    missing_source = sources[issue.source_contract_ids[0]]
    assert not missing_source.available
    assert not missing_source.expectation_authority


def test_missing_and_conflicting_expectations_fail_closed() -> None:
    pack = build_vfs_contract_pack()

    with pytest.raises(VfsContractPackError, match="must stay unresolved"):
        replace(
            pack.issues[0],
            state=ExpectationState.RESOLVED,
        )
    with pytest.raises(VfsContractPackError, match="at least two sources"):
        ExpectationIssue(
            issue_id="issue:one-sided-conflict",
            kind=IssueKind.CONFLICT,
            subject="One source cannot establish a conflict.",
            source_contract_ids=("source:vfs-026-acceptance",),
            positions=("position-a", "position-b"),
            state=ExpectationState.CONFLICTING,
        )

    conflict = ExpectationIssue(
        issue_id="issue:example-conflict",
        kind=IssueKind.CONFLICT,
        subject="Two reviewed facade contracts disagree.",
        source_contract_ids=(
            "source:vfs-026-acceptance",
            "source:vfs-026-surface:python",
        ),
        positions=("requires stable bytes", "requires implicit text"),
        state=ExpectationState.CONFLICTING,
    )
    conflicted_pack = replace(pack, issues=pack.issues + (conflict,))
    conflict_record = conflicted_pack.to_record()["issues"][-1]
    assert conflict_record["state"] == "conflicting"
    assert conflict_record["resolution"] is None


def test_observations_and_unavailable_sources_cannot_gain_authority() -> None:
    common = {
        "source_id": "source:test",
        "locator": "test://source",
        "revision": "r1",
        "summary": "Test source.",
        "reviewed": True,
    }
    with pytest.raises(VfsContractPackError, match="observation-only"):
        SourceContract(
            kind=ContractSourceKind.IMPLEMENTATION_OBSERVATION,
            **common,
        )
    with pytest.raises(VfsContractPackError, match="available and reviewed"):
        SourceContract(
            kind=ContractSourceKind.REVIEWED_INTERFACE,
            available=False,
            **common,
        )


def test_pack_rejects_incomplete_or_unbacked_resolved_contracts() -> None:
    pack = build_vfs_contract_pack()

    with pytest.raises(VfsContractPackError, match="schema must be"):
        replace(pack, schema="vfs/contract-pack@invented")
    with pytest.raises(VfsContractPackError, match="operation matrix is incomplete"):
        replace(pack.surfaces[0], operations=pack.surfaces[0].operations[:-1])
    with pytest.raises(VfsContractPackError, match="mutates must be a boolean"):
        replace(pack.operations[0], mutates="false")  # type: ignore[arg-type]

    unbacked = replace(
        pack.operations[0],
        source_contract_ids=("source:missing-backend-atomicity-contract",),
    )
    operations = (unbacked,) + pack.operations[1:]
    with pytest.raises(
        VfsContractPackError,
        match="resolves an expectation without reviewed authority",
    ):
        replace(pack, operations=operations)


def test_canonical_vectors_cover_semantic_edge_cases() -> None:
    pack = build_vfs_contract_pack()
    vectors = {item.vector_id: item for item in pack.vectors}
    required = {
        "vector:path:nfc-dot-segments",
        "vector:path:root-traversal-denied",
        "vector:mount:component-boundary",
        "vector:write:utf8-byte-accounting",
        "vector:seek:byte-offset",
        "vector:stat:cid-size",
        "vector:remove:non-empty",
        "vector:journal:duplicate-replay",
        "vector:version:stale-write",
        "vector:auth:precedes-cache",
        "vector:resource:list-limit",
        "vector:degradation:no-silent-fallback",
    }
    assert set(vectors) == required
    assert vectors["vector:path:nfc-dot-segments"].expected["path"] == "/café/data"
    assert (
        vectors["vector:path:root-traversal-denied"].expected["error"]["code"]
        == VfsErrorCode.TRAVERSAL_DENIED.value
    )
    assert vectors["vector:write:utf8-byte-accounting"].expected["size"] == 2
    assert vectors["vector:journal:duplicate-replay"].expected["commits"] == 1
    assert (
        vectors["vector:degradation:no-silent-fallback"].expected["degraded"]
        is False
    )
    assert all(vector.state is ExpectationState.RESOLVED for vector in vectors.values())
    assert all(vector.invariant_ids for vector in vectors.values())


def test_facade_examples_include_compatible_incompatible_and_unresolved_cases() -> None:
    pack = build_vfs_contract_pack()
    sources = {item.source_id: item for item in pack.sources}

    assert {item.surface for item in pack.facade_examples} == set(PublicSurface)
    assert {item.compatibility for item in pack.facade_examples} == {
        FacadeCompatibility.COMPATIBLE,
        FacadeCompatibility.INCOMPATIBLE,
        FacadeCompatibility.UNRESOLVED,
    }
    for example in pack.facade_examples:
        source_contracts = [
            sources[source_id] for source_id in example.source_contract_ids
        ]
        if example.compatibility is FacadeCompatibility.UNRESOLVED:
            assert not any(
                source.expectation_authority for source in source_contracts
            )
        else:
            assert any(source.expectation_authority for source in source_contracts)


def test_alias_validation_and_atomic_publication(tmp_path) -> None:
    pack = canonical_vfs_contract_pack()
    assert_vfs_contract_pack_complete(pack)
    assert isinstance(pack, VfsContractPack)

    destination = tmp_path / "nested" / "vfs-contract-pack.json"
    published = publish_vfs_contract_pack(destination, pack)
    assert published == destination.resolve()
    assert json.loads(destination.read_text(encoding="utf-8")) == pack.to_record()
    assert not list(destination.parent.glob("*.tmp"))

    first_bytes = destination.read_bytes()
    assert publish_vfs_contract_pack(destination, pack) == destination.resolve()
    assert destination.read_bytes() == first_bytes
