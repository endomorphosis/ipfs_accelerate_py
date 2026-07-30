from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.vfs_contract_pack import (
    CONTRACT_PACK_AUTHORIZES_REPAIR,
    CONTRACT_PACK_IS_COMPLETION_EVIDENCE,
    CONTRACT_PACK_IS_CORRECTNESS_EVIDENCE,
    DRIFT_INVENTORY_AUTHORIZES_REPAIR,
    DRIFT_INVENTORY_IS_COMPLETION_EVIDENCE,
    DRIFT_INVENTORY_IS_CORRECTNESS_EVIDENCE,
    DRIFT_INVENTORY_VARIANT_PRESENCE_IS_DEFECT,
    VFS_CANONICAL_OPERATION_MATRIX_CLAIM_SCHEMA,
    VFS_CANONICAL_OPERATION_MATRIX_EVIDENCE_TERMS,
    VFS_CANONICAL_OPERATION_MATRIX_GOAL_ID,
    VFS_CANONICAL_OPERATION_MATRIX_GOAL_PACKET_ID,
    VFS_CANONICAL_OPERATION_MATRIX_OBJECTIVE_REVISION,
    VFS_CANONICAL_OPERATION_MATRIX_PACKET_EVIDENCE_TERMS,
    VFS_CANONICAL_OPERATION_MATRIX_PACKET_GOAL_IDS,
    VFS_CANONICAL_OPERATION_MATRIX_PACKET_TASK_IDS,
    VFS_CANONICAL_OPERATION_MATRIX_PARENT_GOAL_ID,
    VFS_CANONICAL_OPERATION_MATRIX_REQUIRED_INVARIANTS,
    VFS_CANONICAL_OPERATION_MATRIX_SCHEMA,
    VFS_CANONICAL_OPERATION_MATRIX_TASK_ID,
    VFS_CONTRACT_PACK_GOAL_ID,
    VFS_CONTRACT_PACK_SCHEMA,
    VFS_CONTRACT_PACK_VERSION,
    VFS_DIFFERENTIAL_CONTRACT_WITNESS_SCHEMA,
    VFS_DRIFT_INVENTORY_GOAL_ID,
    VFS_DRIFT_INVENTORY_OBJECTIVE_REVISION,
    VFS_DRIFT_INVENTORY_SCHEMA,
    VFS_DRIFT_INVENTORY_SOURCE_REVISION,
    VFS_DRIFT_INVENTORY_TASK_ID,
    CanonicalVfsContractPack,
    ContractSourceKind,
    DataMode,
    DriftAssessment,
    DriftFindingKind,
    DriftSurfaceKind,
    ExecutionMode,
    ExpectationIssue,
    ExpectationState,
    FacadeCompatibility,
    IssueKind,
    OperationSupport,
    PublicSurface,
    SourceContract,
    VfsContractPack,
    VfsContractPackError,
    VfsErrorCode,
    VfsInvariantKind,
    VfsOperation,
    all_covered_evidence_terms,
    assert_vfs_canonical_operation_matrix_complete,
    assert_vfs_contract_pack_complete,
    assert_vfs_drift_inventory_complete,
    build_vfs_contract_pack,
    build_vfs_drift_inventory,
    canonical_operation_matrix_evidence,
    canonical_operation_matrix_evidence_terms,
    canonical_vfs_contract_pack,
    canonical_vfs_drift_inventory,
    covered_evidence_terms,
    packet_evidence_terms,
    prove_vfs_canonical_operation_matrix,
    publish_vfs_contract_pack,
    publish_vfs_drift_inventory,
    vfs_canonical_operation_matrix_satisfies_objective,
)
from ipfs_accelerate_py.agent_supervisor.vfs_differential_harness import (
    VFS_DIFFERENTIAL_EVIDENCE_KINDS,
    VFS_DIFFERENTIAL_PACKET_GOAL_IDS,
    VFS_DIFFERENTIAL_WITNESS_SCHEMA,
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


def test_canonical_matrix_evidence_terms_bind_vfs_g158_packet() -> None:
    assert (
        canonical_operation_matrix_evidence()
        == VFS_CANONICAL_OPERATION_MATRIX_SCHEMA
        == "vfs/canonical-operation-matrix@1"
    )
    assert canonical_operation_matrix_evidence_terms() == (
        "vfs/canonical-operation-matrix@1",
    )
    assert (
        covered_evidence_terms()
        == VFS_CANONICAL_OPERATION_MATRIX_EVIDENCE_TERMS
    )
    assert packet_evidence_terms() == (
        "vfs/differential-contract-witness@1",
        "vfs/canonical-operation-matrix@1",
    )
    assert (
        all_covered_evidence_terms()
        == VFS_CANONICAL_OPERATION_MATRIX_PACKET_EVIDENCE_TERMS
        == packet_evidence_terms()
    )

    assert VFS_CANONICAL_OPERATION_MATRIX_GOAL_ID == "VFS-G158"
    assert VFS_CANONICAL_OPERATION_MATRIX_TASK_ID == "VFS-073"
    assert VFS_CANONICAL_OPERATION_MATRIX_PARENT_GOAL_ID == "VFS-G090"
    assert VFS_CANONICAL_OPERATION_MATRIX_OBJECTIVE_REVISION == (
        "baguqeeramjx4cofpxl4tvz57mno5f3hx6nfxkwp65ydb7il6vjw5hirigdaa"
    )
    assert VFS_CANONICAL_OPERATION_MATRIX_GOAL_PACKET_ID == (
        "goal_packet/vfs_drift/ipfs_accelerate_py/1ad8c79bee6a"
    )
    assert VFS_CANONICAL_OPERATION_MATRIX_PACKET_GOAL_IDS == (
        "VFS-G091",
        "VFS-G158",
    )
    assert VFS_CANONICAL_OPERATION_MATRIX_PACKET_TASK_IDS == (
        "VFS-077",
        "VFS-073",
    )

    # Both packet members publish exactly the same shared evidence vocabulary.
    assert VFS_DIFFERENTIAL_CONTRACT_WITNESS_SCHEMA == (
        VFS_DIFFERENTIAL_WITNESS_SCHEMA
    )
    assert (
        VFS_CANONICAL_OPERATION_MATRIX_PACKET_GOAL_IDS
        == VFS_DIFFERENTIAL_PACKET_GOAL_IDS
    )
    assert (
        VFS_CANONICAL_OPERATION_MATRIX_PACKET_EVIDENCE_TERMS
        == VFS_DIFFERENTIAL_EVIDENCE_KINDS
    )


def test_canonical_matrix_claim_proves_complete_structural_coverage() -> None:
    pack = build_vfs_contract_pack()
    inventory = build_vfs_drift_inventory(pack)

    assert_vfs_canonical_operation_matrix_complete(pack, inventory)
    assert vfs_canonical_operation_matrix_satisfies_objective(pack, inventory)

    claim = prove_vfs_canonical_operation_matrix(pack, inventory)
    assert claim["schema"] == VFS_CANONICAL_OPERATION_MATRIX_CLAIM_SCHEMA
    assert claim["evidence"] == "vfs/canonical-operation-matrix@1"
    assert claim["evidence_terms"] == ["vfs/canonical-operation-matrix@1"]
    assert claim["requirement_id"] == VFS_CANONICAL_OPERATION_MATRIX_SCHEMA
    assert claim["goal_id"] == "VFS-G158"
    assert claim["parent_goal_id"] == "VFS-G090"
    assert claim["task_id"] == "VFS-073"
    assert claim["objective_revision"] == (
        VFS_CANONICAL_OPERATION_MATRIX_OBJECTIVE_REVISION
    )
    assert claim["goal_packet_id"] == (
        VFS_CANONICAL_OPERATION_MATRIX_GOAL_PACKET_ID
    )
    assert claim["packet_goal_ids"] == ["VFS-G091", "VFS-G158"]
    assert claim["packet_task_ids"] == ["VFS-077", "VFS-073"]
    assert claim["packet_evidence_terms"] == [
        "vfs/differential-contract-witness@1",
        "vfs/canonical-operation-matrix@1",
    ]
    assert claim["bindings"] == {
        "contract_pack_content_id": pack.content_id,
        "drift_inventory_content_id": inventory.content_id,
        "contract_version": VFS_CONTRACT_PACK_VERSION,
    }

    coverage = claim["coverage"]
    assert set(coverage["operations"]) == {
        operation.value for operation in VfsOperation
    }
    assert coverage["operation_count"] == len(VfsOperation)
    assert set(coverage["public_surfaces"]) == {
        surface.value for surface in PublicSurface
    }
    assert coverage["public_surface_count"] == len(PublicSurface)
    assert set(coverage["required_invariant_kinds"]) == {
        kind.value for kind in VfsInvariantKind
    }
    assert set(VFS_CANONICAL_OPERATION_MATRIX_REQUIRED_INVARIANTS) == set(
        VfsInvariantKind
    )
    assert set(coverage["resolved_invariant_kinds"]) == {
        kind.value for kind in VfsInvariantKind
    }
    assert set(coverage["execution_modes"]) == {"sync", "async"}
    assert set(coverage["drift_surface_kinds"]) == {
        kind.value for kind in DriftSurfaceKind
    }
    assert {
        "duplicate_candidate",
        "manifest_drift",
        "variant_presence",
    }.issubset(coverage["drift_finding_kinds"])
    assert coverage["unresolved_issue_ids"] == [
        "issue:backend-specific-atomicity"
    ]
    assert coverage["variant_presence_is_defect"] is False
    assert coverage["repair_decision_count"] == 0
    assert coverage["matrix_complete"] is True

    assert claim["sibling_evidence_requirements"] == [
        {
            "evidence": "vfs/differential-contract-witness@1",
            "goal_id": "VFS-G091",
            "task_id": "VFS-077",
            "status": "external_runtime_witness_required",
        }
    ]
    assert claim["satisfied"] is True
    assert claim["claim_level"] == "structural_contract"
    assert claim["claims_runtime_conformance"] is False
    assert claim["authoritative"] is False
    assert claim["completion_authoritative"] is False
    assert claim["semantic_authority"] is False
    assert claim["authorizes_repair"] is False
    assert claim["content_id"] == _canonical_digest(claim)
    assert claim == prove_vfs_canonical_operation_matrix().copy()


def test_canonical_matrix_objective_check_fails_closed_on_forgery() -> None:
    pack = build_vfs_contract_pack()
    inventory = build_vfs_drift_inventory(pack)

    object.__setattr__(pack, "operations", pack.operations[:-1])
    assert not vfs_canonical_operation_matrix_satisfies_objective(pack, inventory)
    with pytest.raises(VfsContractPackError, match="complete VfsOperation"):
        assert_vfs_canonical_operation_matrix_complete(pack, inventory)

    valid_pack = build_vfs_contract_pack()
    forged_inventory = build_vfs_drift_inventory(valid_pack)
    manifest_finding = next(
        finding
        for finding in forged_inventory.findings
        if finding.kind is DriftFindingKind.MANIFEST_DRIFT
    )
    object.__setattr__(
        manifest_finding, "kind", DriftFindingKind.CONTRACT_DRIFT
    )
    assert not vfs_canonical_operation_matrix_satisfies_objective(
        valid_pack, forged_inventory
    )
    with pytest.raises(VfsContractPackError, match="manifest finding"):
        assert_vfs_canonical_operation_matrix_complete(
            valid_pack, forged_inventory
        )


def test_drift_inventory_identity_tracks_the_objective_heap() -> None:
    inventory = build_vfs_drift_inventory()
    record = inventory.to_record()

    assert inventory.schema == VFS_DRIFT_INVENTORY_SCHEMA == "vfs/drift-inventory@1"
    assert inventory.goal_id == VFS_DRIFT_INVENTORY_GOAL_ID == "VFS-G090"
    assert inventory.task_id == VFS_DRIFT_INVENTORY_TASK_ID == "VFS-045"
    assert (
        inventory.objective_revision
        == VFS_DRIFT_INVENTORY_OBJECTIVE_REVISION
        == "baguqeerahsnzkm2u6e6qvh6hnjyrwwwhyf6usdlocisaibw5zyk4ujektotq"
    )
    assert (
        inventory.source_revision
        == VFS_DRIFT_INVENTORY_SOURCE_REVISION
        == "git:f6a574375febbcf9a46fcd24bbc7bc5cfb551de5"
    )
    assert inventory.contract_pack_id == build_vfs_contract_pack().content_id
    assert record["evidence_kinds"] == ["vfs/drift-inventory@1"]
    assert record["authority"] == {
        "completion_evidence": False,
        "correctness_evidence": False,
        "authorizes_repair": False,
        "variant_presence_is_defect": False,
    }
    assert not DRIFT_INVENTORY_IS_COMPLETION_EVIDENCE
    assert not DRIFT_INVENTORY_IS_CORRECTNESS_EVIDENCE
    assert not DRIFT_INVENTORY_AUTHORIZES_REPAIR
    assert not DRIFT_INVENTORY_VARIANT_PRESENCE_IS_DEFECT
    assert inventory.content_id == _canonical_digest(record)
    assert inventory.to_record() == build_vfs_drift_inventory().to_record()
    assert inventory.to_record() == canonical_vfs_drift_inventory().to_record()
    assert json.loads(inventory.to_json()) == record


def test_drift_inventory_covers_surface_families_and_canonical_operations() -> None:
    pack = build_vfs_contract_pack()
    inventory = build_vfs_drift_inventory(pack)
    evidence = {item.evidence_id: item for item in inventory.evidence}
    sources = {item.source_id: item for item in pack.sources}

    assert {
        surface_kind
        for finding in inventory.findings
        for surface_kind in finding.surface_kinds
    } == set(DriftSurfaceKind)
    assert {
        operation
        for finding in inventory.findings
        for operation in finding.canonical_operations
    } == set(VfsOperation)
    assert {item.locator for item in inventory.evidence} >= {
        "ipfs_kit_py/ipfs_kit_py/ipfs_fsspec.py",
        "ipfs_kit_py/ipfs_kit_py/vfs_manager.py",
        "ipfs_kit_py/ipfs_kit_py/bucket_vfs_manager.py",
        "ipfs_kit_py/ipfs_kit_py/filesystem_journal.py",
        "ipfs_kit_py/ipfs_kit_py/vfs_version_tracker.py",
        "ipfs_kit_py/ipfs_kit_py/iroh_vfs.py",
    }
    assert all(
        item.reviewed and item.available and not item.expectation_authority
        for item in inventory.evidence
    )

    for finding in inventory.findings:
        assert finding.evidence_ids
        assert all(evidence_id in evidence for evidence_id in finding.evidence_ids)
        assert finding.canonical_operations
        assert all(
            pack.operation_contract(operation).state is ExpectationState.RESOLVED
            for operation in finding.canonical_operations
        )
        assert all(
            sources[source_id].expectation_authority
            for source_id in finding.source_contract_ids
        )

    core = next(
        item
        for item in inventory.findings
        if item.finding_id == "finding:vfs-core-fsspec"
    )
    assert set(core.canonical_operations) == set(VfsOperation)
    assert {"VFSCore", "IPFSFSSpecFileSystem"}.issubset(
        evidence["evidence:vfs-core-fsspec"].observed_symbols
    )


def test_inventory_findings_are_separate_from_repair_decisions() -> None:
    inventory = build_vfs_drift_inventory()
    record = inventory.to_record()

    assert record["inventory_findings"]
    assert record["repair_decisions"] == []
    assert inventory.repair_decisions == ()
    assert all(item.defect_label is None for item in inventory.findings)
    assert all(item.repair_decision is None for item in inventory.findings)

    variants = [
        item
        for item in inventory.findings
        if item.kind is DriftFindingKind.VARIANT_PRESENCE
    ]
    assert variants
    assert all(item.variant_presence_only for item in variants)
    assert all(item.assessment is DriftAssessment.OBSERVED for item in variants)
    duplicate = next(
        item
        for item in inventory.findings
        if item.kind is DriftFindingKind.DUPLICATE_CANDIDATE
    )
    assert duplicate.assessment is DriftAssessment.UNRESOLVED
    assert duplicate.defect_label is None

    manifest = next(
        item
        for item in inventory.findings
        if item.kind is DriftFindingKind.MANIFEST_DRIFT
    )
    assert manifest.assessment is DriftAssessment.DRIFT
    assert set(manifest.evidence_ids) == {
        "evidence:mcp-vfs-tools",
        "evidence:mcp-js-tools-manifest",
    }
    placeholder = next(
        item
        for item in inventory.findings
        if item.kind is DriftFindingKind.CONTRACT_DRIFT
    )
    assert placeholder.assessment is DriftAssessment.DRIFT

    with pytest.raises(VfsContractPackError, match="cannot contain repair"):
        replace(inventory.findings[0], repair_decision="replace the module")
    with pytest.raises(VfsContractPackError, match="cannot assign a defect"):
        replace(variants[0], defect_label="broken")
    with pytest.raises(VfsContractPackError, match="must remain an observation"):
        replace(variants[0], assessment=DriftAssessment.DRIFT)


def test_drift_inventory_fails_closed_on_incomplete_or_unreviewed_mapping() -> None:
    pack = build_vfs_contract_pack()
    inventory = build_vfs_drift_inventory(pack)

    without_manifest = tuple(
        item
        for item in inventory.findings
        if item.kind is not DriftFindingKind.MANIFEST_DRIFT
    )
    with pytest.raises(VfsContractPackError):
        replace(inventory, findings=without_manifest)

    bad_evidence = replace(
        inventory.findings[0],
        evidence_ids=("evidence:not-in-inventory",),
    )
    with pytest.raises(VfsContractPackError, match="evidence coverage differs"):
        replace(
            inventory,
            findings=(bad_evidence,) + inventory.findings[1:],
        )

    wrong_pack = replace(inventory, contract_pack_id="sha256:" + "0" * 64)
    with pytest.raises(VfsContractPackError, match="does not match"):
        assert_vfs_drift_inventory_complete(wrong_pack, pack)

    unreviewed_mapping = replace(
        inventory.findings[0],
        source_contract_ids=("source:missing-backend-atomicity-contract",),
    )
    unreviewed_inventory = replace(
        inventory,
        findings=(unreviewed_mapping,) + inventory.findings[1:],
    )
    with pytest.raises(VfsContractPackError, match="lacks reviewed mapping authority"):
        assert_vfs_drift_inventory_complete(unreviewed_inventory, pack)


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

    inventory = canonical_vfs_drift_inventory()
    inventory_destination = tmp_path / "nested" / "vfs-drift-inventory.json"
    inventory_published = publish_vfs_drift_inventory(
        inventory_destination, inventory
    )
    assert inventory_published == inventory_destination.resolve()
    assert (
        json.loads(inventory_destination.read_text(encoding="utf-8"))
        == inventory.to_record()
    )
    assert not list(inventory_destination.parent.glob("*.tmp"))
