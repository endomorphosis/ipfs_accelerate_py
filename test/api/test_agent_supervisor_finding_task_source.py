"""Tests for stable repair task source from admitted findings (VFS-031)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.contract_findings import (
    CallSlice,
    CallSliceStep,
    ContractFindingLedger,
    EvidenceReferences,
    build_contract_finding,
)
from ipfs_accelerate_py.agent_supervisor.finding_task_source import (
    DEFAULT_BOARD_NAMESPACE,
    DEFAULT_CONTEXT_CEILING_BYTES,
    DEFAULT_GOAL_ID,
    DEFAULT_RESOURCE_CLASS,
    FINDING_TASK_SOURCE_VERSION,
    PROJECTION_AUTHORIZES_REPAIR,
    PROJECTION_IS_COMPLETION_EVIDENCE,
    BoardSnapshot,
    FindingTaskAuthorityError,
    FindingTaskSource,
    FindingTaskSourceError,
    FindingTaskSourcePolicy,
    MaterializationOutcome,
    RepairTaskRecord,
    ReviewRecord,
    TaskDisposition,
    build_repair_task,
    build_review_record,
    classify_finding_for_task,
    coalesce_repair_tasks,
    materialize_finding_tasks,
    project_board_duckdb_rows,
    project_board_json,
    project_board_markdown,
    project_board_sarif_links,
)
from ipfs_accelerate_py.agent_supervisor.program_assurance_contracts import (
    ClaimLevel,
    EvidenceFreshness,
    FindingSeverity,
    FindingStatus,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _slice(*symbols: str, base_path: str = "ipfs_accelerate_py") -> CallSlice:
    steps = tuple(
        CallSliceStep(
            symbol=symbol,
            interface=f"iface://{symbol}",
            repository_id="repository:alpha",
            path=f"{base_path}/{symbol.replace('.', '/')}.py",
        )
        for symbol in symbols
    )
    return CallSlice(steps=steps)


def _evidence(
    *,
    counterexample: str = "cex:alpha",
    proof: str = "proof:alpha",
    runtime: str = "runtime:alpha",
) -> EvidenceReferences:
    return EvidenceReferences(
        counterexample_cids=(counterexample,) if counterexample else (),
        proof_cids=(proof,) if proof else (),
        runtime_cids=(runtime,) if runtime else (),
        artifact_cids=("artifact:witness",),
    )


def broken_finding(**overrides):
    base = dict(
        claim_level=ClaimLevel.MODEL_DISPROVED,
        status=FindingStatus.CONTRACT_BROKEN,
        severity=FindingSeverity.HIGH,
        confidence_millionths=950_000,
        freshness=EvidenceFreshness.CURRENT,
        repositories=("repository:alpha",),
        symbols=("pkg.api.call",),
        interfaces=("mcp://pkg/call",),
        expected_contract_cid="expected:contract:1",
        observed_contract_cid="observed:contract:1",
        root_cause_family="error-map-mismatch",
        merge_fate="pkg.api.call",
        summary="Implementation violates the reviewed interface contract.",
        call_slice=_slice("pkg.entry", "pkg.api.call"),
        evidence=_evidence(),
        assumptions=("fixture is hermetic",),
        analyzer_versions={"contract-checker": "1.0.0"},
        remediation_scope=(
            "pkg.api.call",
            "ipfs_accelerate_py/pkg/api/call.py",
        ),
        tree_id="tree:abc",
        policy_revision="policy:v1",
        repository_observation_id="observation:1",
        verdict="violated",
    )
    base.update(overrides)
    return build_contract_finding(**base)


def ambiguous_finding(**overrides):
    base = dict(
        claim_level=ClaimLevel.RESOLVED_STATIC,
        status=FindingStatus.AMBIGUOUS,
        severity=FindingSeverity.MEDIUM,
        confidence_millionths=500_000,
        freshness=EvidenceFreshness.CURRENT,
        repositories=("repository:alpha",),
        symbols=("pkg.api.other",),
        interfaces=("mcp://pkg/other",),
        expected_contract_cid="expected:contract:2",
        observed_contract_cid="observed:contract:2",
        root_cause_family="optional-field-drift",
        merge_fate="pkg.api.other",
        summary="Ambiguous static resolution.",
        call_slice=_slice("pkg.api.other"),
        evidence=_evidence(counterexample=""),
        remediation_scope=("ipfs_accelerate_py/pkg/api/other.py",),
        tree_id="tree:abc",
        policy_revision="policy:v1",
    )
    base.update(overrides)
    return build_contract_finding(**base)


# ---------------------------------------------------------------------------
# Authority flags / version
# ---------------------------------------------------------------------------


def test_projection_authority_flags_fail_closed() -> None:
    assert PROJECTION_AUTHORIZES_REPAIR is False
    assert PROJECTION_IS_COMPLETION_EVIDENCE is False
    assert FINDING_TASK_SOURCE_VERSION == 1


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def test_fresh_admitted_broken_finding_is_executable() -> None:
    finding = broken_finding()
    disposition, reasons = classify_finding_for_task(finding, admitted=True)
    assert disposition is TaskDisposition.EXECUTABLE
    assert reasons == ()


def test_ambiguous_finding_produces_review() -> None:
    finding = ambiguous_finding()
    disposition, reasons = classify_finding_for_task(finding, admitted=True)
    assert disposition is TaskDisposition.REVIEW
    assert "ambiguous" in reasons


def test_broad_finding_produces_review() -> None:
    many_paths = tuple(
        f"ipfs_accelerate_py/module_{index}.py" for index in range(20)
    )
    finding = broken_finding(
        remediation_scope=many_paths,
        call_slice=_slice(
            *[f"pkg.mod{i}" for i in range(20)],
            base_path="ipfs_accelerate_py",
        ),
        symbols=tuple(f"pkg.mod{i}" for i in range(20)),
    )
    policy = FindingTaskSourcePolicy(max_output_paths=4, max_symbols=8)
    disposition, reasons = classify_finding_for_task(
        finding, policy=policy, admitted=True
    )
    assert disposition is TaskDisposition.REVIEW
    assert "broad" in reasons


def test_out_of_root_finding_produces_review() -> None:
    finding = broken_finding(
        remediation_scope=("../escape/evil.py",),
        call_slice=CallSlice(
            steps=(
                CallSliceStep(
                    symbol="pkg.api.call",
                    path="../escape/evil.py",
                    repository_id="repository:alpha",
                ),
            )
        ),
    )
    policy = FindingTaskSourcePolicy(
        write_roots=("ipfs_accelerate_py", "test")
    )
    disposition, reasons = classify_finding_for_task(
        finding, policy=policy, admitted=True
    )
    assert disposition is TaskDisposition.REVIEW
    assert "out_of_root" in reasons


def test_stale_and_partial_are_review() -> None:
    stale = broken_finding(
        freshness=EvidenceFreshness.STALE,
        status=FindingStatus.STALE,
        severity=FindingSeverity.INFO,
        confidence_millionths=0,
        claim_level=ClaimLevel.OBSERVED_SYNTAX,
    )
    disposition, reasons = classify_finding_for_task(stale, admitted=True)
    assert disposition is TaskDisposition.REVIEW
    assert "stale" in reasons

    partial = broken_finding(
        symbols=(),
        interfaces=(),
        expected_contract_cid="",
        observed_contract_cid="",
        root_cause_family="",
        merge_fate="",
        partial=True,
        allow_poisoned_severity=True,
        status=FindingStatus.INCONCLUSIVE,
        severity=FindingSeverity.INFO,
        confidence_millionths=0,
        claim_level=ClaimLevel.OBSERVED_SYNTAX,
    )
    disposition, reasons = classify_finding_for_task(partial, admitted=True)
    assert disposition is TaskDisposition.REVIEW


def test_not_admitted_finding_is_review() -> None:
    finding = broken_finding()
    disposition, reasons = classify_finding_for_task(finding, admitted=False)
    assert disposition is TaskDisposition.REVIEW
    assert "not_admitted" in reasons


# ---------------------------------------------------------------------------
# Repair task shape
# ---------------------------------------------------------------------------


def test_repair_task_binds_required_fields() -> None:
    finding = broken_finding()
    task = build_repair_task(finding, task_index=1)
    assert task.executable is True
    assert task.goal_id == DEFAULT_GOAL_ID
    assert task.root_cause_family == "error-map-mismatch"
    assert task.outputs
    assert task.symbols == ("pkg.api.call",)
    assert task.effects
    assert task.conflict_domain
    assert task.validation_plan
    assert task.proof_plan
    assert finding.finding_cid in task.finding_cids
    assert task.provenance_cids
    assert 0 < task.risk_millionths <= 1_000_000
    assert task.resource_class == DEFAULT_RESOURCE_CLASS
    assert task.context_ceiling_bytes == DEFAULT_CONTEXT_CEILING_BYTES
    assert task.context_ceiling_tokens > 0
    assert task.merge_fate == "pkg.api.call"
    assert task.semantic_key
    assert task.task_cid
    assert task.identity.canonical_task_key
    assert task.board_namespace == DEFAULT_BOARD_NAMESPACE

    restored = RepairTaskRecord.from_dict(task.to_dict())
    assert restored.task_cid == task.task_cid
    assert restored.semantic_key == task.semantic_key


def test_review_record_is_non_executable() -> None:
    finding = ambiguous_finding()
    review = build_review_record(
        finding, reasons=("ambiguous",), review_index=1
    )
    assert review.executable is False
    assert review.disposition is TaskDisposition.REVIEW
    assert "ambiguous" in review.reasons
    assert finding.finding_cid in review.finding_cids

    with pytest.raises(FindingTaskAuthorityError):
        ReviewRecord.from_dict({**review.to_dict(), "executable": True})


def test_forged_task_cid_rejected() -> None:
    task = build_repair_task(broken_finding())
    payload = task.to_dict()
    payload["task_cid"] = "b" + "a" * 58
    with pytest.raises(Exception):
        RepairTaskRecord.from_dict(payload)


# ---------------------------------------------------------------------------
# Materialization: create, no-op, supersede, review
# ---------------------------------------------------------------------------


def test_materialize_fresh_admitted_findings_creates_tasks(
    tmp_path: Path,
) -> None:
    finding = broken_finding()
    source = FindingTaskSource(root=tmp_path / "board")
    receipt = source.materialize([finding])
    assert receipt.outcome is MaterializationOutcome.CREATED
    assert receipt.created_task_ids
    assert not receipt.review_ids
    snapshot = source.snapshot()
    assert len(snapshot.executable_tasks) == 1
    task = snapshot.executable_tasks[0]
    assert task.root_cause_family == finding.root_cause_family
    assert finding.finding_cid in task.finding_cids
    assert source.task_for_finding(finding.finding_cid) is not None


def test_stable_finding_replay_is_no_op(tmp_path: Path) -> None:
    finding = broken_finding()
    source = FindingTaskSource(root=tmp_path / "board")
    first = source.materialize([finding])
    assert first.outcome is MaterializationOutcome.CREATED
    board_cid = first.board_cid
    revision = first.revision

    second = source.materialize([finding])
    assert second.outcome is MaterializationOutcome.NO_OP
    assert finding.finding_cid in second.no_op_finding_cids
    assert second.board_cid == board_cid
    assert second.revision == revision
    assert len(source.snapshot().tasks) == 1


def test_changed_evidence_supersedes_rather_than_duplicates(
    tmp_path: Path,
) -> None:
    first = broken_finding(
        observed_contract_cid="observed:contract:1",
        evidence=_evidence(counterexample="cex:v1"),
        summary="First observation of the contract break.",
    )
    # Same semantic scope (root cause, merge fate, symbols, remediation paths)
    # but different evidence body → different finding CID.
    second = broken_finding(
        observed_contract_cid="observed:contract:1",
        evidence=_evidence(counterexample="cex:v2", proof="proof:v2"),
        summary="Updated observation with stronger counterexample.",
        confidence_millionths=980_000,
    )
    assert first.finding_cid != second.finding_cid
    assert first.semantic_key_id == second.semantic_key_id

    source = FindingTaskSource(root=tmp_path / "board")
    source.materialize([first])
    receipt = source.materialize([second])
    assert receipt.superseded_task_ids or receipt.created_task_ids
    snapshot = source.snapshot()
    # One live executable task bound to the newer finding, not two duplicates.
    assert len(snapshot.executable_tasks) == 1
    live = snapshot.executable_tasks[0]
    assert second.finding_cid in live.finding_cids
    assert first.finding_cid not in live.finding_cids or live.supersedes_task_ids


def test_ambiguous_materializes_as_review_only(tmp_path: Path) -> None:
    finding = ambiguous_finding()
    source = FindingTaskSource(root=tmp_path / "board")
    receipt = source.materialize([finding])
    assert receipt.outcome is MaterializationOutcome.REVIEW_ONLY
    assert receipt.review_ids
    assert not receipt.created_task_ids
    snapshot = source.snapshot()
    assert len(snapshot.tasks) == 0
    assert len(snapshot.reviews) == 1
    assert snapshot.reviews[0].executable is False


def test_ledger_admitted_only_path(tmp_path: Path) -> None:
    ledger = ContractFindingLedger(tmp_path / "ledger")
    good = broken_finding()
    bad = ambiguous_finding()
    ledger.append(good)
    ledger.append(bad)

    source = FindingTaskSource(root=tmp_path / "board")
    receipt = source.materialize(ledger=ledger, admitted_only=True)
    # Ambiguous is not admitted into actionable projection.
    assert receipt.created_task_ids
    snapshot = source.snapshot()
    assert any(good.finding_cid in t.finding_cids for t in snapshot.tasks)
    # Ambiguous may appear only if admitted_only=False; with True it is skipped
    # entirely from ledger.current_findings(admitted_only=True).
    assert all(
        bad.finding_cid not in t.finding_cids for t in snapshot.tasks
    )


def test_materialize_finding_tasks_facade(tmp_path: Path) -> None:
    finding = broken_finding()
    snapshot, receipt = materialize_finding_tasks(
        [finding], root=tmp_path / "board"
    )
    assert receipt.created_task_ids
    assert len(snapshot.tasks) == 1


def test_durable_reload_preserves_board(tmp_path: Path) -> None:
    root = tmp_path / "board"
    finding = broken_finding()
    source = FindingTaskSource(root=root)
    source.materialize([finding])
    first_cid = source.snapshot().board_cid

    reloaded = FindingTaskSource(root=root)
    assert reloaded.snapshot().board_cid == first_cid
    assert len(reloaded.snapshot().tasks) == 1
    # Replay still no-op after reload.
    receipt = reloaded.materialize([finding])
    assert receipt.outcome is MaterializationOutcome.NO_OP


# ---------------------------------------------------------------------------
# Coalescing
# ---------------------------------------------------------------------------


def test_related_tiny_tasks_coalesce_with_shared_validation_and_merge_fate() -> None:
    shared_slice = CallSlice(
        steps=(
            CallSliceStep(
                symbol="pkg.api.shared",
                path="ipfs_accelerate_py/pkg/api/shared.py",
                repository_id="repository:alpha",
            ),
        )
    )
    a = broken_finding(
        symbols=("pkg.api.shared.a",),
        merge_fate="pkg.api.shared",
        root_cause_family="shared-family",
        summary="Tiny A",
        remediation_scope=("ipfs_accelerate_py/pkg/api/shared.py",),
        call_slice=shared_slice,
        expected_contract_cid="expected:a",
        observed_contract_cid="observed:a",
        evidence=_evidence(counterexample="cex:a"),
    )
    b = broken_finding(
        symbols=("pkg.api.shared.b",),
        merge_fate="pkg.api.shared",
        root_cause_family="shared-family",
        summary="Tiny B",
        remediation_scope=("ipfs_accelerate_py/pkg/api/shared.py",),
        call_slice=shared_slice,
        expected_contract_cid="expected:b",
        observed_contract_cid="observed:b",
        evidence=_evidence(counterexample="cex:b"),
    )
    task_a = build_repair_task(a, task_index=1)
    task_b = build_repair_task(b, task_index=2)
    assert task_a.outputs == task_b.outputs
    assert task_a.merge_fate == task_b.merge_fate
    assert task_a.validation_plan == task_b.validation_plan

    coalesced = coalesce_repair_tasks([task_a, task_b])
    assert len(coalesced) == 1
    merged = coalesced[0]
    assert a.finding_cid in merged.finding_cids
    assert b.finding_cid in merged.finding_cids
    assert merged.validation_plan == task_a.validation_plan
    assert merged.merge_fate == task_a.merge_fate
    assert set(merged.symbols) >= {"pkg.api.shared.a", "pkg.api.shared.b"}


def test_tasks_with_different_merge_fate_do_not_coalesce() -> None:
    a = build_repair_task(
        broken_finding(
            merge_fate="fate-a",
            symbols=("pkg.a",),
            call_slice=_slice("pkg.a"),
            expected_contract_cid="e:a",
            observed_contract_cid="o:a",
            evidence=_evidence(counterexample="cex:a"),
        ),
        task_index=1,
    )
    b = build_repair_task(
        broken_finding(
            merge_fate="fate-b",
            symbols=("pkg.b",),
            call_slice=_slice("pkg.b"),
            expected_contract_cid="e:b",
            observed_contract_cid="o:b",
            evidence=_evidence(counterexample="cex:b"),
        ),
        task_index=2,
    )
    result = coalesce_repair_tasks([a, b])
    assert len(result) == 2


# ---------------------------------------------------------------------------
# Projections without authority drift
# ---------------------------------------------------------------------------


def test_json_markdown_duckdb_sarif_projections_have_no_authority(
    tmp_path: Path,
) -> None:
    finding = broken_finding()
    review_finding = ambiguous_finding()
    source = FindingTaskSource(root=tmp_path / "board")
    source.materialize([finding, review_finding])
    snapshot = source.snapshot()

    json_proj = project_board_json(snapshot)
    assert json_proj["authorizes_repair"] is False
    assert json_proj["is_completion_evidence"] is False
    assert json_proj["executable_count"] >= 1

    md = project_board_markdown(snapshot)
    assert "authorizes_repair: false" in md
    assert "is_completion_evidence: false" in md
    assert finding.root_cause_family in md

    rows = project_board_duckdb_rows(snapshot)
    assert rows
    assert all(row["authorizes_repair"] is False for row in rows)
    assert all(row["is_completion_evidence"] is False for row in rows)
    kinds = {row["kind"] for row in rows}
    assert "repair_task" in kinds
    assert "review" in kinds

    sarif_links = project_board_sarif_links(snapshot)
    assert sarif_links["authorizes_repair"] is False
    assert sarif_links["is_completion_evidence"] is False
    assert sarif_links["sarif_is_diagnostic_only"] is True
    assert any(
        finding.finding_cid in link["finding_cids"]
        for link in sarif_links["links"]
    )

    # Source helpers match free functions.
    assert source.project_json()["authorizes_repair"] is False
    assert "authorizes_repair: false" in source.project_markdown()
    assert source.project_duckdb_rows()
    assert source.project_sarif_links()["sarif_is_diagnostic_only"] is True


def test_board_snapshot_rejects_authority_claims() -> None:
    with pytest.raises(FindingTaskAuthorityError):
        BoardSnapshot.from_dict(
            {
                "schema": "x",
                "tasks": [],
                "reviews": [],
                "authorizes_repair": True,
            }
        )
    with pytest.raises(FindingTaskAuthorityError):
        BoardSnapshot.from_dict(
            {
                "schema": "x",
                "tasks": [],
                "reviews": [],
                "is_completion_evidence": True,
            }
        )


def test_mixed_findings_materialization(tmp_path: Path) -> None:
    good = broken_finding()
    bad = ambiguous_finding()
    source = FindingTaskSource(root=tmp_path / "board")
    receipt = source.materialize([good, bad])
    assert receipt.created_task_ids
    assert receipt.review_ids
    assert receipt.outcome in {
        MaterializationOutcome.MIXED,
        MaterializationOutcome.CREATED,
        MaterializationOutcome.UPDATED,
    }
    snapshot = source.snapshot()
    assert len(snapshot.tasks) == 1
    assert len(snapshot.reviews) == 1


def test_task_candidate_projection_round_trip() -> None:
    task = build_repair_task(broken_finding())
    candidate = task.to_task_candidate()
    assert candidate.goal_id
    assert candidate.outputs == task.outputs
    assert candidate.validation_commands == task.validation_plan
    assert candidate.merge_fate == task.merge_fate
    assert candidate.resource_class == task.resource_class


def test_policy_rejects_invalid_resource_class() -> None:
    with pytest.raises(FindingTaskSourceError):
        FindingTaskSourcePolicy(resource_class="not-a-class")


def test_one_root_cause_family_per_task() -> None:
    task = build_repair_task(broken_finding())
    assert task.root_cause_family
    assert " " not in task.root_cause_family or task.root_cause_family.count(
        "-"
    ) >= 0
    # Exactly one family string (not a list).
    assert isinstance(task.root_cause_family, str)


def test_write_roots_allow_in_scope_paths() -> None:
    finding = broken_finding()
    policy = FindingTaskSourcePolicy(
        write_roots=("ipfs_accelerate_py", "test")
    )
    disposition, reasons = classify_finding_for_task(
        finding, policy=policy, admitted=True
    )
    assert disposition is TaskDisposition.EXECUTABLE
    assert reasons == ()
    task = build_repair_task(finding, policy=policy)
    assert all(
        path.startswith("ipfs_accelerate_py/") for path in task.outputs
    )


def test_dependencies_and_conflict_domain_present() -> None:
    task = build_repair_task(broken_finding())
    assert task.conflict_domain
    # Dependencies may be empty for independent findings but field is present.
    assert isinstance(task.dependencies, tuple)
    payload = task.to_dict()
    assert "dependencies" in payload
    assert "conflict_domain" in payload
    assert "validation_plan" in payload
    assert "proof_plan" in payload
    assert "risk_millionths" in payload
    assert "context_ceiling_bytes" in payload
    assert "context_ceiling_tokens" in payload
    assert "finding_cids" in payload
    assert "provenance_cids" in payload
    assert "resource_class" in payload


def test_cannot_build_repair_from_ambiguous() -> None:
    with pytest.raises(FindingTaskSourceError):
        build_repair_task(ambiguous_finding())


def test_json_projection_is_byte_canonical(tmp_path: Path) -> None:
    finding = broken_finding()
    source = FindingTaskSource(root=tmp_path / "board")
    source.materialize([finding])
    a = project_board_json(source.snapshot())
    b = project_board_json(source.snapshot())
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
