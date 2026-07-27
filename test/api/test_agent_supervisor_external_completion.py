from __future__ import annotations

import base64
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.goal_completion import CompletionEvidence
from ipfs_accelerate_py.agent_supervisor.core.external_completion import (
    ExternalArtifactIdentity,
    ExternalCompletionAuthority,
    ExternalCompletionRequirement,
    ExternalOperationalCompletionReceipt,
    HSSLEV2398A61,
    evaluate_external_completion_authority,
    inspect_external_source,
    load_external_completion_authority,
    validate_cid,
)
from ipfs_accelerate_py.agent_supervisor.objectives.objective_daemon import (
    build_arg_parser,
    run_objective_daemon,
)
from ipfs_accelerate_py.agent_supervisor.objectives.objective_graph import (
    build_bundle_task_payloads,
    generate_objective_todos,
    materialize_task_dependency_dag,
    parse_goal_heap,
    scan_objective_gaps,
)
from ipfs_accelerate_py.agent_supervisor.objective_tracker import (
    completion_tree_identity,
    reconcile_objective_goal_completion,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.task_identity import canonical_content_cid


EVIDENCE_TERM = "HSSLEV_EXTERNAL_OPERATION_COMPLETE"
OBSERVED_AT = "2026-07-25T12:00:00+00:00"


def _cid(name: str) -> str:
    return canonical_content_cid({"synthetic_test_identity": name})


def test_external_completion_cids_require_canonical_multiformats_encoding():
    canonical = _cid("canonical")
    body = canonical[1:]
    padding = "=" * ((8 - len(body) % 8) % 8)
    raw = base64.b32decode(body.upper() + padding)
    # Encode CID version 1 as the overlong varint 0x81 0x00. It decodes to
    # the same integer but is not a canonical multiformats representation.
    noncanonical_raw = b"\x81\x00" + raw[1:]
    noncanonical = (
        "b"
        + base64.b32encode(noncanonical_raw)
        .decode("ascii")
        .rstrip("=")
        .lower()
    )

    assert validate_cid(canonical, field_name="canonical") == canonical
    with pytest.raises(ValueError, match="valid CIDv1"):
        validate_cid(noncanonical, field_name="noncanonical")


def test_external_completion_implementation_marker_is_not_a_run_receipt():
    marker = HSSLEV2398A61()
    assert "external operational completion authority" in marker
    assert "completed" not in marker


def _git(cwd: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    return completed.stdout.strip()


def _seed_repo(tmp_path: Path) -> tuple[Path, Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    objective_path = repo / "objective.md"
    todo_path = repo / "todo.md"
    proof_path = repo / "src" / "implementation.py"
    proof_path.parent.mkdir()
    proof_path.write_text(
        f"IMPLEMENTATION_MARKER = {EVIDENCE_TERM!r}\n",
        encoding="utf-8",
    )
    objective_path.write_text(
        f"""# Objective Heap

## EXT-G001 External operational run

- Status: active
- Priority: P0
- Track: benchmark
- Evidence: {EVIDENCE_TERM}
- Completion authority: external
- Acceptance: Execute the operational protocol outside this repository.
- Validation: test -f src/implementation.py
- Gap task: Produce a source-bound external completion receipt.
""",
        encoding="utf-8",
    )
    todo_path.write_text("# Drained task board\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "seed synthetic operational objective")
    return repo, objective_path, todo_path


def _authority(
    repo: Path,
    objective_path: Path,
    *,
    include_receipt: bool = True,
    artifacts: tuple[ExternalArtifactIdentity, ...] | None = None,
    run_plan_cid: str | None = None,
    parent_ledger_cid: str | None = None,
) -> ExternalCompletionAuthority:
    source = inspect_external_source(
        repo,
        objective_path=objective_path,
    ).identity
    producer_id = _cid("external-producer")
    validator_id = _cid("independent-validator")
    required_artifact_ids = (_cid("runtime-receipt-slot"), _cid("resource-slot"))
    requirement = ExternalCompletionRequirement(
        goal_id="EXT-G001",
        evidence_term=EVIDENCE_TERM,
        source_identity_cid=source.source_identity_cid,
        run_plan_cid=_cid("run-plan"),
        parent_ledger_cid=_cid("parent-ledger"),
        required_artifact_ids=required_artifact_ids,
        expected_producer_id=producer_id,
        expected_validator_id=validator_id,
    )
    if artifacts is None:
        artifacts = (
            ExternalArtifactIdentity(
                artifact_id=required_artifact_ids[0],
                artifact_cid=_cid("runtime-receipt"),
            ),
            ExternalArtifactIdentity(
                artifact_id=required_artifact_ids[1],
                artifact_cid=_cid("resource-receipt"),
            ),
        )
    receipts = ()
    if include_receipt:
        receipts = (
            ExternalOperationalCompletionReceipt(
                goal_id="EXT-G001",
                evidence_term=EVIDENCE_TERM,
                source=source,
                run_plan_cid=run_plan_cid or requirement.run_plan_cid,
                parent_ledger_cid=(
                    parent_ledger_cid or requirement.parent_ledger_cid
                ),
                artifacts=artifacts,
                producer_id=producer_id,
                validator_id=validator_id,
                validator_receipt_cid=_cid("validator-receipt"),
                observed_at=OBSERVED_AT,
                fresh_until="2026-07-25T13:00:00+00:00",
            ),
        )
    return ExternalCompletionAuthority(
        requirements=(requirement,),
        receipts=receipts,
    )


def _completion_gate(
    repo: Path,
    objective_path: Path,
) -> dict[str, object]:
    identity = completion_tree_identity(
        repo,
        objective_path=objective_path,
    )
    binding = {
        "repository_id": identity.repository_id,
        "tree_id": identity.tree_id,
    }
    return {
        "coverage": {
            "verified": True,
            "repository_tree": identity.tree_id,
            "evaluated_at": OBSERVED_AT,
            "criteria": [
                {"criterion": EVIDENCE_TERM, "status": "verified"}
            ],
        },
        "analyzer_health": {
            "status": "healthy",
            "healthy": True,
            "safe_for_completion_reasoning": True,
            "exhaustive": True,
        },
        "exhaustion_quorum": {
            "satisfied": True,
            "required_members": 2,
            "member_count": 2,
            "binding": binding,
            "members": [
                {
                    "member_id": "synthetic-exhaustive",
                    "evidence_channel": "exhaustive",
                    "receipt_cid": _cid("exhaustive"),
                    "scan_mode": "exhaustive",
                    "analyzer_version": "external-completion-test/v1",
                    "passed": True,
                    "analyzer_health": {
                        "status": "healthy",
                        "healthy": True,
                    },
                    "exhaustive": True,
                    "safe_for_completion_reasoning": True,
                    "conclusive": True,
                    "contradicted": False,
                    "finished_at": OBSERVED_AT,
                    "binding": binding,
                },
                {
                    "member_id": "synthetic-audit",
                    "evidence_channel": "audit",
                    "receipt_cid": _cid("audit"),
                    "scan_mode": "audit",
                    "analyzer_version": "external-completion-test/v1",
                    "passed": True,
                    "analyzer_health": {
                        "status": "healthy",
                        "healthy": True,
                    },
                    "exhaustive": True,
                    "safe_for_completion_reasoning": True,
                    "conclusive": True,
                    "contradicted": False,
                    "finished_at": OBSERVED_AT,
                    "binding": binding,
                },
            ],
        },
    }


def test_declared_external_goal_is_governed_before_first_authority(
    tmp_path,
):
    repo, objective_path, todo_path = _seed_repo(tmp_path)

    result = reconcile_objective_goal_completion(
        repo_root=repo,
        objective_path=objective_path,
        todo_path=todo_path,
        completion_gate_records={
            "EXT-G001": _completion_gate(repo, objective_path)
        },
        now=OBSERVED_AT,
    )

    assert result.verified_goal_ids == []
    assert result.external_completion["governed_goal_ids"] == [
        "EXT-G001"
    ]
    decision = result.decisions["EXT-G001"]["external_completion"]
    assert decision["results"][0]["reason_codes"] == [
        "external_authority_not_supplied"
    ]


@pytest.mark.parametrize(
    "authority_field",
    (
        "Completion authority kind: external",
        "External completion required: true",
    ),
)
def test_external_authority_aliases_are_governed_during_reconciliation(
    tmp_path,
    authority_field,
):
    repo, objective_path, todo_path = _seed_repo(tmp_path)
    objective_text = objective_path.read_text(encoding="utf-8")
    objective_path.write_text(
        objective_text.replace(
            "Completion authority: external",
            authority_field,
        ),
        encoding="utf-8",
    )
    _git(repo, "add", "objective.md")
    _git(repo, "commit", "-m", "use external authority declaration alias")

    result = reconcile_objective_goal_completion(
        repo_root=repo,
        objective_path=objective_path,
        todo_path=todo_path,
        completion_gate_records={
            "EXT-G001": _completion_gate(repo, objective_path)
        },
        now=OBSERVED_AT,
    )

    assert result.verified_goal_ids == []
    assert result.external_completion["governed_goal_ids"] == ["EXT-G001"]
    decision = result.decisions["EXT-G001"]["external_completion"]
    assert decision["results"][0]["reason_codes"] == [
        "external_authority_not_supplied"
    ]


def test_external_completion_is_two_phase_and_marker_text_is_not_authority(
    tmp_path,
):
    repo, objective_path, todo_path = _seed_repo(tmp_path)
    marker_only = _authority(
        repo,
        objective_path,
        include_receipt=False,
    )
    local_identity = completion_tree_identity(
        repo,
        objective_path=objective_path,
    )
    locally_asserted = CompletionEvidence(
        acceptance_criterion=EVIDENCE_TERM,
        producing_task_or_scan="local-task-metadata",
        validation_receipt={"attempted": True, "passed": True},
        validation_passed=True,
        repository_tree=local_identity.tree_id,
        freshness=True,
        observed_at=OBSERVED_AT,
        provenance_cid=_cid("locally-asserted-evidence"),
    )
    first = reconcile_objective_goal_completion(
        repo_root=repo,
        objective_path=objective_path,
        todo_path=todo_path,
        completion_gate_records={
            "EXT-G001": _completion_gate(repo, objective_path)
        },
        completion_evidence_records={"EXT-G001": [locally_asserted]},
        external_completion_authority=marker_only,
        now=OBSERVED_AT,
    )

    assert first.verified_goal_ids == []
    assert first.provisional_goal_ids == ["EXT-G001"]
    assert first.external_completion["results"][0]["reason_codes"] == [
        "external_receipt_missing"
    ]
    marker_goal = parse_goal_heap(
        objective_path.read_text(encoding="utf-8")
    )[0]
    assert marker_goal.status == "provisionally_complete"

    _git(repo, "add", "objective.md")
    _git(repo, "commit", "-m", "record provisional objective state")
    authority = _authority(repo, objective_path)
    provisional = reconcile_objective_goal_completion(
        repo_root=repo,
        objective_path=objective_path,
        todo_path=todo_path,
        completion_gate_records={
            "EXT-G001": _completion_gate(repo, objective_path)
        },
        external_completion_authority=authority,
        now=OBSERVED_AT,
    )
    assert provisional.verified_goal_ids == ["EXT-G001"]

    completed_goal = parse_goal_heap(
        objective_path.read_text(encoding="utf-8")
    )[0]
    assert completed_goal.status == "verified_complete"
    persisted = json.loads(completed_goal.fields["completion_evidence_records"])
    serialized = json.dumps(persisted, sort_keys=True)
    assert authority.receipts[0].receipt_cid in serialized
    assert authority.receipts[0].validator_receipt_cid in serialized
    assert authority.receipts[0].source.outer_commit in serialized
    assert "src/implementation.py" not in serialized
    assert "IMPLEMENTATION_MARKER" not in serialized

    revoked = ExternalCompletionAuthority(
        requirements=authority.requirements,
        receipts=(),
    )
    reopened = reconcile_objective_goal_completion(
        repo_root=repo,
        objective_path=objective_path,
        todo_path=todo_path,
        completion_gate_records={
            "EXT-G001": _completion_gate(repo, objective_path)
        },
        external_completion_authority=revoked,
        now=OBSERVED_AT,
    )
    assert reopened.reopened_goal_ids == ["EXT-G001"]


def test_verified_external_gate_stays_nonlocal_while_downstream_work_reopens(
    tmp_path,
):
    repo, objective_path, todo_path = _seed_repo(tmp_path)
    objective_path.write_text(
        objective_path.read_text(encoding="utf-8")
        + f"""

## EXT-G002 Downstream local implementation

- Status: active
- Parent: EXT-G001
- Priority: P1
- Track: benchmark
- Evidence: HSSL_DOWNSTREAM_STAGE_MISSING_EVIDENCE
- Outputs: src/downstream.py
- Validation: true
- Gap task: Implement the authorized downstream stage.
""",
        encoding="utf-8",
    )
    _git(repo, "add", "objective.md")
    _git(repo, "commit", "-m", "add externally gated downstream goal")

    first = reconcile_objective_goal_completion(
        repo_root=repo,
        objective_path=objective_path,
        todo_path=todo_path,
        completion_gate_records={
            "EXT-G001": _completion_gate(repo, objective_path)
        },
        external_completion_authority=_authority(repo, objective_path),
        now=OBSERVED_AT,
    )
    assert first.provisional_goal_ids == ["EXT-G001"]

    _git(repo, "add", "objective.md")
    _git(repo, "commit", "-m", "record external gate provisional state")
    completed = reconcile_objective_goal_completion(
        repo_root=repo,
        objective_path=objective_path,
        todo_path=todo_path,
        completion_gate_records={
            "EXT-G001": _completion_gate(repo, objective_path)
        },
        external_completion_authority=_authority(repo, objective_path),
        now=OBSERVED_AT,
    )
    assert completed.verified_goal_ids == ["EXT-G001"]

    findings = scan_objective_gaps(
        repo,
        objective_path=objective_path,
        max_findings=10,
        embedding_min_score=2.0,
    )
    assert [finding.goal_id for finding in findings] == ["EXT-G002"]
    assert findings[0].parent_goal_ids == []
    assert findings[0].external_authority_blockers == []

    bundle_dir = repo / "data" / "agent_supervisor" / "objective_bundles"
    generated = generate_objective_todos(
        repo_root=repo,
        objective_path=objective_path,
        todo_path=todo_path,
        discovery_dir=repo / "data" / "agent_supervisor" / "discovery",
        bundle_dir=bundle_dir,
        task_prefix="DOWNSTREAM-",
        max_findings=10,
        persist_ast_dataset=False,
        write_todo_vector_index=False,
    )
    assert [record.finding.goal_id for record in generated] == ["EXT-G002"]
    downstream_payload = build_bundle_task_payloads(
        bundle_dir / "index.json"
    )[0]
    assert downstream_payload["claimable"] is True
    assert downstream_payload["ready_member_task_ids"] == ["DOWNSTREAM-001"]

    legacy_external_task = materialize_task_dependency_dag(
        [
            {
                "task_id": "LEGACY-EXTERNAL-GATE",
                "task_cid": "cid-legacy-external-gate",
                "goal_id": "EXT-G001",
                "completion_authority": "external",
                "status": "completed",
            }
        ],
        merge_receipts={
            "cid-legacy-external-gate": {
                "status": "succeeded",
                "receipt_cid": "local-merge-receipt",
            }
        },
    )
    legacy_schedule = legacy_external_task.schedule[0]
    assert legacy_schedule.claimable is False
    assert "cid-legacy-external-gate" in (
        legacy_external_task.invalid_task_cids
    )
    assert any(
        repair.kind == "external_authority_required"
        for repair in legacy_external_task.repair_evidence
    )


def test_reopened_external_gate_cannot_advance_descendant_in_same_reconciliation(
    tmp_path,
):
    repo, objective_path, todo_path = _seed_repo(tmp_path)
    child_marker = "HSSL_DESCENDANT_MUST_REMAIN_GATED"
    child_source = repo / "src" / "downstream.py"
    child_source.write_text(
        f"DOWNSTREAM_MARKER = {child_marker!r}\n",
        encoding="utf-8",
    )
    objective_path.write_text(
        f"""# Objective Heap

## EXT-G001 Previously verified external gate

- Status: verified_complete
- Priority: P0
- Track: benchmark
- Evidence: {EVIDENCE_TERM}
- Completion authority: external
- External completion authority CID: forged-authority-routing-hint
- External completion receipt CIDs: ["forged-receipt-routing-hint"]
- External completion validation: [{{"goal_id":"EXT-G001","evidence_term":"{EVIDENCE_TERM}","valid":true,"receipt_cid":"forged-receipt-routing-hint"}}]
- Completion evidence records: [{{"acceptance_criterion":"{EVIDENCE_TERM}","provenance_cid":"forged-receipt-routing-hint","metadata":{{"external_operational_completion":true}}}}]
- Validation: test -f src/implementation.py

## EXT-G002 Locally visible downstream evidence

- Status: active
- Parent: EXT-G001
- Priority: P1
- Track: benchmark
- Evidence: {child_marker}
- Validation: test -f src/downstream.py
""",
        encoding="utf-8",
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "seed stale external routing hint")

    result = reconcile_objective_goal_completion(
        repo_root=repo,
        objective_path=objective_path,
        todo_path=todo_path,
        now=OBSERVED_AT,
    )

    assert result.reopened_goal_ids == ["EXT-G001"]
    assert "EXT-G002" not in result.decisions
    goals = {goal.goal_id: goal for goal in parse_goal_heap(
        objective_path.read_text(encoding="utf-8")
    )}
    assert goals["EXT-G001"].status == "reopened"
    assert goals["EXT-G002"].status == "active"


def test_daemon_without_reconciliation_fences_recorded_gate_but_keeps_local_work(
    tmp_path,
):
    repo, objective_path, todo_path = _seed_repo(tmp_path)
    objective_path.write_text(
        """# Objective Heap

## EXT-G001 Recorded external operational gate with declaration removed

- Status: verified_complete
- Priority: P0
- Track: benchmark
- Evidence: HSSL_RECORDED_EXTERNAL_GATE
- External completion authority CID: recorded-authority-routing-hint
- External completion receipt CIDs: ["recorded-receipt-routing-hint"]
- External completion validation: [{"goal_id":"EXT-G001","evidence_term":"HSSL_RECORDED_EXTERNAL_GATE","valid":true,"receipt_cid":"recorded-receipt-routing-hint"}]
- Completion evidence records: [{"acceptance_criterion":"HSSL_RECORDED_EXTERNAL_GATE","provenance_cid":"recorded-receipt-routing-hint","metadata":{"external_operational_completion":true}}]
- Validation: true

## EXT-G002 Externally gated downstream work

- Status: active
- Parent: EXT-G001
- Priority: P0
- Track: benchmark
- Evidence: HSSL_GATED_DOWNSTREAM_MISSING
- Outputs: src/gated_downstream.py
- Validation: true

## HSSL-G231 Local implementation readiness

- Status: active
- Priority: P1
- Track: benchmark
- Evidence: HSSL_LOCAL_READINESS_MISSING
- Outputs: src/local_readiness.py
- Validation: true
""",
        encoding="utf-8",
    )
    _git(repo, "add", "objective.md")
    _git(repo, "commit", "-m", "seed recorded gate and local readiness")
    bundle_dir = repo / "bundles"
    args = build_arg_parser().parse_args(
        [
            "--repo-root",
            str(repo),
            "--objective-path",
            str(objective_path),
            "--todo-path",
            str(todo_path),
            "--discovery-dir",
            str(repo / "discovery"),
            "--bundle-dir",
            str(bundle_dir),
            "--dataset-dir",
            str(repo / "datasets"),
            "--task-prefix",
            "SAFE-",
            "--max-findings",
            "10",
            "--no-reconcile-goal-completion",
            "--no-persist-ast-dataset",
        ]
    )

    payload = run_objective_daemon(args)

    assert payload["objective_completion_reconciliation_enabled"] is False
    assert (
        payload["recorded_external_completion_trusted_for_generation"]
        is False
    )
    index = json.loads((bundle_dir / "index.json").read_text(encoding="utf-8"))
    generated_goal_ids = {
        str(task.get("goal_id") or "")
        for bundle in index["bundles"].values()
        for task in bundle["tasks"]
    }
    assert generated_goal_ids == {"HSSL-G231"}
    todo_text = todo_path.read_text(encoding="utf-8")
    assert "HSSL_GATED_DOWNSTREAM_MISSING" not in todo_text
    assert "HSSL_LOCAL_READINESS_MISSING" in todo_text


def test_external_completion_rejects_dirty_stale_and_incomplete_artifacts(
    tmp_path,
):
    repo, objective_path, _todo_path = _seed_repo(tmp_path)
    valid = _authority(repo, objective_path)
    (repo / "untracked-change.py").write_text(
        "DIRTY = True\n",
        encoding="utf-8",
    )
    dirty = evaluate_external_completion_authority(
        valid,
        repo_root=repo,
        objective_path=objective_path,
        goal_evidence_terms={"EXT-G001": [EVIDENCE_TERM]},
        now=OBSERVED_AT,
    )
    assert dirty.evidence_records["EXT-G001"] == ()
    assert "current_source_dirty" in dirty.results[0].reason_codes

    (repo / "untracked-change.py").unlink()
    original_objective = objective_path.read_text(encoding="utf-8")
    objective_path.write_text(
        original_objective + "\n<!-- uncommitted supervisor state -->\n",
        encoding="utf-8",
    )
    dirty_objective = evaluate_external_completion_authority(
        valid,
        repo_root=repo,
        objective_path=objective_path,
        goal_evidence_terms={"EXT-G001": [EVIDENCE_TERM]},
        now=OBSERVED_AT,
    )
    assert dirty_objective.evidence_records["EXT-G001"] == ()
    assert "current_source_dirty" in (
        dirty_objective.results[0].reason_codes
    )

    objective_path.write_text(original_objective, encoding="utf-8")
    (repo / "src" / "implementation.py").write_text(
        "IMPLEMENTATION_MARKER = 'new source revision'\n",
        encoding="utf-8",
    )
    _git(repo, "add", "src/implementation.py")
    _git(repo, "commit", "-m", "advance synthetic source")
    stale_source = evaluate_external_completion_authority(
        valid,
        repo_root=repo,
        objective_path=objective_path,
        goal_evidence_terms={"EXT-G001": [EVIDENCE_TERM]},
        now=OBSERVED_AT,
    )
    stale_codes = set(stale_source.results[0].reason_codes)
    assert "external_source_commit_mismatch" in stale_codes
    assert "external_source_tree_mismatch" in stale_codes

    repo2, objective2, _todo2 = _seed_repo(tmp_path / "second")
    source2 = inspect_external_source(
        repo2,
        objective_path=objective2,
    ).identity
    missing_artifact = ExternalArtifactIdentity(
        artifact_id=_cid("runtime-receipt-slot"),
        artifact_cid=_cid("runtime-receipt"),
    )
    incomplete = _authority(
        repo2,
        objective2,
        artifacts=(missing_artifact,),
    )
    incomplete_result = evaluate_external_completion_authority(
        incomplete,
        repo_root=repo2,
        objective_path=objective2,
        goal_evidence_terms={"EXT-G001": [EVIDENCE_TERM]},
        now=OBSERVED_AT,
    )
    assert incomplete_result.evidence_records["EXT-G001"] == ()
    assert "external_artifacts_missing" in (
        incomplete_result.results[0].reason_codes
    )
    assert source2.clean is True


def test_dirty_objective_heap_reopens_externally_verified_goal(tmp_path):
    repo, objective_path, todo_path = _seed_repo(tmp_path)
    authority = _authority(repo, objective_path)
    gate = _completion_gate(repo, objective_path)
    provisional = reconcile_objective_goal_completion(
        repo_root=repo,
        objective_path=objective_path,
        todo_path=todo_path,
        completion_gate_records={"EXT-G001": gate},
        external_completion_authority=authority,
        now=OBSERVED_AT,
    )
    assert provisional.provisional_goal_ids == ["EXT-G001"]

    _git(repo, "add", "objective.md")
    _git(repo, "commit", "-m", "record provisional objective state")
    authority = _authority(repo, objective_path)
    gate = _completion_gate(repo, objective_path)
    completed = reconcile_objective_goal_completion(
        repo_root=repo,
        objective_path=objective_path,
        todo_path=todo_path,
        completion_gate_records={"EXT-G001": gate},
        external_completion_authority=authority,
        now=OBSERVED_AT,
    )
    assert completed.verified_goal_ids == ["EXT-G001"]

    _git(repo, "add", "objective.md")
    _git(repo, "commit", "-m", "record verified objective state")
    current_authority = _authority(repo, objective_path)
    current_gate = _completion_gate(repo, objective_path)
    objective_path.write_text(
        objective_path.read_text(encoding="utf-8")
        + "\n<!-- uncommitted objective mutation -->\n",
        encoding="utf-8",
    )

    reopened = reconcile_objective_goal_completion(
        repo_root=repo,
        objective_path=objective_path,
        todo_path=todo_path,
        completion_gate_records={"EXT-G001": current_gate},
        external_completion_authority=current_authority,
        now=OBSERVED_AT,
    )

    assert reopened.reopened_goal_ids == ["EXT-G001"]
    assert "current_source_dirty" in reopened.external_completion[
        "results"
    ][0]["reason_codes"]


def test_omitting_explicit_authority_cannot_reuse_persisted_external_receipt(
    tmp_path,
):
    repo, objective_path, todo_path = _seed_repo(tmp_path)
    authority = _authority(repo, objective_path)
    provisional = reconcile_objective_goal_completion(
        repo_root=repo,
        objective_path=objective_path,
        todo_path=todo_path,
        completion_gate_records={
            "EXT-G001": _completion_gate(repo, objective_path)
        },
        external_completion_authority=authority,
        now=OBSERVED_AT,
    )
    assert provisional.provisional_goal_ids == ["EXT-G001"]

    _git(repo, "add", "objective.md")
    _git(repo, "commit", "-m", "record provisional objective state")
    authority = _authority(repo, objective_path)
    completed = reconcile_objective_goal_completion(
        repo_root=repo,
        objective_path=objective_path,
        todo_path=todo_path,
        completion_gate_records={
            "EXT-G001": _completion_gate(repo, objective_path)
        },
        external_completion_authority=authority,
        now=OBSERVED_AT,
    )
    assert completed.verified_goal_ids == ["EXT-G001"]

    _git(repo, "add", "objective.md")
    _git(repo, "commit", "-m", "record verified objective state")
    reopened = reconcile_objective_goal_completion(
        repo_root=repo,
        objective_path=objective_path,
        todo_path=todo_path,
        completion_gate_records={
            "EXT-G001": _completion_gate(repo, objective_path)
        },
        now=OBSERVED_AT,
    )

    assert reopened.reopened_goal_ids == ["EXT-G001"]
    external_result = reopened.decisions["EXT-G001"][
        "external_completion"
    ]["results"][0]
    assert external_result["reason_codes"] == [
        "external_authority_not_supplied"
    ]
    reopened_goal = parse_goal_heap(
        objective_path.read_text(encoding="utf-8")
    )[0]
    assert json.loads(
        reopened_goal.fields["completion_evidence_records"]
    ) == []


def test_external_source_is_rechecked_after_legacy_migration_rewrite(
    tmp_path,
):
    repo, objective_path, todo_path = _seed_repo(tmp_path)
    objective_path.write_text(
        objective_path.read_text(encoding="utf-8").replace(
            "- Status: active",
            "- Status: completed",
        ),
        encoding="utf-8",
    )
    _git(repo, "add", "objective.md")
    _git(repo, "commit", "-m", "seed legacy completed objective")
    authority = _authority(repo, objective_path)

    result = reconcile_objective_goal_completion(
        repo_root=repo,
        objective_path=objective_path,
        todo_path=todo_path,
        completion_gate_records={
            "EXT-G001": _completion_gate(repo, objective_path)
        },
        external_completion_authority=authority,
        now=OBSERVED_AT,
    )

    assert result.verified_goal_ids == []
    assert "current_source_dirty" in result.external_completion[
        "results"
    ][0]["reason_codes"]
    assert result.state_counts["verified_complete"] == 0


def test_external_reconciliation_never_reads_configured_excluded_bytes(
    tmp_path,
    monkeypatch,
):
    repo, objective_path, todo_path = _seed_repo(tmp_path)
    authority = _authority(repo, objective_path)
    excluded_root = repo / "private-corpus"
    excluded_root.mkdir()
    excluded_source = excluded_root / "opaque-input.bin"
    excluded_source.write_bytes(b"synthetic private bytes")
    excluded_resolved = excluded_source.resolve()
    original_open = Path.open
    original_read_bytes = Path.read_bytes

    def guarded_open(path, *args, **kwargs):
        assert path.resolve() != excluded_resolved
        return original_open(path, *args, **kwargs)

    def guarded_read_bytes(path, *args, **kwargs):
        assert path.resolve() != excluded_resolved
        return original_read_bytes(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)

    result = reconcile_objective_goal_completion(
        repo_root=repo,
        objective_path=objective_path,
        todo_path=todo_path,
        external_completion_authority=authority,
        scan_exclude_paths=["private-corpus"],
        now=OBSERVED_AT,
    )

    assert result.verified_goal_ids == []
    assert "current_source_dirty" in result.external_completion[
        "results"
    ][0]["reason_codes"]


def test_external_completion_schema_rejects_duplicates_mismatches_and_disclosure(
    tmp_path,
):
    repo, objective_path, _todo_path = _seed_repo(tmp_path)
    valid = _authority(repo, objective_path)
    receipt = valid.receipts[0]

    duplicate_artifact_payload = receipt.to_dict()
    duplicate_artifact_payload["artifacts"] = [
        receipt.artifacts[0].to_dict(),
        receipt.artifacts[0].to_dict(),
    ]
    duplicate_artifact_payload.pop("receipt_cid")
    with pytest.raises(ValueError, match="duplicate artifact_id"):
        ExternalOperationalCompletionReceipt.from_dict(
            duplicate_artifact_payload
        )

    disclosure_payload = receipt.to_dict()
    disclosure_payload["artifact_path"] = "/private/corpus/result.json"
    with pytest.raises(ValueError, match="unsupported fields") as disclosure:
        ExternalOperationalCompletionReceipt.from_dict(disclosure_payload)
    assert "artifact_path" not in str(disclosure.value)
    assert "/private/corpus" not in str(disclosure.value)

    status_disclosure_payload = receipt.to_dict()
    status_disclosure_payload["status"] = "/private/corpus/result.json"
    status_disclosure_payload.pop("receipt_cid")
    with pytest.raises(ValueError, match="unsupported external receipt status"):
        ExternalOperationalCompletionReceipt.from_dict(
            status_disclosure_payload
        )

    unexpected_binding_receipt = ExternalOperationalCompletionReceipt(
        **{
            **{
                key: value
                for key, value in receipt.__dict__.items()
                if key not in {"receipt_cid", "goal_id"}
            },
            "goal_id": "EXT-UNEXPECTED",
        }
    )
    with pytest.raises(ValueError, match="no matching requirement"):
        ExternalCompletionAuthority(
            requirements=valid.requirements,
            receipts=(unexpected_binding_receipt,),
        )

    marker_path = tmp_path / "marker-only.json"
    marker_path.write_text(
        json.dumps(
            {
                "schema": valid.to_dict()["schema"],
                "requirements": [valid.requirements[0].to_dict()],
                "receipts": [EVIDENCE_TERM],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="receipt entries must be objects"):
        load_external_completion_authority(marker_path)

    mismatched = _authority(
        repo,
        objective_path,
        run_plan_cid=_cid("wrong-run-plan"),
        parent_ledger_cid=_cid("wrong-parent-ledger"),
    )
    mismatch_result = evaluate_external_completion_authority(
        mismatched,
        repo_root=repo,
        objective_path=objective_path,
        goal_evidence_terms={"EXT-G001": [EVIDENCE_TERM]},
        now=OBSERVED_AT,
    )
    assert set(mismatch_result.results[0].reason_codes) >= {
        "external_run_plan_mismatch",
        "external_parent_ledger_mismatch",
    }

    stale_receipt = ExternalOperationalCompletionReceipt(
        **{
            **{
                key: value
                for key, value in receipt.__dict__.items()
                if key not in {"receipt_cid", "observed_at", "fresh_until"}
            },
            "observed_at": "2026-07-25T10:00:00+00:00",
            "fresh_until": "2026-07-25T13:00:00+00:00",
        }
    )
    stale_authority = ExternalCompletionAuthority(
        requirements=valid.requirements,
        receipts=(stale_receipt,),
    )
    stale_result = evaluate_external_completion_authority(
        stale_authority,
        repo_root=repo,
        objective_path=objective_path,
        goal_evidence_terms={"EXT-G001": [EVIDENCE_TERM]},
        now=OBSERVED_AT,
    )
    assert "external_receipt_stale" in stale_result.results[0].reason_codes

    self_validated_receipt = ExternalOperationalCompletionReceipt(
        **{
            **{
                key: value
                for key, value in receipt.__dict__.items()
                if key not in {"receipt_cid", "validator_id"}
            },
            "validator_id": receipt.producer_id,
        }
    )
    self_validated = ExternalCompletionAuthority(
        requirements=valid.requirements,
        receipts=(self_validated_receipt,),
    )
    self_validated_result = evaluate_external_completion_authority(
        self_validated,
        repo_root=repo,
        objective_path=objective_path,
        goal_evidence_terms={"EXT-G001": [EVIDENCE_TERM]},
        now=OBSERVED_AT,
    )
    assert set(self_validated_result.results[0].reason_codes) >= {
        "external_validator_mismatch",
        "external_validator_not_independent",
    }

    duplicate_receipt = ExternalCompletionAuthority(
        requirements=valid.requirements,
        receipts=(
            valid.receipts[0],
            ExternalOperationalCompletionReceipt(
                **{
                    **{
                        key: value
                        for key, value in valid.receipts[0].__dict__.items()
                        if key != "receipt_cid"
                    },
                    "validator_receipt_cid": _cid(
                        "second-validator-receipt"
                    ),
                }
            ),
        ),
    )
    duplicate_result = evaluate_external_completion_authority(
        duplicate_receipt,
        repo_root=repo,
        objective_path=objective_path,
        goal_evidence_terms={"EXT-G001": [EVIDENCE_TERM]},
        now=datetime(2026, 7, 25, 12, 0, tzinfo=timezone.utc),
    )
    assert duplicate_result.evidence_records["EXT-G001"] == ()
    assert duplicate_result.results[0].reason_codes == (
        "external_receipt_duplicate",
    )


def test_external_source_identity_covers_recursive_gitlinks_without_paths(
    tmp_path,
):
    nested = tmp_path / "nested"
    nested.mkdir()
    _git(nested, "init")
    _git(nested, "checkout", "-b", "main")
    _git(nested, "config", "user.name", "Test User")
    _git(nested, "config", "user.email", "test@example.invalid")
    (nested / "nested.py").write_text("NESTED = True\n", encoding="utf-8")
    _git(nested, "add", ".")
    _git(nested, "commit", "-m", "seed nested")

    child = tmp_path / "child"
    child.mkdir()
    _git(child, "init")
    _git(child, "checkout", "-b", "main")
    _git(child, "config", "user.name", "Test User")
    _git(child, "config", "user.email", "test@example.invalid")
    (child / "child.py").write_text("CHILD = True\n", encoding="utf-8")
    _git(child, "add", ".")
    _git(child, "commit", "-m", "seed child")
    _git(
        child,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(nested),
        "nested-component",
    )
    _git(child, "commit", "-am", "add nested gitlink")

    repo, objective_path, _todo_path = _seed_repo(tmp_path / "outer")
    _git(
        repo,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(child),
        "child-component",
    )
    _git(repo, "commit", "-am", "add child gitlink")
    _git(
        repo,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "update",
        "--init",
        "--recursive",
    )

    inspection = inspect_external_source(
        repo,
        objective_path=objective_path,
    )
    assert inspection.valid is True
    assert sorted(
        item.depth for item in inspection.identity.recursive_gitlinks
    ) == [
        0,
        1,
    ]
    serialized = json.dumps(inspection.identity.to_dict(), sort_keys=True)
    assert "child-component" not in serialized
    assert "nested-component" not in serialized

    child_checkout = repo / "child-component"
    (child_checkout / "child.py").write_text(
        "CHILD = False\n",
        encoding="utf-8",
    )
    dirty = inspect_external_source(repo, objective_path=objective_path)
    assert dirty.valid is False
    assert "gitlink_checkout_dirty" in dirty.reason_codes

    _git(child_checkout, "restore", "child.py")
    nested_checkout = child_checkout / "nested-component"
    (nested_checkout / "nested.py").write_text(
        "NESTED = False\n",
        encoding="utf-8",
    )
    _git(nested_checkout, "add", "nested.py")
    _git(
        nested_checkout,
        "-c",
        "user.name=Test User",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "advance nested checkout only",
    )
    mismatched = inspect_external_source(
        repo,
        objective_path=objective_path,
    )
    assert mismatched.valid is False
    assert "gitlink_head_mismatch" in mismatched.reason_codes


def test_objective_daemon_loads_explicit_external_authority_without_path_disclosure(
    tmp_path,
):
    repo, objective_path, todo_path = _seed_repo(tmp_path)
    authority = _authority(repo, objective_path, include_receipt=False)
    authority_path = tmp_path / "external-authority.json"
    authority_path.write_text(
        json.dumps(authority.to_dict(), sort_keys=True),
        encoding="utf-8",
    )
    args = build_arg_parser().parse_args(
        [
            "--repo-root",
            str(repo),
            "--objective-path",
            str(objective_path),
            "--todo-path",
            str(todo_path),
            "--objective-external-completion-receipt-path",
            str(authority_path),
            "--max-findings",
            "0",
            "--no-persist-ast-dataset",
            "--no-generate-bounded-work",
        ]
    )

    payload = run_objective_daemon(args)

    assert (
        payload["objective_external_completion_authority_cid"]
        == authority.authority_cid
    )
    assert payload["objective_external_completion_governed_goal_ids"] == [
        "EXT-G001"
    ]
    assert payload["objective_external_completion"]["results"][0][
        "reason_codes"
    ] == ["external_receipt_missing"]
    assert "objective_external_completion_receipt_path" not in payload
    assert str(authority_path) not in json.dumps(payload, sort_keys=True)
