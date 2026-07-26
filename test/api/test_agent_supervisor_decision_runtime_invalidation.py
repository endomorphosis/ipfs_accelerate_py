from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.decision_runtime import (
    INCREMENTAL_REVALIDATION_REQUIREMENT_ID,
    DecisionRuntime,
    DecisionRuntimeConfig,
    DecisionRuntimeDenied,
    RuntimeInvalidationReceipt,
    canonical_dependency_change,
)
from ipfs_accelerate_py.agent_supervisor.event_log import (
    SemanticChange,
    SemanticChangeIntegrityError,
    SemanticChangeKind,
    append_semantic_change_event,
    initial_event_cursor,
    latest_event_cursor,
    read_semantic_change_page,
)
from ipfs_accelerate_py.agent_supervisor.proof_scope_index import (
    CrossDomainArtifact,
    CrossDomainArtifactKind,
    ProofInputKind,
    ProofScopeIndex,
    ProofScopeKey,
)
from ipfs_accelerate_py.agent_supervisor.runtime_cas import RuntimeCAS
from ipfs_accelerate_py.agent_supervisor.supervisor_recovery import (
    RecoveryDisposition,
    RecoveryFault,
    SupervisorRecovery,
)


def _artifact(
    artifact_id: str,
    kind: CrossDomainArtifactKind,
    *,
    root_id: str,
    scope_key: ProofScopeKey | None = None,
    dependency: CrossDomainArtifact | None = None,
    payload: dict[str, str] | None = None,
) -> CrossDomainArtifact:
    return CrossDomainArtifact(
        artifact_id=artifact_id,
        kind=kind,
        root_id=root_id,
        canonical_id=f"canonical:{artifact_id}",
        scope_keys=(scope_key,) if scope_key is not None else (),
        dependency_ids=(
            (dependency.artifact_id,) if dependency is not None else ()
        ),
        dependency_versions=(
            ((dependency.artifact_id, dependency.version_id),)
            if dependency is not None
            else ()
        ),
        payload=payload or {},
    )


def _proof_index() -> tuple[ProofScopeIndex, tuple[str, ...]]:
    root_id = "proof-root:v1"
    changed = ProofScopeKey(ProofInputKind.PROGRAM_SNAPSHOT, "program")
    other = ProofScopeKey(ProofInputKind.PROGRAM_SNAPSHOT, "independent")
    retrieval = _artifact(
        "retrieval:changed",
        CrossDomainArtifactKind.CACHE,
        root_id=root_id,
        scope_key=changed,
        payload={"cache_kind": "retrieval"},
    )
    context = _artifact(
        "context:changed",
        CrossDomainArtifactKind.CONTEXT,
        root_id=root_id,
        dependency=retrieval,
    )
    plan = _artifact(
        "plan:changed",
        CrossDomainArtifactKind.PLAN,
        root_id=root_id,
        dependency=context,
    )
    proof = _artifact(
        "proof:changed",
        CrossDomainArtifactKind.PROOF,
        root_id=root_id,
        dependency=plan,
    )
    permit = _artifact(
        "permit:changed",
        CrossDomainArtifactKind.PERMIT,
        root_id=root_id,
        dependency=proof,
    )
    validation = _artifact(
        "validation:changed",
        CrossDomainArtifactKind.VALIDATION,
        root_id=root_id,
        dependency=permit,
    )
    cache = _artifact(
        "cache:changed",
        CrossDomainArtifactKind.CACHE,
        root_id=root_id,
        dependency=validation,
        payload={"cache_kind": "validation"},
    )
    merge = _artifact(
        "merge:changed",
        CrossDomainArtifactKind.MERGE,
        root_id=root_id,
        dependency=cache,
        payload={"receipt_kind": "merge"},
    )
    completion = _artifact(
        "completion:changed",
        CrossDomainArtifactKind.MERGE,
        root_id=root_id,
        dependency=merge,
        payload={"receipt_kind": "completion"},
    )
    independent = _artifact(
        "cache:independent",
        CrossDomainArtifactKind.CACHE,
        root_id=root_id,
        scope_key=other,
        payload={"cache_kind": "retrieval"},
    )
    affected = (
        retrieval.artifact_id,
        context.artifact_id,
        plan.artifact_id,
        proof.artifact_id,
        permit.artifact_id,
        validation.artifact_id,
        cache.artifact_id,
        merge.artifact_id,
        completion.artifact_id,
    )
    return (
        ProofScopeIndex(
            blobs=(),
            obligations=(),
            receipts=(),
            root_id=root_id,
            artifacts=(
                retrieval,
                context,
                plan,
                proof,
                permit,
                validation,
                cache,
                merge,
                completion,
                independent,
            ),
        ),
        affected,
    )


def _change(previous: str = "program:v1", current: str = "program:v2") -> SemanticChange:
    return canonical_dependency_change(
        SemanticChangeKind.WORKTREE,
        subject_id="program",
        previous_root_id=previous,
        current_root_id=current,
        scope_kind=ProofInputKind.PROGRAM_SNAPSHOT.value,
        scope_value="program",
    )


def _recomputers() -> dict[str, object]:
    return {
        family: (
            lambda identities, _change_value, _index, name=family: {
                "artifact_id": f"recomputed:{name}:{','.join(identities)}"
            }
        )
        for family in ("context", "plan", "proof", "validation")
    }


def test_change_vocabulary_is_closed_and_content_addressed() -> None:
    assert INCREMENTAL_REVALIDATION_REQUIREMENT_ID
    assert {item.value for item in SemanticChangeKind} == {
        "worktree",
        "ast",
        "effect",
        "intent_ir",
        "legal_ir",
        "security_ir",
        "policy",
        "tool_catalog",
        "capability",
        "proof",
        "monitor",
        "lease",
        "observed_effect",
    }
    for kind in SemanticChangeKind:
        value = canonical_dependency_change(
            kind,
            subject_id=f"subject:{kind.value}",
            previous_root_id=f"{kind.value}:v1",
            current_root_id=f"{kind.value}:v2",
        )
        restored = SemanticChange.from_dict(value.to_dict())
        assert restored == value
        assert restored.change_id == value.change_id


def test_reverse_scope_invalidates_every_and_only_dependent_artifact() -> None:
    index, affected = _proof_index()
    runtime = DecisionRuntime(DecisionRuntimeConfig(mode="off"))

    result = runtime.apply_dependency_change(
        _change(),
        index,
        recompute=_recomputers(),
    )
    receipt = result.receipt

    assert set(result.proof_index.invalidated_artifact_ids) == set(affected)
    assert result.proof_index.active_artifact_ids == ("cache:independent",)
    assert receipt.retrieval_ids == ("retrieval:changed",)
    assert receipt.context_ids == ("context:changed",)
    assert receipt.plan_suffix_ids == ("plan:changed",)
    assert receipt.permit_ids == ("permit:changed",)
    assert receipt.proof_ids == ("proof:changed",)
    assert receipt.validation_ids == ("validation:changed",)
    assert receipt.cache_ids == ("cache:changed",)
    assert receipt.merge_receipt_ids == ("merge:changed",)
    assert receipt.completion_receipt_ids == ("completion:changed",)
    assert receipt.preserved_artifact_ids == ("cache:independent",)
    assert receipt.authoritative
    assert not receipt.reason_codes
    assert len(receipt.recomputed_artifact_ids) == 4
    assert (
        RuntimeInvalidationReceipt.from_dict(receipt.to_dict()).receipt_id
        == receipt.receipt_id
    )

    with pytest.raises(DecisionRuntimeDenied) as caught:
        runtime.apply_dependency_change(
            _change(),
            result.proof_index,
            recompute=_recomputers(),
        )
    assert "duplicate_semantic_change" in caught.value.reason_codes


def test_event_replay_rejects_duplicate_and_missing_root_transitions(
    tmp_path: Path,
) -> None:
    duplicate_log = tmp_path / "duplicate.jsonl"
    duplicate_cursor = initial_event_cursor(duplicate_log)
    change = _change()
    append_semantic_change_event(duplicate_log, change)
    append_semantic_change_event(duplicate_log, change)

    with pytest.raises(SemanticChangeIntegrityError, match="duplicate"):
        read_semantic_change_page(
            duplicate_log,
            duplicate_cursor,
            expected_roots={"program": "program:v1"},
        )

    reordered_log = tmp_path / "reordered.jsonl"
    reordered_cursor = initial_event_cursor(reordered_log)
    append_semantic_change_event(reordered_log, change)
    append_semantic_change_event(
        reordered_log,
        _change(previous="program:v1", current="program:v3"),
    )
    with pytest.raises(SemanticChangeIntegrityError, match="missing or reordered"):
        read_semantic_change_page(
            reordered_log,
            reordered_cursor,
            expected_roots={"program": "program:v1"},
        )


def test_replay_binds_physical_cursor_and_preserves_independent_branch(
    tmp_path: Path,
) -> None:
    event_log = tmp_path / "events.jsonl"
    cursor = initial_event_cursor(event_log)
    append_semantic_change_event(event_log, _change())
    index, _affected = _proof_index()
    runtime = DecisionRuntime(DecisionRuntimeConfig(mode="off"))

    changed, receipts, replay_cursor = runtime.replay_dependency_events(
        event_log,
        cursor,
        index,
        recompute=_recomputers(),
    )

    assert replay_cursor == latest_event_cursor(event_log)
    assert receipts[0].event_cursor == replay_cursor
    assert receipts[0].current_roots["program"] == "program:v2"
    assert changed.active_artifact_ids == ("cache:independent",)


def test_corrupt_cas_invalidation_journal_enters_fail_closed_quarantine(
    tmp_path: Path,
) -> None:
    RuntimeCAS(tmp_path)
    journal = tmp_path / "invalidation-transactions" / ("0" * 64 + ".json")
    journal.write_bytes(b'{"partial":')

    restored = RuntimeCAS(tmp_path)
    audit = restored.audit()

    assert restored.quarantined
    assert not audit.healthy
    assert "corrupt_invalidation_journal" in audit.issue_codes
    assert not journal.exists()
    assert list((tmp_path / "quarantine").iterdir())


def test_recovery_replays_same_roots_and_cursor_and_fences_old_permits(
    tmp_path: Path,
) -> None:
    event_log = tmp_path / "events.jsonl"
    append_semantic_change_event(event_log, _change())
    checkpoint_cursor = latest_event_cursor(event_log)
    recovery = SupervisorRecovery(tmp_path / "recovery")
    recovery.checkpoint(
        repository_id="repository:fixture",
        tree_id="tree:fixture",
        generation=1,
        state={"phase": "proof"},
        cursor=checkpoint_cursor,
        semantic_roots={"program": "program:v2"},
        proof_index_id="proof-index:v2",
        cas_invalidation_id="cas-head:v2",
        fencing_epoch=4,
    )
    append_semantic_change_event(
        event_log,
        _change(previous="program:v2", current="program:v3"),
    )
    head = latest_event_cursor(event_log)

    receipt = recovery.recover(
        incident_id="crash-replay",
        fault=RecoveryFault.PROCESS_CRASH,
        repository_id="repository:fixture",
        tree_id="tree:fixture",
        event_log_path=event_log,
        current_semantic_roots={"program": "program:v3"},
        current_event_cursor=head,
        current_proof_index_id="proof-index:v3",
        current_cas_invalidation_id="cas-head:v3",
        current_fencing_token=5,
        observed_fencing_token=4,
        precrash_permit_ids=("permit:old",),
        fence_permits=lambda permit_ids, epoch: (
            permit_ids == ("permit:old",) and epoch == 5
        ),
        replay_events=lambda _checkpoint: {
            "event_cursor": head,
            "semantic_roots": {"program": "program:v3"},
            "proof_index_id": "proof-index:v3",
            "cas_invalidation_id": "cas-head:v3",
            "invalidated_permit_ids": ("permit:old",),
        },
    )

    assert receipt.disposition is RecoveryDisposition.RECOVERED
    assert receipt.replay_cursor == head
    assert receipt.result_semantic_roots == {"program": "program:v3"}
    assert receipt.precrash_permit_ids == ("permit:old",)
    assert receipt.invalidated_permit_ids == ("permit:old",)
    assert receipt.fencing_epoch == 5
    assert receipt.proof_index_id == "proof-index:v3"
    assert receipt.cas_invalidation_id == "cas-head:v3"
    assert "fence_precrash_permits" in receipt.actions
    assert "replay_dependency_events" in receipt.actions
    assert (
        recovery.recover(
            incident_id="crash-replay",
            fault=RecoveryFault.PROCESS_CRASH,
            repository_id="repository:fixture",
            tree_id="tree:fixture",
            event_log_path=event_log,
            current_semantic_roots={"program": "program:v3"},
            current_event_cursor=head,
            current_proof_index_id="proof-index:v3",
            current_cas_invalidation_id="cas-head:v3",
            current_fencing_token=5,
            observed_fencing_token=4,
            precrash_permit_ids=("permit:old",),
        ).receipt_id
        == receipt.receipt_id
    )


def test_recovery_refuses_unreplayed_tail_and_root_race(tmp_path: Path) -> None:
    event_log = tmp_path / "events.jsonl"
    append_semantic_change_event(event_log, _change())
    recovery = SupervisorRecovery(tmp_path / "recovery")
    recovery.checkpoint(
        repository_id="repository:fixture",
        tree_id="tree:fixture",
        generation=1,
        state={"phase": "proof"},
        cursor=latest_event_cursor(event_log),
        semantic_roots={"program": "program:v2"},
    )
    append_semantic_change_event(
        event_log,
        _change(previous="program:v2", current="program:v3"),
    )

    missed = recovery.recover(
        incident_id="missed-tail",
        fault=RecoveryFault.PROCESS_CRASH,
        repository_id="repository:fixture",
        tree_id="tree:fixture",
        event_log_path=event_log,
    )
    assert missed.disposition is RecoveryDisposition.FAILED_CLOSED
    assert missed.reason_code == "unreplayed_events"

    roots = iter(
        (
            {"program": "program:v2"},
            {"program": "program:raced"},
        )
    )
    raced = recovery.recover(
        incident_id="root-race",
        fault=RecoveryFault.PROCESS_CRASH,
        repository_id="repository:fixture",
        tree_id="tree:fixture",
        current_semantic_roots={"program": "program:v2"},
        root_reader=lambda: next(roots),
    )
    assert raced.disposition is RecoveryDisposition.FAILED_CLOSED
    assert raced.reason_code == "root_race"
