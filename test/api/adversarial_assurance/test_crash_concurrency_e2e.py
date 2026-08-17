"""AAE-060: Qualify crash recovery, deterministic replay, cancellation, and concurrent stale writers.

Validates ``AAECrashConcurrencyQualification@1`` / ``aae/crash-e2e@1``:

* All ten required pipeline crash points restart safely.
* Completed immutable evidence survives process restart and recovery.
* Ambiguous or partial execution claims never become terminal success.
* CAS permits exactly one current writer; stale writers fail closed.
* Worktrees and process trees are ownership-fenced (cleanup / cancel).

Crash points are the plan §13 persistence/concurrency injection sites (exactly
ten). Each site is exercised with an injected interruption, reopen + recovery,
and an idempotent or clean restart that leaves durable evidence consistent.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

import pytest

from ipfs_accelerate_py.agent_supervisor.adversarial_assurance.workers import (
    MutationWorkerBudget,
    MutationWorkerCancellation,
    MutationWorkerCheckpointStore,
    MutationWorkerDisposition,
    MutationWorkerPool,
    MutationWorkerTask,
)
from ipfs_accelerate_py.agent_supervisor.adversarial_assurance.worktrees import (
    MutationWorktreeFenceError,
    MutationWorktreePhase,
    MutationWriteScope,
    create_mutation_worktree,
    recover_mutation_worktree,
)
from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
    CleanupDisposition,
    WorktreeLifecycleStore,
    normalize_workspace_path,
)
from ipfs_accelerate_py.agent_supervisor.runtime.resource_scheduler import (
    HostResourceSnapshot,
    ResourcePolicy,
    ResourceScheduler,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.core import pid_alive
from ipfs_datasets_py.logic.software_contracts.adversarial_assurance import (
    AssuranceTerminalStatus,
    HeldOutResult,
    SignatureVerificationStatus,
)
from ipfs_datasets_py.tests.unit.logic.software_contracts.adversarial_assurance import (
    test_receipt_contracts as receipt_fixtures,
)
from ipfs_kit_py.adversarial_assurance_store.artifacts import (
    DurableAssuranceArtifactStore,
    cid_for_assurance_artifact,
)
from ipfs_kit_py.adversarial_assurance_store.campaigns import (
    CampaignPhase,
    CampaignTransitionError,
    DurableMutationCampaignRepository,
    ExecutionClaimStatus,
    admit_campaign_receipt_payload,
    assert_terminal_success_admissible,
)
from ipfs_kit_py.adversarial_assurance_store.contracts import (
    AssuranceArtifactKind,
    AssuranceStoreStatus,
)
from ipfs_kit_py.adversarial_assurance_store.merkle import (
    DurableAssuranceCampaignMerkleRepository,
    MerkleSetKind,
    build_merkle_set_commitment,
    cid_for_merkle_set,
)
from ipfs_kit_py.adversarial_assurance_store.policy import (
    DurableAssurancePolicyRepository,
)
from ipfs_kit_py.adversarial_assurance_store.recovery import (
    REQUIRED_CAS_INTERRUPTION_POINTS,
    AssuranceRecoveryAdmissionError,
    assert_terminal_claim_not_ambiguous,
    assert_writer_fence,
    recover_assurance_campaigns,
)
from ipfs_kit_py.mcp_server.mcplusplus.coordination_storage import (
    DurableCoordinationStore,
    cid_for_artifact,
    cid_for_bytes,
)

# ---------------------------------------------------------------------------
# Qualification constants
# ---------------------------------------------------------------------------

INTERFACE = "AAECrashConcurrencyQualification@1"
EVIDENCE = "aae/crash-e2e@1"
TASK_ID = "AAE-060"
BUNDLE = "adversarial-assurance/crash-concurrency-e2e"

# Plan §13: exactly ten pipeline crash injection sites.
REQUIRED_PIPELINE_CRASH_POINTS: tuple[str, ...] = (
    "after_mutant_creation",
    "during_worktree_setup",
    "during_test_execution",
    "during_proof_execution",
    "after_receipt_persistence",
    "during_diagnosis",
    "during_evaluation",
    "during_root_update",
    "before_policy_cas",
    "after_cas_before_cleanup",
)

WORKSPACE = "aae060-worker"
CAMPAIGN_ID = "camp-aae060"

# CAS boundaries that complete *before* the durable head advances.
_PRE_DURABLE_CAS = frozenset(
    {"before_transaction", "after_expectation_verification"}
)

MOD_SOURCE = textwrap.dedent(
    '''\
    """Sample module for AAE-060 crash concurrency e2e."""


    def fn(flag: bool) -> int:
        if flag:
            return 1
        return 0
    '''
)

MOD_MUTATED = textwrap.dedent(
    '''\
    """Sample module for AAE-060 crash concurrency e2e."""


    def fn(flag: bool) -> int:
        if not flag:
            return 1
        return 0
    '''
)


class InjectedCrash(RuntimeError):
    """Stand-in for a process stopping at a declared pipeline crash point."""


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _block(store: DurableCoordinationStore, name: str, **extra: Any) -> str:
    payload = {"schema": "example/aae060@1", "name": name}
    payload.update(extra)
    return store.put(payload, expected_cid=cid_for_artifact(payload), replicate=False)[
        "cid"
    ]


def _plan_cid() -> str:
    return cid_for_bytes(b"aae060-campaign-plan")


def _policy_plan_cid() -> str:
    return cid_for_bytes(b"aae060-campaign-policy")


def _builder_state(
    *,
    phase: CampaignPhase | str,
    execution_claim_status: ExecutionClaimStatus | str,
    receipt_cid: str | None = None,
    campaign_id: str = CAMPAIGN_ID,
    artifact_cids: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "campaign_id": campaign_id,
        "phase": phase if isinstance(phase, str) else phase.value,
        "execution_claim_status": (
            execution_claim_status
            if isinstance(execution_claim_status, str)
            else execution_claim_status.value
        ),
        "plan_cid": _plan_cid(),
        "policy_cid": _policy_plan_cid(),
        "receipt_cid": receipt_cid,
        "artifact_cids": list(artifact_cids or []),
    }


def _put_receipt(
    campaigns: DurableMutationCampaignRepository,
    *,
    op_suffix: str = "1",
) -> str:
    sealed = admit_campaign_receipt_payload(receipt_fixtures._campaign().to_dict())
    expected = cid_for_assurance_artifact(
        AssuranceArtifactKind.ASSURANCE_CAMPAIGN_RECEIPT, sealed
    )
    head = campaigns.current_receipts_history(WORKSPACE)
    result = campaigns.persist_campaign_receipt(
        WORKSPACE,
        sealed,
        expected_cid=expected,
        artifact_operation_id=f"receipt-art-{op_suffix}",
        history_operation_id=f"receipt-hist-{op_suffix}",
        expected_history_generation=head.generation,
        expected_history_head_cid=head.head_cid,
        replicate=False,
    )
    assert result.artifact.local_durable is True
    return expected


def _put_leaf(store: DurableCoordinationStore, label: str) -> str:
    payload = {"schema": "example/aae060-merkle-leaf@1", "label": label}
    return store.put(payload, expected_cid=cid_for_artifact(payload), replicate=False)[
        "cid"
    ]


def _commit_all_sets(
    merkle: DurableAssuranceCampaignMerkleRepository,
    coordination: DurableCoordinationStore,
) -> dict[str, str]:
    set_cids: dict[str, str] = {}
    for kind in MerkleSetKind:
        members = [
            _put_leaf(coordination, f"{kind.value}-a"),
            _put_leaf(coordination, f"{kind.value}-b"),
        ]
        sealed = build_merkle_set_commitment(
            workspace=WORKSPACE,
            campaign_id=CAMPAIGN_ID,
            set_kind=kind,
            member_cids=members,
            operation_id=f"set-op-{kind.value}",
        )
        expected = cid_for_merkle_set(sealed)
        result = merkle.commit_merkle_set(
            WORKSPACE,
            campaign_id=CAMPAIGN_ID,
            set_kind=kind,
            member_cids=members,
            expected_cid=expected,
            operation_id=f"set-op-{kind.value}",
        )
        assert result.local_durable is True
        set_cids[kind.value] = expected
    return set_cids


def _git(
    cwd: Path, *args: str, check: bool = True
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )
    if check and completed.returncode != 0:
        raise AssertionError(
            f"git {' '.join(args)} failed: {completed.stderr or completed.stdout}"
        )
    return completed


def _init_repo(root: Path) -> tuple[str, str]:
    root.mkdir(parents=True, exist_ok=True)
    _git(root, "init")
    _git(root, "config", "user.email", "aae060@example.com")
    _git(root, "config", "user.name", "AAE060")
    _git(root, "checkout", "-b", "main")
    (root / "mod.py").write_text(MOD_SOURCE, encoding="utf-8")
    (root / "pkg").mkdir()
    (root / "pkg" / "util.py").write_text("X = 1\n", encoding="utf-8")
    (root / "README.md").write_text("# fixture\n", encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-m", "baseline")
    head = _git(root, "rev-parse", "HEAD").stdout.strip()
    tree = _git(root, "rev-parse", "HEAD^{tree}").stdout.strip()
    return head, tree


def _lifecycle_store(repo: Path, tmp_path: Path) -> WorktreeLifecycleStore:
    return WorktreeLifecycleStore(
        repo_root=repo,
        store_dir=tmp_path / "lifecycle",
        lease_seconds=300.0,
    )


def _scope(**overrides: object) -> MutationWriteScope:
    payload: dict[str, object] = {
        "allowed_paths": ("mod.py", "pkg/"),
        "effect_paths": ("mod.py",),
        "task_owned_paths": ("mod.py", "pkg/"),
    }
    payload.update(overrides)
    return MutationWriteScope.from_dict(payload)


def _host(*, worker_limit: int = 4) -> HostResourceSnapshot:
    return HostResourceSnapshot(
        worker_limit=worker_limit,
        available_worker_capacity=worker_limit,
        active_workers=0,
        memory_available_bytes=8 * 1024 * 1024 * 1024,
        disk_available_bytes=8 * 1024 * 1024 * 1024,
        memory_total_bytes=16 * 1024 * 1024 * 1024,
        disk_total_bytes=64 * 1024 * 1024 * 1024,
        capabilities=("cpu",),
        resource_classes=(
            "cpu-large",
            "cpu-small",
            "cpu-medium",
            "cpu-validation",
            "cpu-proof-solver",
        ),
    )


def _pool(
    tmp_path: Path,
    *,
    max_concurrency: int = 2,
    default_timeout_seconds: float = 5.0,
) -> MutationWorkerPool:
    return MutationWorkerPool.create(
        max_concurrency=max_concurrency,
        default_timeout_seconds=default_timeout_seconds,
        checkpoint_dir=tmp_path / "worker-checkpoints",
        resource_scheduler=ResourceScheduler(
            ResourcePolicy(max_lanes=max_concurrency)
        ),
        host_snapshot=_host(worker_limit=max(max_concurrency, 4)),
    )


def _advance_to(
    campaigns: DurableMutationCampaignRepository,
    *,
    target: CampaignPhase,
    receipt_cid: str | None = None,
    claim: ExecutionClaimStatus = ExecutionClaimStatus.COMPLETE,
    op_prefix: str = "adv",
) -> str:
    """Drive campaign state from genesis to ``target`` (inclusive)."""

    sequence: list[tuple[CampaignPhase, ExecutionClaimStatus, str | None]] = [
        (CampaignPhase.PLANNED, ExecutionClaimStatus.NONE, None),
        (CampaignPhase.EXECUTING, ExecutionClaimStatus.COMPLETE, None),
    ]
    if target in (
        CampaignPhase.DIAGNOSING,
        CampaignPhase.EVALUATING,
        CampaignPhase.COMPLETE,
    ):
        if target is CampaignPhase.DIAGNOSING:
            sequence.append((CampaignPhase.DIAGNOSING, claim, None))
        else:
            sequence.append(
                (CampaignPhase.DIAGNOSING, ExecutionClaimStatus.COMPLETE, None)
            )
            if target is CampaignPhase.EVALUATING:
                sequence.append((CampaignPhase.EVALUATING, claim, None))
            else:
                sequence.append(
                    (CampaignPhase.EVALUATING, ExecutionClaimStatus.COMPLETE, None)
                )
                sequence.append(
                    (CampaignPhase.COMPLETE, ExecutionClaimStatus.COMPLETE, receipt_cid)
                )

    head = campaigns.current_campaign_state(WORKSPACE)
    gen = head.generation
    state_cid = head.state_cid
    for phase, claim_status, rcid in sequence:
        result = campaigns.transition_campaign_state(
            WORKSPACE,
            state=_builder_state(
                phase=phase,
                execution_claim_status=claim_status,
                receipt_cid=rcid,
            ),
            expected_generation=gen,
            expected_state_cid=state_cid,
            operation_id=f"{op_prefix}-{phase.value}",
        )
        assert result.status is AssuranceStoreStatus.UPDATED
        gen = gen + 1
        state_cid = result.state_cid
        assert state_cid is not None
    assert campaigns.current_campaign_state(WORKSPACE).phase is target
    assert state_cid is not None
    return state_cid


# ---------------------------------------------------------------------------
# Interface / inventory
# ---------------------------------------------------------------------------


def test_qualification_interface_and_ten_crash_points() -> None:
    assert INTERFACE == "AAECrashConcurrencyQualification@1"
    assert EVIDENCE == "aae/crash-e2e@1"
    assert TASK_ID == "AAE-060"
    assert BUNDLE == "adversarial-assurance/crash-concurrency-e2e"
    assert len(REQUIRED_PIPELINE_CRASH_POINTS) == 10
    assert len(set(REQUIRED_PIPELINE_CRASH_POINTS)) == 10
    # Durable CAS points remain available for recovery qualification.
    assert len(REQUIRED_CAS_INTERRUPTION_POINTS) == 6


def test_required_crash_points_match_plan_injection_sites() -> None:
    expected = {
        "after_mutant_creation",
        "during_worktree_setup",
        "during_test_execution",
        "during_proof_execution",
        "after_receipt_persistence",
        "during_diagnosis",
        "during_evaluation",
        "during_root_update",
        "before_policy_cas",
        "after_cas_before_cleanup",
    }
    assert set(REQUIRED_PIPELINE_CRASH_POINTS) == expected


# ---------------------------------------------------------------------------
# Per-crash-point restart harness
# ---------------------------------------------------------------------------


def _run_after_mutant_creation(tmp_path: Path) -> dict[str, Any]:
    root = tmp_path / "store-mutant"
    mutant_cid: str | None = None
    try:
        with DurableCoordinationStore(root) as store:
            mutant_cid = _block(store, "mutant-artifact", body=MOD_MUTATED)
            raise InjectedCrash("after_mutant_creation")
    except InjectedCrash:
        pass
    assert mutant_cid is not None
    with DurableCoordinationStore(root) as store:
        report = recover_assurance_campaigns(store)
        assert store.has(mutant_cid) is True
        # Restart continues: attach mutant as a planned campaign artifact.
        campaigns = DurableMutationCampaignRepository(store)
        planned = campaigns.transition_campaign_state(
            WORKSPACE,
            state=_builder_state(
                phase=CampaignPhase.PLANNED,
                execution_claim_status=ExecutionClaimStatus.NONE,
                artifact_cids=[mutant_cid],
            ),
            expected_generation=0,
            expected_state_cid=None,
            operation_id="mutant-plan",
        )
        assert planned.status is AssuranceStoreStatus.UPDATED
        # Deterministic replay of the same plan op is idempotent.
        replay = campaigns.transition_campaign_state(
            WORKSPACE,
            state=_builder_state(
                phase=CampaignPhase.PLANNED,
                execution_claim_status=ExecutionClaimStatus.NONE,
                artifact_cids=[mutant_cid],
            ),
            expected_generation=0,
            expected_state_cid=None,
            operation_id="mutant-plan",
        )
        assert replay.status is AssuranceStoreStatus.UNCHANGED
        assert replay.reason_code == "idempotent_replay"
        return {
            "crash_point": "after_mutant_creation",
            "mutant_cid": mutant_cid,
            "phase": CampaignPhase.PLANNED.value,
            "report_errors": list(report.errors),
            "restarted_safely": True,
        }


def _run_during_worktree_setup(tmp_path: Path) -> dict[str, Any]:
    repo = tmp_path / "repo-setup"
    head, tree = _init_repo(repo)
    store = _lifecycle_store(repo, tmp_path / "lc-setup")
    parent = tmp_path / "owned-wts-setup"
    parent.mkdir(parents=True, exist_ok=True)
    workspace = parent / "interrupted-setup"

    record = store.begin_preparing(
        task_id="AAE-060-setup",
        attempt=1,
        lane_id="lane-0",
        workspace_path=workspace,
        branch="aae-mutant/setup-a1",
        merge_target="HEAD",
    )
    digest = hashlib.sha256(
        normalize_workspace_path(workspace).encode("utf-8")
    ).hexdigest()[:16]
    journal_path = Path(store.store_dir) / f"aae-attempt-{digest}.json"  # type: ignore[arg-type]
    journal_path.write_text(
        json.dumps(
            {
                "schema": "ipfs_accelerate_py/agent-supervisor/adversarial-assurance-mutation-worktree-attempt@1",
                "phase": "preparing",
                "lease_id": record.lease_id,
                "fence": record.fence,
                "repo_root": str(repo),
                "worktree_path": str(workspace),
                "worktree_parent": str(parent),
                "base_commit": head,
                "base_tree": tree,
                "task_id": "AAE-060-setup",
                "attempt": 1,
            }
        ),
        encoding="utf-8",
    )
    # Simulate crash mid-setup: no worktree created, journal left preparing.
    try:
        raise InjectedCrash("during_worktree_setup")
    except InjectedCrash:
        pass

    recovery = recover_mutation_worktree(
        lifecycle_store=store,
        worktree_path=workspace,
        repo_root=repo,
        worktree_parent=parent,
        caller_lease_id=record.lease_id,
    )
    assert recovery["recovered"] is True
    assert "marked_terminal" in recovery["actions"]
    loaded = store.load_workspace(workspace)
    assert loaded is not None and loaded.is_terminal

    # Restart: create a fresh owned worktree for attempt 2.
    with create_mutation_worktree(
        repo_root=repo,
        worktree_path=parent / "setup-retry",
        worktree_parent=parent,
        base_commit=head,
        base_tree=tree,
        task_id="AAE-060-setup",
        attempt=2,
        lifecycle_store=store,
    ) as isolated:
        assert isolated.phase is MutationWorktreePhase.READY
        applied = isolated.apply_replacements(
            {"mod.py": MOD_MUTATED},
            _scope(),
            lease_id=isolated.lease_id,
            fence=isolated.fence,
        )
        assert applied.applied is True
        assert (repo / "mod.py").read_text(encoding="utf-8") == MOD_SOURCE
        lease = isolated.lease_id
        fence = isolated.fence

    return {
        "crash_point": "during_worktree_setup",
        "recovered": True,
        "restarted_safely": True,
        "production_untouched": True,
        "final_lease": lease,
        "final_fence": fence,
    }


def _run_worker_crash(
    tmp_path: Path,
    *,
    crash_point: str,
    task_id: str,
) -> dict[str, Any]:
    """Cancel an in-flight command worker (test or proof) and restart cleanly."""

    pool = _pool(tmp_path / f"pool-{crash_point}", default_timeout_seconds=10.0)
    cancel = MutationWorkerCancellation()
    ready = tmp_path / f"ready-{crash_point}.flag"
    script = tmp_path / f"slow-{crash_point}.py"
    script.write_text(
        "import pathlib, time\n"
        f"pathlib.Path({str(ready)!r}).write_text('1')\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    try:
        future = pool.submit(
            MutationWorkerTask(
                task_id=task_id,
                command=[sys.executable, str(script)],
                cwd=str(tmp_path),
                environment={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
            ),
            cancellation=cancel,
        )
        deadline = time.time() + 5.0
        while time.time() < deadline and not ready.exists():
            time.sleep(0.02)
        assert ready.exists(), "child never became ready"
        # Injected crash at execution boundary → cooperative cancel + fence.
        cancel.cancel(reason=f"crash:{crash_point}")
        result = future.result(timeout=10.0)
        assert result.disposition is MutationWorkerDisposition.CANCELLED
        assert result.publication_allowed is False
        assert result.payload is None
        assert result.infrastructure.process_tree_fenced is True
        assert result.infrastructure.cancelled is True
        pid = result.infrastructure.pid
        if pid is not None:
            deadline = time.time() + 2.0
            while time.time() < deadline and pid_alive(pid):
                time.sleep(0.05)
            assert not pid_alive(pid)

        # Restart: same semantic work under a new attempt succeeds.
        restart = pool.run(
            MutationWorkerTask(
                task_id=f"{task_id}-restart",
                command=[
                    sys.executable,
                    "-c",
                    "print('ok')",
                ],
                cwd=str(tmp_path),
                environment={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
            )
        )
        assert restart.disposition is MutationWorkerDisposition.COMPLETED
        assert restart.publication_allowed is True
        return {
            "crash_point": crash_point,
            "cancelled": True,
            "process_tree_fenced": True,
            "restarted_safely": True,
            "restart_disposition": restart.disposition.value,
        }
    finally:
        pool.shutdown(wait=True, cancel=True)


def _run_during_test_execution(tmp_path: Path) -> dict[str, Any]:
    return _run_worker_crash(
        tmp_path,
        crash_point="during_test_execution",
        task_id="aae060-test-exec",
    )


def _run_during_proof_execution(tmp_path: Path) -> dict[str, Any]:
    return _run_worker_crash(
        tmp_path,
        crash_point="during_proof_execution",
        task_id="aae060-proof-exec",
    )


def _run_after_receipt_persistence(tmp_path: Path) -> dict[str, Any]:
    root = tmp_path / "store-receipt"
    receipt_cid: str | None = None
    with DurableCoordinationStore(root) as store:
        campaigns = DurableMutationCampaignRepository(store)
        receipt_cid = _put_receipt(campaigns, op_suffix="persist")
        assert store.has(receipt_cid) is True
        try:
            raise InjectedCrash("after_receipt_persistence")
        except InjectedCrash:
            pass

    assert receipt_cid is not None
    with DurableCoordinationStore(root) as recovered:
        report = recover_assurance_campaigns(recovered)
        assert recovered.has(receipt_cid) is True
        artifacts = DurableAssuranceArtifactStore(recovered)
        try:
            verified = artifacts.get_verified_artifact(
                receipt_cid,
                expected_kind=AssuranceArtifactKind.ASSURANCE_CAMPAIGN_RECEIPT,
            )
            assert verified is not None
        finally:
            artifacts.close()
        # History head still projects the receipt after restart.
        campaigns = DurableMutationCampaignRepository(recovered)
        hist = campaigns.current_receipts_history(WORKSPACE)
        assert hist.generation >= 1
        assert hist.head_cid is not None
        return {
            "crash_point": "after_receipt_persistence",
            "receipt_cid": receipt_cid,
            "immutable_survived": True,
            "history_generation": hist.generation,
            "report_errors": list(report.errors),
            "restarted_safely": True,
        }


def _run_campaign_phase_crash(
    tmp_path: Path,
    *,
    crash_point: str,
    prior_phase: CampaignPhase,
    next_phase: CampaignPhase,
    cas_boundary: str = "after_sqlite_commit",
) -> dict[str, Any]:
    """Interrupt a campaign-state CAS between prior and next phase, then recover."""

    root = tmp_path / f"store-{crash_point}"
    assert cas_boundary in REQUIRED_CAS_INTERRUPTION_POINTS

    def interrupt(point: str) -> None:
        if point == cas_boundary:
            raise InjectedCrash(f"{crash_point}:{point}")

    seed_cid: str | None = None
    with DurableCoordinationStore(root) as setup:
        campaigns = DurableMutationCampaignRepository(setup)
        seed_cid = _advance_to(
            campaigns, target=prior_phase, op_prefix=f"{crash_point}-seed"
        )
        prior_head = campaigns.current_campaign_state(WORKSPACE)
        prior_generation = prior_head.generation
        prior_state_cid = prior_head.state_cid

    assert seed_cid is not None
    assert prior_state_cid is not None

    with DurableCoordinationStore(root, crash_injector=interrupt) as store:
        campaigns = DurableMutationCampaignRepository(store)
        with pytest.raises(InjectedCrash):
            campaigns.transition_campaign_state(
                WORKSPACE,
                state=_builder_state(
                    phase=next_phase,
                    execution_claim_status=ExecutionClaimStatus.COMPLETE,
                ),
                expected_generation=prior_generation,
                expected_state_cid=prior_state_cid,
                operation_id=f"{crash_point}-interrupted",
            )

    with DurableCoordinationStore(root) as recovered:
        report = recover_assurance_campaigns(recovered)
        campaigns = DurableMutationCampaignRepository(recovered)
        head = campaigns.current_campaign_state(WORKSPACE)
        if cas_boundary in _PRE_DURABLE_CAS:
            assert head.generation == prior_generation
            assert head.phase is prior_phase
        else:
            assert head.generation == prior_generation + 1
            assert head.phase is next_phase

        replay = campaigns.transition_campaign_state(
            WORKSPACE,
            state=_builder_state(
                phase=next_phase,
                execution_claim_status=ExecutionClaimStatus.COMPLETE,
            ),
            expected_generation=prior_generation,
            expected_state_cid=prior_state_cid,
            operation_id=f"{crash_point}-interrupted",
        )
        if cas_boundary in _PRE_DURABLE_CAS:
            assert replay.status is AssuranceStoreStatus.UPDATED
        else:
            assert replay.status is AssuranceStoreStatus.UNCHANGED
            assert replay.reason_code == "idempotent_replay"
        final = campaigns.current_campaign_state(WORKSPACE)
        assert final.phase is next_phase
        assert final.generation == prior_generation + 1
        return {
            "crash_point": crash_point,
            "cas_boundary": cas_boundary,
            "final_phase": final.phase.value,
            "final_generation": final.generation,
            "report_errors": list(report.errors),
            "restarted_safely": True,
            "deterministic_replay": True,
        }


def _run_during_diagnosis(tmp_path: Path) -> dict[str, Any]:
    return _run_campaign_phase_crash(
        tmp_path,
        crash_point="during_diagnosis",
        prior_phase=CampaignPhase.EXECUTING,
        next_phase=CampaignPhase.DIAGNOSING,
    )


def _run_during_evaluation(tmp_path: Path) -> dict[str, Any]:
    return _run_campaign_phase_crash(
        tmp_path,
        crash_point="during_evaluation",
        prior_phase=CampaignPhase.DIAGNOSING,
        next_phase=CampaignPhase.EVALUATING,
    )


def _run_during_root_update(tmp_path: Path) -> dict[str, Any]:
    root = tmp_path / "store-root"
    cas_boundary = "after_sqlite_commit"

    def interrupt(point: str) -> None:
        if point == cas_boundary:
            raise InjectedCrash(f"during_root_update:{point}")

    set_cids: dict[str, str] = {}
    with DurableCoordinationStore(root) as setup:
        artifacts = DurableAssuranceArtifactStore(setup)
        merkle = DurableAssuranceCampaignMerkleRepository(setup, artifacts=artifacts)
        set_cids = _commit_all_sets(merkle, setup)
        artifacts.close()

    with DurableCoordinationStore(root, crash_injector=interrupt) as store:
        artifacts = DurableAssuranceArtifactStore(store)
        merkle = DurableAssuranceCampaignMerkleRepository(store, artifacts=artifacts)
        with pytest.raises(InjectedCrash):
            merkle.commit_campaign_roots(
                WORKSPACE,
                campaign_id=CAMPAIGN_ID,
                set_commitments=set_cids,
                expected_generation=0,
                expected_root_cid=None,
                operation_id="root-interrupted",
            )
        artifacts.close()

    with DurableCoordinationStore(root) as recovered:
        report = recover_assurance_campaigns(recovered)
        artifacts = DurableAssuranceArtifactStore(recovered)
        merkle = DurableAssuranceCampaignMerkleRepository(
            recovered, artifacts=artifacts
        )
        try:
            head = merkle.current_merkle_root(WORKSPACE)
            # Post-commit crash: head advanced; set completeness preserved.
            assert head.generation == 1
            assert head.required_set_completeness is True
            replay = merkle.commit_campaign_roots(
                WORKSPACE,
                campaign_id=CAMPAIGN_ID,
                set_commitments=set_cids,
                expected_generation=0,
                expected_root_cid=None,
                operation_id="root-interrupted",
            )
            assert replay.status is AssuranceStoreStatus.UNCHANGED
            assert merkle.current_merkle_root(WORKSPACE).generation == 1
            return {
                "crash_point": "during_root_update",
                "root_generation": 1,
                "required_set_completeness": True,
                "report_errors": list(report.errors),
                "restarted_safely": True,
                "deterministic_replay": True,
            }
        finally:
            artifacts.close()


def _run_before_policy_cas(tmp_path: Path) -> dict[str, Any]:
    root = tmp_path / "store-before-cas"
    cas_boundary = "before_transaction"

    def interrupt(point: str) -> None:
        if point == cas_boundary:
            raise InjectedCrash(f"before_policy_cas:{point}")

    with DurableCoordinationStore(root) as setup:
        successor = _block(setup, "policy-before-cas")

    with DurableCoordinationStore(root, crash_injector=interrupt) as store:
        policy = DurableAssurancePolicyRepository(store)
        with pytest.raises(InjectedCrash):
            policy.compare_and_swap_policy(
                WORKSPACE,
                expected_generation=0,
                expected_policy_cid=None,
                new_policy_cid=successor,
                operation_id="policy-before-cas",
            )

    with DurableCoordinationStore(root) as recovered:
        report = recover_assurance_campaigns(recovered)
        policy = DurableAssurancePolicyRepository(recovered)
        head = policy.current_policy(WORKSPACE)
        # Pre-durable: head unchanged; immutable successor still present.
        assert head.generation == 0
        assert head.policy_cid is None
        assert recovered.has(successor) is True
        # Restart completes the CAS.
        done = policy.compare_and_swap_policy(
            WORKSPACE,
            expected_generation=0,
            expected_policy_cid=None,
            new_policy_cid=successor,
            operation_id="policy-before-cas",
        )
        assert done.status is AssuranceStoreStatus.UPDATED
        assert policy.current_policy(WORKSPACE).policy_cid == successor
        return {
            "crash_point": "before_policy_cas",
            "pre_crash_generation": 0,
            "final_policy_cid": successor,
            "immutable_successor_survived": True,
            "report_errors": list(report.errors),
            "restarted_safely": True,
        }


def _run_after_cas_before_cleanup(tmp_path: Path) -> dict[str, Any]:
    """CAS succeeds, then crash before disposable worktree cleanup; restart cleans."""

    root = tmp_path / "store-after-cas"
    repo = tmp_path / "repo-cleanup"
    head, tree = _init_repo(repo)
    lifecycle = _lifecycle_store(repo, tmp_path / "lc-cleanup")
    parent = tmp_path / "owned-wts-cleanup"
    parent.mkdir(parents=True, exist_ok=True)

    policy_cid: str | None = None
    with DurableCoordinationStore(root) as store:
        policy_cid = _block(store, "policy-after-cas")
        policy = DurableAssurancePolicyRepository(store)
        cas = policy.compare_and_swap_policy(
            WORKSPACE,
            expected_generation=0,
            expected_policy_cid=None,
            new_policy_cid=policy_cid,
            operation_id="policy-after-cas-ok",
        )
        assert cas.status is AssuranceStoreStatus.UPDATED

    # Disposable worktree still live when process crashes after CAS.
    isolated = create_mutation_worktree(
        repo_root=repo,
        worktree_path=parent / "post-cas-mutant",
        worktree_parent=parent,
        base_commit=head,
        base_tree=tree,
        task_id="AAE-060-cleanup",
        attempt=1,
        lifecycle_store=lifecycle,
    )
    live_lease = isolated.lease_id
    live_fence = isolated.fence
    wt_path = isolated.worktree_path
    try:
        isolated.apply_replacements(
            {"mod.py": MOD_MUTATED},
            _scope(),
            lease_id=live_lease,
            fence=live_fence,
        )
        # Crash after CAS, before cleanup.
        try:
            raise InjectedCrash("after_cas_before_cleanup")
        except InjectedCrash:
            pass

        # Stale peer cannot clean or apply.
        with pytest.raises(MutationWorktreeFenceError):
            isolated.cleanup(lease_id="not-the-owner", fence=live_fence)
        decision = lifecycle.evaluate_cleanup(workspace_path=wt_path)
        assert decision.allowed is False
        assert decision.disposition is CleanupDisposition.DENY

        # Owner restarts and cleans only the owned disposable worktree.
        cleaned = isolated.cleanup(
            lease_id=live_lease,
            fence=isolated.fence,
            reason="post_cas_restart_cleanup",
        )
        assert cleaned["cleaned"] is True
        assert not wt_path.exists()
        # Production untouched; policy CAS head still current.
        assert (repo / "mod.py").read_text(encoding="utf-8") == MOD_SOURCE
        with DurableCoordinationStore(root) as recovered:
            report = recover_assurance_campaigns(recovered)
            policy = DurableAssurancePolicyRepository(recovered)
            head_pol = policy.current_policy(WORKSPACE)
            assert head_pol.policy_cid == policy_cid
            assert head_pol.generation == 1
            return {
                "crash_point": "after_cas_before_cleanup",
                "policy_cid": policy_cid,
                "worktree_cleaned": True,
                "stale_cleanup_denied": True,
                "production_untouched": True,
                "report_errors": list(report.errors),
                "restarted_safely": True,
            }
    finally:
        if wt_path.exists():
            recover_mutation_worktree(
                lifecycle_store=lifecycle,
                worktree_path=wt_path,
                repo_root=repo,
                worktree_parent=parent,
                caller_lease_id=live_lease,
            )


_CRASH_RUNNERS: dict[str, Callable[[Path], dict[str, Any]]] = {
    "after_mutant_creation": _run_after_mutant_creation,
    "during_worktree_setup": _run_during_worktree_setup,
    "during_test_execution": _run_during_test_execution,
    "during_proof_execution": _run_during_proof_execution,
    "after_receipt_persistence": _run_after_receipt_persistence,
    "during_diagnosis": _run_during_diagnosis,
    "during_evaluation": _run_during_evaluation,
    "during_root_update": _run_during_root_update,
    "before_policy_cas": _run_before_policy_cas,
    "after_cas_before_cleanup": _run_after_cas_before_cleanup,
}


@pytest.mark.parametrize("crash_point", REQUIRED_PIPELINE_CRASH_POINTS)
def test_each_required_crash_point_restarts_safely(
    tmp_path: Path, crash_point: str
) -> None:
    runner = _CRASH_RUNNERS[crash_point]
    result = runner(tmp_path / crash_point)
    assert result["crash_point"] == crash_point
    assert result["restarted_safely"] is True


def test_all_ten_crash_points_covered_by_runners() -> None:
    assert set(_CRASH_RUNNERS) == set(REQUIRED_PIPELINE_CRASH_POINTS)
    assert len(_CRASH_RUNNERS) == 10


# ---------------------------------------------------------------------------
# Immutable evidence survival
# ---------------------------------------------------------------------------


def test_completed_immutable_evidence_survives_restart(tmp_path: Path) -> None:
    root = tmp_path / "immutable-survive"
    with DurableCoordinationStore(root) as store:
        artifacts = DurableAssuranceArtifactStore(store)
        campaigns = DurableMutationCampaignRepository(store, artifacts=artifacts)
        policy = DurableAssurancePolicyRepository(store)
        mutant_cid = _block(store, "completed-mutant")
        receipt_cid = _put_receipt(campaigns, op_suffix="complete")

        planned = campaigns.transition_campaign_state(
            WORKSPACE,
            state=_builder_state(
                phase=CampaignPhase.PLANNED,
                execution_claim_status=ExecutionClaimStatus.NONE,
                artifact_cids=[mutant_cid],
            ),
            expected_generation=0,
            expected_state_cid=None,
            operation_id="ok-plan",
        )
        executing = campaigns.transition_campaign_state(
            WORKSPACE,
            state=_builder_state(
                phase=CampaignPhase.EXECUTING,
                execution_claim_status=ExecutionClaimStatus.COMPLETE,
                artifact_cids=[mutant_cid],
            ),
            expected_generation=1,
            expected_state_cid=planned.state_cid,
            operation_id="ok-exec",
        )
        diagnosing = campaigns.transition_campaign_state(
            WORKSPACE,
            state=_builder_state(
                phase=CampaignPhase.DIAGNOSING,
                execution_claim_status=ExecutionClaimStatus.COMPLETE,
                artifact_cids=[mutant_cid],
            ),
            expected_generation=2,
            expected_state_cid=executing.state_cid,
            operation_id="ok-diag",
        )
        evaluating = campaigns.transition_campaign_state(
            WORKSPACE,
            state=_builder_state(
                phase=CampaignPhase.EVALUATING,
                execution_claim_status=ExecutionClaimStatus.COMPLETE,
                artifact_cids=[mutant_cid],
            ),
            expected_generation=3,
            expected_state_cid=diagnosing.state_cid,
            operation_id="ok-eval",
        )
        complete = campaigns.transition_campaign_state(
            WORKSPACE,
            state=_builder_state(
                phase=CampaignPhase.COMPLETE,
                execution_claim_status=ExecutionClaimStatus.COMPLETE,
                receipt_cid=receipt_cid,
                artifact_cids=[mutant_cid],
            ),
            expected_generation=4,
            expected_state_cid=evaluating.state_cid,
            operation_id="ok-complete",
        )
        assert complete.status is AssuranceStoreStatus.UPDATED

        policy_cid = _block(store, "policy-live")
        promo = policy.compare_and_swap_policy(
            WORKSPACE,
            expected_generation=0,
            expected_policy_cid=None,
            new_policy_cid=policy_cid,
            operation_id="policy-live",
        )
        assert promo.status is AssuranceStoreStatus.UPDATED
        artifacts.close()

    with DurableCoordinationStore(root) as recovered:
        report = recover_assurance_campaigns(recovered)
        assert report.errors == ()
        assert len(report.reconstructed_campaign_heads) == 1
        camp = report.reconstructed_campaign_heads[0]
        assert camp.phase is CampaignPhase.COMPLETE
        assert camp.receipt_cid == receipt_cid
        assert camp.execution_claim_status is ExecutionClaimStatus.COMPLETE
        assert recovered.has(mutant_cid) is True
        assert recovered.has(receipt_cid) is True
        assert len(report.reconstructed_policy_heads) == 1
        assert report.reconstructed_policy_heads[0].policy_cid == policy_cid

        artifacts = DurableAssuranceArtifactStore(recovered)
        try:
            verified = artifacts.get_verified_artifact(
                receipt_cid,
                expected_kind=AssuranceArtifactKind.ASSURANCE_CAMPAIGN_RECEIPT,
            )
            assert verified is not None
        finally:
            artifacts.close()


# ---------------------------------------------------------------------------
# Ambiguous / partial claims fail closed
# ---------------------------------------------------------------------------


def test_ambiguous_and_partial_claims_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(AssuranceRecoveryAdmissionError, match="partial|ambiguous"):
        assert_terminal_claim_not_ambiguous(
            phase=CampaignPhase.COMPLETE,
            execution_claim_status=ExecutionClaimStatus.PARTIAL,
            receipt_cid=cid_for_bytes(b"receipt"),
        )
    with pytest.raises(AssuranceRecoveryAdmissionError, match="partial|ambiguous"):
        assert_terminal_claim_not_ambiguous(
            phase=CampaignPhase.COMPLETE,
            execution_claim_status=ExecutionClaimStatus.AMBIGUOUS,
            receipt_cid=cid_for_bytes(b"receipt"),
        )
    with pytest.raises(CampaignTransitionError):
        assert_terminal_success_admissible(
            phase=CampaignPhase.COMPLETE,
            execution_claim_status=ExecutionClaimStatus.AMBIGUOUS,
            receipt_cid=cid_for_bytes(b"receipt"),
        )

    root = tmp_path / "ambiguous-reject"
    with DurableCoordinationStore(root) as store:
        campaigns = DurableMutationCampaignRepository(store)
        planned = campaigns.transition_campaign_state(
            WORKSPACE,
            state=_builder_state(
                phase=CampaignPhase.PLANNED,
                execution_claim_status=ExecutionClaimStatus.NONE,
            ),
            expected_generation=0,
            expected_state_cid=None,
            operation_id="amb-plan",
        )
        executing = campaigns.transition_campaign_state(
            WORKSPACE,
            state=_builder_state(
                phase=CampaignPhase.EXECUTING,
                execution_claim_status=ExecutionClaimStatus.PARTIAL,
            ),
            expected_generation=1,
            expected_state_cid=planned.state_cid,
            operation_id="amb-exec",
        )
        assert executing.status is AssuranceStoreStatus.UPDATED
        evaluating = campaigns.transition_campaign_state(
            WORKSPACE,
            state=_builder_state(
                phase=CampaignPhase.EVALUATING,
                execution_claim_status=ExecutionClaimStatus.PARTIAL,
            ),
            expected_generation=2,
            expected_state_cid=executing.state_cid,
            operation_id="amb-eval",
        )
        assert evaluating.status is AssuranceStoreStatus.UPDATED
        with pytest.raises(CampaignTransitionError, match="partial|ambiguous"):
            campaigns.transition_campaign_state(
                WORKSPACE,
                state=_builder_state(
                    phase=CampaignPhase.COMPLETE,
                    execution_claim_status=ExecutionClaimStatus.PARTIAL,
                    receipt_cid=cid_for_bytes(b"fake-receipt"),
                ),
                expected_generation=3,
                expected_state_cid=evaluating.state_cid,
                operation_id="amb-complete",
            )

    with DurableCoordinationStore(root) as recovered:
        report = recover_assurance_campaigns(recovered)
        assert len(report.reconstructed_campaign_heads) == 1
        head = report.reconstructed_campaign_heads[0]
        assert head.phase is CampaignPhase.EVALUATING
        assert head.execution_claim_status is ExecutionClaimStatus.PARTIAL
        assert not any(
            item.phase is CampaignPhase.COMPLETE
            for item in report.reconstructed_campaign_heads
        )


def test_unverified_signed_receipt_cannot_complete(tmp_path: Path) -> None:
    root = tmp_path / "unverified-receipt"
    payload = receipt_fixtures._campaign(
        header=receipt_fixtures._header(
            "assurance_campaign_receipt",
            terminal_status=AssuranceTerminalStatus.REJECTED,
        ),
        held_out_result=HeldOutResult.FAILED,
        signature=receipt_fixtures._signature(
            signature_verification_status=SignatureVerificationStatus.UNVERIFIED
        ),
    ).to_dict()
    with DurableCoordinationStore(root) as store:
        campaigns = DurableMutationCampaignRepository(store)
        with pytest.raises(Exception, match="signature|unverified"):
            admit_campaign_receipt_payload(payload)
        head = campaigns.current_campaign_state(WORKSPACE)
        assert head.generation == 0
        report = recover_assurance_campaigns(store)
        assert report.reconstructed_campaign_heads == ()


# ---------------------------------------------------------------------------
# CAS: one current writer
# ---------------------------------------------------------------------------


def test_cas_permits_one_current_writer(tmp_path: Path) -> None:
    root = tmp_path / "cas-one-writer"
    with DurableCoordinationStore(root) as setup:
        one = _block(setup, "c1")
        two = _block(setup, "c2")
        seed = _block(setup, "seed")
        DurableAssurancePolicyRepository(setup).compare_and_swap_policy(
            "seed-ws",
            expected_generation=0,
            expected_policy_cid=None,
            new_policy_cid=seed,
            operation_id="seed-op",
        )

    with DurableCoordinationStore(root) as recovered:
        recover_assurance_campaigns(recovered)

    def attempt(cid: str, operation_id: str) -> str:
        with DurableCoordinationStore(root) as store:
            policy = DurableAssurancePolicyRepository(store)
            result = policy.compare_and_swap_policy(
                WORKSPACE,
                expected_generation=0,
                expected_policy_cid=None,
                new_policy_cid=cid,
                operation_id=operation_id,
            )
            return result.status.value

    with ThreadPoolExecutor(max_workers=2) as pool:
        statuses = list(
            pool.map(
                lambda args: attempt(*args),
                ((one, "writer-1"), (two, "writer-2")),
            )
        )
    assert sorted(statuses) == ["conflict", "updated"]

    with DurableCoordinationStore(root) as store:
        report = recover_assurance_campaigns(store)
        policy = DurableAssurancePolicyRepository(store)
        head = policy.current_policy(WORKSPACE)
        assert head.generation == 1
        assert head.policy_cid in (one, two)
        assert any(
            item.policy_cid == head.policy_cid
            for item in report.reconstructed_policy_heads
        )
        # Stale writer fence rejects pre-CAS expectation.
        with pytest.raises(AssuranceRecoveryAdmissionError, match="stale writer"):
            assert_writer_fence(
                expected_generation=0,
                expected_head_cid=None,
                current_generation=head.generation,
                current_head_cid=head.policy_cid,
            )
        stale = policy.compare_and_swap_policy(
            WORKSPACE,
            expected_generation=0,
            expected_policy_cid=None,
            new_policy_cid=_block(store, "stale-attempt"),
            operation_id="stale-writer",
        )
        assert stale.status is AssuranceStoreStatus.CONFLICT
        assert stale.reason_code == "stale_expectation"
        assert policy.current_policy(WORKSPACE).policy_cid == head.policy_cid


# ---------------------------------------------------------------------------
# Worktree / process fencing
# ---------------------------------------------------------------------------


def test_worktree_owner_fence_blocks_stale_cleanup(tmp_path: Path) -> None:
    repo = tmp_path / "repo-fence"
    head, tree = _init_repo(repo)
    store = _lifecycle_store(repo, tmp_path)
    parent = tmp_path / "owned-wts-fence"
    isolated = create_mutation_worktree(
        repo_root=repo,
        worktree_path=parent / "peer",
        worktree_parent=parent,
        base_commit=head,
        base_tree=tree,
        task_id="AAE-060-fence",
        attempt=1,
        lifecycle_store=store,
    )
    try:
        live_lease = isolated.lease_id
        live_fence = isolated.fence
        stale_fence = live_fence - 1 if live_fence > 1 else 0

        with pytest.raises(MutationWorktreeFenceError):
            isolated.apply_replacements(
                {"mod.py": MOD_MUTATED},
                _scope(),
                lease_id=live_lease,
                fence=stale_fence,
            )
        with pytest.raises(MutationWorktreeFenceError):
            isolated.cleanup(lease_id="not-the-owner", fence=live_fence)

        decision = store.evaluate_cleanup(workspace_path=isolated.worktree_path)
        assert decision.allowed is False
        assert decision.disposition is CleanupDisposition.DENY

        cleaned = isolated.cleanup(
            lease_id=isolated.lease_id,
            fence=isolated.fence,
            reason="owner_done",
        )
        assert cleaned["cleaned"] is True
        assert not isolated.worktree_path.exists()
        assert (repo / "mod.py").read_text(encoding="utf-8") == MOD_SOURCE
    finally:
        if not isolated._closed and isolated.worktree_path.exists():
            recover_mutation_worktree(
                lifecycle_store=store,
                worktree_path=isolated.worktree_path,
                repo_root=repo,
                worktree_parent=parent,
            )


def test_process_tree_fenced_on_cancellation(tmp_path: Path) -> None:
    result = _run_during_test_execution(tmp_path / "process-fence")
    assert result["process_tree_fenced"] is True
    assert result["cancelled"] is True
    assert result["restarted_safely"] is True


def test_worker_checkpoint_restart_recovers_incomplete(tmp_path: Path) -> None:
    store = MutationWorkerCheckpointStore(tmp_path / "ck")
    store.mark_running(
        "orphan-task",
        lease_id="lease-x",
        pool_id="pool-old",
        attempt=1,
    )
    incomplete = store.list_incomplete()
    assert len(incomplete) == 1

    pool = MutationWorkerPool(
        MutationWorkerBudget(max_concurrency=1),
        resource_scheduler=ResourceScheduler(ResourcePolicy(max_lanes=1)),
        host_snapshot=_host(),
        checkpoint_dir=tmp_path / "ck",
        pool_id="pool-new",
    )
    try:
        recovered = pool.recover()
        assert len(recovered) == 1
        result = recovered[0]
        assert result.task_id == "orphan-task"
        assert (
            result.disposition is MutationWorkerDisposition.INFRASTRUCTURE_FAILURE
        )
        assert result.infrastructure.restart_recovered is True
        assert "restart_recovered_incomplete" in result.infrastructure.reason_codes
        assert store.list_incomplete() == ()
    finally:
        pool.shutdown(wait=True)


# ---------------------------------------------------------------------------
# Deterministic full-pipeline replay after multi-point recovery
# ---------------------------------------------------------------------------


def test_deterministic_replay_across_campaign_and_policy(tmp_path: Path) -> None:
    """Idempotent operation_ids converge to one durable head after restarts."""

    root = tmp_path / "deterministic"
    with DurableCoordinationStore(root) as store:
        campaigns = DurableMutationCampaignRepository(store)
        policy = DurableAssurancePolicyRepository(store)
        receipt_cid = _put_receipt(campaigns, op_suffix="det")
        complete_cid = _advance_to(
            campaigns,
            target=CampaignPhase.COMPLETE,
            receipt_cid=receipt_cid,
            op_prefix="det",
        )
        policy_cid = _block(store, "det-policy")
        first = policy.compare_and_swap_policy(
            WORKSPACE,
            expected_generation=0,
            expected_policy_cid=None,
            new_policy_cid=policy_cid,
            operation_id="det-policy-cas",
        )
        assert first.status is AssuranceStoreStatus.UPDATED
        second = policy.compare_and_swap_policy(
            WORKSPACE,
            expected_generation=0,
            expected_policy_cid=None,
            new_policy_cid=policy_cid,
            operation_id="det-policy-cas",
        )
        assert second.status is AssuranceStoreStatus.UNCHANGED
        assert second.reason_code == "idempotent_replay"

    with DurableCoordinationStore(root) as recovered:
        report = recover_assurance_campaigns(recovered)
        assert report.errors == ()
        campaigns = DurableMutationCampaignRepository(recovered)
        policy = DurableAssurancePolicyRepository(recovered)
        head = campaigns.current_campaign_state(WORKSPACE)
        assert head.phase is CampaignPhase.COMPLETE
        assert head.state_cid == complete_cid
        assert policy.current_policy(WORKSPACE).policy_cid == policy_cid

        # Replay after recovery still idempotent.
        again = policy.compare_and_swap_policy(
            WORKSPACE,
            expected_generation=0,
            expected_policy_cid=None,
            new_policy_cid=policy_cid,
            operation_id="det-policy-cas",
        )
        assert again.status is AssuranceStoreStatus.UNCHANGED
        assert policy.current_policy(WORKSPACE).generation == 1


def test_no_partial_promotion_after_interrupted_promote(tmp_path: Path) -> None:
    root = tmp_path / "no-partial-promo"
    boundary = "after_sqlite_commit"

    def interrupt(point: str) -> None:
        if point == boundary:
            raise InjectedCrash(point)

    with DurableCoordinationStore(root) as setup:
        candidate = _block(setup, "cand")
        evaluation = _block(setup, "eval")
        auth = _block(setup, "auth")
        policy_cid = _block(setup, "promo-policy")

    with DurableCoordinationStore(root, crash_injector=interrupt) as store:
        policy = DurableAssurancePolicyRepository(store)
        with pytest.raises(InjectedCrash):
            policy.promote_policy(
                WORKSPACE,
                expected_generation=0,
                expected_policy_cid=None,
                new_policy_cid=policy_cid,
                operation_id="promote-interrupted",
                candidate_cid=candidate,
                evaluation_cid=evaluation,
                authorization_cid=auth,
            )

    with DurableCoordinationStore(root) as recovered:
        report = recover_assurance_campaigns(recovered)
        policy = DurableAssurancePolicyRepository(recovered)
        # promote_policy CAS-es policy only; promotion-state head must not be invented.
        assert policy.current_promotion(WORKSPACE).generation == 0
        assert policy.current_promotion(WORKSPACE).promotion_cid is None
        assert report.reconstructed_promotion_heads == ()
        assert policy.current_policy(WORKSPACE).generation == 1
        assert policy.current_policy(WORKSPACE).policy_cid == policy_cid
