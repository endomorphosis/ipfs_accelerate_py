"""Focused fail-closed tests for the bounded PCTDD occurrence recoveries."""

from __future__ import annotations

import hashlib
import json
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    database_portal_bridge as bridge_module,
    implementation_daemon as daemon_module,
    implementation_supervisor as supervisor_module,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    ImplementationRetryDeferred,
    PCTDD_RETAINED_CANDIDATE_QUARANTINE_SCHEMA,
    PCTDD_RETAINED_CANDIDATE_RETRY_CONSUMPTION_SCHEMA,
    PCTDD_RETAINED_CANDIDATE_RETRY_CREDIT_SCHEMA,
    PCTDD_RETAINED_CANDIDATE_TASK_CID,
    PCTDD_RETAINED_CANDIDATE_TASK_ID,
    PortalImplementationDaemon,
    PortalTask,
    PortalTaskState,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
    PortalImplementationSupervisor,
)


_MANIFEST_SCHEMA = "test/fenced-provider-migration-manifest@1"
_MANIFEST_REVISION = "pctdd-provider-recovery-2026-09-02"


def _git(repo: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    if check:
        assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _repo(tmp_path: Path) -> tuple[Path, str, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Test")
    (repo / "tracked.txt").write_text("baseline\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-qm", "baseline")
    baseline = _git(repo, "rev-parse", "HEAD")
    (repo / "tracked.txt").write_text("divergent\n", encoding="utf-8")
    _git(repo, "commit", "-qam", "divergent")
    divergent = _git(repo, "rev-parse", "HEAD")
    return repo, baseline, divergent


def _occurrences(tmp_path: Path, baseline: str) -> list[dict[str, object]]:
    return [
        {
            "task_alias": task,
            "task_cid": f"cid:{task.lower()}",
            "branch": f"implementation/{task.lower()}-exact-occurrence",
            "baseline_ref": baseline,
            "workspace_path": str(tmp_path / f"missing-{task.lower()}"),
        }
        for task in ("PCTDD-006", "PCTDD-007", "PCTDD-034")
    ]


def _manifest_id(occurrences: list[dict[str, object]]) -> str:
    payload = {
        "schema": _MANIFEST_SCHEMA,
        "revision": _MANIFEST_REVISION,
        "operator_owned": True,
        "one_shot": True,
        "occurrences": occurrences,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        default=str,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _supervisor(repo: Path, tmp_path: Path) -> tuple[PortalImplementationSupervisor, list[tuple[str, dict[str, object]]]]:
    supervisor = object.__new__(PortalImplementationSupervisor)
    supervisor.config = SimpleNamespace(
        repo_root=repo,
        worktree_root=tmp_path / "managed-worktrees",
        state_dir=repo / "state",
        state_path=repo / "state" / "task-state.json",
        todo_path=repo / "todo.md",
        state_prefix="test",
        merge_target_branch="master",
        worktree_scan_cache_enabled=False,
        worktree_scan_cache_ttl_seconds=0.0,
        worktree_scan_cache_path=None,
        worktree_submodule_paths=(),
    )
    supervisor._checkout_mutation_context = threading.local()
    supervisor.board_namespace = (
        "parallel-content-sealing-proof-carrying-tdd-v1"
    )
    events: list[tuple[str, dict[str, object]]] = []
    supervisor._record_event = lambda event_type, payload: events.append(
        (event_type, dict(payload))
    )
    return supervisor, events


def _run_cleanup(
    supervisor: PortalImplementationSupervisor,
    occurrences: list[dict[str, object]],
    *,
    manifest_id: str | None = None,
) -> dict[str, object]:
    lock_path = supervisor._repo_merge_lock_path()
    metadata = supervisor._supervisor_checkout_lock_metadata(
        operation="cleanup_backlogged_worktrees",
    )
    lease, reason, _existing = supervisor._acquire_supervisor_checkout_lease(
        lock_path,
        metadata,
    )
    assert lease is not None, reason
    try:
        with supervisor._supervisor_checkout_transaction(lease):
            return supervisor._cleanup_fenced_provider_migration_branches_locked(
                checkout_lease=lease,
                worktree_prune_result=subprocess.CompletedProcess(
                    args=("git", "worktree", "prune"),
                    returncode=0,
                    stdout="",
                    stderr="",
                ),
                occurrences=occurrences,
                manifest_schema=_MANIFEST_SCHEMA,
                manifest_id=(manifest_id or _manifest_id(occurrences)),
            )
    finally:
        supervisor._release_supervisor_checkout_lease(
            lease,
            operation="cleanup_backlogged_worktrees",
        )


def test_exact_baseline_refs_are_cas_deleted_and_true_absence_is_preserved(
    tmp_path: Path,
) -> None:
    repo, baseline, _divergent = _repo(tmp_path)
    occurrences = _occurrences(tmp_path, baseline)
    for occurrence in (occurrences[0], occurrences[2]):
        _git(repo, "update-ref", f"refs/heads/{occurrence['branch']}", baseline)
    supervisor, events = _supervisor(repo, tmp_path)

    result = _run_cleanup(supervisor, occurrences)

    assert result["prerequisites_satisfied"] is True
    assert result["deleted_count"] == 2
    assert result["already_absent_count"] == 1
    for occurrence in occurrences:
        assert not _git(
            repo,
            "show-ref",
            "--verify",
            "--hash",
            f"refs/heads/{occurrence['branch']}",
            check=False,
        )
    assert len(events) == 1
    assert events[0][0] == "fenced_provider_migration_branch_cleanup"
    assert events[0][1]["authorizes_recovery_credit"] is False
    assert events[0][1]["evidence_authoritative"] is False
    assert all(
        item["recovery_credit_issued"] is False
        for item in events[0][1]["branches"]
    )


def test_wrong_manifest_id_fails_without_ref_mutation(tmp_path: Path) -> None:
    repo, baseline, _divergent = _repo(tmp_path)
    occurrences = _occurrences(tmp_path, baseline)
    ref = f"refs/heads/{occurrences[0]['branch']}"
    _git(repo, "update-ref", ref, baseline)
    supervisor, _events = _supervisor(repo, tmp_path)

    result = _run_cleanup(
        supervisor,
        occurrences,
        manifest_id="sha256:" + "0" * 64,
    )

    assert result["prerequisites_satisfied"] is False
    assert result["reason"] == "operator_occurrence_manifest_invalid"
    assert _git(repo, "show-ref", "--verify", "--hash", ref) == baseline


def test_direct_cleanup_without_current_lease_preserves_ref(tmp_path: Path) -> None:
    repo, baseline, _divergent = _repo(tmp_path)
    occurrences = _occurrences(tmp_path, baseline)
    ref = f"refs/heads/{occurrences[0]['branch']}"
    _git(repo, "update-ref", ref, baseline)
    supervisor, _events = _supervisor(repo, tmp_path)

    result = supervisor._cleanup_fenced_provider_migration_branches_locked(
        worktree_prune_result=subprocess.CompletedProcess(
            args=(), returncode=0, stdout="", stderr=""
        ),
        occurrences=occurrences,
        manifest_schema=_MANIFEST_SCHEMA,
        manifest_id=_manifest_id(occurrences),
    )

    assert result["reason"] == "checkout_mutation_lease_not_current"
    assert _git(repo, "show-ref", "--verify", "--hash", ref) == baseline


def test_foreign_board_never_loads_or_mutates_occurrence_manifest(
    tmp_path: Path,
) -> None:
    repo, _baseline, _divergent = _repo(tmp_path)
    supervisor, _events = _supervisor(repo, tmp_path)
    supervisor.board_namespace = "foreign-board"

    result = supervisor._cleanup_fenced_provider_migration_branches_locked()

    assert result == {
        "attempted": False,
        "reason": "occurrence_manifest_board_not_active",
        "prerequisites_satisfied": False,
        "authorizes_recovery_credit": False,
        "evidence_authoritative": False,
    }


def test_divergent_ref_is_preserved(tmp_path: Path) -> None:
    repo, baseline, divergent = _repo(tmp_path)
    occurrences = _occurrences(tmp_path, baseline)
    ref = f"refs/heads/{occurrences[0]['branch']}"
    _git(repo, "update-ref", ref, divergent)
    supervisor, _events = _supervisor(repo, tmp_path)

    result = _run_cleanup(supervisor, occurrences)

    assert result["prerequisites_satisfied"] is False
    assert result["branches"][0]["reason"] == "ref_target_mismatch"
    assert _git(repo, "show-ref", "--verify", "--hash", ref) == divergent


def test_linked_worktree_ref_is_preserved(tmp_path: Path) -> None:
    repo, baseline, _divergent = _repo(tmp_path)
    occurrences = _occurrences(tmp_path, baseline)
    branch = str(occurrences[0]["branch"])
    ref = f"refs/heads/{branch}"
    _git(repo, "update-ref", ref, baseline)
    linked = tmp_path / "linked"
    _git(repo, "worktree", "add", "-q", str(linked), branch)
    supervisor, _events = _supervisor(repo, tmp_path)

    result = _run_cleanup(supervisor, occurrences)

    assert result["prerequisites_satisfied"] is False
    assert result["branches"][0]["reason"] == "linked_worktree_present"
    assert _git(repo, "show-ref", "--verify", "--hash", ref) == baseline


@pytest.mark.parametrize("kind", ["directory", "symlink"])
def test_present_or_symlink_workspace_blocks_cleanup(
    tmp_path: Path,
    kind: str,
) -> None:
    repo, baseline, _divergent = _repo(tmp_path)
    occurrences = _occurrences(tmp_path, baseline)
    workspace = Path(str(occurrences[0]["workspace_path"]))
    if kind == "directory":
        workspace.mkdir()
    else:
        workspace.symlink_to(tmp_path / "missing-target")
    ref = f"refs/heads/{occurrences[0]['branch']}"
    _git(repo, "update-ref", ref, baseline)
    supervisor, _events = _supervisor(repo, tmp_path)

    result = _run_cleanup(supervisor, occurrences)

    assert result["branches"][0]["reason"] == "workspace_present"
    assert _git(repo, "show-ref", "--verify", "--hash", ref) == baseline


def test_malformed_manifest_ref_is_rejected(tmp_path: Path) -> None:
    repo, baseline, _divergent = _repo(tmp_path)
    occurrences = _occurrences(tmp_path, baseline)
    occurrences[0]["branch"] = "implementation/../foreign"
    supervisor, _events = _supervisor(repo, tmp_path)

    result = _run_cleanup(supervisor, occurrences)

    assert result["branches"][0]["reason"] == "invalid_manifest_ref_binding"


def test_concurrent_ref_change_causes_cas_rejection_and_preserves_ref(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, baseline, divergent = _repo(tmp_path)
    occurrences = _occurrences(tmp_path, baseline)
    ref = f"refs/heads/{occurrences[0]['branch']}"
    _git(repo, "update-ref", ref, baseline)
    supervisor, _events = _supervisor(repo, tmp_path)
    real_run = subprocess.run
    raced = False

    def racing_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal raced
        if args[:4] == ["git", "update-ref", "-d", ref] and not raced:
            raced = True
            real_run(
                ["git", "update-ref", ref, divergent],
                cwd=repo,
                text=True,
                capture_output=True,
                check=False,
            )
        return real_run(args, **kwargs)

    monkeypatch.setattr(supervisor_module.subprocess, "run", racing_run)

    result = _run_cleanup(supervisor, occurrences)

    assert raced
    assert result["branches"][0]["reconciled"] is False
    assert result["branches"][0]["reason"] == "cas_delete_or_reprobe_failed"
    assert _git(repo, "show-ref", "--verify", "--hash", ref) == divergent


def test_failed_post_delete_reprobe_never_reports_reconciled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, baseline, _divergent = _repo(tmp_path)
    occurrences = _occurrences(tmp_path, baseline)
    ref = f"refs/heads/{occurrences[0]['branch']}"
    _git(repo, "update-ref", ref, baseline)
    supervisor, _events = _supervisor(repo, tmp_path)
    real_run = subprocess.run
    deleted = False

    def reprobe_failure(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal deleted
        if args[:4] == ["git", "update-ref", "-d", ref]:
            result = real_run(args, **kwargs)
            deleted = result.returncode == 0
            return result
        if (
            deleted
            and args[:5] == ["git", "rev-parse", "--verify", "--quiet", ref]
        ):
            return subprocess.CompletedProcess(args=args, returncode=2, stdout="", stderr="reprobe failed")
        return real_run(args, **kwargs)

    monkeypatch.setattr(supervisor_module.subprocess, "run", reprobe_failure)

    result = _run_cleanup(supervisor, occurrences)

    assert result["branches"][0]["reconciled"] is False
    assert result["branches"][0]["reason"] == "cas_delete_or_reprobe_failed"


@pytest.mark.parametrize("failure", ["command", "malformed"])
def test_inconclusive_worktree_registry_never_deletes_ref(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    repo, baseline, _divergent = _repo(tmp_path)
    occurrences = _occurrences(tmp_path, baseline)
    ref = f"refs/heads/{occurrences[0]['branch']}"
    _git(repo, "update-ref", ref, baseline)
    supervisor, _events = _supervisor(repo, tmp_path)
    real_run = subprocess.run
    update_ref_called = False

    def failed_registry(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[object]:
        nonlocal update_ref_called
        if args == ["git", "worktree", "list", "--porcelain", "-z"]:
            return subprocess.CompletedProcess(
                args=args,
                returncode=1 if failure == "command" else 0,
                stdout=(b"" if failure == "command" else b"worktree /truncated\0"),
                stderr=b"registry unavailable",
            )
        if args[:3] == ["git", "update-ref", "-d"]:
            update_ref_called = True
        return real_run(args, **kwargs)

    monkeypatch.setattr(supervisor_module.subprocess, "run", failed_registry)

    result = _run_cleanup(supervisor, occurrences)

    assert result["prerequisites_satisfied"] is False
    assert result["reason"] in {
        "worktree_registry_command_failed",
        "worktree_registry_malformed",
    }
    assert update_ref_called is False
    assert _git(repo, "show-ref", "--verify", "--hash", ref) == baseline


def _p005_task() -> PortalTask:
    return PortalTask(
        task_id=PCTDD_RETAINED_CANDIDATE_TASK_ID,
        title="Prepared canonical contracts",
        status="ready",
        completion="automatic",
        priority="P0",
        track="content",
        canonical_task_key="pctdd-005",
        canonical_task_cid=PCTDD_RETAINED_CANDIDATE_TASK_CID,
        board_namespace="parallel-content-sealing-proof-carrying-tdd-v1",
    )


def _p005_retry_daemon(
    tmp_path: Path,
    *,
    successor: bool = True,
) -> tuple[PortalImplementationDaemon, PortalTask, list[dict[str, object]], dict[str, object]]:
    daemon = object.__new__(PortalImplementationDaemon)
    task = _p005_task()
    events: list[dict[str, object]] = []
    issuer_birth = {
        "pid": 110,
        "start_time_ticks": 1000,
        "boot_id": "boot-a",
        "parent_pid": 10,
    }
    consumer_birth = {
        "pid": 210 if successor else 110,
        "start_time_ticks": 2000 if successor else 1000,
        "boot_id": "boot-a",
        "parent_pid": 20 if successor else 10,
    }
    daemon.use_ephemeral_worktree = True
    daemon._controller_session_id = (
        "portal-session:successor" if successor else "portal-session:issuer"
    )
    daemon._implementation_dispatch_process_birth = SimpleNamespace(
        to_dict=lambda: dict(consumer_birth)
    )
    daemon._identity_for_task = lambda _task: SimpleNamespace(
        canonical_task_cid=PCTDD_RETAINED_CANDIDATE_TASK_CID,
        board_namespace=task.board_namespace,
    )
    daemon._canonical_ref = lambda _task: PCTDD_RETAINED_CANDIDATE_TASK_CID
    daemon._iter_events = lambda: tuple(events)
    daemon._record_event = lambda event_type, payload: events.append(
        {"type": event_type, **dict(payload)}
    )
    prior = tmp_path / "quarantined-dirty-candidate"
    prior.mkdir(parents=True)
    quarantine: dict[str, object] = {
        "schema": PCTDD_RETAINED_CANDIDATE_QUARANTINE_SCHEMA,
        "task_id": PCTDD_RETAINED_CANDIDATE_TASK_ID,
        "task_cid": PCTDD_RETAINED_CANDIDATE_TASK_CID,
        "board_namespace": task.board_namespace,
        "preserved_attempt_count": 3,
        "attempt_consumption_preserved": True,
        "worktree_path": str(prior),
        "branch": "implementation/pctdd-005-quarantined",
        "baseline_ref": "1" * 40,
        "candidate_head": "2" * 40,
        "candidate_diff_fingerprint": "sha256:" + "3" * 64,
        "finding_digest": "sha256:" + "4" * 64,
        "proposal_id": "proposal:p005",
        "proposal_receipt_id": "receipt:p005",
        "issuer_process_birth": issuer_birth,
        "issuer_session_id": "portal-session:issuer",
        "preserved_dirty": True,
        "merge_allowed": False,
        "sanitize_allowed": False,
    }
    quarantine["quarantine_id"] = daemon._retained_candidate_record_id(
        quarantine
    )
    credit: dict[str, object] = {
        "schema": PCTDD_RETAINED_CANDIDATE_RETRY_CREDIT_SCHEMA,
        "quarantine_id": quarantine["quarantine_id"],
        **{
            name: quarantine[name]
            for name in (
                "task_id",
                "task_cid",
                "board_namespace",
                "preserved_attempt_count",
                "worktree_path",
                "branch",
                "baseline_ref",
                "candidate_head",
                "candidate_diff_fingerprint",
                "finding_digest",
                "issuer_process_birth",
                "issuer_session_id",
            )
        },
        "attempt_consumption_preserved": True,
        "fresh_isolated_only": True,
        "one_shot": True,
    }
    credit["retry_credit_id"] = daemon._retained_candidate_record_id(credit)
    issuance = {
        "type": "dirty_worktree_candidate_quarantined_retry_issued",
        "task_id": task.task_id,
        "task_cid": PCTDD_RETAINED_CANDIDATE_TASK_CID,
        "quarantine": quarantine,
        "retry_credit": credit,
    }
    events.append(issuance)
    return daemon, task, events, issuance


def test_retained_candidate_credit_is_successor_only_and_one_shot(
    tmp_path: Path,
) -> None:
    successor, task, events, issuance = _p005_retry_daemon(tmp_path)
    assert successor._issued_retained_candidate_retry_credit(
        task,
        require_successor=True,
    ) is not None

    # A second otherwise-valid issuance is ambiguity, never another credit.
    events.append(dict(issuance))
    assert successor._issued_retained_candidate_retry_credit(
        task,
        require_successor=True,
    ) is None

    issuer, issuer_task, _issuer_events, _ = _p005_retry_daemon(
        tmp_path / "issuer",
        successor=False,
    )
    assert issuer._issued_retained_candidate_retry_credit(
        issuer_task,
        require_successor=True,
    ) is None


def test_retained_candidate_issuance_write_response_loss_is_idempotent(
    tmp_path: Path,
) -> None:
    daemon, task, events, issuance = _p005_retry_daemon(tmp_path)
    events.clear()
    result = {
        "task_id": task.task_id,
        "task_cid": PCTDD_RETAINED_CANDIDATE_TASK_CID,
        "quarantined": True,
        "quarantine": issuance["quarantine"],
        "retry_credit": issuance["retry_credit"],
    }

    def append_then_lose_response(
        event_type: str,
        payload: object,
    ) -> None:
        events.append({"type": event_type, **dict(payload)})
        raise OSError("injected durable append response loss")

    daemon._record_event = append_then_lose_response

    assert daemon._publish_retained_candidate_retry_issuance(
        task,
        result=result,
        quarantine=issuance["quarantine"],
        retry_credit=issuance["retry_credit"],
    ) is True
    assert result["retry_credit_write_response_lost"] is True
    assert result["retry_credit_event_recorded"] is True
    assert len(events) == 1
    assert daemon._issued_retained_candidate_retry_credit(
        task,
        require_successor=True,
    ) is not None


def test_retained_candidate_credit_forces_no_prior_attempt_seed_below_limit(
    tmp_path: Path,
) -> None:
    daemon, _task, _events, _issuance = _p005_retry_daemon(tmp_path)
    daemon._prior_attempt_seed_plan = lambda **_kwargs: pytest.fail(
        "quarantined prior attempt was seeded"
    )

    plan = daemon._implementation_seed_plan_for_attempt(
        state=PortalTaskState(),
        attempt=1,
        retry_no_change_probe_only=False,
        retained_candidate_retry_required=True,
    )

    assert dict(plan) == {"reuse_prior_attempt": False}


def test_cleanup_preserves_exact_quarantined_candidate_after_issuance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, _baseline, _divergent = _repo(tmp_path)
    supervisor, _events = _supervisor(repo, tmp_path)
    worktree_root = supervisor.config.worktree_root
    worktree_root.mkdir(parents=True)
    candidate = worktree_root / "pctdd-005-quarantined"
    branch = "implementation/pctdd-005-quarantined"
    _git(repo, "worktree", "add", "-qb", branch, str(candidate), "HEAD")
    head = _git(candidate, "rev-parse", "HEAD")
    (candidate / "tracked.txt").write_text(
        "rejected validation transport mutation\n",
        encoding="utf-8",
    )
    status_before = _git(
        candidate,
        "status",
        "--porcelain",
        "--untracked-files=all",
    )
    diff_before = _git(candidate, "diff", "--binary")

    retry_daemon, _task, _retry_events, issuance = _p005_retry_daemon(
        tmp_path / "credit"
    )
    quarantine = dict(issuance["quarantine"])
    quarantine.pop("quarantine_id")
    quarantine.update(
        {
            "worktree_path": str(candidate),
            "branch": branch,
            "baseline_ref": head,
            "candidate_head": head,
        }
    )
    quarantine["quarantine_id"] = (
        retry_daemon._retained_candidate_record_id(quarantine)
    )
    credit = {
        "schema": PCTDD_RETAINED_CANDIDATE_RETRY_CREDIT_SCHEMA,
        "quarantine_id": quarantine["quarantine_id"],
        **{
            name: quarantine[name]
            for name in (
                "task_id",
                "task_cid",
                "board_namespace",
                "preserved_attempt_count",
                "worktree_path",
                "branch",
                "baseline_ref",
                "candidate_head",
                "candidate_diff_fingerprint",
                "finding_digest",
                "issuer_process_birth",
                "issuer_session_id",
            )
        },
        "attempt_consumption_preserved": True,
        "fresh_isolated_only": True,
        "one_shot": True,
    }
    credit["retry_credit_id"] = (
        retry_daemon._retained_candidate_record_id(credit)
    )
    event = {
        "type": "dirty_worktree_candidate_quarantined_retry_issued",
        "task_id": PCTDD_RETAINED_CANDIDATE_TASK_ID,
        "task_cid": PCTDD_RETAINED_CANDIDATE_TASK_CID,
        "quarantine": quarantine,
        "retry_credit": credit,
    }
    event_path = (
        supervisor.config.state_dir
        / f"{supervisor.config.state_prefix}_events.jsonl"
    )
    event_path.parent.mkdir(parents=True, exist_ok=True)
    event_path.write_text(
        json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        supervisor,
        "_rescue_dirty_worktree",
        lambda *_args, **_kwargs: pytest.fail(
            "quarantined rejected delta entered generic rescue"
        ),
    )

    lock_path = supervisor._repo_merge_lock_path()
    lease, reason, _existing = supervisor._acquire_supervisor_checkout_lease(
        lock_path,
        supervisor._supervisor_checkout_lock_metadata(
            operation="cleanup_backlogged_worktrees"
        ),
    )
    assert lease is not None, reason
    try:
        with supervisor._supervisor_checkout_transaction(lease):
            result = supervisor._cleanup_backlogged_worktrees_locked(
                checkout_lease=lease,
            )
    finally:
        supervisor._release_supervisor_checkout_lease(
            lease,
            operation="cleanup_backlogged_worktrees",
        )

    held = [
        item
        for item in result["skipped"]
        if item.get("reason") == "retained_candidate_quarantine_hold"
    ]
    assert len(held) == 1
    assert held[0]["merge_allowed"] is False
    assert held[0]["sanitize_allowed"] is False
    assert candidate.is_dir()
    assert _git(candidate, "symbolic-ref", "--short", "HEAD") == branch
    assert _git(candidate, "rev-parse", "HEAD") == head
    assert _git(
        candidate,
        "status",
        "--porcelain",
        "--untracked-files=all",
    ) == status_before
    assert _git(candidate, "diff", "--binary") == diff_before

    # Consumption does not revoke archival preservation authority.
    event_path.write_text(
        event_path.read_text(encoding="utf-8")
        + json.dumps(
            {
                "type": "dirty_worktree_candidate_retry_credit_consumed",
                "retry_credit_id": credit["retry_credit_id"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    holds = supervisor._retained_candidate_cleanup_holds()
    assert str(candidate.resolve()) in holds


def test_retained_candidate_credit_is_consumed_before_launch_boundary_failure(
    tmp_path: Path,
) -> None:
    daemon, task, events, _issuance = _p005_retry_daemon(tmp_path)
    credit_state = daemon._issued_retained_candidate_retry_credit(
        task,
        require_successor=True,
    )
    assert credit_state is not None
    credit = dict(credit_state["retry_credit"])
    daemon._is_git_worktree = lambda _path: True
    daemon._git_current_branch = lambda _path: str(credit["branch"])
    daemon._resolved_commit_ref = lambda _path, _ref: str(
        credit["candidate_head"]
    )
    daemon._run_git = lambda *_args, **_kwargs: subprocess.CompletedProcess(
        args=("git", "status"),
        returncode=0,
        stdout=" M rejected_transport.py\n",
        stderr="",
    )
    daemon._proposal_scope_paths = lambda _task: ("rejected_transport.py",)
    daemon._collect_proposal_candidate_diff = lambda *_args, **_kwargs: (
        (SimpleNamespace(path="rejected_transport.py"),),
        (),
    )
    daemon._proposal_candidate_fingerprint = lambda _entries: str(
        credit["candidate_diff_fingerprint"]
    )
    launch_calls: list[str] = []

    def failed_launch_boundary(*_args: object, **_kwargs: object) -> None:
        launch_calls.append("boundary")
        raise RuntimeError("injected launch-boundary failure")

    daemon._mark_provider_launch_boundary = failed_launch_boundary
    fresh = tmp_path / "fresh-isolated"
    fresh.mkdir()

    with pytest.raises(RuntimeError, match="launch-boundary failure"):
        daemon._admit_provider_launch_with_retained_candidate_credit(
            task,
            credit_state=credit_state,
            attempt=4,
            worktree_path=fresh,
            branch_name="implementation/pctdd-005-fresh",
            baseline_ref="5" * 40,
            state=PortalTaskState(),
        )

    consumptions = [
        event
        for event in events
        if event.get("type")
        == "dirty_worktree_candidate_retry_credit_consumed"
    ]
    assert launch_calls == ["boundary"]
    assert len(consumptions) == 1
    consumption = consumptions[0]["consumption"]
    assert consumption["schema"] == PCTDD_RETAINED_CANDIDATE_RETRY_CONSUMPTION_SCHEMA
    assert (
        daemon._issued_retained_candidate_retry_credit(
            task,
            require_successor=True,
        )
        is None
    )

    # A successor/reentry sees the durable terminal consumption and cannot
    # cross the launch boundary a second time.
    with pytest.raises(ImplementationRetryDeferred):
        daemon._admit_provider_launch_with_retained_candidate_credit(
            task,
            credit_state=credit_state,
            attempt=4,
            worktree_path=fresh,
            branch_name="implementation/pctdd-005-fresh",
            baseline_ref="5" * 40,
            state=PortalTaskState(),
        )
    assert launch_calls == ["boundary"]
    assert sum(
        event.get("type")
        == "dirty_worktree_candidate_retry_credit_consumed"
        for event in events
    ) == 1


def _shared_fenced_provider_recovery_daemon(
    *,
    initial_state: str,
    response_loss_call: int | None,
    post_admission_drift: bool = False,
) -> tuple[
    daemon_module.DatabaseImplementationDaemon,
    SimpleNamespace,
    list[str],
    list[int],
]:
    daemon = object.__new__(daemon_module.DatabaseImplementationDaemon)
    attempt_record = {
        "attempt_id": "attempt:exact-pinned-provider",
        "task_cid": "task:exact-pinned-provider",
        "status": "blocked",
        "committed_phase": "provider",
    }
    attempt = SimpleNamespace(
        attempt_id=attempt_record["attempt_id"],
        to_dict=lambda: dict(attempt_record),
    )
    original = {
        "attempt_id": attempt.attempt_id,
        "claim_id": "claim:exact-pinned-provider",
    }
    evidence = {
        "schema": (
            bridge_module.
            DATABASE_PORTAL_FENCED_PROVIDER_UNPUBLISHED_REARM_EVIDENCE_SCHEMA
        ),
        "evidence_id": "sha256:" + "6" * 64,
        "migration_manifest_id": "sha256:" + "c" * 64,
        "migration_credit_id": "sha256:" + "d" * 64,
    }
    base_revision = 41
    revision = base_revision + (1 if initial_state == "admitting" else 0)
    fence = {
        "schema": "test/no-provider-rearm-fence@1",
        "saga_id": "sha256:" + "7" * 64,
        "saga_nonce": "no-provider-rearm:" + "8" * 24,
        "state": initial_state,
        "evidence_id": evidence["evidence_id"],
        "blocked_receipt_digest": "sha256:" + "9" * 64,
        "blocked_revision": 40,
        "retrying_revision": 41,
        "admitted_revision": 0 if initial_state == "pending" else 43,
        "immutable_receipt_digest": "sha256:" + "a" * 64,
    }
    receipt = {
        "schema": "test/retry-budget@1",
        "operation": daemon_module.DATABASE_UNKNOWN_OUTCOME_REARM_OPERATION,
        "no_provider_rearm_saga_id": fence["saga_id"],
        "no_provider_rearm_evidence_id": evidence["evidence_id"],
        "no_provider_rearm_evidence": evidence,
        "no_provider_rearm_original_block_receipt": original,
        "no_provider_rearm_fence": fence,
    }
    outer_snapshot: dict[str, object] = {
        "schema": daemon_module.DATABASE_FENCED_PROVIDER_OUTER_SNAPSHOT_SCHEMA,
        "attempt_record": attempt_record,
        "provider_dispatch": {"state": "fenced", "result": "unpublished"},
        "phase_history": [{"phase": "provider", "state": "fenced"}],
        "provider_invocation_absent": True,
        "effect_claim_absent": True,
        "effect_dispatch_absent": True,
        "recovery_dispatch_fence_id": "sha256:" + "b" * 64,
    }
    outer_snapshot["snapshot_id"] = (
        daemon._database_no_provider_rearm_digest(outer_snapshot)
    )
    receipt["fenced_provider_outer_attempt_snapshot"] = outer_snapshot
    current = SimpleNamespace(
        task_cid="task:exact-pinned-provider",
        task_alias="PCTDD-006",
        status="retrying" if initial_state == "pending" else "blocked",
        revision=revision,
        body={"completion_receipt": receipt},
    )

    class Source:
        def __init__(self, task: SimpleNamespace) -> None:
            self.task = task

        def list_tasks(self, *, limit: int) -> SimpleNamespace:
            assert limit > 0
            return SimpleNamespace(tasks=(self.task,))

        def get(self, task_cid: str) -> SimpleNamespace | None:
            return self.task if task_cid == self.task.task_cid else None

    source = Source(current)
    daemon._task_source = source
    daemon.open = lambda: daemon
    daemon.get_attempt = lambda attempt_id: (
        attempt if attempt_id == attempt.attempt_id else None
    )
    daemon._dispatch_journal_entry = lambda *_args, **kwargs: (
        {"state": "fenced", "result": "unpublished"}
        if kwargs.get("dispatch_kind") == "provider"
        else None
    )
    daemon.phase_history = lambda _attempt_id: [
        {"phase": "provider", "state": "fenced"}
    ]
    daemon.provider_invocation_recorded = lambda *_args, **_kwargs: None
    daemon.effect_claim_recorded = lambda *_args, **_kwargs: None
    dispatch_fence_state = {"value": "sealed"}

    def dispatch_fence(_attempt: object) -> dict[str, object]:
        return {
            "fence_id": outer_snapshot["recovery_dispatch_fence_id"],
            "snapshot_id": outer_snapshot["snapshot_id"],
            "state": dispatch_fence_state["value"],
            "evidence_id": evidence["evidence_id"],
            "migration_manifest_id": evidence["migration_manifest_id"],
            "migration_credit_id": evidence["migration_credit_id"],
            "attempt_id": original["attempt_id"],
            "task_cid": current.task_cid,
        }

    def transition_dispatch_fence(
        _attempt: object,
        *,
        expected_state: str,
        new_state: str,
    ) -> dict[str, object]:
        assert dispatch_fence_state["value"] == expected_state
        dispatch_fence_state["value"] = new_state
        return dispatch_fence(_attempt)

    daemon._fenced_provider_recovery_dispatch_fence = dispatch_fence
    daemon._transition_fenced_provider_recovery_dispatch_fence = (
        transition_dispatch_fence
    )
    revalidations: list[str] = []
    artifact_drift = {"value": False}

    def revalidate(
        selected_attempt: object,
        *,
        outer_block_receipt: object,
        expected_evidence: object,
    ) -> bool:
        assert selected_attempt is attempt
        assert outer_block_receipt == original
        assert expected_evidence == evidence
        revalidations.append("revalidated")
        return not artifact_drift["value"]

    def execute(
        selected_attempt: object,
        *,
        outer_block_receipt: object,
        expected_evidence: object,
        callback: object,
    ) -> object:
        assert revalidate(
            selected_attempt,
            outer_block_receipt=outer_block_receipt,
            expected_evidence=expected_evidence,
        )
        result = callback()
        if post_admission_drift and dispatch_fence_state["value"] == "admitted":
            artifact_drift["value"] = True
        return result

    daemon._database_portal_bridge = SimpleNamespace(
        fenced_provider_unpublished_migration_available=(
            lambda selected_attempt, *, outer_block_receipt: (
                selected_attempt is attempt
                and outer_block_receipt == original
            )
        ),
        revalidate_fenced_provider_unpublished_rearm_evidence=revalidate,
        execute_with_revalidated_fenced_provider_unpublished=execute,
    )
    daemon._no_provider_rearm_fence_state = lambda task: str(
        task.body["completion_receipt"]["no_provider_rearm_fence"]["state"]
    )
    cas_calls: list[int] = []

    def cas(
        task_cid: str,
        *,
        expected_revision: int,
        new_status: str,
        receipt: object,
    ) -> SimpleNamespace:
        assert task_cid == source.task.task_cid
        assert expected_revision == source.task.revision
        cas_calls.append(expected_revision)
        source.task = SimpleNamespace(
            task_cid=source.task.task_cid,
            task_alias=source.task.task_alias,
            status=new_status,
            revision=expected_revision + 1,
            body={"completion_receipt": dict(receipt)},
        )
        result = SimpleNamespace(task=source.task)
        if response_loss_call == len(cas_calls):
            raise RuntimeError("injected committed CAS response loss")
        return result

    daemon._cas_task_status_database = cas
    daemon._shared_no_provider_rearm_compensation_receipt = (
        lambda *_args, **_kwargs: pytest.fail(
            "exact manifest occurrence entered generic compensation"
        )
    )
    daemon._test_recovery_dispatch_fence_state = dispatch_fence_state
    daemon._test_recovery_artifact_drift = artifact_drift
    return daemon, source, revalidations, cas_calls


@pytest.mark.parametrize(
    ("initial_state", "response_loss_call"),
    (("pending", 1), ("admitting", 1), ("pending", 2)),
    ids=("pending-response-loss", "admitting-response-loss", "admitted-response-loss"),
)
def test_exact_fenced_provider_shared_recovery_moves_forward_once(
    initial_state: str,
    response_loss_call: int,
) -> None:
    daemon, source, revalidations, cas_calls = (
        _shared_fenced_provider_recovery_daemon(
            initial_state=initial_state,
            response_loss_call=response_loss_call,
        )
    )
    claims: list[str] = []

    def claim_once() -> str | None:
        state = source.task.body["completion_receipt"][
            "no_provider_rearm_fence"
        ]["state"]
        if state != "admitted" or claims:
            return None
        claims.append("claimed")
        return "claimed"

    # Pending/admitting are durable no-dispatch fences.
    assert claim_once() is None

    outcomes = daemon._reconcile_shared_no_provider_rearm_fences()

    assert len(outcomes) == 1
    assert outcomes[0]["rearm_committed"] is True
    assert outcomes[0]["recovered_forward"] is True
    final_receipt = source.task.body["completion_receipt"]
    assert final_receipt["no_provider_rearm_fence"]["state"] == "admitted"
    assert "no_provider_rearm_compensation" not in final_receipt
    assert len(cas_calls) == (2 if initial_state == "pending" else 1)
    assert len(revalidations) >= len(cas_calls) * 2
    assert claim_once() == "claimed"
    assert claim_once() is None
    assert claims == ["claimed"]


def test_post_admission_drift_cannot_revoke_irrevocable_one_shot() -> None:
    daemon, source, _revalidations, cas_calls = (
        _shared_fenced_provider_recovery_daemon(
            initial_state="admitting",
            response_loss_call=None,
            post_admission_drift=True,
        )
    )

    recovery = daemon._recover_shared_fenced_provider_rearm(source.task)

    assert recovery == {
        "rearmed": True,
        "rearm_committed": True,
        "recovered_forward": True,
        "admitted_revision": 43,
    }
    assert len(cas_calls) == 1
    # The bridge's callback-last return is the linearization point.  Mutable
    # nested/ref evidence changes after it returns cannot revoke or revive the
    # exact one-shot authorization.
    assert source.task.status == "retrying"
    assert source.task.body["completion_receipt"][
        "no_provider_rearm_fence"
    ]["state"] == "admitted"
    assert daemon._test_recovery_dispatch_fence_state["value"] == "admitted"
    assert daemon._test_recovery_artifact_drift["value"] is True
    assert daemon._automatic_claim_forbidden_current(source.task) is False

    # Restoring the mutable state cannot create a second authorization either.
    daemon._test_recovery_artifact_drift["value"] = False
    assert daemon._automatic_claim_forbidden_current(source.task) is False
    assert daemon._recover_shared_fenced_provider_rearm(source.task) == {
        "rearmed": True,
        "rearm_committed": True,
        "recovered_forward": True,
        "admitted_revision": 43,
        "recovery_required": False,
    }


def test_fenced_provider_manifest_near_miss_remains_unmodified_and_unclaimable(
) -> None:
    daemon, source, _revalidations, cas_calls = (
        _shared_fenced_provider_recovery_daemon(
            initial_state="pending",
            response_loss_call=None,
        )
    )
    original_receipt = json.loads(
        json.dumps(source.task.body["completion_receipt"], sort_keys=True)
    )
    daemon._database_portal_bridge.fenced_provider_unpublished_migration_available = (
        lambda *_args, **_kwargs: False
    )

    outcomes = daemon._reconcile_shared_no_provider_rearm_fences()

    assert outcomes == [
        {
            "task_cid": source.task.task_cid,
            "task_alias": source.task.task_alias,
            "operation": "database_no_provider_rearm_shared_fence_recovery",
            "rearmed": False,
            "saga_id": "sha256:" + "7" * 64,
            "prior_fence_state": "pending",
            "recovery_required": True,
        }
    ]
    assert source.task.body["completion_receipt"] == original_receipt
    assert cas_calls == []


@pytest.mark.parametrize(
    "drift",
    ("phase", "provider_dispatch", "provider_result", "effect_claim", "effect_dispatch"),
)
def test_fenced_provider_outer_state_drift_never_advances_credit(
    drift: str,
) -> None:
    daemon, source, _revalidations, cas_calls = (
        _shared_fenced_provider_recovery_daemon(
            initial_state="pending",
            response_loss_call=None,
        )
    )
    if drift == "phase":
        daemon.phase_history = lambda _attempt_id: [
            {"phase": "provider", "state": "returned-late"}
        ]
    elif drift == "provider_dispatch":
        daemon._dispatch_journal_entry = lambda *_args, **kwargs: (
            {"state": "returned", "result": "late"}
            if kwargs.get("dispatch_kind") == "provider"
            else None
        )
    elif drift == "provider_result":
        daemon.provider_invocation_recorded = (
            lambda *_args, **_kwargs: {"status": "late"}
        )
    elif drift == "effect_claim":
        daemon.effect_claim_recorded = (
            lambda *_args, **_kwargs: {"status": "late"}
        )
    else:
        daemon._dispatch_journal_entry = lambda *_args, **kwargs: (
            {"state": "returned", "result": "late"}
            if kwargs.get("dispatch_kind") == "effect"
            else {"state": "fenced", "result": "unpublished"}
        )

    outcomes = daemon._reconcile_shared_no_provider_rearm_fences()

    assert outcomes[0]["recovery_required"] is True
    assert source.task.body["completion_receipt"][
        "no_provider_rearm_fence"
    ]["state"] == "pending"
    assert cas_calls == []


def test_exact_manifest_candidate_bypasses_mutating_terminal_landed_probe(
) -> None:
    daemon = object.__new__(daemon_module.DatabaseImplementationDaemon)
    attempt = SimpleNamespace(attempt_id="attempt:pctdd-034-9")
    receipt = {
        "schema": daemon_module.DATABASE_RETRY_BUDGET_SCHEMA,
        "operation": "database_unknown_outcome_blocked",
        "reason": "callback_authority_incomplete_blocked",
        "forced_block": True,
        "authority_outcome": "unknown",
        "retry_exhausted": True,
        "attempt_id": attempt.attempt_id,
        "attempt_number": 9,
        "attempts_used": 1,
        "unknown_outcome_rearm_count": 0,
        "terminal_reconciliation": {"evidence_id": "exact-link"},
    }
    task = SimpleNamespace(
        task_cid="task:pctdd-034-exact",
        task_alias="PCTDD-034",
        body={"completion_receipt": receipt},
    )
    daemon.task_prefix = "PCTDD-"
    daemon.strict_task_sharding = False
    daemon.task_shard_count = 1
    daemon._task_source = SimpleNamespace(
        list_tasks=lambda **_kwargs: SimpleNamespace(tasks=(task,))
    )
    daemon.open = lambda: daemon
    daemon.get_attempt = lambda attempt_id: (
        attempt if attempt_id == attempt.attempt_id else None
    )
    probe_calls: list[str] = []
    daemon._database_portal_bridge = SimpleNamespace(
        fenced_provider_unpublished_migration_available=(
            lambda selected, *, outer_block_receipt: (
                selected is attempt and outer_block_receipt == receipt
            )
        ),
        reconcile_quiesced_attempt=lambda *_args, **_kwargs: (
            probe_calls.append("mutated")
        ),
    )

    assert daemon.reconcile_blocked_terminal_landed_tasks() == []
    assert daemon.reconcile_blocked_terminal_landed_tasks() == []
    assert probe_calls == []
