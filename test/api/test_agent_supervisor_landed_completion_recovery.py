"""End-to-end proof for database-authoritative landed completion recovery."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.runtime.event_log import (
    append_jsonl_event,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
    duckdb_available,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
    QUACK_OWNER_MUTATION_MAX_PARAMETER_BYTES,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    landed_completion_recovery as landed_recovery,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import (
    DatabasePortalBridgeError,
    DatabasePortalExecutionBridge,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    ATTEMPT_PHASE_COMPLETE,
    DatabaseImplementationDaemon,
    DatabaseTaskAttempt,
    PortalImplementationDaemon,
    PortalTaskState,
    parse_task_file,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.landed_completion_recovery import (
    DATABASE_LANDED_COMPLETION_CLAIM_SEED_SCHEMA,
    DATABASE_LANDED_COMPLETION_RECOVERY_SCHEMA,
    LandedCompletionRecoveryError,
    build_landed_completion_claim_seed,
    discover_landed_completion_recovery,
    revalidate_landed_completion_repository,
    verify_landed_completion_claim_seed,
    verify_landed_completion_recovery_receipt,
)

pytestmark = pytest.mark.skipif(
    not duckdb_available(),
    reason="DuckDB is required for database landed-completion recovery",
)


def _git(repo: Path, *argv: str) -> str:
    completed = subprocess.run(
        ["git", *argv],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _landed_repository(
    root: Path,
    *,
    task_alias: str,
) -> tuple[str, str, str, tuple[str, ...]]:
    root.mkdir(parents=True)
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "landed-recovery@example.invalid")
    _git(root, "config", "user.name", "Landed Recovery Test")
    (root / "README.md").write_text("base\n", encoding="utf-8")
    _git(root, "add", "--", "README.md")
    _git(root, "commit", "-qm", "base")
    _git(root, "branch", "-M", "main")

    rescue_branch = "rescue/recovery-task"
    _git(root, "checkout", "-qb", rescue_branch)
    output = root / "src" / "recovered.py"
    validation = root / "tests" / "check_recovered.py"
    output.parent.mkdir(parents=True)
    validation.parent.mkdir(parents=True)
    output.write_text("VALUE = 1\n", encoding="utf-8")
    validation.write_text(
        "from pathlib import Path\n"
        "assert Path('src/recovered.py').read_text() == 'VALUE = 1\\n'\n",
        encoding="utf-8",
    )
    outputs = ("src/recovered.py", "tests/check_recovered.py")
    _git(root, "add", "--", *outputs)
    _git(root, "commit", "-qm", f"rescue {task_alias.lower()} landed output")
    candidate = _git(root, "rev-parse", "HEAD")

    _git(root, "checkout", "-q", "main")
    _git(
        root,
        "merge",
        "--no-ff",
        "-m",
        "integrate landed recovery candidate",
        rescue_branch,
    )
    integration = _git(root, "rev-parse", "HEAD")
    tree = _git(root, "rev-parse", "HEAD^{tree}")
    return candidate, integration, tree, outputs


class _FreshValidationCompletingPortal:
    """Use Portal's real uncached no-change gate, then publish completion."""

    def __init__(
        self,
        paths: Any,
        task_alias: str,
        *,
        repository: Path,
        observations: dict[str, Any],
    ) -> None:
        self.paths = paths
        self.task_alias = task_alias
        self.repository = repository
        self.observations = observations
        self.portal: PortalImplementationDaemon | None = None

    def run_once(self) -> dict[str, object]:
        self.paths.implementation_logs.mkdir(parents=True, exist_ok=True)
        self.portal = PortalImplementationDaemon(
            todo_path=self.paths.task_projection,
            state_path=self.paths.state,
            strategy_path=self.paths.strategy,
            events_path=self.paths.events,
            repo_root=self.repository,
            implement=False,
            implementation_log_dir=self.paths.implementation_logs,
            validation_cache_dir=self.paths.root / "validation-cache",
            use_ephemeral_worktree=False,
            task_header_prefix="REC-",
        )
        tasks = parse_task_file(
            self.paths.task_projection,
            task_header_prefix="REC-",
        )
        assert len(tasks) == 1
        task = tasks[0]
        self.observations["projection_claim_seed"] = json.loads(
            str(task.metadata["landed completion recovery"])
        )
        state = PortalTaskState.load(self.paths.state)
        baseline = _git(self.repository, "rev-parse", "HEAD")

        force_uncached_values: list[bool] = []
        original = self.portal._run_validation_commands

        def record_force_uncached(*args: Any, **kwargs: Any) -> dict[str, Any]:
            force_uncached_values.append(kwargs.get("force_uncached") is True)
            return original(*args, **kwargs)

        self.portal._run_validation_commands = record_force_uncached  # type: ignore[method-assign]
        validation = self.portal._run_retry_no_change_pre_dispatch_validation(
            self.repository,
            task,
            self.paths.implementation_logs / "landed-validation.log",
            state=state,
            attempt=1,
            baseline_ref=baseline,
            protected_path_snapshot=None,
            branch_name="main",
            prepare_workspace=False,
        )
        assert validation is not None
        policy = validation.get("no_change_policy_gate")
        binding = validation.get("candidate_binding")
        bypass = validation.get("pre_dispatch_no_change")
        assert validation.get("attempted") is True
        assert validation.get("passed") is True
        assert isinstance(policy, dict) and policy.get("accepted") is True
        assert isinstance(binding, dict) and binding.get("verified") is True
        assert isinstance(bypass, dict)
        assert bypass.get("provider_dispatched") is False
        assert force_uncached_values == [True]
        self.observations["validation"] = validation
        self.observations["force_uncached"] = tuple(force_uncached_values)

        state.last_implementation_task_id = self.task_alias
        state.last_implementation_task_cid = task.canonical_task_cid
        state.last_implementation_returncode = 0
        state.save(self.paths.state)
        projection = self.paths.task_projection.read_text(encoding="utf-8")
        self.paths.task_projection.write_text(
            projection.replace("- Status: ready", "- Status: completed"),
            encoding="utf-8",
        )
        append_jsonl_event(
            self.paths.events,
            "task_completed",
            {
                "task_id": self.task_alias,
                "canonical_task_cid": task.canonical_task_cid,
                "completion_authoritative": False,
                "source": "fresh_landed_completion_validation",
            },
        )
        return {
            "task_count": 1,
            "completed_count": 1,
            "active_task_id": self.task_alias,
            "implementation_result": {
                "task_id": self.task_alias,
                "attempt": 1,
                "returncode": 0,
                "provider_dispatched": False,
                "reason": "landed_candidate_freshly_validated",
            },
        }

    def close_event_runtime(self) -> None:
        if self.portal is not None:
            self.portal.close_event_runtime()


def test_blocked_landed_output_completes_only_after_new_claim_validation(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    task_alias = "REC-001"
    task_cid = "task:cid:landed-recovery"
    candidate, integration, target_tree, outputs = _landed_repository(
        repository,
        task_alias=task_alias,
    )
    observations: dict[str, Any] = {}
    holder: dict[str, Any] = {}
    provider_attempts: list[DatabaseTaskAttempt] = []
    now = {"ms": 1_000}

    def provider(attempt: DatabaseTaskAttempt) -> dict[str, Any]:
        provider_attempts.append(attempt)
        if len(provider_attempts) == 1:
            raise DatabasePortalBridgeError("portal_provider_failed")
        task = holder["daemon"].task_source.get(attempt.task_cid)
        assert task is not None
        receipt = dict(task.body["completion_receipt"])
        observations["database_claim_receipt"] = receipt
        observations["claim_receipt_bytes"] = len(
            json.dumps(
                receipt,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        return dict(holder["bridge"].run_provider(attempt))

    def effect(
        attempt: DatabaseTaskAttempt,
        provider_result: dict[str, Any],
    ) -> dict[str, Any]:
        return dict(holder["bridge"].apply_effect(attempt, provider_result))

    def validation(
        attempt: DatabaseTaskAttempt,
        effect_result: dict[str, Any],
    ) -> dict[str, Any]:
        return dict(holder["bridge"].validate_effect(attempt, effect_result))

    daemon = DatabaseImplementationDaemon(
        database_path=tmp_path / "control.duckdb",
        coordination_path=tmp_path / "coordination.duckdb",
        execution_path=tmp_path / "execution.duckdb",
        owner_session_id="session:landed-recovery",
        authority_mode="embedded",
        task_source_kind="duckdb",
        lease_ms=5_000,
        max_task_attempts=1,
        provider_fn=provider,
        effect_fn=effect,
        validation_fn=validation,
        landed_completion_recovery_fn=(
            lambda attempt: holder["bridge"].recover_landed_completion(attempt)
        ),
        require_real_execution=True,
        clock_ms=lambda: now["ms"],
    )
    holder["daemon"] = daemon

    def portal_factory(paths: Any, alias: str) -> _FreshValidationCompletingPortal:
        return _FreshValidationCompletingPortal(
            paths,
            alias,
            repository=repository,
            observations=observations,
        )

    bridge = DatabasePortalExecutionBridge(
        task_source=daemon.task_source,
        attempt_root=tmp_path / "portal-attempts",
        portal_factory=portal_factory,
        repository_root=repository,
        merge_target_ref="main",
        task_header_prefix="REC-",
        max_passes=1,
        max_task_attempts=1,
    )
    holder["bridge"] = bridge

    try:
        daemon.materialize_population(
            {
                "repository_tree_id": "git-tree:" + target_tree,
                "objectives": [
                    {
                        "objective_id": "objective:landed-recovery",
                        "objective_alias": "REC-O001",
                        "title": "Recover landed output",
                        "goal_cid": "goal:landed-recovery",
                        "goal_alias": "REC-G001",
                        "status": "open",
                    }
                ],
                "tasks": [
                    {
                        "task_cid": task_cid,
                        "task_id": task_alias,
                        "goal_cid": "goal:landed-recovery",
                        "status": "ready",
                        "priority": "P0",
                        "ordinal": 1,
                        "title": "Freshly validate an already-landed output",
                        "completion": "auto",
                        "outputs": [{"path": path} for path in outputs],
                        "validations": [
                            {"argv": ["python -B tests/check_recovered.py"]}
                        ],
                        "acceptance": [
                            {"criterion": "Declared validation passes uncached"}
                        ],
                    }
                ],
            },
            repository_tree_id="git-tree:" + target_tree,
        )

        first = daemon.run_once()
        source_attempt = daemon.get_attempt(first["attempt_id"])
        assert source_attempt is not None
        assert source_attempt.status == "failed"
        assert source_attempt.attempt_number == 1
        blocked = daemon.task_source.get(task_cid)
        assert blocked is not None and blocked.status == "blocked"
        blocked_receipt = blocked.body["completion_receipt"]
        assert blocked_receipt["operation"] == "database_portal_terminal_failure"

        recovery_pass = daemon.run_once()
        recoveries = recovery_pass["terminal_portal_reconciliations"]
        assert len(recoveries) == 1
        recovery = recoveries[0]
        assert recovery["status"] == "retrying"
        proof = recovery["landed_completion_recovery_evidence"]
        assert proof["schema"] == DATABASE_LANDED_COMPLETION_RECOVERY_SCHEMA
        assert proof["source_attempt_id"] == source_attempt.attempt_id
        assert proof["candidate_commit"] == candidate
        assert proof["integrating_merge"] == integration
        assert proof["declared_outputs"] == list(outputs)
        assert (
            revalidate_landed_completion_repository(
                proof,
                repo_root=repository,
                target_ref="main",
            )["current_target_tree"]
            == target_tree
        )
        retrying = daemon.task_source.get(task_cid)
        assert retrying is not None and retrying.status == "retrying"
        retrying_receipt = retrying.body["completion_receipt"]
        assert (
            retrying_receipt["operation"]
            == "database_portal_landed_completion_revalidation"
        )

        # The control-owner retry CAS cannot revoke a still-live coordination
        # lease.  Once that exact source fence expires, the next pass verifies
        # the retry projection and acquires a strictly newer claim.
        now["ms"] = 6_001
        second = (
            recovery_pass
            if recovery_pass["implementation_result"] is not None
            else daemon.run_once()
        )
        assert second["implementation_result"] is not None, second
        assert second["implementation_result"]["status"] == "succeeded"

        assert len(provider_attempts) == 2
        target_attempt = daemon.get_attempt(second["attempt_id"])
        assert target_attempt is not None
        assert target_attempt.attempt_number == 2
        assert target_attempt.status == "succeeded"
        assert target_attempt.committed_phase == ATTEMPT_PHASE_COMPLETE
        assert target_attempt.owner_session_id == daemon.owner_session_id
        assert target_attempt.fencing_token > source_attempt.fencing_token
        assert target_attempt.fence_epoch >= source_attempt.fence_epoch
        claim_receipt = observations["database_claim_receipt"]
        assert claim_receipt["operation"] == "database_claim"
        assert claim_receipt["owner_session_id"] == target_attempt.owner_session_id
        claim_recovery = claim_receipt["landed_completion_recovery_seed"]
        assert claim_recovery == proof
        claim_seed = observations["projection_claim_seed"]
        assert claim_seed["schema"] == DATABASE_LANDED_COMPLETION_CLAIM_SEED_SCHEMA
        assert claim_seed["recovery_receipt"] == proof
        assert claim_seed["target_attempt_id"] == target_attempt.attempt_id
        assert claim_seed["target_claim_id"] == target_attempt.claim_id
        assert claim_seed["target_attempt_number"] == target_attempt.attempt_number
        assert claim_seed["target_owner_session_id"] == target_attempt.owner_session_id
        assert claim_seed["target_fencing_token"] == target_attempt.fencing_token
        assert claim_seed["target_fence_epoch"] == target_attempt.fence_epoch
        assert claim_seed["target_lease_id"] == target_attempt.lease_id
        assert claim_seed["validated_target_commit"] == _git(
            repository, "rev-parse", "main"
        )
        assert observations["claim_receipt_bytes"] < (
            QUACK_OWNER_MUTATION_MAX_PARAMETER_BYTES
        )
        assert observations["force_uncached"] == (True,)
        assert (
            observations["validation"]["pre_dispatch_no_change"]["provider_dispatched"]
            is False
        )

        completed = daemon.task_source.get(task_cid)
        assert completed is not None and completed.status == "completed"
        completion_receipt = completed.body["completion_receipt"]
        assert completion_receipt["operation"] == "database_complete"
        prepared = daemon.coordinator.get_prepared_task_completion(task_cid)
        assert prepared is not None
        assert prepared["attempt_id"] == target_attempt.attempt_id
        assert prepared["claim_id"] == target_attempt.claim_id
    finally:
        daemon.close()


def test_landed_recovery_revalidation_rejects_a_different_target_ref(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    task_alias = "REC-002"
    _candidate, _integration, _tree, outputs = _landed_repository(
        repository,
        task_alias=task_alias,
    )
    # The end-to-end test above proves the valid path.  This regression keeps
    # a caller from replaying the same content-addressed proof through an
    # arbitrary ref instead of the bridge's configured merge target.
    proof = discover_landed_completion_recovery(
        repo_root=repository,
        target_ref="main",
        task_cid="task:cid:target-ref",
        task_alias=task_alias,
        declared_outputs=outputs,
        source_attempt_id="attempt:source",
        source_claim_id="claim:source",
        source_lease_id="lease:source",
        source_owner_session_id="session:source",
        source_attempt_number=1,
        source_fencing_token=1,
        source_fence_epoch=0,
        source_execution_revision=2,
        source_execution_finished_at_ms=3,
        source_control_revision=4,
    )
    assert proof is not None
    with pytest.raises(
        RuntimeError,
        match="configured merge target",
    ):
        revalidate_landed_completion_repository(
            proof,
            repo_root=repository,
            target_ref="HEAD",
        )


def test_landed_claim_seed_requires_new_fence_and_exact_owner(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    task_alias = "REC-003"
    _candidate, _integration, target_tree, outputs = _landed_repository(
        repository,
        task_alias=task_alias,
    )
    proof = discover_landed_completion_recovery(
        repo_root=repository,
        target_ref="main",
        task_cid="task:cid:claim-identity",
        task_alias=task_alias,
        declared_outputs=outputs,
        source_attempt_id="attempt:source",
        source_claim_id="claim:source",
        source_lease_id="lease:source",
        source_owner_session_id="session:source",
        source_attempt_number=4,
        source_fencing_token=7,
        source_fence_epoch=2,
        source_execution_revision=8,
        source_execution_finished_at_ms=9,
        source_control_revision=10,
    )
    assert proof is not None
    seed_kwargs = {
        "target_task_cid": "task:cid:claim-identity",
        "target_task_alias": task_alias,
        "target_attempt_id": "attempt:target",
        "target_claim_id": "claim:target",
        "target_owner_session_id": "session:target",
        "target_attempt_number": 5,
        "target_fencing_token": 8,
        "target_fence_epoch": 2,
        "target_lease_id": "lease:target",
        "validated_target_commit": _git(repository, "rev-parse", "main"),
        "validated_target_tree": target_tree,
    }
    seed = build_landed_completion_claim_seed(proof, **seed_kwargs)
    assert (
        verify_landed_completion_claim_seed(
            seed,
            target_owner_session_id="session:target",
        )
        == seed
    )
    with pytest.raises(LandedCompletionRecoveryError, match="identity"):
        verify_landed_completion_claim_seed(
            seed,
            target_owner_session_id="session:foreign",
        )
    with pytest.raises(LandedCompletionRecoveryError, match="identity"):
        build_landed_completion_claim_seed(
            proof,
            **{**seed_kwargs, "target_fencing_token": 7},
        )
    with pytest.raises(LandedCompletionRecoveryError, match="identity"):
        build_landed_completion_claim_seed(
            proof,
            **{**seed_kwargs, "target_fence_epoch": 1},
        )


def test_landed_recovery_receipt_rejects_safe_outputs_over_serialized_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repo"
    task_alias = "REC-004"
    _candidate, _integration, target_tree, outputs = _landed_repository(
        repository,
        task_alias=task_alias,
    )
    proof = discover_landed_completion_recovery(
        repo_root=repository,
        target_ref="main",
        task_cid="task:cid:receipt-bound",
        task_alias=task_alias,
        declared_outputs=outputs,
        source_attempt_id="attempt:source",
        source_claim_id="claim:source",
        source_lease_id="lease:source",
        source_owner_session_id="session:source",
        source_attempt_number=1,
        source_fencing_token=1,
        source_fence_epoch=0,
        source_execution_revision=2,
        source_execution_finished_at_ms=3,
        source_control_revision=4,
    )
    assert proof is not None

    oversized = dict(proof)
    oversized["declared_outputs"] = [
        f"generated/{ordinal:03d}-{'x' * 120}.py" for ordinal in range(256)
    ]
    unsigned = dict(oversized)
    unsigned.pop("proof_id")
    oversized["proof_id"] = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(
                unsigned,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
    )
    assert (
        len(
            json.dumps(
                oversized,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        )
        > landed_recovery._MAX_RECOVERY_RECEIPT_BYTES
    )

    with pytest.raises(
        LandedCompletionRecoveryError,
        match="conservative serialized byte bound",
    ):
        verify_landed_completion_recovery_receipt(oversized)

    seed_kwargs = {
        "target_task_cid": "task:cid:receipt-bound",
        "target_task_alias": task_alias,
        "target_attempt_id": "attempt:target",
        "target_claim_id": "claim:target",
        "target_owner_session_id": "session:target",
        "target_attempt_number": 2,
        "target_fencing_token": 2,
        "target_fence_epoch": 0,
        "target_lease_id": "lease:target",
        "validated_target_commit": _git(repository, "rev-parse", "main"),
        "validated_target_tree": target_tree,
    }
    seed = build_landed_completion_claim_seed(proof, **seed_kwargs)
    seed_bytes = len(
        json.dumps(
            seed,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )
    monkeypatch.setattr(
        landed_recovery,
        "_MAX_CLAIM_SEED_BYTES",
        seed_bytes - 1,
    )
    with pytest.raises(
        LandedCompletionRecoveryError,
        match="conservative serialized byte bound",
    ):
        build_landed_completion_claim_seed(proof, **seed_kwargs)


def test_landed_recovery_fails_closed_when_alias_history_is_truncated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repo"
    task_alias = "REC-005"
    _candidate, _integration, _tree, outputs = _landed_repository(
        repository,
        task_alias=task_alias,
    )
    (repository / "alias-noise.txt").write_text("noise\n", encoding="utf-8")
    _git(repository, "add", "--", "alias-noise.txt")
    _git(repository, "commit", "-qm", f"unrelated follow-up for {task_alias}")
    monkeypatch.setattr(landed_recovery, "_MAX_ALIAS_CANDIDATES", 1)

    assert (
        discover_landed_completion_recovery(
            repo_root=repository,
            target_ref="main",
            task_cid="task:cid:alias-truncation",
            task_alias=task_alias,
            declared_outputs=outputs,
            source_attempt_id="attempt:source",
            source_claim_id="claim:source",
            source_lease_id="lease:source",
            source_owner_session_id="session:source",
            source_attempt_number=1,
            source_fencing_token=1,
            source_fence_epoch=0,
            source_execution_revision=2,
            source_execution_finished_at_ms=3,
            source_control_revision=4,
        )
        is None
    )


def test_landed_recovery_fails_closed_when_merge_history_is_truncated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repo"
    task_alias = "REC-006"
    _candidate, _integration, _tree, outputs = _landed_repository(
        repository,
        task_alias=task_alias,
    )
    _git(repository, "checkout", "-qb", "unrelated-merge")
    (repository / "merge-noise.txt").write_text("noise\n", encoding="utf-8")
    _git(repository, "add", "--", "merge-noise.txt")
    _git(repository, "commit", "-qm", "unrelated side branch")
    _git(repository, "checkout", "-q", "main")
    _git(
        repository,
        "merge",
        "--no-ff",
        "-m",
        "integrate unrelated side branch",
        "unrelated-merge",
    )
    monkeypatch.setattr(landed_recovery, "_MAX_MERGES", 1)

    assert (
        discover_landed_completion_recovery(
            repo_root=repository,
            target_ref="main",
            task_cid="task:cid:merge-truncation",
            task_alias=task_alias,
            declared_outputs=outputs,
            source_attempt_id="attempt:source",
            source_claim_id="claim:source",
            source_lease_id="lease:source",
            source_owner_session_id="session:source",
            source_attempt_number=1,
            source_fencing_token=1,
            source_fence_epoch=0,
            source_execution_revision=2,
            source_execution_finished_at_ms=3,
            source_control_revision=4,
        )
        is None
    )
