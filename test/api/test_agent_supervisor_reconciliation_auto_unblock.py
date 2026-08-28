from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import (
    PREPARED_COMPLETION_STATUS,
    TASK_COMPLETION_PREPARATION_SCHEMA,
    DatabaseCoordinator,
)
from ipfs_accelerate_py.agent_supervisor.objectives.backlog_refinery import (
    reconciliation_guardrail_records,
    resolved_reconciliation_guardrail_keys,
)
from ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner import (
    DatabaseProgramConfig,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import (
    DatabasePortalExecutionBridge,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    DatabaseTaskAttempt,
    PortalTask,
    TodoImplementationDaemon,
    TodoTaskState,
    parse_task_file,
    portal_task_identity,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
    TodoImplementationSupervisor,
    TodoSupervisorConfig,
)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"git {' '.join(args)} failed in {repo}:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    return result.stdout.strip()


def _init_repo(path: Path) -> Path:
    path.mkdir()
    _git(path, "init")
    _git(path, "checkout", "-b", "main")
    _git(path, "config", "user.name", "Auto Unblock Test")
    _git(path, "config", "user.email", "auto-unblock@example.invalid")
    return path


def _supervisor(
    repo: Path,
    *,
    worktree_root: Path,
    worktree_submodule_paths: tuple[str, ...] = (),
    todo_text: str = "# Todos\n",
    database_program: DatabaseProgramConfig | None = None,
) -> TodoImplementationSupervisor:
    todo_path = repo / "todo.md"
    todo_path.write_text(todo_text, encoding="utf-8")
    state_dir = repo / "state"
    return TodoImplementationSupervisor(
        TodoSupervisorConfig(
            todo_path=todo_path,
            state_path=state_dir / "task_state.json",
            strategy_path=state_dir / "strategy.json",
            events_path=state_dir / "events.jsonl",
            state_dir=state_dir,
            repo_root=repo,
            worktree_root=worktree_root,
            worktree_submodule_paths=worktree_submodule_paths,
            database_program=database_program,
        )
    )


def _database_program() -> DatabaseProgramConfig:
    return DatabaseProgramConfig(
        authority_mode="quack",
        task_source_kind="duckdb",
        endpoint_secret_handle="handle:test-reconciliation",
        quack_endpoint="quack:127.0.0.1:41317",
        store_id="state/control.duckdb",
        store_generation="test-generation",
        schema_revision="1",
    )


class _ProjectionTaskSource:
    def __init__(self, record: object) -> None:
        self.record = record

    def get_task(self, task_cid: str) -> object | None:
        return (
            self.record
            if task_cid == str(getattr(self.record, "task_cid", "") or "")
            else None
        )

    def snapshot(self) -> object:
        return SimpleNamespace(repository_tree_id="tree:completion-proof")


def _seed_completed_database_portal_attempt(
    supervisor: TodoImplementationSupervisor,
    *,
    task_alias: str,
    task_cid: str,
    task_revision: int = 1,
    attempt_number: int = 4,
    completion_receipt_extension: dict[str, object] | None = None,
) -> tuple[object, object, object, object]:
    """Persist the same immutable attempt evidence Portal execution uses."""

    attempt = DatabaseTaskAttempt(
        attempt_id=f"attempt:{task_alias.lower()}",
        claim_id=f"claim:{task_alias.lower()}",
        task_cid=task_cid,
        task_alias=task_alias,
        attempt_number=attempt_number,
        owner_session_id=f"owner:{task_alias.lower()}",
        fencing_token=4,
        fence_epoch=1,
        lease_id=f"lease:{task_alias.lower()}",
        committed_phase="claimed",
        status="running",
        started_at_ms=1786946656000,
    )
    record = SimpleNamespace(
        task_alias=task_alias,
        task_cid=task_cid,
        goal_cid="goal:completion-proof",
        plan_cid="plan:completion-proof",
        revision=task_revision,
        priority="P1",
        dependencies=(),
        outputs=({"path": "src/app.py"},),
        validations=({"argv": ["python", "-m", "pytest", "focused.py"]},),
        acceptance=({"criterion": "Focused validation passes"},),
        body={
            "objective": "Land the exact completed implementation",
            "completion": "auto",
            "track": "ops",
            "allowed_paths": ["src/app.py"],
            "completion_contract": "Focused validation passes",
        },
    )
    attempt_root = (
        supervisor.config.state_dir
        / f"{supervisor.config.state_prefix}_database_portal_attempts"
    )
    bridge = DatabasePortalExecutionBridge(
        task_source=_ProjectionTaskSource(record),
        attempt_root=attempt_root,
        portal_factory=lambda _paths, _alias: object(),
    )
    paths, binding = bridge._ensure_attempt_projection(attempt, record)
    [projected_task] = parse_task_file(
        paths.task_projection,
        f"## {task_alias.split('-', 1)[0]}-",
    )
    portal_identity = portal_task_identity(
        projected_task,
        todo_path=paths.task_projection,
    )

    preparation = {
        "schema": TASK_COMPLETION_PREPARATION_SCHEMA,
        "task_cid": task_cid,
        "attempt_id": binding["attempt_id"],
        # The obsolete rescue is attempt 1; a later retry is the accepted
        # canonical completion.  Reconciliation must not conflate them.
        "attempt_number": binding["attempt_number"],
        "claim_id": binding["claim_id"],
        "lease_id": binding["lease_id"],
        "owner_session_id": binding["owner_session_id"],
        "fencing_token": binding["fencing_token"],
        "fence_epoch": binding["fence_epoch"],
        "control_expected_revision": binding["task_revision"],
        "control_expected_status": "in_progress",
        "evidence_digest": "sha256:passed",
        "prepared_at_ms": 1786946656000,
        "body": {"validation": {"outcome": "passed"}},
    }
    preparation["preparation_digest"] = (
        DatabaseCoordinator._preparation_digest(preparation)
    )
    preparation["status"] = PREPARED_COMPLETION_STATUS
    preparation["replayed"] = False
    completion_receipt: dict[str, object] = {
        "operation": "database_complete",
        "attempt_id": binding["attempt_id"],
        "claim_id": binding["claim_id"],
        "lease_id": binding["lease_id"],
        "owner_session_id": binding["owner_session_id"],
        "fencing_token": binding["fencing_token"],
        "fence_epoch": binding["fence_epoch"],
        "evidence_digest": "sha256:passed",
        "coordination_preparation": preparation,
        "validation": {
            "outcome": "passed",
            "evidence_digest": "sha256:passed",
        },
    }
    if completion_receipt_extension:
        assert not set(completion_receipt) & set(completion_receipt_extension)
        completion_receipt.update(completion_receipt_extension)
    canonical_task = SimpleNamespace(
        task_alias=task_alias,
        task_cid=task_cid,
        status="completed",
        revision=record.revision + 1,
        body={"completion_receipt": completion_receipt},
    )
    return paths, projected_task, portal_identity, canonical_task


def _seed_rescue_worktree(
    repo: Path,
    *,
    worktree_root: Path,
    task_alias: str,
    fingerprint: str,
) -> tuple[str, Path]:
    branch_name = (
        "rescue/worktree/implementation-"
        f"{task_alias.lower()}-{fingerprint[:12]}-"
        "attempt-1-1786946656-2b2ccc291a04"
    )
    marker = repo / "src" / "app.py"
    _git(repo, "checkout", "-b", branch_name)
    marker.write_text("leftover rescue change\n", encoding="utf-8")
    _git(repo, "commit", "-am", "stale rescue of completed task")
    _git(repo, "checkout", "main")
    marker.write_text("landed completion\n", encoding="utf-8")
    _git(repo, "commit", "-am", "completion already on main")
    worktree_path = worktree_root / f"{task_alias.lower()}-rescue"
    _git(repo, "worktree", "add", str(worktree_path), branch_name)
    return branch_name, worktree_path


def _completed_rescue_proof_fixture(
    tmp_path: Path,
    *,
    task_alias: str,
    task_cid: str = "",
    task_revision: int = 1,
    attempt_number: int = 4,
    completion_receipt_extension: dict[str, object] | None = None,
) -> tuple[Path, object, Path, object, object, object, object]:
    repo = _init_repo(tmp_path / "repo")
    marker = repo / "src" / "app.py"
    marker.parent.mkdir()
    marker.write_text("base\n", encoding="utf-8")
    _git(repo, "add", "src/app.py")
    _git(repo, "commit", "-m", "base")
    worktree_root = repo / "worktrees"
    supervisor = _supervisor(
        repo,
        worktree_root=worktree_root,
        todo_text=(
            "# Agent Todos\n\n"
            f"## {task_alias} Already landed\n\n"
            "- Status: completed\n"
            "- Completion: validated-implementation\n"
            "- Priority: P1\n"
            "- Track: ops\n"
            "- Outputs: src/app.py\n"
        ),
        database_program=_database_program(),
    )
    task_cid = task_cid or f"task-cid:{task_alias.lower()}"
    paths, projected_task, portal_identity, canonical_task = (
        _seed_completed_database_portal_attempt(
            supervisor,
            task_alias=task_alias,
            task_cid=task_cid,
            task_revision=task_revision,
            attempt_number=attempt_number,
            completion_receipt_extension=completion_receipt_extension,
        )
    )
    return (
        repo,
        supervisor,
        worktree_root,
        paths,
        projected_task,
        portal_identity,
        canonical_task,
    )


_PCSM_HISTORICAL_COMPLETION_CASES = (
    (
        "PCSM-023",
        "baguqeeraupubf5hqv7sg3de6ylhu4lfbhmdh6stvwow5lzli7h37fux435ia",
        7,
        2,
        "baguqeeraks4c4cc2gody65itaawqajdt3b5bcaotitrs6v4kexxqvpuau2xq",
    ),
    (
        "PCSM-040",
        "baguqeeracpy55nht3icjbnyopccuv7iqojmedl3y2bchtxmusowfpzjwnaea",
        15,
        4,
        "baguqeerasfnaw2h6jgj2au27ohgs2x2gecg3qwhs5vi7yoygfnare277dscq",
    ),
)


def _pcsm_historical_execution_route_receipt_fields(
    *,
    task_alias: str,
    task_cid: str,
    task_contract_cid: str,
) -> dict[str, object]:
    route = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "task-execution-route-binding@1"
        ),
        "policy_id": (
            "baguqeerahvwjwfbexnsqdflng54qvtuqoiwf4junsjaafyuhvl2fmjkeettq"
        ),
        "plan_root_cid": (
            "baguqeerax2gqeo7n5uy3uw3q7h5zn7ruicibrzfy65yv2vachdvbfnnvyfla"
        ),
        "repository_tree_id": "cd81f5731ee64c29161830c9933d6739e0dd3eb3",
        "source_revision": 1,
        "task_cid": task_cid,
        "task_alias": task_alias,
        "task_revision": 1,
        "task_contract_cid": task_contract_cid,
        "execution_mode": "grok-codex",
    }
    return {
        "execution_route_binding": route,
        "execution_route_policy_id": route["policy_id"],
        "execution_route_origin_revision": route["task_revision"],
    }


def _seed_parent_with_submodule(tmp_path: Path) -> tuple[Path, Path]:
    child_source = tmp_path / "child-source"
    child_source.mkdir()
    _git(child_source, "init")
    _git(child_source, "checkout", "-b", "main")
    _git(child_source, "config", "user.name", "Auto Unblock Test")
    _git(child_source, "config", "user.email", "auto-unblock@example.invalid")
    (child_source / "child.txt").write_text("base\n", encoding="utf-8")
    _git(child_source, "add", "child.txt")
    _git(child_source, "commit", "-m", "child base")

    repo = _init_repo(tmp_path / "parent")
    _git(
        repo,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(child_source),
        "ipfs_datasets_py",
    )
    _git(repo, "add", ".gitmodules", "ipfs_datasets_py")
    _git(repo, "commit", "-m", "add datasets submodule")
    submodule = repo / "ipfs_datasets_py"
    _git(submodule, "checkout", "main")
    _git(submodule, "config", "user.name", "Auto Unblock Test")
    _git(submodule, "config", "user.email", "auto-unblock@example.invalid")
    return repo, submodule


def test_redundant_dirty_treats_matching_submodule_working_tree_as_redundant(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = _init_repo(tmp_path / "repo")
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "base")
    worktree_root = repo / "worktrees"
    supervisor = _supervisor(
        repo,
        worktree_root=worktree_root,
        worktree_submodule_paths=("ipfs_datasets_py",),
    )
    monkeypatch.setattr(
        supervisor,
        "_submodule_gitlink_matches_target",
        lambda _worktree, relative, _ref: relative == "ipfs_datasets_py",
    )

    result = supervisor._redundant_dirty_worktree_status(
        repo,
        [" m ipfs_datasets_py"],
        "main",
    )

    assert result["redundant"] is True
    assert result["reason"] == "submodule_working_tree_matches_gitlink"


def test_uppercase_submodule_gitlink_move_is_not_auto_classified_redundant(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = _init_repo(tmp_path / "repo")
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "base")
    worktree_root = repo / "worktrees"
    supervisor = _supervisor(
        repo,
        worktree_root=worktree_root,
        worktree_submodule_paths=("ipfs_datasets_py",),
    )
    monkeypatch.setattr(
        supervisor,
        "_submodule_gitlink_matches_target",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        supervisor,
        "_worktree_file_matches_ref",
        lambda *_args, **_kwargs: False,
    )

    result = supervisor._redundant_dirty_worktree_status(
        repo,
        [" M ipfs_datasets_py"],
        "main",
    )

    assert result["redundant"] is False
    assert result["reason"] == "submodule_gitlink_diverged"


def test_cleanup_removes_merged_worktree_with_matching_submodule_working_tree(
    tmp_path: Path,
) -> None:
    repo, submodule = _seed_parent_with_submodule(tmp_path)
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "parent readme")
    branch_name = "implementation/portal-113-attempt-1-123"
    _git(repo, "checkout", "-b", branch_name)
    (repo / "README.md").write_text("branch\n", encoding="utf-8")
    _git(repo, "commit", "-am", "branch change")
    _git(repo, "checkout", "main")
    _git(repo, "merge", "--no-ff", "--no-edit", branch_name)

    worktree_root = repo / "worktrees"
    worktree_path = worktree_root / "portal-113"
    _git(repo, "worktree", "add", str(worktree_path), branch_name)
    _git(
        worktree_path,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "update",
        "--init",
        "--force",
        "--checkout",
        "--",
        "ipfs_datasets_py",
    )
    recorded = ""
    for line in _git(worktree_path, "ls-tree", "HEAD", "--", "ipfs_datasets_py").splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0] == "160000":
            recorded = parts[2]
    assert recorded
    _git(worktree_path / "ipfs_datasets_py", "checkout", "--force", recorded)
    nested = worktree_path / "ipfs_datasets_py" / "child.txt"
    nested.write_text("local submodule dirt\n", encoding="utf-8")
    porcelain = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=worktree_path,
        text=True,
        capture_output=True,
        check=False,
    ).stdout
    assert any(
        line[3:].strip() == "ipfs_datasets_py" and line[:2] in {" m", " M", "M "}
        for line in porcelain.splitlines()
        if len(line) > 3
    ), porcelain

    supervisor = _supervisor(
        repo,
        worktree_root=worktree_root,
        worktree_submodule_paths=("ipfs_datasets_py",),
    )
    result = supervisor.cleanup_backlogged_worktrees()

    assert result["removed_count"] == 1
    assert result["removed"][0]["dirty_redundancy"]["redundant"] is True
    assert result["removed"][0]["dirty_redundancy"]["reason"] == (
        "submodule_working_tree_matches_gitlink"
    )
    assert not worktree_path.exists()
    records = reconciliation_guardrail_records(
        reconciliation_result={"attempted": True, "main_checkout_dirty": False},
        cleanup_result=result,
    )
    assert not any(item["kind"] == "dirty_backlogged_worktree" for item in records)


def test_reconcile_skips_completed_rescue_leftover_before_preflight(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = _init_repo(tmp_path / "repo")
    marker = repo / "src" / "app.py"
    marker.parent.mkdir()
    marker.write_text("base\n", encoding="utf-8")
    _git(repo, "add", "src/app.py")
    _git(repo, "commit", "-m", "base")
    worktree_root = repo / "worktrees"
    supervisor = _supervisor(
        repo,
        worktree_root=worktree_root,
        todo_text=(
            "# Agent Todos\n\n"
            "## PORTAL-060 Already landed\n\n"
            "- Status: completed\n"
            "- Completion: validated-implementation\n"
            "- Priority: P1\n"
            "- Track: ops\n"
            "- Outputs: src/app.py\n"
        ),
        database_program=_database_program(),
    )
    task_cid = "task-cid:portal-060"
    paths, projected_task, portal_identity, canonical_task = (
        _seed_completed_database_portal_attempt(
            supervisor,
            task_alias="PORTAL-060",
            task_cid=task_cid,
        )
    )
    database_fingerprint = hashlib.sha256(
        task_cid.encode("utf-8")
    ).hexdigest()
    projection_fingerprint = projected_task.canonical_task_key.rsplit(
        "/", 1
    )[-1]
    assert len(
        {
            database_fingerprint,
            projection_fingerprint,
            portal_identity.semantic_fingerprint,
        }
    ) == 3
    assert paths.task_projection.is_file()
    branch_name, worktree_path = _seed_rescue_worktree(
        repo,
        worktree_root=worktree_root,
        task_alias="PORTAL-060",
        fingerprint=portal_identity.semantic_fingerprint,
    )
    completion_proof = supervisor._database_completion_receipt_proof(
        canonical_task,
        expected_alias="PORTAL-060",
        branch=branch_name,
    )
    assert completion_proof["verified"] is True
    assert completion_proof["branch_attempt_number"] == 1
    assert completion_proof["completion_attempt_number"] == 4
    for projection_field, invalid_value in (
        ("status", "completed"),
        ("replayed", 1),
    ):
        forged_body = json.loads(json.dumps(canonical_task.body))
        forged_body["completion_receipt"]["coordination_preparation"][
            projection_field
        ] = invalid_value
        forged_task = SimpleNamespace(
            task_alias=canonical_task.task_alias,
            task_cid=canonical_task.task_cid,
            status=canonical_task.status,
            revision=canonical_task.revision,
            body=forged_body,
        )
        forged_proof = supervisor._database_completion_receipt_proof(
            forged_task,
            expected_alias="PORTAL-060",
            branch=branch_name,
        )
        assert forged_proof["verified"] is False
        assert forged_proof["reason"] == (
            "database_completion_preparation_unverified"
        )
    monkeypatch.setattr(
        supervisor,
        "_canonical_completed_reconciliation_task",
        lambda **_kwargs: {
            "applicable": True,
            "authority_available": True,
            **completion_proof,
        },
    )

    result = supervisor.reconcile_backlogged_worktrees()

    assert result["preflight_blocked_count"] == 0
    assert result["reconciled_count"] == 0
    assert any(item["reason"] == "completed_task_leftover" for item in result["skipped"])
    leftover = next(
        item for item in result["skipped"] if item["reason"] == "completed_task_leftover"
    )
    assert leftover["task_id"] == "PORTAL-060"
    assert leftover["prune_result"]["removed"] is True
    assert leftover["prune_result"]["branch_preserved"] is True
    assert not worktree_path.exists()
    assert _git(repo, "rev-parse", branch_name) == leftover["head"]
    records = reconciliation_guardrail_records(
        reconciliation_result=result,
        cleanup_result={"skipped": []},
    )
    assert not any(item["kind"] == "preflight_merge_conflict" for item in records)


@pytest.mark.parametrize(
    (
        "task_alias",
        "task_cid",
        "task_revision",
        "attempt_number",
        "task_contract_cid",
    ),
    _PCSM_HISTORICAL_COMPLETION_CASES,
)
def test_completed_rescue_accepts_exact_pcsm_historical_route_receipt_shapes(
    tmp_path: Path,
    task_alias: str,
    task_cid: str,
    task_revision: int,
    attempt_number: int,
    task_contract_cid: str,
) -> None:
    route_fields = _pcsm_historical_execution_route_receipt_fields(
        task_alias=task_alias,
        task_cid=task_cid,
        task_contract_cid=task_contract_cid,
    )
    (
        repo,
        supervisor,
        worktree_root,
        _paths,
        _projected_task,
        portal_identity,
        canonical_task,
    ) = _completed_rescue_proof_fixture(
        tmp_path,
        task_alias=task_alias,
        task_cid=task_cid,
        task_revision=task_revision,
        attempt_number=attempt_number,
        completion_receipt_extension=route_fields,
    )
    branch_name, _worktree_path = _seed_rescue_worktree(
        repo,
        worktree_root=worktree_root,
        task_alias=task_alias,
        fingerprint=portal_identity.semantic_fingerprint,
    )

    receipt = canonical_task.body["completion_receipt"]
    assert set(receipt) == {
        "operation",
        "attempt_id",
        "claim_id",
        "lease_id",
        "owner_session_id",
        "fencing_token",
        "fence_epoch",
        "evidence_digest",
        "coordination_preparation",
        "validation",
        "execution_route_binding",
        "execution_route_policy_id",
        "execution_route_origin_revision",
    }
    assert {
        key: receipt[key]
        for key in (
            "execution_route_binding",
            "execution_route_policy_id",
            "execution_route_origin_revision",
        )
    } == route_fields

    proof = supervisor._database_completion_receipt_proof(
        canonical_task,
        expected_alias=task_alias,
        branch=branch_name,
    )

    assert proof["verified"] is True
    assert proof["revision"] == task_revision + 1
    assert proof["completion_attempt_number"] == attempt_number
    assert proof["branch_attempt_number"] == 1


@pytest.mark.parametrize(
    ("fault", "expected_reason"),
    (
        (
            "partial_route_receipt",
            "database_completion_receipt_shape_invalid",
        ),
        (
            "route_lineage_tampered",
            "database_completion_execution_route_unverified",
        ),
        (
            "route_origin_after_attempt",
            "database_completion_execution_route_unverified",
        ),
        (
            "route_origin_boolean",
            "database_completion_execution_route_unverified",
        ),
        (
            "duplicate_attempt_projection",
            "database_portal_attempt_projection_ambiguous",
        ),
    ),
)
def test_pcsm_historical_route_receipt_tamper_or_ambiguity_is_rejected(
    tmp_path: Path,
    fault: str,
    expected_reason: str,
) -> None:
    task_alias, task_cid, task_revision, attempt_number, task_contract_cid = (
        _PCSM_HISTORICAL_COMPLETION_CASES[0]
    )
    route_fields = _pcsm_historical_execution_route_receipt_fields(
        task_alias=task_alias,
        task_cid=task_cid,
        task_contract_cid=task_contract_cid,
    )
    (
        repo,
        supervisor,
        worktree_root,
        paths,
        _projected_task,
        portal_identity,
        canonical_task,
    ) = _completed_rescue_proof_fixture(
        tmp_path,
        task_alias=task_alias,
        task_cid=task_cid,
        task_revision=task_revision,
        attempt_number=attempt_number,
        completion_receipt_extension=route_fields,
    )
    branch_name, worktree_path = _seed_rescue_worktree(
        repo,
        worktree_root=worktree_root,
        task_alias=task_alias,
        fingerprint=portal_identity.semantic_fingerprint,
    )
    receipt = canonical_task.body["completion_receipt"]
    if fault == "partial_route_receipt":
        receipt.pop("execution_route_policy_id")
    elif fault == "route_lineage_tampered":
        receipt["execution_route_policy_id"] = (
            "baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        )
    elif fault == "route_origin_after_attempt":
        route = receipt["execution_route_binding"]
        assert isinstance(route, dict)
        route["task_revision"] = task_revision + 1
        receipt["execution_route_origin_revision"] = task_revision + 1
    elif fault == "route_origin_boolean":
        receipt["execution_route_origin_revision"] = True
    elif fault == "duplicate_attempt_projection":
        duplicate = (
            repo
            / "sibling-state"
            / "sibling_database_portal_attempts"
            / paths.root.name
        )
        shutil.copytree(paths.root, duplicate)
    else:  # pragma: no cover - the parameter table is intentionally closed.
        raise AssertionError(f"unknown fault: {fault}")

    proof = supervisor._database_completion_receipt_proof(
        canonical_task,
        expected_alias=task_alias,
        branch=branch_name,
    )

    assert proof["verified"] is False
    assert proof["reason"] == expected_reason
    assert worktree_path.is_dir()


@pytest.mark.parametrize(
    ("fault", "expected_reason"),
    (
        ("database_identity", "branch_task_generation_mismatch"),
        ("projection_identity", "branch_task_generation_mismatch"),
        ("missing", "database_portal_attempt_projection_missing"),
        ("duplicate", "database_portal_attempt_projection_ambiguous"),
        ("tampered", "database_portal_attempt_projection_unverified"),
    ),
)
def test_completed_rescue_unverified_identity_is_preserved(
    tmp_path: Path,
    monkeypatch,
    fault: str,
    expected_reason: str,
) -> None:
    (
        repo,
        supervisor,
        worktree_root,
        paths,
        projected_task,
        portal_identity,
        canonical_task,
    ) = _completed_rescue_proof_fixture(
        tmp_path,
        task_alias="PORTAL-061",
    )
    branch_fingerprint = portal_identity.semantic_fingerprint
    if fault == "database_identity":
        branch_fingerprint = hashlib.sha256(
            canonical_task.task_cid.encode("utf-8")
        ).hexdigest()
    elif fault == "projection_identity":
        branch_fingerprint = projected_task.canonical_task_key.rsplit(
            "/", 1
        )[-1]
    branch_name, worktree_path = _seed_rescue_worktree(
        repo,
        worktree_root=worktree_root,
        task_alias="PORTAL-061",
        fingerprint=branch_fingerprint,
    )
    if fault == "missing":
        shutil.rmtree(paths.root)
    elif fault == "duplicate":
        duplicate = (
            repo
            / "sibling-state"
            / "sibling_database_portal_attempts"
            / paths.root.name
        )
        shutil.copytree(paths.root, duplicate)
    elif fault == "tampered":
        projection_text = paths.task_projection.read_text(encoding="utf-8")
        paths.task_projection.write_text(
            projection_text.replace(
                "Land the exact completed implementation",
                "Tampered completed implementation",
            ),
            encoding="utf-8",
        )

    completion_proof = supervisor._database_completion_receipt_proof(
        canonical_task,
        expected_alias="PORTAL-061",
        branch=branch_name,
    )
    assert completion_proof["verified"] is False
    assert completion_proof["reason"] == expected_reason
    if fault.endswith("_identity"):
        assert branch_fingerprint != portal_identity.semantic_fingerprint
        assert completion_proof["expected_fingerprint"] == (
            portal_identity.semantic_fingerprint[:12]
        )
    monkeypatch.setattr(
        supervisor,
        "_canonical_completed_reconciliation_task",
        lambda **_kwargs: {
            "applicable": True,
            "authority_available": True,
            **completion_proof,
        },
    )

    result = supervisor.reconcile_backlogged_worktrees()

    preserved = next(
        item
        for item in result["skipped"]
        if item["reason"] == "canonical_completion_proof_unavailable"
    )
    assert preserved["completion_proof"]["reason"] == expected_reason
    assert worktree_path.is_dir()
    assert _git(repo, "rev-parse", branch_name) == preserved["head"]


def test_markdown_completed_status_is_not_canonical_rescue_prune_authority(
    tmp_path: Path,
) -> None:
    repo = _init_repo(tmp_path / "repo")
    supervisor = _supervisor(
        repo,
        worktree_root=repo / "worktrees",
        todo_text=(
            "# Agent Todos\n\n"
            "## PORTAL-060 Completed only in Markdown\n\n"
            "- Status: completed\n"
        ),
    )

    proof = supervisor._canonical_completed_reconciliation_task(
        branch=(
            "rescue/worktree/implementation-portal-060-"
            "d2c6b71e0110-attempt-1-1-deadbeef0000"
        ),
        known_task_ids=("PORTAL-060",),
    )

    assert proof == {
        "applicable": False,
        "authority_available": False,
        "verified": False,
        "reason": "database_task_authority_not_configured",
    }


def test_completed_rescue_with_last_moment_data_is_preserved_nonblocking(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = _init_repo(tmp_path / "repo")
    tracked = repo / "tracked.txt"
    tracked.write_text("base\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-m", "base")
    branch_name = (
        "rescue/worktree/implementation-pcar-000-b62604280cfd-"
        "attempt-1-1787277595-5b3d1f2e245d"
    )
    _git(repo, "branch", branch_name)
    worktree_root = repo / "worktrees"
    worktree_path = worktree_root / "pcar-rescue"
    _git(repo, "worktree", "add", str(worktree_path), branch_name)
    (worktree_path / "tracked.txt").write_text(
        "superseded attempt\n",
        encoding="utf-8",
    )
    _git(worktree_path, "commit", "-am", "superseded attempt")
    untracked = worktree_path / "last-moment-evidence.txt"
    untracked.write_text("must survive\n", encoding="utf-8")
    expected_head = _git(repo, "rev-parse", branch_name)
    supervisor = _supervisor(
        repo,
        worktree_root=worktree_root,
        todo_text=(
            "# Agent Todos\n\n"
            "## PCAR-000 Canonically completed\n\n"
            "- Status: todo\n"
        ),
        database_program=_database_program(),
    )
    monkeypatch.setattr(
        supervisor,
        "_canonical_completed_reconciliation_task",
        lambda **_kwargs: {
            "applicable": True,
            "authority_available": True,
            "verified": True,
            "reason": "database_completed_task_verified",
            "task_id": "PCAR-000",
            "task_cid": "canonical:pcar-000",
            "status": "completed",
            "revision": 11,
        },
    )
    monkeypatch.setattr(
        supervisor,
        "_rescue_dirty_worktree",
        lambda *_args, **_kwargs: pytest.fail(
            "verified completed rescue must not be rewritten"
        ),
    )

    reconciliation = supervisor.reconcile_backlogged_worktrees()

    leftover = next(
        item
        for item in reconciliation["skipped"]
        if item["reason"] == "completed_task_leftover"
    )
    result = leftover["prune_result"]
    assert reconciliation["preflight_blocked_count"] == 0
    assert result["removed"] is False
    assert result["preserved_nonblocking"] is True
    assert result["reason"] == (
        "completed_rescue_worktree_preserved_not_clean"
    )
    assert result["branch_preserved"] is True
    assert worktree_path.is_dir()
    assert untracked.read_text(encoding="utf-8") == "must survive\n"
    assert _git(repo, "rev-parse", branch_name) == expected_head

    cached = supervisor.reconcile_backlogged_worktrees()

    assert cached["scan_cache_hit_count"] >= 1
    assert any(
        item.get("reason") == "completed_task_leftover"
        and item.get("cached") is True
        for item in cached["skipped"]
    )


def test_duckdb_reconciliation_guardrail_never_mutates_sealed_markdown_projection(
    tmp_path: Path,
) -> None:
    repo = _init_repo(tmp_path / "repo")
    sealed_board = (
        "# Sealed task projection\n\n"
        "## PCAR-000 Existing task\n\n"
        "- Status: completed\n"
    )
    supervisor = _supervisor(
        repo,
        worktree_root=repo / "worktrees",
        todo_text=sealed_board,
        database_program=_database_program(),
    )

    findings = supervisor.record_reconciliation_guardrails(
        {
            "attempted": True,
            "processed": [
                {
                    "branch": "rescue/worktree/implementation-pcar-000-attempt-1",
                    "path": str(repo / "worktrees" / "rescue"),
                    "target_ref": "main",
                    "preflight_result": {
                        "attempted": True,
                        "mergeable": False,
                        "reason": "preflight_merge_conflict",
                        "conflict_paths": ["sealed.json"],
                    },
                }
            ],
        },
        {"attempted": True, "skipped": []},
    )

    assert findings == []
    assert supervisor.config.todo_path.read_text(encoding="utf-8") == sealed_board
    assert not supervisor._reconciliation_guardrail_discovery_dir().exists()
    events = [
        json.loads(line)
        for line in supervisor.config.events_path.read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert events[-1]["type"] == "generated_board_mutation_suppressed"
    assert events[-1]["reason"] == "immutable_database_authority_projection"
    assert events[-1]["task_source_kind"] == "duckdb"
    assert events[-1]["canonical_blocks_mutated"] is False


@pytest.mark.parametrize(
    "producer",
    (
        "guardrail-release",
        "dependency-guardrail",
        "reconciliation-guardrail",
        "retry-budget",
    ),
)
def test_duckdb_suppresses_every_markdown_guardrail_producer(
    tmp_path: Path,
    producer: str,
) -> None:
    repo = _init_repo(tmp_path / f"repo-{producer}")
    sealed_board = "# Sealed database projection\n"
    supervisor = _supervisor(
        repo,
        worktree_root=repo / "worktrees",
        todo_text=sealed_board,
        database_program=_database_program(),
    )
    callback_calls: list[str] = []

    result = supervisor._run_generated_board_producer(
        producer=producer,
        commit_outputs=False,
        callback=lambda: callback_calls.append(producer),
    )

    assert result == []
    assert callback_calls == []
    assert supervisor.config.todo_path.read_text(encoding="utf-8") == sealed_board
    event = json.loads(
        supervisor.config.events_path.read_text(encoding="utf-8").splitlines()[
            -1
        ]
    )
    assert event["type"] == "generated_board_mutation_suppressed"
    assert event["producer"] == producer
    assert event["canonical_blocks_mutated"] is False


def test_unreadable_completed_rescue_preserves_submodule_data_without_reset_surface(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = _init_repo(tmp_path / "repo")
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "base")
    branch_name = (
        "rescue/worktree/implementation-pcar-000-b62604280cfd-"
        "attempt-1-1787277595-5b3d1f2e245d"
    )
    _git(repo, "branch", branch_name)
    worktree_root = repo / "worktrees"
    leftover = worktree_root / "unreadable-rescue"
    _git(repo, "worktree", "add", str(leftover), branch_name)
    (leftover / "README.md").write_text(
        "superseded attempt\n",
        encoding="utf-8",
    )
    _git(leftover, "commit", "-am", "superseded attempt")
    submodule_evidence = leftover / "ipfs_datasets_py" / "untracked-proof.txt"
    submodule_evidence.parent.mkdir()
    submodule_evidence.write_text("preserve me\n", encoding="utf-8")
    expected_head = _git(repo, "rev-parse", branch_name)
    supervisor = _supervisor(
        repo,
        worktree_root=worktree_root,
        worktree_submodule_paths=("ipfs_datasets_py",),
        todo_text=(
            "# Agent Todos\n\n"
            "## PCAR-000 Canonically completed\n\n"
            "- Status: todo\n"
        ),
        database_program=_database_program(),
    )
    monkeypatch.setattr(
        supervisor,
        "_canonical_completed_reconciliation_task",
        lambda **_kwargs: {
            "applicable": True,
            "authority_available": True,
            "verified": True,
            "reason": "database_completed_task_verified",
            "task_id": "PCAR-000",
            "task_cid": "canonical:pcar-000",
            "status": "completed",
            "revision": 11,
        },
    )

    def unreadable_submodule_status(_path: Path) -> list[str]:
        raise RuntimeError("nested submodule status is unavailable")

    monkeypatch.setattr(
        supervisor,
        "_git_status_short_strict",
        unreadable_submodule_status,
    )

    result = supervisor.reconcile_backlogged_worktrees()

    completed_skip = next(
        item
        for item in result["skipped"]
        if item["reason"] == "completed_task_leftover"
    )
    preservation = completed_skip["prune_result"]
    assert result["preflight_blocked_count"] == 0
    assert preservation["removed"] is False
    assert preservation["preserved_nonblocking"] is True
    assert preservation["reason"] == (
        "completed_rescue_worktree_preserved_status_unavailable"
    )
    assert preservation["branch_preserved"] is True
    assert leftover.is_dir()
    assert submodule_evidence.read_text(encoding="utf-8") == "preserve me\n"
    assert _git(repo, "rev-parse", branch_name) == expected_head
    assert not hasattr(supervisor, "apply_safe_reconciliation_remediations")
    assert not hasattr(supervisor, "_reset_matching_submodule_working_trees")


def test_resolved_keys_retire_cleared_preflight_and_unsupported_status() -> None:
    resolved = resolved_reconciliation_guardrail_keys(
        reconciliation_result={
            "attempted": True,
            "preflight_blocked_count": 0,
            "processed": [],
        },
        cleanup_result={
            "attempted": True,
            "dirty_worktree_groups": {},
            "skipped": [],
        },
    )

    assert "reconciliation_guardrail:preflight_merge_conflict" in resolved
    assert (
        "reconciliation_guardrail:dirty_backlogged_worktree:unsupported_status" in resolved
    )


def test_resolved_keys_keep_active_preflight_and_unsupported_status() -> None:
    resolved = resolved_reconciliation_guardrail_keys(
        reconciliation_result={
            "attempted": True,
            "preflight_blocked_count": 1,
            "processed": [
                {
                    "merged": False,
                    "preflight_result": {"mergeable": False, "conflict_paths": ["src/app.py"]},
                }
            ],
        },
        cleanup_result={
            "attempted": True,
            "dirty_worktree_groups": {"unsupported_status": {"count": 1}},
            "skipped": [
                {
                    "reason": "dirty_worktree",
                    "dirty_redundancy": {"reason": "unsupported_status"},
                }
            ],
        },
    )

    assert "reconciliation_guardrail:preflight_merge_conflict" not in resolved
    assert (
        "reconciliation_guardrail:dirty_backlogged_worktree:unsupported_status" not in resolved
    )


def test_resolved_keys_do_not_retire_dirty_cards_when_cleanup_is_locked() -> None:
    resolved = resolved_reconciliation_guardrail_keys(
        cleanup_result={
            "attempted": True,
            "reason": "checkout_mutation_lock_exists",
            "dirty_worktree_groups": {},
            "skipped": [],
        }
    )

    assert (
        "reconciliation_guardrail:dirty_backlogged_worktree:unsupported_status" not in resolved
    )


def test_run_implementation_skips_board_completed_task(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    (repo / "src").mkdir()
    (repo / "src" / "app.py").write_text("base\n", encoding="utf-8")
    todo_path = repo / "todo.md"
    todo_path.write_text(
        "# Agent Todos\n\n"
        "## PORTAL-071 Already landed\n\n"
        "- Status: completed\n"
        "- Completion: validated-implementation\n"
        "- Priority: P1\n"
        "- Track: evaluation\n"
        "- Outputs: src/app.py\n",
        encoding="utf-8",
    )
    state_dir = repo / "state"
    daemon = TodoImplementationDaemon(
        todo_path=todo_path,
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        task_header_prefix="## PORTAL-",
        implementation_command="must-not-run",
    )
    task = PortalTask(
        task_id="PORTAL-071",
        title="Already landed",
        status="todo",
        completion="manual",
        priority="P1",
        track="evaluation",
        outputs=["src/app.py"],
    )

    result = daemon._run_implementation(task, TodoTaskState())

    assert result["skipped"] is True
    assert result["reason"] == "completed_task_leftover"
    assert result["attempt_consumed"] is False
    assert result["provider_dispatched"] is False


def test_supervisor_releases_completed_leftover_execution(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = _init_repo(tmp_path / "repo")
    (repo / "todo.md").write_text(
        "# Agent Todos\n\n"
        "## PORTAL-071 Already landed\n\n"
        "- Status: completed\n"
        "- Completion: validated-implementation\n"
        "- Priority: P1\n"
        "- Track: evaluation\n"
        "- Outputs: src/app.py\n",
        encoding="utf-8",
    )
    state_dir = repo / "state"
    state_path = state_dir / "task_state.json"
    TodoTaskState(
        active_task_id="PORTAL-071",
        active_task_title="Already landed",
        active_attempt=3,
        active_phase="implementing",
        active_worktree_path=str(repo / "worktrees" / "portal-071-attempt-3"),
        active_branch="implementation/portal-071-attempt-3",
        implementation_in_progress=True,
    ).save(state_path)
    supervisor = TodoImplementationSupervisor(
        TodoSupervisorConfig(
            todo_path=repo / "todo.md",
            state_path=state_path,
            strategy_path=state_dir / "strategy.json",
            events_path=state_dir / "events.jsonl",
            state_dir=state_dir,
            repo_root=repo,
            worktree_root=repo / "worktrees",
            task_prefix="## PORTAL-",
        )
    )
    stop_calls: list[float] = []
    monkeypatch.setattr(
        supervisor,
        "_terminate_managed_daemon_tree",
        lambda grace_seconds=1.0: stop_calls.append(grace_seconds)
        or {"terminated": True, "pid": 4321, "quiesced": True},
    )

    result = supervisor.release_completed_leftover_execution()

    assert result["released"] is True
    assert result["reason"] == "completed_task_leftover"
    assert result["active_task_id"] == "PORTAL-071"
    assert stop_calls == [2.0]
    recovered = TodoTaskState.load(state_path)
    assert recovered.implementation_in_progress is False
    assert recovered.active_task_id == "PORTAL-071"
    assert recovered.active_worktree_path == ""
    assert recovered.active_attempt == 0


def test_supervisor_does_not_stop_live_incomplete_task(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = _init_repo(tmp_path / "repo")
    (repo / "todo.md").write_text(
        "# Agent Todos\n\n"
        "## PORTAL-072 Still open\n\n"
        "- Status: todo\n"
        "- Completion: manual\n"
        "- Priority: P1\n"
        "- Track: ops\n"
        "- Outputs: src/app.py\n",
        encoding="utf-8",
    )
    state_dir = repo / "state"
    state_path = state_dir / "task_state.json"
    TodoTaskState(
        active_task_id="PORTAL-072",
        active_attempt=1,
        active_phase="implementing",
        active_worktree_path=str(repo / "worktrees" / "portal-072"),
        implementation_in_progress=True,
    ).save(state_path)
    supervisor = TodoImplementationSupervisor(
        TodoSupervisorConfig(
            todo_path=repo / "todo.md",
            state_path=state_path,
            strategy_path=state_dir / "strategy.json",
            events_path=state_dir / "events.jsonl",
            state_dir=state_dir,
            repo_root=repo,
            worktree_root=repo / "worktrees",
            task_prefix="## PORTAL-",
        )
    )
    monkeypatch.setattr(
        supervisor,
        "_terminate_managed_daemon_tree",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("must not stop live work")),
    )

    result = supervisor.release_completed_leftover_execution()

    assert result["released"] is False
    assert result["reason"] == "active_task_not_completed"
    assert TodoTaskState.load(state_path).implementation_in_progress is True
