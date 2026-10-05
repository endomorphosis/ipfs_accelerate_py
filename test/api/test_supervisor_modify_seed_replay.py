"""Real proposal and Git replay handoffs for an existing scoped output."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as module
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalImplementationDaemon,
    PortalTask,
    PortalTaskState,
    ValidationProjectDependencyPreflightDeferred,
)
from ipfs_accelerate_py.agent_supervisor.validation.project_dependency_preflight import (
    PROJECT_DEPENDENCY_PROBE_SCHEMA,
    SCOPED_PROJECT_DEPENDENCY_CONTRACT_SCHEMA_V2,
)
from test.api.test_agent_supervisor_prior_attempt_seed import _git


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _modify_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, pollution=""):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "Replay Test")
    _git(repo, "config", "user.email", "replay@example.invalid")
    control = tmp_path / "control"
    daemon = PortalImplementationDaemon(
        todo_path=control / "tasks.todo.md",
        state_path=control / "state.json",
        strategy_path=control / "strategy.json",
        events_path=control / "events.jsonl",
        repo_root=repo,
        worktree_root=tmp_path / "worktrees",
        worktree_pool_enabled=False,
    )
    target = "tests/unit/test_retry_seed_contract.py"
    command = f"python3 -m pytest {target} -q"
    task = PortalTask(
        task_id="MODIFY-SEED-001", title="Replay an existing scoped test output",
        status="todo", completion="manual", priority="P1", track="test",
        outputs=[target], validation=[command],
        board_namespace="modify-seed-board",
        canonical_task_key="modify-seed-task-key",
        canonical_task_cid="modify-seed-task-cid",
    )
    daemon._register_task_identities([task])
    identity = daemon._identity_for_task(task)
    requirements = ["pytest>=9.0,<10"]
    setup = f"from setuptools import setup\nsetup(extras_require={{'test': {requirements!r}}})\n".encode()
    before = "# café baseline\r\ndef test_replayed_seed():\r\n    assert False".encode()
    after = "# café replayed\r\ndef test_replayed_seed():\r\n    assert True".encode()
    (repo / "setup.py").write_bytes(setup)
    (repo / target).parent.mkdir(parents=True)
    (repo / target).write_bytes(before)
    (repo / "untouched.py").write_text("VALUE = 'baseline'\n")
    (repo / "pyproject.toml").write_text(f'''
[project]
name = "modify-seed-contract"
version = "1.0.0"
requires-python = ">=3.12"
dynamic = ["dependencies"]

[tool.ipfs-accelerate-agent-supervisor.project-dependency-preflight]
schema = {json.dumps(SCOPED_PROJECT_DEPENDENCY_CONTRACT_SCHEMA_V2)}
requires-python = ">=3.12"
authority = {{ file = "setup.py", sha256 = "{_sha(setup)}", extra = "test", extra-requirements-sha256 = "{_sha(json.dumps(requirements, separators=(',', ':')).encode())}" }}
targets = [
  {{ target = "{target}", validation-command-sha256 = "{_sha(command.encode())}", requirements = {json.dumps(requirements)}, task = {{ board-namespace = "{identity.board_namespace}", canonical-task-cid = "{identity.canonical_task_cid}", declared-output = "{target}" }}, baseline = {{ state = "present", sha256 = "{_sha(before)}" }} }},
]
'''.lstrip())
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "existing scoped target baseline")
    baseline = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-b", "prior-attempt")
    (repo / target).write_bytes(after)
    if pollution:
        (repo / "outside.py").write_text("UNAUTHORIZED = True\n")
        if pollution == "staged":
            _git(repo, "add", "outside.py")
    # Both the persisted source receipt and replay gate come from the real owner.
    validation = daemon._validate_implementation_patch(
        repo, task, baseline_ref=baseline, allow_scope_adjudication=False,
    )
    assert validation.accepted, daemon._compact_proposal_validation(validation)
    assert validation.proposal.changed_paths == (target,)
    _git(repo, "add", target)
    _git(repo, "commit", "-m", "prior attempt modifies scoped target")
    seed = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "main")
    workspace = tmp_path / "retry"
    _git(repo, "worktree", "add", "-b", "retry-modify", str(workspace), baseline)

    actual_preflight = module.preflight_validation_project_dependencies
    calls = []

    def deterministic_probe(workspace_path, commands, **kwargs):
        receipt = actual_preflight(
            workspace_path, commands, **kwargs,
            probe_runner=lambda *_args, **_kwargs: {
                "schema": PROJECT_DEPENDENCY_PROBE_SCHEMA,
                "passed": True, "reason": "project_dependencies_satisfied",
                "projects": [],
            },
        )
        calls.append(receipt)
        return receipt

    monkeypatch.setattr(module, "preflight_validation_project_dependencies", deterministic_probe)
    return SimpleNamespace(
        repo=repo, daemon=daemon, task=task, target=target,
        baseline=baseline, seed=seed, workspace=workspace,
        before=before, after=after, calls=calls,
    )


def _preflight(case, authority=None):
    return case.daemon._require_validation_project_dependency_preflight(
        workspace_path=case.workspace, task=case.task, attempt=2,
        branch_name="retry-modify", prior_seed_authority=authority,
    )


def _replay(case):
    baseline_receipt = _preflight(case)
    assert baseline_receipt["projects"][0]["scoped_validation_target_materialization_state"] == "present"
    result = case.daemon._apply_prior_attempt_seed(
        case.workspace, task=case.task,
        seed_plan={"reuse_prior_attempt": True, "seed_ref": case.seed},
        baseline_ref=case.baseline,
    )
    assert result["applied"] is True, result
    assert result["pre_dispatch_proposal_gate"]["accepted"] is True
    assert (case.workspace / case.target).read_bytes() == case.after
    assert _git(case.workspace, "rev-parse", "HEAD") == case.baseline
    authority = case.daemon._dependency_prior_seed_authority(
        task=case.task, baseline_ref=case.baseline,
        baseline_receipt=baseline_receipt, seed_apply=result,
    )
    return result, authority


def test_modify_seed_real_proposals_pass_both_dependency_preflights(tmp_path, monkeypatch):
    case = _modify_retry(tmp_path, monkeypatch)
    result, authority = _replay(case)
    # Keep the post-replay owner as the first red: old code rejects these bytes
    # as v2_present_target_digest_mismatch instead of issuing seed authority.
    receipt = _preflight(case, authority)
    assert receipt["passed"] is True
    assert receipt["projects"][0]["scoped_validation_target_materialization_state"] == "authenticated-prior-seed"
    assert result["seeded_outputs"] == [{
        "path": case.target, "change_kind": "modify",
        "before_sha256": _sha(case.before),
        "before_git_blob_id": _git(case.repo, "rev-parse", f"{case.baseline}:{case.target}"),
        "sha256": _sha(case.after),
        "git_blob_id": _git(case.repo, "hash-object", str(case.workspace / case.target)),
    }]
    assert authority["schema"].endswith("@2")
    assert _git(case.workspace, "rev-parse", "HEAD") == case.baseline
    assert _git(case.workspace, "diff", "--name-only", "HEAD") == case.target
    assert _git(case.workspace, "ls-files", "--others", "--exclude-standard") == ""


@pytest.mark.parametrize("pollution", ["staged", "untracked"])
def test_modify_seed_never_attests_out_of_scope_source_files(tmp_path, monkeypatch, pollution):
    case = _modify_retry(tmp_path, monkeypatch, pollution=pollution)
    result, authority = _replay(case)
    receipt = _preflight(case, authority)
    assert receipt["passed"] is True
    assert [row["path"] for row in result["seeded_outputs"]] == [case.target]
    assert not (case.workspace / "outside.py").exists()
    assert _git(case.workspace, "diff", "--name-only", "HEAD") == case.target
    assert _git(case.workspace, "rev-parse", "HEAD") == case.baseline


def _reseal(authority):
    body = {key: value for key, value in authority.items() if key != "authority_sha256"}
    authority["authority_sha256"] = _sha(json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode())


@pytest.mark.parametrize("mutation", [
    "before_bytes", "before_blob", "after_bytes", "after_blob",
    "board_namespace", "task_identity", "baseline_identity", "duplicate_output",
    "malformed_output", "unattested_output",
])
def test_modify_seed_rejects_resealed_tampering_after_real_replay(tmp_path, monkeypatch, mutation):
    case = _modify_retry(tmp_path, monkeypatch)
    _, authority = _replay(case)
    assert authority is not None
    authority = deepcopy(authority)
    output = authority["seeded_outputs"][0]
    if mutation == "before_bytes":
        output["before_sha256"] = "f" * 64
    elif mutation == "before_blob":
        output["before_git_blob_id"] = "f" * 40
    elif mutation == "after_bytes":
        (case.workspace / case.target).write_bytes(case.after + b"\n# unvalidated drift\n")
    elif mutation == "after_blob":
        output["git_blob_id"] = "f" * 40
    elif mutation == "board_namespace":
        authority["board_namespace"] = "foreign-board"
    elif mutation == "task_identity":
        authority["canonical_task_cid"] = "foreign-task"
    elif mutation == "baseline_identity":
        authority["baseline_commit_id"] = case.seed
        authority["repository_tree_id"] = "git-tree:" + _git(case.repo, "rev-parse", f"{case.seed}^{{tree}}")
        authority["proposal_repository_tree_id"] = authority["repository_tree_id"]
    elif mutation == "duplicate_output":
        authority["seeded_outputs"].append(dict(output))
    elif mutation == "malformed_output":
        output["before_sha256"] = True
    elif mutation == "unattested_output":
        authority["authorized_paths"] = ["outside.py"]
    _reseal(authority)
    with pytest.raises(ValidationProjectDependencyPreflightDeferred) as raised:
        _preflight(case, authority)
    assert raised.value.receipt["passed"] is False
    assert raised.value.receipt["projects"][0]["contract_error_reason"].startswith("v2_prior_seed_")
    assert _git(case.workspace, "rev-parse", "HEAD") == case.baseline


@pytest.mark.parametrize("identity", ["canonical_task_key", "canonical_task_cid", "board_namespace"])
def test_foreign_task_cannot_replay_real_accepted_modify_proposal(tmp_path, monkeypatch, identity):
    case = _modify_retry(tmp_path, monkeypatch)
    foreign = replace(case.task, **{identity: "foreign"})
    result = case.daemon._apply_prior_attempt_seed(
        case.workspace, task=foreign,
        seed_plan={"reuse_prior_attempt": True, "seed_ref": case.seed},
        baseline_ref=case.baseline,
    )
    assert result["applied"] is False
    assert result["reason"] == "prior_seed_accepted_proposal_missing"
    assert (case.workspace / case.target).read_bytes() == case.before
    assert _git(case.workspace, "status", "--porcelain") == ""
    assert _git(case.workspace, "rev-parse", "HEAD") == case.baseline


@pytest.mark.parametrize("source_namespace", [None, "foreign-board"])
def test_modify_seed_authority_requires_source_proposal_namespace(tmp_path, monkeypatch, source_namespace):
    case = _modify_retry(tmp_path, monkeypatch)
    result, authority = _replay(case)
    assert authority is not None
    detached_result = deepcopy(result)
    if source_namespace is None:
        detached_result["proposal_authority"].pop("board_namespace")
    else:
        detached_result["proposal_authority"]["board_namespace"] = source_namespace
    assert case.daemon._dependency_prior_seed_authority(
        task=case.task, baseline_ref=case.baseline,
        baseline_receipt=authority["baseline_receipt"], seed_apply=detached_result,
    ) is None
    assert _git(case.workspace, "rev-parse", "HEAD") == case.baseline


def test_foreign_namespace_is_refused_before_modify_replay(tmp_path, monkeypatch):
    case = _modify_retry(tmp_path, monkeypatch)
    case.task = replace(case.task, board_namespace="foreign-board")
    with pytest.raises(ValidationProjectDependencyPreflightDeferred):
        _preflight(case)
    assert (case.workspace / case.target).read_bytes() == case.before
    assert _git(case.workspace, "status", "--porcelain") == ""
    assert _git(case.workspace, "rev-parse", "HEAD") == case.baseline


class _ProviderCommandReached(BaseException):
    """Stop immediately after the actual pre-dispatch owners admit the retry."""


def test_modify_retry_reaches_provider_command_only_after_post_replay_preflight(tmp_path, monkeypatch):
    case = _modify_retry(tmp_path, monkeypatch)
    state = PortalTaskState(
        last_implementation_commit=case.seed,
        last_implementation_branch="prior-attempt",
        last_implementation_returncode=78,
    )
    prompt = case.daemon._build_implementation_prompt(case.task, attempt=2)
    assert case.task.title in prompt
    reached = []

    def stop_before_provider(workspace, **kwargs):
        # This is an effect barrier, not a replacement admission decision.
        assert kwargs["prompt"] == prompt
        assert kwargs["attempt"] == 2
        assert [receipt["passed"] for receipt in case.calls] == [True, True]
        assert [receipt["projects"][0]["scoped_validation_target_materialization_state"] for receipt in case.calls] == ["present", "authenticated-prior-seed"]
        assert (workspace / case.target).read_bytes() == case.after
        assert _git(workspace, "rev-parse", "HEAD") == case.baseline
        reached.append(workspace)
        raise _ProviderCommandReached()

    monkeypatch.setattr(case.daemon, "_build_implementation_command", stop_before_provider)
    with pytest.raises(_ProviderCommandReached):
        case.daemon._run_implementation_in_ephemeral_worktree(
            task=case.task, state=state, attempt=2, started_at=module.utc_now(),
            log_path=case.daemon.implementation_log_dir / "retry.log", prompt=prompt,
        )
    assert len(reached) == 1


def test_modify_retry_refuses_provider_admission_after_candidate_byte_drift(tmp_path, monkeypatch):
    case = _modify_retry(tmp_path, monkeypatch)
    state = PortalTaskState(
        last_implementation_commit=case.seed,
        last_implementation_branch="prior-attempt",
        last_implementation_returncode=78,
    )
    prompt = case.daemon._build_implementation_prompt(case.task, attempt=2)
    real_preflight = module.preflight_validation_project_dependencies
    workspaces = []

    def drift_at_post_replay_boundary(workspace, commands, **kwargs):
        if kwargs.get("prior_seed_authority"):
            assert (workspace / case.target).read_bytes() == case.after
            (workspace / case.target).write_bytes(case.after + b"\n# changed after proposal acceptance\n")
            workspaces.append(workspace)
        return real_preflight(workspace, commands, **kwargs)

    monkeypatch.setattr(module, "preflight_validation_project_dependencies", drift_at_post_replay_boundary)
    monkeypatch.setattr(case.daemon, "_build_implementation_command", lambda *_args, **_kwargs: pytest.fail("dependency refusal reached provider command"))
    result = case.daemon._run_implementation_in_ephemeral_worktree(
        task=case.task, state=state, attempt=2, started_at=module.utc_now(),
        log_path=case.daemon.implementation_log_dir / "retry.log", prompt=prompt,
    )
    assert result["provider_dispatched"] is False
    assert len(workspaces) == 1
    assert [receipt["passed"] for receipt in case.calls] == [True, False]
    assert case.calls[-1]["projects"][0]["contract_error_reason"] == "v2_prior_seed_target_content_mismatch"
    assert _git(case.repo, "rev-parse", "main") == case.baseline
