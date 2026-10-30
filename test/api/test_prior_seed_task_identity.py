"""A lane's last commit cannot donate overlapping bytes to another task."""
from dataclasses import asdict, replace

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import PortalTaskState
from test.api.test_agent_supervisor_prior_attempt_seed import _daemon, _task, _git, _authorize_seed


@pytest.fixture
def seed_case(tmp_path):
    repo = tmp_path / "repo"; repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "Fixture")
    _git(repo, "config", "user.email", "fixture@example.invalid")
    page = repo / "templates/workspace.html"; page.parent.mkdir(); page.write_text("<main>Chat</main>\n")
    _git(repo, "add", "."); _git(repo, "commit", "-m", "base")
    baseline = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-b", "prior")
    page.write_text("<main>Evidence library</main>\n")
    _git(repo, "add", "."); _git(repo, "commit", "-m", "candidate")
    candidate = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "main")
    daemon = _daemon(repo)
    chat = replace(_task("templates/workspace.html"), task_id="CIMP-UX-07", title="Chat presentation")
    evidence = replace(chat, task_id="CIMP-UX-15", title="Evidence library")
    # The old chat receipt really authorizes this overlapping path. It still
    # cannot authorize the bytes of the later evidence-task commit.
    _authorize_seed(daemon, chat, "templates/workspace.html")
    return daemon, chat, evidence, page, baseline, candidate


def state_for(daemon, task, candidate):
    identity = daemon._identity_for_task(task)
    return PortalTaskState(last_implementation_task_id=task.task_id,
        last_implementation_task_cid=identity.canonical_task_cid,
        last_implementation_task_key=identity.canonical_task_key,
        last_implementation_commit=candidate, last_implementation_branch="prior",
        implementation_attempts={task.task_id: 2},
        implementation_attempts_by_cid={identity.canonical_task_cid: 2})


def record_owner(daemon, task, candidate):
    identity = daemon._identity_for_task(task)
    daemon._record_event("failed_validation_worktree_preserved", {"task_id": task.task_id,
        "canonical_task_cid": identity.canonical_task_cid,
        "canonical_task_key": identity.canonical_task_key,
        "board_namespace": identity.board_namespace,
        "implementation_commit": candidate, "preserved_commit": candidate, "preserved": True})


@pytest.mark.parametrize("damage", ["other_task", "foreign_commit", "revised_task", "missing_commit_owner"])
def test_foreign_or_unproved_seed_cannot_replay_overlapping_output(seed_case, damage):
    daemon, chat, evidence, page, baseline, candidate = seed_case
    state = state_for(daemon, evidence if damage == "other_task" else chat, candidate)
    if damage != "missing_commit_owner": record_owner(daemon, evidence, candidate)
    requested = replace(chat, title="Revised chat contract") if damage == "revised_task" else chat
    before = (page.read_bytes(), _git(daemon.repo_root, "rev-parse", "HEAD"),
        _git(daemon.repo_root, "ls-files", "--stage"), asdict(state))
    assert daemon._prior_seed_proposal_authority(chat)["ok"] is True
    plan = daemon._prior_attempt_seed_plan(task=requested, state=state, attempt=3)
    assert plan["reuse_prior_attempt"] is False
    assert plan["reason"] in {"prior_attempt_task_identity_mismatch", "prior_seed_commit_task_identity_unproved"}
    applied = daemon._apply_prior_attempt_seed(daemon.repo_root, task=requested, seed_plan=plan, baseline_ref=baseline)
    assert applied["applied"] is False
    assert (page.read_bytes(), _git(daemon.repo_root, "rev-parse", "HEAD"),
        _git(daemon.repo_root, "ls-files", "--stage"), asdict(state)) == before


@pytest.mark.parametrize("branch_fallback", [False, True])
def test_actual_same_task_preserved_commit_remains_available_as_seed(seed_case, branch_fallback):
    daemon, chat, _, page, baseline, candidate = seed_case
    record_owner(daemon, chat, candidate)
    state = state_for(daemon, chat, candidate)
    if branch_fallback: state.last_implementation_commit = ""
    before = (page.read_bytes(), _git(daemon.repo_root, "rev-parse", "HEAD"), asdict(state))
    plan = daemon._prior_attempt_seed_plan(task=chat, state=state, attempt=3)
    assert plan["reuse_prior_attempt"] is True and plan["seed_ref"] == candidate
    assert plan["reason"] == "prior_failed_attempt_commit"
    assert (page.read_bytes(), _git(daemon.repo_root, "rev-parse", "HEAD"), asdict(state)) == before
