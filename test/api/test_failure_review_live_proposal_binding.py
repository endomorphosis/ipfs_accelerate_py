"""A genuine live binding is not missing candidate-path diagnostics."""
from copy import deepcopy
from dataclasses import asdict
import subprocess
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as runtime
from ipfs_accelerate_py.agent_supervisor.todo_daemon import portal_task_state_control_plane as plane
from ipfs_accelerate_py.agent_supervisor.validation import proposal_validation as proposals
from ipfs_accelerate_py.agent_supervisor.proof.code_proof_obligations import CandidateDiffEntry, DiffChangeKind


INCOMPLETE = "diagnostic_path_evidence_incomplete"
HARD_DENIAL = "protected_path_forbidden"
OMITTED = "[truncated original_bytes=41 sha256=" + "a" * 64 + " omitted_items=2]"


def _git(root, *args):
    result = subprocess.run(["git", *args], cwd=root, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setenv(plane.ENV_DUCKDB, str(tmp_path / "control-state.duckdb"))
    monkeypatch.setenv(plane.ENV_DUCKLAKE, str(tmp_path / "history"))
    monkeypatch.delenv(plane.ENV_QUACK, raising=False)
    for name in ("IPFS_ACCELERATE_AGENT_STATE_GRANT_BROKER_SOCKET",
                 "IPFS_ACCELERATE_AGENT_STATE_GRANT_BROKER_SECRET_FD"):
        monkeypatch.delenv(name, raising=False)
    repo = tmp_path / "repo"
    repo.mkdir(mode=0o700)
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "Fixture")
    _git(repo, "config", "user.email", "fixture@example.invalid")
    board = repo / "todo.md"
    board.write_text("# Tasks\n\n## LIVE-001 Diagnose a genuine validation failure\n\n"
                     "- Status: todo\n- Completion: validation\n- Outputs: feature.py, test_feature.py\n"
                     "- Validation: python -m unittest -q test_feature.py\n"
                     "- Acceptance: The implementation passes the actual test.\n")
    (repo / "feature.py").write_text("VALUE = 1\n")
    (repo / "test_feature.py").write_text(
        "import unittest\nfrom feature import VALUE\n\nclass TestFeature(unittest.TestCase):\n"
        "    def test_actual_value(self):\n        self.assertEqual(VALUE, 3)\n")
    (repo / ".gitignore").write_text("__pycache__/\n.pytest_cache/\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "synthetic pending contract")
    baseline = _git(repo, "rev-parse", "HEAD")
    state_dir = tmp_path / "state"
    daemon = runtime.PortalImplementationDaemon(
        todo_path=board, state_path=state_dir / "task-state.json",
        strategy_path=state_dir / "strategy.json", events_path=state_dir / "events.jsonl",
        repo_root=repo, task_header_prefix="## LIVE-", merge_target_branch="main",
        max_task_attempts=3, implementation_timeout=30, implementation_protected_paths=())
    task = runtime.parse_task_file(board, "## LIVE-")[0]
    identity = daemon._identity_for_task(task)
    state = runtime.PortalTaskState(implementation_attempts={task.task_id: 1},
        implementation_attempts_by_cid={identity.canonical_task_cid: 1},
        task_identities={task.task_id: identity.to_dict()})
    state.save(daemon.state_path)
    return SimpleNamespace(repo=repo, board=board, daemon=daemon, task=task,
                           baseline=baseline, state=state, state_dir=state_dir)


def _live():
    entry = CandidateDiffEntry(old_path="feature.py", new_path="feature.py",
        change_kind=DiffChangeKind.MODIFY, before_source="VALUE = 1\n", after_source="VALUE = 2\n")
    proposal = proposals.ImplementationProposal(task_id="LIVE-001", accepted_plan_id="plan:fixture",
        repository_id="repository:fixture", repository_tree_id="tree:fixture", objective_id="goal:fixture",
        baseline_id="baseline:fixture", candidate_diff=(entry,), declared_paths=("feature.py",))
    result = proposals.validate_implementation_proposal(proposal,
        policy=proposals.ProposalValidationPolicy(allowed_paths=("feature.py",)))
    assert type(result) is proposals.ProposalValidationResult and result.accepted
    return result


def _payload(live):
    return {"passed": False, "attempted": True, "returncode": 1,
        "reason": "declared_validation_failed", "error": "validation_command_failed",
        "failed_commands": ["python -m unittest -q test_feature.py"],
        "proposal_gate": {"accepted": True, "changed_paths": list(live.proposal.changed_paths),
                          "reason_codes": []},
        "proposal_validation": live}


def _review(world, result, live):
    return world.daemon._apply_implementation_failure_review(
        task=world.task, attempt=1, workspace_path=world.repo,
        validation_result=result, proposal_validation=live, baseline_ref=world.baseline)


@pytest.mark.parametrize("untracked_test", [False, True])
def test_real_native_failing_command_retains_live_binding_and_accurate_paths(world, untracked_test):
    if untracked_test:
        # Preserve identical file bytes but make the declared new test absent
        # from the baseline, matching Native's pre-commit task10 candidate.
        test_source = (world.repo / "test_feature.py").read_text()
        _git(world.repo, "rm", "test_feature.py")
        _git(world.repo, "commit", "-m", "declared test will be newly materialized")
        world.baseline = _git(world.repo, "rev-parse", "HEAD")
        (world.repo / "test_feature.py").write_text(test_source)
    (world.repo / "feature.py").write_text("VALUE = 2\n")
    result = world.daemon._run_validation_with_candidate_binding(
        world.repo, world.task, world.state_dir / "validation.log", baseline_ref=world.baseline)
    live = result["proposal_validation"]
    assert type(live) is proposals.ProposalValidationResult and live.accepted
    assert result["passed"] is False and result["error"] == "validation_command_failed"
    assert result["failed_commands"]
    assert "AssertionError: 2 != 3" in (world.state_dir / "validation.log").read_text()
    gate = deepcopy(result["proposal_gate"])
    returned = _review(world, result, live)
    review = returned["failure_review"]
    assert review["decision"] == "guide_rescue" and review["accepted"] is False
    assert "validation_command_failed" in review["reason_codes"]
    assert INCOMPLETE not in review["reason_codes"]
    assert returned["proposal_validation"] is result["proposal_validation"] is live
    assert returned["proposal_gate"] == result["proposal_gate"] == gate
    assert "failure_review" not in result
    assert returned["passed"] is False and returned["returncode"] != 0
    assert asdict(runtime.PortalTaskState.load(world.daemon.state_path)) == asdict(world.state)
    assert runtime.parse_task_file(world.board, "## LIVE-")[0].status == "todo"


@pytest.mark.parametrize("case", ["missing_gate", "missing_paths", "none_paths", "truncated",
                                 "empty_paths", "foreign_paths", "oversized"])
def test_opaque_or_incomplete_gate_cannot_gain_scope_acceptance(world, case):
    live = _live()
    result = _payload(live)
    result.update(reason="proposal_gate_failed", error="proposal_validation_failed", failed_commands=[])
    result["selection"] = {"changed_files": ["src/allowed.py"]}
    result["scope_adjudication"] = {"accepted": True, "justified_paths": ["src"]}
    result["proposal_gate"]["reason_codes"] = ["path_outside_scope"]
    if case == "missing_gate":
        result.pop("proposal_gate")
    elif case == "missing_paths":
        result["proposal_gate"].pop("changed_paths")
    else:
        result["proposal_gate"]["changed_paths"] = {
            "none_paths": None, "truncated": [OMITTED], "empty_paths": [],
            "foreign_paths": ["foreign/other.py"], "oversized": ["src/allowed.py"] * 129}[case]
    returned = _review(world, result, live)
    review = returned["failure_review"]
    assert INCOMPLETE in review["reason_codes"]
    assert review["accepted"] is False and returned["passed"] is False
    if case != "oversized":
        assert review["changed_paths"] != ["src/allowed.py"]
    assert returned["proposal_validation"] is result["proposal_validation"] is live


@pytest.mark.parametrize("case", ["different_object", "lookalike", "subclass"])
def test_only_identical_exact_native_result_is_normalized(world, case):
    live = _live()
    result = _payload(live)
    if case == "different_object":
        result["proposal_validation"] = _live()
    elif case == "lookalike":
        live = SimpleNamespace(accepted=True, proposal=live.proposal)
        result["proposal_validation"] = live
    else:
        subclass = type("LookalikeSubclass", (type(live),), {})
        live = subclass(proposal=live.proposal, policy=live.policy, receipt=live.receipt)
        result["proposal_validation"] = live
    returned = _review(world, result, live)
    assert INCOMPLETE in returned["failure_review"]["reason_codes"]
    assert returned["failure_review"]["accepted"] is False
    assert returned["proposal_validation"] is result["proposal_validation"]


def test_normalized_live_binding_still_retains_hard_denial(world):
    live = _live()
    result = _payload(live)
    result["proposal_gate"]["reason_codes"] = [HARD_DENIAL]
    returned = _review(world, result, live)
    assert returned["failure_review"]["decision"] == "reject"
    assert HARD_DENIAL in returned["failure_review"]["finding_codes"]
    assert returned["failure_review"]["accepted"] is False
    assert returned["proposal_validation"] is live
