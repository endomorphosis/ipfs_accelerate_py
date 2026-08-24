from __future__ import annotations

from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.self_hosting import ExperimentPlan, SelfHostingQualificationHarness
from ipfs_accelerate_py.proof_context import create_ordinary_python_repository
from ipfs_accelerate_py.proof_context.bootstrap import RUNTIME_CID


CID = "bafkreigh2akiscaildc6ic4z2zgw6eo6wgbdbr7r7z5t2wea5l5lyz66ri"


def _plan(kind="live", **task_changes):
    task = {"task_id": "fixture-task", "task_specification_cid": CID, "proposal": {"files": {"src/demo/change.py": "VALUE = 2\n"}, "declared_files": ["src/demo/change.py"]}}
    task.update(task_changes)
    return ExperimentPlan.from_mapping({"engine_id": RUNTIME_CID, "package_id": "ipfs-accelerate-py", "package_identity": "sha256:" + "2" * 64, "repository_id": "fixture/self-hosting", "repository_state_cid": CID, "configuration_id": "fixture-a", "configuration_cid": CID, "evidence_kind": kind, "tasks": [task]})


def test_live_attempt_uses_runtime_and_discards_disposable_worktree(tmp_path: Path):
    repo = create_ordinary_python_repository(tmp_path / "repo")
    before = (repo / ".git").read_text() if (repo / ".git").is_file() else None
    evidence = SelfHostingQualificationHarness(_plan(), repo, worktree_parent=tmp_path / "worktrees").run()
    attempt = evidence["attempts"][0]
    assert evidence["qualification"] is None and evidence["not_a_qualification"] is True
    assert attempt["provenance"] == "live"
    assert attempt["identities"]["repository_id"] == "fixture/self-hosting"
    assert attempt["identities"]["engine_id"] == RUNTIME_CID
    assert attempt["identities"]["configuration_id"] == "fixture-a"
    assert attempt["identities"]["task_specification_cid"] == CID
    assert attempt["worktree"]["cleanup"]["discarded"] is True
    assert before is None


def test_replay_is_labelled_and_never_runs_a_worktree(tmp_path: Path):
    plan = _plan("replayed", replay_record={"status": "verification_failed", "artifact_cid": CID})
    evidence = SelfHostingQualificationHarness(plan).run()
    attempt = evidence["attempts"][0]
    assert attempt["provenance"] == "replayed"
    assert attempt["worktree"]["disposable"] is False
    assert attempt["status"] == "verification_failed"
    assert attempt["failure"]["status"] == "verification_failed"
    assert evidence["evidence_kind"] == "replayed"
    assert evidence["authority"] == {"execution": False, "qualification": False, "self_approval": False}


def test_simulation_is_explicitly_separate_from_live_evidence(tmp_path: Path):
    repo = create_ordinary_python_repository(tmp_path / "repo")
    attempt = SelfHostingQualificationHarness(
        _plan("simulated"), repo, worktree_parent=tmp_path / "worktrees"
    ).run()["attempts"][0]
    assert attempt["provenance"] == "simulated"
    assert attempt["engine_record"]["provenance"] == "simulated"
    assert attempt["worktree"]["cleanup"]["discarded"] is True


def test_bad_patch_preserves_typed_failure_or_runtime_rejection(tmp_path: Path):
    repo = create_ordinary_python_repository(tmp_path / "repo")
    plan = _plan(proposal={"files": {"../escape.py": "nope"}, "declared_files": ["../escape.py"]})
    attempt = SelfHostingQualificationHarness(plan, repo, worktree_parent=tmp_path / "worktrees").run()["attempts"][0]
    assert attempt["status"] in {"rejected", "invalid", "infrastructure_failure"}
    assert attempt["failure"] is not None or attempt["engine_record"] is not None
