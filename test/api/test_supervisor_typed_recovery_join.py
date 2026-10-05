"""Typed recovery capture must survive the actual accepted-tree boundary."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_implementation_route import AgentImplementationControlPlanePin
from ipfs_accelerate_py.agent_supervisor.runtime import multi_supervisor_runner as runner
from ipfs_accelerate_py.agent_supervisor.runtime.artifact_store import BoundedArtifactStore
from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import DatabasePortalExecutionBridge
from ipfs_accelerate_py.agent_supervisor.validation.project_dependency_preflight import (
    canonical_project_dependency_preflight_receipt_bytes,
    project_dependency_preflight_error_receipt,
)


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-c", "user.name=Recovery Fixture", "-c", "user.email=fixture@example.invalid", *args],
        cwd=root, text=True,
    ).strip()


def _runtime(tmp_path: Path):
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    (root / ".gitignore").write_text("data/recovery/run-v1/\n")
    (root / "code.py").write_text("value = 1\n")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "accepted source")
    head, tree = _git(root, "rev-parse", "HEAD"), _git(root, "rev-parse", "HEAD^{tree}")
    runtime = root / "data/recovery/run-v1"
    roots = tuple(runtime / name for name in ("state", "worktrees", "merge-queue"))
    for directory in (runtime, *roots, roots[0] / "lane-0"):
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        directory.chmod(0o700)
    state = roots[0] / "lane-0"
    store_root = state / "dependency-preflight-artifacts"
    receipt = project_dependency_preflight_error_receipt(root, ["true"], RuntimeError("fixture"))
    with BoundedArtifactStore(store_root) as store:
        reference = store.put_blob(
            canonical_project_dependency_preflight_receipt_bytes(receipt),
            kind="validation_project_dependency_preflight_receipt",
            retention_class="checkpoint", media_type="application/json",
        )
    bindings = ({"slice_id": "slice-0", "lane_id": "lane-0", "lane_index": 0,
                 "task_ids": ("FIX-1",), "task_cids": ("task:fixture",)},)
    capture = dict(root=root, runtime_roots=roots,
                   owner_bound_artifacts=(state / "launch.log", state / "other-owned.log"),
                   runtime_bindings=bindings, slice_id="slice-0", lane_id="lane-0",
                   state_dir=state, state_prefix="fixture_lane_0")
    # Pin construction/sealed-descriptor verification is a separate boundary.
    # This test supplies the typed pin to the production repository validator.
    pin = AgentImplementationControlPlanePin(
        schema="fixture", runner_path="fixture", runner_sha256="a" * 64,
        capsule_root="fixture", capsule_id="fixture", source_head=head,
        source_tree=tree, archive_sha256="a" * 64,
    )
    birth = dict(
        accepted_tree_root=root, source_head=head, source_tree=tree,
        control_plane_pin=pin, recovery_repository_head=head, recovery_repository_tree=tree,
        recovery_runtime_roots=roots, recovery_owner_bound_artifacts=capture["owner_bound_artifacts"],
        recovery_runtime_bindings=bindings, recovery_slice_id="slice-0", recovery_lane_id="lane-0",
        recovery_state_dir=state, recovery_state_prefix="fixture_lane_0",
        recovery_launch_artifact_paths=(state / "launch.log",),
    )
    digest = reference.digest.removeprefix("sha256:")
    blob = store_root / "blobs/sha256" / digest[:2] / (digest + ".blob")
    return capture, birth, store_root, blob


@pytest.mark.parametrize("mutation", ["none", "blob", "manifest", "missing", "foreign", "symlink", "late_signed", "late_unsigned"])
def test_snapshot_revalidates_at_accepted_tree_boundary(tmp_path: Path, mutation: str) -> None:
    capture, birth, store_root, blob = _runtime(tmp_path)
    birth["recovery_artifacts"] = runner._snapshot_plan_bound_recovery_artifacts(**capture)
    runner._validate_plan_bound_accepted_tree(**birth)
    if mutation == "blob":
        blob.write_bytes(blob.read_bytes() + b" ")
    elif mutation == "manifest":
        with BoundedArtifactStore(store_root):
            pass
    elif mutation == "missing":
        blob.unlink()
    elif mutation == "foreign":
        (capture["state_dir"] / "foreign.json").write_text("{}")
    elif mutation == "symlink":
        retained = tmp_path / "outside.blob"
        blob.rename(retained)
        blob.symlink_to(retained)
    elif mutation in {"late_signed", "late_unsigned"}:
        name = "launch.log" if mutation == "late_signed" else "other-owned.log"
        target = capture["state_dir"] / name
        target.write_bytes(b"owned launch fixture\n")
        target.chmod(0o600)
    if mutation in {"none", "late_signed"}:
        runner._validate_plan_bound_accepted_tree(**birth)
    else:
        with pytest.raises(ValueError):
            runner._validate_plan_bound_accepted_tree(**birth)


def _portal(capture):
    attempt = SimpleNamespace(
        task_cid="task:fixture", task_alias="FIX-1", attempt_id="attempt:fixture",
        claim_id="claim:fixture", attempt_number=1, lease_id="lease:fixture",
        owner_session_id="session:fixture", fencing_token=1, fence_epoch=1, body={},
    )
    record = SimpleNamespace(task_alias="FIX-1", revision=1, goal_cid="", plan_cid="", body={})
    portal_root = capture["state_dir"] / "fixture_lane_0_database_portal_attempts"
    attempt_root = portal_root / hashlib.sha256(attempt.attempt_id.encode()).hexdigest()[:24]
    attempt_root.mkdir(parents=True, mode=0o700)
    portal_root.chmod(0o700)
    paths = DatabasePortalExecutionBridge._direct_attempt_paths(attempt_root)
    projection = DatabasePortalExecutionBridge._render_projection_seed(attempt, record)
    binding = DatabasePortalExecutionBridge._binding(attempt, record, projection)
    paths.binding.write_text(json.dumps(binding, sort_keys=True, indent=2) + "\n")
    paths.task_projection.write_text(projection)
    for path in (paths.binding, paths.task_projection):
        path.chmod(0o600)
    return portal_root, attempt_root, paths


@pytest.mark.parametrize("mutation", ["none", "task_cid", "directory", "projection", "duplicate_json", "foreign_log"])
def test_current_owner_portal_projection_is_bound_to_the_selected_slice(tmp_path: Path, mutation: str) -> None:
    capture, birth, _, _ = _runtime(tmp_path)
    portal_root, attempt_root, paths = _portal(capture)
    if mutation == "task_cid":
        capture["runtime_bindings"] = ({**capture["runtime_bindings"][0], "task_cids": ("task:foreign",)},)
    elif mutation == "directory":
        attempt_root.rename(portal_root / ("f" * 24))
    elif mutation == "projection":
        paths.task_projection.write_text(paths.task_projection.read_text() + "- Outputs: foreign.py\n")
    elif mutation == "duplicate_json":
        original = paths.binding.read_text()
        paths.binding.write_text('{"schema":"duplicate",' + original[1:])
    elif mutation == "foreign_log":
        paths.implementation_logs.mkdir(mode=0o700)
        (paths.implementation_logs / "foreign-attempt-1.log").write_text("foreign")
    if mutation == "none":
        evidence = runner._snapshot_plan_bound_recovery_artifacts(**capture)
        assert paths.task_projection.relative_to(capture["root"]).as_posix() in {item["path"] for item in evidence}
        runner._validate_plan_bound_accepted_tree(**birth, recovery_artifacts=evidence)
    else:
        with pytest.raises(ValueError):
            runner._snapshot_plan_bound_recovery_artifacts(**capture)


@pytest.mark.parametrize("reader", ["stable", "absent", "lock"])
def test_recovery_open_race_cannot_block_on_fifo(tmp_path: Path, reader: str) -> None:
    script = r'''
import os, sys
from pathlib import Path
from ipfs_accelerate_py.agent_supervisor.runtime import multi_supervisor_runner as runner
root = Path(sys.argv[1]); path = root / "target"
kind = sys.argv[2]
if kind != "absent": path.write_bytes(b"")
original_open = os.open
def swapped_open(candidate, flags, *args, **kwargs):
    if Path(candidate) == path:
        if path.exists(): path.unlink()
        os.mkfifo(path)
    return original_open(candidate, flags, *args, **kwargs)
os.open = swapped_open
try:
    if kind == "lock": runner._acquire_plan_bound_dependency_preflight_lock(root, path)
    else: runner._read_stable_regular_bytes(path)
except (ValueError, runner._StableArtifactReadError):
    print("refused", flush=True)
else:
    raise AssertionError("FIFO replacement was accepted")
'''
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(tmp_path), reader],
        capture_output=True, text=True, timeout=10, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "refused"


@pytest.mark.parametrize("mutation", ["none", "foreign_task", "foreign_event", "authority", "oversized", "duplicate", "late_mutation"])
def test_portal_guidance_joins_current_projection_and_retains_immutable_bytes(tmp_path: Path, mutation: str) -> None:
    from ipfs_accelerate_py.agent_supervisor.runtime.prior_seed_guidance import (
        SCHEMA, guidance_artifact_path,
    )
    capture, birth, _, _ = _runtime(tmp_path)
    _, attempt_root, paths = _portal(capture)
    identity = DatabasePortalExecutionBridge._projection_task_identity(
        paths.task_projection.read_text(), projection_path=paths.task_projection,
        task_alias="FIX-1",
    )
    binding = dict(schema=SCHEMA, **identity, attempt=2, event_source=str(attempt_root / "portal-events.jsonl"))
    if mutation == "foreign_task":
        binding["canonical_task_cid"] = "task:foreign"
    if mutation == "foreign_event":
        binding["event_source"] = str(attempt_root / "foreign-events.jsonl")
    artifact = guidance_artifact_path(binding)
    artifact.parent.mkdir(mode=0o700)
    guidance = "Use the current source; the former seed changed an unrelated file."
    record = dict(binding, guidance=guidance, guidance_sha256=hashlib.sha256(guidance.encode()).hexdigest(),
                  status="pending", candidate_worktree=str(tmp_path / "candidate"),
                  proof_authority=False, execution_authority=False, completion_authority=False)
    if mutation == "authority":
        record["proof_authority"] = True
    raw = json.dumps(record, sort_keys=True) + "\n"
    if mutation == "oversized":
        raw += " " * 65536
    elif mutation == "duplicate":
        raw = '{"schema":"duplicate",' + raw[1:]
    artifact.write_text(raw)
    artifact.chmod(0o600)
    if mutation in {"none", "late_mutation"}:
        evidence = runner._snapshot_plan_bound_recovery_artifacts(**capture)
        assert artifact.relative_to(capture["root"]).as_posix() in {item["path"] for item in evidence}
        runner._validate_plan_bound_accepted_tree(**birth, recovery_artifacts=evidence)
        if mutation == "late_mutation":
            record["status"] = "consumed"
            artifact.write_text(json.dumps(record, sort_keys=True) + "\n")
            with pytest.raises(ValueError):
                runner._validate_plan_bound_accepted_tree(**birth, recovery_artifacts=evidence)
    else:
        with pytest.raises(ValueError):
            runner._snapshot_plan_bound_recovery_artifacts(**capture)


@pytest.mark.parametrize("filename", ["fixture_lane_1_supervisor.pid", "fixture_lane_1_supervisor.out", ".fixture_lane_1_supervisor.pid.update.lock", "fixture_lane_1_managed_daemon.pid", "fixture_lane_1_ensure_status.json"])
def test_reassigned_recovery_custody_checks_ordinary_live_sibling(tmp_path: Path, filename: str) -> None:
    capture, _, _, _ = _runtime(tmp_path)
    lane_id = "recovery-1-0123456789ab"
    selected = {**capture["runtime_bindings"][0], "lane_id": lane_id}
    sibling = {"slice_id": "slice-1", "lane_id": "lane-1", "lane_index": 1,
               "task_ids": ("FIX-2",), "task_cids": ("task:sibling",)}
    state = capture["runtime_roots"][0] / lane_id
    capture["state_dir"].rename(state)
    capture.update(state_dir=state, lane_id=lane_id, state_prefix=runner._plan_bound_reassigned_state_prefix(selected),
                   runtime_bindings=(selected, sibling), owner_bound_artifacts=())
    sibling_dir = capture["runtime_roots"][0] / "lane-1"
    sibling_dir.mkdir(mode=0o700)
    artifact = sibling_dir / filename
    artifact.write_text("mutable sibling state\n")
    artifact.chmod(0o600)
    evidence = runner._snapshot_plan_bound_recovery_artifacts(**capture)
    assert artifact.relative_to(capture["root"]).as_posix() not in {item["path"] for item in evidence}
    artifact.chmod(0o666)
    with pytest.raises(ValueError):
        runner._snapshot_plan_bound_recovery_artifacts(**capture)
