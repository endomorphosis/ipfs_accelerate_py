"""Retained donor custody derives from an actual fenced owner transfer."""
from __future__ import annotations

import hashlib
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.control import plan_execution_store as owner
from ipfs_accelerate_py.agent_supervisor.control.lifecycle_orchestrator import LifecycleProfile
from ipfs_accelerate_py.agent_supervisor.control.plan_execution_store import (
    ProductionParallelPlanAdapter,
)
from ipfs_accelerate_py.agent_supervisor.control.profile_authority import DEFAULT_SCOPED_ROUTE_ID
from ipfs_accelerate_py.agent_supervisor.runtime import multi_supervisor_runner as runner
from ipfs_accelerate_py.agent_supervisor.task_sources.plan_revision_store import PlanRevisionStore
from test.api.test_agent_supervisor_configured_board_scheduler import (
    _fenced_plan_children,
    _isolated_local_profile_lifecycle_registry,
)


def _transfer(tmp_path: Path):
    repo, _board, _receipt, donor, recipient, process = _fenced_plan_children(
        tmp_path, models=("grok-4.7", "gpt-6.1-sol"), route_id=DEFAULT_SCOPED_ROUTE_ID,
    )
    token = hashlib.sha256(f"{donor.revision_cid}:{donor.slice_id}:1".encode()).hexdigest()[:12]
    lane_id = f"recovery-1-{token}"
    recipient = replace(
        recipient, name=lane_id, lane_id=lane_id,
        state_dir=str(Path(donor.state_dir).parent / lane_id),
        state_prefix=f"recovery_1_{token}",
    )
    adopted = runner.reassign_fenced_plan_bound_child(
        donor=donor, recipient=recipient, donor_process=process, repo_root=repo,
    )
    adapter = ProductionParallelPlanAdapter(PlanRevisionStore(repo / adopted.plan_revision_store_path))
    return repo, donor, adopted, process, adapter


def test_runtime_binding_retains_exact_fenced_donor_scope(tmp_path: Path) -> None:
    repo, donor, adopted, process, adapter = _transfer(tmp_path)
    bindings = adapter.recovery_runtime_bindings(
        revision_cid=adopted.revision_cid,
        slice_manifest_cid=adopted.slice_manifest_cid,
    )
    selected = next(item for item in bindings if item["slice_id"] == adopted.slice_id)
    scopes = selected["retained_donor_scopes"]
    assert len(scopes) == 1
    scope = scopes[0]
    transfer_cid, transfer = adapter.load_slice_reassignment(
        revision_cid=adopted.revision_cid, slice_id=adopted.slice_id,
    )
    assert scope == {
        "lane_id": donor.lane_id,
        "state_dir": str(repo / donor.state_dir),
        "state_prefix": donor.state_prefix,
        "reassignment_cid": transfer_cid,
        "donor_process_birth_cid": transfer.donor_process_birth_cid,
        "launch_process_birth_cid": process._agent_supervisor_process_birth_cid,
        "attempt_absence_cid": transfer.attempt_absence_cid,
    }
    assert selected["lane_id"] == adopted.lane_id
    assert all(item["lane_id"] != donor.lane_id for item in bindings)
    assert all(item["retained_donor_scopes"] == [] for item in bindings if item is not selected)


@pytest.mark.parametrize("mutation", [
    "duplicate_directory", "missing_prefix", "foreign_directory", "prefix_mismatch",
    "traversal", "launch_profile_mismatch", "attempted", "absence_path",
])
def test_scope_projection_rejects_mixed_profile_and_absence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str,
) -> None:
    """Exercise scope joins after the existing owner has verified the chain."""
    repo, donor, adopted, _process, adapter = _transfer(tmp_path)
    reassignment = adapter.load_slice_reassignment(
        revision_cid=adopted.revision_cid, slice_id=adopted.slice_id,
    )
    assert reassignment is not None
    transfer = reassignment[1]
    read = owner._secure_store_cas
    process = read(adapter.plan_revision_store, transfer.donor_process_birth_cid)
    profile = LifecycleProfile.from_dict(process["profile"])
    argv = list(profile.argv)
    if mutation == "duplicate_directory":
        argv += ["--state-dir", donor.state_dir]
    elif mutation == "missing_prefix":
        position = argv.index("--state-prefix")
        del argv[position:position + 2]
    elif mutation in {"foreign_directory", "traversal"}:
        value = str(Path(donor.state_dir).parent / "foreign")
        if mutation == "traversal":
            value = str(Path(donor.state_dir).parent / "transit" / ".." / donor.lane_id)
        argv[argv.index("--state-dir") + 1] = value
    elif mutation in {"prefix_mismatch", "launch_profile_mismatch"}:
        argv[argv.index("--state-prefix") + 1] = "foreign_lane_0"
    changed_profile = replace(profile, argv=tuple(argv), profile_id="").to_dict()

    def changed_cas(store, cid):
        payload = read(store, cid)
        if cid == transfer.donor_process_birth_cid:
            return {**payload, "profile": changed_profile}
        if cid == process["launch_process_birth_cid"] and mutation != "launch_profile_mismatch":
            return {**payload, "profile": changed_profile}
        if cid == transfer.attempt_absence_cid:
            if mutation == "attempted":
                return {**payload, "never_attempted": False}
            if mutation == "absence_path":
                return {**payload, "state_path": str(repo / "foreign_task_state.json")}
        return payload

    monkeypatch.setattr(owner, "_secure_store_cas", changed_cas)
    with adapter.plan_revision_store._thread_lock, adapter.plan_revision_store._guard():
        with pytest.raises(owner.ExecutionPlanError, match="retained donor"):
            adapter._retained_donor_runtime_scopes_locked(reassignment)


def test_runtime_binding_rejects_redirected_retained_donor_state(tmp_path: Path) -> None:
    repo, donor, adopted, _process, adapter = _transfer(tmp_path)
    state = repo / donor.state_dir
    outside = tmp_path / "redirected-donor"
    if state.exists():
        state.rename(outside)
    else:
        outside.mkdir()
    state.symlink_to(outside, target_is_directory=True)
    # Reconstructing the signed profile already detects the redirected state
    # root, before the new custody projection can admit it.
    with pytest.raises(owner.ExecutionPlanError, match="donor process fence lifecycle evidence is invalid"):
        adapter.recovery_runtime_bindings(
            revision_cid=adopted.revision_cid,
            slice_manifest_cid=adopted.slice_manifest_cid,
        )


def test_retained_donor_authority_reader_cannot_block_on_fifo_swap(tmp_path: Path) -> None:
    script = r'''
import os, sys
from pathlib import Path
from ipfs_accelerate_py.agent_supervisor.control import plan_execution_store as owner
path = Path(sys.argv[1])
path.write_bytes(b"{}")
path.chmod(0o600)
original_open = os.open
swapped = False
def swap_open(target, flags, *args, **kwargs):
    global swapped
    if Path(target) == path and not swapped:
        swapped = True
        path.unlink()
        os.mkfifo(path, 0o600)
    return original_open(target, flags, *args, **kwargs)
owner.os.open = swap_open
try:
    owner._stable_authority_json(path)
except owner.ExecutionPlanError as exc:
    assert swapped and "changed before open" in str(exc), str(exc)
else:
    raise AssertionError("FIFO replacement was accepted")
'''
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "authority.json")],
        cwd=Path(owner.__file__).resolve().parents[3],
        capture_output=True, text=True, timeout=20, check=False,
    )
    assert completed.returncode == 0, completed.stderr
