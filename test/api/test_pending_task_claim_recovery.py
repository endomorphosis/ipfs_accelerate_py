"""Pending custody release uses real Native source, claims, events and Git."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as native
from ipfs_accelerate_py.agent_supervisor.todo_daemon import pending_task_claim_recovery as recovery
from ipfs_accelerate_py.agent_supervisor.todo_daemon import current_incident_adjudication as common
from test.api.test_agent_supervisor_implementation_protected_paths import (
    _persist_live_shaped_interrupted_attempt, _protected_git_worktree_daemon,
)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    path.chmod(0o600)


@pytest.fixture
def pending(tmp_path, monkeypatch, request):
    monkeypatch.setenv(native.AUTOMATIC_RETRY_REARM_ENV, "0")
    reconciler = native.CrashFenceReconciler
    monkeypatch.setattr(native, "CrashFenceReconciler", lambda daemon: reconciler(daemon, max_hold_seconds=10))
    daemon, repo, workspace, _ = _protected_git_worktree_daemon(tmp_path)
    daemon.state_path = tmp_path / "state/portal_task_state.json"
    daemon.strategy_path = tmp_path / "state/portal_strategy.json"
    if "dependency_marker" in request.fixturenames:
        dependency = tmp_path / "runtime-dependency"
        dependency.mkdir()
        subprocess.run(["git", "init", "-q", str(dependency)], check=True)
        (dependency / "runtime.py").write_text("DEPENDENCY = 1\n")
        subprocess.run(["git", "add", "."], cwd=dependency, check=True)
        subprocess.run(["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "fixture"], cwd=dependency, check=True)
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=dependency, text=True).strip()
        subprocess.run(["git", "update-index", "--add", "--cacheinfo", "160000," + commit + ",external/runtime"], cwd=repo, check=True)
        subprocess.run(["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "fixture dependency"], cwd=repo, check=True)
        baseline = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        subprocess.run(["git", "merge", "--ff-only", baseline], cwd=workspace, check=True, capture_output=True)
        (workspace / "external").mkdir(exist_ok=True)
        subprocess.run(["git", "clone", "--shared", str(dependency), str(workspace / "external/runtime")], check=True, capture_output=True)
    parameter = getattr(request, "param", None)
    ordinal = parameter.get("attempt", 1) if isinstance(parameter, dict) else 1
    candidate = parameter == "candidate" or isinstance(parameter, dict) and parameter.get("candidate", False)
    task, lease, claim_path = _persist_live_shaped_interrupted_attempt(daemon, workspace=workspace, attempt=ordinal)
    directory = daemon.state_path.parent
    directory.chmod(0o700)
    # These births belong to real exited local Python processes. No provider,
    # container or task implementation is launched by this synthetic fixture.
    birth_code = "import json; from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import current_process_birth; print(json.dumps(current_process_birth().to_dict()))"
    wrapper = """import json,subprocess,sys
from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import current_process_birth
launch=current_process_birth().to_dict()
child=json.loads(subprocess.run([sys.executable,'-c',sys.argv[1]],capture_output=True,text=True,check=True).stdout)
print(json.dumps({'launch':launch,'dispatch':child}))
"""
    tree = json.loads(subprocess.run([sys.executable, "-c", wrapper, birth_code],
                                    capture_output=True, text=True, check=True, timeout=30).stdout)
    life = daemon.worktree_lifecycle.load_workspace(workspace)
    life = replace(life, owner=native.ProcessBirthIdentity.from_dict(tree["dispatch"]),
                   lane_id=f"{directory}:0/1:{tree['dispatch']['pid']}")
    write(daemon.worktree_lifecycle.workspace_path_for(workspace), life.to_dict())
    claim = json.loads(claim_path.read_text())
    claim.update(pid=tree["dispatch"]["pid"], owner_process_birth=tree["dispatch"])
    claim["dispatch_authority_cid"] = daemon._implementation_dispatch_authority_cid(claim)
    write(claim_path, claim)
    lease_value = json.loads(lease.read_text())
    lease_value.pop("lease_role", None)
    lease_value.update(pid=tree["dispatch"]["pid"], owner_process_birth=tree["dispatch"])
    write(lease, lease_value)
    start = next(row for row in daemon._iter_events() if row["type"] == "implementation_started")
    log = Path(start["log_path"])
    log.write_text("synthetic invocation metadata; no provider execution\n")
    log.chmod(0o600); os.utime(log, (1, 1))
    # Reissue the synthetic history through Native's authenticated append API,
    # including the same typed active-lifecycle projection as real dispatch.
    daemon.events_path.unlink()
    daemon.events_path.with_name(daemon.events_path.name + ".manifest.json").unlink(missing_ok=True)
    reserved = {"timestamp", "type", "stream_id", "snapshot_id", "sequence", "event_id", "previous_event_id"}
    daemon._record_event("implementation_started", {**{k:v for k,v in start.items() if k not in reserved},
        "worktree_lifecycle": {"state": "active", "fence": life.fence, "lease_id": life.lease_id}})
    if candidate:
        output = workspace / "src/example.py"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("VALUE = 34\n")
        subprocess.run(["git", "add", "--", "src/example.py"], cwd=workspace, check=True, capture_output=True)
        # Explicit synthetic accepted history, not a validation/completion
        # receipt. Its proposal identity is independently checked against the
        # real Native builder in the candidate tests below.
        daemon._record_event("implementation_proposal_validated", {"task_id": task.task_id,
            "accepted": True, "attempted": True, "reason_codes": [],
            "repository_tree_id": start["baseline_ref"],
            "proposal_id": recovery._candidate_proposal_id(daemon, task, workspace, start["baseline_ref"]),
            "receipt_id": "1" * 64, "policy_id": "2" * 64})
        daemon.worktree_lifecycle.mark_settling(workspace, lease_id=life.lease_id,
            expected_fence=life.fence, expected_record=life)
    code = """import json,sys
from pathlib import Path
from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import WorktreeLifecycleStore
store=WorktreeLifecycleStore(Path(sys.argv[1]))
value=store.reclaim_dead_owner_for_controlled_restart(sys.argv[2],expected_state_dir=sys.argv[3])
print(json.dumps(value.to_dict()))
"""
    if getattr(request, "param", None) == "current_maintenance":
        terminal = daemon.worktree_lifecycle.reclaim_dead_owner_for_controlled_restart(
            workspace, expected_state_dir=directory).to_dict()
    else:
        terminal = json.loads(subprocess.run([sys.executable, "-c", code, str(repo), str(workspace), str(directory)],
                                            capture_output=True, text=True, check=True, timeout=30).stdout)
    assert terminal["owner"]["pid"] != claim["pid"]
    assert terminal["owner"]["parent_pid"] != tree["launch"]["pid"]
    cleanup = daemon.reconcile_quiesced_active_attempt()
    assert cleanup["blocked"] and cleanup["reason"] == "task_claim_reconciliation_blocked", cleanup
    launcher = {"status": "stopped", "repo_root": str(repo.resolve()), "state_dir": str(directory),
        "board_path": str(daemon.todo_path), "plan_id": "synthetic-plan", "started_at": 1.0,
        "owner_token": "private-fixture-token-not-for-release-receipt", "argv": ["private-fixture-argv"],
        "identity": {"pid": tree["launch"]["pid"], "start_ticks": tree["launch"]["start_time_ticks"],
                     "boot_id": tree["launch"]["boot_id"]}}
    write(directory / "launcher.json", launcher)
    write(directory / "container_shutdown.json", {"status": "stopped", "quiesced": True,
        "reason": "owned_container_closure_verified", "repo_root": str(repo.resolve()), "state_dir": str(directory),
        "plan_id": "synthetic-plan", "started_at": 1.0,
        "owner_token_sha256": hashlib.sha256(launcher["owner_token"].encode()).hexdigest(),
        "attempt_refunded": False, "completion_authority": False, "provider_callback_settled": False})
    write(daemon.strategy_path, {"blocked_tasks": [], "deprioritized_tasks": []})
    daemon.implement = False
    from ipfs_accelerate_py.agent_supervisor.runtime import grok_cli_runner
    monkeypatch.setattr(grok_cli_runner, "_docker_isolation_binary", lambda: "/synthetic-docker")
    actual_run = subprocess.run
    def run(argv, **kwargs):
        if argv[0] == "/synthetic-docker":
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        return actual_run(argv, **kwargs)
    monkeypatch.setattr(common.subprocess, "run", run)
    return daemon, task, workspace, claim_path


def test_real_pending_claim_release_preserves_state_source_ordinals_and_exact_replay(pending):
    daemon, task, workspace, claim = pending
    before = asdict(native.PortalTaskState.load(daemon.state_path))
    files = {p:p.read_bytes() for p in (daemon.todo_path, daemon.strategy_path,
        daemon.worktree_lifecycle.workspace_path_for(workspace), daemon.state_path.parent / "launcher.json")}
    entries = sorted(str(path) for path in daemon.state_path.parent.rglob("*"))
    request = daemon.review_quiesced_pending_task_claim(task_id=task.task_id)
    assert sorted(str(path) for path in daemon.state_path.parent.rglob("*")) == entries
    result = daemon.reconcile_quiesced_pending_task_claim(request=request)
    assert result["reconciled"] and not result["blocked"], result
    assert not claim.exists()
    assert asdict(native.PortalTaskState.load(daemon.state_path)) == before
    assert all(path.read_bytes() == raw for path,raw in files.items())
    assert all(result[key] is False for key in recovery._FALSE)
    assert result["task_status_unchanged"] and result["counters_preserved"]
    receipt = Path(result["receipt_path"])
    assert receipt.stat().st_mode & 0o777 == 0o600
    assert receipt.parent.stat().st_mode & 0o777 == 0o700
    assert "private-fixture-token" not in receipt.read_text() and "private-fixture-argv" not in receipt.read_text()
    events = daemon._iter_events()
    assert events[-1]["type"] == "implementation_pending_task_claim_released"
    assert not any(event["type"] in recovery._OUTCOMES for event in events)
    again = daemon.reconcile_quiesced_pending_task_claim(request=request)
    assert again["reason"] == "pending_claim_already_released" and again["receipt_id"] == result["receipt_id"], again
    assert daemon._task_attempt(native.PortalTaskState.load(daemon.state_path), task) == 2


@pytest.mark.parametrize("damage", ["runtime", "state", "claim", "source", "waiting", "manual_hold", "quarantine",
    "active", "protected", "production", "implement", "live_birth", "unknown_birth", "foreign_launcher",
    "shutdown", "baseline_dirty", "later_outcome", "later_start", "partial_history", "wrong_lifecycle_index",
    "maintenance_history", "foreign_pin", "replacement_claim"])
def test_refusal_preserves_pending_claim_and_native_accounting(pending, monkeypatch, damage):
    daemon, task, workspace, claim = pending
    request = daemon.review_quiesced_pending_task_claim(task_id=task.task_id)
    if damage == "runtime": request["runtime_revision"] = "sha256:" + "0" * 64
    elif damage in {"state", "active", "waiting"}:
        state = native.PortalTaskState.load(daemon.state_path)
        if damage == "waiting": state.waiting_task_ids = [task.task_id]
        elif damage == "active": state.active_phase = "implementing"
        else: state.implementation_attempts[task.task_id] += 1
        state.save(daemon.state_path)
    elif damage in {"claim", "replacement_claim"}:
        value = json.loads(claim.read_text());value["lease_id"] = "new-authority";write(claim,value)
    elif damage == "source": daemon.todo_path.write_text(daemon.todo_path.read_text().replace("src/example.py", "src/other.py"))
    elif damage == "manual_hold": daemon.todo_path.write_text(daemon.todo_path.read_text() + "\n- Manual hold: true\n")
    elif damage == "quarantine": write(daemon.strategy_path, {"autonomous_unstall_quarantines": [{"task_id": task.task_id}]})
    elif damage == "protected": write(daemon._implementation_protected_active_snapshot_path(), {"held": True})
    elif damage == "production": daemon.production_provider_policy = object()
    elif damage == "implement": daemon.implement = True
    elif damage == "unknown_birth": monkeypatch.setattr(native, "owner_liveness", lambda *a,**kw:native.OwnerLiveness.UNKNOWN)
    elif damage == "live_birth":
        value = json.loads(claim.read_text());birth = native.current_process_birth();value.update(pid=birth.pid,owner_process_birth=birth.to_dict())
        value["dispatch_authority_cid"]=daemon._implementation_dispatch_authority_cid(value);write(claim,value)
    elif damage in {"foreign_launcher", "shutdown"}:
        path=daemon.state_path.parent / ("launcher.json" if damage=="foreign_launcher" else "container_shutdown.json")
        value=json.loads(path.read_text());value["repo_root"]="/foreign";write(path,value)
    elif damage == "baseline_dirty": (workspace / "changed.py").write_text("VALUE = 1\n")
    elif damage == "later_outcome": daemon._record_event("implementation_finished", {"task_id":task.task_id,"attempt":1,"success":False})
    elif damage == "later_start": daemon._record_event("implementation_started", {"task_id":task.task_id,"attempt":2})
    elif damage == "partial_history":
        with daemon.events_path.open("ab") as handle: handle.write(b"{incomplete")
    elif damage == "wrong_lifecycle_index":
        path=Path(request["pins"]["lifecycle_index"]["path"]);value=json.loads(path.read_text());value["fence"]+=1;write(path,value)
    elif damage == "maintenance_history":
        path=daemon.worktree_lifecycle.workspace_path_for(workspace);value=json.loads(path.read_text());value["terminal_reason"]="unproved";write(path,value)
    elif damage == "foreign_pin": request["pins"]["lifecycle"]["path"] = str(workspace / "foreign.json")
    saved = claim.read_bytes();before=asdict(native.PortalTaskState.load(daemon.state_path));board=daemon.todo_path.read_bytes()
    if damage not in {"runtime", "foreign_pin", "claim", "replacement_claim"}:
        with pytest.raises((ValueError, OSError, TypeError)):
            daemon.review_quiesced_pending_task_claim(task_id=task.task_id)
    result=daemon.reconcile_quiesced_pending_task_claim(request=request)
    assert result["blocked"] and not result["reconciled"], (damage,result)
    assert claim.read_bytes()==saved and daemon.todo_path.read_bytes()==board
    assert asdict(native.PortalTaskState.load(daemon.state_path))==before
    assert not (workspace / "foreign.json.update.lock").exists()


@pytest.mark.parametrize("stage", ["prepared", "released"])
def test_interrupted_private_publication_can_resume_exact_request_without_new_authority(pending, monkeypatch, stage):
    daemon, task, _, claim = pending
    request=daemon.review_quiesced_pending_task_claim(task_id=task.task_id)
    before=asdict(native.PortalTaskState.load(daemon.state_path))
    actual=daemon._write_private_task_claim_release_receipt
    def interrupted(path,value):
        if value.get("phase")==stage: raise OSError("synthetic durable-publication interruption")
        return actual(path,value)
    monkeypatch.setattr(daemon,"_write_private_task_claim_release_receipt",interrupted)
    first=daemon.reconcile_quiesced_pending_task_claim(request=request)
    assert first["blocked"]
    assert claim.exists() is (stage=="prepared")
    monkeypatch.setattr(daemon,"_write_private_task_claim_release_receipt",actual)
    resumed=daemon.reconcile_quiesced_pending_task_claim(request=request)
    assert resumed["reconciled"] and not claim.exists(),resumed
    assert asdict(native.PortalTaskState.load(daemon.state_path))==before


def test_released_claim_replacement_cannot_be_deleted_by_old_receipt(pending):
    daemon,task,_,claim=pending
    request=daemon.review_quiesced_pending_task_claim(task_id=task.task_id)
    original=claim.read_bytes()
    assert daemon.reconcile_quiesced_pending_task_claim(request=request)["reconciled"]
    claim.write_bytes(original);claim.chmod(0o600)
    result=daemon.reconcile_quiesced_pending_task_claim(request=request)
    assert result["blocked"] and claim.read_bytes()==original


def test_ignored_declared_output_is_not_an_unchanged_pending_attempt(pending):
    daemon,task,workspace,claim=pending
    request=daemon.review_quiesced_pending_task_claim(task_id=task.task_id)
    path=Path(subprocess.run(["git","rev-parse","--git-path","info/exclude"],cwd=workspace,
        capture_output=True,text=True,check=True).stdout.strip())
    if not path.is_absolute(): path=workspace/path
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a") as handle: handle.write("\nsrc/example.py\n")
    target=workspace/"src/example.py";target.parent.mkdir();target.write_text("OPERATOR_EDIT = True\n")
    assert not subprocess.run(["git","status","--porcelain","--untracked-files=all"],cwd=workspace,
        capture_output=True,text=True,check=True).stdout.strip()
    result=daemon.reconcile_quiesced_pending_task_claim(request=request)
    assert result["blocked"] and result["reason"]=="declared_baseline_changed" and claim.exists(),result


def test_real_caller_held_queue_settlement_does_not_reacquire_writer_or_deadlock(pending):
    daemon,task,_,claim=pending
    from ipfs_accelerate_py.agent_supervisor.merge.merge_queue import hold_merge_queue_settlement
    with hold_merge_queue_settlement(daemon.merge_queue.queue_dir,
            target_repository_id=daemon.merge_target_repository_id,
            target_branch=daemon.resolved_merge_target_branch,lock_timeout_seconds=0.01) as settlement:
        assert settlement["settled"]
        request=daemon.review_quiesced_pending_task_claim(task_id=task.task_id)
        assert request["merge_queue"]==settlement
        result=daemon.reconcile_quiesced_pending_task_claim(request=request)
        assert result["reconciled"] and not claim.exists(),result


def test_new_queue_work_after_review_cannot_release_pending_claim(pending):
    daemon,task,_,claim=pending
    request=daemon.review_quiesced_pending_task_claim(task_id=task.task_id)
    daemon.merge_queue.enqueue(branch_name="another-pending-branch",task_id="OTHER-1",
        commit_sha=request["baseline_ref"],target_repository_id=daemon.merge_target_repository_id,
        target_branch=daemon.resolved_merge_target_branch)
    result=daemon.reconcile_quiesced_pending_task_claim(request=request)
    assert result["blocked"] and claim.exists(),result


def test_identical_claim_bytes_replaced_after_review_fail_inode_cas(pending):
    daemon,task,_,claim=pending
    request=daemon.review_quiesced_pending_task_claim(task_id=task.task_id)
    raw=claim.read_bytes();replacement=claim.with_name("replacement.tmp")
    replacement.write_bytes(raw);replacement.chmod(0o600);replacement.replace(claim)
    result=daemon.reconcile_quiesced_pending_task_claim(request=request)
    assert result["blocked"] and result["reason"]=="pending_claim_pinned_claim_identity_changed"
    assert claim.read_bytes()==raw


def test_standalone_helper_pins_loaded_native_not_decoy_sibling_source(tmp_path):
    copied=tmp_path/"pending_task_claim_recovery.py";copied.write_bytes(Path(recovery.__file__).read_bytes())
    (tmp_path/"implementation_daemon.py").write_text("raise RuntimeError('untrusted sibling must not import')\n")
    spec=importlib.util.spec_from_file_location(recovery.__package__+".standalone_pending_fixture",copied)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    sources=module.runtime_sources()
    assert sources[".implementation_daemon"]["path"]==str(Path(native.__file__).resolve())
    assert sources[".implementation_daemon"]["sha256"]==hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()
    assert sources["pending_task_claim_recovery"]["path"]==str(copied)


@pytest.mark.parametrize("pending", ["current_maintenance"], indirect=True)
def test_same_public_recovery_call_can_release_its_own_terminal_maintenance_record(pending):
    daemon,task,workspace,claim=pending
    record=daemon.worktree_lifecycle.load_workspace(workspace)
    assert record.owner==native.current_process_birth()
    before=asdict(native.PortalTaskState.load(daemon.state_path))
    request=daemon.review_quiesced_pending_task_claim(task_id=task.task_id)
    result=daemon.reconcile_quiesced_pending_task_claim(request=request)
    assert result["reconciled"] and not claim.exists(),result
    proof=json.loads(Path(result["receipt_path"]).read_text())["maintenance_owner_proof"]
    assert proof=={"source":"exact_current_recovery_process","owner":record.owner.to_dict()}
    assert asdict(native.PortalTaskState.load(daemon.state_path))==before


@pytest.mark.parametrize("damage", ["foreign_live", "reused_birth", "altered_current_birth", "unknown"])
def test_terminal_maintenance_exception_never_accepts_foreign_reused_or_unknown_birth(pending,monkeypatch,damage):
    daemon,_,_,_=pending
    from ipfs_accelerate_py.agent_supervisor.merge import worktree_lifecycle
    if damage in {"foreign_live","reused_birth"}:
        child=subprocess.Popen([sys.executable,"-c","import time; time.sleep(30)"])
        try:
            owner=worktree_lifecycle.read_process_birth(child.pid)
            assert owner is not None
            if damage=="reused_birth":owner=replace(owner,start_time_ticks=owner.start_time_ticks+1)
            with pytest.raises(ValueError,match="terminal_maintenance_owner_foreign_or_reused"):
                recovery._terminal_maintenance_owner(daemon,native,owner)
        finally:
            child.terminate();child.wait(timeout=5)
    else:
        owner=native.current_process_birth()
        if damage=="unknown":
            def unavailable(*args,**kwargs):raise OSError("synthetic process inspection unavailable")
            monkeypatch.setattr(worktree_lifecycle,"read_process_birth",unavailable)
            with pytest.raises(OSError):recovery._terminal_maintenance_owner(daemon,native,owner)
        else:
            owner=replace(owner,parent_pid=owner.parent_pid+1)
            with pytest.raises(ValueError,match="terminal_maintenance_owner_foreign_or_reused"):
                recovery._terminal_maintenance_owner(daemon,native,owner)
