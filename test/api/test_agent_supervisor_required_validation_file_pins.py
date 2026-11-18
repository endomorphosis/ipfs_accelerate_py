"""Input availability fences run before local/provider attempts or workspaces."""
import hashlib
import json
import subprocess

import pytest
import ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon as module


def task(rows=None):
    metadata = {} if rows is None else {"required validation file pins": json.dumps(rows)}
    return module.PortalTask(task_id="PINS-001", title="Pinned input", status="ready", completion="manual",
        priority="P0", track="runtime", outputs=["receipt.json"], validation=["python3 -m pytest tests/test_receipt.py"],
        metadata=metadata, canonical_task_cid="task:pinned-input")


def pin(path, content=b"retained data"):
    path.write_bytes(content)
    return {"path": str(path), "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


@pytest.fixture
def daemon(tmp_path, monkeypatch):
    monkeypatch.setenv(module.REQUIRE_PROVIDER_TASK_CONTRACT_ENV, "1")
    outer = object.__new__(module.PortalImplementationDaemon)
    outer.repo_root = tmp_path
    outer._canonical_ref = lambda item: item.canonical_task_cid
    outer._proposal_scope_paths = lambda _item: ["receipt.json"]
    return outer


@pytest.mark.parametrize("writing", [True, False])
def test_exact_file_pins_allow_provider_and_local_contracts(daemon, tmp_path, writing):
    daemon._require_provider_task_contract(task([pin(tmp_path / "input")]), provider_writing=writing)


def test_absent_metadata_and_disabled_policy_do_not_read_files(daemon, monkeypatch):
    monkeypatch.setattr(module.os, "open", lambda *_a, **_k: pytest.fail("unrequested read"))
    daemon._require_provider_task_contract(task(), provider_writing=False)
    monkeypatch.setenv(module.REQUIRE_PROVIDER_TASK_CONTRACT_ENV, "0")
    daemon._require_provider_task_contract(task([{"bad": "pin"}]), provider_writing=True)


@pytest.mark.parametrize("change,reason", [("missing", "missing"), ("size", "size_changed"), ("digest", "sha256_changed")])
def test_missing_or_changed_file_defers_and_exact_return_recovers(daemon, tmp_path, change, reason):
    path = tmp_path / "input"; row = pin(path)
    if change == "missing": path.unlink()
    elif change == "size": path.write_bytes(b"different size")
    else: path.write_bytes(b"changed bytes")
    with pytest.raises(module.ProviderTaskContractDeferred) as caught:
        daemon._require_provider_task_contract(task([row]), provider_writing=False)
    receipt = caught.value.receipt
    assert caught.value.reason == "provider_task_contract_required_validation_file_pins_unavailable"
    assert receipt["unavailable_validation_file_pins"] == [{"path": str(path), "reason": reason}]
    assert receipt["attempt_consumed"] is receipt["provider_call_allowed"] is False
    assert caught.value.backoff_seconds == 300
    path.write_bytes(b"retained data")
    daemon._require_provider_task_contract(task([row]), provider_writing=False)


@pytest.mark.parametrize("rows", [[], {}, "not-json", [{"path": "relative", "bytes": 1, "sha256": "a"*64}],
    [{"path": "/tmp/../escape", "bytes": 1, "sha256": "a"*64}],
    [{"path": "/tmp/\ud800", "bytes": 1, "sha256": "a"*64}],
    [{"path": "/tmp/pin", "bytes": True, "sha256": "a"*64}],
    [{"path": "/tmp/pin", "bytes": 2**30+1, "sha256": "a"*64}],
    [{"path": "/tmp/pin", "bytes": 1, "sha256": "A"*64}],
    [{"path": "/tmp/pin", "bytes": 1, "sha256": "a"*64, "extra": 1}],
    [{"path": f"/tmp/pin-{n}", "bytes": 1, "sha256": "a"*64} for n in range(65)],
    [{"path": f"/tmp/pin-{n}", "bytes": 2**30, "sha256": "a"*64} for n in range(3)]])
def test_invalid_or_oversized_declaration_never_opens_files(daemon, monkeypatch, rows):
    monkeypatch.setattr(module.os, "open", lambda *_a, **_k: pytest.fail("invalid pin opened"))
    with pytest.raises(module.ProviderTaskContractDeferred) as caught:
        daemon._require_provider_task_contract(task(rows), provider_writing=False)
    assert caught.value.reason == "provider_task_contract_invalid_required_validation_file_pins"


def test_symlink_pin_is_not_followed(daemon, tmp_path, monkeypatch):
    target = tmp_path / "target"; row = pin(target)
    link = tmp_path / "link"; link.symlink_to(target); row["path"] = str(link)
    monkeypatch.setattr(module.os, "open", lambda *_a, **_k: pytest.fail("symlink followed"))
    with pytest.raises(module.ProviderTaskContractDeferred) as caught:
        daemon._require_provider_task_contract(task([row]), provider_writing=False)
    assert caught.value.receipt["unavailable_validation_file_pins"][0]["reason"] == "noncanonical_or_symlink"


def test_replaced_file_fails_even_with_identical_bytes(daemon, tmp_path, monkeypatch):
    path = tmp_path / "input"; content = b"x"*(2*1024*1024); row = pin(path, content)
    original = module.os.read; replaced = False
    def read(descriptor, limit):
        nonlocal replaced
        value = original(descriptor, limit)
        if value and not replaced:
            replaced = True; path.unlink(); path.write_bytes(content)
        return value
    monkeypatch.setattr(module.os, "read", read)
    with pytest.raises(module.ProviderTaskContractDeferred) as caught:
        daemon._require_provider_task_contract(task([row]), provider_writing=False)
    assert caught.value.receipt["unavailable_validation_file_pins"][0]["reason"] == "file_changed_during_read"


def test_fingerprint_normalizes_order_and_binds_expected_pinset(daemon, tmp_path):
    a,b = pin(tmp_path / "a"),pin(tmp_path / "b"); (tmp_path / "a").unlink(); (tmp_path / "b").unlink()
    receipts=[]
    for rows in ([a,b],[b,a],[{**a,"sha256":"f"*64},b]):
        with pytest.raises(module.ProviderTaskContractDeferred) as caught:
            daemon._require_provider_task_contract(task(rows), provider_writing=False)
        receipts.append(caught.value.receipt)
    assert receipts[0] == receipts[1]
    assert receipts[0]["contract_fingerprint"] != receipts[2]["contract_fingerprint"]


def test_missing_local_pin_spends_no_attempt_or_workspace_or_merge_and_next_task_advances(tmp_path, monkeypatch):
    monkeypatch.setenv(module.REQUIRE_PROVIDER_TASK_CONTRACT_ENV,"1")
    repo=tmp_path/"repo";repo.mkdir();board=repo/"tasks.md"
    row={"path":str(tmp_path/"missing-input"),"bytes":1,"sha256":"a"*64}
    board.write_text("# Tasks\n\n## PINS-001 Local pinned task\n\n- status: todo\n- completion: manual\n- priority: P0\n- track: runtime\n"
        "- provider role: deterministic-only\n- context budget tokens: 0\n- outputs: receipt.json\n- validation: python3 -m pytest tests/test_receipt.py\n"
        "- required validation file pins: "+json.dumps([row])+"\n\n## PINS-002 Independent task\n\n- status: todo\n- completion: manual\n"
        "- priority: P1\n- track: runtime\n- outputs: result.json\n- validation: python3 -m pytest tests/test_receipt.py\n")
    for command in (["git","init","-q","-b","main"],["git","add","tasks.md"],
        ["git","-c","user.name=Pin Test","-c","user.email=pins@example.invalid","commit","-qm","task pins"]):
        subprocess.run(command,cwd=repo,check=True)
    state=tmp_path/"state";queue=tmp_path/"merge-queue"
    outer=module.PortalImplementationDaemon(todo_path=board,state_path=state/"state.json",strategy_path=state/"strategy.json",
        events_path=state/"events.jsonl",repo_root=repo,task_header_prefix="## PINS-",implement=True,
        use_ephemeral_worktree=True,worktree_root=tmp_path/"worktrees",merge_queue_dir=queue)
    reached=[]
    def boundary(item):
        reached.append(item.task_id);raise module.ImplementationRetryDeferred("test boundary",backoff_seconds=300)
    monkeypatch.setattr(outer,"_require_primary_provider_readiness",boundary)
    monkeypatch.setattr(outer,"_compile_implementation_context",lambda *_a:pytest.fail("local dispatch reached"))
    monkeypatch.setattr(outer,"_enqueue_merge_candidate",lambda *_a,**_k:pytest.fail("merge queued"))
    first=outer.run_once()["implementation_result"]
    assert first["task_id"]=="PINS-001" and first["reason"]=="provider_task_contract_required_validation_file_pins_unavailable"
    assert first["provider_dispatched"] is first["attempt_consumed"] is False
    assert first["deferred"] is first["retryable"] is True and reached==[]
    second=outer.run_once()["implementation_result"]
    assert second["task_id"]=="PINS-002" and reached==["PINS-002"]
    saved=module.PortalTaskState.load(outer.state_path)
    assert not saved.implementation_attempts and not saved.implementation_attempts_by_cid
    assert not list((tmp_path/"worktrees").glob("workspace_*")) and not list(queue.rglob("*.json"))
    events=[json.loads(line) for line in outer.events_path.read_text().splitlines()]
    assert not any(e.get("type") in {"implementation_started","provider_invocation_committed","provider_callback_started"} for e in events)

    # Later polling must retain canonical queue fences while the board still
    # labels both tasks ready.
    outer.run_once()
    fenced=module.PortalTaskState.load(outer.state_path)
    assert fenced.selectable_ready_count==2
    assert fenced.selection_idle_reason=="implementation_retry_deferred:all_eligible_ready_tasks_in_cooldown"
    from types import SimpleNamespace
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_supervisor as parent
    supervisor=object.__new__(parent.PortalImplementationSupervisor)
    supervisor.config=SimpleNamespace(implement=True,reconciliation_only=False,
        implementation_log_stall_seconds=30,daemon_interval=1,check_interval=1,repo_root=repo)
    supervisor._watchdog_startup_grace_seconds=lambda:0
    supervisor._active_agent_worker_processes=lambda:[]
    supervisor._active_validation_subprocess_exists=lambda:False
    supervisor._current_supervisor_checkout_lease=lambda:None
    supervisor._repo_merge_lock_path=lambda:tmp_path/"merge.lock"
    child=SimpleNamespace(pid=0)
    pressure=supervisor._idle_ready_child_watchdog_status(fenced,child,now_monotonic=1000)
    assert pressure["implementation_progress_status"]=="backpressure"
    assert pressure["idle_ready_seconds"]==0 and not pressure["idle_ready_recycle_safe"]
    # Partial coverage must never suppress a real dispatch stall.
    independent=next(item for item in outer._load_tasks() if item.task_id=="PINS-002")
    outer.task_queue.entries[outer._canonical_ref(independent)].cooldown_until=0
    outer.task_queue.save()
    outer.implement=False
    outer.run_once()
    runnable=module.PortalTaskState.load(outer.state_path)
    assert runnable.selection_idle_reason==""
    first=supervisor._idle_ready_child_watchdog_status(runnable,child,now_monotonic=1001)
    later=supervisor._idle_ready_child_watchdog_status(runnable,child,now_monotonic=2000)
    assert first["implementation_progress_status"]=="awaiting_dispatch"
    assert later["implementation_progress_status"]=="ready_work_stalled"
    assert not runnable.implementation_attempts and not runnable.implementation_attempts_by_cid
