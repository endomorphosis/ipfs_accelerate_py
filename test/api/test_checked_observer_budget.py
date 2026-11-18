"""Real signed inputs and budget owner; explicitly stubbed current proof scope.

These controls do not qualify native runtime birth, current proof acceptance,
workers, providers or task completion. Their exact native runtime instances are
uninitialized data fixtures; the broker's separate native authority is absent.
"""
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from copy import copy, deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
from types import MappingProxyType, SimpleNamespace

import pytest

from test.api.test_terminal_semantic_call_review import case as coherent_case
from test.api.test_checked_audit_context import selected


@pytest.fixture
def control(selected, monkeypatch, tmp_path):
    from ipfs_accelerate_py.agent_supervisor.runtime import checked_observer_budget as module
    from benchmarks.agent_supervisor.container_coding import terminal_semantic_budget_delivery as delivery
    from ipfs_accelerate_py.agent_supervisor.runtime.code_source_inspection import _PublicCapture
    owner = delivery.budgets.create_semantic_context_budget(binding=selected["prepare"]["checked_binding"],
        policy=delivery.budgets.ContextBudgetPolicy())
    capture = _PublicCapture(selected["root"])
    for name in [*selected["sources"], selected["prepare"]["public_instruction"]["source_path"]]:
        capture.read(name, 65536)
    state = {"entered": False, "closed": False, "checks": 0, "oncheck": None}
    @contextmanager
    def explicitly_unqualified_delivery(**kwargs):
        state["entered"] = True
        def check():
            state["checks"] += 1
            capture.verify_originals()
            if state["oncheck"] is not None: state["oncheck"]()
            capture.verify_originals()
        try:
            check(); yield {"check": check}; check()
        finally:
            state["closed"] = True
    real_delivery = delivery._current_delivery_scope
    monkeypatch.setattr(delivery, "_current_delivery_scope", explicitly_unqualified_delivery)
    args = dict(budget_owner=owner, context_packet=selected["prepare"]["context_packet"],
        admission=selected["prepare"]["admission"], public_instruction=selected["prepare"]["public_instruction"],
        boundary=selected["load"]["boundary"], index=None, repository=selected["root"], expected_head=None,
        native_capsules=None, code_views=None, tool_policy=None, policy_roots=None, policy_observer=None,
        scheduler=None, capacity_control=None, basis={})
    scopes = []
    def create(**changes):
        scope = module.create_checked_observer_budget_scope(**{**args, **changes})
        scopes.append(scope); return scope
    yield SimpleNamespace(module=module, delivery=delivery, selected=selected, owner=owner,
        args=args, create=create, state=state, tmp=tmp_path, real_delivery=real_delivery)
    for scope in scopes:
        if scope._closed: continue
        if scope._runtime is None: scope.close_unlaunched_construction()
        else:
            _stopped(scope._runtime)
            scope.close_after_native_stop(scope._runtime)
    if not owner.snapshot()["closed"]: owner.close()


def _command(scope, selected):
    public = selected["prepare"]["public_instruction"]
    return ["/opt/ipfs-supervisor/bin/router-worker", "--observe-model-input-only", "--model", "fixture",
        "--timeout", "10", "--max-output-tokens", "128", "--purpose", "coding",
        "--semantic-repository", str(selected["root"]), "--semantic-transport", "native",
        "--intent-requirements-transport", "expanded", "--code-retrieval-codec", "expanded",
        "--code-retrieval-wire-delivery", "inline", "--public-instruction-artifact", public["artifact"],
        "--public-instruction-sha256", public["sha256"], "--public-instruction-task-cid", public["task_cid"],
        "--checked-context-owner-scope-cid", scope.context_cid]


def _runtime(scope, control, command=None):
    from ipfs_accelerate_py.agent_supervisor.entrypoints import admitted_benchmark_runtime as native
    runtime = native.AdmittedBenchmarkRuntime.__new__(native.AdmittedBenchmarkRuntime)
    runtime.run_id = "admitted-" + "a" * 32
    runtime.state = control.tmp / "authored-native-runtime-state"; runtime.state.mkdir(exist_ok=True)
    runtime.repository = control.selected["root"]
    runtime.admission = deepcopy(control.args["admission"])
    runtime.context_bundle = {"artifact": "authored-signed-data-not-runtime-birth", "sha256": "a" * 64}
    runtime.signature = {"authored_untrusted_runtime_fixture": True}
    runtime.checked_observer_budget_scope = scope
    runtime._children = []
    for name in ("server", "source", "profile", "service", "coordinator", "local_profile", "owner_identity"):
        setattr(runtime, name, object())
    runtime.process = SimpleNamespace(snapshot=lambda _: SimpleNamespace(members=()))
    runtime._context_refresh_stop_receipt = None
    command = _command(scope, control.selected) if command is None else command
    runtime.manifest = {"schema": "isolated-admitted-supervisor-launch@1", "run_id": runtime.run_id,
        "state_root": str(runtime.state), "repository_root": str(runtime.repository),
        "local_admission": deepcopy(runtime.admission), "checked_observer_budget_scope": scope.material_binding,
        "argv": ["authored", "--implementation-command", shlex.join(command)], "candidate_runner": {"authored": True}}
    runtime.manifest_id = native._digest(runtime.manifest)
    return runtime


def _bound(control):
    scope = control.create(); command = _command(scope, control.selected); runtime = _runtime(scope, control, command)
    scope.bind_runtime(runtime, command)
    return scope, runtime, command


def _dispatch(control, scope, command, **changes):
    return scope.prepare_dispatch(**{"native_prompt": control.selected["prepare"]["native_prompt"],
        "worktree": control.selected["workspace"], "base_command": command, **changes})


def _stopped(runtime, *, code=0):
    from ipfs_accelerate_py.agent_supervisor.control.control_contracts import (
        Operation, OperationAuthority, OperationResult, OperationStatus,
    )
    runtime._context_refresh_stop_receipt = OperationResult(
        request_id="authored-stop-result", operation=Operation.STOP,
        authority=OperationAuthority.MUTATION, status=OperationStatus.SUCCEEDED,
        repository_id="authored-repository", tree_id="authored-tree",
        objective_id="authored-objective", policy_id="authored-policy", caller="authored-test",
        data={"old_tree_fenced": True, "isolated_worker_cleanup": {"returncode": code, "single_worker": True}})
    return runtime._context_refresh_stop_receipt


def _receipt(scope):
    from test.api.test_terminal_checked_worker_budget import _receipt as authored_receipt
    record = scope.prepared_record
    result = authored_receipt(record["preparation"])
    result.update(schema="supervisor-model-input-observation@2", checked_context=record["preflight"],
        semantic_context_verification=record["verification"])
    return result


def test_worker_import_of_scope_module_uses_only_standard_library(monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import checked_observer_budget
    import builtins
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert not name.startswith(("benchmarks", "ipfs_datasets_py"))
        assert "proof" not in name and "duckdb" not in name
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    spec = importlib.util.spec_from_file_location("authored_worker_import_only", checked_observer_budget.__file__)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    assert value.SCHEMA == "checked-observer-budget-scope@1"


def test_factory_retains_originals_and_explicit_transport_producers_before_runtime(control):
    scope = control.create()
    assert control.state["entered"] and control.state["checks"] > 0
    material = scope.material_binding
    assert material["budget_binding"] == control.owner.binding and material["budget_policy"] == control.owner.policy.to_dict()
    assert material["task_cid"] == control.selected["prepare"]["task_cid"]
    assert material["context_sha256"] == hashlib.sha256(control.args["context_packet"].encode()).hexdigest()
    paths = {Path(row["path"]).name for row in material["owner_producer"]["files"]}
    assert {"checked_observer_budget.py", "checked_context_dispatch.py", "admitted_benchmark_runtime.py",
        "supervisor_runtime.py", "implementation_daemon.py", "container_worker_deployment.py"} <= paths
    material["budget_binding"]["view_id"] = "detached mutation"
    assert scope.material_binding["budget_binding"] == control.owner.binding
    assert all(scope.material_binding[key] is False for key in control.module.AUTHORITY)


def _public_storage_root(tmp_path):
    tmp_path.chmod(0o755)
    parent = tmp_path / ".runtime"; parent.mkdir(); parent.chmod(0o755)
    return tmp_path, parent / "checked-audit-context"


def test_new_checked_storage_is_public_under_restrictive_umask_using_only_created_fd(tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import checked_observer_budget as module
    root, child = _public_storage_root(tmp_path)
    before = (root.stat().st_mode, (root / ".runtime").stat().st_mode)
    calls = []; original = os.fchmod
    def capture(fd, mode):
        calls.append((os.fstat(fd).st_mode & 0o777, mode, os.fstat(fd).st_ino))
        original(fd, mode)
    monkeypatch.setattr(module.os, "fchmod", capture)
    previous = os.umask(0o077)  # Test-only reproduction in the isolated pytest child.
    guard = None
    try:
        guard = module._prepare_checked_storage(repository=root, directory=".runtime/checked-audit-context")
        assert os.umask(0o077) == 0o077
        assert calls == [(0o700, 0o755, child.stat().st_ino)]
        assert child.stat().st_mode & 0o777 == 0o755
        assert (root.stat().st_mode, (root / ".runtime").stat().st_mode) == before
        (child / "generated-public.json").write_text("{}")
        guard()  # Generated entries may change size/times, not directory identity/mode.
    finally:
        os.umask(previous)
        if guard is not None: guard.close()


def test_existing_public_checked_storage_is_never_chmodded(tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import checked_observer_budget as module
    root, child = _public_storage_root(tmp_path); child.mkdir(); child.chmod(0o755)
    monkeypatch.setattr(module.os, "fchmod", lambda *_: pytest.fail("existing directory was chmodded"))
    guard = module._prepare_checked_storage(repository=root, directory=".runtime/checked-audit-context")
    try: guard()
    finally: guard.close()


@pytest.mark.parametrize("which", ["root_private", "runtime_private", "runtime_writable", "runtime_symlink",
                                   "child_private", "child_writable", "child_symlink"])
def test_existing_unsafe_checked_storage_refuses_without_repair(tmp_path, monkeypatch, which):
    from ipfs_accelerate_py.agent_supervisor.runtime import checked_observer_budget as module
    root, child = _public_storage_root(tmp_path); parent = root / ".runtime"
    if which == "root_private": root.chmod(0o700)
    elif which == "runtime_private": parent.chmod(0o700)
    elif which == "runtime_writable": parent.chmod(0o775)
    elif which == "runtime_symlink":
        parent.rename(root / "original-runtime"); parent.symlink_to(root / "original-runtime", target_is_directory=True)
    elif which == "child_symlink":
        target = root / "unrelated-private"; target.mkdir(); target.chmod(0o700)
        child.symlink_to(target, target_is_directory=True)
    else:
        child.mkdir(); child.chmod(0o700 if which == "child_private" else 0o775)
    paths = [path for path in (root, parent, child, root / "unrelated-private") if path.exists() or path.is_symlink()]
    before = [(path, path.lstat().st_ino, path.lstat().st_mode) for path in paths]
    monkeypatch.setattr(module.os, "fchmod", lambda *_: pytest.fail("existing unsafe directory was repaired"))
    with pytest.raises(ValueError):
        module._prepare_checked_storage(repository=root, directory=".runtime/checked-audit-context")
    assert [(path, path.lstat().st_ino, path.lstat().st_mode) for path in paths] == before
    if not which.startswith("child_"): assert not child.exists()


def test_checked_storage_retains_original_directory_against_same_mode_replacement(tmp_path):
    from ipfs_accelerate_py.agent_supervisor.runtime import checked_observer_budget as module
    root, child = _public_storage_root(tmp_path)
    guard = module._prepare_checked_storage(repository=root, directory=".runtime/checked-audit-context")
    try:
        child.rename(child.with_name("original-checked-context")); child.mkdir(); child.chmod(0o755)
        with pytest.raises(ValueError, match="directory changed"): guard()
    finally: guard.close()


def test_factory_preprovisions_fresh_public_checked_storage_before_original_instruction_custody(control):
    child = control.selected["root"] / ".runtime/checked-audit-context"
    for path in child.iterdir():
        assert path.is_file() and not path.is_symlink()
        path.unlink()
    child.rmdir()
    previous = os.umask(0o077)  # This test child has no live runtime or broker threads.
    try:
        scope, runtime, command = _bound(control)
        assert child.stat().st_mode & 0o777 == 0o755
        scope.require_current()
        result = _dispatch(control, scope, command)
        assert "--checked-context-artifact" in result["command"]
        assert control.owner.snapshot()["records"][0]["status"] == "reserved"
        assert os.umask(0o077) == 0o077
    finally: os.umask(previous)


def test_real_delivery_gate_refuses_authored_non_owner_proof_fixture_before_scope_creation(control, monkeypatch):
    monkeypatch.setattr(control.delivery, "_current_delivery_scope", control.real_delivery)
    with pytest.raises(ValueError, match="exact current owner"):
        control.create()
    assert control.owner.snapshot()["records"] == []


def test_owner_thread_can_prepare_and_capture_original_scope_without_copy_or_serialization(control):
    scope, runtime, command = _bound(control)
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(_dispatch, control, scope, command).result(timeout=30)
        pool.submit(scope.capture_allocated).result(timeout=30)
        pool.submit(scope.require_current).result(timeout=30)
    assert result["preparation"]["event"] == control.owner.snapshot()["records"][0]["event"]
    assert control.owner.snapshot()["records"][0]["status"] == "reserved"
    scope.abort_pending()


def test_retained_producer_fd_rejects_same_byte_replacement_of_an_explicit_test_producer(control, monkeypatch):
    producer = control.tmp / "authored-producer.py"; producer.write_text("# explicitly unqualified producer witness\n")
    original = control.delivery.forecast.metadata._producer
    def extra(modules): return original([*modules, SimpleNamespace(__file__=str(producer))])
    monkeypatch.setattr(control.delivery.forecast.metadata, "_producer", extra)
    scope = control.create()
    raw = producer.read_bytes(); producer.rename(producer.with_suffix(".old")); producer.write_bytes(raw)
    with pytest.raises(ValueError, match="producer changed"):
        scope.require_current()
    assert control.owner.snapshot()["records"] == []


@pytest.mark.parametrize("operation", [copy, deepcopy])
def test_scope_copy_is_never_another_original_owner(control, operation):
    scope = control.create()
    with pytest.raises(ValueError, match="cannot be copied"): operation(scope)


def test_forked_pid_and_rebound_runtime_are_rejected_before_native_callbacks(control, monkeypatch):
    scope, runtime, command = _bound(control)
    original_pid = os.getpid()
    with monkeypatch.context() as local:
        local.setattr(os, "getpid", lambda: original_pid + 1)
        with pytest.raises(ValueError, match="process-local"): scope.require_current()
    with pytest.raises(ValueError, match="exactly one"): scope.bind_runtime(runtime, command)
    foreign = _runtime(scope, control, command)
    with pytest.raises(ValueError): scope.bind_runtime(foreign, command)
    assert control.owner.snapshot()["records"] == []


@pytest.mark.parametrize("change", ["owner_missing", "owner_duplicate", "owner_foreign", "artifact", "validation", "compact", "provider"])
def test_alternate_or_malformed_recipe_never_reserves(control, change):
    scope = control.create(); command = _command(scope, control.selected)
    flag = "--checked-context-owner-scope-cid"
    if change == "owner_missing": index=command.index(flag); del command[index:index+2]
    elif change == "owner_duplicate": command += [flag, scope.context_cid]
    elif change == "owner_foreign": command[-1] = "foreign"
    elif change == "artifact": command += ["--checked-context-artifact", "/unapproved"]
    elif change == "validation": command[0] = "/opt/ipfs-supervisor/bin/validation-worker"
    elif change == "compact": command[command.index("--code-retrieval-codec")+1] = "compact"
    elif change == "provider": command.remove("--observe-model-input-only")
    runtime = _runtime(scope, control, command)
    with pytest.raises(ValueError): scope.bind_runtime(runtime, command)
    assert control.owner.snapshot()["records"] == []


@pytest.mark.parametrize("change", ["signature", "manifest", "profile", "scope", "admission"])
def test_immutable_native_runtime_joins_reject_rebinding(control, change):
    scope, runtime, command = _bound(control)
    saved = {"signature": runtime.signature, "manifest": runtime.manifest,
        "profile": runtime.profile, "scope": runtime.checked_observer_budget_scope, "admission": runtime.admission}
    try:
        if change == "signature": runtime.signature = {"different": True}
        elif change == "manifest": runtime.manifest = {**runtime.manifest, "different": True}
        elif change == "profile": runtime.profile = object()
        elif change == "scope": runtime.manifest["checked_observer_budget_scope"] = {}
        else: runtime.admission = {}
        with pytest.raises(ValueError): scope.require_current()
        assert control.owner.snapshot()["records"] == []
    finally:
        runtime.signature = saved["signature"]; runtime.manifest = saved["manifest"]
        runtime.manifest["checked_observer_budget_scope"] = scope.material_binding
        runtime.profile = saved["profile"]; runtime.checked_observer_budget_scope = saved["scope"]
        runtime.admission = saved["admission"]


def test_actual_emitted_prompt_is_preflighted_and_original_budget_reserved_before_command(control):
    scope, runtime, command = _bound(control)
    result = _dispatch(control, scope, command)
    assert result["schema"] == "checked-observer-budget-dispatch@1"
    assert "--checked-context-owner-scope-cid" not in result["command"]
    assert result["command"][-6:] == ["--checked-context-artifact", scope.prepared_record["descriptor"]["artifact"],
        "--checked-context-sha256", scope.prepared_record["descriptor"]["sha256"], "--checked-context-task-cid", scope.task_cid]
    snapshot = control.owner.snapshot()
    assert len(snapshot["records"]) == 1 and snapshot["records"][0]["status"] == "reserved"
    assert result["preparation"]["event"] == snapshot["records"][0]["event"]
    assert result["preflight"]["native_prompt_sha256"] == hashlib.sha256(control.selected["prepare"]["native_prompt"].encode()).hexdigest()
    detached = scope.prepared_record; detached["preparation"]["event"]["new_content"]["text"] = "changed"
    assert scope.prepared_record["preparation"]["event"] == snapshot["records"][0]["event"]
    with pytest.raises(ValueError): _dispatch(control, scope, command)
    scope.abort_pending()
    assert control.owner.snapshot()["records"][0]["status"] == "aborted"
    assert control.owner.snapshot()["totals"] == snapshot["totals"]


def test_budget_preflight_refusal_returns_no_derived_command_and_no_partial_charge(control):
    owner = control.delivery.budgets.create_semantic_context_budget(binding=control.owner.binding,
        policy=control.delivery.budgets.ContextBudgetPolicy(max_model_calls=0))
    scope = control.create(budget_owner=owner)
    command = _command(scope, control.selected); runtime = _runtime(scope, control, command); scope.bind_runtime(runtime, command)
    try:
        with pytest.raises(control.delivery.budgets.SemanticContextBudgetExceeded): _dispatch(control, scope, command)
        assert owner.snapshot()["records"] == []
        with pytest.raises(ValueError): _dispatch(control, scope, command)
    finally:
        _stopped(runtime); scope.close_after_native_stop(runtime); owner.close()


def test_changed_actual_prompt_cannot_use_cached_native_task_preflight(control):
    scope, runtime, command = _bound(control)
    with pytest.raises(ValueError): _dispatch(control, scope, command, native_prompt="Foreign emitted task")
    assert control.owner.snapshot()["records"] == []


def test_preflight_guard_closes_before_allocated_permission_transition_then_new_guard_is_retained(control):
    scope, runtime, command = _bound(control); _dispatch(control, scope, command)
    workspace = control.selected["workspace"]
    workspace.chmod(0o3770)
    for row in scope.prepared_record["preflight"]["source_scope"] + [scope.prepared_record["preflight"]["signed_instruction"]]:
        path = workspace / row["path"]
        if path.parent != workspace: path.parent.chmod(0o2770)
        path.chmod(0o660)
    assert workspace.stat().st_mode & 0o7777 == 0o3770
    scope.capture_allocated(); scope.require_current()
    receipt = _receipt(scope)
    result = scope.commit_observation(receipt)
    assert result["budget_snapshot"]["records"][0]["status"] == "committed"
    assert result["local_reservation_committed"] is True
    assert all(result[key] is False for key in control.module.AUTHORITY)
    with pytest.raises(ValueError): scope.commit_observation(receipt)


def test_allocated_same_byte_replacement_after_capture_aborts_without_refund(control):
    scope, runtime, command = _bound(control); _dispatch(control, scope, command); scope.capture_allocated()
    path = control.selected["workspace"] / "caller.py"; raw = path.read_bytes(); backup = path.with_name(path.name+".authored-original")
    path.rename(backup); path.write_bytes(raw); path.chmod(0o644)
    with pytest.raises(ValueError): scope.commit_observation(_receipt(scope))
    scope.abort_pending()
    assert control.owner.snapshot()["records"][0]["status"] == "aborted"


@pytest.mark.parametrize("which", ["original_source", "proof_callback", "boundary", "instruction"])
def test_original_source_proof_or_input_drift_blocks_runtime_dispatch(control, which):
    scope, runtime, command = _bound(control)
    restore = None
    if which == "original_source":
        path=control.selected["root"] / "caller.py"; raw=path.read_bytes(); path.write_bytes(raw)
    elif which == "proof_callback":
        control.state["oncheck"] = lambda: (_ for _ in ()).throw(ValueError("explicit untrusted proof callback drift"))
    elif which == "boundary":
        original=control.args["boundary"]["container_id"]; control.args["boundary"]["container_id"]="d"*64
        restore=lambda:control.args["boundary"].update(container_id=original)
    else:
        original=control.args["public_instruction"]["sha256"];control.args["public_instruction"]["sha256"]="d"*64
        restore=lambda:control.args["public_instruction"].update(sha256=original)
    try:
        with pytest.raises(ValueError): _dispatch(control, scope, command)
        assert control.owner.snapshot()["records"] == []
    finally:
        if restore:restore()
        control.state["oncheck"] = None


def test_signed_literal_same_byte_replacement_before_start_invalidates_factory_custody(control):
    scope, runtime, command = _bound(control)
    path = control.selected["root"] / control.args["public_instruction"]["source_path"]
    raw = path.read_bytes(); backup = path.with_name(path.name + ".authored-original")
    path.rename(backup); path.write_bytes(raw); path.chmod(0o644)
    with pytest.raises(ValueError): scope.require_current()
    assert control.owner.snapshot()["records"] == []


def test_stop_cleanup_remains_available_after_proof_callback_drift_and_aborts_charged_pending(control):
    scope, runtime, command = _bound(control); _dispatch(control, scope, command)
    totals = control.owner.snapshot()["totals"]
    control.state["oncheck"] = lambda: (_ for _ in ()).throw(ValueError("publication must fail; STOP must remain available"))
    with pytest.raises(ValueError): scope.require_current()
    _stopped(runtime)
    result = scope.close_after_native_stop(runtime)
    assert result["retained_guards_closed"] is True and control.state["closed"]
    assert control.owner.snapshot()["totals"] == totals and control.owner.snapshot()["records"][0]["status"] == "aborted"
    with pytest.raises(ValueError, match="closed"): scope.require_current()


def test_native_immutable_stop_result_closes_original_scope_without_refunding_pending(control):
    scope, runtime, command = _bound(control); _dispatch(control, scope, command)
    totals = control.owner.snapshot()["totals"]
    stop = _stopped(runtime)
    assert type(stop.data) is MappingProxyType
    assert type(stop.data["isolated_worker_cleanup"]) is MappingProxyType
    assert runtime._context_refresh_stopped() is True
    result = scope.close_after_native_stop(runtime)
    assert result["retained_guards_closed"] is True and scope.is_closed is True
    assert control.owner.snapshot()["totals"] == totals
    assert control.owner.snapshot()["records"][0]["status"] == "aborted"
    assert control.state["closed"] is True


def test_unlaunched_and_fenced_stop_are_closed_original_resource_predicates(control):
    scope = control.create(); assert scope.is_closed is False
    result = scope.close_unlaunched_construction()
    assert scope.is_closed is True
    assert result["unlaunched"] is True and control.state["closed"]
    scope, runtime, command = _bound(control)
    runtime.checked_context_dispatch_broker = SimpleNamespace(can_close_unlaunched=lambda: False)
    with pytest.raises(ValueError, match="broker"): scope.close_unlaunched_construction(runtime)
    runtime.checked_context_dispatch_broker = SimpleNamespace(can_close_unlaunched=lambda: True)
    assert scope.close_unlaunched_construction(runtime)["unlaunched"] is True
    assert scope.is_closed is True


@pytest.mark.parametrize("code", [None, True, False, 1])
def test_bool_or_failed_worker_cleanup_cannot_release_original_scope(control, code):
    scope, runtime, command = _bound(control); _stopped(runtime, code=code)
    with pytest.raises(ValueError): scope.close_after_native_stop(runtime)
    assert not scope._closed
