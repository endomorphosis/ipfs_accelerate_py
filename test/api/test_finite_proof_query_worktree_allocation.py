"""Genuine host native claim, Portal projection and worktree allocation.

The constructor and native ownership are real; installed launcher custody and
sudo cleanup are explicit host fixtures. No supervisor or UID1001 worker is
launched, and these tests do not qualify the Docker dispatch boundary.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
import time

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import finite_proof_query_worktree_allocation as allocation
from ipfs_accelerate_py.agent_supervisor.runtime import finite_proof_query_worker_dispatch as dispatch
from ipfs_accelerate_py.agent_supervisor.task_sources import typed_state_owner as typed
from test.api.test_finite_integer_codebase import finite_tools, finite_git  # noqa: F401
from test.api.test_finite_proof_query_worker_context import _native_proof_query_worker_case
from test.api.test_finite_proof_query_execution import reserve
from test.api.test_finite_repository_execution import _daemon


def _retain_initial_live_custody(case, destination):
    """Retain genuine positive native reads before mutations and teardown.

    This is host-fixture evidence. It contains no client token or signing key
    and does not turn historical custody into a current worker capability.
    """
    from ipfs_accelerate_py.agent_supervisor.planning import finite_integer_source_custody as files
    from ipfs_accelerate_py.agent_supervisor.runtime import finite_proof_query_execution as proof
    from ipfs_accelerate_py.agent_supervisor.runtime import finite_proof_query_worker_source_custody as source_custody
    from ipfs_accelerate_py.agent_supervisor.runtime import finite_repository_execution as native_execution
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import canonical_json_bytes

    destination = Path(destination)
    destination.mkdir(mode=0o700)
    raw_directory = destination / "raw"
    raw_directory.mkdir(mode=0o700)
    manifest, total_bytes = [], 0
    checkpoint = lambda: 90.0
    cap, seed = case["worker_source_custody"], case["source_baseline"]

    def retain(raw, *, role, original_path, original_identity, source):
        nonlocal total_bytes
        total_bytes += len(raw)
        assert len(manifest) < 64 and total_bytes <= 4 * 1024 * 1024
        path = raw_directory / (str(len(manifest)).zfill(3) + ".bin")
        path.write_bytes(raw); path.chmod(0o400)
        assert path.read_bytes() == raw
        manifest.append({"role": role, "source_path": str(original_path),
            "capture_source": source, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
            "original_read_identity": list(original_identity), "retained_path": str(path)})

    def current(path, role):
        row, raw = files._read(Path(path), role=role, bound=files.LIMITS["git_file_bytes"], checkpoint=checkpoint)
        info = row.path.lstat()
        retain(raw, role=role, original_path=row.path, original_identity=row.witness,
               source="genuine-current-physical-read")
        manifest[-1].update(owner_uid=info.st_uid, owner_gid=info.st_gid, link_count=info.st_nlink)
        return raw

    with case["native"].server._lock, case["closure"]._catalog.store._lock:
        claim = dispatch._require_native_dispatch_claim(server=case["native"].server,
            expected_population=case["broker"]._population, request=case["dispatch_request"],
            peer_identity=case["dispatch_peer"], client_id=case["runtime"].client_id,
            store_id=case["runtime"].owner_identity.store_id)
        allocation_body = allocation.FrozenFiniteProofQueryWorktreeAllocation.require_current(
            case["allocation"], checkpoint=checkpoint)
        source_body = source_custody.FrozenWorkerSourceCustody.require_current(cap, checkpoint)
        proof.FrozenFiniteProofQueryExecutionClosure.require_worker_dispatch_current(
            case["closure"], scope=case["scope"], worker_source_custody=cap)
        bodies = {"allocation": allocation_body, "source-baseline": seed.material_binding,
                  "source-custody": source_body}
        for name, value in bodies.items():
            path = destination / (name + ".json")
            path.write_bytes(canonical_json_bytes(value)); path.chmod(0o400)
        current(cap._marker.path, "linked-marker")
        for name, row in cap._files:
            current(row.path, "linked:" + name)
        baseline_files = dict(seed._git_files)
        for name, raw in seed._git_raw:
            row = baseline_files[name]
            assert (len(raw), hashlib.sha256(raw).hexdigest()) == (row.size, row.sha)
            retain(raw, role="original-canonical:" + name, original_path=row.path,
                original_identity=row.witness, source="original-sealed-baseline-raw")
        store, workspace = case["allocation_store"], case["workspace"]
        record = store._load_strict_workspace_record(workspace)
        assert json.loads(current(store.workspace_path_for(workspace), "native-lifecycle")) == record.to_dict()
        current(store.task_index_path_for(canonical_task_cid=record.canonical_task_cid,
            task_id=record.task_id, attempt=record.attempt), "native-task-index")
        current(case["portal_paths"].binding, "native-portal-binding")
        current(case["portal_paths"].task_projection, "native-portal-projection")
        population = native_execution._physical_native(case["native"].server._connection)
        # Close again after every retention read; the native locks remain held.
        proof.FrozenFiniteProofQueryExecutionClosure.require_worker_dispatch_current(
            case["closure"], scope=case["scope"], worker_source_custody=cap)
        assert allocation.FrozenFiniteProofQueryWorktreeAllocation.require_current(
            case["allocation"], checkpoint=checkpoint) == allocation_body
        assert source_custody.FrozenWorkerSourceCustody.require_current(cap, checkpoint) == source_body
        receipt = {"schema": "host-native-initial-worktree-custody-retention@1",
            "capture_phase": "initial-successful-native-guard-before-test-mutations-and-linked-teardown",
            "captured_utc_ns": time.time_ns(), "native_claim": claim,
            "native_task_population": population, "native_lifecycle": record.to_dict(),
            "native_portal_identity": dict(case["portal_identity"]),
            "native_peer_identity": list(case["dispatch_peer"]),
            "worktree_path": str(workspace), "scope": str(case["scope"]._output / "execution-scope.json")
                if hasattr(case["scope"], "_output") else str(case["runtime"].directory),
            "material_bodies": {name: {"path": str(destination / (name + ".json")),
                "sha256": hashlib.sha256(canonical_json_bytes(value)).hexdigest()} for name, value in bodies.items()},
            "raw_files": manifest, "retained_raw_bytes": total_bytes,
            "all_native_guards_closed_before_and_after_reads": True,
            "deployment_launcher_and_sudo_cleanup": "Explicit host fixtures.",
            "historical_only": True, "current_dispatch_authority": False,
            "isolated_worker_birth_qualified": False, "proof_authority": False,
            "completion_authority": False, "global_convergence_proved": False,
            "model_off": True, "training_steps": 0}
        # This historical native receipt includes the real lifecycle's finite
        # float timestamps. Formal material bodies above retain DAG-JSON;
        # their serializer must not be reused for this native observation.
        receipt_raw = json.dumps(receipt, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False).encode("utf-8")
        assert len(receipt_raw) <= 1024 * 1024
        path = destination / "receipt.json"
        path.write_bytes(receipt_raw); path.chmod(0o400)
    return path


@contextmanager
def _native_proof_query_allocation_case(root, tools, monkeypatch, *, gc_before_allocation=False):
    """Share one genuine sealed owner fixture with the source-custody suite."""
    from ipfs_accelerate_py.agent_supervisor.entrypoints import admitted_benchmark_runtime as entry
    from ipfs_accelerate_py.agent_supervisor.runtime import candidate_execution as boundary
    from ipfs_accelerate_py.agent_supervisor.task_sources.quack_state_client import QuackStateClient
    from ipfs_accelerate_py.agent_supervisor.task_sources.typed_database_task_source import (
        TypedDatabaseTaskSource, daemon_required_owner_operations,
        daemon_required_owner_command_operations,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import TYPED_STATE_OWNER_TOKEN_ENV
    from ipfs_accelerate_py.agent_supervisor.task_sources.board_control_plane import infer_board_namespace
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import DatabasePortalExecutionBridge
    from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import WorktreeLifecycleStore

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with _native_proof_query_worker_case(root / "native", tools) as case:
        binding = {"schema": boundary.SCHEMA,
            "argv": ["/opt/ipfs-supervisor/bin/validation-worker"], "files": {},
            "owner_uid": os.geteuid(), "worker_uid": 1001,
            "namespaces": {name: os.readlink("/proc/self/ns/" + name) for name in ("pid", "mnt", "net")},
            "completion_authority": False}
        with monkeypatch.context() as patch:
            patch.setattr(boundary, "bind_candidate_runner", lambda argv: deepcopy(binding))
            patch.setattr(boundary, "verify_candidate_runner", lambda value: value)
            patch.setattr(boundary, "_root_file",
                lambda path: hashlib.sha256(str(path).encode("utf-8")).hexdigest())
            native_run = subprocess.run
            def host_cleanup(command, *args, **kwargs):
                if list(command[:5]) == ["/usr/bin/sudo", "-n", "-u", "benchmarkworker", "--"]:
                    assert command[-1] == "--cleanup"
                    return subprocess.CompletedProcess(command, 0, b"", b"")
                return native_run(command, *args, **kwargs)
            patch.setattr(subprocess, "run", host_cleanup)
            with tempfile.TemporaryDirectory(prefix="fpq-alloc-") as short_root:
                with reserve(case, root / "reserved") as (scope, closure):
                    worktrees = root / "worktrees"
                    worktrees.mkdir(mode=0o700)
                    runtime = entry.AdmittedBenchmarkRuntime.create(Path(short_root) / "launch",
                        admission=case["admission"]["local_admission"], server=case["native"].server,
                        source=case["native"].source, implement=True,
                        implementation_command=scope.to_dict()["payload"]["candidate"]["implementation_command"],
                        candidate_runner_argv=binding["argv"], finite_execution_scope=scope,
                        worker_worktree_root=worktrees)
                    try:
                        broker = runtime.proof_query_dispatch_broker
                        server, original = case["native"].server, case["native"]
                        token, grant = server.issue_typed_client_grant_record(
                            client_id=runtime.client_id, process_birth_id=original.identity.process_birth_id,
                            peer_pid=os.getpid(), allowed_operations=daemon_required_owner_operations(),
                            allowed_command_operations=daemon_required_owner_command_operations())
                        credentials = replace(original.credentials, client_id=runtime.client_id, token=token)
                        with patch.context() as client_environment:
                            client_environment.setenv(TYPED_STATE_OWNER_TOKEN_ENV, token)
                            client = QuackStateClient(owner_id=runtime.client_id,
                                store_id=original.identity.store_id,
                                process_birth_id=original.identity.process_birth_id)
                            client.attach(original.identity.listen_uri, server_id=original.identity.server_id)
                            source = TypedDatabaseTaskSource(client,
                                execution_route_policy=original.credentials.execution_route_policy)
                            local_case = dict(case, native=replace(original,
                                client=client, source=source, credentials=credentials))
                            driver = _daemon(local_case, "allocation-claim")
                            workspace, branch, store, record = None, None, None, None
                            try:
                                attempt = driver.claim_next()
                                assert attempt is not None and attempt.task_cid == case["offset_cid"]
                                task = source.get_task(attempt.task_cid)
                                target = finite_git(case["repository"], "branch", "--show-current").strip()
                                def no_provider(*_args):
                                    pytest.fail("allocation test unexpectedly dispatched a provider")
                                bridge = DatabasePortalExecutionBridge(task_source=source,
                                    attempt_root=runtime.state / "run" / "admitted_database_portal_attempts",
                                    portal_factory=no_provider, repo_root=case["repository"],
                                    board_namespace=infer_board_namespace(merge_target_branch=target,
                                        todo_path=Path(server.config.database_path), state_prefix="admitted"),
                                    merge_target_branch=target, task_header_prefix="## " + task.task_alias)
                                paths, portal_binding = bridge._ensure_attempt_projection(attempt, task)
                                identity = bridge._prior_projection_identity(paths, portal_binding)
                                workspace, branch = worktrees / "exact-native-allocation", "implementation/finite-proof-query-allocation"
                                if gc_before_allocation:
                                    finite_git(case["repository"], "gc", "--prune=now")
                                store = WorktreeLifecycleStore(case["repository"])
                                record = store.begin_preparing(task_id=task.task_alias,
                                    canonical_task_cid=identity["canonical_task_cid"],
                                    attempt=(1 << 52) | (attempt.attempt_number * (1 << 16)) | 1,
                                    lane_id=str(paths.root) + ":0/1:" + str(os.getpid()),
                                    workspace_path=workspace, branch=branch, merge_target=target,
                                    state_dir=str(paths.root))
                                finite_git(case["repository"], "worktree", "add", "-b", branch,
                                           str(workspace), case["manifest"]["payload"]["baseline_commit"])
                                record = store.mark_active(workspace, lease_id=record.lease_id,
                                                           expected_fence=record.fence)
                                request = {"schema": dispatch.SCHEMA, "operation": "prepare",
                                    "context_cid": case["worker_context"]["context_cid"],
                                    "command": scope.to_dict()["payload"]["candidate"]["argv"],
                                    "worktree": str(workspace), "task_id": task.task_alias,
                                    "database_task_cid": attempt.task_cid,
                                    "database_attempt_id": attempt.attempt_id,
                                    "database_claim_id": attempt.claim_id,
                                    "database_attempt_number": attempt.attempt_number}
                                left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
                                try:
                                    peer = typed._kernel_peer_identity(left)
                                finally:
                                    left.close(); right.close()
                                with server._lock, closure._catalog.store._lock:
                                    claim = dispatch._require_native_dispatch_claim(server=server,
                                        expected_population=broker._population, request=request,
                                        peer_identity=peer, client_id=runtime.client_id,
                                        store_id=runtime.owner_identity.store_id)
                                    broker._worktree_current(request, claim=claim, peer_identity=peer)
                                case.update(scope=scope, closure=closure, runtime=runtime, broker=broker,
                                    allocation=broker._allocation, worker_source_custody=broker._worker_source_custody,
                                    source_baseline=broker._worker_source_baseline,
                                    workspace=workspace, branch=branch, allocation_store=store,
                                    allocation_record=record, portal_paths=paths, portal_binding=portal_binding,
                                    portal_identity=identity, dispatch_request=request, dispatch_claim=claim,
                                    dispatch_peer=peer, dispatch_grant=grant, dispatch_driver=driver)
                                case["initial_live_custody_receipt"] = _retain_initial_live_custody(
                                    case, root / "initial-live-native-custody")
                                yield case
                            finally:
                                if record is not None and store is not None:
                                    current = store.load_workspace(workspace)
                                    if current is not None and not current.is_terminal:
                                        store.mark_terminal(workspace, lease_id=current.lease_id,
                                            expected_fence=current.fence, reason="host-allocation-test-complete")
                                if workspace is not None and workspace.exists():
                                    finite_git(case["repository"], "worktree", "remove", "--force", str(workspace))
                                if branch is not None:
                                    finite_git(case["repository"], "branch", "-D", branch)
                                driver.close(); source.close(); client.close()
                                server._command_gateway.revoke_grant(grant.grant_id)
                    finally:
                        runtime.stop()
                        scope.finish_runtime(runtime)
                        runtime.close()
                        assert not broker.socket_path.exists() and not broker._thread.is_alive()
                        shutil.copytree(runtime.directory, root / "retained-native-constructor-state")


@pytest.fixture(scope="module")
def native_proof_query_allocation_case(tmp_path_factory, finite_tools):
    patch = pytest.MonkeyPatch()
    try:
        with _native_proof_query_allocation_case(
                tmp_path_factory.mktemp("finite-proof-query-native-allocation"), finite_tools, patch) as case:
            yield case
    finally:
        patch.undo()


def _check(case):
    with case["native"].server._lock, case["closure"]._catalog.store._lock:
        return allocation.FrozenFiniteProofQueryWorktreeAllocation.require_current(
            case["allocation"], checkpoint=lambda: 90.0)


def test_native_allocation_matches_distinct_portal_identity_actual_claim_and_kernel_birth(native_proof_query_allocation_case):
    case = native_proof_query_allocation_case
    material = _check(case)
    assert type(case["allocation"]) is allocation.FrozenFiniteProofQueryWorktreeAllocation
    assert material["database_task_cid"] == case["offset_cid"]
    assert material["portal_canonical_task_cid"] == case["portal_identity"]["canonical_task_cid"]
    assert material["database_task_cid"] != material["portal_canonical_task_cid"]
    assert material["lifecycle_owner_birth"]["pid"] == os.getpid()
    assert material["worktree_path"] == str(case["workspace"])
    assert material["branch"] == case["branch"]
    assert material["native_worktree_lease_checked"] is True
    assert material["independent_attempt_lease_checked"] is False
    assert material["proof_authority"] is material["process_origin_attested"] is False


@pytest.mark.parametrize("field", ["task_id", "canonical_task_cid", "attempt", "branch", "state", "owner", "lease_id"])
def test_native_lifecycle_tuple_corruption_refuses(native_proof_query_allocation_case, field):
    case = native_proof_query_allocation_case
    path = case["allocation_store"].workspace_path_for(case["workspace"])
    before = path.read_bytes()
    value = json.loads(before)
    if field == "owner":
        value["owner"]["start_time_ticks"] += 1
    elif field == "attempt":
        value[field] += 65536
    elif field == "state":
        value[field] = "terminal"
    else:
        value[field] = "changed-unbound"
    path.write_text(json.dumps(value))
    try:
        with pytest.raises((ValueError, RuntimeError)):
            _check(case)
    finally:
        path.write_bytes(before)
    assert _check(case)["allocation_cid"] == case["allocation"].material_binding["allocation_cid"]


def test_exact_task_index_required_even_with_valid_native_workspace_record(native_proof_query_allocation_case):
    case = native_proof_query_allocation_case
    record = case["allocation_store"].load_workspace(case["workspace"])
    path = case["allocation_store"].task_index_path_for(canonical_task_cid=record.canonical_task_cid,
        task_id=record.task_id, attempt=record.attempt)
    before = path.read_bytes()
    value = json.loads(before); value["workspace_path"] = str(case["workspace"].parent / "foreign")
    path.write_text(json.dumps(value))
    try:
        with pytest.raises((ValueError, RuntimeError)):
            _check(case)
    finally:
        path.write_bytes(before)


@pytest.mark.parametrize("path_key", ["binding", "task_projection"])
def test_authentic_portal_binding_and_projected_task_remain_required(native_proof_query_allocation_case, path_key):
    case = native_proof_query_allocation_case
    path = getattr(case["portal_paths"], path_key)
    before, mode = path.read_bytes(), path.stat().st_mode & 0o777
    path.chmod(0o600)
    path.write_bytes(before + b"\nchanged-unbound\n")
    path.chmod(mode)
    try:
        with pytest.raises((ValueError, RuntimeError)):
            _check(case)
    finally:
        path.chmod(0o600); path.write_bytes(before); path.chmod(mode)


def test_real_worktree_lease_renewal_preserves_exact_allocation_without_cid_rebase(native_proof_query_allocation_case):
    case = native_proof_query_allocation_case
    before = _check(case)
    store = case["allocation_store"]
    record = store.load_workspace(case["workspace"])
    renewed = store.renew_lease(case["workspace"], lease_id=record.lease_id, expected_fence=record.fence)
    assert renewed.fence == record.fence + 1
    assert _check(case) == before


def test_actual_native_grant_must_retain_full_birth_ack_budget(native_proof_query_allocation_case):
    case = native_proof_query_allocation_case
    server, gateway = case["native"].server, case["native"].server._command_gateway
    with gateway._grants_lock:
        matches = [(key, value) for key, value in gateway._grants.items()
                   if value.grant_id == case["dispatch_grant"].grant_id]
        assert len(matches) == 1
        key, original = matches[0]
        gateway._grants[key] = replace(original, expires_at=time.time_ns() // 1_000_000 + 1000)
    try:
        with server._lock:
            claim = dispatch._require_native_dispatch_claim(server=server,
                expected_population=case["broker"]._population, request=case["dispatch_request"],
                peer_identity=case["dispatch_peer"], client_id=case["runtime"].client_id,
                store_id=case["runtime"].owner_identity.store_id)
            with pytest.raises(dispatch.FiniteProofQueryDispatchError, match="acknowledgement budget"):
                dispatch._require_native_grant_ack_budget(server=server, claim=claim)
    finally:
        with gateway._grants_lock:
            gateway._grants[key] = original
    assert dispatch._require_native_grant_ack_budget(server=server, claim=case["dispatch_claim"]) == original.expires_at


def test_inert_allocation_material_cannot_authorize_native_custody(native_proof_query_allocation_case):
    case = native_proof_query_allocation_case
    material = _check(case)
    with pytest.raises((ValueError, RuntimeError)):
        case["source_baseline"].bind_allocation(allocation=material, checkpoint=lambda: 90.0)


def test_native_worktree_lease_expiry_refuses_current_record_and_peer(native_proof_query_allocation_case):
    case = native_proof_query_allocation_case
    path = case["allocation_store"].workspace_path_for(case["workspace"])
    before = path.read_bytes()
    value = json.loads(before); value["expires_at"] = time.time() - 1.0
    path.write_text(json.dumps(value))
    try:
        with pytest.raises((ValueError, RuntimeError), match="lease"):
            _check(case)
    finally:
        path.write_bytes(before)


def test_cached_allocation_material_cannot_upgrade_or_rebind_ownership(native_proof_query_allocation_case):
    case, cap = native_proof_query_allocation_case, native_proof_query_allocation_case["allocation"]
    before = cap._material
    material = json.loads(before); material["proof_authority"] = True
    cap._material = json.dumps(material).encode()
    try:
        with pytest.raises((ValueError, RuntimeError), match="rebound"):
            _check(case)
    finally:
        cap._material = before
    assert _check(case)["proof_authority"] is False


def test_genuine_receiving_refusal_has_bounded_stage_and_real_message_digest(native_proof_query_allocation_case):
    case, broker = native_proof_query_allocation_case, native_proof_query_allocation_case["broker"]
    channel = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        channel.connect(str(broker.socket_path))
        channel.settimeout(10)
        dispatch._send(channel, case["dispatch_request"])
        response = dispatch._receive(channel)
    finally:
        channel.close()
    assert response["ok"] is False
    observation = broker.observations[-1]
    # The actual factory has no supervisor child/bootstrap in this host
    # fixture. A caller cannot promote its current native grant into that role.
    assert observation["boundary"] == "native_daemon_to_owner_wrapper"
    assert observation["stage"] == "native_claim"
    assert observation["error_type"] == "FiniteProofQueryDispatchError"
    assert observation["message_sha256"] == hashlib.sha256(
        b"dispatch peer lacks its exact native child bootstrap receipt").hexdigest()
    assert observation["worker_birth_observed"] is False
    assert observation["unacknowledged_child_birth_possible"] is False
    assert "message" not in observation and "token" not in observation


def test_genuine_same_grant_renewal_preserves_allocation_and_full_birth_ack_budget(native_proof_query_allocation_case):
    case = native_proof_query_allocation_case
    before = _check(case)
    gateway = case["native"].server._command_gateway
    with gateway._grants_lock:
        matches = [(key, value) for key, value in gateway._grants.items()
                   if value.grant_id == case["dispatch_grant"].grant_id]
        assert len(matches) == 1
        key, original = matches[0]
    original_floor = case["allocation"]._last_grant_expiry
    try:
        renewed = gateway.renew_grant(case["dispatch_grant"].grant_id, ttl_seconds=7200)
        assert renewed.grant_id == case["dispatch_grant"].grant_id
        assert renewed.expires_at > original.expires_at
        assert _check(case) == before
        assert dispatch._require_native_grant_ack_budget(
            server=case["native"].server, claim=case["dispatch_claim"]) == renewed.expires_at
        current = dict(case["dispatch_claim"], grant_expires_at_ms=renewed.expires_at)
        assert dispatch._same_native_dispatch_claim(current, case["dispatch_claim"])
        assert not dispatch._same_native_dispatch_claim(case["dispatch_claim"], current)
        evidence = {"schema": "genuine-native-grant-renewal-control@1", "grant_id": renewed.grant_id,
            "original_expires_at_ms": original.expires_at,
            "renewed_expires_at_ms": renewed.expires_at,
            "actual_native_gateway_renew_grant_called": True,
            "allocation_cid_preserved": _check(case)["allocation_cid"] == before["allocation_cid"],
            "full_birth_ACK_budget_checked": True, "proof_authority": False}
    finally:
        # Restore the exact deliberate control in the genuine native table and
        # its monotonic validation floor. Final readback rereads that table.
        with gateway._grants_lock:
            gateway._grants[key] = original
        case["allocation"]._last_grant_expiry = original_floor
    assert _check(case) == before
    actual = gateway._require_active_grant(original, peer_identity=case["dispatch_peer"])
    assert actual == original
    evidence.update(actual_original_grant_restored=True,
                    restored_current_gateway_expires_at_ms=actual.expires_at)
    (case["root"] / "native-grant-renewal-control.json").write_text(json.dumps(evidence, indent=2) + "\n")


@pytest.mark.parametrize("field", ["_claim_bytes", "_fields"])
def test_cached_derived_claim_or_truncated_fields_cannot_authorize_allocation(native_proof_query_allocation_case, field):
    case, cap = native_proof_query_allocation_case, native_proof_query_allocation_case["allocation"]
    before = getattr(cap, field)
    setattr(cap, field, b"changed-unbound" if field == "_claim_bytes" else ())
    try:
        with pytest.raises((ValueError, RuntimeError), match="rebound"):
            _check(case)
    finally:
        setattr(cap, field, before)
    assert _check(case)["proof_authority"] is False
