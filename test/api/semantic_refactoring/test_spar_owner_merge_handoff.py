"""Fresh native origin and three exact-peer clients on real retained DuckDB."""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts.ops.agent_supervisor import spar_merge_owner as role
from scripts.ops.agent_supervisor import spar_merge_owner_handoff as handoff
from ipfs_accelerate_py.agent_supervisor.task_sources import (
    owner_merge_bootstrap as wire,
)
from ipfs_accelerate_py.agent_supervisor.merge.checkout_lock import (
    checkout_repository_id,
)
from ipfs_accelerate_py.agent_supervisor.merge.merge_queue import MergeQueue
from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
    FakeQuackTransport,
    build_server,
)
from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
    current_process_birth,
)
from ipfs_accelerate_py.agent_supervisor.merge.database_worktree_registry import (
    process_birth_id,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.state_owner_bootstrap import (
    STATE_OWNER_BOOTSTRAP_RESPONSE_SCHEMA,
)
from test.api.test_agent_supervisor_quack_state_server import _compatible_report
from test.api.semantic_refactoring.test_launch_source_amendment_task_source import (
    _fixture as launch_fixture,
)


@pytest.fixture
def fresh_args(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "--quiet", str(repo)], check=True)
    (repo / "runtime").mkdir()
    _, policy, amendment, *_ = launch_fixture()
    scope = {
        "board_namespace": amendment.board_namespace,
        "config_cid": amendment.launch_config_cid,
        "plan_cid": amendment.bootstrap_plan_root_cid,
        "lane_id": "0",
        "attempt_root": str(
            repo / "runtime/state/lane-0/spar_lane_0_database_portal_attempts"
        ),
    }
    args = {
        "queue_root": repo / "runtime/native-queue",
        "repository_id": checkout_repository_id(repo),
        "target_branch": "main",
        "store_id": "spar-native-queue",
        "source_commit": amendment.launch_source_head,
        "source_tree": amendment.launch_repository_tree_id,
        "scopes": [scope],
    }
    return repo, args, policy, amendment


def test_fresh_origin_created_once_and_restarted_without_legacy_import(fresh_args):
    _, args, *_ = fresh_args
    first = handoff.prepare_fresh_native_queue(**args)
    second = handoff.prepare_fresh_native_queue(**args)
    assert second.database_uuid == first.database_uuid
    assert second.manifest_cid == first.manifest_cid
    with role.open_duckdb_connection(second.database_path) as connection:
        assert (
            connection.execute("SELECT COUNT(*) FROM merge_requests").fetchone()[0] == 0
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM " + handoff.ORIGIN_TABLE
            ).fetchone()[0]
            == 1
        )


@pytest.mark.parametrize(
    "kind", ["directory", "legacy_empty", "legacy_processing", "json_override"]
)
def test_existing_legacy_namespace_cannot_be_called_fresh(fresh_args, kind):
    _, args, *_ = fresh_args
    root = args["queue_root"]
    if kind in ("directory", "json_override"):
        root.mkdir(mode=0o700)
        if kind == "json_override":
            (root / "fresh-origin.json").write_text(
                json.dumps({"schema": handoff.ORIGIN_SCHEMA, "admitted": True})
            )
    else:
        queue = MergeQueue(
            root,
            target_repository_id=args["repository_id"],
            target_branch="main",
            require_target_binding=True,
        )
        root.chmod(0o700)
        if kind == "legacy_processing":
            request = queue.enqueue(
                branch_name="work/unknown", task_id="SPAR-031", commit_sha="a" * 40
            )
            queue.claim_pending_request(request, consumer_id="unknown-old-consumer")
    before = {
        p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }
    with pytest.raises(role.SparMergeOwnerError):
        handoff.prepare_fresh_native_queue(**args)
    assert before == {
        p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }


@pytest.mark.parametrize("change", ["config", "plan", "target", "uuid", "copy"])
def test_fresh_origin_rejects_changed_native_binding(fresh_args, tmp_path, change):
    _, args, *_ = fresh_args
    prepared = handoff.prepare_fresh_native_queue(**args)
    changed = dict(args)
    if change in ("config", "plan"):
        changed["scopes"] = [dict(args["scopes"][0])]
        changed["scopes"][0][change + "_cid"] = "sha256:" + "0" * 64
    elif change == "target":
        changed["target_branch"] = "foreign"
    elif change == "copy":
        import shutil

        changed["queue_root"] = tmp_path / "copied"
        shutil.copytree(args["queue_root"], changed["queue_root"])
    else:
        with role.open_duckdb_connection(prepared.database_path) as connection:
            connection.execute(
                "UPDATE control_plane_metadata SET value='00000000-0000-0000-0000-000000000001' WHERE key='database_uuid'"
            )
    with pytest.raises(role.SparMergeOwnerError):
        handoff.prepare_fresh_native_queue(**changed)


@pytest.fixture
def pair(fresh_args, request):
    real_transport = getattr(request, "param", "fake") == "native"
    repo, args, policy, amendment = fresh_args
    prepared = handoff.prepare_fresh_native_queue(**args)
    queue = role.start_queue_owner(
        prepared,
        state_dir=args["queue_root"] / "owner",
        transport=None if real_transport else FakeQuackTransport(),
        capability_probe=None if real_transport else lambda **_: _compatible_report(),
    )
    task = build_server(
        database_path=repo / "runtime/task.duckdb",
        state_dir=repo / "runtime/task-owner",
        repository_id=args["repository_id"],
        store_id="spar-task-owner",
        port=0,
        allow_legacy_board_unstall=False,
        transport=None if real_transport else FakeQuackTransport(),
        capability_probe=None if real_transport else lambda **_: _compatible_report(),
    )
    try:
        task.start()
        board = SimpleNamespace(board_namespace=amendment.board_namespace, max_lanes=1)
        issuer = handoff.NativeMergeBundleIssuer(
            handoff.NativeQueueOwner(queue, prepared), amendment=amendment, board=board
        )
        birth = current_process_birth()
        client_id = "database-implementation-daemon:" + board.board_namespace + "-0"
        request = {
            "schema": wire.REQUEST_SCHEMA,
            "request_id": "a" * 32,
            "pid": os.getpid(),
            "process_birth": birth.to_dict(),
            "process_birth_id": process_birth_id(birth),
            "client_id": client_id,
            "store_id": task.identity.store_id,
            "config_cid": amendment.launch_config_cid,
            "plan_cid": amendment.bootstrap_plan_root_cid,
        }
        token, grant = task.issue_typed_client_grant_record(
            client_id=client_id,
            process_birth_id=request["process_birth_id"],
            peer_pid=os.getpid(),
            allowed_operations=("count_tasks",),
            ttl_seconds=300,
        )
        response = {
            "schema": STATE_OWNER_BOOTSTRAP_RESPONSE_SCHEMA,
            "ok": True,
            "endpoint": task.identity.listen_uri,
            "socket_path": str(task.typed_command_socket_path()),
            "store_id": task.identity.store_id,
            "server_id": task.identity.server_id,
            "client_id": client_id,
            "process_birth_id": request["process_birth_id"],
            "token": token,
            "execution_route_policy": policy.to_dict(),
        }
        yield SimpleNamespace(
            repo=repo,
            args=args,
            amendment=amendment,
            policy=policy,
            queue=queue,
            task=task,
            issuer=issuer,
            request=request,
            task_response=response,
            session=board.board_namespace + "-0",
            task_grant=grant,
        )
    finally:
        task.stop()
        queue.stop()


def issue(pair):
    return pair.issuer.issue(
        pair.request,
        session=pair.session,
        task_response=pair.task_response,
        task_owner_identity=pair.task.identity.to_dict(),
    )


def test_actual_paired_clients_attach_and_preserve_durable_cursor(pair):
    response = issue(pair)
    bundle = wire.OwnerMergeBootstrapBundle.from_response(
        response, request=pair.request, peer_pid=os.getpid(), peer_uid=os.geteuid()
    )

    assert bundle.queue.token not in repr(bundle)
    scope = pair.args["scopes"][0]
    attached = bundle.attach_merge_runtime(
        repository_root=pair.repo,
        attempt_root=Path(scope["attempt_root"]),
        board_namespace=scope["board_namespace"],
        lane_id="0",
        admitted_config_cid=scope["config_cid"],
        admitted_plan_cid=scope["plan_cid"],
    )
    try:
        api = attached.runtime.client
        head = api.load_cursors()
        changed = dict(head["cursors"], pending_requests="known:cursor")
        updated = api.cas_cursors(
            expected_revision=head["revision"],
            expected_state_cid=head["state_cid"],
            cursors=changed,
            operation_id="advance:one",
        )
        assert updated["revision"] == 1 and api.load_cursors()["cursors"] == changed
        attached.runtime.validate_factory_binding(
            repository_root=pair.repo,
            attempt_root=Path(scope["attempt_root"]),
            board_namespace=scope["board_namespace"],
            lane_id="0",
            target_branch="main",
            admitted_config_cid=scope["config_cid"],
            admitted_plan_cid=scope["plan_cid"],
        )
    finally:
        attached.close()
    assert (
        pair.issuer.replay(
            pair.request,
            session=pair.session,
            task_owner_identity=pair.task.identity.to_dict(),
        )
        == response
    )


def test_fresh_bundle_reaches_actual_factory_without_filesystem_fallback(
    pair, monkeypatch
):
    from test.api.test_agent_supervisor_owner_recovery_integration import (
        BindingDaemon,
        CapturingPortal,
    )
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
        parse_args,
    )
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon_runner import (
        bind_database_portal_execution_from_args,
    )

    response = issue(pair)
    bundle = wire.OwnerMergeBootstrapBundle.from_response(
        response, request=pair.request, peer_pid=os.getpid(), peer_uid=os.geteuid()
    )
    scope = pair.args["scopes"][0]
    state = Path(scope["attempt_root"]).parent
    parsed = parse_args(
        [
            "--task-source-kind",
            "duckdb",
            "--authority-mode",
            "quack",
            "--board-namespace",
            scope["board_namespace"],
            "--database-path",
            str(pair.task.config.database_path),
            "--todo-path",
            str(state / "board.md"),
            "--state-dir",
            str(state),
            "--state-prefix",
            "spar_lane_0",
            "--task-shard-index",
            "0",
            "--merge-queue-dir",
            str(state / "forbidden-filesystem-fallback"),
            "--merge-target-branch",
            "main",
            "--implement",
            "--once",
        ]
    )
    attached = bundle.attach_merge_runtime(
        repository_root=pair.repo,
        attempt_root=Path(scope["attempt_root"]),
        board_namespace=scope["board_namespace"],
        lane_id="0",
        admitted_config_cid=scope["config_cid"],
        admitted_plan_cid=scope["plan_cid"],
    )
    monkeypatch.setattr(
        MergeQueue,
        "__init__",
        lambda *a, **k: pytest.fail("factory created legacy queue"),
    )
    try:
        daemon = BindingDaemon()
        bridge = bind_database_portal_execution_from_args(
            daemon,
            parsed,
            repo_root=pair.repo,
            portal_daemon_class=CapturingPortal,
            owner_merge_runtime=attached.runtime,
            admitted_owner_merge_config_cid=scope["config_cid"],
            admitted_owner_merge_plan_cid=scope["plan_cid"],
        )
        assert bridge.merge_queue is attached.runtime.queue
        assert daemon.callbacks["post_merge_recovery"]() is None
        assert not (state / "forbidden-filesystem-fallback").exists()
        assert not (pair.repo / ".merge-queue").exists()
    finally:
        attached.close()


@pytest.mark.parametrize("lost_response", [False, True])
def test_native_broker_hands_bundle_to_actual_grandchild_peer(
    pair, monkeypatch, lost_response
):
    from test.api.semantic_refactoring.test_bootstrap_controls import _materializer
    from ipfs_accelerate_py.agent_supervisor.task_sources import state_owner_bootstrap

    module = _materializer()
    listener = module._new_bootstrap_listener(lane_count=1)
    board = SimpleNamespace(
        board_namespace=pair.amendment.board_namespace,
        max_lanes=1,
        task_prefix="SPAR-",
        resolved_database_program=lambda: SimpleNamespace(
            store_id=pair.task.identity.store_id,
            quack_endpoint=pair.task.identity.listen_uri,
        ),
    )
    issuer = handoff.NativeMergeBundleIssuer(
        handoff.NativeQueueOwner(pair.queue, pair.issuer.owner.prepared),
        amendment=pair.amendment,
        board=board,
    )
    broker = module._SparStateOwnerBootstrapBroker(
        channel=listener,
        server=pair.task,
        board=board,
        paths={"owner": pair.task.config.state_dir},
        execution_route_policy=pair.policy,
        merge_bundle_issuer=issuer,
    )
    dropped = []
    if lost_response:
        original = state_owner_bootstrap._send_frame

        def send(channel, payload):
            if payload.get("schema") == wire.RESPONSE_SCHEMA and not dropped:
                dropped.append(True)
                raise BrokenPipeError("injected lost private response")
            return original(channel, payload)

        monkeypatch.setattr(state_owner_bootstrap, "_send_frame", send)
    source_root = Path(wire.__file__).resolve().parents[3]
    scope = pair.args["scopes"][0]
    context = {
        "source": str(source_root),
        "fd": listener.fileno(),
        "client": pair.request["client_id"],
        "store": pair.task.identity.store_id,
        "config": scope["config_cid"],
        "plan": scope["plan_cid"],
        "repo": str(pair.repo),
        "scope": scope,
    }
    daemon_code = r"""
import json,sys
from pathlib import Path
c=json.loads(sys.argv[1]);sys.path.insert(0,c['source'])
from ipfs_accelerate_py.agent_supervisor.task_sources.owner_merge_bootstrap import request_owner_merge_bootstrap,REQUEST_SCHEMA,receive_bundle_frame
from ipfs_accelerate_py.agent_supervisor.task_sources.state_owner_bootstrap import _connect_inherited_listener,_send_frame,StateOwnerBootstrapError
from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import current_process_birth
from ipfs_accelerate_py.agent_supervisor.merge.database_worktree_registry import process_birth_id
import os,uuid
birth=current_process_birth()
bad={'schema':REQUEST_SCHEMA,'request_id':uuid.uuid4().hex,'pid':os.getpid(),'process_birth':birth.to_dict(),'process_birth_id':process_birth_id(birth),'client_id':c['client'],'store_id':c['store'],'config_cid':'foreign-config','plan_cid':c['plan']}
with _connect_inherited_listener(os.dup(c['fd']),timeout_seconds=2) as channel:
    _send_frame(channel,bad)
    try:
        receive_bundle_frame(channel)
    except (StateOwnerBootstrapError,EOFError):
        pass
    else:
        raise AssertionError('foreign source request was accepted')
bundle=request_owner_merge_bootstrap(c['fd'],client_id=c['client'],store_id=c['store'],config_cid=c['config'],plan_cid=c['plan'],timeout_seconds=8)
s=c['scope']
attached=bundle.attach_merge_runtime(repository_root=Path(c['repo']),attempt_root=Path(s['attempt_root']),board_namespace=s['board_namespace'],lane_id=s['lane_id'],admitted_config_cid=c['config'],admitted_plan_cid=c['plan'])
try:
    assert attached.runtime.client.load_cursors()['revision']==0
    print(json.dumps({'paired':True,'tokens_exported':False}),flush=True)
finally:
    attached.close()
"""
    supervisor_code = "import subprocess,sys,json; c=json.loads(sys.argv[1]); r=subprocess.run([sys.executable,'-c',sys.argv[2],sys.argv[1]],pass_fds=(c['fd'],));sys.exit(r.returncode)"
    command = [
        sys.executable,
        "-c",
        supervisor_code,
        json.dumps(context),
        daemon_code,
        "--board-namespace",
        board.board_namespace,
        "--task-shard-count",
        "1",
        "--task-shard-index",
        "0",
        "--state-prefix",
        "spar_lane_0",
        "--database-owner-session-id",
        pair.session,
        "--state-owner-bootstrap-store-id",
        pair.task.identity.store_id,
        "--state-owner-bootstrap-fd",
        str(listener.fileno()),
        "--owner-merge-bootstrap-profile",
        handoff.BUNDLE_PROFILE,
    ]
    broker.start()
    try:
        result = subprocess.run(
            command,
            pass_fds=(listener.fileno(),),
            env={"PATH": os.defpath, "LANG": "C.UTF-8", "OPENBLAS_NUM_THREADS": "1"},
            capture_output=True,
            text=True,
            timeout=25,
        )
        assert result.returncode == 0, result.stderr[-1500:]
        assert json.loads(result.stdout.strip()) == {
            "paired": True,
            "tokens_exported": False,
        }
        assert len(issuer.grants[pair.session]) == 2
        assert len(issuer.pending) == 1
        assert broker.rejection_count >= 1
        assert not broker.stopping.is_set()
        assert pair.task.ready()["ready"] and pair.queue.ready()["ready"]
        assert bool(dropped) == lost_response
        assert (
            broker.current_by_session[pair.session]["daemon_process_birth"]["pid"]
            != os.getpid()
        )
    finally:
        broker.stop()


def test_partial_queue_grant_failure_revokes_only_new_grants(pair, monkeypatch):
    original = pair.queue.issue_typed_client_grant_record
    created = []

    def issue_one(**kwargs):
        if created:
            raise RuntimeError("injected second grant refusal")
        token, grant = original(**kwargs)
        created.append(grant.grant_id)
        return token, grant

    monkeypatch.setattr(pair.queue, "issue_typed_client_grant_record", issue_one)
    with pytest.raises(RuntimeError, match="second grant"):
        issue(pair)
    assert pair.issuer.grants == {} and pair.issuer.pending == {}
    assert created[0] in pair.queue._command_gateway._revoked_grants
    assert pair.task_grant.grant_id not in pair.task._command_gateway._revoked_grants


def test_revoked_queue_grant_cannot_replay_prior_bundle(pair):
    issue(pair)
    pair.queue.revoke_typed_client_grant(pair.issuer.grants[pair.session][0])
    from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import (
        TypedStateOwnerError,
    )

    with pytest.raises(TypedStateOwnerError):
        pair.issuer.replay(
            pair.request,
            session=pair.session,
            task_owner_identity=pair.task.identity.to_dict(),
        )


@pytest.mark.parametrize(
    "change",
    [
        "peer",
        "uid",
        "source",
        "scope",
        "same_owner",
        "same_token",
        "wrong_request",
        "extra",
    ],
)
def test_bundle_wrong_peer_generation_scope_is_denied(pair, change):
    response = json.loads(json.dumps(issue(pair)))
    peer_pid, peer_uid = os.getpid(), os.geteuid()
    if change == "peer":
        peer_pid += 10000000
    elif change == "uid":
        peer_uid += 1
    elif change == "source":
        response["scope_binding"]["config_cid"] = "foreign"
    elif change == "scope":
        response["recovery_scope_cid"] = "foreign"
    elif change == "same_owner":
        response["queue_owner_identity"] = response["task_owner_identity"]
    elif change == "same_token":
        response["queue"]["token"] = response["task"]["token"]
    elif change == "wrong_request":
        response["request_id"] = "b" * 32
    else:
        response["grant_authority"] = True
    with pytest.raises(wire.StateOwnerBootstrapError):
        wire.OwnerMergeBootstrapBundle.from_response(
            response, request=pair.request, peer_pid=peer_pid, peer_uid=peer_uid
        )


@pytest.mark.parametrize(
    "raw",
    [
        b'{"same":1,"same":2}',
        b'{"x":NaN}',
        b"[" * 2048 + b"0" + b"]" * 2048,
        b'{"x":' + b"[" * 13 + b"0" + b"]" * 13 + b"}",
    ],
)
def test_bounded_frame_rejects_ambiguous_or_deep_json(raw):
    left, right = socket.socketpair()
    try:
        left.sendall(len(raw).to_bytes(4, "big") + raw)
        with pytest.raises(wire.StateOwnerBootstrapError):
            wire.receive_bundle_frame(right)
    finally:
        left.close()
        right.close()


def test_actual_native_fresh_entry_preserves_origin_and_refuses_live_reopen(
    fresh_args, monkeypatch
):
    repo, args, _, amendment = fresh_args
    board = SimpleNamespace(
        repo_root=repo,
        task_prefix="SPAR-",
        board_namespace=amendment.board_namespace,
        max_lanes=1,
        payload={
            "merge_target_branch": "main",
            "runtime_paths": {
                "root": "runtime",
                "state": "runtime/state",
                "merge_queue": "runtime/native-queue",
            },
        },
        path=lambda raw: repo / raw,
    )
    owner = handoff.start_native_queue_for_launch(
        board=board,
        paths={},
        amendment=amendment,
        profile=handoff.FRESH_PROFILE,
        transport=FakeQuackTransport(),
        capability_probe=_compatible_report,
    )
    try:
        assert owner.prepared.manifest["scope_bindings"] == args["scopes"]
        assert owner.server.ready()["ready"] is True

        def forbidden_open(*args, **kwargs):
            raise AssertionError("active canonical DB must not be opened")

        with monkeypatch.context() as guard:
            guard.setattr(role, "open_duckdb_connection", forbidden_open)
            with pytest.raises(role.SparMergeOwnerError, match="lock"):
                handoff.start_native_queue_for_launch(
                    board=board,
                    paths={},
                    amendment=amendment,
                    profile=handoff.FRESH_PROFILE,
                    transport=FakeQuackTransport(),
                    capability_probe=_compatible_report,
                )
    finally:
        owner.close()
    resumed = handoff.start_native_queue_for_launch(
        board=board,
        paths={},
        amendment=amendment,
        profile=handoff.FRESH_PROFILE,
        transport=FakeQuackTransport(),
        capability_probe=_compatible_report,
    )
    try:
        assert resumed.prepared.database_uuid == owner.prepared.database_uuid
        assert resumed.prepared.manifest_cid == owner.prepared.manifest_cid
        assert (
            resumed.server.identity.to_dict()["generation"]
            != owner.server.identity.to_dict()["generation"]
        )
    finally:
        resumed.close()


@pytest.mark.parametrize("bad", ["legacy_profile", "foreign_board", "dict_amendment"])
def test_native_fresh_entry_denies_missing_admission_before_directory_creation(
    fresh_args, bad
):
    repo, args, _, amendment = fresh_args
    board = SimpleNamespace(board_namespace=amendment.board_namespace)
    profile = handoff.FRESH_PROFILE
    if bad == "legacy_profile":
        profile = handoff.LEGACY_PROFILE
    elif bad == "foreign_board":
        board.board_namespace = "foreign"
    else:
        amendment = amendment.to_dict()
    with pytest.raises(role.SparMergeOwnerError):
        handoff.start_native_queue_for_launch(
            board=board, paths={}, amendment=amendment, profile=profile
        )
    assert not args["queue_root"].exists()


def test_daemon_pair_request_hardens_before_credentials_and_preserves_current_source(
    monkeypatch,
):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
        implementation_daemon as daemon,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime import process_security

    events = []
    result = object()
    monkeypatch.setattr(
        process_security,
        "establish_state_authority_process_boundary",
        lambda: events.append("harden"),
    )

    def request(fd, **kwargs):
        assert events == ["harden"]
        assert fd == 21 and kwargs == {
            "client_id": "client",
            "store_id": "task",
            "config_cid": "current-config",
            "plan_cid": "current-plan",
        }
        events.append("credentials")
        return result

    monkeypatch.setattr(wire, "request_owner_merge_bootstrap", request)
    assert (
        daemon._request_process_bound_state_owner_bootstrap(
            21,
            client_id="client",
            store_id="task",
            merge_config_cid="current-config",
            merge_plan_cid="current-plan",
        )
        is result
    )
    assert events == ["harden", "credentials"]


@pytest.mark.parametrize(
    "bad", ["board", "fd", "amendment_required", "amendment_missing"]
)
def test_daemon_main_rejects_partial_pair_before_any_program_or_provider_path(
    monkeypatch, bad
):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
        implementation_daemon as daemon,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime import process_security

    _, _, amendment, *_ = launch_fixture()
    args = SimpleNamespace(
        owner_merge_bootstrap_profile=handoff.BUNDLE_PROFILE,
        board_namespace=amendment.board_namespace,
        state_owner_bootstrap_fd=21,
        require_launch_source_amendment=True,
    )
    if bad == "board":
        args.board_namespace = "foreign"
    elif bad == "fd":
        args.state_owner_bootstrap_fd = -1
    elif bad == "amendment_required":
        args.require_launch_source_amendment = False
    elif bad == "amendment_missing":
        amendment = None
    monkeypatch.setattr(
        process_security, "harden_state_authority_process", lambda: None
    )
    monkeypatch.setattr(
        process_security, "capture_state_authority_credentials", lambda: None
    )
    monkeypatch.setattr(daemon, "parse_args", lambda argv: args)
    monkeypatch.setattr(
        daemon, "_launch_source_amendment_from_args", lambda args, repo_root: amendment
    )

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "program/provider path reached without native pair admission"
        )

    monkeypatch.setattr(daemon, "database_program_from_daemon_namespace", forbidden)
    with pytest.raises(RuntimeError, match="complete admitted SPAR"):
        daemon.main([])


@pytest.mark.parametrize("pair", ["native"], indirect=True)
def test_actual_two_quack_roles_share_controller_without_capability_collision(pair):
    response = issue(pair)
    bundle = wire.OwnerMergeBootstrapBundle.from_response(
        response, request=pair.request, peer_pid=os.getpid(), peer_uid=os.geteuid()
    )
    assert pair.task.identity.process_birth == pair.queue.identity.process_birth
    assert pair.task.identity.listen_uri != pair.queue.identity.listen_uri
    assert pair.task.identity.database_uuid != pair.queue.identity.database_uuid
    assert (
        pair.task.typed_command_socket_path() != pair.queue.typed_command_socket_path()
    )
    scope = pair.args["scopes"][0]
    attached = bundle.attach_merge_runtime(
        repository_root=pair.repo,
        attempt_root=Path(scope["attempt_root"]),
        board_namespace=scope["board_namespace"],
        lane_id="0",
        admitted_config_cid=scope["config_cid"],
        admitted_plan_cid=scope["plan_cid"],
    )
    try:
        assert attached.runtime.client.load_cursors()["revision"] == 0
        # Both identities and database writer exclusions remain independent even
        # though their immutable capability reports are registered by one PID.
        for server in (pair.task, pair.queue):
            assert server.ready()["ready"] is True
            assert role._writer_lock_held(server.config.database_path)
            role._require_independent_writer_excluded(server.config.database_path)
            server.checkpoint()
            assert role._writer_lock_held(server.config.database_path)
            role._require_independent_writer_excluded(server.config.database_path)
    finally:
        attached.close()


def test_restart_preserves_creation_source_pins_as_history_not_current_admission(
    fresh_args,
):
    _, args, *_ = fresh_args
    original = handoff.prepare_fresh_native_queue(**args)
    current = dict(args, source_commit="2" * 40, source_tree="3" * 40)
    resumed = handoff.prepare_fresh_native_queue(**current)
    assert resumed.manifest_cid == original.manifest_cid
    assert resumed.manifest["source_commit"] == args["source_commit"]
    assert resumed.manifest["source_tree"] == args["source_tree"]
    # This local schema helper cannot authenticate source. Actual supervise
    # separately admits the current clean-source LaunchSourceAmendment first.
    assert not hasattr(resumed, "source_admission")


def test_operator_denied_current_amendment_prevents_queue_creation_and_grants(
    fresh_args, monkeypatch
):
    from test.api.semantic_refactoring.test_bootstrap_controls import _materializer
    from ipfs_accelerate_py.agent_supervisor.runtime import (
        configured_board_scheduler as scheduler,
    )

    repo, args, policy, _ = fresh_args
    module = _materializer()
    board = SimpleNamespace(
        max_lanes=1, resolved_database_program=lambda: SimpleNamespace(store_id="task")
    )
    events = []
    monkeypatch.setattr(module, "_load_config", lambda path: (board, {}))
    monkeypatch.setattr(module, "_assert_start_not_held", lambda board: None)

    def clean(config):
        events.append("current_clean_source")
        return ("2" * 40, "3" * 40)

    monkeypatch.setattr(module, "_assert_clean_current_tree", clean)
    monkeypatch.setattr(
        module,
        "_source_forest",
        lambda *a, **k: {"source_forest_root": "current-forest"},
    )
    monkeypatch.setattr(
        scheduler, "preflight_configured_board", lambda board: {"valid": True}
    )
    monkeypatch.setattr(
        scheduler, "configured_board_launch_plan", lambda *a, **k: {"argv": []}
    )
    monkeypatch.setattr(module, "_runtime_paths", lambda board: {})
    monkeypatch.setattr(module, "_execution_route_policy", lambda paths: policy)
    monkeypatch.setattr(module, "_launch_source_forest_receipt", lambda **kwargs: {})
    monkeypatch.setattr(module, "_harden_runtime_directories", lambda *a, **k: None)
    monkeypatch.setattr(
        handoff, "configured_queue_root", lambda board: args["queue_root"]
    )

    def denied(**kwargs):
        assert (
            kwargs["source_head"] == "2" * 40 and kwargs["repository_tree"] == "3" * 40
        )
        events.append("current_amendment_denied")
        raise module.OperatorError("current native amendment refused")

    monkeypatch.setattr(module, "_admit_launch_source_amendment", denied)

    def forbidden(*a, **k):
        raise AssertionError("owner or grants reached before current native admission")

    monkeypatch.setattr(module, "_start_state_owner", forbidden)
    monkeypatch.setattr(handoff, "start_native_queue_for_launch", forbidden)
    with pytest.raises(module.OperatorError, match="current native amendment refused"):
        module.supervise(
            repo / "config.json",
            implement=True,
            merge_owner_profile=handoff.FRESH_PROFILE,
        )
    assert events == ["current_clean_source", "current_amendment_denied"]
    assert not args["queue_root"].exists()


def test_supervisor_command_and_daemon_parser_preserve_selected_pair_profile(tmp_path):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
        PortalSupervisorConfig,
        PortalImplementationSupervisor,
    )
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
        parse_args,
    )

    config = PortalSupervisorConfig(
        todo_path=tmp_path / "todo.md",
        state_path=tmp_path / "state.json",
        strategy_path=tmp_path / "strategy.json",
        events_path=tmp_path / "events.jsonl",
        state_dir=tmp_path,
        repo_root=tmp_path,
    )
    # Command forwarding is separate from constructor/current-source admission,
    # exercised by native operator and denial tests above.
    config.owner_merge_bootstrap_profile = handoff.BUNDLE_PROFILE
    supervisor = object.__new__(PortalImplementationSupervisor)
    supervisor.config = config
    supervisor.board_namespace = config.board_namespace
    command = supervisor._build_daemon_command()
    assert command.count("--owner-merge-bootstrap-profile") == 1
    index = command.index("--owner-merge-bootstrap-profile")
    assert command[index + 1] == handoff.BUNDLE_PROFILE
    # Retained interpreter/module argv precedes the ordinary daemon arguments.
    parsed = parse_args(command[command.index("--todo-path") :])
    assert parsed.owner_merge_bootstrap_profile == handoff.BUNDLE_PROFILE
