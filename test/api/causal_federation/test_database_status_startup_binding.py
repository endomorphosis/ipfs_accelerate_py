"""Real local Quack startup binds read-only status before socket publication."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import socket

import pytest

from test.api.causal_federation.test_typed_state_owner import _install
from ipfs_accelerate_py.agent_supervisor.merge.merge_queue import MergeQueue
from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
    QuackStateServerControlError, QuackStateServerOwnershipError, ServerLifecycle, build_server,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import open_duckdb_connection
from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import (
    STATUS_BOOTSTRAP_CLIENT_ID, TypedStateOwnerConnection, TypedStateOwnerError, TypedStateOwnerGateway,
)

CID = "task:typed-owner"
REPOSITORY = "repository:sha256:" + "a" * 64


@pytest.fixture
def startup(tmp_path):
    database = tmp_path / "control.duckdb"
    _install(database)
    with open_duckdb_connection(database) as connection:
        connection.execute("UPDATE tasks SET plan_cid=?, identity_json=?, body_json=?",
            ["plan:sealed", json.dumps({"repository_tree_id": "tree:sealed"}),
             json.dumps({"board_namespace": "board:sealed"})])
    queue = MergeQueue(tmp_path / "queue", target_repository_id=REPOSITORY,
                       target_branch="main", require_target_binding=True)
    queue.enqueue(branch_name="candidate/first", task_id="CASF-TYPED", canonical_task_id=CID,
        canonical_task_key=CID, commit_sha="a" * 40, metadata={"completion_task_cids": {"CASF-TYPED": CID}})
    servers = []
    def make():
        server = build_server(database_path=database, state_dir=tmp_path / "owner", port=0,
            store_id="control.duckdb", secret_handle="handle:startup-status-test",
            typed_command_socket_path=tmp_path / "owner.sock", allow_legacy_board_unstall=False)
        servers.append(server)
        return server
    arguments = dict(board_namespace="board:sealed", plan_root_cid="plan:sealed",
        repository_tree_id="tree:sealed", task_cids=[CID], queue_dir=queue.queue_dir,
        target_repository_id=REPOSITORY, target_branch="main")
    try:
        yield make, queue, arguments
    finally:
        for server in reversed(servers):
            server.stop()
            assert server._command_gateway is None
            assert server._connection is None and server._owner is None
        refs = []
        for fd in Path('/proc/self/fd').iterdir():
            try:
                target = os.readlink(fd)
            except OSError:
                continue
            if target == str(tmp_path) or target.startswith(str(tmp_path) + '/'):
                refs.append(target)
        assert not refs and not list(tmp_path.rglob('.git'))
        details = tmp_path.stat()
        with (tmp_path.parent / 'closed-startup-fixtures.jsonl').open('a') as ledger:
            ledger.write(json.dumps({'path': str(tmp_path), 'device': details.st_dev,
                'inode': details.st_ino, 'own_open_references': refs, 'servers_closed': True}) + '\n')
        shutil.rmtree(tmp_path)


def client(server):
    return TypedStateOwnerConnection(socket_path=server.typed_command_socket_path(),
        token=server.typed_command_token_path().read_text().strip(),
        client_id=STATUS_BOOTSTRAP_CLIENT_ID, process_birth_id="birth:startup-reader",
        store_id="control.duckdb", status_bootstrap=True)


def test_real_startup_binds_full_database_then_queue_before_publication(startup, monkeypatch):
    make, _queue, arguments = startup
    server = make(); calls = []
    for name in ('bind_database_status_scope', 'bind_legacy_merge_queue_status_scope', 'start'):
        original = getattr(TypedStateOwnerGateway, name)
        def observed(gateway, *args, _name=name, _original=original, **kwargs):
            assert not server.typed_command_socket_path().exists()
            assert not server.typed_command_token_path().exists()
            calls.append(_name)
            return _original(gateway, *args, **kwargs)
        monkeypatch.setattr(TypedStateOwnerGateway, name, observed)
    server.configure_database_status_before_start(**arguments)
    with pytest.raises(QuackStateServerControlError, match='rebound'):
        server.configure_database_status_before_start(**arguments)
    arguments['task_cids'].append('task:caller-mutated-after-binding')
    server.start()
    assert calls == ['bind_database_status_scope', 'bind_legacy_merge_queue_status_scope', 'start']
    connection = client(server)
    try:
        result = connection.legacy_merge_queue_task_observation([CID], task_cid=CID)
        assert result['population_complete'] and len(result['matching_rows']) == 1
        assert all(result[key] is False for key in ('guard_retained', 'claim_authority',
                                                   'retry_authorized', 'completion_authority'))
    finally:
        connection.close()
    with pytest.raises(QuackStateServerControlError, match='rebound'):
        server.configure_database_status_before_start(**arguments)


@pytest.mark.parametrize('failure', ['population', 'target', 'replacement', 'wal', 'busy', 'interrupt'])
def test_failed_binding_has_no_partial_status_and_preserves_foreign_artifacts(startup, failure, monkeypatch):
    make, queue, arguments = startup
    server = make()
    if failure == 'population': arguments['task_cids'] = ['task:missing']
    if failure == 'target': arguments['target_branch'] = 'foreign'
    server.configure_database_status_before_start(**arguments)
    lock = None
    if failure == 'replacement':
        path = queue.queue_dir / 'merge_queue.duckdb'
        path.rename(path.with_suffix('.old')); shutil.copyfile(path.with_suffix('.old'), path)
    elif failure == 'wal': (queue.queue_dir / 'merge_queue.duckdb.wal').write_bytes(b'unknown writer')
    elif failure == 'busy':
        lock = (queue.queue_dir / '.merge_queue.duckdb.lock').open('r+')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    error = KeyboardInterrupt('binding interrupted')
    if failure == 'interrupt':
        def interrupted(*args, **kwargs): raise error
        monkeypatch.setattr(TypedStateOwnerGateway, 'bind_legacy_merge_queue_status_scope', interrupted)
    foreign = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    foreign.bind(str(server.typed_command_socket_path())); foreign.listen()
    socket_inode = server.typed_command_socket_path().stat().st_ino
    token = server.typed_command_token_path(); token.parent.mkdir(exist_ok=True)
    token.write_bytes(b'prior-status-credential-preserved')
    try:
        with pytest.raises(BaseException) as refused: server.start()
        if failure == 'interrupt': assert refused.value is error
        assert server.lifecycle is ServerLifecycle.FAILED
        assert server._connection is None and server._owner is None and server._command_gateway is None
        server.stop()
        assert server.typed_command_socket_path().stat().st_ino == socket_inode
        assert token.read_bytes() == b'prior-status-credential-preserved'
        with server.owner_lock_path().open('r+') as candidate:
            fcntl.flock(candidate, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        foreign.close()
        if lock is not None: lock.close()


def test_second_configured_start_cannot_remove_live_owner_status(startup):
    make, _queue, arguments = startup
    first = make(); first.configure_database_status_before_start(**arguments); first.start()
    socket_inode = first.typed_command_socket_path().stat().st_ino
    token = first.typed_command_token_path().read_bytes()
    status = first.status_path().read_bytes()
    second = make(); second.configure_database_status_before_start(**arguments)
    with pytest.raises(QuackStateServerOwnershipError): second.start()
    second.stop()
    assert first.typed_command_socket_path().stat().st_ino == socket_inode
    assert first.typed_command_token_path().read_bytes() == token
    assert first.status_path().read_bytes() == status
    assert first.ready()


@pytest.mark.parametrize('mode', ['unconfigured', 'database-only'])
def test_existing_unconfigured_or_database_only_startup_does_not_admit_queue(startup, mode):
    make, _queue, arguments = startup
    server = make()
    if mode == 'database-only':
        server.configure_database_status_before_start(**{k: arguments[k] for k in
            ('board_namespace', 'plan_root_cid', 'repository_tree_id', 'task_cids')})
    server.start()
    assert server.ready()
    if mode == 'unconfigured':
        with pytest.raises(TypedStateOwnerError): client(server)
    else:
        connection = client(server)
        try:
            assert connection.completion_closeout_snapshot([CID])['completion_authority'] is False
            with pytest.raises(TypedStateOwnerError):
                connection.legacy_merge_queue_task_observation([CID], task_cid=CID)
        finally: connection.close()


@pytest.mark.parametrize('invalid', ['duplicate', 'nontext', 'partial-queue'])
def test_invalid_configuration_is_refused_without_starting_or_opening_stores(startup, invalid):
    make, _queue, arguments = startup
    server = make()
    if invalid == 'duplicate': arguments['task_cids'] = [CID, CID]
    elif invalid == 'nontext': arguments['task_cids'] = [True]
    else: arguments['queue_dir'] = None
    with pytest.raises(QuackStateServerControlError): server.configure_database_status_before_start(**arguments)
    assert server.lifecycle is ServerLifecycle.CREATED
    assert server._connection is None and server._owner is None
    assert not server.typed_command_socket_path().exists()
