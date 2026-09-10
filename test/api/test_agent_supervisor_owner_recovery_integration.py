"""Real DuckDB/typed socket recovery; only admitted Git scratch is local."""

from __future__ import annotations

import subprocess
import concurrent.futures
from types import SimpleNamespace

import pytest

from test.api.causal_federation.test_typed_state_owner import _gateway, _install
from ipfs_accelerate_py.agent_supervisor.merge.checkout_lock import (
    checkout_repository_id,
)
from ipfs_accelerate_py.agent_supervisor.merge.merge_queue import MergeQueue
from ipfs_accelerate_py.agent_supervisor.merge.merge_train import MergeTrain
from ipfs_accelerate_py.agent_supervisor.merge.owner_merge_queue import (
    OwnerMergeQueueClient,
    SERVICE_OPERATIONS as QUEUE_OPERATIONS,
)
from ipfs_accelerate_py.agent_supervisor.merge.owner_merge_queue_adapter import (
    OwnerMergeQueueAdapter,
)
from ipfs_accelerate_py.agent_supervisor.merge.owner_recovery_adapter import (
    OwnerMergeRecoveryRuntime,
    OwnerRecoveryCursorConflict,
    OwnerRecoveryRuntimeError,
)
from ipfs_accelerate_py.agent_supervisor.merge.owner_recovery_runtime import (
    OwnerRecoveryRuntimeClient,
    SERVICE_OPERATIONS as RECOVERY_OPERATIONS,
)
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.residual_authority import (
    SPAR_BOARD_NAMESPACE,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import (
    TypedStateOwnerConnection,
    TypedStateOwnerError,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import (
    DatabasePortalExecutionBridge,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    parse_args,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon_runner import (
    bind_database_portal_execution_from_args,
)


@pytest.fixture
def admitted(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    for args in (
        ["init", "-q", "-b", "main"],
        ["config", "user.name", "Recovery Test"],
        ["config", "user.email", "recovery@example.invalid"],
        ["commit", "-qm", "base", "--allow-empty"],
    ):
        subprocess.run(["git", *args], cwd=repo, check=True)
    repository_id = checkout_repository_id(repo)
    queue_dir = tmp_path / "canonical-queue"
    MergeQueue(
        queue_dir,
        target_repository_id=repository_id,
        target_branch="main",
        require_target_binding=True,
    )
    db = queue_dir / "merge_queue.duckdb"
    _install(db)
    gateway, connection = _gateway(db, tmp_path / "owner.sock")
    connection.execute(
        "UPDATE store_generations SET birth_id=?",
        [gateway.identity["process_birth_id"]],
    )
    gateway.bind_legacy_merge_queue_service(
        expected_identity=dict(gateway.identity),
        repository_id=repository_id,
        target_branch="main",
        max_age_seconds=3600,
        max_queue_size=100,
        max_processing=10,
        max_attempts=3,
        max_worktree_bytes=None,
    )
    state_dir = tmp_path / "native-state"
    binding = {
        "board_namespace": SPAR_BOARD_NAMESPACE,
        "config_cid": "sha256:" + "a" * 64,
        "plan_cid": "sha256:" + "b" * 64,
        "lane_id": "0",
        "attempt_root": str(state_dir / "spar_database_portal_attempts"),
    }
    migration = gateway.provision_legacy_merge_recovery_schema(
        expected_identity=dict(gateway.identity),
        repository_id=repository_id,
        target_branch="main",
        migration_id="migration:integration",
        scope_bindings=[binding],
    )
    gateway.bind_legacy_merge_recovery_service(
        expected_identity=dict(gateway.identity),
        repository_id=repository_id,
        target_branch="main",
    )
    scope_cid = migration["scope_cids"][0]
    clients = []

    def attach(consumer="native:lane-0", *, pair=True, binding_override=None):
        entity = {
            "repository_id": repository_id,
            "target_branch": "main",
            "consumer_id": consumer,
        }

        def client(operations, scopes):
            token, grant = gateway.issue_grant(
                client_id=consumer,
                process_birth_id="birth:" + consumer,
                allowed_operations=tuple(operations),
                entity_scopes=scopes,
            )
            connection = TypedStateOwnerConnection(
                socket_path=gateway.socket_path,
                token=token,
                client_id=consumer,
                process_birth_id="birth:" + consumer,
                store_id="control.duckdb",
            )
            clients.append(connection)
            return connection, grant

        qc, qgrant = client(QUEUE_OPERATIONS, entity)
        queue = OwnerMergeQueueAdapter(OwnerMergeQueueClient(qc, **entity))
        if not pair:
            return SimpleNamespace(queue=queue, queue_grant=qgrant)
        rc, grant = client(
            RECOVERY_OPERATIONS, {**entity, "recovery_scope_cid": scope_cid}
        )
        api = OwnerRecoveryRuntimeClient(rc, **entity, recovery_scope_cid=scope_cid)
        supplied = {**binding, **(binding_override or {})}
        runtime = OwnerMergeRecoveryRuntime(
            queue, api, repository_root=repo, **supplied
        )
        return SimpleNamespace(
            queue=queue, runtime=runtime, api=api, grant=grant, queue_grant=qgrant
        )

    value = SimpleNamespace(
        repo=repo,
        queue_dir=queue_dir,
        state_dir=state_dir,
        gateway=gateway,
        connection=connection,
        binding=binding,
        attach=attach,
    )
    try:
        yield value
    finally:
        for client in clients:
            client.close()
        gateway.stop()
        connection.close()


def bridge_for(own, pair):
    return DatabasePortalExecutionBridge(
        task_source=object(),
        attempt_root=own.binding["attempt_root"],
        repository_root=own.repo,
        portal_factory=lambda *_: None,
        merge_queue=pair.queue,
        merge_target_branch="main",
    )


def train_for(own, pair, **kwargs):
    return MergeTrain(
        repo_root=own.repo, queue=pair.queue, target_branch="main", **kwargs
    )


def test_bridge_cursor_survives_reconstruction_and_stale_cas(admitted):
    own = admitted
    first = own.attach()
    bridge = bridge_for(own, first)
    cursors = bridge._load_post_merge_recovery_cursors()
    cursors["completed_requests"] = "request:064"
    bridge._save_post_merge_recovery_cursors(cursors)
    second = own.attach()
    other = bridge_for(own, second)
    assert other._load_post_merge_recovery_cursors() == cursors
    newer = dict(cursors, completed_requests="request:096")
    other._save_post_merge_recovery_cursors(newer)
    with pytest.raises(OwnerRecoveryCursorConflict):
        bridge._save_post_merge_recovery_cursors(
            dict(cursors, completed_requests="request:032")
        )
    assert first.api.load_cursors()["cursors"] == newer
    assert not own.state_dir.exists()
    assert not (own.repo / ".merge-queue").exists()


def test_train_uses_exact_owner_consumer_and_preserves_receipt_versions(admitted):
    own = admitted
    first = own.attach()
    train = train_for(own, first)
    assert train.owner_id == first.runtime.consumer_id
    assert train.consumer_lock_path is None and train.receipt_dir is None
    assert train.distributed_publication_ledger_path is None
    assert not (own.repo / ".merge-queue").exists()
    assert not (own.queue_dir / "train").exists()
    with pytest.raises(OwnerRecoveryRuntimeError, match="active train lease"):
        train._write_receipt("stable-key", {"status": "prepared"})
    assert train.run_under_consumer_lease(
        lambda: train._write_receipt("stable-key", {"status": "prepared"})
    ) == (True, None)
    assert train.run_under_consumer_lease(
        lambda: train._write_receipt(
            "stable-key", {"status": "validated", "accepted": False}
        )
    ) == (True, None)
    second = own.attach()
    reconstructed = train_for(own, second)
    assert reconstructed._read_receipt("stable-key") == {
        "status": "validated",
        "accepted": False,
    }
    assert second.api.get_receipt("stable-key", revision=1)["receipt"] == {
        "status": "prepared"
    }
    assert second.api.get_receipt("stable-key")["revision"] == 2
    assert not list(first.runtime.local_state_dir.rglob("*receipt*.json"))


def test_actual_train_claim_and_git_merge_use_owner_receipts_without_task_acceptance(
    admitted,
):
    own = admitted
    pair = own.attach()

    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=own.repo, check=True, capture_output=True, text=True
        ).stdout.strip()

    git("switch", "-c", "work/candidate")
    (own.repo / "result.txt").write_text("disposable candidate\n")
    git("add", "result.txt")
    git("commit", "-qm", "candidate")
    candidate = git("rev-parse", "HEAD")
    git("switch", "main")
    request = pair.queue.enqueue(
        branch_name="work/candidate",
        task_id="task:disposable",
        canonical_task_id="task:disposable",
        commit_sha=candidate,
        lane_id="0",
    )
    train = train_for(own, pair)
    result = train.run_once()
    assert result["status"] == "merged"
    assert git("show", "main:result.txt") == "disposable candidate"
    completed = pair.queue.get(request.request_id)
    assert completed.status == "completed"
    key = train._dedupe_key(request.canonical_identity, candidate)
    receipt = pair.api.get_receipt(key)
    assert receipt["receipt"]["status"] == "merged"
    # Queue completion and transport receipts cannot accept a canonical task.
    assert (
        own.connection.execute(
            "SELECT status FROM tasks WHERE task_cid='task:typed-owner'"
        ).fetchone()[0]
        == "ready"
    )
    assert not (own.repo / ".merge-queue").exists()
    assert not (own.queue_dir / "train").exists()


def test_unknown_callback_retains_canonical_lease_across_reconstructed_train(admitted):
    own = admitted
    first, second = own.attach(), own.attach("native:other-lane")
    train, other = train_for(own, first), train_for(own, second)

    def unknown():
        assert other.run_under_consumer_lease(
            lambda: pytest.fail("concurrent consumer")
        ) == (False, None)
        raise OSError("unobserved callback closure")

    with pytest.raises(OSError):
        train.run_under_consumer_lease(unknown)
    assert (
        train.status()["owner_recovery"]["consumer_custody"]
        == "unknown_callback_custody_retained"
    )
    own.connection.execute("UPDATE legacy_merge_recovery_leases SET expires_at=0")
    assert other.run_under_consumer_lease(
        lambda: pytest.fail("expiry is not closure")
    ) == (False, None)
    third = own.attach()
    assert train_for(own, third).run_under_consumer_lease(
        lambda: pytest.fail("new client cannot adopt old lease")
    ) == (False, None)
    with pytest.raises(OwnerRecoveryRuntimeError, match="unknown merge callback"):
        train.run_under_consumer_lease(lambda: None)


def test_caught_nested_unknown_callback_does_not_release_outer_lease(admitted):
    first, second = admitted.attach(), admitted.attach("native:other-lane")
    train, other = train_for(admitted, first), train_for(admitted, second)

    def unknown():
        raise OSError("unknown nested callback")

    def outer():
        with pytest.raises(OSError):
            train.run_under_consumer_lease(unknown)

    with pytest.raises(OwnerRecoveryRuntimeError, match="nested unknown"):
        train.run_under_consumer_lease(outer)
    assert other.run_under_consumer_lease(
        lambda: pytest.fail("nested uncertainty released custody")
    ) == (False, None)


def test_another_thread_cannot_publish_under_active_consumer_lease(admitted):
    pair = admitted.attach()
    train = train_for(admitted, pair)

    def outer():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                train._write_receipt, "foreign-thread", {"accepted": False}
            )
            with pytest.raises(OwnerRecoveryRuntimeError, match="active train lease"):
                future.result(timeout=5)
        train._write_receipt("same-thread", {"accepted": False})

    assert train.run_under_consumer_lease(outer) == (True, None)
    assert pair.api.get_receipt("foreign-thread") is None
    assert pair.api.get_receipt("same-thread")["revision"] == 1


def test_lost_acquire_reply_replays_exact_pre_callback_operation(admitted, monkeypatch):
    pair = admitted.attach()
    train = train_for(admitted, pair)
    acquire = pair.api.acquire_consumer_lease
    operations, callbacks = [], []

    def lost_once(**kwargs):
        operations.append(kwargs["operation_id"])
        result = acquire(**kwargs)
        if len(operations) == 1:
            raise OSError("committed acquire reply was lost")
        return result

    monkeypatch.setattr(pair.api, "acquire_consumer_lease", lost_once)
    with pytest.raises(OSError):
        train.run_under_consumer_lease(lambda: callbacks.append("entered"))
    assert callbacks == []
    assert (
        train.status()["owner_recovery"]["consumer_custody"]
        == "exact_pre_callback_acquire_reply_pending"
    )
    assert train.run_under_consumer_lease(lambda: callbacks.append("entered")) == (
        True,
        None,
    )
    assert len(operations) == 2 and operations[0] == operations[1]
    assert callbacks == ["entered"]
    row = admitted.connection.execute(
        "SELECT fence_epoch,state FROM legacy_merge_recovery_leases"
    ).fetchone()
    assert (row[0], row[1]) == (1, "released")


def test_lost_release_reply_replays_only_known_completed_callback_release(
    admitted, monkeypatch
):
    pair = admitted.attach()
    train = train_for(admitted, pair)
    release = pair.api.release_consumer_lease
    operations, callbacks = [], []

    def lost_once(**kwargs):
        operations.append(kwargs["operation_id"])
        result = release(**kwargs)
        if len(operations) == 1:
            raise OSError("committed release reply was lost")
        return result

    monkeypatch.setattr(pair.api, "release_consumer_lease", lost_once)
    with pytest.raises(OSError):
        train.run_under_consumer_lease(lambda: callbacks.append("first"))
    assert callbacks == ["first"]
    assert (
        train.status()["owner_recovery"]["consumer_custody"]
        == "exact_release_reply_pending"
    )
    assert train.run_under_consumer_lease(lambda: callbacks.append("second")) == (
        True,
        None,
    )
    assert len(operations) == 3 and operations[0] == operations[1] != operations[2]
    assert callbacks == ["first", "second"]
    row = admitted.connection.execute(
        "SELECT fence_epoch,state FROM legacy_merge_recovery_leases"
    ).fetchone()
    assert (row[0], row[1]) == (2, "released")


@pytest.mark.parametrize(
    "field,value",
    [
        ("board_namespace", "foreign"),
        ("config_cid", "sha256:" + "c" * 64),
        ("plan_cid", "sha256:" + "c" * 64),
        ("lane_id", "2"),
        ("attempt_root", "/foreign/attempts"),
    ],
)
def test_exact_native_scope_cannot_be_reinterpreted(admitted, field, value):
    with pytest.raises(OwnerRecoveryRuntimeError, match="namespace"):
        admitted.attach(binding_override={field: value})
    assert not admitted.state_dir.exists()


def test_unpaired_queue_refuses_train_and_bridge_before_filesystem_fallback(admitted):
    pair = admitted.attach(pair=False)
    with pytest.raises(ValueError, match="recovery runtime"):
        train_for(admitted, pair)
    with pytest.raises(ValueError, match="recovery runtime"):
        bridge_for(admitted, pair)
    assert not (admitted.repo / ".merge-queue").exists()
    assert not admitted.state_dir.exists()


def test_authority_read_failure_is_never_an_empty_cursor_or_receipt(admitted):
    pair = admitted.attach()
    bridge = bridge_for(admitted, pair)
    train = train_for(admitted, pair)
    admitted.gateway.revoke_grant(pair.grant.grant_id)
    with pytest.raises(TypedStateOwnerError):
        bridge._load_post_merge_recovery_cursors()
    with pytest.raises(TypedStateOwnerError):
        train._read_receipt("missing")
    assert not (admitted.repo / ".merge-queue").exists()


def test_queue_grant_cannot_substitute_for_separately_scoped_recovery_grant(admitted):
    pair = admitted.attach()
    queue_client = pair.queue._client
    unauthorized = OwnerRecoveryRuntimeClient(
        queue_client.connection,
        repository_id=queue_client.repository_id,
        target_branch=queue_client.target_branch,
        consumer_id=queue_client.consumer_id,
        recovery_scope_cid=pair.api.recovery_scope_cid,
    )
    with pytest.raises(TypedStateOwnerError):
        unauthorized.describe_scope()
    assert pair.api.load_cursors()["revision"] == 0


class BindingDaemon:
    task_source = object()

    def __init__(self):
        self.callbacks = {}

    def bind_execution_callbacks(self, **values):
        self.callbacks.update(values)

    def bind_post_merge_recovery(self, callback):
        self.callbacks["post_merge_recovery"] = callback

    def bind_superseded_consumed_attempt_recovery(self, callback):
        pass

    def bind_protected_preservation_recovery(self, callback):
        pass

    def bind_protected_reconciliation_self_lock_recovery(self, callback):
        pass

    def bind_merge_train_recovery(self, **values):
        self.callbacks["merge_train_recovery"] = values

    def bind_pending_merge_consume(self, callback):
        pass

    @staticmethod
    def _database_portal_evidence_digest(value):
        return "sha256:" + "0" * 64

    @staticmethod
    def recover_blocked_post_merge_declared_outputs(value):
        pytest.fail("empty queue cannot recover task")

    @staticmethod
    def preauthorize_post_merge_declared_output_recovery(value):
        pytest.fail("empty queue cannot authorize task")


class CapturingPortal:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def arguments(own, *, namespace=SPAR_BOARD_NAMESPACE, authority="quack"):
    return parse_args(
        [
            "--task-source-kind",
            "duckdb",
            "--authority-mode",
            authority,
            "--board-namespace",
            namespace,
            "--database-path",
            str(own.queue_dir / "merge_queue.duckdb"),
            "--todo-path",
            str(own.state_dir / "board.md"),
            "--state-dir",
            str(own.state_dir),
            "--state-prefix",
            "spar",
            "--merge-queue-dir",
            str(own.state_dir / "forbidden-legacy-fallback"),
            "--merge-target-branch",
            "main",
            "--task-shard-index",
            "0",
            "--implement",
            "--once",
        ]
    )


def test_native_factory_injects_paired_queue_before_full_empty_recovery_pass(
    admitted, monkeypatch
):
    pair = admitted.attach()
    monkeypatch.setattr(
        MergeQueue,
        "__init__",
        lambda *a, **k: pytest.fail("native factory opened a legacy queue"),
    )
    daemon = BindingDaemon()
    bridge = bind_database_portal_execution_from_args(
        daemon,
        arguments(admitted),
        repo_root=admitted.repo,
        portal_daemon_class=CapturingPortal,
        owner_merge_runtime=pair.runtime,
        admitted_owner_merge_config_cid=admitted.binding["config_cid"],
        admitted_owner_merge_plan_cid=admitted.binding["plan_cid"],
    )
    assert bridge.merge_queue is pair.queue
    assert daemon.callbacks["merge_train_recovery"]["merge_queue"] is pair.queue
    assert daemon.callbacks["post_merge_recovery"]() is None
    assert pair.api.load_cursors()["revision"] == 0
    assert not (admitted.repo / ".merge-queue").exists()
    assert not (admitted.state_dir / "forbidden-legacy-fallback").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("task_shard_index", 1),
        ("state_prefix", "other"),
        ("merge_target_branch", "foreign"),
        ("board_namespace", "foreign"),
    ],
)
def test_factory_rejects_drift_after_native_pair_admission(admitted, field, value):
    pair = admitted.attach()
    parsed = arguments(admitted)
    setattr(parsed, field, value)
    daemon = BindingDaemon()
    with pytest.raises(OwnerRecoveryRuntimeError, match="factory differs"):
        bind_database_portal_execution_from_args(
            daemon,
            parsed,
            repo_root=admitted.repo,
            portal_daemon_class=CapturingPortal,
            owner_merge_runtime=pair.runtime,
            admitted_owner_merge_config_cid=admitted.binding["config_cid"],
            admitted_owner_merge_plan_cid=admitted.binding["plan_cid"],
        )
    assert not daemon.callbacks
    assert not admitted.state_dir.exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("admitted_owner_merge_config_cid", "sha256:" + "c" * 64),
        ("admitted_owner_merge_plan_cid", "sha256:" + "c" * 64),
        ("admitted_owner_merge_config_cid", None),
        ("admitted_owner_merge_plan_cid", None),
    ],
)
def test_factory_requires_current_native_config_and_plan_admission(
    admitted, field, value
):
    pair = admitted.attach()
    values = {
        "admitted_owner_merge_config_cid": admitted.binding["config_cid"],
        "admitted_owner_merge_plan_cid": admitted.binding["plan_cid"],
    }
    values[field] = value
    parsed = arguments(admitted)
    # Arbitrary parsed values are not a substitute for the native handoff.
    parsed.config_cid = admitted.binding["config_cid"]
    parsed.plan_cid = admitted.binding["plan_cid"]
    daemon = BindingDaemon()
    with pytest.raises(OwnerRecoveryRuntimeError, match="factory differs"):
        bind_database_portal_execution_from_args(
            daemon,
            parsed,
            repo_root=admitted.repo,
            portal_daemon_class=CapturingPortal,
            owner_merge_runtime=pair.runtime,
            **values,
        )
    assert not daemon.callbacks
    assert not admitted.state_dir.exists()


@pytest.mark.parametrize(
    "namespace,authority",
    [
        (SPAR_BOARD_NAMESPACE, "embedded_exclusive"),
        (SPAR_BOARD_NAMESPACE, "quack"),
        ("other-board", "quack"),
    ],
)
def test_configured_native_factory_refuses_missing_admission_pair_before_workers(
    admitted, monkeypatch, namespace, authority
):
    monkeypatch.setattr(
        MergeQueue, "__init__", lambda *a, **k: pytest.fail("legacy fallback")
    )
    daemon = BindingDaemon()
    with pytest.raises(RuntimeError, match="native migration and paired grants"):
        bind_database_portal_execution_from_args(
            daemon,
            arguments(admitted, namespace=namespace, authority=authority),
            repo_root=admitted.repo,
            portal_daemon_class=CapturingPortal,
        )
    assert daemon.callbacks == {}
    assert not admitted.state_dir.exists()
