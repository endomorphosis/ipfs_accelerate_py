"""Host-admitted EAAEF daemon gateway loads pinned extensions without INSTALL."""

from __future__ import annotations

import copy
import hashlib
import os
import time
from pathlib import Path
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_daemon_gateway import (
    QuackDaemonGatewayError,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.eaaef_host_admitted_daemon_gateway import (
    _CAS_TASK_STATUS_SQL,
    _admitted_home_directory,
    _admitted_httpfs_extension,
    _connect_admitted_duckdb,
    _submit_owner_mutation,
)


class _FakeConnection:
    def __init__(self) -> None:
        self.statements: list[str] = []

    def execute(self, statement: str, parameters: Any = None) -> None:
        del parameters
        self.statements.append(statement)


class _FakeDuckDB:
    def __init__(self) -> None:
        self.connection = _FakeConnection()

    def connect(self, database: str) -> _FakeConnection:
        assert database == ":memory:"
        return self.connection


def _pin_extension_pair(tmp_path: Path) -> tuple[Path, Path]:
    directory = (
        tmp_path / ".duckdb" / "extensions" / "v1.5.5" / "linux_arm64"
    )
    directory.mkdir(parents=True)
    quack = directory / "quack.duckdb_extension"
    httpfs = directory / "httpfs.duckdb_extension"
    quack.write_bytes(b"quack")
    httpfs.write_bytes(b"httpfs")
    return quack, httpfs


def _file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_admitted_httpfs_is_the_pinned_quack_sibling(tmp_path: Path) -> None:
    quack, httpfs = _pin_extension_pair(tmp_path)
    assert _admitted_httpfs_extension(quack) == httpfs


def test_admitted_httpfs_rejects_a_missing_sibling(tmp_path: Path) -> None:
    quack = tmp_path / "quack.duckdb_extension"
    quack.write_bytes(b"quack")
    with pytest.raises(QuackDaemonGatewayError, match="httpfs"):
        _admitted_httpfs_extension(quack)


def test_admitted_home_directory_is_the_duckdb_dotdir_parent(tmp_path: Path) -> None:
    quack, _httpfs = _pin_extension_pair(tmp_path)
    assert _admitted_home_directory(quack) == tmp_path


def test_connect_admitted_duckdb_loads_httpfs_then_quack_without_install(
    tmp_path: Path,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
        eaaef_host_admitted_daemon_gateway as gateway_module,
    )

    quack, httpfs = _pin_extension_pair(tmp_path)
    extensions = gateway_module._seal_admitted_extensions(
        quack_path=quack,
        quack_sha256=_file_digest(quack),
        httpfs_path=httpfs,
        httpfs_sha256=_file_digest(httpfs),
    )
    duckdb = _FakeDuckDB()
    try:
        connection = _connect_admitted_duckdb(duckdb, extensions)
        assert connection is duckdb.connection
        escaped_httpfs = str(extensions.httpfs_path).replace("'", "''")
        escaped_quack = str(extensions.quack_path).replace("'", "''")
        assert connection.statements == [
            "SET autoinstall_known_extensions=false",
            "SET autoload_known_extensions=false",
            f"LOAD '{escaped_httpfs}'",
            f"LOAD '{escaped_quack}'",
        ]
        assert all(
            "INSTALL" not in statement for statement in connection.statements
        )
    finally:
        extensions.close()


def test_import_admitted_duckdb_consumes_the_immutable_receipt_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
        eaaef_host_admitted_daemon_gateway as gateway_module,
    )
    from ipfs_accelerate_py.agent_supervisor.validation import (
        eaaef_host_admission,
    )

    module_path = tmp_path / "site-packages/duckdb/__init__.py"
    module_path.parent.mkdir(parents=True)
    module_path.write_text("# immutable receipt target\n", encoding="utf-8")
    native_module_path = tmp_path / "site-packages/_duckdb.test.so"
    native_module_path.write_bytes(b"native-duckdb")
    extension, httpfs = _pin_extension_pair(tmp_path)
    imported = SimpleNamespace(__version__="1.5.5", __file__=str(module_path))
    imported_native = SimpleNamespace(__file__=str(native_module_path))
    monkeypatch.setattr(
        gateway_module.importlib,
        "import_module",
        lambda name: {
            "duckdb": imported,
            "_duckdb": imported_native,
        }[name],
    )
    extension_fingerprint = (
        eaaef_host_admission.REQUIRED_QUACK_EXTENSION_FINGERPRINT
    )
    monkeypatch.setattr(
        eaaef_host_admission,
        "APPROVED_IMPORT_ROOT",
        module_path.parent.parent,
    )
    receipt = {
        "decision": "admitted",
        "evidence": {
            "required_duckdb": "1.5.5",
            "required_quack": "1.5.5+core",
            "required_quack_extension_version": "c154811",
            "observed_duckdb": "1.5.5",
            "observed_module_path": str(module_path),
            "observed_module_sha256": _file_digest(module_path),
            "observed_native_module_path": str(native_module_path),
            "observed_native_module_sha256": _file_digest(native_module_path),
            "required_quack_extension_fingerprint": extension_fingerprint,
            "required_quack_platform": "linux-aarch64",
            "under_approved_import_root": True,
            "quack_extension_sha256": _file_digest(extension),
            "httpfs_extension_path": str(httpfs),
            "httpfs_extension_sha256": _file_digest(httpfs),
            "quack_probe": {
                "passes_health_check": True,
                "extension": {
                    "extension_version": "c154811",
                    "installed_from": "core",
                    "install_path": str(extension),
                },
                "extension_fingerprint": extension_fingerprint,
                "platform_name": "linux",
                "platform_machine": "aarch64",
            },
        },
    }

    duckdb, sealed_extensions = gateway_module._import_admitted_duckdb(
        receipt
    )

    assert duckdb is imported
    try:
        sealed_httpfs, sealed_quack = sealed_extensions.load_paths()
        assert sealed_httpfs.read_bytes() == b"httpfs"
        assert sealed_quack.read_bytes() == b"quack"
        extension.write_bytes(b"mutated-after-import")
        httpfs.write_bytes(b"mutated-after-import")
        connection = _connect_admitted_duckdb(
            _FakeDuckDB(),
            sealed_extensions,
        )
        assert connection.statements[-2:] == [
            f"LOAD '{sealed_httpfs}'",
            f"LOAD '{sealed_quack}'",
        ]
        assert sealed_httpfs.read_bytes() == b"httpfs"
        assert sealed_quack.read_bytes() == b"quack"
    finally:
        sealed_extensions.close()
    extension.write_bytes(b"quack")
    httpfs.write_bytes(b"httpfs")

    substituted_fingerprint = copy.deepcopy(receipt)
    substituted_fingerprint["evidence"][
        "required_quack_extension_fingerprint"
    ] = "sha256:" + "1" * 64
    substituted_fingerprint["evidence"]["quack_probe"][
        "extension_fingerprint"
    ] = "sha256:" + "1" * 64
    with pytest.raises(QuackDaemonGatewayError, match="capability pins"):
        gateway_module._import_admitted_duckdb(substituted_fingerprint)

    substituted = tmp_path / "site-packages/other/duckdb/__init__.py"
    substituted.parent.mkdir(parents=True)
    substituted.write_text("# same version, different module\n", encoding="utf-8")
    monkeypatch.setattr(
        gateway_module.importlib,
        "import_module",
        lambda name: (
            SimpleNamespace(__version__="1.5.5", __file__=str(substituted))
            if name == "duckdb"
            else imported_native
        ),
    )
    with pytest.raises(QuackDaemonGatewayError, match="module is not"):
        gateway_module._import_admitted_duckdb(receipt)

    monkeypatch.setattr(
        gateway_module.importlib,
        "import_module",
        lambda name: imported if name == "duckdb" else imported_native,
    )
    extension.write_bytes(b"mutated-quack")
    with pytest.raises(QuackDaemonGatewayError, match="file digest differs"):
        gateway_module._import_admitted_duckdb(receipt)


def test_connect_rejects_a_replaced_sealed_extension_alias(tmp_path: Path) -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
        eaaef_host_admitted_daemon_gateway as gateway_module,
    )

    quack, httpfs = _pin_extension_pair(tmp_path)
    extensions = gateway_module._seal_admitted_extensions(
        quack_path=quack,
        quack_sha256=_file_digest(quack),
        httpfs_path=httpfs,
        httpfs_sha256=_file_digest(httpfs),
    )
    try:
        extensions.quack_path.unlink()
        os.symlink(str(quack), extensions.quack_path)
        with pytest.raises(QuackDaemonGatewayError, match="alias changed"):
            _connect_admitted_duckdb(_FakeDuckDB(), extensions)
    finally:
        extensions.close()


def test_host_admitted_factories_reject_dead_command_fabric(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
        eaaef_host_admitted_daemon_gateway as gateway_module,
    )

    receipts = {
        task_id: {"decision": "admitted", "evidence": {}}
        for task_id in (
            "EAAEF-182",
            "EAAEF-185",
            "EAAEF-186",
            "EAAEF-187",
            "EAAEF-188",
            "EAAEF-189",
            "EAAEF-191",
        )
    }
    monkeypatch.setattr(
        gateway_module,
        "_eaaef_source_addressed_host_receipts",
        lambda *args, **kwargs: receipts,
    )

    def _unexpected_import(receipt):
        del receipt
        raise AssertionError("dead command fabric must reject before DuckDB import")

    monkeypatch.setattr(
        gateway_module,
        "_import_admitted_duckdb",
        _unexpected_import,
    )
    program = SimpleNamespace(
        authority_mode="quack",
        quack_endpoint="quack:127.0.0.1:19495",
        endpoint_secret_handle="secret-handle:eaaef-quack-owner-v1",
    )

    # The source-only overlay has its own explicit gate, independent of the
    # command-fabric verifier.  Even a future edit of one gate cannot silently
    # make these placeholder authority seams reachable.
    assert gateway_module.build_eaaef_host_admitted_command_gateway(
        repo_root=tmp_path,
        program=program,
        owner_session_id="owner",
        expected_source_head="a" * 40,
        expected_source_tree="b" * 40,
    ) is None
    assert gateway_module.build_eaaef_host_admitted_container_dispatcher_factory(
        repo_root=tmp_path,
        expected_source_head="a" * 40,
        expected_source_tree="b" * 40,
    ) is None
    monkeypatch.setattr(
        gateway_module,
        "_SOURCE_ONLY_SCAFFOLDING_RUNTIME_ENABLED",
        True,
    )
    assert gateway_module.build_eaaef_host_admitted_command_gateway(
        repo_root=tmp_path,
        program=program,
        owner_session_id="owner",
        expected_source_head="a" * 40,
        expected_source_tree="b" * 40,
    ) is None
    assert gateway_module.build_eaaef_host_admitted_container_dispatcher_factory(
        repo_root=tmp_path,
        expected_source_head="a" * 40,
        expected_source_tree="b" * 40,
    ) is None


def test_factory_uses_daemon_execution_repository_property() -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.eaaef_host_admitted_daemon_gateway import (
        _Execution,
    )

    class _Daemon:
        def __init__(self) -> None:
            self.execution_repository = object()

    daemon = _Daemon()
    assert daemon.execution_repository is not None
    # The closed execution component is what reserve/commit consume.
    assert hasattr(_Execution, "reserve_effect")
    assert hasattr(_Execution, "commit_effect")


def test_record_defaults_missing_task_dependencies() -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.eaaef_host_admitted_daemon_gateway import (
        _Record,
    )

    task = _Record({"task_cid": "cid:1", "task_alias": "EAAEF-010", "status": "todo"})
    assert tuple(task.dependencies) == ()
    assert task.body == {}
    assert task.task_cid == "cid:1"
    assert dict(task)["task_cid"] == "cid:1"
    assert task["status"] == "todo"


def test_owner_mutation_rejects_non_cas_sql(tmp_path: Path) -> None:
    inbox = tmp_path / "mutations"
    inbox.mkdir()
    with pytest.raises(QuackDaemonGatewayError, match="signed command fabric"):
        _submit_owner_mutation(
            mutation_dir=inbox,
            sql="DELETE FROM tasks",
            parameters=[],
            timeout_seconds=0.2,
        )


def test_owner_mutation_never_publishes_bare_cas_request(tmp_path: Path) -> None:
    inbox = tmp_path / "mutations"
    inbox.mkdir()
    with pytest.raises(QuackDaemonGatewayError, match="signed command fabric"):
        _submit_owner_mutation(
            mutation_dir=inbox,
            sql=_CAS_TASK_STATUS_SQL,
            parameters=["in_progress", 3, "2026-08-21T00:00:00Z", "cid:1", 2],
            timeout_seconds=2.0,
        )
    assert list(inbox.iterdir()) == []


def test_owned_patch_cid_hashes_only_owned_files(tmp_path: Path) -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.eaaef_host_admitted_daemon_gateway import (
        _OWNED_RELATIVE_PATHS,
        _owned_patch_cid,
    )

    for relative in _OWNED_RELATIVE_PATHS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("eaaef-010\n", encoding="utf-8")
    (tmp_path / "unrelated.txt").write_text("noise\n", encoding="utf-8")
    first = _owned_patch_cid(tmp_path)
    (tmp_path / "unrelated.txt").write_text("changed\n", encoding="utf-8")
    assert _owned_patch_cid(tmp_path) == first
    owned = tmp_path / _OWNED_RELATIVE_PATHS[0]
    owned.write_text("changed-owned\n", encoding="utf-8")
    assert _owned_patch_cid(tmp_path) != first


def test_ensure_attempt_returns_the_attempt_record() -> None:
    from types import SimpleNamespace

    from ipfs_accelerate_py.agent_supervisor.todo_daemon.eaaef_host_admitted_daemon_gateway import (
        _Execution,
    )

    gateway = SimpleNamespace(
        capability=SimpleNamespace(content_id="sha256:" + "a" * 64),
        _attempts={},
    )
    execution = _Execution(gateway)
    stored = execution.ensure_attempt(
        attempt={
            "attempt_id": "attempt:1",
            "claim_id": "claim:1",
            "task_cid": "cid:1",
            "task_alias": "EAAEF-010",
            "attempt_number": 1,
            "owner_session_id": "owner",
            "fencing_token": 1,
            "fence_epoch": 1,
            "lease_id": "lease:1",
            "committed_phase": "claimed",
            "status": "running",
            "started_at_ms": 1,
            "revision": 1,
            "body": {},
        },
        claimed_phase={"phase": "claimed", "revision": 1},
    )
    assert stored["attempt_id"] == "attempt:1"
    assert execution.get_attempt("attempt:1")["attempt_id"] == "attempt:1"
    running = execution.list_running_attempts(owner_session_id="owner")
    assert len(running) == 1


def test_git_worktree_reuses_sibling_with_owned_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.eaaef_host_admitted_daemon_gateway import (
        _OWNED_RELATIVE_PATHS,
        _git_worktree,
    )

    worktrees = (
        tmp_path
        / "data/agent_supervisor/external_agent_autonomous_execution_fabric"
        / "run-v14/worktrees"
    )
    sibling = worktrees / "eaaef-010-aaaaaaaaaaaaaaaa"
    for relative in _OWNED_RELATIVE_PATHS:
        path = sibling / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("owned\n", encoding="utf-8")

    def _forbidden_run(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise AssertionError("git worktree add must not run when owned files exist")

    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.eaaef_host_admitted_daemon_gateway.subprocess.run",
        _forbidden_run,
    )
    found = _git_worktree(
        tmp_path,
        attempt_id="sha256:" + ("b" * 64),
        task_id="EAAEF-010",
        owned=_OWNED_RELATIVE_PATHS,
    )
    assert found == sibling


def test_commit_effect_returns_the_accepted_result() -> None:
    from types import SimpleNamespace

    from ipfs_accelerate_py.agent_supervisor.todo_daemon.eaaef_host_admitted_daemon_gateway import (
        _Execution,
    )

    gateway = SimpleNamespace(
        capability=SimpleNamespace(content_id="sha256:" + "a" * 64),
    )
    accepted = {
        "schema": "accepted",
        "status": "succeeded",
        "accepted": True,
        "task_result_accepted": False,
        "merge_admitted": False,
    }
    committed = _Execution(gateway).commit_effect(
        kind="external_agent_container_dispatch",
        record_id="claim:1",
        claim={"claim_cid": "claim:1"},
        result=accepted,
    )
    assert dict(committed) == accepted


def test_focused_test_receipt_uses_ini_cache_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.eaaef_host_admitted_daemon_gateway import (
        _focused_test_receipt_cid,
    )

    captured: dict[str, Any] = {}

    class _Completed:
        returncode = 0
        stdout = "29 passed in 0.50s\n"
        stderr = ""

    def _fake_run(argv: list[str], **kwargs: Any) -> _Completed:
        captured["argv"] = list(argv)
        captured["env"] = dict(kwargs.get("env") or {})
        return _Completed()

    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.eaaef_host_admitted_daemon_gateway.subprocess.run",
        _fake_run,
    )
    receipt = _focused_test_receipt_cid(tmp_path)
    assert receipt.startswith("sha256:")
    assert "--cache-dir" not in captured["argv"]
    assert captured["argv"].count("-o") >= 1
    assert any(
        str(item).startswith("cache_dir=") for item in captured["argv"]
    )
    assert captured["env"].get("PYTEST_ADDOPTS") == ""


def test_host_merge_admission_is_reviewed_patch_when_host_lacks_files(
    tmp_path: Path,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.eaaef_host_admitted_daemon_gateway import (
        _OWNED_RELATIVE_PATHS,
        _REVIEWER_DID,
        _cid,
        _host_merge_admission,
        _owned_patch_cid,
    )
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.external_agent_container_dispatcher import (
        ExternalAgentContainerWorkerDispatcher,
        ExternalAgentContainerWorkPacket,
    )

    worktrees = (
        tmp_path
        / "data/agent_supervisor/external_agent_autonomous_execution_fabric"
        / "run-v14/worktrees"
    )
    sibling = worktrees / "eaaef-010-aaaaaaaaaaaaaaaa"
    for relative in _OWNED_RELATIVE_PATHS:
        path = sibling / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("owned\n", encoding="utf-8")
    packet = ExternalAgentContainerWorkPacket(
        task_id="EAAEF-010",
        task_cid="sha256:" + ("1" * 64),
        attempt_id="attempt:eaaef:1",
        attempt_number=1,
        plan_revision_cid="sha256:" + ("2" * 64),
        repository_tree="3" * 40,
        semantic_state_root="sha256:" + ("4" * 64),
        worktree_id="sha256:" + ("5" * 64),
        planned_container_id="sha256:" + ("6" * 64),
        worker_principal_did="did:key:zworker",
        provider_principal_did="did:key:zprovider",
        provider="grok",
        model_route_cid="sha256:" + ("7" * 64),
        container_profile_cid="sha256:" + ("8" * 64),
        image_digest="sha256:" + ("9" * 64),
        network_authorization_cid="sha256:" + ("a" * 64),
        lease_id="lease:eaaef:1",
        fencing_token=1,
        fence_epoch=1,
        idempotency_key="eaaef:dispatch:1",
        effect_scope_cid="sha256:" + ("b" * 64),
        gateway_binding_cid="sha256:" + ("c" * 64),
    )
    claim = ExternalAgentContainerWorkerDispatcher._dispatch_claim(packet)
    patch = _owned_patch_cid(sibling)
    admission = _host_merge_admission(
        packet=packet,
        effect={
            "claim_cid": claim["claim_cid"],
            "accepted_result_receipt_id": _cid({"accepted": True}),
            "patch_artifact_cid": patch,
        },
        repo_root=tmp_path,
        owned=_OWNED_RELATIVE_PATHS,
    )
    assert admission is not None
    assert admission["decision"] == "accepted"
    assert admission["delivery_mode"] == "reviewed_patch"
    assert admission["merge_commit"] == ""
    assert admission["reviewer_principal_did"] == _REVIEWER_DID
    assert admission["patch_artifact_cid"] == patch


def test_ready_tasks_skip_unmet_dependencies() -> None:
    import json
    from types import SimpleNamespace

    from ipfs_accelerate_py.agent_supervisor.todo_daemon.eaaef_host_admitted_daemon_gateway import (
        _TaskSource,
    )

    done = "sha256:" + ("1" * 64)
    peer = "sha256:" + ("2" * 64)
    ready_cid = "sha256:" + ("3" * 64)
    waiting = "sha256:" + ("4" * 64)
    catalog = [
        {
            "task_cid": done,
            "task_alias": "EAAEF-010",
            "status": "completed",
            "ordinal": 1,
        },
        {
            "task_cid": peer,
            "task_alias": "EAAEF-012",
            "status": "todo",
            "ordinal": 2,
        },
        {
            "task_cid": ready_cid,
            "task_alias": "EAAEF-011",
            "status": "todo",
            "ordinal": 3,
        },
        {
            "task_cid": waiting,
            "task_alias": "EAAEF-015",
            "status": "todo",
            "ordinal": 4,
        },
    ]
    bodies = {
        done: {
            "task_cid": done,
            "task_alias": "EAAEF-010",
            "status": "completed",
            "body_json": "{}",
        },
        peer: {
            "task_cid": peer,
            "task_alias": "EAAEF-012",
            "status": "todo",
            "body_json": json.dumps({"dependency_task_cids": [done]}),
        },
        ready_cid: {
            "task_cid": ready_cid,
            "task_alias": "EAAEF-011",
            "status": "todo",
            "body_json": json.dumps({"dependency_task_cids": [done]}),
        },
        waiting: {
            "task_cid": waiting,
            "task_alias": "EAAEF-015",
            "status": "todo",
            "body_json": json.dumps({"dependency_task_cids": [peer]}),
        },
    }

    class _Client:
        def paginate(self, name: str, cursor: int = 0, limit: int = 50) -> Any:
            del name, cursor, limit
            return SimpleNamespace(items=catalog, exhausted=True, next_cursor=None)

        def execute(self, name: str, params: dict[str, Any]) -> list[dict[str, Any]]:
            del name
            return [bodies[str(params["task_cid"])]]

    gateway = SimpleNamespace(
        capability=SimpleNamespace(content_id="sha256:" + ("a" * 64)),
        _client=_Client(),
    )
    page = _TaskSource(gateway).ready_tasks(limit=8)
    assert [task.task_alias for task in page.tasks] == ["EAAEF-011", "EAAEF-012"]


def test_configured_board_never_promotes_raw_live_quack_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.runtime import (
        configured_board_scheduler as scheduler,
    )

    class _Board:
        board_namespace = scheduler.EAAEF_BOARD_NAMESPACE
        payload = {"schema": scheduler.EAAEF_SCHEDULER_SCHEMA}
        database_program = object()
        repo_root = Path("/tmp")

        def path(self, relative: str) -> Path:
            return Path("/tmp") / relative

    monkeypatch.setattr(
        scheduler,
        "_eaaef_live_quack_status_overlay",
        lambda board: {"EAAEF-011": "todo", "EAAEF-010": "completed"},
    )
    overlay = scheduler._eaaef_task_status_overlay(_Board())
    assert overlay == {}
