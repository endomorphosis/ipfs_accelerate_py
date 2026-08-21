from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts/materialize_agent_supervisor_procedure_compiler_program.py"


def _load_materializer() -> ModuleType:
    spec = importlib.util.spec_from_file_location("pcpc_materializer_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _scheduler_config() -> dict[str, object]:
    return json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_proof_carrying_procedure_compiler_scheduler.json"
        ).read_text(encoding="utf-8")
    )


def _create_duckdb(path: Path) -> None:
    import duckdb

    path.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(path))
    try:
        connection.execute("CREATE TABLE authority_marker(value INTEGER)")
        connection.execute("INSERT INTO authority_marker VALUES (1)")
    finally:
        connection.close()


def _hermetic_trusted_home(
    module: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[ModuleType, object, Path]:
    launcher = module._load_program_launcher_module()
    base = launcher.load_program_config(REPO_ROOT)
    extension_directory = tmp_path / "source-extensions"
    extension_directory.mkdir()
    hashes: dict[str, str] = {}
    for index, name in enumerate(base.qualification_extension_hashes):
        content = f"materializer-extension-{index}".encode()
        (extension_directory / name).write_bytes(content)
        hashes[name] = hashlib.sha256(content).hexdigest()
    config = replace(
        base,
        extension_directory=extension_directory,
        extension_hashes={name: hashes[name] for name in base.extension_hashes},
        projection_extension_hashes={
            name: hashes[name] for name in base.projection_extension_hashes
        },
        state_root=tmp_path / "state",
    )
    config.state_root.mkdir(mode=0o700)
    home = launcher._build_qualification_home(config)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv(module.TRUSTED_DUCKDB_HOME_ENV, str(home))
    return launcher, config, home


def _qualification(module: ModuleType) -> dict[str, object]:
    commands = []
    for argv in module.QUALIFICATION_COMMANDS:
        commands.append(
            {
                "argv": list(argv),
                "returncode": 0,
                "elapsed_ms": 1,
                "stdout_bytes": 0,
                "stderr_bytes": 0,
                "stdout_sha256": "0" * 64,
                "stderr_sha256": "0" * 64,
                "stdout_tail": "",
                "stderr_tail": "",
            }
        )
    payload: dict[str, object] = {
        "schema": ("ipfs_accelerate_py/agent-supervisor/procedure-compiler-p0-qualification@2"),
        "program": module.PROGRAM,
        "repository_commit": "commit-current",
        "repository_tree": "tree-current",
        "branch": module.BRANCH,
        "commands": commands,
        "p0_tasks": list(module.P0_TASKS),
        "test_evidence_class": "current_tree_hermetic",
        "simulated": False,
    }
    payload["qualification_cid"] = module.content_identity(payload)
    return payload


def test_qualification_recomputes_identity_and_rejects_receipt_shaped_edit() -> None:
    module = _load_materializer()
    receipt = _qualification(module)
    assert module._stored_qualification_receipt_is_intact(
        receipt, head="commit-current", tree="tree-current"
    )

    receipt["test_evidence_class"] = "current_tree_hermetic_but_forged"
    assert not module._stored_qualification_receipt_is_intact(
        receipt, head="commit-current", tree="tree-current"
    )


def test_materialization_receipt_identity_must_be_recomputed() -> None:
    module = _load_materializer()
    receipt = {"program": module.PROGRAM, "ready_task_ids": ["PCPC-009"]}
    receipt["receipt_cid"] = module.content_identity(receipt)
    assert module._has_valid_embedded_identity(receipt, identity_field="receipt_cid")

    receipt["ready_task_ids"] = []
    assert not module._has_valid_embedded_identity(receipt, identity_field="receipt_cid")


def test_direct_materializer_rejects_ambient_home_before_qualification_or_database_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_materializer()
    ambient = tmp_path / "ambient-home"
    ambient.mkdir()
    qualification_called = False

    def qualification_must_not_run() -> dict[str, object]:
        nonlocal qualification_called
        qualification_called = True
        raise AssertionError("qualification ran before trusted HOME admission")

    monkeypatch.setenv("HOME", str(ambient))
    monkeypatch.delenv(module.TRUSTED_DUCKDB_HOME_ENV, raising=False)
    monkeypatch.setattr(module, "_qualify_exact_tree", qualification_must_not_run)

    with pytest.raises(module.MaterializationError, match="trusted DuckDB HOME"):
        module.materialize()

    assert qualification_called is False
    assert not (tmp_path / "state").exists()


def test_direct_materializer_rejects_sealed_home_extension_digest_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_materializer()
    launcher, config, home = _hermetic_trusted_home(module, tmp_path, monkeypatch)
    target = (
        home
        / ".duckdb/extensions/v1.5.5/linux_arm64/ducklake.duckdb_extension"
    )
    target.chmod(0o600)
    target.write_bytes(b"digest drift")
    target.chmod(0o400)

    with pytest.raises(module.MaterializationError, match="validation failed"):
        module._require_trusted_duckdb_home(
            launcher_module=launcher, program_config=config
        )


def test_materializer_disables_extension_install_and_autoload() -> None:
    module = _load_materializer()
    calls: list[tuple[str, object]] = []

    class FakeResult:
        @staticmethod
        def fetchall() -> list[tuple[str, str]]:
            return [
                ("allow_unsigned_extensions", "false"),
                ("autoinstall_known_extensions", "false"),
                ("autoload_known_extensions", "false"),
                ("temp_directory", ""),
            ]

    class FakeConnection:
        closed = False

        def execute(self, sql: str, parameters: object = None) -> FakeResult:
            calls.append((sql, parameters))
            return FakeResult()

        def close(self) -> None:
            self.closed = True

    class FakeDuckDB:
        config: dict[str, bool] | None = None

        @classmethod
        def connect(cls, database: str, *, config: dict[str, bool]) -> FakeConnection:
            assert database == ":memory:"
            cls.config = config
            return FakeConnection()

    connection = module._sealed_duckdb_connection(FakeDuckDB)

    assert connection.closed is False
    assert FakeDuckDB.config == {
        "allow_unsigned_extensions": False,
        "autoinstall_known_extensions": False,
        "autoload_known_extensions": False,
        "temp_directory": "",
    }
    assert len(calls) == 1
    assert "duckdb_settings()" in calls[0][0]
    assert "INSTALL" not in SCRIPT.read_text(encoding="utf-8")


def test_external_access_is_allowlisted_then_locked_before_projection(
    tmp_path: Path,
) -> None:
    catalog = tmp_path / "history" / "catalog.ducklake"
    data = tmp_path / "history" / "data"
    code = f"""
import importlib.util
import sys
from pathlib import Path
import duckdb
spec = importlib.util.spec_from_file_location("pcpc_external_lock_child", {str(SCRIPT)!r})
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
catalog = Path({str(catalog)!r})
data = Path({str(data)!r})
catalog.parent.mkdir()
data.mkdir()
connection = module._sealed_duckdb_connection(duckdb)
module._seal_external_access(
    connection,
    allowed_paths=(catalog, Path(f"{{catalog}}.wal")),
    allowed_directories=(data,),
)
assert connection.execute(
    "SELECT current_setting('enable_external_access'), "
    "current_setting('lock_configuration'), current_setting('temp_directory')"
).fetchone() == (False, True, "")
for statement in (
    "SELECT * FROM read_csv_auto('/etc/passwd')",
    "SELECT * FROM read_csv_auto('https://example.com/data.csv')",
):
    try:
        connection.execute(statement)
    except Exception as exc:
        assert "file system operations are disabled" in str(exc)
    else:
        raise AssertionError("external path unexpectedly admitted")
try:
    connection.execute("SET enable_external_access = true")
except Exception as exc:
    assert "configuration has been locked" in str(exc)
else:
    raise AssertionError("locked configuration changed")
connection.close()
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_ducklake_projection_filesystem_outage_is_typed_and_non_authoritative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_materializer()
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    history = (
        tmp_path
        / "state/agent_supervisor_proof_carrying_procedure_compiler/history"
    )
    history.parent.mkdir(parents=True)
    history.write_text("projection root unavailable\n", encoding="utf-8")

    result = module._project_ducklake(
        config=_scheduler_config(),
        run={},
        qualification={},
        tasks=(),
    )

    assert result["projected"] is False
    assert result["authority"] is False
    assert "FileExistsError" in result["reason"]


def test_unsafe_projection_config_is_rejected_before_qualification_or_control_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_materializer()
    config = _scheduler_config()
    projection = config["ducklake_projection_program"]
    assert isinstance(projection, dict)
    projection["catalog_path"] = "docs/redteam.ducklake"
    qualification_called = False

    def qualification_must_not_run() -> dict[str, object]:
        nonlocal qualification_called
        qualification_called = True
        raise AssertionError("unsafe projection reached qualification")

    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(module, "_require_trusted_duckdb_home", lambda: tmp_path)
    monkeypatch.setattr(module, "_read_json", lambda _path: config)
    monkeypatch.setattr(module, "_qualify_exact_tree", qualification_must_not_run)

    with pytest.raises(module.MaterializationError, match="configuration is unsafe"):
        module.materialize()

    assert qualification_called is False
    assert not (
        tmp_path / module.CONTROL_DATABASE_RELATIVE
    ).exists()


def test_existing_materialization_cannot_bypass_fresh_qualification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_materializer()

    def reject_stale_or_fabricated_evidence() -> dict[str, object]:
        raise module.MaterializationError("fresh qualification required")

    monkeypatch.setattr(module, "_qualify_exact_tree", reject_stale_or_fabricated_evidence)
    monkeypatch.setattr(module, "_require_trusted_duckdb_home", lambda: Path("/trusted"))
    with pytest.raises(module.MaterializationError, match="fresh qualification required"):
        module.verify_existing()


def test_existing_materialization_reopens_with_recomputed_current_plan_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_materializer()
    database_path = tmp_path / module.CONTROL_DATABASE_RELATIVE
    _create_duckdb(database_path)
    config = _scheduler_config()
    config["runtime_paths"]["evidence"] = "evidence"
    expected_plan_cid = "baguqeera" + "a" * 48
    observed: dict[str, object] = {}

    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(module, "_require_trusted_duckdb_home", lambda: Path("/trusted"))
    monkeypatch.setattr(
        module,
        "_qualify_exact_tree",
        lambda: {
            "qualification_cid": "baguqeera" + "b" * 48,
            "repository_commit": "commit-current",
            "repository_tree": "tree-current",
            "simulated": False,
        },
    )
    monkeypatch.setattr(
        module,
        "_git",
        lambda *args: (
            "commit-current" if args[-1] == "HEAD" else "tree-current"
        ),
    )
    monkeypatch.setattr(
        module,
        "_read_json",
        lambda path: (
            config
            if Path(path).name
            == Path(module.CONFIG_RELATIVE).name
            else {}
        ),
    )
    monkeypatch.setattr(
        module,
        "_population",
        lambda **kwargs: ({}, expected_plan_cid),
    )

    class CapturedPlanRoot(RuntimeError):
        pass

    def capture_source(*args: object, **kwargs: object) -> object:
        observed.update(kwargs)
        raise CapturedPlanRoot

    monkeypatch.setattr(module, "DatabaseTaskSource", capture_source)

    with pytest.raises(CapturedPlanRoot):
        module.verify_existing()
    assert observed["repository_tree_id"] == "tree-current"
    assert observed["plan_root_cid"] == expected_plan_cid
    assert observed


@pytest.mark.parametrize("replay_failure", (False, True))
def test_verify_replays_only_on_private_copy_and_preserves_authority_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replay_failure: bool,
) -> None:
    module = _load_materializer()
    database_path = tmp_path / module.CONTROL_DATABASE_RELATIVE
    _create_duckdb(database_path)
    config = _scheduler_config()
    config["runtime_paths"]["evidence"] = "evidence"
    head = "commit-current"
    tree = "tree-current"
    expected_plan_cid = "baguqeera" + "c" * 48
    opened_paths: list[Path] = []

    fresh_qualification = {
        "qualification_cid": "baguqeera" + "d" * 48,
        "repository_commit": head,
        "repository_tree": tree,
        "simulated": False,
    }
    stored_qualification = {"qualification_cid": "baguqeera" + "e" * 48}
    materialization_receipt = {
        "schema": "ipfs_accelerate_py/agent-supervisor/procedure-compiler-materialization@1",
        "program": module.PROGRAM,
        "repository_commit": head,
        "repository_tree": tree,
        "plan_root_cid": expected_plan_cid,
        "qualification_cid": stored_qualification["qualification_cid"],
        "simulated": False,
    }

    records = []
    for index in range(32):
        status = "completed" if index < 9 else "ready"
        records.append(
            SimpleNamespace(
                task_alias=f"PCPC-{index:03d}",
                to_dict=lambda index=index, status=status: {
                    "task_cid": f"task-{index}",
                    "task_alias": f"PCPC-{index:03d}",
                    "goal_cid": "goal",
                    "status": status,
                    "revision": 1,
                    "body": {
                        "repository_commit": head,
                        "repository_tree": tree,
                    },
                },
            )
        )

    class FakeSource:
        def __init__(self, path: Path, **kwargs: object) -> None:
            del kwargs
            opened = Path(path)
            assert opened != database_path
            assert opened.parent != database_path.parent
            assert opened.stat().st_mode & 0o777 == 0o600
            opened_paths.append(opened)

        def __enter__(self) -> FakeSource:
            return self

        def __exit__(self, *exc: object) -> None:
            del exc

        def list_tasks(self, *, limit: int) -> SimpleNamespace:
            assert limit == 64
            return SimpleNamespace(tasks=tuple(records))

        def ready_tasks(self, *, limit: int) -> SimpleNamespace:
            assert limit == 64
            return SimpleNamespace(tasks=tuple(records[index] for index in (9, 11, 13)))

        @staticmethod
        def snapshot() -> SimpleNamespace:
            return SimpleNamespace(
                plan_root_cid=expected_plan_cid,
                projection_cid="projection-current",
            )

        @staticmethod
        def get_plan(plan_cid: str) -> dict[str, object]:
            assert plan_cid == expected_plan_cid
            return {"body": {"repository_commit": head, "repository_tree": tree}}

        @staticmethod
        def projection_matches_events() -> bool:
            if replay_failure:
                raise RuntimeError("private replay failed")
            return True

    def read_fixture(path: Path) -> dict[str, object]:
        if Path(path).name == Path(module.CONFIG_RELATIVE).name:
            return config
        if Path(path).name == "materialization.json":
            return materialization_receipt
        return stored_qualification

    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(module, "_require_trusted_duckdb_home", lambda: Path("/trusted"))
    monkeypatch.setattr(module, "_qualify_exact_tree", lambda: fresh_qualification)
    monkeypatch.setattr(
        module,
        "_git",
        lambda *args: head if args[-1] == "HEAD" else tree,
    )
    monkeypatch.setattr(module, "_read_json", read_fixture)
    monkeypatch.setattr(module, "_population", lambda **kwargs: ({}, expected_plan_cid))
    monkeypatch.setattr(module, "_has_valid_embedded_identity", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        module, "_stored_qualification_receipt_is_intact", lambda *args, **kwargs: True
    )
    monkeypatch.setattr(module, "DatabaseTaskSource", FakeSource)
    before = hashlib.sha256(database_path.read_bytes()).hexdigest()

    if replay_failure:
        with pytest.raises(RuntimeError, match="private replay failed"):
            module.verify_existing()
    else:
        assert module.verify_existing()["valid"] is True

    assert opened_paths
    assert hashlib.sha256(database_path.read_bytes()).hexdigest() == before


@pytest.mark.parametrize("dangling_symlink", (False, True))
def test_private_replay_copy_rejects_wal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dangling_symlink: bool,
) -> None:
    module = _load_materializer()
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    database_path = tmp_path / module.CONTROL_DATABASE_RELATIVE
    _create_duckdb(database_path)
    wal_path = Path(f"{database_path}.wal")
    if dangling_symlink:
        wal_path.symlink_to(tmp_path / "absent-wal-target")
    else:
        wal_path.write_bytes(b"pending transaction")
    before = hashlib.sha256(database_path.read_bytes()).hexdigest()

    with pytest.raises(module.MaterializationError, match="offline verification"):
        with module._disposable_control_database_copy(database_path):
            raise AssertionError("live authority was admitted")

    assert hashlib.sha256(database_path.read_bytes()).hexdigest() == before


def test_private_replay_copy_rejects_live_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import duckdb

    module = _load_materializer()
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    database_path = tmp_path / module.CONTROL_DATABASE_RELATIVE
    _create_duckdb(database_path)
    before = hashlib.sha256(database_path.read_bytes()).hexdigest()
    writer = duckdb.connect(str(database_path))
    try:
        with pytest.raises(module.MaterializationError, match="read-only verification"):
            with module._disposable_control_database_copy(database_path):
                raise AssertionError("live writer was admitted")
    finally:
        writer.close()

    assert hashlib.sha256(database_path.read_bytes()).hexdigest() == before


def test_private_replay_copy_rejects_live_status_but_ignores_dead_archived_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
        current_process_birth,
    )

    module = _load_materializer()
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    database_path = tmp_path / module.CONTROL_DATABASE_RELATIVE
    _create_duckdb(database_path)
    status_path = (
        tmp_path
        / module.QUACK_OWNER_STATE_RELATIVE
        / "quack-state-server.status.json"
    )
    status_path.parent.mkdir()
    birth = current_process_birth().to_dict()
    status_path.write_text(
        json.dumps(
            {
                "lifecycle": "ready",
                "identity": {"status": "ready", "process_birth": birth},
            }
        ),
        encoding="utf-8",
    )
    before = hashlib.sha256(database_path.read_bytes()).hexdigest()

    with pytest.raises(module.MaterializationError, match="Quack owner status"):
        with module._disposable_control_database_copy(database_path):
            raise AssertionError("live status was admitted")

    dead_birth = dict(birth)
    dead_birth["pid"] = 2_000_000_000
    status_path.write_text(
        json.dumps(
            {
                "lifecycle": "ready",
                "identity": {"status": "ready", "process_birth": dead_birth},
            }
        ),
        encoding="utf-8",
    )
    with module._disposable_control_database_copy(database_path) as copied:
        assert copied.is_file()

    assert hashlib.sha256(database_path.read_bytes()).hexdigest() == before


@pytest.mark.parametrize("alias_kind", ("symlink", "hardlink"))
def test_private_replay_copy_rejects_database_aliases_without_mutating_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, alias_kind: str
) -> None:
    module = _load_materializer()
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    target = tmp_path / "outside.duckdb"
    _create_duckdb(target)
    database_path = tmp_path / module.CONTROL_DATABASE_RELATIVE
    database_path.parent.mkdir(parents=True)
    if alias_kind == "symlink":
        database_path.symlink_to(target)
    else:
        database_path.hardlink_to(target)
    before = hashlib.sha256(target.read_bytes()).hexdigest()

    with pytest.raises(module.MaterializationError):
        with module._disposable_control_database_copy(database_path):
            raise AssertionError("aliased authority was admitted")

    assert hashlib.sha256(target.read_bytes()).hexdigest() == before


def test_projection_parity_replay_uses_copy_and_detects_real_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import duckdb

    module = _load_materializer()
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    database_path = tmp_path / module.CONTROL_DATABASE_RELATIVE
    population = {
        "objectives": [
            {
                "goal_cid": "goal:root",
                "goal_alias": "G-ROOT",
                "title": "Root",
            }
        ],
        "plans": [
            {
                "plan_cid": "plan:root",
                "goal_cid": "goal:root",
                "plan_alias": "PLAN",
            }
        ],
        "taskboard": [
            {
                "task_cid": "task:one",
                "task_id": "T-ONE",
                "goal_cid": "goal:root",
                "plan_cid": "plan:root",
                "status": "ready",
            }
        ],
    }
    with module.DatabaseTaskSource(
        database_path,
        repository_tree_id="tree:one",
        plan_root_cid="plan:root",
    ) as source:
        source.materialize(
            population,
            repository_tree_id="tree:one",
            plan_root_cid="plan:root",
        )
    connection = duckdb.connect(str(database_path))
    try:
        connection.execute("UPDATE tasks SET status = 'failed' WHERE task_cid = 'task:one'")
    finally:
        connection.close()
    before = hashlib.sha256(database_path.read_bytes()).hexdigest()

    assert module._projection_matches_events_on_disposable_copy(
        database_path,
        repository_tree_id="tree:one",
        plan_root_cid="plan:root",
    ) is False
    assert hashlib.sha256(database_path.read_bytes()).hexdigest() == before


def _command(argv: list[str], *, returncode: int, output: str) -> dict[str, object]:
    return {
        "argv": argv,
        "returncode": returncode,
        "elapsed_ms": 1,
        "stdout_bytes": len(output.encode("utf-8")),
        "stderr_bytes": 0,
        "stdout_sha256": "1" * 64,
        "stderr_sha256": "0" * 64,
        "stdout_tail": output,
        "stderr_tail": "",
    }


def _producer_fixture(module: ModuleType) -> tuple[dict[str, object], dict[str, object]]:
    failing = {
        "reason_code": "typed_known_failure",
        "required_output_fragments": ["test_known_failure", "KeyError: 'known'"],
        "signature": "known deterministic failure",
    }
    baseline: dict[str, object] = {
        "sibling_release_bindings": [],
        "test_producers": [
            {
                "producer_id": "TP-PASS",
                "command": ["python", "-m", "pytest", "-q", "pass.py"],
                "expected": {
                    "collected": 2,
                    "passed": 2,
                    "failed": 0,
                    "errors": 0,
                    "returncode": 0,
                },
                "source_bindings": [
                    {
                        "path": "pass.py",
                        "blob_id": "1" * 40,
                        "current_blob_id": "3" * 40,
                    }
                ],
            },
            {
                "producer_id": "TP-FAIL",
                "command": ["python", "-m", "pytest", "-q", "fail.py"],
                "expected": {
                    "collected": 2,
                    "passed": 1,
                    "failed": 1,
                    "errors": 0,
                    "returncode": 1,
                },
                "expected_failure": failing,
                "source_bindings": [{"path": "fail.py", "blob_id": "2" * 40}],
            },
        ],
    }
    inventory: dict[str, object] = {
        "dispositions": [
            {
                "authority": "AvailableAuthority",
                "status": "available_with_caveats",
                "test_producer_bindings": ["TP-PASS", "TP-FAIL"],
            },
            {
                "authority": "MissingAuthority",
                "status": "missing",
                "test_producer_bindings": [],
            },
        ]
    }
    return baseline, inventory


def _gitlink_fixture(module: ModuleType) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/procedure-compiler-exact-gitlink-checkouts@1"
        ),
        "program": module.PROGRAM,
        "repository_commit": "commit-current",
        "repository_tree": "tree-current",
        "bindings": [],
        "binding_count": 0,
        "auto_updated": False,
        "simulated": False,
    }
    payload["gitlink_receipt_cid"] = module.content_identity(payload)
    return payload


def test_current_prerequisite_execution_binds_exact_counts_and_authorities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_materializer()
    baseline, inventory = _producer_fixture(module)

    def execute(argv: list[str], *, timeout: int) -> tuple[dict[str, object], str]:
        assert timeout == module.PREREQUISITE_PRODUCER_TIMEOUT_SECONDS
        if argv[-1] == "pass.py":
            output, returncode = "..\n2 passed in 0.01s\n", 0
        else:
            output, returncode = (
                "FAILED fail.py::test_known_failure - KeyError: 'known'\n"
                "1 failed, 1 passed in 0.02s\n",
                1,
            )
        return _command(argv, returncode=returncode, output=output), output

    monkeypatch.setattr(module, "_captured_command_receipt", execute)
    execution = module._execute_prerequisite_test_producers(
        baseline=baseline,
        inventory=inventory,
        head="commit-current",
        tree="tree-current",
        gitlinks=_gitlink_fixture(module),
    )
    assert execution["producer_count"] == 2
    assert execution["typed_expected_failure_count"] == 1
    assert execution["authority_count"] == 2
    assert execution["all_declared_outcomes_matched"] is True
    by_producer = {item["producer_id"]: item for item in execution["producer_receipts"]}
    assert by_producer["TP-PASS"]["source_blob_ids"] == ["3" * 40]
    by_authority = {item["authority"]: item for item in execution["authority_receipts"]}
    assert by_authority["AvailableAuthority"]["producer_ids"] == ["TP-FAIL", "TP-PASS"]
    assert (
        by_authority["MissingAuthority"]["evidence_disposition"]
        == "not_applicable_missing_authority"
    )

    def read_fixture(path: Path) -> dict[str, object]:
        return baseline if path.name == "baseline.json" else inventory

    monkeypatch.setattr(module, "_read_json", read_fixture)
    assert module._stored_prerequisite_execution_is_intact(
        execution, head="commit-current", tree="tree-current"
    )

    forged = copy.deepcopy(execution)
    producer = forged["producer_receipts"][0]
    old_cid = producer.pop("producer_receipt_cid")
    producer["unknown_normative_field"] = "not admitted"
    producer["producer_receipt_cid"] = module.content_identity(producer)
    for authority in forged["authority_receipts"]:
        if old_cid in authority["producer_receipt_cids"]:
            authority["producer_receipt_cids"] = [
                producer["producer_receipt_cid"] if item == old_cid else item
                for item in authority["producer_receipt_cids"]
            ]
            authority.pop("authority_receipt_cid")
            authority["authority_receipt_cid"] = module.content_identity(authority)
    forged.pop("execution_cid")
    forged["execution_cid"] = module.content_identity(forged)
    assert not module._stored_prerequisite_execution_is_intact(
        forged, head="commit-current", tree="tree-current"
    )


def test_current_prerequisite_execution_rejects_count_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_materializer()
    baseline, inventory = _producer_fixture(module)

    def drifted(argv: list[str], *, timeout: int) -> tuple[dict[str, object], str]:
        del timeout
        output = "3 passed in 0.01s\n"
        return _command(argv, returncode=0, output=output), output

    monkeypatch.setattr(module, "_captured_command_receipt", drifted)
    with pytest.raises(module.MaterializationError, match="producer TP-PASS drifted"):
        module._execute_prerequisite_test_producers(
            baseline=baseline,
            inventory=inventory,
            head="commit-current",
            tree="tree-current",
            gitlinks=_gitlink_fixture(module),
        )


def test_current_prerequisite_execution_rejects_untyped_same_count_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_materializer()
    baseline, inventory = _producer_fixture(module)
    baseline = copy.deepcopy(baseline)
    baseline["test_producers"] = [baseline["test_producers"][1]]
    inventory = {
        "dispositions": [
            {
                "authority": "AvailableAuthority",
                "status": "available_with_caveats",
                "test_producer_bindings": ["TP-FAIL"],
            }
        ]
    }

    def wrong_failure(argv: list[str], *, timeout: int) -> tuple[dict[str, object], str]:
        del timeout
        output = "FAILED fail.py::test_other - RuntimeError: other\n1 failed, 1 passed in 0.01s\n"
        return _command(argv, returncode=1, output=output), output

    monkeypatch.setattr(module, "_captured_command_receipt", wrong_failure)
    with pytest.raises(module.MaterializationError, match="typed expected failure fragments"):
        module._execute_prerequisite_test_producers(
            baseline=baseline,
            inventory=inventory,
            head="commit-current",
            tree="tree-current",
            gitlinks=_gitlink_fixture(module),
        )


def test_exact_gitlink_checkout_is_a_read_only_qualification_precondition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_materializer()
    commit = "a" * 40
    baseline = {
        "sibling_release_bindings": [
            {"binding_id": "gitlink:sibling", "path": "sibling", "gitlink_commit": commit}
        ]
    }
    monkeypatch.setattr(module, "_object_id", lambda revision, path: commit)
    monkeypatch.setattr(
        module,
        "_git",
        lambda *args: f"-{commit} sibling" if args[:2] == ("submodule", "status") else "",
    )
    with pytest.raises(module.MaterializationError, match="exact sibling checkout required"):
        module._verify_exact_gitlink_checkouts(baseline, head="commit-current", tree="tree-current")
