from __future__ import annotations

import ast
import fcntl
import hashlib
import importlib.util
import json
import os
import select
import shlex
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
    OwnerLiveness,
    ProcessBirthIdentity,
    owner_liveness,
    read_process_birth,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
    MigrationDriftError,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
    install_control_plane_schema,
    load_datasets_authoritative_operational_catalog,
    verify_datasets_authoritative_operational_schema,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import (
    open_intent_repository,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_capabilities import (
    probe_quack_capabilities,
)

ROOT = Path(__file__).resolve().parents[2]
OPERATOR_PATH = ROOT / (
    "scripts/run_logic_governed_compositional_verification_fabric_quack.py"
)


def _operator() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "lgcvf_quack_successor", OPERATOR_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _lgcvf_test_execution_route_policy(operator: ModuleType) -> object:
    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        TaskRecord,
        TaskSourceSnapshot,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.task_execution_route_policy import (
        GROK_CODEX_EXECUTION_MODE,
        TaskExecutionRoutePolicy,
    )

    plan_root_cid = "plan:test-lgcvf-bootstrap"
    tasks = tuple(
        TaskRecord(
            task_cid=f"task:test-lgcvf-bootstrap:{index:02d}",
            task_alias=alias,
            goal_cid="goal:test-lgcvf-bootstrap",
            plan_cid=plan_root_cid,
            ordinal=index + 1,
            status="ready",
            revision=1,
        )
        for index, alias in enumerate(operator.LGCVF_TASK_ALIASES)
    )
    snapshot = TaskSourceSnapshot(
        source_schema="test-lgcvf-task-source@1",
        schema_version=1,
        plan_root_cid=plan_root_cid,
        repository_tree_id="tree:test-lgcvf-bootstrap",
        projection_cid="projection:test-lgcvf-bootstrap",
        formal_plan_id=plan_root_cid,
        source_identity="source:test-lgcvf-bootstrap",
        revision=1,
        event_cursor=0,
        goal_count=1,
        task_count=len(tasks),
        dependency_count=0,
        terminal=False,
        objective_count=1,
        plan_count=1,
    )
    return TaskExecutionRoutePolicy.seal(
        snapshot=snapshot,
        tasks=tasks,
        execution_modes={
            task.task_alias: GROK_CODEX_EXECUTION_MODE for task in tasks
        },
    )


def _seed_datasets_profile(database: Path) -> None:
    operator = _operator()
    operator.datasets_profile_migration(database)
    repository = open_intent_repository(
        database,
        owner_id="lgcvf-quack-successor-test:seed",
        install_schema=False,
    )
    try:
        repository.upsert_objective(
            objective_id="objective:test",
            objective_alias="O",
            title="Synthetic objective",
        )
        repository.upsert_goal(
            goal_cid="goal:test",
            goal_alias="G",
            title="Synthetic goal",
            objective_id="objective:test",
        )
        repository.upsert_plan(
            plan_cid="plan:test",
            plan_alias="P",
            goal_cid="goal:test",
        )
        repository.upsert_task(
            task_cid="task:test",
            task_alias="LGCVF-TEST",
            goal_cid="goal:test",
            plan_cid="plan:test",
            objective_id="objective:test",
            ordinal=1,
            status="ready",
        )
    finally:
        repository.close()


def test_datasets_profile_callback_rejects_default_catalog_drift(
    tmp_path: Path,
) -> None:
    operator = _operator()
    database = tmp_path / "control.duckdb"

    report = operator.datasets_profile_migration(database)
    verification = verify_datasets_authoritative_operational_schema(database)
    expected_catalog = load_datasets_authoritative_operational_catalog().fingerprint()

    assert verification["valid"] is True
    assert report.schema_fingerprint == verification["schema_fingerprint"]
    assert report.catalog_fingerprint == expected_catalog
    assert verification["catalog_fingerprint"] == expected_catalog
    with pytest.raises(MigrationDriftError):
        install_control_plane_schema(database)


def test_native_resume_materialization_has_exact_four_task_frontier(
    tmp_path: Path,
) -> None:
    operator = _operator()
    materializer = importlib.import_module(
        "scripts."
        "materialize_logic_governed_compositional_verification_fabric_control_plane"
    )
    config, _raw = operator._load_native_resume_config(ROOT)
    formal_path = ROOT / str(config["formal_plan_path"])
    todo_path = ROOT / str(config["taskboard_path"])
    formal_plan = materializer.FormalWorkPlan.from_dict(
        json.loads(formal_path.read_text(encoding="utf-8"))
    )
    population = materializer.project_population(
        config,
        formal_plan=formal_plan,
        todo_text=todo_path.read_text(encoding="utf-8"),
        source={
            "accelerator_head": operator._git_text(
                ROOT,
                ("rev-parse", "HEAD"),
                noun="test source HEAD",
            ),
            "accelerator_tree": operator._git_text(
                ROOT,
                ("rev-parse", "HEAD^{tree}"),
                noun="test source tree",
            ),
            "source_forest_root": "sha256:" + ("a" * 64),
        },
    )
    stage = tmp_path / "run-v39.stage-test"
    stage.mkdir(mode=0o700)
    staged_config = operator._native_resume_stage_config(
        config,
        root=tmp_path,
        stage=stage,
    )

    receipt = materializer._materialize_canonical(
        staged_config,
        population,
        root=tmp_path,
        recheck_source=False,
    )
    profile = operator._verify_profile(stage / "control.duckdb")
    operator._privatize_and_sync_native_resume_stage(stage)
    operator._validate_native_bootstrap_receipt(
        receipt,
        config=config,
        database_paths={
            "control": "run-v39.stage-test/control.duckdb",
            "coordination": "run-v39.stage-test/control.coordination.duckdb",
            "execution": "run-v39.stage-test/control.execution.duckdb",
        },
        source_head=str(population["source_head"]),
        repository_tree_id=str(population["repository_tree_id"]),
        population_root=str(population["population_root"]),
        plan_root_cid=str(population["plan_root_cid"]),
        schema_fingerprint=str(profile["schema_fingerprint"]),
        catalog_fingerprint=str(profile["catalog_fingerprint"]),
    )
    operator._verify_native_resume_stage_allowlist(
        stage,
        include_provenance=False,
    )
    unexpected = stage / "undeclared-runtime-object"
    unexpected.write_bytes(b"unexpected")
    unexpected.chmod(0o600)
    with pytest.raises(operator.SuccessorOperatorError, match="exact allowlist"):
        operator._verify_native_resume_stage_allowlist(
            stage,
            include_provenance=False,
        )
    unexpected.unlink()
    semantic_tampers: list[dict[str, object]] = []
    wrong_path = json.loads(json.dumps(receipt))
    wrong_path["database_paths"]["control"] = "wrong/control.duckdb"
    semantic_tampers.append(wrong_path)
    extra_schema_authority = json.loads(json.dumps(receipt))
    extra_schema_authority["schema_install"]["unvalidated_authority"] = True
    semantic_tampers.append(extra_schema_authority)
    extra_progress = json.loads(json.dumps(receipt))
    extra_progress["verification"]["control"]["unvalidated_progress"] = {
        "claims": 9
    }
    extra_progress["verification"].pop("verification_root")
    extra_progress["verification"]["verification_root"] = operator._content_id(
        extra_progress["verification"]
    )
    semantic_tampers.append(extra_progress)
    integer_task_ids = json.loads(json.dumps(receipt))
    integer_task_ids["materialization"]["registered_task_cids"] = list(range(28))
    integer_task_ids["materialization"]["bootstrap_completed_task_cids"] = list(
        range(7)
    )
    integer_task_ids["materialization"]["task_source"]["task_cids"] = list(
        range(28)
    )
    semantic_tampers.append(integer_task_ids)
    boolean_writer_count = json.loads(json.dumps(receipt))
    boolean_writer_count["maximum_writer_processes"] = True
    semantic_tampers.append(boolean_writer_count)
    for tampered in semantic_tampers:
        tampered.pop("receipt_cid")
        tampered["receipt_cid"] = operator._content_id(tampered)
        with pytest.raises(operator.SuccessorOperatorError, match="semantics differ"):
            operator._validate_native_bootstrap_receipt(
                tampered,
                config=config,
                database_paths={
                    "control": "run-v39.stage-test/control.duckdb",
                    "coordination": (
                        "run-v39.stage-test/control.coordination.duckdb"
                    ),
                    "execution": "run-v39.stage-test/control.execution.duckdb",
                },
                source_head=str(population["source_head"]),
                repository_tree_id=str(population["repository_tree_id"]),
                population_root=str(population["population_root"]),
                plan_root_cid=str(population["plan_root_cid"]),
                schema_fingerprint=str(profile["schema_fingerprint"]),
                catalog_fingerprint=str(profile["catalog_fingerprint"]),
            )
    projection = operator._verify_native_resume_projection(
        stage / "control.duckdb",
        config=config,
    )

    assert projection["completed_count"] == 7
    assert projection["todo_count"] == 19
    assert projection["blocked_count"] == 2
    assert projection["ready_task_ids"] == [
        "LGCVF-051",
        "LGCVF-060",
        "LGCVF-070",
        "LGCVF-080",
    ]


def test_successor_bootstrap_invokes_protected_recovery_verifier_isolated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _operator()
    observed: dict[str, object] = {}
    report = {
        "valid": True,
        "target_generation": "lgcvf-run-v17",
        "stores_unchanged": True,
        "source_database_statuses_read": False,
        "completed_count": 13,
        "todo_count": 13,
        "blocked_count": 2,
        "ready_task_ids": ["LGCVF-081"],
    }

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        observed["command"] = command
        observed["kwargs"] = kwargs
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(report),
            stderr="",
        )

    monkeypatch.setattr(operator.subprocess, "run", run)

    assert operator._canonical_recovery_verification(tmp_path) == report
    command = observed["command"]
    assert isinstance(command, list)
    assert command[:4] == [sys.executable, "-I", "-S", "-B"]
    assert command[-1] == "recovery-verify"


def test_lgcvf_lane_supervisor_import_closure_excludes_optional_host_crypto() -> None:
    """The sealed ``-S`` lane must import before optional EAAEF crypto is used."""

    probe = r'''import runpy,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve()
sys.path[:0]=[str(root),str(root/'ipfs_datasets_py')]
namespace=runpy.run_module(
    'ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor',
    run_name='ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor',
    alter_sys=True,
)
if not callable(namespace.get('main')): raise SystemExit(10)
if 'cryptography' in sys.modules: raise SystemExit(11)
if 'ipfs_accelerate_py.agent_supervisor.entrypoints.local_profile' in sys.modules: raise SystemExit(12)
'''
    completed = subprocess.run(
        (sys.executable, "-I", "-S", "-B", "-c", probe, str(ROOT)),
        cwd=ROOT,
        env={
            "HOME": str(Path.home()),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": "/usr/bin:/bin",
        },
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_lgcvf_sealed_module_sibling_keeps_exact_proc_fd_member() -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
        implementation_daemon,
    )

    origin = Path(
        "/proc/self/fd/71/ipfs_accelerate_py/agent_supervisor/"
        "todo_daemon/implementation_daemon.py"
    )

    assert implementation_daemon._trusted_quota_fallback_script_path(
        origin
    ) == Path(
        "/proc/self/fd/71/ipfs_accelerate_py/agent_supervisor/"
        "grok_cli_runner.py"
    )
    assert implementation_daemon._trusted_provider_fallback_script_path(
        origin
    ) == Path(
        "/proc/self/fd/71/ipfs_accelerate_py/agent_supervisor/"
        "provider_fallback_runner.py"
    )
    with pytest.raises(ValueError, match="sibling name is not admitted"):
        implementation_daemon._trusted_packaged_sibling_script_path(
            origin,
            "lookalike_runner.py",
        )


def test_lgcvf_sealed_provider_commands_keep_exact_proc_fd_members(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
        implementation_daemon,
    )

    grok_runner = Path(
        "/proc/self/fd/71/ipfs_accelerate_py/agent_supervisor/"
        "grok_cli_runner.py"
    )
    fallback_runner = grok_runner.with_name("provider_fallback_runner.py")
    monkeypatch.setattr(
        implementation_daemon,
        "_TRUSTED_QUOTA_FALLBACK_SCRIPT",
        grok_runner,
    )
    monkeypatch.setattr(
        implementation_daemon,
        "_TRUSTED_PROVIDER_FALLBACK_SCRIPT",
        fallback_runner,
    )
    monkeypatch.setattr(
        implementation_daemon,
        "_grok_binary",
        lambda: "/usr/bin/true",
    )

    grok_command = implementation_daemon._grok_cli_trusted_failure_command(
        workspace_path=tmp_path,
        model="grok-4.6",
    )
    fallback_command = implementation_daemon._ordered_provider_fallback_command(
        workspace_path=tmp_path,
        primary_provider="grok",
        primary_command=grok_command,
        fallback_provider="codex",
        fallback_command=["/usr/bin/true"],
    )

    assert grok_command[1] == str(grok_runner)
    assert fallback_command[1] == str(fallback_runner)
    assert implementation_daemon._uses_packaged_provider_fallback_runner(
        shlex.join([implementation_daemon.sys.executable, str(fallback_runner)])
    )
    assert implementation_daemon._uses_packaged_provider_fallback_runner(
        shlex.join([str(fallback_runner)])
    )
    assert not implementation_daemon._uses_packaged_provider_fallback_runner(
        shlex.join(
            [
                implementation_daemon.sys.executable,
                str(fallback_runner).replace("/fd/71/", "/fd/72/"),
            ]
        )
    )


def test_verified_run_v17_clone_is_no_overwrite_and_content_addressed(
    tmp_path: Path,
) -> None:
    operator = _operator()
    source = tmp_path / "run-v17" / "control.duckdb"
    target = tmp_path / "run-v23" / "control.duckdb"
    provenance = tmp_path / "run-v23" / "evidence" / "provenance.json"
    source.parent.mkdir(parents=True)
    _seed_datasets_profile(source)
    source_digest = operator._sha256_regular_file(source)
    recovery = {
        "valid": True,
        "target_generation": "lgcvf-run-v17",
        "stores_unchanged": True,
        "source_database_statuses_read": False,
        "verification_root": "sha256:" + ("ab" * 32),
        "receipt_cid": "baguqeera-test-recovery",
    }

    receipt = operator.clone_verified_successor(
        source,
        target,
        provenance,
        recovery_verification=recovery,
    )

    assert source_digest == operator._sha256_regular_file(source)
    assert source_digest == operator._sha256_regular_file(target)
    assert {item.name for item in target.parent.iterdir()} == {
        "control.duckdb",
        "evidence",
    }
    assert {item.name for item in provenance.parent.iterdir()} == {provenance.name}
    assert not tuple(tmp_path.glob("run-v23.stage-*"))
    for path, expected_mode in (
        (target.parent, 0o700),
        (provenance.parent, 0o700),
        (target, 0o600),
        (provenance, 0o600),
    ):
        metadata = path.stat()
        assert metadata.st_mode & 0o777 == expected_mode
        if path.is_file():
            assert metadata.st_nlink == 1
    assert receipt["source_generation"] == "lgcvf-run-v17"
    assert receipt["target_generation"] == "lgcvf-run-v23"
    assert receipt["source_database_statuses_read"] is False
    assert (
        operator._strict_json(
            provenance,
            expected_schema=operator.PROVENANCE_SCHEMA,
        )
        == receipt
    )
    with pytest.raises(operator.SuccessorOperatorError, match="overwrite"):
        operator.clone_verified_successor(
            source,
            target,
            provenance,
            recovery_verification=recovery,
        )


@pytest.mark.parametrize("existing_kind", ("directory", "file", "dangling_link"))
def test_successor_bootstrap_rejects_any_preexisting_generation(
    tmp_path: Path, existing_kind: str
) -> None:
    operator = _operator()
    source = tmp_path / "run-v17" / "control.duckdb"
    target = tmp_path / "run-v23" / "control.duckdb"
    provenance = tmp_path / "run-v23" / "evidence" / "provenance.json"
    source.parent.mkdir(parents=True)
    _seed_datasets_profile(source)
    if existing_kind == "directory":
        target.parent.mkdir()
    elif existing_kind == "file":
        target.parent.write_bytes(b"occupied")
    else:
        target.parent.symlink_to("missing-run-v23")
    before = os.lstat(target.parent)
    recovery = {
        "valid": True,
        "target_generation": "lgcvf-run-v17",
        "stores_unchanged": True,
        "source_database_statuses_read": False,
    }

    with pytest.raises(operator.SuccessorOperatorError, match="overwrite"):
        operator.clone_verified_successor(
            source,
            target,
            provenance,
            recovery_verification=recovery,
        )

    after = os.lstat(target.parent)
    assert (after.st_dev, after.st_ino, after.st_mode) == (
        before.st_dev,
        before.st_ino,
        before.st_mode,
    )


def test_successor_receipt_failure_never_publishes_database_only_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    operator = _operator()
    source = tmp_path / "run-v17" / "control.duckdb"
    target = tmp_path / "run-v23" / "control.duckdb"
    provenance = tmp_path / "run-v23" / "evidence" / "provenance.json"
    source.parent.mkdir(parents=True)
    _seed_datasets_profile(source)

    def fail_receipt(*args: object, **kwargs: object) -> None:
        raise OSError("injected receipt failure")

    monkeypatch.setattr(operator, "_atomic_json", fail_receipt)
    with pytest.raises(OSError, match="injected receipt failure"):
        operator.clone_verified_successor(
            source,
            target,
            provenance,
            recovery_verification={
                "valid": True,
                "target_generation": "lgcvf-run-v17",
                "stores_unchanged": True,
                "source_database_statuses_read": False,
            },
        )

    assert not os.path.lexists(target.parent)
    assert not tuple(tmp_path.glob("run-v23.stage-*"))


def test_successor_stage_remains_clean_during_reverification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    operator = _operator()
    repository = tmp_path / "repo"
    repository.mkdir()
    (repository / ".gitignore").write_text("run-v*/\n", encoding="utf-8")
    for arguments in (
        ("init", "-q"),
        ("config", "user.email", "lgcvf-test@example.invalid"),
        ("config", "user.name", "LGCVF Test"),
        ("add", ".gitignore"),
        ("commit", "-qm", "stage ignore boundary"),
    ):
        subprocess.run(
            ["/usr/bin/git", *arguments],
            cwd=repository,
            check=True,
            env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
        )
    source = repository / "run-v17" / "control.duckdb"
    target = repository / "run-v23" / "control.duckdb"
    provenance = repository / "run-v23" / "evidence" / "provenance.json"
    source.parent.mkdir()
    _seed_datasets_profile(source)
    operator._require_ignored_successor(repository)
    original_atomic_json = operator._atomic_json
    observed_status: list[str] = []

    def inspect_stage(path: Path, value: object, *, replace: bool) -> None:
        observed_status.append(
            operator._git_text(
                repository,
                (
                    "status",
                    "--porcelain=v1",
                    "--untracked-files=all",
                    "--ignore-submodules=none",
                ),
                noun="staged successor test inventory",
            )
        )
        original_atomic_json(path, value, replace=replace)

    monkeypatch.setattr(operator, "_atomic_json", inspect_stage)
    operator.clone_verified_successor(
        source,
        target,
        provenance,
        recovery_verification={
            "valid": True,
            "target_generation": "lgcvf-run-v17",
            "stores_unchanged": True,
            "source_database_statuses_read": False,
        },
    )

    assert observed_status == [""]
    assert not tuple(repository.glob("run-v23.stage-*"))


@pytest.mark.parametrize(
    ("custody_change", "error"),
    (
        ("database_hardlink", "bounded private"),
        ("database_wal", "live WAL"),
        ("coordination_wal", "coordination database has a live WAL"),
        ("execution_wal", "execution database has a live WAL"),
        ("provenance_symlink", "unreadable"),
    ),
)
def test_successor_load_rejects_aliases_and_live_wal(
    tmp_path: Path, custody_change: str, error: str
) -> None:
    operator = _operator()
    source = tmp_path / "run-v17" / "control.duckdb"
    target = tmp_path / "run-v23" / "control.duckdb"
    provenance = tmp_path / "run-v23" / "evidence" / "provenance.json"
    source.parent.mkdir(parents=True)
    _seed_datasets_profile(source)
    operator.clone_verified_successor(
        source,
        target,
        provenance,
        recovery_verification={
            "valid": True,
            "target_generation": "lgcvf-run-v17",
            "stores_unchanged": True,
            "source_database_statuses_read": False,
        },
    )
    paths = {
        "source_database": source,
        "successor_database": target,
        "provenance": provenance,
    }
    assert operator._load_provenance(paths, root=tmp_path)["target_database"] == str(
        target
    )

    if custody_change == "database_hardlink":
        os.link(target, target.with_name("hidden-control-alias.duckdb"))
    elif custody_change == "database_wal":
        target.with_name(target.name + ".wal").touch(mode=0o600)
    elif custody_change == "coordination_wal":
        target.with_name("control.coordination.duckdb.wal").touch(mode=0o600)
    elif custody_change == "execution_wal":
        target.with_name("control.execution.duckdb.wal").touch(mode=0o600)
    else:
        receipt = operator._strict_json(provenance)
        alias = provenance.with_name("identical-provenance.json")
        operator._atomic_json(alias, receipt, replace=False)
        provenance.unlink()
        provenance.symlink_to(alias.name)

    with pytest.raises(operator.SuccessorOperatorError, match=error):
        operator._load_provenance(paths, root=tmp_path)


def test_controller_lock_is_held_before_locked_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    operator = _operator()
    paths = operator._paths(tmp_path)
    held = operator._open_private_lock(paths["controller_lock"])
    fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    entered = False

    def locked_run(*args: object, **kwargs: object) -> int:
        nonlocal entered
        entered = True
        return 0

    monkeypatch.setattr(operator, "_run_locked_successor", locked_run)
    try:
        with pytest.raises(operator.SuccessorOperatorError, match="owns the lock"):
            operator.run_successor(
                tmp_path / "candidate.json",
                root=tmp_path,
                implement=False,
                duration_seconds=1.0,
            )
    finally:
        fcntl.flock(held.fileno(), fcntl.LOCK_UN)
        held.close()
    assert entered is False


def test_generation_bound_controller_lock_rejects_run_directory_replacement(
    tmp_path: Path,
) -> None:
    operator = _operator()
    paths = operator._paths(tmp_path)
    generation = paths["controller_lock"].parent
    generation.mkdir(mode=0o700, parents=True)
    displaced = generation.with_name(generation.name + ".displaced")

    with pytest.raises(
        operator.SuccessorOperatorError,
        match="generation/controller lock binding changed",
    ):
        with operator._exclusive_projection_checkpoint(paths):
            generation.rename(displaced)
            generation.mkdir(mode=0o700)
            replacement_lock = generation / paths["controller_lock"].name
            replacement_lock.touch(mode=0o600)

    assert displaced.is_dir()
    assert generation.is_dir()
    assert (displaced / paths["controller_lock"].name).is_file()
    assert (generation / paths["controller_lock"].name).is_file()


def test_runtime_inventory_admits_only_exact_clean_initialized_gitlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _operator()
    parent = tmp_path / "parent"
    nested = parent / "runtime" / "module"
    nested.mkdir(parents=True)

    def run_git(repository: Path, *arguments: str) -> str:
        completed = subprocess.run(
            [
                str(operator.GIT_EXECUTABLE),
                "-c",
                "core.hooksPath=/dev/null",
                "-c",
                "user.name=LGCVF Test",
                "-c",
                "user.email=lgcvf-test@example.invalid",
                *arguments,
            ],
            cwd=repository,
            env={
                "GIT_CONFIG_GLOBAL": "/dev/null",
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_NO_REPLACE_OBJECTS": "1",
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
                "PATH": "/usr/bin:/bin",
            },
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        return completed.stdout.strip()

    run_git(parent, "init", "-q")
    run_git(nested, "init", "-q")
    nested.joinpath(".gitignore").write_text("ignored.data\n", encoding="utf-8")
    nested.joinpath("payload.txt").write_text("sealed nested source\n", encoding="utf-8")
    run_git(nested, "add", ".gitignore", "payload.txt")
    run_git(nested, "commit", "-qm", "nested source")
    nested_head = run_git(nested, "rev-parse", "HEAD")
    run_git(
        parent,
        "update-index",
        "--add",
        "--cacheinfo",
        f"160000,{nested_head},runtime/module",
    )
    run_git(parent, "commit", "-qm", "parent gitlink")
    parent_head = run_git(parent, "rev-parse", "HEAD")

    stowed_nested = tmp_path / "stowed-nested"
    nested.rename(stowed_nested)
    nested.mkdir()
    empty_inventory = operator._tracked_runtime_inventory(
        parent,
        head=parent_head,
        pathspecs=("runtime",),
        noun="test runtime",
    )
    nested.rmdir()
    stowed_nested.rename(nested)

    inventory = operator._tracked_runtime_inventory(
        parent,
        head=parent_head,
        pathspecs=("runtime",),
        noun="test runtime",
    )
    assert inventory == empty_inventory
    assert inventory["tracked_object_count"] == 1
    assert str(inventory["tracked_inventory_root"]).startswith("sha256:")

    with pytest.raises(operator.SuccessorOperatorError, match="nesting is too deep"):
        operator._tracked_runtime_inventory(
            parent,
            head=parent_head,
            pathspecs=("runtime",),
            noun="test runtime",
            _gitlink_depth=operator.MAX_RUNTIME_GITLINK_DEPTH + 1,
        )
    with pytest.raises(operator.SuccessorOperatorError, match="gitlink cycle differs"):
        operator._tracked_runtime_inventory(
            parent,
            head=parent_head,
            pathspecs=("runtime",),
            noun="test runtime",
            _gitlink_chain=frozenset({(str(nested.resolve()), nested_head)}),
        )

    runtime = parent / "runtime"
    outside_runtime = tmp_path / "outside-runtime"
    runtime.rename(outside_runtime)
    runtime.symlink_to(outside_runtime, target_is_directory=True)
    with pytest.raises(operator.SuccessorOperatorError, match="gitlink custody differs"):
        operator._tracked_runtime_inventory(
            parent,
            head=parent_head,
            pathspecs=("runtime",),
            noun="test runtime",
        )
    runtime.unlink()
    outside_runtime.rename(runtime)

    nested.joinpath("untracked.txt").write_text("drift\n", encoding="utf-8")
    with pytest.raises(
        operator.SuccessorOperatorError,
        match="initialized gitlink custody differs",
    ):
        operator._tracked_runtime_inventory(
            parent,
            head=parent_head,
            pathspecs=("runtime",),
            noun="test runtime",
        )
    nested.joinpath("untracked.txt").unlink()

    nested.joinpath("payload.txt").write_text("tracked drift\n", encoding="utf-8")
    with pytest.raises(
        operator.SuccessorOperatorError,
        match="initialized gitlink custody differs",
    ):
        operator._tracked_runtime_inventory(
            parent,
            head=parent_head,
            pathspecs=("runtime",),
            noun="test runtime",
        )
    nested.joinpath("payload.txt").write_text(
        "sealed nested source\n", encoding="utf-8"
    )

    run_git(nested, "update-index", "--skip-worktree", "payload.txt")
    with pytest.raises(operator.SuccessorOperatorError, match="special index flags"):
        operator._tracked_runtime_inventory(
            parent,
            head=parent_head,
            pathspecs=("runtime",),
            noun="test runtime",
        )
    run_git(nested, "update-index", "--no-skip-worktree", "payload.txt")

    nested.joinpath("ignored.data").write_text("ignored drift\n", encoding="utf-8")
    with pytest.raises(operator.SuccessorOperatorError, match="ignored executable"):
        operator._tracked_runtime_inventory(
            parent,
            head=parent_head,
            pathspecs=("runtime",),
            noun="test runtime",
        )
    nested.joinpath("ignored.data").unlink()

    git_text = operator._git_text
    injected_ignored_drift = False

    def inject_ignored_drift_after_scan(
        repository: Path,
        arguments: tuple[str, ...],
        *,
        noun: str,
    ) -> str:
        nonlocal injected_ignored_drift
        result = git_text(repository, arguments, noun=noun)
        if (
            repository == nested
            and arguments[:3] == ("ls-files", "--others", "--ignored")
            and not injected_ignored_drift
        ):
            injected_ignored_drift = True
            nested.joinpath("ignored.data").write_text(
                "late ignored drift\n", encoding="utf-8"
            )
        return result

    monkeypatch.setattr(operator, "_git_text", inject_ignored_drift_after_scan)
    with pytest.raises(
        operator.SuccessorOperatorError,
        match="ignored executable|initialized gitlink custody changed",
    ):
        operator._tracked_runtime_inventory(
            parent,
            head=parent_head,
            pathspecs=("runtime",),
            noun="test runtime",
        )
    assert injected_ignored_drift is True
    nested.joinpath("ignored.data").unlink()
    monkeypatch.setattr(operator, "_git_text", git_text)

    nested.joinpath("payload.txt").write_text("alternate source\n", encoding="utf-8")
    run_git(nested, "add", "payload.txt")
    run_git(nested, "commit", "-qm", "alternate nested source")
    alternate_head = run_git(nested, "rev-parse", "HEAD")
    with pytest.raises(
        operator.SuccessorOperatorError,
        match="initialized gitlink custody differs",
    ):
        operator._tracked_runtime_inventory(
            parent,
            head=parent_head,
            pathspecs=("runtime",),
            noun="test runtime",
        )
    run_git(nested, "reset", "--hard", nested_head)

    regular_git_blob_oid = operator._regular_git_blob_oid
    switched_head = False

    def switch_head_after_hash(path: Path, *, noun: str) -> str:
        nonlocal switched_head
        observed_oid = regular_git_blob_oid(path, noun=noun)
        if path == nested / "payload.txt" and not switched_head:
            switched_head = True
            run_git(nested, "reset", "--hard", alternate_head)
        return observed_oid

    monkeypatch.setattr(operator, "_regular_git_blob_oid", switch_head_after_hash)
    with pytest.raises(
        operator.SuccessorOperatorError,
        match="initialized gitlink custody changed",
    ):
        operator._tracked_runtime_inventory(
            parent,
            head=parent_head,
            pathspecs=("runtime",),
            noun="test runtime",
        )
    assert switched_head is True


def test_git_observation_decode_failure_is_typed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _operator()

    def fail_decode(*_args: object, **_kwargs: object) -> None:
        raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")

    monkeypatch.setattr(operator.subprocess, "run", fail_decode)
    with pytest.raises(
        operator.SuccessorOperatorError,
        match="test Git observation could not be observed",
    ):
        operator._git_text(tmp_path, ("status",), noun="test Git observation")


def test_profile_read_only_verifier_cannot_request_writable_database(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources import (
        control_plane_schema as schema_module,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources import (
        duckdb_state as state_module,
    )

    operator = _operator()
    database = tmp_path / "control.duckdb"
    database.write_bytes(b"immutable-test-database\n")
    database.chmod(0o600)
    inventory_before = tuple(sorted(item.name for item in tmp_path.iterdir()))
    raw_before = database.read_bytes()
    connector_modes: list[bool] = []

    class RawConnection:
        def close(self) -> None:
            return None

    class WrappedConnection:
        def close(self) -> None:
            return None

    def connect(
        _engine: object,
        path: str,
        *,
        read_only: bool,
    ) -> RawConnection:
        assert path.startswith("/proc/self/fd/")
        assert read_only is True
        connector_modes.append(read_only)
        return RawConnection()

    def verifier(path: Path) -> dict[str, object]:
        with globals()["open_duckdb_connection"](path):
            return {
                "valid": True,
                "schema_fingerprint": "schema:test-read-only",
                "catalog_fingerprint": "catalog:test-read-only",
            }

    assert verifier.__closure__ is None
    monkeypatch.setattr(
        schema_module,
        "verify_datasets_authoritative_operational_schema",
        verifier,
    )
    monkeypatch.setattr(
        schema_module,
        "load_datasets_authoritative_operational_catalog",
        lambda: SimpleNamespace(
            fingerprint=lambda: "catalog:test-read-only"
        ),
    )
    monkeypatch.setattr(state_module, "connect_duckdb_with_policy", connect)
    monkeypatch.setattr(
        state_module.DuckDBConnection,
        "wrap",
        staticmethod(lambda _raw: WrappedConnection()),
    )
    directory_descriptor = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        verification = operator._verify_profile(
            Path(f"/proc/self/fd/{directory_descriptor}/{database.name}"),
            read_only=True,
        )
    finally:
        os.close(directory_descriptor)

    assert verification["valid"] is True
    assert connector_modes == [True]
    assert database.read_bytes() == raw_before
    assert tuple(sorted(item.name for item in tmp_path.iterdir())) == (
        inventory_before
    )


def test_live_launch_verifies_provenance_before_import_retarget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py import llm_router

    operator = _operator()
    events: list[str] = []
    provenance_loaded = False
    raw_provenance = {"receipt_cid": "baguqeera-test-live-provenance"}
    verified_provenance = dict(raw_provenance)
    continuity = {"current_head": "a" * 40, "current_tree": "b" * 40}
    native_launch = object()
    capsule_pin = object()
    prior_signal_handlers = {
        signal.SIGINT: object(),
        signal.SIGTERM: object(),
    }
    signal_installations = 0

    class Capsule:
        descriptor = 991

    capsule = Capsule()

    class RetargetReached(BaseException):
        pass

    class SealedArchive:
        def __init__(self, path: str, *, mode: str) -> None:
            assert path == "/proc/self/fd/991"
            assert mode == "r"

        def __enter__(self) -> SealedArchive:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self, _member: str) -> bytes:
            return b"sealed"

    def load_raw(_paths: object, **_kwargs: object) -> dict[str, str]:
        events.append("raw")
        return raw_provenance

    def prepare(**kwargs: object) -> dict[str, object]:
        events.append("prepare")
        assert kwargs["provenance"] is raw_provenance
        return {
            "launch_home": tmp_path / "qualification-home",
            "native_launch": native_launch,
            "capsule_pin": capsule_pin,
            "capsule": capsule,
            "archive_path": "/proc/self/fd/991",
            "continuity": continuity,
        }

    def preload(observed_launch: object) -> None:
        events.append("native")
        assert observed_launch is native_launch

    def load_verified(
        _paths: object,
        *,
        root: Path,
        expected_receipt: object,
    ) -> dict[str, str]:
        nonlocal provenance_loaded
        events.append("provenance")
        assert root == tmp_path
        assert expected_receipt is raw_provenance
        provenance_loaded = True
        return verified_provenance

    def post_provenance_preload() -> tuple[str, ...]:
        events.append("post_provenance_preload")
        assert provenance_loaded is True
        return ("ipfs_accelerate_py.loaded_during_provenance",)

    def refresh_manifest(
        observed_pin: object,
        observed_descriptor: int,
    ) -> tuple[str, dict[str, str]]:
        events.append("manifest")
        assert provenance_loaded is True
        assert observed_pin is capsule_pin
        assert observed_descriptor == capsule.descriptor
        return (
            "/proc/self/fd/991",
            {"sealed.py": "sha256:" + ("c" * 64)},
        )

    def audit(**kwargs: object) -> tuple[str, ...]:
        events.append("audit")
        assert provenance_loaded is True
        assert callable(kwargs["read_member"])
        return ("ipfs_accelerate_py.loaded_during_provenance",)

    def final_continuity(root: Path) -> dict[str, str]:
        events.append("final_continuity")
        assert provenance_loaded is True
        assert root == tmp_path
        return dict(continuity)

    def retarget(**kwargs: object) -> tuple[str, ...]:
        events.append("retarget")
        assert provenance_loaded is True
        assert kwargs == {"root": tmp_path, "archive_path": "/proc/self/fd/991"}
        raise RetargetReached

    def set_signal(signum: int, handler: object) -> object:
        nonlocal signal_installations
        name = signal.Signals(signum).name
        if signal_installations < len(prior_signal_handlers):
            events.append(f"install_{name}")
            signal_installations += 1
            return prior_signal_handlers[signum]
        assert handler is prior_signal_handlers[signum]
        events.append(f"restore_{name}")
        return handler

    monkeypatch.setattr(
        operator,
        "_load_lgcvf_live_raw_provenance_receipt",
        load_raw,
    )
    monkeypatch.setattr(
        operator,
        "_prepare_lgcvf_configured_board_live_launch",
        prepare,
    )
    monkeypatch.setattr(
        llm_router,
        "preload_agent_supervisor_native_dependency",
        preload,
    )
    monkeypatch.setattr(operator, "_load_provenance", load_verified)
    monkeypatch.setattr(
        operator,
        "_preload_lgcvf_live_controller_dependency_closure",
        post_provenance_preload,
    )
    monkeypatch.setattr(
        operator,
        "_lgcvf_live_sealed_manifest_inventory",
        refresh_manifest,
    )
    monkeypatch.setattr(
        operator,
        "_audit_lgcvf_live_loaded_repository_modules",
        audit,
    )
    monkeypatch.setattr(operator, "_candidate_runtime_continuity", final_continuity)
    monkeypatch.setattr(
        operator,
        "_retarget_lgcvf_live_repository_imports",
        retarget,
    )
    monkeypatch.setattr(operator.zipfile, "ZipFile", SealedArchive)
    monkeypatch.setattr(operator.signal, "signal", set_signal)
    monkeypatch.setattr(
        operator,
        "_close_lgcvf_configured_board_live_launch",
        lambda _launch: events.append("cleanup"),
    )
    for name in tuple(os.environ):
        if name.startswith("LD_") or name == "GLIBC_TUNABLES":
            monkeypatch.delenv(name)

    with pytest.raises(RetargetReached):
        operator._run_locked_successor(
            tmp_path / "candidate.json",
            root=tmp_path,
            implement=True,
            duration_seconds=float("inf"),
        )

    assert verified_provenance == raw_provenance
    assert verified_provenance is not raw_provenance
    assert events == [
        "raw",
        "prepare",
        "install_SIGINT",
        "install_SIGTERM",
        "native",
        "provenance",
        "post_provenance_preload",
        "manifest",
        "audit",
        "final_continuity",
        "retarget",
        "cleanup",
        "restore_SIGINT",
        "restore_SIGTERM",
    ]


def test_live_restart_recovery_runs_after_capsule_preparation_and_native_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py import llm_router

    operator = _operator()
    paths, provenance, _continuity = _stopped_state_continuity_fixture(
        operator, tmp_path, monkeypatch
    )
    events: list[str] = []
    native_launch = object()

    class AdmissionReached(BaseException):
        pass

    def raw(_paths: object, **_kwargs: object) -> dict[str, object]:
        events.append("raw")
        return provenance

    def prepare(**kwargs: object) -> dict[str, object]:
        events.append("prepare")
        assert kwargs["stopped_restart"] is True
        return {
            "launch_home": tmp_path / "qualification-home",
            "native_launch": native_launch,
            "stopped_restart": True,
        }

    def preload(observed: object) -> None:
        events.append("native")
        assert observed is native_launch

    original_restore = operator._restore_or_retire_stopped_restart_admission

    def restore(receipt_paths: dict[str, Path]) -> str:
        events.append("restore")
        return original_restore(receipt_paths)

    def recover(*args: object, **kwargs: object) -> None:
        events.append("recover")
        assert kwargs["lock_custody"] is custody

    def verify(**kwargs: object) -> None:
        events.append("verify")
        assert kwargs["lock_custody"] is custody
        raise AdmissionReached

    monkeypatch.setattr(
        operator,
        "_load_lgcvf_live_raw_provenance_receipt",
        raw,
    )
    monkeypatch.setattr(
        operator,
        "_prepare_lgcvf_configured_board_live_launch",
        prepare,
    )
    monkeypatch.setattr(
        llm_router,
        "preload_agent_supervisor_native_dependency",
        preload,
    )
    monkeypatch.setattr(
        operator,
        "_restore_or_retire_stopped_restart_admission",
        restore,
    )
    monkeypatch.setattr(
        operator,
        "_recover_interrupted_stopped_state_continuity",
        recover,
    )
    monkeypatch.setattr(
        operator,
        "_verify_lgcvf_live_provenance_before_import_retarget",
        verify,
    )
    monkeypatch.setattr(
        operator,
        "_close_lgcvf_configured_board_live_launch",
        lambda _launch: events.append("cleanup"),
    )
    for name in tuple(os.environ):
        if name.startswith("LD_") or name == "GLIBC_TUNABLES":
            monkeypatch.delenv(name)

    custody = operator._open_generation_bound_controller_lock(paths)
    handle = custody["lock_handle"]
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(AdmissionReached):
            operator._run_locked_successor(
                tmp_path / "candidate.json",
                root=tmp_path,
                implement=True,
                duration_seconds=float("inf"),
                _locked_paths=paths,
                _lock_custody=custody,
            )
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        operator._close_generation_bound_controller_lock(custody)

    assert events == [
        "raw",
        "prepare",
        "native",
        "restore",
        "recover",
        "verify",
        "cleanup",
    ]


def _live_controller_source_audit_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, Path, ModuleType, dict[str, bytes], dict[str, str]]:
    root = tmp_path / "repo"
    (root / "ipfs_datasets_py").mkdir(parents=True)
    operator_path = root / (
        "scripts/run_logic_governed_compositional_verification_fabric_quack.py"
    )
    module_path = root / "ipfs_accelerate_py/sealed_dependency.py"
    operator_path.parent.mkdir(parents=True)
    module_path.parent.mkdir(parents=True)
    operator_path.write_bytes(b"operator_identity = 'sealed'\n")
    module_path.write_bytes(b"dependency_identity = 'sealed'\n")
    module_name = "ipfs_accelerate_py.sealed_dependency"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    payloads = {
        operator_path.relative_to(root).as_posix(): operator_path.read_bytes(),
        module_path.relative_to(root).as_posix(): module_path.read_bytes(),
    }
    manifest = {
        relative: "sha256:" + hashlib.sha256(raw).hexdigest()
        for relative, raw in payloads.items()
    }
    return root, operator_path, module_path, module, payloads, manifest


def test_live_controller_source_audit_rejects_post_import_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _operator()
    (
        root,
        operator_path,
        module_path,
        module,
        payloads,
        manifest,
    ) = _live_controller_source_audit_fixture(tmp_path)
    module_name = "ipfs_accelerate_py.sealed_dependency"
    monkeypatch.setattr(
        operator,
        "LGCVF_LIVE_CONTROLLER_PRELOAD_MODULES",
        (module_name,),
    )
    outer_main = ModuleType("__main__")
    outer_main.__file__ = str(operator_path)
    arguments = {
        "root": root,
        "operator_path": operator_path,
        "manifest_files": manifest,
        "read_member": payloads.__getitem__,
        "modules": {"__main__": outer_main, module_name: module},
    }

    assert operator._audit_lgcvf_live_loaded_repository_modules(
        **arguments
    ) == (module_name,)
    operator_path.write_bytes(b"operator_identity = 'mutable'\n")
    with pytest.raises(
        operator.SuccessorOperatorError,
        match="outer operator bytes differ",
    ):
        operator._audit_lgcvf_live_loaded_repository_modules(**arguments)
    operator_path.write_bytes(payloads[operator_path.relative_to(root).as_posix()])
    module_path.write_bytes(b"dependency_identity = 'mutable'\n")
    with pytest.raises(
        operator.SuccessorOperatorError,
        match="module bytes differ",
    ):
        operator._audit_lgcvf_live_loaded_repository_modules(**arguments)


def test_live_controller_source_audit_rejects_loaded_origin_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _operator()
    (
        root,
        operator_path,
        _module_path,
        module,
        payloads,
        manifest,
    ) = _live_controller_source_audit_fixture(tmp_path)
    module_name = "ipfs_accelerate_py.sealed_dependency"
    monkeypatch.setattr(
        operator,
        "LGCVF_LIVE_CONTROLLER_PRELOAD_MODULES",
        (module_name,),
    )
    assert module.__spec__ is not None
    module.__spec__.origin = str(tmp_path / "substituted.py")

    with pytest.raises(
        operator.SuccessorOperatorError,
        match="module origin differs",
    ):
        operator._audit_lgcvf_live_loaded_repository_modules(
            root=root,
            operator_path=operator_path,
            manifest_files=manifest,
            read_member=payloads.__getitem__,
            modules={module_name: module},
        )


def test_live_controller_retargets_all_repository_package_search_paths(
    tmp_path: Path,
) -> None:
    operator = _operator()
    root = tmp_path / "repo"
    nested = root / "ipfs_datasets_py"
    package_root = root / "ipfs_accelerate_py"
    nested.mkdir(parents=True)
    package_root.mkdir()
    package_init = package_root / "__init__.py"
    package_init.write_bytes(b"# sealed package\n")
    spec = importlib.util.spec_from_file_location(
        "ipfs_accelerate_py",
        package_init,
        submodule_search_locations=[str(package_root)],
    )
    assert spec is not None and spec.loader is not None
    package = importlib.util.module_from_spec(spec)
    path_entries = [str(root), str(nested), "/usr/lib/python3.12"]
    foreign_finder = object()
    meta_path = [
        foreign_finder,
        operator.importlib.machinery.BuiltinImporter,
        operator.importlib.machinery.FrozenImporter,
        operator.importlib.machinery.PathFinder,
    ]

    retargeted = operator._retarget_lgcvf_live_repository_imports(
        root=root,
        archive_path="/proc/self/fd/991",
        modules={"ipfs_accelerate_py": package},
        path_entries=path_entries,
        meta_path=meta_path,
    )

    assert retargeted == ("ipfs_accelerate_py",)
    assert path_entries == [
        "/proc/self/fd/991/ipfs_datasets_py",
        "/proc/self/fd/991",
        "/usr/lib/python3.12",
    ]
    assert meta_path == [
        operator.importlib.machinery.BuiltinImporter,
        operator.importlib.machinery.FrozenImporter,
        operator.importlib.machinery.PathFinder,
    ]
    assert list(package.__path__) == [
        "/proc/self/fd/991/ipfs_accelerate_py"
    ]
    assert package.__spec__ is not None
    assert list(package.__spec__.submodule_search_locations or ()) == [
        "/proc/self/fd/991/ipfs_accelerate_py"
    ]


def test_live_extension_home_is_inside_ignored_successor_runtime() -> None:
    operator = _operator()
    relative = operator.LGCVF_LIVE_QUALIFICATION_HOMES_RELATIVE

    assert relative.parent == operator.SUCCESSOR_RUN_RELATIVE
    assert relative.name == "qualification-homes"
    for candidate in (relative, relative / ("a" * 64)):
        ignored = subprocess.run(
            [
                "/usr/bin/git",
                "check-ignore",
                "--quiet",
                "--",
                candidate.as_posix(),
            ],
            cwd=ROOT,
            env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
            check=False,
        )
        assert ignored.returncode == 0


def test_tracked_runtime_inventory_hashes_bytes_and_rejects_ignored_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    operator = _operator()
    repository = tmp_path / "repo"
    package = repository / "package"
    package.mkdir(parents=True)
    module = package / "module.py"
    module.write_text("value = 1\n", encoding="utf-8")
    (repository / ".gitignore").write_text(
        "__pycache__/\npackage/shadow.py\n", encoding="utf-8"
    )
    for arguments in (
        ("init", "-q"),
        ("config", "user.email", "lgcvf-test@example.invalid"),
        ("config", "user.name", "LGCVF Test"),
        ("add", ".gitignore", "package/module.py"),
        ("commit", "-qm", "inventory"),
    ):
        subprocess.run(
            ["/usr/bin/git", *arguments],
            cwd=repository,
            check=True,
            env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
        )
    head = subprocess.check_output(
        ["/usr/bin/git", "rev-parse", "HEAD"],
        cwd=repository,
        text=True,
    ).strip()
    receipt = operator._tracked_runtime_inventory(
        repository,
        head=head,
        pathspecs=("package",),
        noun="test runtime",
    )
    assert receipt["tracked_object_count"] == 1
    assert "ignored_pycache_quarantined" not in receipt
    assert "pycache_prefix" not in receipt

    ignored_pycache = package / "__pycache__" / "module.cpython-312.pyc"
    ignored_pycache.parent.mkdir()
    ignored_pycache.write_bytes(b"quarantined bytecode is never imported")
    assert (
        operator._tracked_runtime_inventory(
            repository,
            head=head,
            pathspecs=("package",),
            noun="test runtime",
        )
        == receipt
    )

    probe = """
import importlib.util
import json
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("lgcvf_inventory_probe", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print(json.dumps(module._tracked_runtime_inventory(
    Path(sys.argv[2]),
    head=sys.argv[3],
    pathspecs=("package",),
    noun="test runtime",
), sort_keys=True))
"""

    def fresh_process_inventory() -> dict[str, object]:
        completed = subprocess.run(
            [
                sys.executable,
                "-I",
                "-B",
                "-c",
                probe,
                str(OPERATOR_PATH),
                str(repository),
                head,
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        value = json.loads(completed.stdout)
        assert isinstance(value, dict)
        return value

    assert fresh_process_inventory() == fresh_process_inventory() == receipt

    second_quarantine = tmp_path / "second-pycache-quarantine"
    second_quarantine.mkdir(mode=0o700)
    monkeypatch.setattr(operator.sys, "pycache_prefix", str(second_quarantine))
    assert (
        operator._tracked_runtime_inventory(
            repository,
            head=head,
            pathspecs=("package",),
            noun="test runtime",
        )
        == receipt
    )

    module.write_text("value = 2\n", encoding="utf-8")
    with pytest.raises(operator.SuccessorOperatorError, match="bytes differ"):
        operator._tracked_runtime_inventory(
            repository,
            head=head,
            pathspecs=("package",),
            noun="test runtime",
        )
    module.write_text("value = 1\n", encoding="utf-8")
    (package / "shadow.py").write_text("value = 3\n", encoding="utf-8")
    with pytest.raises(operator.SuccessorOperatorError, match="ignored executable"):
        operator._tracked_runtime_inventory(
            repository,
            head=head,
            pathspecs=("package",),
            noun="test runtime",
        )


def test_sealed_manifest_identity_and_policy_authority_are_fail_closed(
    tmp_path: Path,
) -> None:
    operator = _operator()
    manifest_path = tmp_path / "manifest.json"
    addressed = {
        "schema": operator.FRESH_RECOVERY_MANIFEST_SCHEMA,
        "value": "sealed-test",
    }
    addressed["manifest_cid"] = operator._content_id(addressed)
    operator._atomic_json(manifest_path, addressed, replace=False)
    assert (
        operator._strict_addressed_json(
            manifest_path,
            expected_schema=operator.FRESH_RECOVERY_MANIFEST_SCHEMA,
            identity_field="manifest_cid",
            noun="test manifest",
        )
        == addressed
    )
    tampered = dict(addressed)
    tampered["value"] = "changed"
    operator._atomic_json(manifest_path, tampered, replace=True)
    with pytest.raises(operator.SuccessorOperatorError, match="content identity"):
        operator._strict_addressed_json(
            manifest_path,
            expected_schema=operator.FRESH_RECOVERY_MANIFEST_SCHEMA,
            identity_field="manifest_cid",
            noun="test manifest",
        )

    config = json.loads(
        (
            ROOT / "config/agent_supervisor_logic_governed_compositional_"
            "verification_fabric_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    policy = config["fresh_generation_recovery"]
    false_authority = {
        "candidate_authored_validation": True,
        "validation_self_authority": False,
        "validation_completion_authoritative": False,
        "source_database_statuses_read": False,
        "source_database_completion_records_imported": False,
        "synthetic_source_disposition": "quarantined_not_imported",
        "network_isolation_enforced": True,
        "model_provider_route": "none",
        "task_implementation_complete": False,
        "test_qualification_complete": False,
        "objective_complete": False,
        "release_qualified": False,
        "production_authorized": False,
    }
    common = {
        "source_generation": policy["source_generation"],
        "target_generation": policy["target_generation"],
        "source_head": "a" * 40,
        "source_tree": "b" * 40,
        "source_evidence_cid": "source:test",
        "plan_root_cid": config["plan_binding"]["formal_plan_content_id"],
        "population_root": "population:test",
        "validation_qualification_cid": "qualification:test",
        **false_authority,
    }
    manifest = {
        **common,
        "source_runtime_root": policy["source_runtime_root"],
        "target_runtime_root": policy["target_runtime_root"],
        "completion_partition": {
            "construction_completed_task_ids": list(
                operator.CONSTRUCTION_COMPLETED_TASK_IDS
            ),
            "recovered_completed_task_ids": list(operator.RECOVERED_COMPLETED_TASK_IDS),
            "rejected_synthetic_task_ids": list(operator.TODO_TASK_IDS),
            "preserved_blocked_task_ids": list(operator.BLOCKED_TASK_IDS),
            "completed_count": 13,
            "todo_count": 13,
            "blocked_count": 2,
        },
        "retained_completion_binding": {
            "binding_cid": policy["retained_completion_binding_cid"],
            "construction_completion_count": 7,
            "delta_cid": policy["retained_delta_cid"],
            "dynamic_completion_receipt_count": 5,
            "logical_completion_count": 12,
            "path": policy["retained_revision_receipt_path"],
            "protected_blocker_binding_cid": policy[
                "retained_protected_blocker_binding_cid"
            ],
            "receipt_cid": policy["retained_revision_receipt_cid"],
            "sha256": policy["retained_revision_receipt_sha256"],
            "successor_revision_cid": policy["retained_successor_revision_cid"],
        },
        "wrong_default_quarantine": {
            "incident_manifest_path": policy["wrong_default_incident_manifest_path"],
            "incident_manifest_sha256": policy[
                "wrong_default_incident_manifest_sha256"
            ],
            "incident_manifest_cid": policy["wrong_default_incident_manifest_cid"],
            "contaminated_coordination_manifest_path": policy[
                "contaminated_coordination_projection_path"
            ],
            "contaminated_coordination_manifest_sha256": policy[
                "contaminated_coordination_projection_sha256"
            ],
            "contaminated_coordination_manifest_cid": policy[
                "contaminated_coordination_projection_manifest_cid"
            ],
            "rejected_record_set_cid": policy[
                "contaminated_coordination_rejected_record_set_cid"
            ],
            "rejected_contaminated_coordination_projection_root": policy[
                "rejected_contaminated_coordination_projection_root"
            ],
            "rejected_synthetic_task_ids": list(operator.TODO_TASK_IDS),
            "disposition": "preserved_forensic_quarantine_not_imported",
            "source_database_opened": False,
        },
        "merge_completion_evidence": [
            dict(item) for item in policy["merge_completions"]
        ],
    }
    receipt = {
        **common,
        "completed_task_ids": list(operator.COMPLETED_TASK_IDS),
        "todo_task_ids": list(operator.TODO_TASK_IDS),
        "blocked_task_ids": list(operator.BLOCKED_TASK_IDS),
        "completed_count": 13,
        "todo_count": 13,
        "blocked_count": 2,
        "atomic_publish": True,
    }
    operator._validate_recovery_policy_projection(
        config=config,
        manifest=manifest,
        receipt=receipt,
    )
    receipt["production_authorized"] = True
    with pytest.raises(operator.SuccessorOperatorError, match="projection|ceiling"):
        operator._validate_recovery_policy_projection(
            config=config,
            manifest=manifest,
            receipt=receipt,
        )


def test_sealed_continuity_cli_requires_all_six_raw_byte_pins() -> None:
    operator = _operator()
    parser = operator._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["bootstrap-sealed-continuity", "--source-root", "/tmp/run-v17"]
        )
    pin = "sha256:" + ("ab" * 32)
    parsed = parser.parse_args(
        [
            "bootstrap-sealed-continuity",
            "--source-root",
            "/tmp/run-v17",
            "--control-sha256",
            pin,
            "--coordination-sha256",
            pin,
            "--execution-sha256",
            pin,
            "--bootstrap-sha256",
            pin,
            "--manifest-sha256",
            pin,
            "--recovery-receipt-sha256",
            pin,
        ]
    )
    assert parsed.command == "bootstrap-sealed-continuity"
    assert parsed.control_sha256 == pin


def test_owner_socket_and_ducklake_projection_are_physically_separate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _operator()
    production_paths = operator._paths(ROOT)
    socket_path = production_paths["owner_socket"]

    assert socket_path.parent.name == f"ipfs-accelerate-lgcvf-{os.geteuid()}"
    assert len(os.fsencode(socket_path)) <= operator.UNIX_SOCKET_PATH_CEILING
    assert str(ROOT) not in str(socket_path)
    operator._prepare_private_owner_socket(socket_path)
    metadata = os.lstat(socket_path.parent)
    assert oct(metadata.st_mode & 0o777) == "0o700"
    assert metadata.st_uid == os.geteuid()

    monkeypatch.setattr(
        operator,
        "_extension_preflight",
        lambda: {
            "available": True,
            "extensions": {
                "quack": "test",
                "ducklake": "test",
                "httpfs": "test",
            },
            "automatic_installation_permitted": False,
        },
    )
    preflight = operator.projection_preflight(tmp_path)
    runtime_paths = operator._paths(tmp_path)
    assert preflight["valid"] is False
    assert preflight["capability"]["available"] is True
    assert preflight["source_database_present"] is False
    assert preflight["provenance_receipt_present"] is False
    assert preflight["source_admitted"] is False
    assert Path(preflight["projection_root"]) == runtime_paths["projection_root"]
    assert (
        runtime_paths["projection_root"] / "control.duckdb"
        != runtime_paths["successor_database"]
    )
    assert preflight["authoritative"] is False
    assert preflight["scheduling_authority"] is False
    assert preflight["completion_authority"] is False
    assert preflight["read_by_scheduler"] is False
    assert preflight["requires_stopped_checkpoint"] is True
    assert preflight["restart_authority"] is False
    assert preflight["source_admission_mode"] == ""
    assert preflight["stopped_state_continuity_receipt_cid"] == ""
    assert preflight["projection_root_present"] is False
    assert preflight["projection_receipt_present"] is False


def _stopped_state_continuity_fixture(
    operator: ModuleType,
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, Path], dict[str, object], dict[str, object]]:
    from ipfs_accelerate_py.agent_supervisor.merge import worktree_lifecycle
    from ipfs_accelerate_py.utils.cid_utils import cid_from_sha256_digest

    paths = operator._paths(root)
    databases = operator._successor_state_databases(paths)
    paths["successor_database"].parent.mkdir(mode=0o700, parents=True)
    paths["controller_lock"].touch(mode=0o600)
    paths["controller_lock"].chmod(0o600)
    for name, database in databases.items():
        database.write_bytes(f"stopped-{name}\n".encode())
        database.chmod(0o600)
    bootstrap = (
        paths["successor_database"].parent
        / "evidence"
        / "bootstrap"
        / "materialization.json"
    )
    operator._atomic_json(
        bootstrap,
        {"fixture": "stopped-continuity-bootstrap"},
        replace=False,
    )

    schema_digest = hashlib.sha256(
        b"test-stopped-continuity-schema"
    ).hexdigest()
    schema_cid = cid_from_sha256_digest(
        bytes.fromhex(schema_digest),
        codec="dag-json",
    )
    provenance: dict[str, object] = {
        "receipt_cid": "b" + ("a" * 60),
        "database_uuid": "database:test-stopped-continuity",
        "schema_fingerprint": schema_cid,
        "catalog_fingerprint": "catalog:test-stopped-continuity",
    }
    restart_calls = 0

    def reject_mutated_restart(*args: object, **kwargs: object) -> object:
        nonlocal restart_calls
        restart_calls += 1
        raise operator.SuccessorOperatorError(
            operator.NATIVE_RESUME_LIVE_CONTINUITY_REQUIRED_ERROR
        )

    monkeypatch.setattr(operator, "_load_provenance", reject_mutated_restart)
    monkeypatch.setattr(
        operator,
        "_load_lgcvf_live_raw_provenance_receipt",
        lambda _paths, **_kwargs: provenance,
    )
    monkeypatch.setattr(
        operator,
        "_verify_profile",
        lambda _database, **_kwargs: {
            "schema_fingerprint": provenance["schema_fingerprint"],
            "catalog_fingerprint": provenance["catalog_fingerprint"],
        },
    )
    monkeypatch.setattr(
        operator,
        "_database_identity",
        lambda _database: {
            "database_uuid": provenance["database_uuid"],
            "schema_fingerprint": provenance["schema_fingerprint"],
        },
    )
    monkeypatch.setattr(
        worktree_lifecycle,
        "owner_liveness",
        lambda _birth: OwnerLiveness.DEAD,
    )
    birth = ProcessBirthIdentity(
        pid=2_000_000_001,
        start_time_ticks=17,
        boot_id="test-stopped-continuity",
        parent_pid=1,
    ).to_dict()
    scheduler_birth = ProcessBirthIdentity(
        pid=2_000_000_002,
        start_time_ticks=18,
        boot_id="test-stopped-continuity",
        parent_pid=int(birth["pid"]),
    ).to_dict()
    owner_identity = {
        "server_id": "server:test-stopped-continuity",
        "store_id": operator.SUCCESSOR_DATABASE_RELATIVE.as_posix(),
        "database_uuid": provenance["database_uuid"],
        "schema_fingerprint": f"sha256:{schema_digest}",
        "secret_handle": operator.SECRET_HANDLE,
        "process_birth": birth,
    }
    stopped = operator._status_payload(
        lifecycle="stopped",
        controller_birth=birth,
        provenance_cid=str(provenance["receipt_cid"]),
        owner_identity=owner_identity,
        scheduler_birth=scheduler_birth,
        scheduler_returncode=0,
        projection_root=paths["projection_root"],
    )
    operator._write_status(paths["controller_status"], stopped)
    paths["owner_state"].mkdir(mode=0o700, parents=True)
    operator._atomic_json(
        paths["owner_state"] / "quack-state-server.status.json",
        {
            "schema": operator.QUACK_STATE_SERVER_STATUS_SCHEMA,
            "lifecycle": "stopped",
            "database_path": str(paths["successor_database"]),
            "state_dir": str(paths["owner_state"]),
            "store_id": operator.SUCCESSOR_DATABASE_RELATIVE.as_posix(),
            "secret_handle": operator.SECRET_HANDLE,
            "owner_marker_path": str(
                paths["successor_database"].with_name(
                    ".control.duckdb.state-owner.json"
                )
            ),
            "identity": {**owner_identity, "status": "stopped"},
        },
        replace=False,
    )
    final_source_continuity = {
        "approved_branch": operator.APPROVED_BOARD_BRANCH,
        "resolved_remote_head": "f" * 40,
        "current_head": "f" * 40,
        "current_tree": "e" * 40,
        "candidate_worktree_clean": True,
        "datasets_head": "d" * 40,
        "datasets_tree": "c" * 40,
        "datasets_worktree_clean": True,
        "python_bytecode_quarantine": {"enabled": True},
        "superproject_runtime_inventory": {"tracked_object_count": 1},
        "datasets_runtime_inventory": {"tracked_object_count": 1},
    }
    monkeypatch.setattr(
        operator,
        "_candidate_runtime_continuity",
        lambda _root: final_source_continuity,
    )
    monkeypatch.setattr(
        operator,
        "_observe_candidate_runtime_continuity",
        lambda _root, *, require_resolved_remote: final_source_continuity,
    )
    monkeypatch.setattr(
        operator,
        "_validate_stopped_projection_native_provenance",
        lambda *args, **kwargs: None,
    )
    recovery_io_paths = operator._stopped_recovery_io_paths(paths, None)
    recovery_anchors = operator._capture_stopped_recovery_anchors(
        paths,
        root=root,
        stopped_status=stopped,
        provenance=provenance,
        io_paths=recovery_io_paths,
    )
    stopped = operator._bind_stopped_recovery_anchors_status(
        stopped,
        recovery_anchors,
    )
    operator._write_status(paths["controller_status"], stopped)
    continuity = operator._write_stopped_state_continuity(
        paths,
        root=root,
        stopped_status=stopped,
        provenance=provenance,
        owner_checkpoint={
            "checkpointed": True,
            "server_id": owner_identity["server_id"],
            "database_path": str(paths["successor_database"]),
            "at": "2026-08-26T00:00:00Z",
        },
        owner_stop={
            "stopped": True,
            "server_id": owner_identity["server_id"],
            "at": "2026-08-26T00:00:01Z",
        },
    )
    bound_status = operator._bind_stopped_state_continuity_status(
        stopped, continuity
    )
    operator._write_status(paths["controller_status"], bound_status)
    continuity["test_restart_calls"] = lambda: restart_calls
    return paths, provenance, continuity


def test_owner_schema_fingerprint_bridge_is_exact_dag_json_sha256_only() -> None:
    from ipfs_accelerate_py.utils.cid_utils import cid_from_sha256_digest

    operator = _operator()
    digest = hashlib.sha256(b"real-owner-schema-shape").digest()
    dag_json_cid = cid_from_sha256_digest(digest, codec="dag-json")
    raw_cid = cid_from_sha256_digest(digest, codec="raw")
    owner_fingerprint = f"sha256:{digest.hex()}"

    assert operator._owner_schema_fingerprint_matches_canonical_cid(
        owner_fingerprint,
        dag_json_cid,
    ) is True
    assert operator._owner_schema_fingerprint_matches_canonical_cid(
        owner_fingerprint,
        raw_cid,
    ) is False
    assert operator._owner_schema_fingerprint_matches_canonical_cid(
        "sha256:" + ("0" * 64),
        dag_json_cid,
    ) is False
    assert operator._owner_schema_fingerprint_matches_canonical_cid(
        owner_fingerprint.upper(),
        dag_json_cid,
    ) is False
    assert operator._owner_schema_fingerprint_matches_canonical_cid(
        owner_fingerprint,
        dag_json_cid.upper(),
    ) is False


@pytest.mark.parametrize("mismatch", ("wrong_digest", "wrong_codec"))
def test_stopped_recovery_binds_raw_database_schema_cid_to_owner_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mismatch: str,
) -> None:
    from ipfs_accelerate_py.utils.cid_utils import cid_from_sha256_digest

    operator = _operator()
    paths, provenance, _continuity = _stopped_state_continuity_fixture(
        operator,
        tmp_path,
        monkeypatch,
    )
    _unpublish_stopped_continuity(operator, paths, legacy=True)
    provenance_digest = hashlib.sha256(b"test-stopped-continuity-schema").digest()
    database_schema = cid_from_sha256_digest(
        (
            hashlib.sha256(b"wrong-database-schema").digest()
            if mismatch == "wrong_digest"
            else provenance_digest
        ),
        codec="dag-json" if mismatch == "wrong_digest" else "raw",
    )
    monkeypatch.setattr(
        operator,
        "_database_identity",
        lambda _database: {
            "database_uuid": provenance["database_uuid"],
            "schema_fingerprint": database_schema,
        },
    )

    with pytest.raises(
        operator.SuccessorOperatorError,
        match="database identity differs from provenance",
    ):
        operator.stopped_recovery_preflight(tmp_path)

    assert not paths["stopped_state_continuity"].exists()


def _unpublish_stopped_continuity(
    operator: ModuleType,
    paths: dict[str, Path],
    *,
    legacy: bool,
) -> dict[str, object]:
    bound = operator._strict_json(paths["controller_status"])
    unbound = dict(bound)
    unbound.pop("status_cid")
    unbound.pop("stopped_state_continuity_receipt_cid")
    unbound.pop("stopped_state_continuity_status_cid")
    if legacy:
        unbound.pop("stopped_recovery_anchors")
    unbound["status_cid"] = operator._content_id(unbound)
    operator._write_status(paths["controller_status"], unbound)
    paths["stopped_state_continuity"].unlink()
    return unbound


def test_stopped_continuity_authorizes_only_same_generation_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _operator()
    paths, provenance, continuity = _stopped_state_continuity_fixture(
        operator, tmp_path, monkeypatch
    )

    admitted = operator._load_projection_source_continuity(paths, root=tmp_path)

    assert admitted["admission_mode"] == (
        operator.STOPPED_STATE_CONTINUITY_ADMISSION_MODE
    )
    assert admitted["receipt"]["receipt_cid"] == continuity["receipt_cid"]
    assert admitted["receipt"]["restart_authority"] is True
    assert admitted["receipt"]["same_generation_restart_only"] is True
    assert admitted["receipt"]["scheduling_authority"] is False
    assert admitted["receipt"]["completion_authority"] is False
    assert continuity["test_restart_calls"]() == 0

    bound = operator._strict_json(paths["controller_status"])
    anchor = dict(bound)
    anchor.pop("status_cid")
    anchor.pop("stopped_state_continuity_receipt_cid")
    anchor.pop("stopped_state_continuity_status_cid")
    anchor["status_cid"] = operator._content_id(anchor)
    operator._write_status(paths["controller_status"], anchor)
    custody = operator._open_generation_bound_controller_lock(paths)
    handle = custody["lock_handle"]
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        completed = operator._recover_interrupted_stopped_state_continuity(
            paths,
            root=tmp_path,
            lock_custody=custody,
            provenance=provenance,
        )
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        operator._close_generation_bound_controller_lock(custody)
    assert completed["receipt_cid"] == continuity["receipt_cid"]
    assert operator._load_stopped_restart_provenance(
        paths,
        root=tmp_path,
        provenance=provenance,
    ) == provenance


def test_interrupted_clean_stop_recovers_fresh_restart_receipt_under_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _operator()
    paths, provenance, _continuity = _stopped_state_continuity_fixture(
        operator, tmp_path, monkeypatch
    )
    bound = operator._strict_json(paths["controller_status"])
    anchor = dict(bound)
    anchor.pop("status_cid")
    anchor.pop("stopped_state_continuity_receipt_cid")
    anchor.pop("stopped_state_continuity_status_cid")
    anchor["status_cid"] = operator._content_id(anchor)
    operator._write_status(paths["controller_status"], anchor)
    paths["stopped_state_continuity"].unlink()

    custody = operator._open_generation_bound_controller_lock(paths)
    handle = custody["lock_handle"]
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        recovered = operator._recover_interrupted_stopped_state_continuity(
            paths,
            root=tmp_path,
            lock_custody=custody,
            provenance=provenance,
        )
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        operator._close_generation_bound_controller_lock(custody)

    assert recovered is not None
    evidence = recovered["stop_evidence"]
    assert evidence == {
        "mode": operator.STOPPED_STATE_RECOVERED_EVIDENCE_MODE,
        "recovered_at": recovered["issued_at"],
        "source_controller_status_cid": anchor["status_cid"],
        "recovery_preflight_cid": evidence["recovery_preflight_cid"],
        "recovery_authorization_mode": (
            operator.STOPPED_RECOVERY_DURABLE_ANCHOR_MODE
        ),
        "durable_stopped_anchors_cid": anchor[
            "stopped_recovery_anchors"
        ]["anchors_cid"],
        "historical_owner_receipts_reconstructed": False,
    }
    assert evidence["recovery_preflight_cid"]
    assert "owner_checkpoint" not in evidence
    assert "owner_stop" not in evidence
    assert recovered["restart_authority"] is True
    assert recovered["scheduling_authority"] is False

    # Simulate interruption after the immutable receipt link but before the
    # controller-status cross-binding replacement.
    operator._write_status(paths["controller_status"], anchor)
    custody = operator._open_generation_bound_controller_lock(paths)
    handle = custody["lock_handle"]
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        completed = operator._recover_interrupted_stopped_state_continuity(
            paths,
            root=tmp_path,
            lock_custody=custody,
            provenance=provenance,
        )
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        operator._close_generation_bound_controller_lock(custody)
    assert completed == recovered
    rebound = operator._strict_json(paths["controller_status"])
    assert rebound["stopped_state_continuity_receipt_cid"] == recovered[
        "receipt_cid"
    ]
    assert operator._load_stopped_restart_provenance(
        paths,
        root=tmp_path,
        provenance=provenance,
    ) == provenance


def test_anchored_unbound_stop_recovers_across_monotonic_remote_catchup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _operator()
    paths, provenance, _continuity = _stopped_state_continuity_fixture(
        operator,
        tmp_path,
        monkeypatch,
    )
    anchor = _unpublish_stopped_continuity(operator, paths, legacy=False)
    sealed = anchor["stopped_recovery_anchors"]["final_source_continuity"]
    assert isinstance(sealed, dict)
    advanced = {**sealed, "resolved_remote_head": "a" * 40}
    ancestry_checks: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        operator,
        "_observe_candidate_runtime_continuity",
        lambda _root, *, require_resolved_remote: advanced,
    )
    monkeypatch.setattr(
        operator,
        "_git_quiet",
        lambda _root, arguments, *, noun: ancestry_checks.append(
            tuple(arguments)
        ),
    )

    custody = operator._open_generation_bound_controller_lock(paths)
    handle = custody["lock_handle"]
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        recovered = operator._recover_interrupted_stopped_state_continuity(
            paths,
            root=tmp_path,
            lock_custody=custody,
            provenance=provenance,
        )
        repeated = operator._recover_interrupted_stopped_state_continuity(
            paths,
            root=tmp_path,
            lock_custody=custody,
            provenance=provenance,
        )
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        operator._close_generation_bound_controller_lock(custody)

    assert recovered is not None
    assert recovered["final_source_continuity"] == sealed
    assert recovered["final_source_continuity"] != advanced
    assert repeated is None
    assert ancestry_checks
    rebound = operator._strict_json(paths["controller_status"])
    assert rebound["stopped_state_continuity_receipt_cid"] == recovered[
        "receipt_cid"
    ]


def test_receipt_written_stop_completes_across_monotonic_remote_catchup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _operator()
    paths, provenance, _continuity = _stopped_state_continuity_fixture(
        operator,
        tmp_path,
        monkeypatch,
    )
    receipt = operator._strict_json(paths["stopped_state_continuity"])
    bound = operator._strict_json(paths["controller_status"])
    anchor = dict(bound)
    anchor.pop("status_cid")
    anchor.pop("stopped_state_continuity_receipt_cid")
    anchor.pop("stopped_state_continuity_status_cid")
    anchor["status_cid"] = operator._content_id(anchor)
    assert anchor["status_cid"] == receipt["controller_status_cid"]
    operator._write_status(paths["controller_status"], anchor)
    sealed = receipt["final_source_continuity"]
    assert isinstance(sealed, dict)
    advanced = {**sealed, "resolved_remote_head": "a" * 40}
    monkeypatch.setattr(
        operator,
        "_observe_candidate_runtime_continuity",
        lambda _root, *, require_resolved_remote: advanced,
    )
    monkeypatch.setattr(operator, "_git_quiet", lambda *args, **kwargs: None)

    custody = operator._open_generation_bound_controller_lock(paths)
    handle = custody["lock_handle"]
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        completed = operator._recover_interrupted_stopped_state_continuity(
            paths,
            root=tmp_path,
            lock_custody=custody,
            provenance=provenance,
        )
        repeated = operator._recover_interrupted_stopped_state_continuity(
            paths,
            root=tmp_path,
            lock_custody=custody,
            provenance=provenance,
        )
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        operator._close_generation_bound_controller_lock(custody)

    assert completed == receipt
    assert completed["final_source_continuity"] == sealed
    assert repeated is None
    rebound = operator._strict_json(paths["controller_status"])
    assert rebound["stopped_state_continuity_receipt_cid"] == receipt[
        "receipt_cid"
    ]


@pytest.mark.parametrize("rejection", ("rollback", "not_beneath_current"))
def test_anchored_recovery_rejects_nonmonotonic_remote_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rejection: str,
) -> None:
    operator = _operator()
    paths, provenance, _continuity = _stopped_state_continuity_fixture(
        operator,
        tmp_path,
        monkeypatch,
    )
    anchor = _unpublish_stopped_continuity(operator, paths, legacy=False)
    sealed = anchor["stopped_recovery_anchors"]["final_source_continuity"]
    assert isinstance(sealed, dict)
    observed_remote = "0" * 40 if rejection == "rollback" else "a" * 40
    observed = {**sealed, "resolved_remote_head": observed_remote}
    monkeypatch.setattr(
        operator,
        "_observe_candidate_runtime_continuity",
        lambda _root, *, require_resolved_remote: observed,
    )

    def reject_ancestry(
        _root: Path,
        arguments: tuple[str, ...],
        *,
        noun: str,
    ) -> None:
        ancestor, descendant = arguments[-2:]
        if rejection == "rollback" and ancestor == sealed[
            "resolved_remote_head"
        ]:
            raise operator.SuccessorOperatorError("test remote rollback")
        if (
            rejection == "not_beneath_current"
            and ancestor == observed_remote
            and descendant == sealed["current_head"]
        ):
            raise operator.SuccessorOperatorError(
                "test remote is not beneath current"
            )

    monkeypatch.setattr(operator, "_git_quiet", reject_ancestry)
    status_before = paths["controller_status"].read_bytes()
    custody = operator._open_generation_bound_controller_lock(paths)
    handle = custody["lock_handle"]
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(
            operator.SuccessorOperatorError,
            match="test remote rollback|test remote is not beneath current",
        ):
            operator._recover_interrupted_stopped_state_continuity(
                paths,
                root=tmp_path,
                lock_custody=custody,
                provenance=provenance,
            )
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        operator._close_generation_bound_controller_lock(custody)

    assert paths["controller_status"].read_bytes() == status_before
    assert not paths["stopped_state_continuity"].exists()


def test_receipt_publication_signal_gap_keeps_one_recoverable_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _operator()
    paths, provenance, _continuity = _stopped_state_continuity_fixture(
        operator,
        tmp_path,
        monkeypatch,
    )
    anchor = _unpublish_stopped_continuity(operator, paths, legacy=False)
    original_complete = (
        operator._complete_interrupted_stopped_recovery_publication
    )

    class SimulatedSignalGap(RuntimeError):
        pass

    def interrupt_after_receipt(*args: object, **kwargs: object) -> None:
        raise SimulatedSignalGap

    monkeypatch.setattr(
        operator,
        "_complete_interrupted_stopped_recovery_publication",
        interrupt_after_receipt,
    )
    custody = operator._open_generation_bound_controller_lock(paths)
    handle = custody["lock_handle"]
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(SimulatedSignalGap):
            operator._recover_interrupted_stopped_state_continuity(
                paths,
                root=tmp_path,
                lock_custody=custody,
                provenance=provenance,
            )
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        operator._close_generation_bound_controller_lock(custody)

    receipt_metadata = os.lstat(paths["stopped_state_continuity"])
    assert receipt_metadata.st_nlink == 1
    assert operator._strict_json(paths["controller_status"]) == anchor
    assert not tuple(
        paths["stopped_state_continuity"].parent.glob(
            f".{paths['stopped_state_continuity'].name}.*"
        )
    )

    monkeypatch.setattr(
        operator,
        "_complete_interrupted_stopped_recovery_publication",
        original_complete,
    )
    custody = operator._open_generation_bound_controller_lock(paths)
    handle = custody["lock_handle"]
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        completed = operator._recover_interrupted_stopped_state_continuity(
            paths,
            root=tmp_path,
            lock_custody=custody,
            provenance=provenance,
        )
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        operator._close_generation_bound_controller_lock(custody)
    assert completed is not None
    assert operator._strict_json(paths["controller_status"])[
        "stopped_state_continuity_receipt_cid"
    ] == completed["receipt_cid"]


def test_interrupted_receipt_publication_rejects_extra_field_before_status_bind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _operator()
    paths, provenance, _continuity = _stopped_state_continuity_fixture(
        operator, tmp_path, monkeypatch
    )
    unbound = _unpublish_stopped_continuity(
        operator,
        paths,
        legacy=False,
    )
    # Recreate the receipt-written/status-unbound crash phase, but make the
    # receipt independently content-addressed with a foreign field.
    receipt = dict(_continuity)
    receipt.pop("test_restart_calls")
    receipt.pop("receipt_cid")
    receipt["foreign_authority_field"] = False
    receipt["receipt_cid"] = operator._content_id(receipt)
    operator._atomic_json(
        paths["stopped_state_continuity"],
        receipt,
        replace=False,
    )
    status_before = paths["controller_status"].read_bytes()

    custody = operator._open_generation_bound_controller_lock(paths)
    handle = custody["lock_handle"]
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(
            operator.SuccessorOperatorError,
            match="continuity receipt shape differs",
        ):
            operator._recover_interrupted_stopped_state_continuity(
                paths,
                root=tmp_path,
                lock_custody=custody,
                provenance=provenance,
            )
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        operator._close_generation_bound_controller_lock(custody)

    assert paths["controller_status"].read_bytes() == status_before
    assert operator._strict_json(paths["controller_status"]) == unbound


def test_legacy_stop_requires_deterministic_reviewed_preflight_before_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _operator()
    paths, provenance, _continuity = _stopped_state_continuity_fixture(
        operator, tmp_path, monkeypatch
    )
    legacy_status = _unpublish_stopped_continuity(
        operator,
        paths,
        legacy=True,
    )
    generation = paths["controller_lock"].parent

    def generation_inventory() -> tuple[tuple[object, ...], ...]:
        observed: list[tuple[object, ...]] = []
        for item in sorted(generation.rglob("*")):
            metadata = os.lstat(item)
            relative = item.relative_to(generation).as_posix()
            if item.is_file():
                observed.append(
                    (
                        relative,
                        "file",
                        metadata.st_mode & 0o777,
                        hashlib.sha256(item.read_bytes()).hexdigest(),
                    )
                )
            else:
                observed.append(
                    (relative, "directory", metadata.st_mode & 0o777)
                )
        return tuple(observed)

    inventory_before = generation_inventory()
    lock_metadata_before = os.stat(paths["controller_lock"])
    exact_lock_stat_before = (
        lock_metadata_before.st_dev,
        lock_metadata_before.st_ino,
        lock_metadata_before.st_mode,
        lock_metadata_before.st_nlink,
        lock_metadata_before.st_size,
        lock_metadata_before.st_mtime_ns,
        lock_metadata_before.st_ctime_ns,
    )
    fixture_verify_profile = operator._verify_profile
    read_only_profile_paths: list[Path] = []

    def verify_profile_read_only(
        path: Path,
        *,
        sealed_descriptor: int | None = None,
        read_only: bool = False,
    ) -> dict[str, object]:
        assert read_only is True
        assert sealed_descriptor is None
        assert str(path).startswith("/proc/self/fd/")
        read_only_profile_paths.append(path)
        return fixture_verify_profile(path, read_only=read_only)

    monkeypatch.setattr(operator, "_verify_profile", verify_profile_read_only)
    status_before = paths["controller_status"].read_bytes()
    owner_before = (
        paths["owner_state"] / "quack-state-server.status.json"
    ).read_bytes()
    databases_before = {
        name: database.read_bytes()
        for name, database in operator._successor_state_databases(paths).items()
    }

    first = operator.stopped_recovery_preflight(tmp_path)
    second = operator.stopped_recovery_preflight(tmp_path)

    assert first["schema"] == operator.STOPPED_RECOVERY_PREFLIGHT_SCHEMA
    assert first["preflight_cid"] == second["preflight_cid"]
    assert first["reviewed_pins"] == second["reviewed_pins"]
    assert first["legacy_explicit_review_required"] is True
    assert first["generic_recovery_authorized"] is False
    assert first["restart_authority"] is False
    assert first["reviewed_pins"]["controller_status_cid"] == (
        legacy_status["status_cid"]
    )
    assert first["reviewed_pins"]["controller_status"] == legacy_status
    assert first["reviewed_pins"]["source_provenance_cid"] == (
        provenance["receipt_cid"]
    )
    assert first["reviewed_pins"]["durable_stopped_anchors_cid"] == ""
    assert first["operation"] == operator.STOPPED_RECOVERY_OPERATION
    assert first["preflight_cid"] == operator._content_id(
        {
            "schema": operator.STOPPED_RECOVERY_PREFLIGHT_SCHEMA,
            "operation": operator.STOPPED_RECOVERY_OPERATION,
            "reviewed_pins": first["reviewed_pins"],
        }
    )
    assert read_only_profile_paths
    assert not paths["stopped_state_continuity"].exists()
    assert paths["controller_status"].read_bytes() == status_before
    assert (
        paths["owner_state"] / "quack-state-server.status.json"
    ).read_bytes() == owner_before
    assert {
        name: database.read_bytes()
        for name, database in operator._successor_state_databases(paths).items()
    } == databases_before
    assert generation_inventory() == inventory_before
    lock_metadata_after = os.stat(paths["controller_lock"])
    assert (
        lock_metadata_after.st_dev,
        lock_metadata_after.st_ino,
        lock_metadata_after.st_mode,
        lock_metadata_after.st_nlink,
        lock_metadata_after.st_size,
        lock_metadata_after.st_mtime_ns,
        lock_metadata_after.st_ctime_ns,
    ) == exact_lock_stat_before

    custody = operator._open_generation_bound_controller_lock(paths)
    handle = custody["lock_handle"]
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(
            operator.SuccessorOperatorError,
            match="legacy stopped status is not self-anchored",
        ):
            operator._recover_interrupted_stopped_state_continuity(
                paths,
                root=tmp_path,
                lock_custody=custody,
                provenance=provenance,
            )
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        operator._close_generation_bound_controller_lock(custody)
    assert not paths["stopped_state_continuity"].exists()
    assert paths["controller_status"].read_bytes() == status_before
    assert generation_inventory() == inventory_before

    with pytest.raises(
        operator.SuccessorOperatorError,
        match="reviewed stopped recovery preflight CID differs",
    ):
        operator.recover_stopped_continuity(
            tmp_path,
            reviewed_preflight_cid="b" + ("z" * 60),
        )
    assert not paths["stopped_state_continuity"].exists()
    assert paths["controller_status"].read_bytes() == status_before
    assert generation_inventory() == inventory_before

    result = operator.recover_stopped_continuity(
        tmp_path,
        reviewed_preflight_cid=str(first["preflight_cid"]),
    )
    receipt = operator._strict_json(paths["stopped_state_continuity"])
    evidence = receipt["stop_evidence"]
    assert result["schema"] == operator.STOPPED_RECOVERY_RESULT_SCHEMA
    assert result["recovered"] is True
    assert result["preflight_cid"] == first["preflight_cid"]
    assert result["stopped_state_continuity_receipt_cid"] == (
        receipt["receipt_cid"]
    )
    assert evidence["recovery_authorization_mode"] == (
        operator.STOPPED_RECOVERY_REVIEWED_LEGACY_MODE
    )
    assert evidence["recovery_preflight_cid"] == first["preflight_cid"]
    assert evidence["durable_stopped_anchors_cid"] == ""
    monkeypatch.setattr(operator, "_verify_profile", fixture_verify_profile)
    assert operator._load_stopped_restart_provenance(
        paths,
        root=tmp_path,
        provenance=provenance,
    ) == provenance


@pytest.mark.parametrize("lock_state", ("absent", "unsafe_mode"))
def test_read_only_stopped_preflight_never_creates_or_repairs_controller_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lock_state: str,
) -> None:
    operator = _operator()
    paths, _provenance, _continuity = _stopped_state_continuity_fixture(
        operator,
        tmp_path,
        monkeypatch,
    )
    _unpublish_stopped_continuity(operator, paths, legacy=True)
    if lock_state == "absent":
        paths["controller_lock"].unlink()
        metadata_before = None
    else:
        paths["controller_lock"].chmod(0o640)
        metadata_before = os.stat(paths["controller_lock"])

    with pytest.raises(
        operator.SuccessorOperatorError,
        match="existing controller lock is unavailable|controller lock custody is unsafe",
    ):
        operator.stopped_recovery_preflight(tmp_path)

    if metadata_before is None:
        assert not os.path.lexists(paths["controller_lock"])
    else:
        metadata_after = os.stat(paths["controller_lock"])
        assert (
            metadata_after.st_dev,
            metadata_after.st_ino,
            metadata_after.st_mode,
            metadata_after.st_nlink,
            metadata_after.st_size,
            metadata_after.st_mtime_ns,
            metadata_after.st_ctime_ns,
        ) == (
            metadata_before.st_dev,
            metadata_before.st_ino,
            metadata_before.st_mode,
            metadata_before.st_nlink,
            metadata_before.st_size,
            metadata_before.st_mtime_ns,
            metadata_before.st_ctime_ns,
        )


def test_stopped_preflight_rejects_provenance_change_at_final_reread(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _operator()
    paths, provenance, _continuity = _stopped_state_continuity_fixture(
        operator,
        tmp_path,
        monkeypatch,
    )
    _unpublish_stopped_continuity(operator, paths, legacy=True)
    status_before = paths["controller_status"].read_bytes()
    calls = 0

    def changing_provenance(
        _paths: dict[str, Path],
        **_kwargs: object,
    ) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls >= 3:
            return {**provenance, "receipt_cid": "b" + ("z" * 60)}
        return provenance

    monkeypatch.setattr(
        operator,
        "_load_lgcvf_live_raw_provenance_receipt",
        changing_provenance,
    )
    with pytest.raises(
        operator.SuccessorOperatorError,
        match="evidence changed during preflight",
    ):
        operator.stopped_recovery_preflight(tmp_path)

    assert calls >= 3
    assert paths["controller_status"].read_bytes() == status_before
    assert not paths["stopped_state_continuity"].exists()


def test_legacy_reviewed_preflight_cid_expires_when_remote_observation_moves(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _operator()
    paths, _provenance, sealed = _stopped_state_continuity_fixture(
        operator,
        tmp_path,
        monkeypatch,
    )
    _unpublish_stopped_continuity(operator, paths, legacy=True)
    first = operator.stopped_recovery_preflight(tmp_path)
    original_source = sealed["final_source_continuity"]
    assert isinstance(original_source, dict)
    advanced = {**original_source, "resolved_remote_head": "a" * 40}
    monkeypatch.setattr(
        operator,
        "_observe_candidate_runtime_continuity",
        lambda _root, *, require_resolved_remote: advanced,
    )

    with pytest.raises(
        operator.SuccessorOperatorError,
        match="reviewed stopped recovery preflight CID differs",
    ):
        operator.recover_stopped_continuity(
            tmp_path,
            reviewed_preflight_cid=str(first["preflight_cid"]),
        )

    assert not paths["stopped_state_continuity"].exists()


@pytest.mark.parametrize("tamper", ("database", "owner_status", "source"))
def test_anchored_stop_recovery_rejects_post_stop_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    operator = _operator()
    paths, provenance, _continuity = _stopped_state_continuity_fixture(
        operator, tmp_path, monkeypatch
    )
    _unpublish_stopped_continuity(operator, paths, legacy=False)
    if tamper == "database":
        with paths["successor_database"].open("ab") as handle:
            handle.write(b"post-stop-tamper\n")
    elif tamper == "owner_status":
        owner_status = paths["owner_state"] / "quack-state-server.status.json"
        changed_owner_status = operator._strict_json(
            owner_status,
            verify_content_identity=False,
        )
        changed_owner_status["test_tamper"] = True
        operator._atomic_json(owner_status, changed_owner_status, replace=True)
    else:
        original = operator._observe_candidate_runtime_continuity(
            tmp_path,
            require_resolved_remote=False,
        )
        changed = {**original, "current_head": "9" * 40}
        monkeypatch.setattr(
            operator,
            "_observe_candidate_runtime_continuity",
            lambda _root, *, require_resolved_remote: changed,
        )

    custody = operator._open_generation_bound_controller_lock(paths)
    handle = custody["lock_handle"]
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(
            operator.SuccessorOperatorError,
            match=(
                "durable stopped recovery anchors differ|"
                "stopped-state final source continuity differs"
            ),
        ):
            operator._recover_interrupted_stopped_state_continuity(
                paths,
                root=tmp_path,
                lock_custody=custody,
                provenance=provenance,
            )
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        operator._close_generation_bound_controller_lock(custody)
    assert not paths["stopped_state_continuity"].exists()


@pytest.mark.parametrize(
    "malformed",
    ("zero_births", "aliased_scheduler", "boolean_returncode"),
)
def test_stopped_recovery_rejects_nonexistent_or_aliased_process_births(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    malformed: str,
) -> None:
    operator = _operator()
    paths, _provenance, _continuity = _stopped_state_continuity_fixture(
        operator, tmp_path, monkeypatch
    )
    status = _unpublish_stopped_continuity(operator, paths, legacy=True)
    status.pop("status_cid")
    if malformed == "zero_births":
        status["controller_birth"] = {}
        status["scheduler_birth"] = {}
        status["owner_identity"] = {
            **status["owner_identity"],
            "process_birth": {},
        }
    elif malformed == "aliased_scheduler":
        status["scheduler_birth"] = status["controller_birth"]
    else:
        status["scheduler_returncode"] = False
    status["status_cid"] = operator._content_id(status)
    operator._write_status(paths["controller_status"], status)

    with pytest.raises(
        operator.SuccessorOperatorError,
        match=(
            "process birth binding is malformed|birth relation differs|"
            "unbound stopped-state controller status differs"
        ),
    ):
        operator.stopped_recovery_preflight(tmp_path)
    assert not paths["stopped_state_continuity"].exists()


def _failed_start_legacy_fixture(
    operator: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, Path], dict[str, object], dict[str, object]]:
    paths, provenance, prior = _stopped_state_continuity_fixture(
        operator,
        tmp_path,
        monkeypatch,
    )
    bound = operator._strict_json(paths["controller_status"])
    assert operator._claim_stopped_state_restart_admission(
        paths,
        expected_restart=True,
        expected_receipt_cid=str(prior["receipt_cid"]),
        expected_controller_status_cid=str(bound["status_cid"]),
    ) is True
    failed = operator._status_payload(
        lifecycle="stopped",
        controller_birth=bound["controller_birth"],
        provenance_cid=str(provenance["receipt_cid"]),
        owner_identity=bound["owner_identity"],
        scheduler_birth=bound["scheduler_birth"],
        scheduler_returncode=-15,
        error=operator.FAILED_START_STATUS_ERROR,
        projection_root=paths["projection_root"],
    )
    operator._write_status(paths["controller_status"], failed)
    return paths, provenance, prior


def test_reviewed_legacy_failed_start_publishes_new_current_byte_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _operator()
    paths, provenance, prior = _failed_start_legacy_fixture(
        operator,
        tmp_path,
        monkeypatch,
    )
    admission_before = paths["stopped_state_restart_admission"].read_bytes()
    status_before = paths["controller_status"].read_bytes()
    database_before = {
        name: path.read_bytes()
        for name, path in operator._successor_state_databases(paths).items()
    }

    first = operator.failed_start_recovery_preflight(tmp_path)
    second = operator.failed_start_recovery_preflight(tmp_path)

    assert first["preflight_cid"] == second["preflight_cid"]
    assert first["reviewed_pins"] == second["reviewed_pins"]
    assert first["legacy_explicit_review_required"] is True
    assert first["restart_authority"] is False
    assert first["reviewed_pins"]["failed_start_reason"] == (
        operator.FAILED_START_REASON_LEGACY_UNCLASSIFIED
    )
    assert paths["controller_status"].read_bytes() == status_before
    assert paths["stopped_state_restart_admission"].read_bytes() == (
        admission_before
    )
    assert {
        name: path.read_bytes()
        for name, path in operator._successor_state_databases(paths).items()
    } == database_before

    result = operator.recover_failed_start_continuity(
        tmp_path,
        reviewed_preflight_cid=str(first["preflight_cid"]),
    )

    receipt = operator._strict_json(paths["stopped_state_continuity"])
    status = operator._strict_json(paths["controller_status"])
    anchors = status["failed_start_recovery_anchors"]
    archive = Path(
        anchors["superseded_restart_admission"]["archive_path"]
    )
    assert result["schema"] == operator.FAILED_START_RECOVERY_RESULT_SCHEMA
    assert receipt["receipt_cid"] != prior["receipt_cid"]
    assert receipt["admission_mode"] == (
        operator.FAILED_START_CONTINUITY_ADMISSION_MODE
    )
    assert receipt["requires_stopped_checkpoint"] is False
    assert receipt["stop_evidence"]["mode"] == (
        operator.FAILED_START_REVIEWED_EVIDENCE_MODE
    )
    assert receipt["stop_evidence"]["historical_owner_receipts_reconstructed"] is False
    assert not paths["stopped_state_restart_admission"].exists()
    assert archive.read_bytes() == admission_before
    admitted = operator._load_stopped_restart_admission(
        paths,
        root=tmp_path,
        provenance=provenance,
    )
    assert admitted["receipt"] == receipt


def test_trusted_failed_start_anchors_replay_after_interrupted_finally(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _operator()
    paths, provenance, prior = _failed_start_legacy_fixture(
        operator,
        tmp_path,
        monkeypatch,
    )
    failed = operator._strict_json(paths["controller_status"])
    io_paths = operator._stopped_recovery_io_paths(paths, None)
    anchors = operator._capture_failed_start_recovery_anchors(
        paths,
        root=tmp_path,
        failed_status=failed,
        provenance=provenance,
        failed_start_reason=operator.FAILED_START_REASON_BOOTSTRAP_TIMEOUT,
        owner_stop={
            "stopped": True,
            "server_id": failed["owner_identity"]["server_id"],
            "at": "2026-08-26T12:37:41Z",
        },
        io_paths=io_paths,
    )
    anchored = operator._bind_failed_start_recovery_anchors_status(
        failed,
        anchors,
    )
    operator._write_status(paths["controller_status"], anchored)

    custody = operator._open_generation_bound_controller_lock(paths)
    handle = custody["lock_handle"]
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt = operator._recover_interrupted_failed_start_continuity(
            paths,
            root=tmp_path,
            lock_custody=custody,
            provenance=provenance,
        )
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        operator._close_generation_bound_controller_lock(custody)

    assert receipt is not None
    assert receipt["receipt_cid"] != prior["receipt_cid"]
    assert receipt["stop_evidence"]["mode"] == (
        operator.FAILED_START_LIVE_OWNER_EVIDENCE_MODE
    )
    assert receipt["stop_evidence"]["failed_start_reason"] == (
        operator.FAILED_START_REASON_BOOTSTRAP_TIMEOUT
    )
    assert operator._load_stopped_restart_admission(
        paths,
        root=tmp_path,
        provenance=provenance,
    )["receipt"] == receipt


@pytest.mark.parametrize(
    "tamper",
    ("database", "owner_status", "source", "wal", "birth"),
)
def test_reviewed_failed_start_recovery_rejects_changed_exact_pins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    operator = _operator()
    paths, _provenance, _prior = _failed_start_legacy_fixture(
        operator,
        tmp_path,
        monkeypatch,
    )
    preflight = operator.failed_start_recovery_preflight(tmp_path)
    if tamper == "database":
        with paths["successor_database"].open("ab") as handle:
            handle.write(b"failed-start-post-review-tamper\n")
    elif tamper == "owner_status":
        owner_path = (
            paths["owner_state"] / "quack-state-server.status.json"
        )
        owner = operator._strict_json(
            owner_path,
            verify_content_identity=False,
        )
        owner["reviewed_tamper"] = True
        operator._atomic_json(owner_path, owner, replace=True)
    elif tamper == "source":
        observed = operator._observe_candidate_runtime_continuity(
            tmp_path,
            require_resolved_remote=False,
        )
        monkeypatch.setattr(
            operator,
            "_observe_candidate_runtime_continuity",
            lambda _root, *, require_resolved_remote: {
                **observed,
                "current_head": "9" * 40,
            },
        )
    elif tamper == "wal":
        paths["successor_database"].with_name(
            paths["successor_database"].name + ".wal"
        ).write_bytes(b"live-wal")
    else:
        status = operator._strict_json(paths["controller_status"])
        status.pop("status_cid")
        status["scheduler_birth"] = {}
        status["status_cid"] = operator._content_id(status)
        operator._write_status(paths["controller_status"], status)

    with pytest.raises(operator.SuccessorOperatorError):
        operator.recover_failed_start_continuity(
            tmp_path,
            reviewed_preflight_cid=str(preflight["preflight_cid"]),
        )
    assert not paths["stopped_state_continuity"].exists()
    assert paths["stopped_state_restart_admission"].exists()


def _published_failed_start_source_maintenance_fixture(
    operator: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    dict[str, Path],
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    paths, provenance, _prior = _failed_start_legacy_fixture(
        operator,
        tmp_path,
        monkeypatch,
    )
    recovery = operator.failed_start_recovery_preflight(tmp_path)
    operator.recover_failed_start_continuity(
        tmp_path,
        reviewed_preflight_cid=str(recovery["preflight_cid"]),
    )
    published = operator._strict_json(paths["stopped_state_continuity"])
    old_source = published["final_source_continuity"]
    assert isinstance(old_source, dict)
    descendant = {
        **old_source,
        "current_head": "9" * 40,
        "current_tree": "8" * 40,
        "datasets_head": "7" * 40,
        "datasets_tree": "6" * 40,
        "superproject_runtime_inventory": {
            "tracked_object_count": 2,
            "tracked_inventory_root": "sha256:" + ("5" * 64),
        },
        "datasets_runtime_inventory": {
            "tracked_object_count": 2,
            "tracked_inventory_root": "sha256:" + ("4" * 64),
        },
    }
    monkeypatch.setattr(
        operator,
        "_observe_candidate_runtime_continuity",
        lambda _root, *, require_resolved_remote: descendant,
    )
    monkeypatch.setattr(operator, "_git_quiet", lambda *args, **kwargs: None)
    return paths, provenance, published, descendant


def _abandoned_owner_fixture(
    operator: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, Path], dict[str, object], dict[str, object]]:
    from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
        ProcessBirthIdentity,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
        OwnerMarker,
    )

    paths, provenance, receipt = _stopped_state_continuity_fixture(
        operator,
        tmp_path,
        monkeypatch,
    )
    owner_status_path = (
        paths["owner_state"] / "quack-state-server.status.json"
    )
    owner_status = operator._strict_json(
        owner_status_path,
        verify_content_identity=False,
    )
    stale_birth = ProcessBirthIdentity(
        pid=2_000_000_003,
        start_time_ticks=19,
        boot_id="test-stopped-continuity",
        parent_pid=1,
    )
    stale_identity = {
        **owner_status["identity"],
        "server_id": "server:test-abandoned-owner",
        "process_birth": stale_birth.to_dict(),
        "status": "ready",
    }
    owner_status["lifecycle"] = "ready"
    owner_status["identity"] = stale_identity
    operator._atomic_json(owner_status_path, owner_status, replace=True)
    marker = OwnerMarker(
        server_id=str(stale_identity["server_id"]),
        process_birth=stale_birth,
        database_path=str(paths["successor_database"]),
        started_at="2026-08-26T12:00:00Z",
        fence_token="test-abandoned-owner-fence",
        generation=2,
    )
    marker_path = paths["successor_database"].with_name(
        ".control.duckdb.state-owner.json"
    )
    marker_path.write_text(
        json.dumps(
            marker.to_dict(),
            sort_keys=True,
            indent=2,
            separators=(",", ": "),
        )
        + "\n",
        encoding="utf-8",
    )
    marker_path.chmod(0o600)
    owner_lock = paths["successor_database"].with_name(
        ".control.duckdb.state-owner.lock"
    )
    owner_lock.touch(mode=0o600)
    owner_lock.chmod(0o600)
    wal = paths["successor_database"].with_name(
        paths["successor_database"].name + ".wal"
    )
    wal.write_bytes(b"test-abandoned-owner-wal")
    # DuckDB creates WALs using the process umask (commonly 0664).  The
    # generation directory itself is private and descriptor-bound.
    wal.chmod(0o664)
    return paths, provenance, receipt


def test_abandoned_owner_preflight_is_stable_and_non_mutating(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _operator()
    paths, _provenance, receipt = _abandoned_owner_fixture(
        operator,
        tmp_path,
        monkeypatch,
    )
    generation = paths["controller_lock"].parent
    before = {
        path.name: path.read_bytes()
        for path in generation.rglob("*")
        if path.is_file()
    }

    first = operator.abandoned_owner_recovery_preflight(tmp_path)
    second = operator.abandoned_owner_recovery_preflight(tmp_path)

    assert first["schema"] == (
        operator.ABANDONED_OWNER_RECOVERY_PREFLIGHT_SCHEMA
    )
    assert first["preflight_cid"] == second["preflight_cid"]
    assert first["reviewed_pins"] == second["reviewed_pins"]
    assert first["automatic_same_source_recovery"] is True
    assert first["reviewed_pins"][
        "stopped_state_continuity_receipt_cid"
    ] == receipt["receipt_cid"]
    assert first["reviewed_pins"]["wal_surfaces"].keys() == {"control"}
    assert first["restart_authority"] is False
    assert {
        path.name: path.read_bytes()
        for path in generation.rglob("*")
        if path.is_file()
    } == before


def test_exact_source_abandoned_owner_is_recovered_before_normal_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _operator()
    paths, _provenance, _receipt = _abandoned_owner_fixture(
        operator,
        tmp_path,
        monkeypatch,
    )
    reviewed = operator.abandoned_owner_recovery_preflight(tmp_path)
    calls: list[dict[str, object]] = []

    def recover(
        observed_paths: dict[str, Path],
        **kwargs: object,
    ) -> dict[str, object]:
        assert observed_paths == paths
        calls.append(dict(kwargs))
        return {"recovered": True}

    monkeypatch.setattr(
        operator,
        "_recover_abandoned_owner_continuity_locked",
        recover,
    )
    custody = operator._open_generation_bound_controller_lock(paths)
    handle = custody["lock_handle"]
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = operator._automatically_recover_abandoned_owner_locked(
            paths,
            root=tmp_path,
            lock_custody=custody,
        )
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        operator._close_generation_bound_controller_lock(custody)

    assert result == {"recovered": True}
    assert calls == [
        {
            "root": tmp_path,
            "lock_custody": custody,
            "reviewed_preflight_cid": reviewed["preflight_cid"],
            "_automatic": True,
        }
    ]


def test_reviewed_abandoned_owner_replays_wal_and_publishes_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ipfs_accelerate_py.agent_supervisor.runtime.configured_board_scheduler as scheduler_module
    import ipfs_accelerate_py.agent_supervisor.runtime.process_security as security_module
    import ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server as server_module
    from ipfs_accelerate_py import llm_router
    from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
        current_process_birth,
    )

    operator = _operator()
    paths, provenance, prior = _abandoned_owner_fixture(
        operator,
        tmp_path,
        monkeypatch,
    )
    preflight = operator.abandoned_owner_recovery_preflight(tmp_path)
    prior_status = operator._strict_json(paths["controller_status"])
    prior_owner = prior_status["owner_identity"]
    process_birth = current_process_birth().to_dict()

    class FakeIdentity:
        server_id = "server:test-recovery-owner"
        listen_uri = "quack:127.0.0.1:24701"
        store_id = operator.SUCCESSOR_DATABASE_RELATIVE.as_posix()
        database_uuid = provenance["database_uuid"]
        schema_fingerprint = prior_owner["schema_fingerprint"]
        secret_handle = operator.SECRET_HANDLE

        def to_dict(self) -> dict[str, object]:
            return {
                **prior_owner,
                "server_id": self.server_id,
                "listen_uri": self.listen_uri,
                "process_birth": process_birth,
                "status": "ready",
            }

    identity = FakeIdentity()

    class FakeVault:
        def resolve(self, _handle: str) -> str:
            return "fixture-recovery-token-never-persisted"

    class FakeServer:
        _vault = FakeVault()

        def typed_command_socket_path(self) -> Path:
            return paths["owner_socket"]

        def start(self) -> FakeIdentity:
            marker = paths["successor_database"].with_name(
                ".control.duckdb.state-owner.json"
            )
            marker.unlink()
            return identity

        def status(self) -> dict[str, object]:
            return {"legacy_board_unstall_enabled": False}

        def checkpoint(self) -> dict[str, object]:
            wal = paths["successor_database"].with_name(
                paths["successor_database"].name + ".wal"
            )
            wal.unlink()
            return {
                "checkpointed": True,
                "server_id": identity.server_id,
                "database_path": str(paths["successor_database"]),
                "at": "2026-08-26T12:01:00Z",
            }

        def stop(self) -> dict[str, object]:
            operator._atomic_json(
                paths["owner_state"] / "quack-state-server.status.json",
                {
                    "schema": operator.QUACK_STATE_SERVER_STATUS_SCHEMA,
                    "lifecycle": "stopped",
                    "database_path": str(paths["successor_database"]),
                    "state_dir": str(paths["owner_state"]),
                    "store_id": operator.SUCCESSOR_DATABASE_RELATIVE.as_posix(),
                    "secret_handle": operator.SECRET_HANDLE,
                    "owner_marker_path": str(
                        paths["successor_database"].with_name(
                            ".control.duckdb.state-owner.json"
                        )
                    ),
                    "identity": {**identity.to_dict(), "status": "stopped"},
                },
                replace=True,
            )
            return {
                "stopped": True,
                "server_id": identity.server_id,
                "at": "2026-08-26T12:01:01Z",
            }

    program = SimpleNamespace(
        quack_endpoint=identity.listen_uri,
        store_id=identity.store_id,
        endpoint_secret_handle=identity.secret_handle,
    )
    monkeypatch.setattr(
        operator,
        "_prepare_lgcvf_configured_board_live_launch",
        lambda **_kwargs: {
            "continuity": preflight["reviewed_pins"]["source_continuity"],
            "launch_home": str(tmp_path / "recovery-home"),
            "native_launch": object(),
            "board": object(),
            "program": program,
            "host": "127.0.0.1",
            "port": 24701,
        },
    )
    monkeypatch.setattr(
        operator,
        "_close_lgcvf_configured_board_live_launch",
        lambda _launch: None,
    )
    monkeypatch.setattr(operator, "_prepare_private_owner_socket", lambda _path: None)
    monkeypatch.setattr(llm_router, "preload_agent_supervisor_native_dependency", lambda _launch: None)
    monkeypatch.setattr(
        scheduler_module,
        "configured_board_launch_plan",
        lambda *_args, **_kwargs: {
            "environment": {operator.DATABASE_PROGRAM_JSON_ENV: "{}"}
        },
    )
    monkeypatch.setattr(
        security_module,
        "establish_state_authority_process_boundary",
        lambda: None,
    )
    monkeypatch.setattr(server_module, "build_server", lambda **_kwargs: FakeServer())
    for name in tuple(os.environ):
        if name.startswith("LD_") or name == "GLIBC_TUNABLES":
            monkeypatch.delenv(name)

    result = operator.recover_abandoned_owner_continuity(
        tmp_path,
        reviewed_preflight_cid=str(preflight["preflight_cid"]),
    )

    receipt = operator._strict_json(paths["stopped_state_continuity"])
    status = operator._strict_json(paths["controller_status"])
    anchors = status["failed_start_recovery_anchors"]
    assert result["schema"] == operator.ABANDONED_OWNER_RECOVERY_RESULT_SCHEMA
    assert result["restart_authority"] is True
    assert receipt["receipt_cid"] != prior["receipt_cid"]
    assert receipt["final_source_continuity"] == (
        preflight["reviewed_pins"]["source_continuity"]
    )
    assert receipt["stop_evidence"]["failed_start_reason"] == (
        operator.FAILED_START_REASON_ABANDONED_OWNER_RECOVERED
    )
    assert status["abandoned_owner_recovery"]["scheduling_attempted"] is False
    expected_prior = {
        key: value
        for key, value in prior.items()
        if key != "test_restart_calls"
    }
    assert anchors["superseded_restart_admission"]["receipt"] == expected_prior
    assert not paths["stopped_state_restart_admission"].exists()
    assert not paths["successor_database"].with_name(
        ".control.duckdb.state-owner.json"
    ).exists()
    assert not paths["successor_database"].with_name(
        paths["successor_database"].name + ".wal"
    ).exists()


def test_failed_start_source_maintenance_reseals_reviewed_descendant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _operator()
    paths, provenance, published, descendant = (
        _published_failed_start_source_maintenance_fixture(
            operator,
            tmp_path,
            monkeypatch,
        )
    )
    receipt_before = paths["stopped_state_continuity"].read_bytes()
    status_before = paths["controller_status"].read_bytes()
    databases_before = {
        name: path.read_bytes()
        for name, path in operator._successor_state_databases(paths).items()
    }
    inventory_before = sorted(path.name for path in paths["controller_lock"].parent.iterdir())

    first = operator.failed_start_source_maintenance_preflight(tmp_path)
    second = operator.failed_start_source_maintenance_preflight(tmp_path)

    assert first["schema"] == (
        operator.FAILED_START_SOURCE_MAINTENANCE_PREFLIGHT_SCHEMA
    )
    assert first["preflight_cid"] == second["preflight_cid"]
    assert first["reviewed_pins"] == second["reviewed_pins"]
    assert first["reviewed_pins"]["source_continuity"] == descendant
    assert first["reviewed_pins"]["controller_status"].get(
        "failed_start_recovery_anchors"
    ) is None
    assert paths["stopped_state_continuity"].read_bytes() == receipt_before
    assert paths["controller_status"].read_bytes() == status_before
    assert sorted(path.name for path in paths["controller_lock"].parent.iterdir()) == (
        inventory_before
    )
    assert {
        name: path.read_bytes()
        for name, path in operator._successor_state_databases(paths).items()
    } == databases_before

    result = operator.reseal_failed_start_source_maintenance(
        tmp_path,
        reviewed_preflight_cid=str(first["preflight_cid"]),
    )

    receipt = operator._strict_json(paths["stopped_state_continuity"])
    status = operator._strict_json(paths["controller_status"])
    superseded = status["failed_start_recovery_anchors"][
        "superseded_restart_admission"
    ]
    archive = Path(superseded["archive_path"])
    assert result["schema"] == (
        operator.FAILED_START_SOURCE_MAINTENANCE_RESULT_SCHEMA
    )
    assert result["repeated"] is False
    assert result[
        "superseded_stopped_state_continuity_receipt_cid"
    ] == published["receipt_cid"]
    assert receipt["receipt_cid"] != published["receipt_cid"]
    assert receipt["final_source_continuity"] == descendant
    assert receipt["stop_evidence"]["recovery_preflight_cid"] == (
        first["preflight_cid"]
    )
    assert superseded["receipt"] == published
    assert archive.read_bytes() == receipt_before
    assert not paths["stopped_state_restart_admission"].exists()
    assert operator._load_stopped_restart_admission(
        paths,
        root=tmp_path,
        provenance=provenance,
    )["receipt"] == receipt
    caught_up = {
        **descendant,
        "resolved_remote_head": descendant["current_head"],
    }
    monkeypatch.setattr(
        operator,
        "_observe_candidate_runtime_continuity",
        lambda _root, *, require_resolved_remote: caught_up,
    )
    repeated = operator.reseal_failed_start_source_maintenance(
        tmp_path,
        reviewed_preflight_cid=str(first["preflight_cid"]),
    )
    assert repeated["repeated"] is True


def test_failed_start_source_maintenance_wrong_cid_is_non_mutating(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _operator()
    paths, _provenance, _published, _descendant = (
        _published_failed_start_source_maintenance_fixture(
            operator,
            tmp_path,
            monkeypatch,
        )
    )
    preflight = operator.failed_start_source_maintenance_preflight(tmp_path)
    receipt_before = paths["stopped_state_continuity"].read_bytes()
    status_before = paths["controller_status"].read_bytes()
    inventory_before = sorted(path.name for path in paths["controller_lock"].parent.iterdir())

    with pytest.raises(
        operator.SuccessorOperatorError,
        match="maintenance preflight CID differs",
    ):
        operator.reseal_failed_start_source_maintenance(
            tmp_path,
            reviewed_preflight_cid=str(preflight["preflight_cid"]) + "wrong",
        )

    assert paths["stopped_state_continuity"].read_bytes() == receipt_before
    assert paths["controller_status"].read_bytes() == status_before
    assert not paths["stopped_state_restart_admission"].exists()
    assert sorted(path.name for path in paths["controller_lock"].parent.iterdir()) == (
        inventory_before
    )


@pytest.mark.parametrize("tamper", ("divergence", "database", "status"))
def test_failed_start_source_maintenance_rejects_divergence_and_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    operator = _operator()
    paths, _provenance, _published, _descendant = (
        _published_failed_start_source_maintenance_fixture(
            operator,
            tmp_path,
            monkeypatch,
        )
    )
    if tamper == "divergence":
        def reject_divergence(
            _root: Path,
            _arguments: object,
            *,
            noun: str,
        ) -> None:
            if noun == "failed-start source maintenance ancestry":
                raise operator.SuccessorOperatorError(
                    "failed-start source maintenance ancestry failed"
                )

        monkeypatch.setattr(operator, "_git_quiet", reject_divergence)
        with pytest.raises(
            operator.SuccessorOperatorError,
            match="source maintenance ancestry failed",
        ):
            operator.failed_start_source_maintenance_preflight(tmp_path)
        assert paths["stopped_state_continuity"].exists()
        return

    preflight = operator.failed_start_source_maintenance_preflight(tmp_path)
    if tamper == "database":
        with paths["successor_database"].open("ab") as handle:
            handle.write(b"post-maintenance-review-tamper\n")
    else:
        status = operator._strict_json(paths["controller_status"])
        status.pop("status_cid")
        status["reviewed_tamper"] = True
        status["status_cid"] = operator._content_id(status)
        operator._write_status(paths["controller_status"], status)

    with pytest.raises(operator.SuccessorOperatorError):
        operator.reseal_failed_start_source_maintenance(
            tmp_path,
            reviewed_preflight_cid=str(preflight["preflight_cid"]),
        )
    assert paths["stopped_state_continuity"].exists()
    assert not paths["stopped_state_restart_admission"].exists()


@pytest.mark.parametrize("interrupt_after_status", (False, True))
def test_failed_start_source_maintenance_resumes_each_claim_boundary_and_repeats(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupt_after_status: bool,
) -> None:
    operator = _operator()
    paths, _provenance, published, _descendant = (
        _published_failed_start_source_maintenance_fixture(
            operator,
            tmp_path,
            monkeypatch,
        )
    )
    preflight = operator.failed_start_source_maintenance_preflight(tmp_path)
    original_write_status = operator._write_status
    interrupted = False

    def interrupt_projected_status(path: Path, payload: dict[str, object]) -> None:
        nonlocal interrupted
        is_projection = (
            payload.get("error") == operator.FAILED_START_STATUS_ERROR
            and "failed_start_recovery_anchors" not in payload
            and "stopped_state_continuity_receipt_cid" not in payload
        )
        if is_projection and not interrupted:
            interrupted = True
            if interrupt_after_status:
                original_write_status(path, payload)
            raise RuntimeError("simulated source-maintenance interruption")
        original_write_status(path, payload)

    monkeypatch.setattr(operator, "_write_status", interrupt_projected_status)
    with pytest.raises(
        RuntimeError,
        match="simulated source-maintenance interruption",
    ):
        operator.reseal_failed_start_source_maintenance(
            tmp_path,
            reviewed_preflight_cid=str(preflight["preflight_cid"]),
        )
    assert not paths["stopped_state_continuity"].exists()
    assert paths["stopped_state_restart_admission"].exists()

    monkeypatch.setattr(operator, "_write_status", original_write_status)
    resumed = operator.reseal_failed_start_source_maintenance(
        tmp_path,
        reviewed_preflight_cid=str(preflight["preflight_cid"]),
    )
    repeated = operator.reseal_failed_start_source_maintenance(
        tmp_path,
        reviewed_preflight_cid=str(preflight["preflight_cid"]),
    )

    assert resumed["repeated"] is False
    assert repeated["repeated"] is True
    assert resumed["stopped_state_continuity_receipt_cid"] == repeated[
        "stopped_state_continuity_receipt_cid"
    ]
    assert repeated[
        "superseded_stopped_state_continuity_receipt_cid"
    ] == published["receipt_cid"]
    assert paths["stopped_state_continuity"].exists()
    assert not paths["stopped_state_restart_admission"].exists()


def test_restart_admission_reads_generation_fd_bound_surfaces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _operator()
    paths, provenance, _continuity = _stopped_state_continuity_fixture(
        operator, tmp_path, monkeypatch
    )
    observed: dict[str, object] = {}
    original_strict_json = operator._strict_json
    original_digests = operator._stopped_state_database_digests
    original_owner = operator._stopped_owner_status_sha256

    def strict_json(path: Path, **kwargs: object) -> dict[str, object]:
        schema = kwargs.get("expected_schema")
        if schema == operator.STOPPED_STATE_CONTINUITY_SCHEMA:
            observed["receipt"] = path
        elif schema == operator.CONTROLLER_STATUS_SCHEMA:
            observed["status"] = path
        return original_strict_json(path, **kwargs)

    def raw_provenance(
        _paths: dict[str, Path],
        *,
        _receipt_path: Path | None = None,
    ) -> dict[str, object]:
        observed["provenance"] = _receipt_path
        return provenance

    def digests(
        _paths: dict[str, Path],
        *,
        _database_paths: dict[str, Path] | None = None,
    ) -> dict[str, dict[str, str]]:
        observed["databases"] = _database_paths
        return original_digests(
            _paths,
            _database_paths=_database_paths,
        )

    def owner_status(
        _paths: dict[str, Path],
        *,
        controller_status: dict[str, object],
        _status_path: Path | None = None,
        _marker_path: Path | None = None,
    ) -> str:
        observed["owner_status"] = _status_path
        observed["owner_marker"] = _marker_path
        return original_owner(
            _paths,
            controller_status=controller_status,
            _status_path=_status_path,
            _marker_path=_marker_path,
        )

    monkeypatch.setattr(operator, "_strict_json", strict_json)
    monkeypatch.setattr(
        operator,
        "_load_lgcvf_live_raw_provenance_receipt",
        raw_provenance,
    )
    monkeypatch.setattr(operator, "_stopped_state_database_digests", digests)
    monkeypatch.setattr(operator, "_stopped_owner_status_sha256", owner_status)
    custody = operator._open_generation_bound_controller_lock(paths)
    handle = custody["lock_handle"]
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        admitted = operator._load_stopped_restart_provenance(
            paths,
            root=tmp_path,
            provenance=provenance,
            lock_custody=custody,
        )
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        operator._close_generation_bound_controller_lock(custody)

    assert admitted == provenance
    assert all(
        str(observed[name]).startswith("/proc/self/fd/")
        for name in (
            "provenance",
            "receipt",
            "status",
            "owner_status",
            "owner_marker",
        )
    )
    database_paths = observed["databases"]
    assert isinstance(database_paths, dict)
    assert set(database_paths) == {"control", "coordination", "execution"}
    assert all(
        str(database).startswith("/proc/self/fd/")
        for database in database_paths.values()
    )


def test_pinned_restart_rejects_provenance_change_at_final_reread(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _operator()
    paths, provenance, _continuity = _stopped_state_continuity_fixture(
        operator,
        tmp_path,
        monkeypatch,
    )
    calls = 0

    def changing_provenance(
        _paths: dict[str, Path],
        **_kwargs: object,
    ) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls >= 2:
            return {**provenance, "receipt_cid": "b" + ("z" * 60)}
        return provenance

    monkeypatch.setattr(
        operator,
        "_load_lgcvf_live_raw_provenance_receipt",
        changing_provenance,
    )
    custody = operator._open_generation_bound_controller_lock(paths)
    handle = custody["lock_handle"]
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(
            operator.SuccessorOperatorError,
            match="pinned stopped-state evidence changed during admission",
        ):
            operator._load_stopped_restart_admission(
                paths,
                root=tmp_path,
                provenance=provenance,
                lock_custody=custody,
            )
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        operator._close_generation_bound_controller_lock(custody)

    assert calls >= 2
    assert paths["stopped_state_continuity"].is_file()
    assert not paths["stopped_state_restart_admission"].exists()


def test_stopped_restart_preparation_allows_clean_head_above_remote(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _operator()
    config = tmp_path / operator.DEFAULT_SUCCESSOR_CONFIG_RELATIVE
    config.parent.mkdir(parents=True)
    config.write_text("{}\n", encoding="utf-8")
    observed: list[bool] = []

    class RuntimeReached(Exception):
        pass

    def observe(
        root: Path,
        *,
        require_resolved_remote: bool,
    ) -> dict[str, object]:
        assert root == tmp_path
        observed.append(require_resolved_remote)
        return {
            "resolved_remote_head": "1" * 40,
            "current_head": "2" * 40,
            "current_tree": "3" * 40,
        }

    monkeypatch.setattr(operator, "_observe_candidate_runtime_continuity", observe)
    monkeypatch.setattr(
        operator,
        "_candidate_runtime_continuity",
        lambda _root: pytest.fail("restart must not require remote equality"),
    )
    monkeypatch.setattr(
        operator,
        "_resolve_installed_duckdb_live_runtime",
        lambda: (_ for _ in ()).throw(RuntimeReached()),
    )

    with pytest.raises(RuntimeReached):
        operator._prepare_lgcvf_configured_board_live_launch(
            root=tmp_path,
            config_path=config,
            provenance={"receipt_cid": "b" + ("a" * 60)},
            stopped_restart=True,
        )

    assert observed == [False]


def test_projection_once_recovers_interrupted_clean_stop_before_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _operator()
    paths, provenance, _continuity = _stopped_state_continuity_fixture(
        operator, tmp_path, monkeypatch
    )
    bound = operator._strict_json(paths["controller_status"])
    anchor = dict(bound)
    anchor.pop("status_cid")
    anchor.pop("stopped_state_continuity_receipt_cid")
    anchor.pop("stopped_state_continuity_status_cid")
    anchor["status_cid"] = operator._content_id(anchor)
    operator._write_status(paths["controller_status"], anchor)
    paths["stopped_state_continuity"].unlink()
    monkeypatch.setattr(operator, "_extension_preflight", lambda: {"available": True})

    class SnapshotReached(Exception):
        pass

    @operator.contextlib.contextmanager
    def snapshots(*args: object, **kwargs: object) -> object:
        assert paths["stopped_state_continuity"].is_file()
        status = operator._strict_json(paths["controller_status"])
        assert status["stopped_state_continuity_receipt_cid"]
        raise SnapshotReached()
        yield {}

    monkeypatch.setattr(operator, "_sealed_stopped_database_snapshots", snapshots)
    custody = operator._open_generation_bound_controller_lock(paths)
    handle = custody["lock_handle"]
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(SnapshotReached):
            operator._project_ducklake_once_locked(
                tmp_path,
                paths=paths,
                lock_custody=custody,
            )
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        operator._close_generation_bound_controller_lock(custody)

    assert operator._load_stopped_restart_provenance(
        paths,
        root=tmp_path,
        provenance=provenance,
    ) == provenance


def test_projection_once_never_auto_mutates_unreviewed_legacy_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _operator()
    paths, _provenance, _continuity = _stopped_state_continuity_fixture(
        operator, tmp_path, monkeypatch
    )
    _unpublish_stopped_continuity(operator, paths, legacy=True)
    status_before = paths["controller_status"].read_bytes()
    monkeypatch.setattr(operator, "_extension_preflight", lambda: {"available": True})

    custody = operator._open_generation_bound_controller_lock(paths)
    handle = custody["lock_handle"]
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(
            operator.SuccessorOperatorError,
            match="legacy stopped status is not self-anchored",
        ):
            operator._project_ducklake_once_locked(
                tmp_path,
                paths=paths,
                lock_custody=custody,
            )
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        operator._close_generation_bound_controller_lock(custody)

    assert not paths["stopped_state_continuity"].exists()
    assert paths["controller_status"].read_bytes() == status_before


def test_clean_stop_restores_only_the_exact_sealed_import_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _operator()
    sealed = (
        "/proc/self/fd/91/ipfs_datasets_py",
        "/proc/self/fd/91",
    )
    projected_path = [*sealed, "/usr/lib/python3"]
    monkeypatch.setattr(operator.sys, "path", projected_path)

    operator._restore_lgcvf_stopped_candidate_import_boundary(
        root=tmp_path,
        sealed_import_roots=sealed,
    )

    assert operator.sys.path[:2] == [
        str(tmp_path),
        str(tmp_path / "ipfs_datasets_py"),
    ]
    operator.sys.path[:2] = ["/proc/self/fd/92", "/proc/self/fd/92/other"]
    with pytest.raises(
        operator.SuccessorOperatorError,
        match="sealed import boundary differs",
    ):
        operator._restore_lgcvf_stopped_candidate_import_boundary(
            root=tmp_path,
            sealed_import_roots=sealed,
        )


def test_interrupted_restart_claim_restores_durable_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _operator()
    paths, provenance, continuity = _stopped_state_continuity_fixture(
        operator, tmp_path, monkeypatch
    )

    bound_status = operator._strict_json(paths["controller_status"])
    assert operator._claim_stopped_state_restart_admission(
        paths,
        expected_restart=True,
        expected_receipt_cid=str(continuity["receipt_cid"]),
        expected_controller_status_cid=str(bound_status["status_cid"]),
    ) is True
    assert not paths["stopped_state_continuity"].exists()
    assert paths["stopped_state_restart_admission"].is_file()
    assert operator._restore_or_retire_stopped_restart_admission(paths) == (
        "restored_interrupted_claim"
    )
    assert paths["stopped_state_continuity"].is_file()
    assert not paths["stopped_state_restart_admission"].exists()
    assert operator._strict_json(paths["stopped_state_continuity"])[
        "receipt_cid"
    ] == continuity["receipt_cid"]
    assert operator._load_stopped_restart_provenance(
        paths,
        root=tmp_path,
        provenance=provenance,
    ) == provenance


def test_restart_claim_after_import_retarget_never_reobserves_candidate_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _operator()
    paths, _provenance, continuity = _stopped_state_continuity_fixture(
        operator,
        tmp_path,
        monkeypatch,
    )
    status = operator._strict_json(paths["controller_status"])
    monkeypatch.setattr(
        operator,
        "_observe_candidate_runtime_continuity",
        lambda *args, **kwargs: pytest.fail(
            "claim boundary must not require candidate import roots"
        ),
    )
    monkeypatch.setattr(
        operator.sys,
        "path",
        ["/proc/self/fd/91/ipfs_datasets_py", "/proc/self/fd/91"],
    )

    assert operator._claim_stopped_state_restart_admission(
        paths,
        expected_restart=True,
        expected_receipt_cid=str(continuity["receipt_cid"]),
        expected_controller_status_cid=str(status["status_cid"]),
    ) is True
    assert paths["stopped_state_restart_admission"].is_file()


def test_owner_start_failure_leaves_claim_recoverable_without_receipt_loss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _operator()
    paths, _provenance, continuity = _stopped_state_continuity_fixture(
        operator, tmp_path, monkeypatch
    )

    class StartFailure(RuntimeError):
        pass

    class Server:
        def start(self) -> None:
            raise StartFailure

    bound_status = operator._strict_json(paths["controller_status"])
    assert operator._claim_stopped_state_restart_admission(
        paths,
        expected_restart=True,
        expected_receipt_cid=str(continuity["receipt_cid"]),
        expected_controller_status_cid=str(bound_status["status_cid"]),
    ) is True
    with pytest.raises(StartFailure):
        Server().start()
    assert not paths["stopped_state_continuity"].exists()
    assert paths["stopped_state_restart_admission"].is_file()
    assert operator._restore_or_retire_stopped_restart_admission(paths) == (
        "restored_interrupted_claim"
    )
    restored = operator._strict_json(paths["stopped_state_continuity"])
    assert restored["receipt_cid"] == continuity["receipt_cid"]


def test_restart_claim_rejects_receipt_deleted_after_pinned_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _operator()
    paths, provenance, continuity = _stopped_state_continuity_fixture(
        operator,
        tmp_path,
        monkeypatch,
    )
    admitted = operator._load_stopped_restart_admission(
        paths,
        root=tmp_path,
        provenance=provenance,
    )
    status = admitted["controller_status"]
    assert isinstance(status, dict)
    paths["stopped_state_continuity"].unlink()

    with pytest.raises(
        operator.SuccessorOperatorError,
        match="restart receipt presence differs from admission",
    ):
        operator._claim_stopped_state_restart_admission(
            paths,
            expected_restart=True,
            expected_receipt_cid=str(continuity["receipt_cid"]),
            expected_controller_status_cid=str(status["status_cid"]),
        )

    assert not paths["stopped_state_restart_admission"].exists()


def test_corrupt_unbound_status_cannot_retire_the_only_claimed_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _operator()
    paths, _provenance, continuity = _stopped_state_continuity_fixture(
        operator,
        tmp_path,
        monkeypatch,
    )
    bound = operator._strict_json(paths["controller_status"])
    assert operator._claim_stopped_state_restart_admission(
        paths,
        expected_restart=True,
        expected_receipt_cid=str(continuity["receipt_cid"]),
        expected_controller_status_cid=str(bound["status_cid"]),
    ) is True
    corrupt = dict(bound)
    corrupt.pop("status_cid")
    corrupt.pop("stopped_state_continuity_receipt_cid")
    corrupt.pop("stopped_state_continuity_status_cid")
    corrupt["scheduler_returncode"] = False
    corrupt["status_cid"] = operator._content_id(corrupt)
    operator._write_status(paths["controller_status"], corrupt)

    with pytest.raises(
        operator.SuccessorOperatorError,
        match="restart admission/status binding differs",
    ):
        operator._restore_or_retire_stopped_restart_admission(paths)

    assert paths["stopped_state_restart_admission"].is_file()
    assert not paths["stopped_state_continuity"].exists()
    assert operator._strict_json(paths["stopped_state_restart_admission"])[
        "receipt_cid"
    ] == continuity["receipt_cid"]


def test_interrupted_claim_restoration_requires_both_status_crosslinks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _operator()
    paths, _provenance, continuity = _stopped_state_continuity_fixture(
        operator,
        tmp_path,
        monkeypatch,
    )
    bound = operator._strict_json(paths["controller_status"])
    assert operator._claim_stopped_state_restart_admission(
        paths,
        expected_restart=True,
        expected_receipt_cid=str(continuity["receipt_cid"]),
        expected_controller_status_cid=str(bound["status_cid"]),
    ) is True
    malformed = dict(bound)
    malformed.pop("status_cid")
    malformed["stopped_state_continuity_status_cid"] = "b" + ("z" * 60)
    malformed["status_cid"] = operator._content_id(malformed)
    operator._write_status(paths["controller_status"], malformed)

    with pytest.raises(
        operator.SuccessorOperatorError,
        match="restart admission/status binding differs",
    ):
        operator._restore_or_retire_stopped_restart_admission(paths)

    assert paths["stopped_state_restart_admission"].is_file()
    assert not paths["stopped_state_continuity"].exists()


def test_fresh_launch_claim_rejects_late_unadmitted_stopped_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _operator()
    paths, _provenance, _continuity = _stopped_state_continuity_fixture(
        operator,
        tmp_path,
        monkeypatch,
    )
    staged = paths["stopped_state_continuity"].with_suffix(".late")
    paths["stopped_state_continuity"].rename(staged)
    assert not paths["stopped_state_continuity"].exists()
    staged.rename(paths["stopped_state_continuity"])

    with pytest.raises(
        operator.SuccessorOperatorError,
        match="restart receipt presence differs from admission",
    ):
        operator._claim_stopped_state_restart_admission(
            paths,
            expected_restart=False,
        )

    assert paths["stopped_state_continuity"].is_file()
    assert not paths["stopped_state_restart_admission"].exists()


def test_stopped_projection_profile_and_identity_use_one_sealed_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _operator()
    paths, provenance, _continuity = _stopped_state_continuity_fixture(
        operator, tmp_path, monkeypatch
    )
    observed: list[tuple[str, str, int | None]] = []
    observed_receipts: dict[str, Path] = {}
    original_strict_json = operator._strict_json

    def raw_provenance(
        _paths: dict[str, Path],
        *,
        _receipt_path: Path | None = None,
    ) -> dict[str, object]:
        assert _receipt_path is not None
        observed_receipts["provenance"] = _receipt_path
        return provenance

    def strict_json(path: Path, **kwargs: object) -> dict[str, object]:
        schema = kwargs.get("expected_schema")
        if schema == operator.STOPPED_STATE_CONTINUITY_SCHEMA:
            observed_receipts["continuity"] = path
        elif schema == operator.CONTROLLER_STATUS_SCHEMA:
            observed_receipts["status"] = path
        return original_strict_json(path, **kwargs)

    def validate_native(*args: object, **kwargs: object) -> None:
        bootstrap_path = kwargs.get("_bootstrap_path")
        assert isinstance(bootstrap_path, Path)
        observed_receipts["bootstrap"] = bootstrap_path

    monkeypatch.setattr(
        operator,
        "_load_lgcvf_live_raw_provenance_receipt",
        raw_provenance,
    )
    monkeypatch.setattr(operator, "_strict_json", strict_json)
    monkeypatch.setattr(
        operator,
        "_validate_stopped_projection_native_provenance",
        validate_native,
    )

    def verify_profile(
        path: Path,
        *,
        sealed_descriptor: int | None = None,
    ) -> dict[str, object]:
        observed.append(("profile", str(path), sealed_descriptor))
        assert sealed_descriptor is not None
        assert str(path) == f"/proc/self/fd/{sealed_descriptor}"
        return {
            "valid": True,
            "schema_fingerprint": provenance["schema_fingerprint"],
            "catalog_fingerprint": provenance["catalog_fingerprint"],
        }

    def database_identity(path: Path) -> dict[str, object]:
        observed.append(("identity", str(path), None))
        return {
            "database_uuid": provenance["database_uuid"],
            "schema_fingerprint": provenance["schema_fingerprint"],
        }

    monkeypatch.setattr(operator, "_verify_profile", verify_profile)
    monkeypatch.setattr(operator, "_database_identity", database_identity)

    with operator._exclusive_projection_checkpoint(paths) as lock_custody:
        with operator._sealed_stopped_database_snapshots(
            paths, lock_custody
        ) as snapshots:
            control = snapshots["control"]
            admitted = operator._load_projection_source_continuity(
                paths,
                root=tmp_path,
                stopped_database_snapshots=snapshots,
                lock_custody=lock_custody,
            )
            snapshot_path = str(control["snapshot_path"])
            snapshot_descriptor = int(control["snapshot_descriptor"])
            assert observed == [
                ("profile", snapshot_path, snapshot_descriptor),
                ("identity", snapshot_path, None),
            ]
            assert admitted["databases"] == operator._validate_stopped_database_snapshots(
                paths, lock_custody, snapshots
            )
            generation_prefix = (
                f"/proc/self/fd/{int(lock_custody['generation_descriptor'])}/"
            )
            assert set(observed_receipts) == {
                "provenance",
                "continuity",
                "status",
                "bootstrap",
            }
            assert all(
                str(path).startswith(generation_prefix)
                for path in observed_receipts.values()
            )
            assert (
                fcntl.fcntl(snapshot_descriptor, fcntl.F_GET_SEALS)
                & operator.STOPPED_SNAPSHOT_REQUIRED_SEALS
                == operator.STOPPED_SNAPSHOT_REQUIRED_SEALS
            )
            with pytest.raises(OSError):
                os.pwrite(snapshot_descriptor, b"tamper", 0)


def test_stopped_task_history_audit_detects_a_noncontiguous_head(
    tmp_path: Path,
) -> None:
    import duckdb

    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        open_duckdb_connection,
    )

    operator = _operator()
    target_cid = "task:test-history-audit:00"
    database = tmp_path / "history-audit.duckdb"
    source = DatabaseTaskSource(database)
    try:
        source.materialize(
            {
                "repository_tree_id": "tree:test-history-audit",
                "plan_root_cid": "plan:test-history-audit",
                "goals": [
                    {
                        "goal_cid": "goal:test-history-audit",
                        "goal_alias": "G-HISTORY-AUDIT",
                        "title": "History audit",
                    }
                ],
                "tasks": [
                    {
                        "task_cid": f"task:test-history-audit:{index:02d}",
                        "task_id": task_alias,
                        "goal_cid": "goal:test-history-audit",
                        "status": "ready",
                        "body": {"title": "History audit"},
                    }
                    for index, task_alias in enumerate(
                        operator.LGCVF_TASK_ALIASES
                    )
                ],
            }
        )
    finally:
        source.close()

    connection = duckdb.connect(str(database), read_only=True)
    try:
        valid = operator._audit_task_history_connection(connection)
    finally:
        connection.close()
    assert valid["valid"] is True
    assert valid["task_count"] == len(operator.LGCVF_TASK_ALIASES)
    assert valid["invalid_task_count"] == 0
    assert valid["tasks"][0]["history_count"] == 1
    assert valid["tasks"][0]["errors"] == []

    connection = open_duckdb_connection(database)
    try:
        connection.execute(
            "UPDATE tasks SET revision = 3 WHERE task_cid = ?",
            [target_cid],
        )
    finally:
        connection.close()
    connection = duckdb.connect(str(database), read_only=True)
    try:
        invalid = operator._audit_task_history_connection(connection)
    finally:
        connection.close()
    assert invalid["valid"] is False
    assert invalid["valid_task_count"] == len(operator.LGCVF_TASK_ALIASES) - 1
    assert invalid["invalid_task_count"] == 1
    assert invalid["tasks"][0]["head_revision"] == 3
    assert invalid["tasks"][0]["history_count"] == 1
    assert invalid["tasks"][0]["errors"] == [
        "history_count_differs_from_head"
    ]

    connection = open_duckdb_connection(database)
    try:
        connection.execute(
            "UPDATE tasks SET revision = 1 WHERE task_cid = ?",
            [target_cid],
        )
        connection.execute(
            "INSERT INTO task_revisions VALUES "
            "('task:orphan-history', 1, 'ready', '{}', "
            "'1970-01-01T00:00:00Z')"
        )
    finally:
        connection.close()
    connection = duckdb.connect(str(database), read_only=True)
    try:
        orphaned = operator._audit_task_history_connection(connection)
    finally:
        connection.close()
    assert orphaned["valid"] is False
    assert orphaned["errors"] == ["orphan_task_revisions_present"]
    assert orphaned["orphan_history_count"] == 1
    assert orphaned["tasks"][0]["valid"] is True

    connection = open_duckdb_connection(database)
    try:
        connection.execute(
            "DELETE FROM task_revisions WHERE task_cid = 'task:orphan-history'"
        )
        connection.execute(
            "UPDATE tasks SET body_json = '{\"value\":NaN}' "
            "WHERE task_cid = ?",
            [target_cid],
        )
        connection.execute(
            "UPDATE task_revisions SET body_json = '{\"value\":NaN}' "
            "WHERE task_cid = ?",
            [target_cid],
        )
    finally:
        connection.close()
    connection = duckdb.connect(str(database), read_only=True)
    try:
        nonfinite = operator._audit_task_history_connection(connection)
    finally:
        connection.close()
    assert nonfinite["valid"] is False
    assert nonfinite["errors"] == []
    assert nonfinite["tasks"][0]["errors"] == [
        "history_body_not_object",
        "task_body_not_object",
    ]

    for rejected_json in ('{"value":1.5}', '{"value":1,"value":2}'):
        connection = open_duckdb_connection(database)
        try:
            connection.execute(
                "UPDATE tasks SET body_json = ? WHERE task_cid = ?",
                [rejected_json, target_cid],
            )
            connection.execute(
                "UPDATE task_revisions SET body_json = ? WHERE task_cid = ?",
                [rejected_json, target_cid],
            )
        finally:
            connection.close()
        connection = duckdb.connect(str(database), read_only=True)
        try:
            rejected = operator._audit_task_history_connection(connection)
        finally:
            connection.close()
        assert rejected["valid"] is False
        assert rejected["tasks"][0]["errors"] == [
            "history_body_not_object",
            "task_body_not_object",
        ]

    missing_cid = (
        f"task:test-history-audit:{len(operator.LGCVF_TASK_ALIASES) - 1:02d}"
    )
    connection = open_duckdb_connection(database)
    try:
        connection.execute(
            "DELETE FROM task_revisions WHERE task_cid = ?",
            [missing_cid],
        )
        connection.execute(
            "DELETE FROM tasks WHERE task_cid = ?",
            [missing_cid],
        )
    finally:
        connection.close()
    connection = duckdb.connect(str(database), read_only=True)
    try:
        below_initial = operator._audit_task_history_connection(connection)
    finally:
        connection.close()
    assert below_initial["valid"] is False
    assert below_initial["errors"] == ["task_population_below_initial_board"]


def test_stopped_task_history_audit_uses_one_sealed_control_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import duckdb

    from ipfs_accelerate_py.agent_supervisor.task_sources import duckdb_state

    operator = _operator()
    _paths, provenance, continuity = _stopped_state_continuity_fixture(
        operator,
        tmp_path,
        monkeypatch,
    )
    observed: dict[str, object] = {}

    class FakeConnection:
        closed = False

        def close(self) -> None:
            self.closed = True

    connection = FakeConnection()

    def connect(
        duckdb_module: object,
        path: str,
        *,
        read_only: bool,
    ) -> FakeConnection:
        assert duckdb_module is duckdb
        observed["snapshot_path"] = path
        observed["read_only"] = read_only
        assert path.startswith("/proc/self/fd/")
        assert Path(path).is_file()
        return connection

    audit = {
        "valid": False,
        "errors": ["orphan_task_revisions_present"],
        "task_count": 1,
        "history_row_count": 2,
        "orphan_history_count": 1,
        "valid_task_count": 1,
        "invalid_task_count": 0,
        "tasks": [{"task_cid": "task:test", "valid": True}],
    }
    monkeypatch.setattr(duckdb_state, "connect_duckdb_with_policy", connect)
    monkeypatch.setattr(
        operator,
        "_audit_task_history_connection",
        lambda observed_connection: (
            audit
            if observed_connection is connection
            else pytest.fail("history audit used a different connection")
        ),
    )

    result = operator.stopped_task_history_audit(tmp_path)

    assert observed["read_only"] is True
    assert connection.closed is True
    assert not Path(str(observed["snapshot_path"])).exists()
    assert result["schema"] == operator.STOPPED_TASK_HISTORY_AUDIT_SCHEMA
    assert result["valid"] is False
    assert result["authoritative"] is False
    assert result["mutation_authority"] is False
    assert result["source_provenance_cid"] == provenance["receipt_cid"]
    assert result["stopped_state_continuity_receipt_cid"] == continuity[
        "receipt_cid"
    ]
    assert result["errors"] == ["orphan_task_revisions_present"]
    assert result["history_row_count"] == 2
    assert result["orphan_history_count"] == 1
    assert result["tasks"] == audit["tasks"]
    result_body = dict(result)
    audit_cid = result_body.pop("audit_cid")
    assert audit_cid == operator._content_id(result_body)
    assert continuity["test_restart_calls"]() == 0


def test_canonical_profile_verifier_reads_sealed_descriptor_without_sidecar(
    tmp_path: Path,
) -> None:
    operator = _operator()
    paths = operator._paths(tmp_path)
    _seed_datasets_profile(paths["successor_database"])
    generation = paths["controller_lock"].parent
    generation.chmod(0o700)
    paths["successor_database"].chmod(0o600)
    source_bytes = paths["successor_database"].read_bytes()
    for name, database in operator._successor_state_databases(paths).items():
        if name != "control":
            database.write_bytes(source_bytes)
            database.chmod(0o600)
    hidden_before = {entry.name for entry in generation.iterdir() if entry.name.startswith(".")}

    with operator._exclusive_projection_checkpoint(paths) as lock_custody:
        with operator._sealed_stopped_database_snapshots(
            paths, lock_custody
        ) as snapshots:
            control = snapshots["control"]
            descriptor = int(control["snapshot_descriptor"])
            snapshot_path = Path(str(control["snapshot_path"]))
            verification = operator._verify_profile(
                snapshot_path,
                sealed_descriptor=descriptor,
            )
            identity = operator._database_identity(snapshot_path)

    assert verification["valid"] is True
    assert identity["database_uuid"]
    assert {
        entry.name for entry in generation.iterdir() if entry.name.startswith(".")
    } == hidden_before


def test_projection_preflight_requires_exact_stopped_mode_without_residue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _operator()
    paths, _provenance, continuity = _stopped_state_continuity_fixture(
        operator, tmp_path, monkeypatch
    )
    monkeypatch.setattr(
        operator,
        "_extension_preflight",
        lambda: {"available": True},
    )

    preflight = operator.projection_preflight(tmp_path)

    assert preflight["valid"] is True
    assert preflight["source_admitted"] is True
    assert preflight["source_admission_mode"] == (
        operator.STOPPED_STATE_CONTINUITY_ADMISSION_MODE
    )
    assert preflight["stopped_state_continuity_receipt_cid"] == (
        continuity["receipt_cid"]
    )
    assert preflight["projection_root_present"] is False
    assert preflight["projection_receipt_present"] is False
    assert not os.path.lexists(paths["projection_root"])
    assert not os.path.lexists(paths["projection_receipt"])


@pytest.mark.parametrize("residue", ("root", "receipt"))
def test_projection_preflight_rejects_stopped_mode_with_projection_residue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    residue: str,
) -> None:
    operator = _operator()
    paths, _provenance, _continuity = _stopped_state_continuity_fixture(
        operator, tmp_path, monkeypatch
    )
    monkeypatch.setattr(
        operator,
        "_extension_preflight",
        lambda: {"available": True},
    )
    if residue == "root":
        paths["projection_root"].mkdir(mode=0o700, parents=True)
    else:
        paths["projection_receipt"].write_text(
            "preserved residue\n",
            encoding="utf-8",
        )
        paths["projection_receipt"].chmod(0o600)

    preflight = operator.projection_preflight(tmp_path)

    assert preflight["valid"] is False
    assert preflight["source_admitted"] is True
    assert preflight["source_admission_mode"] == (
        operator.STOPPED_STATE_CONTINUITY_ADMISSION_MODE
    )
    assert preflight["projection_root_present"] is (residue == "root")
    assert preflight["projection_receipt_present"] is (residue == "receipt")


def test_projection_stopped_continuity_admits_exact_advanced_board_head(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _operator()
    paths, provenance, continuity = _stopped_state_continuity_fixture(
        operator, tmp_path, monkeypatch
    )
    final_continuity = continuity["final_source_continuity"]
    observed: dict[str, object] = {}

    def advanced_head(*args: object, **kwargs: object) -> object:
        raise operator.SuccessorOperatorError(
            operator.NATIVE_RESUME_PROVENANCE_BINDING_ERROR
        )

    def validate(
        observed_paths: dict[str, Path],
        *,
        root: Path,
        receipt: dict[str, object],
        final_continuity: dict[str, object],
        _bootstrap_path: Path | None = None,
    ) -> None:
        observed.update(
            {
                "paths": observed_paths,
                "root": root,
                "receipt": receipt,
                "final_continuity": final_continuity,
            }
        )

    monkeypatch.setattr(operator, "_load_provenance", advanced_head)
    monkeypatch.setattr(
        operator, "_validate_stopped_projection_native_provenance", validate
    )

    admitted = operator._load_projection_source_continuity(paths, root=tmp_path)

    assert admitted["admission_mode"] == (
        operator.STOPPED_STATE_CONTINUITY_ADMISSION_MODE
    )
    assert admitted["provenance"] == provenance
    assert observed == {
        "paths": paths,
        "root": tmp_path,
        "receipt": provenance,
        "final_continuity": final_continuity,
    }


def test_stopped_continuity_observer_allows_clean_local_head_ahead_of_remote(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _operator()
    nested = tmp_path / "ipfs_datasets_py"
    nested.mkdir(mode=0o700)
    quarantine = tmp_path.parent / f"{tmp_path.name}-pycache"
    quarantine.mkdir(mode=0o700)
    remote_head = "1" * 40
    final_head = "2" * 40
    final_tree = "3" * 40
    datasets_head = "4" * 40
    datasets_tree = "5" * 40
    ancestry: list[tuple[Path, tuple[str, ...]]] = []

    monkeypatch.setattr(operator, "_AMBIENT_PYTHONPATH", frozenset())
    monkeypatch.setattr(
        operator,
        "_RUNTIME_PYCACHE",
        SimpleNamespace(name=str(quarantine)),
    )
    monkeypatch.setattr(
        operator.sys,
        "path",
        [str(tmp_path), str(nested)],
    )
    monkeypatch.setattr(operator.sys, "pycache_prefix", str(quarantine))

    def git_text(
        repository: Path,
        arguments: tuple[str, ...],
        *,
        noun: str,
    ) -> str:
        if repository == nested:
            if arguments == ("rev-parse", "HEAD"):
                return datasets_head
            if arguments == ("rev-parse", "HEAD^{tree}"):
                return datasets_tree
            if arguments[0] == "status":
                return ""
        if arguments == ("symbolic-ref", "--short", "HEAD"):
            return operator.APPROVED_BOARD_BRANCH
        if arguments == ("rev-parse", "HEAD"):
            return final_head
        if arguments == ("rev-parse", "HEAD^{tree}"):
            return final_tree
        if arguments[0] == "status":
            return ""
        if arguments[0] == "ls-tree":
            return f"160000 commit {datasets_head}\tipfs_datasets_py"
        if arguments == ("rev-parse", operator.APPROVED_REMOTE_BRANCH_REF):
            return remote_head
        raise AssertionError((repository, arguments, noun))

    monkeypatch.setattr(operator, "_git_text", git_text)
    monkeypatch.setattr(
        operator,
        "_git_quiet",
        lambda repository, arguments, *, noun: ancestry.append(
            (repository, arguments)
        ),
    )
    monkeypatch.setattr(
        operator,
        "_tracked_runtime_inventory",
        lambda *args, **kwargs: {
            "tracked_object_count": 1,
            "tracked_inventory_root": "sha256:" + ("a" * 64),
        },
    )

    observed = operator._observe_candidate_runtime_continuity(
        tmp_path,
        require_resolved_remote=False,
    )

    assert observed["resolved_remote_head"] == remote_head
    assert observed["current_head"] == final_head
    assert ancestry == [
        (
            tmp_path,
            ("merge-base", "--is-ancestor", remote_head, final_head),
        )
    ]
    with pytest.raises(
        operator.SuccessorOperatorError,
        match="not the resolved remote branch",
    ):
        operator._candidate_runtime_continuity(tmp_path)


def test_stopped_projection_continuity_allows_remote_catch_up_to_current_head(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _operator()
    recorded_remote = "1" * 40
    current_head = "3" * 40
    sealed = {
        "approved_branch": operator.APPROVED_BOARD_BRANCH,
        "resolved_remote_head": recorded_remote,
        "current_head": current_head,
        "current_tree": "4" * 40,
        "candidate_worktree_clean": True,
        "datasets_head": "5" * 40,
        "datasets_tree": "6" * 40,
        "datasets_worktree_clean": True,
        "superproject_runtime_inventory": {"root": "sealed"},
        "datasets_runtime_inventory": {"root": "sealed"},
    }
    observed = {**sealed, "resolved_remote_head": current_head}
    ancestry: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        operator,
        "_observe_candidate_runtime_continuity",
        lambda root, *, require_resolved_remote: dict(observed),
    )
    monkeypatch.setattr(
        operator,
        "_git_quiet",
        lambda root, arguments, *, noun: ancestry.append(arguments),
    )

    admitted = operator._observe_stopped_projection_source_continuity(
        tmp_path,
        sealed,
    )

    assert admitted == observed
    assert ancestry == [
        ("merge-base", "--is-ancestor", recorded_remote, current_head)
    ]

    observed["current_tree"] = "9" * 40
    with pytest.raises(
        operator.SuccessorOperatorError,
        match="final source continuity differs",
    ):
        operator._observe_stopped_projection_source_continuity(
            tmp_path,
            sealed,
        )


@pytest.mark.parametrize("movement", ("beyond", "divergent"))
def test_stopped_projection_continuity_rejects_remote_outside_current_ancestry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    movement: str,
) -> None:
    operator = _operator()
    recorded_remote = "1" * 40
    current_head = "3" * 40
    observed_remote = "7" * 40
    sealed = {
        "resolved_remote_head": recorded_remote,
        "current_head": current_head,
        "current_tree": "4" * 40,
    }
    observed = {**sealed, "resolved_remote_head": observed_remote}
    ancestry = (
        {(recorded_remote, observed_remote)} if movement == "beyond" else set()
    )
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        operator,
        "_observe_candidate_runtime_continuity",
        lambda root, *, require_resolved_remote: dict(observed),
    )

    def git_quiet(
        root: Path,
        arguments: tuple[str, ...],
        *,
        noun: str,
    ) -> None:
        edge = (arguments[-2], arguments[-1])
        calls.append(edge)
        if edge not in ancestry:
            raise operator.SuccessorOperatorError("test ancestry rejection")

    monkeypatch.setattr(operator, "_git_quiet", git_quiet)

    with pytest.raises(
        operator.SuccessorOperatorError,
        match="test ancestry rejection",
    ):
        operator._observe_stopped_projection_source_continuity(
            tmp_path,
            sealed,
        )

    if movement == "beyond":
        assert calls == [
            (recorded_remote, observed_remote),
            (observed_remote, current_head),
        ]
    else:
        assert calls == [(recorded_remote, observed_remote)]


def test_stopped_projection_native_provenance_requires_advanced_head_ancestry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _operator()
    paths = operator._paths(tmp_path)
    initial_head = "1" * 40
    initial_tree = "2" * 40
    initial_datasets_head = "3" * 40
    initial_datasets_tree = "4" * 40
    final_head = "5" * 40
    final_datasets_head = "6" * 40
    config_raw = b'{"reviewed":"candidate"}\n'
    ready_task_ids = ["LGCVF-051", "LGCVF-060", "LGCVF-070", "LGCVF-080"]
    initial_projection = {
        "task_count": len(operator.LGCVF_TASK_ALIASES),
        "completed_task_ids": list(operator.CONSTRUCTION_COMPLETED_TASK_IDS),
        "ready_task_ids": ready_task_ids,
        "blocked_task_ids": list(operator.BLOCKED_TASK_IDS),
        "terminal_task_id": "LGCVF-124",
        "goal_count": 14,
        "root_goal_id": "LGCVF-G000",
    }
    config = {"initial_projection": initial_projection}
    cid = "b" + ("a" * 60)
    sha = "sha256:" + ("a" * 64)
    initial_statuses = {
        alias: (
            "completed"
            if alias in operator.CONSTRUCTION_COMPLETED_TASK_IDS
            else "blocked"
            if alias in operator.BLOCKED_TASK_IDS
            else "todo"
        )
        for alias in operator.LGCVF_TASK_ALIASES
    }
    materialized_projection = {
        "task_count": len(operator.LGCVF_TASK_ALIASES),
        "completed_count": len(operator.CONSTRUCTION_COMPLETED_TASK_IDS),
        "todo_count": (
            len(operator.LGCVF_TASK_ALIASES)
            - len(operator.CONSTRUCTION_COMPLETED_TASK_IDS)
            - len(operator.BLOCKED_TASK_IDS)
        ),
        "blocked_count": len(operator.BLOCKED_TASK_IDS),
        "completed_task_ids": list(operator.CONSTRUCTION_COMPLETED_TASK_IDS),
        "ready_task_ids": ready_task_ids,
        "blocked_task_ids": list(operator.BLOCKED_TASK_IDS),
    }
    materialized_projection["projection_root"] = operator._content_id(
        materialized_projection
    )
    provenance: dict[str, object] = {
        "schema": operator.PROVENANCE_SCHEMA,
        "issued_at": "2026-08-26T00:00:00Z",
        "admission_mode": operator.NATIVE_RESUME_ADMISSION_MODE,
        "source_generation": operator.NATIVE_RESUME_SOURCE_GENERATION,
        "target_generation": operator.SUCCESSOR_STORE_GENERATION,
        "source_database": "",
        "target_database": str(paths["successor_database"]),
        "source_head": initial_head,
        "source_tree": initial_tree,
        "source_forest_root": cid,
        "datasets_head": initial_datasets_head,
        "datasets_tree": initial_datasets_tree,
        "candidate_config_path": (
            operator.DEFAULT_SUCCESSOR_CONFIG_RELATIVE.as_posix()
        ),
        "candidate_config_sha256": (
            "sha256:" + hashlib.sha256(config_raw).hexdigest()
        ),
        "population_root": cid,
        "plan_root_cid": cid,
        "initial_projection": initial_projection,
        "materialized_projection": materialized_projection,
        "bootstrap_receipt_cid": cid,
        "bootstrap_verification_root": cid,
        "target_initial_sha256": sha,
        "target_coordination_initial_sha256": sha,
        "target_execution_initial_sha256": sha,
        "database_uuid": "database:test-advanced-head",
        "schema_fingerprint": cid,
        "catalog_fingerprint": cid,
        "initial_projection_reset": True,
        "continuity_completion_records_imported": False,
        "source_database_statuses_read": False,
        "source_database_completion_records_imported": False,
        "quack_required_after_publish": True,
        "direct_multi_process_duckdb_permitted": False,
        "ducklake_projection_authoritative": False,
        "restart_requires_live_continuity_receipt": True,
        "live_continuity_receipt_implemented": False,
        "candidate_authored_validation": True,
        "validation_self_authority": False,
        "validation_completion_authoritative": False,
        "network_isolation_enforced": True,
        "model_provider_route": "none",
        "task_implementation_complete": False,
        "test_qualification_complete": False,
        "objective_complete": False,
        "release_qualified": False,
        "authoritative_for_release": False,
        "production_authorized": False,
        "receipt_cid": cid,
    }
    final_continuity = {
        "approved_branch": operator.APPROVED_BOARD_BRANCH,
        "resolved_remote_head": final_head,
        "current_head": final_head,
        "current_tree": "7" * 40,
        "candidate_worktree_clean": True,
        "datasets_head": final_datasets_head,
        "datasets_tree": "8" * 40,
        "datasets_worktree_clean": True,
    }
    bootstrap = {
        "receipt_cid": cid,
        "population_root": cid,
        "plan_root_cid": cid,
        "verification": {
            "verification_root": cid,
            "control": {
                "statuses": initial_statuses,
                "ready_task_aliases": ready_task_ids,
            },
        },
    }
    git_quiet_calls: list[tuple[Path, tuple[str, ...]]] = []
    initial_gitlink_oid = [initial_datasets_head]

    monkeypatch.setattr(
        operator, "_load_native_resume_config", lambda _root: (config, config_raw)
    )
    def git_text(
        repository: Path,
        arguments: tuple[str, ...],
        *,
        noun: str,
    ) -> str:
        if repository == tmp_path and arguments[0] == "ls-tree":
            return (
                f"160000 commit {initial_gitlink_oid[0]}\t"
                "ipfs_datasets_py"
            )
        return initial_tree if repository == tmp_path else initial_datasets_tree

    monkeypatch.setattr(operator, "_git_text", git_text)

    def git_quiet(
        repository: Path,
        arguments: tuple[str, ...],
        *,
        noun: str,
    ) -> None:
        git_quiet_calls.append((repository, arguments))

    monkeypatch.setattr(operator, "_git_quiet", git_quiet)
    monkeypatch.setattr(
        operator,
        "_target_source_continuity",
        lambda *args, **kwargs: {
            **final_continuity,
            "target_source_head": initial_head,
            "target_source_tree": initial_tree,
        },
    )
    monkeypatch.setattr(operator, "_strict_json", lambda *args, **kwargs: bootstrap)
    monkeypatch.setattr(
        operator, "_validate_native_bootstrap_receipt", lambda *args, **kwargs: None
    )

    operator._validate_stopped_projection_native_provenance(
        paths,
        root=tmp_path,
        receipt=provenance,
        final_continuity=final_continuity,
    )

    assert (
        tmp_path,
        ("merge-base", "--is-ancestor", initial_head, final_head),
    ) in git_quiet_calls
    assert (
        tmp_path / "ipfs_datasets_py",
        (
            "merge-base",
            "--is-ancestor",
            initial_datasets_head,
            final_datasets_head,
        ),
    ) in git_quiet_calls

    drifted = dict(provenance)
    drifted["candidate_config_sha256"] = "sha256:" + ("f" * 64)
    with pytest.raises(
        operator.SuccessorOperatorError,
        match="initial provenance binding differs",
    ):
        operator._validate_stopped_projection_native_provenance(
            paths,
            root=tmp_path,
            receipt=drifted,
            final_continuity=final_continuity,
        )

    initial_gitlink_oid[0] = "9" * 40
    with pytest.raises(
        operator.SuccessorOperatorError,
        match="initial nested gitlink binding differs",
    ):
        operator._validate_stopped_projection_native_provenance(
            paths,
            root=tmp_path,
            receipt=provenance,
            final_continuity=final_continuity,
        )
    initial_gitlink_oid[0] = initial_datasets_head

    drifted_projection = dict(materialized_projection)
    drifted_projection["ready_task_ids"] = list(reversed(ready_task_ids))
    projection_body = dict(drifted_projection)
    projection_body.pop("projection_root")
    drifted_projection["projection_root"] = operator._content_id(projection_body)
    drifted = dict(provenance)
    drifted["materialized_projection"] = drifted_projection
    receipt_body = dict(drifted)
    receipt_body.pop("receipt_cid")
    drifted["receipt_cid"] = operator._content_id(receipt_body)
    with pytest.raises(
        operator.SuccessorOperatorError,
        match="initial projection replay differs",
    ):
        operator._validate_stopped_projection_native_provenance(
            paths,
            root=tmp_path,
            receipt=drifted,
            final_continuity=final_continuity,
        )


@pytest.mark.parametrize("tamper", ("database", "wal", "receipt"))
def test_projection_stopped_continuity_rejects_tamper_and_live_wal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    operator = _operator()
    paths, _provenance, _continuity = _stopped_state_continuity_fixture(
        operator, tmp_path, monkeypatch
    )

    if tamper == "database":
        with paths["successor_database"].open("ab") as handle:
            handle.write(b"tamper\n")
    elif tamper == "wal":
        wal = paths["successor_database"].with_name("control.duckdb.wal")
        wal.write_bytes(b"live\n")
        wal.chmod(0o600)
    else:
        raw = paths["stopped_state_continuity"].read_bytes()
        paths["stopped_state_continuity"].write_bytes(
            raw.replace(b'"restart_authority":true', b'"restart_authority":false')
        )
        paths["stopped_state_continuity"].chmod(0o600)

    with pytest.raises(operator.SuccessorOperatorError):
        operator._load_projection_source_continuity(paths, root=tmp_path)


def test_projection_initial_provenance_does_not_require_stopped_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _operator()
    paths = operator._paths(tmp_path)
    provenance = {"receipt_cid": "b" + ("c" * 60)}
    monkeypatch.setattr(
        operator,
        "_load_provenance",
        lambda _paths, *, root: provenance,
    )
    monkeypatch.setattr(
        operator,
        "_stopped_state_database_digests",
        lambda _paths: {"control": {}, "coordination": {}, "execution": {}},
    )

    admitted = operator._load_projection_source_continuity(paths, root=tmp_path)

    assert admitted["admission_mode"] == (
        operator.INITIAL_PROVENANCE_PROJECTION_ADMISSION_MODE
    )
    assert admitted["receipt"] == {}
    assert not paths["stopped_state_continuity"].exists()
    monkeypatch.setattr(
        operator,
        "_extension_preflight",
        lambda: {"available": True},
    )

    preflight = operator.projection_preflight(tmp_path)

    assert preflight["valid"] is False
    assert preflight["source_admitted"] is True
    assert preflight["source_admission_mode"] == (
        operator.INITIAL_PROVENANCE_PROJECTION_ADMISSION_MODE
    )
    assert preflight["projection_root_present"] is False
    assert preflight["projection_receipt_present"] is False


def test_projection_once_rejects_initial_provenance_without_projection_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _operator()
    paths = operator._paths(tmp_path)
    source_loaded = False
    databases = operator._successor_state_databases(paths)
    paths["controller_lock"].parent.mkdir(mode=0o700, parents=True)
    for name, database in databases.items():
        database.write_bytes(f"initial-{name}\n".encode())
        database.chmod(0o600)
    monkeypatch.setattr(
        operator,
        "_extension_preflight",
        lambda: {"available": True},
    )

    def load_source(*args: object, **kwargs: object) -> object:
        nonlocal source_loaded
        source_loaded = True
        return {
            "admission_mode": (
                operator.INITIAL_PROVENANCE_PROJECTION_ADMISSION_MODE
            )
        }

    monkeypatch.setattr(operator, "_load_projection_source_continuity", load_source)

    with operator._exclusive_projection_checkpoint(paths) as lock_custody:
        with pytest.raises(
            operator.SuccessorOperatorError,
            match="lost typed stopped-state continuity",
        ):
            operator._project_ducklake_once_locked(
                tmp_path,
                paths=paths,
                lock_custody=lock_custody,
            )

    assert source_loaded is True
    assert not os.path.lexists(paths["projection_root"])
    assert not os.path.lexists(paths["projection_receipt"])


def test_projection_once_rejects_residual_root_without_reading_or_deleting_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _operator()
    paths = operator._paths(tmp_path)
    paths["projection_root"].mkdir(mode=0o700, parents=True)
    paths["controller_lock"].parent.chmod(0o700)
    residue = paths["projection_root"] / "foreign-residue"
    residue.write_text("preserve for inspection\n", encoding="utf-8")
    source_loaded = False

    monkeypatch.setattr(
        operator,
        "_extension_preflight",
        lambda: {"available": True},
    )

    def load_source(*args: object, **kwargs: object) -> object:
        nonlocal source_loaded
        source_loaded = True
        return {}

    monkeypatch.setattr(operator, "_load_projection_source_continuity", load_source)

    with operator._exclusive_projection_checkpoint(paths) as lock_custody:
        with pytest.raises(
            operator.SuccessorOperatorError,
            match="refusing to reuse residual DuckLake projection root",
        ):
            operator._project_ducklake_once_locked(
                tmp_path,
                paths=paths,
                lock_custody=lock_custody,
            )

    assert source_loaded is False
    assert residue.read_text(encoding="utf-8") == "preserve for inspection\n"
    assert not os.path.lexists(paths["projection_receipt"])


def test_projection_query_uses_sealed_bytes_and_rejects_path_swap_restore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import duckdb

    operator = _operator()
    paths = operator._paths(tmp_path)
    databases = operator._successor_state_databases(paths)
    paths["controller_lock"].parent.mkdir(mode=0o700, parents=True)
    for name, database in databases.items():
        database.write_bytes(f"sealed-{name}\n".encode())
        database.chmod(0o600)
    expected_control = paths["successor_database"].read_bytes()
    observed_query_paths: list[str] = []
    observed_tasks: list[dict[str, object]] = []

    monkeypatch.setattr(
        operator,
        "_extension_preflight",
        lambda: {"available": True},
    )

    def load_continuity(
        loaded_paths: dict[str, Path],
        *,
        root: Path,
        stopped_database_snapshots: dict[str, dict[str, object]],
        lock_custody: dict[str, object],
    ) -> dict[str, object]:
        assert loaded_paths == paths
        assert root == tmp_path
        return {
            "provenance": {"receipt_cid": "provenance:test-sealed-query"},
            "receipt": {
                "receipt_cid": "continuity:test-sealed-query",
                "controller_status_cid": "status:test-sealed-query",
            },
            "databases": operator._validate_stopped_database_snapshots(
                paths,
                lock_custody,
                stopped_database_snapshots,
            ),
            "admission_mode": operator.STOPPED_STATE_CONTINUITY_ADMISSION_MODE,
        }

    class QueryResult:
        def __init__(self, rows: list[tuple[object, ...]]) -> None:
            self.rows = rows

        def fetchall(self) -> list[tuple[object, ...]]:
            return self.rows

    class SourceConnection:
        def execute(self, statement: str) -> QueryResult:
            if "information_schema.columns" in statement:
                return QueryResult(
                    [
                        ("task_alias",),
                        ("task_cid",),
                        ("status",),
                        ("body_json",),
                        ("ordinal",),
                    ]
                )
            return QueryResult(
                [
                    (
                        "LGCVF-080",
                        "task:test-080",
                        "todo",
                        '{"title":"sealed task","depends_on":[]}',
                        1,
                    )
                ]
            )

        def close(self) -> None:
            return None

    def connect(database: str, *, read_only: bool) -> SourceConnection:
        assert read_only is True
        observed_query_paths.append(database)
        assert database.startswith("/proc/self/fd/")
        displaced = paths["successor_database"].with_name("control.displaced")
        paths["successor_database"].rename(displaced)
        paths["successor_database"].write_bytes(b"foreign-control\n")
        paths["successor_database"].chmod(0o600)
        assert Path(database).read_bytes() == expected_control
        paths["successor_database"].unlink()
        displaced.rename(paths["successor_database"])
        return SourceConnection()

    class Plane:
        backend = "ducklake+quack"
        quack_loaded = True
        ducklake_loaded = True
        ducklake_attached = True

        def __enter__(self) -> Plane:
            paths["projection_root"].joinpath("control.duckdb").write_bytes(b"c")
            paths["projection_root"].joinpath("lake.ducklake").write_bytes(b"l")
            paths["projection_root"].joinpath("lake-data").mkdir(mode=0o700)
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def register_board(self, _namespace: str, **kwargs: object) -> dict[str, str]:
            assert kwargs["source_path"] == str(paths["successor_database"])
            tasks = kwargs["tasks"]
            assert isinstance(tasks, list)
            observed_tasks.extend(tasks)
            return {"board_namespace": "board:test-sealed-query"}

        def aggregate_boards(self) -> dict[str, int]:
            return {"task_count": 1}

    monkeypatch.setattr(operator, "_load_projection_source_continuity", load_continuity)
    monkeypatch.setattr(duckdb, "connect", connect)

    def open_projection(
        root: Path,
        projection_root: Path,
    ) -> Plane:
        assert root == tmp_path
        assert str(projection_root).startswith("/proc/self/fd/")
        return Plane()

    monkeypatch.setattr(
        operator,
        "_open_projection_plane",
        open_projection,
    )
    monkeypatch.setattr(
        operator,
        "_bind_projection_logical_paths",
        lambda *args, **kwargs: {"aggregate": {"task_count": 1}},
    )
    monkeypatch.setattr(
        operator,
        "_validate_projection_root_outputs",
        lambda *args, **kwargs: None,
    )

    with operator._exclusive_projection_checkpoint(paths) as lock_custody:
        with pytest.raises(
            operator.SuccessorOperatorError,
            match="snapshot changed",
        ):
            operator._project_ducklake_once_locked(
                tmp_path,
                paths=paths,
                lock_custody=lock_custody,
            )

    assert len(observed_query_paths) == 1
    assert observed_tasks[0]["task_id"] == "LGCVF-080"
    assert paths["successor_database"].read_bytes() == expected_control
    assert not os.path.lexists(paths["projection_receipt"])


def test_projection_root_claim_rejects_raced_symlink_without_following_it(
    tmp_path: Path,
) -> None:
    operator = _operator()
    paths = operator._paths(tmp_path)
    paths["controller_lock"].parent.mkdir(mode=0o700, parents=True)
    foreign = tmp_path / "foreign-projection"
    foreign.mkdir(mode=0o700)

    with operator._exclusive_projection_checkpoint(paths) as lock_custody:
        assert not os.path.lexists(paths["projection_root"])
        paths["projection_root"].symlink_to(foreign, target_is_directory=True)
        with pytest.raises(
            operator.SuccessorOperatorError,
            match="refusing to reuse residual DuckLake projection root",
        ):
            operator._claim_projection_root(paths, lock_custody)

    assert paths["projection_root"].is_symlink()
    assert foreign.is_dir()


def test_projection_stopped_continuity_rejects_foreign_provenance_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _operator()
    paths = operator._paths(tmp_path)
    raw_read = False

    def foreign_failure(*args: object, **kwargs: object) -> object:
        raise operator.SuccessorOperatorError("foreign provenance failure")

    def raw_provenance(_paths: dict[str, Path]) -> object:
        nonlocal raw_read
        raw_read = True
        return {}

    monkeypatch.setattr(operator, "_load_provenance", foreign_failure)
    monkeypatch.setattr(
        operator, "_load_lgcvf_live_raw_provenance_receipt", raw_provenance
    )

    with pytest.raises(
        operator.SuccessorOperatorError,
        match="foreign provenance failure",
    ):
        operator._load_projection_source_continuity(paths, root=tmp_path)
    assert raw_read is False


def test_projection_extension_policy_is_load_only_and_never_installs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources import (
        board_control_plane as board_control_plane_module,
    )

    operator = _operator()

    class LoadFailureConnection:
        def __init__(self) -> None:
            self.statements: list[str] = []

        def execute(self, statement: str) -> None:
            self.statements.append(statement)
            raise RuntimeError("injected local LOAD failure")

    connection = LoadFailureConnection()
    error = board_control_plane_module._try_load_extension(
        connection,
        "quack",
        allow_install=False,
    )
    assert connection.statements == ["LOAD quack"]
    assert "INSTALL disabled by policy" in error

    observed: dict[str, object] = {}
    sentinel = object()

    def open_projection(
        repo_root: Path,
        *,
        root: Path,
        allow_extension_install: bool,
    ) -> object:
        observed.update(
            {
                "repo_root": repo_root,
                "root": root,
                "allow_extension_install": allow_extension_install,
            }
        )
        return sentinel

    monkeypatch.setattr(
        board_control_plane_module,
        "open_board_control_plane",
        open_projection,
    )
    projection_root = Path("/proc/self/fd/97")
    assert operator._open_projection_plane(tmp_path, projection_root) is sentinel
    assert observed == {
        "repo_root": tmp_path,
        "root": projection_root,
        "allow_extension_install": False,
    }

    identity = type(
        "Identity",
        (),
        {"generation": 1, "schema_revision": "schema-v1"},
    )()
    owner_state = tmp_path / "owner"
    owner_state.mkdir()
    environment = operator._child_environment(
        token="test_token_value",
        identity=identity,
        owner_state=owner_state,
        root=tmp_path,
    )
    assert (
        environment[board_control_plane_module.BOARD_EXTENSION_INSTALL_POLICY_ENV]
        == board_control_plane_module.BOARD_EXTENSION_INSTALL_POLICY_LOAD_ONLY
        == operator.BOARD_EXTENSION_INSTALL_POLICY_LOAD_ONLY
    )
    assert environment[operator.LEGACY_BOARD_UNSTALL_POLICY_ENV] == "disabled"


def test_lgcvf_successor_owner_disables_legacy_board_unstall() -> None:
    tree = ast.parse(OPERATOR_PATH.read_text(encoding="utf-8"))
    launch = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_run_locked_successor"
    )
    calls = [
        node
        for node in ast.walk(launch)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "build_server"
    ]
    assert len(calls) == 1
    policy = next(
        keyword.value
        for keyword in calls[0].keywords
        if keyword.arg == "allow_legacy_board_unstall"
    )
    assert isinstance(policy, ast.Constant) and policy.value is False


def test_typed_retry_writer_and_reader_share_one_closed_vocabulary() -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        TaskRecord,
        TaskSourceIntegrityError,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.typed_database_task_source import (
        TypedDatabaseTaskSource,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import (
        TYPED_RETRYING_RECEIPT_OPERATIONS,
    )
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
        _DATABASE_CONTROL_ATTEMPT_OPERATIONS_BY_STATUS,
    )

    assert (
        _DATABASE_CONTROL_ATTEMPT_OPERATIONS_BY_STATUS["retrying"]
        is TYPED_RETRYING_RECEIPT_OPERATIONS
    )
    assert {
        "database_portal_capacity_retry",
        "database_portal_landed_completion_revalidation",
        "database_portal_protected_preservation_retry",
        "database_portal_protected_preservation_retry_recovery",
        "database_portal_validation_retry_successor_recovery",
    }.issubset(TYPED_RETRYING_RECEIPT_OPERATIONS)

    identity = {
        "attempt_id": "attempt:typed-retry-vocabulary",
        "claim_id": "claim:typed-retry-vocabulary",
        "lease_id": "lease:typed-retry-vocabulary",
        "owner_session_id": "session:typed-retry-vocabulary",
        "attempt_number": 2,
        "fencing_token": 7,
        "fence_epoch": 5,
    }
    revision = 9
    queue_reason = "database_portal_retry:vocabulary"
    retry_not_before_ms = 12_345
    cooldown = {
        "extension": {
            **identity,
            "expected_task_revision": revision - 1,
            "reason": queue_reason,
            "delay_ms": 0,
            "retry_not_before_ms": retry_not_before_ms,
        }
    }

    def retrying_task(operation: str) -> TaskRecord:
        return TaskRecord(
            task_cid="task:typed-retry-vocabulary",
            task_alias="TYPED-RETRY-VOCABULARY",
            goal_cid="goal:typed-retry-vocabulary",
            ordinal=1,
            status="retrying",
            revision=revision,
            body={
                "completion_receipt": {
                    "operation": operation,
                    **identity,
                    "queue_reason": queue_reason,
                    "backoff_ms": 0,
                    "retry_not_before_ms": retry_not_before_ms,
                    "control_expected_revision": revision - 1,
                }
            },
        )

    for operation in TYPED_RETRYING_RECEIPT_OPERATIONS:
        TypedDatabaseTaskSource._validate_retrying_cooldown_binding(
            retrying_task(operation),
            cooldown,
        )
    with pytest.raises(
        TaskSourceIntegrityError,
        match="not an admitted retry transition",
    ):
        TypedDatabaseTaskSource._validate_retrying_cooldown_binding(
            retrying_task("database_claim"),
            cooldown,
        )


def test_projection_plane_pins_all_writes_and_reopens_from_logical_paths(
    tmp_path: Path,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources import (
        board_control_plane as board_module,
    )

    operator = _operator()
    logical_root = tmp_path / "run-v39" / "ducklake-board-projection"
    logical_root.mkdir(mode=0o700, parents=True)
    descriptor = os.open(logical_root, os.O_RDONLY | os.O_DIRECTORY)
    descriptor_root = Path(f"/proc/self/fd/{descriptor}")
    board_namespace = (
        "logic-governed-compositional-verification-fabric-history-shadow-v1"
    )
    tasks = [
        {
            "task_id": "LGCVF-080",
            "status": "todo",
            "title": "descriptor-bound projection",
            "depends_on": [],
            "body": {},
        }
    ]
    try:
        with operator._open_projection_plane(tmp_path, descriptor_root) as plane:
            assert plane.backend == "ducklake+quack"
            assert plane.ducklake_attached is True
            registration = plane.register_board(
                board_namespace,
                source_path="stopped-checkpoint",
                source_kind="duckdb-stopped-checkpoint-observation",
                merge_target_branch=(
                    "agent/logic-governed-compositional-verification-fabric-v1"
                ),
                tasks=tasks,
            )
            binding = operator._bind_projection_logical_paths(
                plane,
                descriptor_root=descriptor_root,
                logical_root=logical_root,
                board_namespace=registration["board_namespace"],
            )
            assert binding["aggregate"]["board_count"] == 1
            assert binding["aggregate"]["ducklake_attached"] is True
            assert binding["logical_board_database"] == str(
                board_module.board_database_path(
                    logical_root,
                    registration["board_namespace"],
                )
            )
    finally:
        os.close(descriptor)

    with board_module.open_board_control_plane(
        tmp_path,
        root=logical_root,
        allow_extension_install=False,
    ) as reopened:
        aggregate = reopened.aggregate_boards()
        boards = reopened.list_boards()
        assert reopened.backend == "ducklake+quack"
        assert reopened.ducklake_attached is True
        assert aggregate["board_count"] == 1
        assert aggregate["ducklake_attached"] is True
        assert len(boards) == 1
        assert boards[0]["duckdb_path"] == str(
            board_module.board_database_path(
                logical_root,
                registration["board_namespace"],
            )
        )


def test_lgcvf_route_sealer_uses_one_temporary_exact_population_grant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.merge.database_worktree_registry import (
        process_birth_id,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources import (
        quack_state_client as client_module,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources import (
        typed_database_task_source as projection_module,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.task_execution_route_policy import (
        GROK_CODEX_EXECUTION_MODE,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import (
        TYPED_STATE_OWNER_SOCKET_ENV,
        TYPED_STATE_OWNER_TOKEN_ENV,
    )

    operator = _operator()
    policy = _lgcvf_test_execution_route_policy(operator)
    controller_birth = read_process_birth(os.getpid())
    assert controller_birth is not None
    birth_id = process_birth_id(controller_birth)
    events: list[object] = []

    class Server:
        def __init__(self) -> None:
            self.issued: list[dict[str, object]] = []
            self.revoked: list[str] = []

        def issue_typed_client_grant_record(
            self, **kwargs: object
        ) -> tuple[str, SimpleNamespace]:
            self.issued.append(dict(kwargs))
            return "temporary-route-sealer-token", SimpleNamespace(
                grant_id="grant:route-sealer"
            )

        def revoke_typed_client_grant(self, grant_id: str) -> None:
            self.revoked.append(grant_id)

    class Client:
        def __init__(self, **kwargs: object) -> None:
            events.append(
                (
                    "client.init",
                    dict(kwargs),
                    os.environ.get(TYPED_STATE_OWNER_TOKEN_ENV),
                    os.environ.get(TYPED_STATE_OWNER_SOCKET_ENV),
                )
            )

        def attach(self, endpoint: str, *, server_id: str) -> None:
            events.append(("client.attach", endpoint, server_id))

        def close(self) -> None:
            events.append("client.close")

    class Projection:
        def __init__(self, client: object, *, owns_client: bool) -> None:
            events.append(("projection.init", client, owns_client))

        def seal_execution_route_policy(
            self, execution_modes: dict[str, str]
        ) -> object:
            assert execution_modes == {
                alias: GROK_CODEX_EXECUTION_MODE
                for alias in operator.LGCVF_TASK_ALIASES
            }
            events.append("projection.seal")
            return policy

        def close(self) -> None:
            events.append("projection.close")

    monkeypatch.setattr(client_module, "QuackStateClient", Client)
    monkeypatch.setattr(
        projection_module,
        "TypedDatabaseTaskSource",
        Projection,
    )
    monkeypatch.setenv(TYPED_STATE_OWNER_TOKEN_ENV, "prior-test-token")
    monkeypatch.setenv(TYPED_STATE_OWNER_SOCKET_ENV, "/prior/test.sock")
    server = Server()
    owner_socket = tmp_path / "typed-owner.sock"
    program = SimpleNamespace(
        store_id="store:lgcvf-bootstrap-test",
        quack_endpoint="quack://lgcvf-bootstrap-test",
    )
    identity = SimpleNamespace(
        process_birth_id=birth_id,
        server_id="server:lgcvf-bootstrap-test",
    )

    result = operator._seal_lgcvf_execution_route_policy(
        server=server,
        program=program,
        identity=identity,
        controller_birth=controller_birth,
        owner_socket=owner_socket,
    )

    assert result is policy
    assert server.issued == [
        {
            "client_id": "lgcvf-route-sealer",
            "process_birth_id": birth_id,
            "allowed_operations": (
                "whoami_metadata",
                "load_store_generation",
                "executor_control_snapshot",
                "executor_task_projection_page",
            ),
            "allowed_command_operations": (),
            "peer_pid": os.getpid(),
            "ttl_seconds": 60.0,
        }
    ]
    assert server.revoked == ["grant:route-sealer"]
    assert events[0] == (
        "client.init",
        {
            "owner_id": "lgcvf-route-sealer",
            "store_id": program.store_id,
            "process_birth_id": birth_id,
        },
        "temporary-route-sealer-token",
        str(owner_socket),
    )
    assert events[1] == (
        "client.attach",
        program.quack_endpoint,
        identity.server_id,
    )
    assert events[-2:] == ["projection.close", "client.close"]
    assert os.environ[TYPED_STATE_OWNER_TOKEN_ENV] == "prior-test-token"
    assert os.environ[TYPED_STATE_OWNER_SOCKET_ENV] == "/prior/test.sock"


@pytest.mark.timeout(30)
def test_lgcvf_bootstrap_broker_mints_four_lane_grants_and_rotates_one(
    tmp_path: Path,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources.typed_database_task_source import (
        daemon_required_owner_command_operations,
        daemon_required_owner_operations,
    )
    from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
        current_process_birth,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.state_owner_bootstrap import (
        STATE_OWNER_BOOTSTRAP_REQUEST_SCHEMA,
        _send_frame,
    )

    operator = _operator()
    policy = _lgcvf_test_execution_route_policy(operator)
    daemon_script = tmp_path / "bootstrap_daemon.py"
    supervisor_script = tmp_path / "lane_supervisor.py"
    scheduler_script = tmp_path / "board_scheduler.py"
    gate = tmp_path / "scheduler.go"
    output_root = tmp_path / "results"
    output_root.mkdir(mode=0o700)
    daemon_script.write_text(
        r'''from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.task_sources.state_owner_bootstrap import (
    request_state_owner_bootstrap,
)

descriptor, client_id, store_id, output_path = sys.argv[1:]
credentials = request_state_owner_bootstrap(
    int(descriptor),
    client_id=client_id,
    store_id=store_id,
    timeout_seconds=15.0,
)
Path(output_path).write_text(
    json.dumps(
        {
            "pid": os.getpid(),
            "client_id": credentials.client_id,
            "store_id": credentials.store_id,
            "server_id": credentials.server_id,
            "process_birth_id": credentials.process_birth_id,
            "token_length": len(credentials.token),
            "execution_route_policy_id": (
                credentials.execution_route_policy.policy_id
            ),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    + "\n",
    encoding="utf-8",
)
time.sleep(2.0)
''',
        encoding="utf-8",
    )
    supervisor_script.write_text(
        r'''from __future__ import annotations

import subprocess
import sys
from pathlib import Path

(
    descriptor,
    session,
    lane_text,
    repository_root,
    output_root_text,
    store_id,
    daemon_script,
) = sys.argv[1:8]
lane = int(lane_text)
for attempt in range(2 if lane == 0 else 1):
    output_path = (
        Path(output_root_text) / f"lane-{lane}-attempt-{attempt}.json"
    )
    process = subprocess.Popen(
        [
            sys.executable,
            daemon_script,
            descriptor,
            f"database-implementation-daemon:{session}",
            store_id,
            str(output_path),
        ],
        cwd=repository_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        pass_fds=(int(descriptor),),
    )
    stdout, stderr = process.communicate(timeout=20.0)
    if process.returncode != 0:
        sys.stderr.buffer.write(stdout + stderr)
        raise SystemExit(process.returncode)
''',
        encoding="utf-8",
    )
    scheduler_script.write_text(
        r'''from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

(
    descriptor,
    gate_text,
    repository_root,
    output_root,
    store_id,
    supervisor_script,
    daemon_script,
) = sys.argv[1:]
deadline = time.monotonic() + 15.0
while not Path(gate_text).is_file():
    if time.monotonic() >= deadline:
        raise TimeoutError("scheduler gate timed out")
    time.sleep(0.01)
processes = []
for lane in range(4):
    session = f"lgcvf-quack-lane-{lane}"
    processes.append(
        subprocess.Popen(
            [
                sys.executable,
                supervisor_script,
                descriptor,
                session,
                str(lane),
                repository_root,
                output_root,
                store_id,
                daemon_script,
                "--board-namespace",
                "logic-governed-compositional-verification-fabric-v1",
                "--task-shard-count",
                "4",
                "--task-shard-index",
                str(lane),
                "--state-prefix",
                f"lgcvf_lane_{lane}",
                "--database-owner-session-id",
                session,
                "--state-owner-bootstrap-fd",
                descriptor,
                "--state-owner-bootstrap-store-id",
                store_id,
            ],
            cwd=repository_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            pass_fds=(int(descriptor),),
        )
    )
failed = []
for process in processes:
    stdout, stderr = process.communicate(timeout=25.0)
    if process.returncode != 0:
        failed.append(
            (process.pid, process.returncode, stdout.decode(), stderr.decode())
        )
if failed:
    raise RuntimeError(f"lane supervisors failed: {failed!r}")
''',
        encoding="utf-8",
    )

    class Server:
        def __init__(self) -> None:
            self.identity = SimpleNamespace(
                server_id="server:lgcvf-bootstrap-broker-test"
            )
            self.issued: list[dict[str, object]] = []
            self.revoked: list[str] = []
            self.renewed: list[str] = []

        def issue_typed_client_grant_record(
            self, **kwargs: object
        ) -> tuple[str, SimpleNamespace]:
            grant_id = f"grant:lgcvf-bootstrap:{len(self.issued) + 1}"
            self.issued.append({**kwargs, "grant_id": grant_id})
            return (
                f"lgcvf-bootstrap-token-{len(self.issued):04d}",
                SimpleNamespace(
                    grant_id=grant_id,
                    expires_at=(
                        int(time.time() * 1_000)
                        + int(float(kwargs["ttl_seconds"]) * 1_000)
                    ),
                ),
            )

        def revoke_typed_client_grant(self, grant_id: str) -> None:
            self.revoked.append(grant_id)

        def renew_typed_client_grant(
            self,
            grant_id: str,
            *,
            ttl_seconds: float,
        ) -> SimpleNamespace:
            assert any(
                record["grant_id"] == grant_id for record in self.issued
            )
            self.renewed.append(grant_id)
            return SimpleNamespace(
                grant_id=grant_id,
                expires_at=(
                    int(time.time() * 1_000) + int(ttl_seconds * 1_000)
                ),
            )

    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    address = "\0lgcvf-broker-test-" + os.urandom(8).hex()
    listener.bind(address)
    listener.listen(8)
    store_id = "store:lgcvf-bootstrap-broker-test"
    server = Server()
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        item
        for item in (str(ROOT), environment.get("PYTHONPATH", ""))
        if item
    )
    scheduler = subprocess.Popen(
        [
            sys.executable,
            str(scheduler_script),
            str(listener.fileno()),
            str(gate),
            str(ROOT),
            str(output_root),
            store_id,
            str(supervisor_script),
            str(daemon_script),
        ],
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        pass_fds=(listener.fileno(),),
        start_new_session=True,
    )
    scheduler_birth = _capture_process_birth(scheduler)
    broker = operator._LgcvfStateOwnerBootstrapBroker(
        channel=listener,
        descriptor=listener.fileno(),
        server=server,
        scheduler_birth=scheduler_birth,
        endpoint="quack://lgcvf-bootstrap-broker-test",
        socket_path=tmp_path / "typed-owner.sock",
        store_id=store_id,
        execution_route_policy=policy,
    )
    broker_stopped = False
    try:
        broker.start()
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as malformed:
            malformed.settimeout(2.0)
            malformed.connect(address)
            _send_frame(
                malformed,
                {
                    "schema": STATE_OWNER_BOOTSTRAP_REQUEST_SCHEMA,
                    "pid": "not-an-integer",
                    "process_birth": {},
                    "process_birth_id": "malformed",
                    "client_id": (
                        "database-implementation-daemon:"
                        + operator.LGCVF_DATABASE_OWNER_SESSIONS[0]
                    ),
                    "store_id": store_id,
                },
            )
        rejection_deadline = time.monotonic() + 2.0
        while broker.rejection_count < 1:
            assert time.monotonic() < rejection_deadline
            assert broker.failure == ""
            time.sleep(0.01)
        assert broker.last_rejection == "SuccessorOperatorError"
        assert broker._thread.is_alive()
        overflow_birth = current_process_birth().to_dict()
        overflow_birth["start_time_ticks"] = "OVERFLOW"
        overflow_payload = json.dumps(
            {
                "schema": STATE_OWNER_BOOTSTRAP_REQUEST_SCHEMA,
                "pid": os.getpid(),
                "process_birth": overflow_birth,
                "process_birth_id": "malformed-overflow",
                "client_id": (
                    "database-implementation-daemon:"
                    + operator.LGCVF_DATABASE_OWNER_SESSIONS[0]
                ),
                "store_id": store_id,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8").replace(b'"OVERFLOW"', b"1e10000")
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as malformed:
            malformed.settimeout(2.0)
            malformed.connect(address)
            malformed.sendall(
                len(overflow_payload).to_bytes(4, "big") + overflow_payload
            )
        rejection_deadline = time.monotonic() + 2.0
        while broker.rejection_count < 2:
            assert time.monotonic() < rejection_deadline
            assert broker.failure == ""
            time.sleep(0.01)
        assert broker.last_rejection == "SuccessorOperatorError"
        assert broker._thread.is_alive()
        gate.write_text("go\n", encoding="utf-8")
        ready_deadline = time.monotonic() + 15.0
        while broker.ready_sessions != operator.LGCVF_DATABASE_OWNER_SESSIONS:
            assert time.monotonic() < ready_deadline
            assert broker.failure == ""
            time.sleep(0.01)
        renewal_session = operator.LGCVF_DATABASE_OWNER_SESSIONS[1]
        with broker._lock:
            broker.current_by_session[renewal_session][
                "grant_renew_after"
            ] = 0.0
            renewal_grant = broker.active_grants[renewal_session]
        renewal_deadline = time.monotonic() + 5.0
        while renewal_grant not in server.renewed:
            assert time.monotonic() < renewal_deadline
            assert broker.failure == ""
            time.sleep(0.01)
        stdout, stderr = scheduler.communicate(timeout=25.0)
        assert scheduler.returncode == 0, (stdout, stderr, broker.failure)
        assert broker.failure == ""
        with broker._lock:
            broker.current_by_session[renewal_session][
                "grant_renew_after"
            ] = 0.0
        renewed_after_live_birth = len(server.renewed)
        broker._renew_due_grants()
        assert len(server.renewed) == renewed_after_live_birth
        assert broker.ready_sessions == operator.LGCVF_DATABASE_OWNER_SESSIONS
        assert len(server.issued) == 5
        assert len(broker.active_grants) == 4
        assert set(broker.active_grants) == set(
            operator.LGCVF_DATABASE_OWNER_SESSIONS
        )

        results = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(output_root.glob("*.json"))
        ]
        assert len(results) == 5
        assert all(
            result["execution_route_policy_id"] == policy.policy_id
            for result in results
        )
        assert all(result["token_length"] >= 16 for result in results)
        assert {result["client_id"] for result in results} == {
            f"database-implementation-daemon:{session}"
            for session in operator.LGCVF_DATABASE_OWNER_SESSIONS
        }
        assert {result["store_id"] for result in results} == {store_id}
        assert {result["server_id"] for result in results} == {
            server.identity.server_id
        }
        assert len({result["process_birth_id"] for result in results}) == 5

        expected_operations = daemon_required_owner_operations()
        expected_commands = daemon_required_owner_command_operations()
        assert all(
            record["allowed_operations"] == expected_operations
            and record["allowed_command_operations"] == expected_commands
            and record["ttl_seconds"]
            == operator.INTERNAL_CLIENT_GRANT_TTL_SECONDS
            for record in server.issued
        )
        assert {record["client_id"] for record in server.issued} == {
            f"database-implementation-daemon:{session}"
            for session in operator.LGCVF_DATABASE_OWNER_SESSIONS
        }
        assert {
            int(record["peer_pid"]) for record in server.issued
        } == {int(result["pid"]) for result in results}

        lane_zero_client = (
            "database-implementation-daemon:"
            + operator.LGCVF_DATABASE_OWNER_SESSIONS[0]
        )
        lane_zero_grants = [
            str(record["grant_id"])
            for record in server.issued
            if record["client_id"] == lane_zero_client
        ]
        assert len(lane_zero_grants) == 2
        assert lane_zero_grants[0] in server.revoked
        assert broker.active_grants[
            operator.LGCVF_DATABASE_OWNER_SESSIONS[0]
        ] == lane_zero_grants[1]

        broker.stop()
        broker_stopped = True
        assert broker._thread.is_alive() is False
        assert broker.active_grants == {}
        assert set(server.revoked) == {
            str(record["grant_id"]) for record in server.issued
        }
    finally:
        if scheduler.poll() is None:
            try:
                os.killpg(scheduler.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            scheduler.wait(timeout=5.0)
        if not broker_stopped:
            broker.stop()
        listener.close()


@pytest.mark.timeout(10)
def test_lgcvf_broker_fences_separate_session_births_before_revocation(
    tmp_path: Path,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
        OwnerLiveness,
        current_process_birth,
        owner_liveness,
    )

    operator = _operator()
    policy = _lgcvf_test_execution_route_policy(operator)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind("\0lgcvf-broker-stop-test-" + os.urandom(8).hex())
    listener.listen(2)
    children = [
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                (
                    "import signal,time;"
                    "signal.signal(signal.SIGTERM,lambda *_:None);"
                    "print('ready',flush=True);time.sleep(60)"
                ),
            ],
            cwd=tmp_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        for _index in range(2)
    ]
    births = tuple(_capture_process_birth(child) for child in children)
    assert all(child.stdout is not None for child in children)
    assert all(child.stdout.readline().strip() == "ready" for child in children)
    revoked_after_fence: list[str] = []

    class Server:
        def revoke_typed_client_grant(self, grant_id: str) -> None:
            assert all(
                owner_liveness(birth) is OwnerLiveness.DEAD
                for birth in births
            )
            revoked_after_fence.append(grant_id)

    broker = operator._LgcvfStateOwnerBootstrapBroker(
        channel=listener,
        descriptor=listener.fileno(),
        server=Server(),
        scheduler_birth=current_process_birth(),
        endpoint="quack://lgcvf-bootstrap-stop-test",
        socket_path=tmp_path / "typed-owner.sock",
        store_id="store:lgcvf-bootstrap-stop-test",
        execution_route_policy=policy,
        process_stop_grace_seconds=0.1,
    )
    session = operator.LGCVF_DATABASE_OWNER_SESSIONS[0]
    broker.current_by_session[session] = {
        "supervisor_process_birth": births[0].to_dict(),
        "daemon_process_birth": births[1].to_dict(),
    }
    broker.active_grants[session] = "grant:lgcvf-bootstrap-stop-test"
    try:
        broker.stop()
        assert revoked_after_fence == [
            "grant:lgcvf-bootstrap-stop-test"
        ]
        assert all(child.poll() is not None for child in children)
        assert broker.active_grants == {}
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5.0)


OWNER_PROCESS = r"""
import importlib.util
import json
import os
import select
import sys
import time
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import build_server


def emit(payload):
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")), flush=True)


configuration = json.loads(sys.stdin.readline())
spec = importlib.util.spec_from_file_location(
    "lgcvf_quack_successor_owner", configuration["operator_path"]
)
if spec is None or spec.loader is None:
    raise RuntimeError("successor operator module is unavailable")
operator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(operator)
os.environ["IPFS_ACCELERATE_AGENT_STATE_STORE_GENERATION"] = configuration[
    "store_generation"
]
server = None
try:
    server = build_server(
        database_path=Path(configuration["database"]),
        state_dir=Path(configuration["owner_state"]),
        host="127.0.0.1",
        port=0,
        store_id=configuration["store_id"],
        repository_id="repository:lgcvf-quack-successor-test",
        secret_handle=configuration["secret_handle"],
        migrate=operator.datasets_profile_migration,
        typed_command_socket_path=Path(configuration["typed_socket"]),
    )
    identity = server.start()
    if server._vault is None:
        raise RuntimeError("owner token vault is unavailable")
    token = server._vault.resolve(identity.secret_handle)
    emit(
        {
            "event": "ready",
            "identity": identity.to_dict(),
            "mutation_inbox": str(server.mutation_inbox_path()),
            "token": token,
            "typed_socket": str(server.typed_command_socket_path()),
        }
    )
    stop_requested = False
    while not stop_requested:
        server.service_mutation_inbox(max_requests=32)
        readable, _, _ = select.select([sys.stdin], [], [], 0.005)
        if not readable:
            continue
        line = sys.stdin.readline()
        if not line:
            stop_requested = True
            continue
        request = json.loads(line)
        action = request.get("action")
        if action == "observe":
            row = server._connection.execute(
                "SELECT status, revision FROM tasks WHERE task_cid = 'task:test'"
            ).fetchone()
            emit(
                {
                    "event": "observed",
                    "lifecycle": server.lifecycle.value,
                    "task": [str(row[0]), int(row[1])],
                }
            )
        elif action == "stop":
            stop_requested = True
        else:
            raise RuntimeError("owner control action is invalid")
    server.stop()
    emit({"event": "stopped"})
except BaseException as exc:
    emit(
        {
            "event": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    )
    raise
"""


WORKER = r"""
import json
import os
import sys
import time
from argparse import Namespace
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
    DatabaseTaskSource,
    TaskSourceConflictError,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon_runner import (
    resolve_database_implementation_paths,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    DatabaseImplementationDaemon,
)

endpoint, lane_text, lane_root_text, ready_text, gate_text, result_text = sys.argv[1:]
lane = int(lane_text)
lane_root = Path(lane_root_text)
ready = Path(ready_text)
gate = Path(gate_text)
result_path = Path(result_text)
sidecars = resolve_database_implementation_paths(
    Namespace(
        state_dir=lane_root,
        state_prefix=f"lgcvf_lane_{lane}",
        authority_mode="quack",
        database_path=None,
        todo_path=None,
        coordination_path=None,
    ),
    authority_mode="quack",
)
source = None
daemon = None
payload = {
    "lane": lane,
    "database_path": str(sidecars["database_path"]),
    "coordination_path": str(sidecars["coordination_path"]),
    "execution_path": str(sidecars["execution_path"]),
}
try:
    source = DatabaseTaskSource(
        endpoint,
        install_schema=False,
        owner_id=f"lgcvf-quack-test:lane:{lane}",
    )
    daemon = DatabaseImplementationDaemon(
        database_path=sidecars["database_path"],
        coordination_path=sidecars["coordination_path"],
        execution_path=sidecars["execution_path"],
        owner_session_id=f"lgcvf-quack-test:lane:{lane}",
        # Provision only the lane-private embedded sidecars here. The shared
        # task CAS below still uses the independently authenticated Quack
        # source; current runtime policy correctly refuses to inject that
        # untyped adapter into DatabaseImplementationDaemon.
        authority_mode="embedded_exclusive",
        state_schema_revision="",
        task_source_kind="duckdb",
        task_shard_count=4,
        task_shard_index=lane,
        strict_task_sharding=True,
    )
    task = source.get_task("LGCVF-TEST")
    if task is None or task.status != "ready" or task.revision != 1:
        raise RuntimeError("lane did not observe the exact initial task head")
    ready.write_text("ready\n", encoding="utf-8")
    deadline = time.monotonic() + 30.0
    while not gate.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError("CAS start gate timed out")
        time.sleep(0.005)
    try:
        result = source.compare_and_set_status(
            "LGCVF-TEST",
            expected_revision=1,
            status="in_progress",
        )
    except TaskSourceConflictError:
        payload["outcome"] = "conflict"
    else:
        payload["outcome"] = "success"
        payload["revision"] = result.revision
except BaseException as exc:
    payload["outcome"] = "error"
    payload["error_type"] = type(exc).__name__
    payload["error"] = str(exc)
finally:
    if daemon is not None:
        daemon.close()
    if source is not None:
        source.close()
    result_path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
"""


def _wait_for_paths(paths: list[Path], processes: list[subprocess.Popen[str]]) -> None:
    deadline = time.monotonic() + 45.0
    while time.monotonic() < deadline:
        if all(path.is_file() for path in paths):
            return
        exited = [
            process.returncode for process in processes if process.poll() is not None
        ]
        if exited:
            raise AssertionError(f"worker exited before the CAS gate: {exited}")
        time.sleep(0.02)
    raise AssertionError("workers did not reach the CAS gate")


def _read_owner_event(
    process: subprocess.Popen[str],
    *,
    timeout_seconds: float,
) -> dict[str, object]:
    assert process.stdout is not None
    deadline = time.monotonic() + timeout_seconds
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise AssertionError("timed out waiting for owner control response")
        readable, _, _ = select.select([process.stdout], [], [], remaining)
        if not readable:
            continue
        line = process.stdout.readline()
        if not line:
            raise AssertionError(
                f"owner control pipe closed with returncode={process.poll()}"
            )
        payload = json.loads(line)
        assert isinstance(payload, dict)
        return payload


def _capture_process_birth(process: subprocess.Popen[str]) -> ProcessBirthIdentity:
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise AssertionError("owner exited before process birth was captured")
        try:
            birth = read_process_birth(process.pid)
        except OSError:
            birth = None
        if birth is not None:
            return birth
        time.sleep(0.01)
    raise AssertionError("owner process birth could not be captured")


def _signal_exact_process(
    process: subprocess.Popen[str],
    birth: ProcessBirthIdentity,
    signum: int,
) -> None:
    if process.poll() is not None:
        return
    if owner_liveness(birth) is not OwnerLiveness.ALIVE:
        return
    try:
        process_group = os.getpgid(birth.pid)
        if process_group == birth.pid:
            os.killpg(process_group, signum)
        else:
            os.kill(birth.pid, signum)
    except ProcessLookupError:
        return


def _bounded_stop_owner(
    process: subprocess.Popen[str],
    birth: ProcessBirthIdentity,
) -> dict[str, object]:
    event: dict[str, object] = {}
    if process.poll() is None:
        assert process.stdin is not None
        try:
            process.stdin.write('{"action":"stop"}\n')
            process.stdin.flush()
            event = _read_owner_event(process, timeout_seconds=5.0)
        except (AssertionError, BrokenPipeError, OSError):
            event = {}
    try:
        process.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        _signal_exact_process(process, birth, signal.SIGTERM)
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            _signal_exact_process(process, birth, signal.SIGKILL)
            process.wait(timeout=2.0)
    stdout_tail, stderr = process.communicate(timeout=1.0)
    return {
        "event": event,
        "returncode": process.returncode,
        "stdout_tail": stdout_tail,
        "stderr": stderr,
    }


def _assert_secret_absent_from_regular_files(root: Path, secret: str) -> None:
    needle = secret.encode("ascii")
    for candidate in root.rglob("*"):
        try:
            candidate.lstat()
        except OSError:
            continue
        if not candidate.is_file() or candidate.is_symlink():
            continue
        with candidate.open("rb") as stream:
            while True:
                block = stream.read(1024 * 1024)
                if not block:
                    break
                assert needle not in block, f"raw Quack token persisted in {candidate}"


def test_real_four_process_quack_cas_has_one_winner_and_private_sidecars(
    tmp_path: Path,
) -> None:
    capability = probe_quack_capabilities()
    if not capability.passes_health_check:
        pytest.skip(
            "preinstalled pinned Quack capability unavailable: "
            f"{capability.status.value}/{capability.reason_code}"
        )

    operator = _operator()
    runtime = tmp_path / "run-v23"
    database = runtime / "control.duckdb"
    owner_state = runtime / "quack-owner"
    logical_generation = "lgcvf-synthetic-v1"
    _seed_datasets_profile(database)
    test_paths = operator._paths(tmp_path)
    operator._prepare_private_owner_socket(test_paths["owner_socket"])
    owner_environment = {
        name: os.environ[name]
        for name in ("HOME", "LANG", "LC_ALL", "PATH", "TMPDIR")
        if name in os.environ
    }
    owner_environment.update(
        {
            "PYTHONPATH": str(ROOT),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    owner = subprocess.Popen(
        [sys.executable, "-c", OWNER_PROCESS],
        cwd=ROOT,
        env=owner_environment,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    owner_birth = _capture_process_birth(owner)
    owner_shutdown: dict[str, object] | None = None
    processes: list[subprocess.Popen[str]] = []
    try:
        owner_configuration = {
            "database": str(database),
            "operator_path": str(OPERATOR_PATH),
            "owner_state": str(owner_state),
            "secret_handle": operator.SECRET_HANDLE,
            "store_generation": logical_generation,
            "store_id": str(database),
            "typed_socket": str(test_paths["owner_socket"]),
        }
        assert owner.stdin is not None
        owner.stdin.write(
            json.dumps(owner_configuration, sort_keys=True, separators=(",", ":"))
            + "\n"
        )
        owner.stdin.flush()
        ready = _read_owner_event(owner, timeout_seconds=30.0)
        assert ready.get("event") == "ready", ready
        identity = ready.get("identity")
        assert isinstance(identity, dict)
        token = ready.get("token")
        assert isinstance(token, str) and token
        endpoint = str(identity["listen_uri"])
        typed_socket = Path(str(ready["typed_socket"]))
        mutation_inbox = Path(str(ready["mutation_inbox"]))
        assert typed_socket == test_paths["owner_socket"]
        assert len(os.fsencode(typed_socket)) <= operator.UNIX_SOCKET_PATH_CEILING
        assert (
            token.encode("ascii") not in Path(f"/proc/{owner.pid}/cmdline").read_bytes()
        )

        sink = operator._token_sink(owner_state)
        gate = tmp_path / "cas.go"
        lane_roots = [runtime / "state" / f"lane-{index}" for index in range(4)]
        ready_paths = [tmp_path / f"lane-{index}.ready" for index in range(4)]
        result_paths = [tmp_path / f"lane-{index}.result.json" for index in range(4)]
        for lane_root in lane_roots:
            lane_root.mkdir(parents=True, mode=0o700)
        environment = dict(owner_environment)
        environment.update(
            {
                operator.TOKEN_ENV: token,
                operator.TOKEN_FILE_ENV: str(sink),
                "IPFS_ACCELERATE_AGENT_STATE_AUTHORITY_MODE": "quack",
                "IPFS_ACCELERATE_AGENT_TASK_SOURCE_KIND": "duckdb",
                "IPFS_ACCELERATE_AGENT_STATE_FAILOVER_POLICY": "fail_closed",
                "IPFS_ACCELERATE_AGENT_STATE_ENDPOINT_SECRET_HANDLE": str(
                    identity["secret_handle"]
                ),
                "IPFS_ACCELERATE_AGENT_QUACK_ENDPOINT": endpoint,
                "IPFS_ACCELERATE_AGENT_STATE_STORE_ID": str(identity["store_id"]),
                "IPFS_ACCELERATE_AGENT_STATE_STORE_GENERATION": logical_generation,
                "IPFS_ACCELERATE_AGENT_STATE_SCHEMA_REVISION": (
                    "datasets-authoritative-operational-v1"
                ),
                "IPFS_ACCELERATE_AGENT_STATE_STORE_LIVE_GENERATION": str(
                    identity["generation"]
                ),
                "IPFS_ACCELERATE_AGENT_STATE_LIVE_SCHEMA_REVISION": str(
                    identity["schema_revision"]
                ),
                "IPFS_ACCELERATE_AGENT_RUNTIME_REGISTRY_PATH": str(owner_state),
                "IPFS_ACCELERATE_AGENT_QUACK_MUTATION_DIR": str(mutation_inbox),
                "IPFS_ACCELERATE_LIFECYCLE_REPOSITORY_ROOT": str(tmp_path),
            }
        )
        for lane in range(4):
            command = [
                sys.executable,
                "-c",
                WORKER,
                endpoint,
                str(lane),
                str(lane_roots[lane]),
                str(ready_paths[lane]),
                str(gate),
                str(result_paths[lane]),
            ]
            assert all(token not in item for item in command)
            processes.append(
                subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    start_new_session=True,
                )
            )
        _wait_for_paths(ready_paths, processes)
        for process in processes:
            cmdline = Path(f"/proc/{process.pid}/cmdline").read_bytes()
            assert token.encode("ascii") not in cmdline
        gate.write_text("go\n", encoding="utf-8")
        outputs = [process.communicate(timeout=45.0) for process in processes]
        assert all(process.returncode == 0 for process in processes), outputs
        results = [
            json.loads(path.read_text(encoding="utf-8")) for path in result_paths
        ]
        assert sorted(result["outcome"] for result in results) == [
            "conflict",
            "conflict",
            "conflict",
            "success",
        ]
        assert {result.get("revision") for result in results} == {None, 2}
        for field in ("database_path", "coordination_path", "execution_path"):
            assert len({result[field] for result in results}) == 4
        assert (
            len(
                {
                    result[field]
                    for result in results
                    for field in (
                        "database_path",
                        "coordination_path",
                        "execution_path",
                    )
                }
            )
            == 12
        )
        for result in results:
            assert Path(result["coordination_path"]).is_file()
            assert Path(result["execution_path"]).is_file()
        owner.stdin.write('{"action":"observe"}\n')
        owner.stdin.flush()
        observed = _read_owner_event(owner, timeout_seconds=10.0)
        assert observed == {
            "event": "observed",
            "lifecycle": "ready",
            "task": ["in_progress", 2],
        }
        assert not tuple(runtime.rglob("*.quack-token"))
        _assert_secret_absent_from_regular_files(tmp_path, token)
    finally:
        for process in processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=3.0)
        owner_shutdown = _bounded_stop_owner(owner, owner_birth)
    assert owner_shutdown["event"] == {"event": "stopped"}, owner_shutdown
    assert owner_shutdown["returncode"] == 0, owner_shutdown
    assert not (owner_state / "typed-state-owner.token").exists()
