from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import json
import os
import select
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType

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
    stage = tmp_path / "run-v31.stage-test"
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
            "control": "run-v31.stage-test/control.duckdb",
            "coordination": "run-v31.stage-test/control.coordination.duckdb",
            "execution": "run-v31.stage-test/control.execution.duckdb",
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
                    "control": "run-v31.stage-test/control.duckdb",
                    "coordination": (
                        "run-v31.stage-test/control.coordination.duckdb"
                    ),
                    "execution": "run-v31.stage-test/control.execution.duckdb",
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

    def load_raw(_paths: object) -> dict[str, str]:
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
        "native",
        "provenance",
        "post_provenance_preload",
        "manifest",
        "audit",
        "final_continuity",
        "retarget",
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
    projection_root = tmp_path / "projection"
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
        authority_mode="quack",
        task_source_kind="duckdb",
        quack_uri=endpoint,
        task_source=source,
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
            lane_root.mkdir(parents=True)
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
