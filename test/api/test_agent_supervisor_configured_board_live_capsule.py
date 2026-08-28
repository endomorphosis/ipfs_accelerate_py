from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from ipfs_accelerate_py.agent_implementation_route import (
    AgentImplementationControlPlanePin,
)
from ipfs_accelerate_py.agent_supervisor.core.multiformats_identity import (
    cid_for_dag_json,
)
from ipfs_accelerate_py.agent_supervisor.runtime import (
    configured_board_extension_projection as extension_projection,
)
from ipfs_accelerate_py.agent_supervisor.runtime import (
    configured_board_live_capsule as capsule,
)
from ipfs_accelerate_py.agent_supervisor.runtime import (
    configured_board_scheduler as scheduler,
)
from ipfs_accelerate_py.agent_supervisor.runtime import (
    multi_supervisor_runner as runner,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_supervisor as implementation,
)

_SYNTHETIC_QUACK_EXTENSION = b"synthetic-quack-extension-v1\x00\x01"
_SYNTHETIC_QUACK_INFO = b'{"extension":"quack","synthetic":true}\n'


def _restore_projection_permissions(root: Path) -> None:
    if not root.exists() or root.is_symlink():
        return
    for current, directories, files in os.walk(root):
        current_path = Path(current)
        os.chmod(current_path, 0o700)
        for name in directories:
            candidate = current_path / name
            if not candidate.is_symlink():
                os.chmod(candidate, 0o700)
        for name in files:
            candidate = current_path / name
            if not candidate.is_symlink():
                os.chmod(candidate, 0o600)


@pytest.fixture
def quack_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[extension_projection.ConfiguredBoardExtensionPin, Path]]:
    sources = tmp_path / "quack-extension-sources"
    sources.mkdir()
    extension = sources / "quack.duckdb_extension"
    info = sources / "quack.duckdb_extension.info"
    extension.write_bytes(_SYNTHETIC_QUACK_EXTENSION)
    info.write_bytes(_SYNTHETIC_QUACK_INFO)
    pin = extension_projection.inspect_configured_board_extension_sources(
        extension,
        info,
        name="quack",
        engine_version="v1.5.5",
        platform="linux_amd64",
    )
    projection_parent = tmp_path / "private-extension-projection"
    projection_parent.mkdir(mode=0o700)
    home = extension_projection.project_configured_board_extension_home(
        pin,
        extension_path=extension,
        info_path=info,
        parent=projection_parent,
    )
    extension_directory = home / ".duckdb/extensions"
    monkeypatch.setenv(
        extension_projection.CONFIGURED_BOARD_EXTENSION_DIRECTORY_ENV,
        str(extension_directory.resolve(strict=True)),
    )
    try:
        yield pin, home
    finally:
        _restore_projection_permissions(home)


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _seed(tmp_path: Path) -> tuple[Path, tuple[str, ...]]:
    root = tmp_path / "repository"
    root.mkdir()
    _git(root, "init", "-q")
    paths = (
        "config/scheduler.json",
        "docs/plan.md",
        "scripts/validate.py",
    )
    for relative in paths:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"accepted:{relative}\n", encoding="utf-8")
    _git(root, "add", ".")
    _git(
        root,
        "-c",
        "user.name=Capsule Test",
        "-c",
        "user.email=capsule@example.invalid",
        "commit",
        "-m",
        "seed",
    )
    return root, paths


def _seed_handoff(tmp_path: Path) -> tuple[Path, tuple[str, ...]]:
    root, paths = _seed(tmp_path)
    entry = root / runner.PLAN_BOUND_ACCEPTED_ENTRY_PATH
    entry.parent.mkdir(parents=True, exist_ok=True)
    entry.write_text("raise SystemExit(0)\n", encoding="utf-8")
    _git(root, "add", entry.relative_to(root).as_posix())
    _git(
        root,
        "-c",
        "user.name=Capsule Test",
        "-c",
        "user.email=capsule@example.invalid",
        "commit",
        "-m",
        "add accepted supervisor entry",
    )
    return root, paths


def _pin(root: Path) -> AgentImplementationControlPlanePin:
    return AgentImplementationControlPlanePin(
        schema="ipfs_accelerate_py.agent_supervisor.accepted-control-plane@2",
        runner_path="ipfs_accelerate_py/agent_supervisor/runtime/multi_supervisor_runner.py",
        runner_sha256="sha256:" + "1" * 64,
        capsule_root="/sealed/capsule",
        capsule_id="sha256:" + "2" * 64,
        source_head=_git(root, "rev-parse", "HEAD"),
        source_tree=_git(root, "rev-parse", "HEAD^{tree}"),
        archive_sha256="sha256:" + "3" * 64,
    )


def _authority() -> dict[str, object]:
    return {
        "authority_mode": "quack",
        "task_source_kind": "duckdb",
        "schema_revision": "datasets-authoritative-operational-v1",
        "failover_policy": "fail_closed",
        "store_id": "data/control.duckdb",
        "store_generation": 2,
        "endpoint_secret_handle": "env://TEST_QUACK_TOKEN",
    }


def _admission(
    root: Path,
    paths: tuple[str, ...],
    quack_projection_pin: extension_projection.ConfiguredBoardExtensionPin,
) -> capsule.ConfiguredBoardLiveCapsuleAdmission:
    return capsule.build_configured_board_live_capsule_admission(
        repo_root=root,
        board_namespace="test-board-v1",
        plan_revision="TEST-PLAN-R2",
        task_prefix="TEST-",
        config_path="config/scheduler.json",
        configuration_root=cid_for_dag_json({"configuration": "root"}),
        control_paths=paths,
        control_plane_pin=_pin(root),
        native_authorization_id="sha256:" + "4" * 64,
        native_dependency_id="sha256:" + "5" * 64,
        native_python_executable_sha256=runner._python_executable_sha256(sys.executable)[1],
        quack_extension_projection=quack_projection_pin,
        database_authority=_authority(),
        max_lanes=2,
        strict_task_sharding=True,
    )


def _native_launch() -> SimpleNamespace:
    launch_json = json.dumps(
        {
            "schema": "test-native-launch@1",
            "accepted_authorization_id": "sha256:" + "4" * 64,
            "dependency_id": "sha256:" + "5" * 64,
            "descriptor": 19,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return SimpleNamespace(
        accepted_authorization_id="sha256:" + "4" * 64,
        pin=SimpleNamespace(
            dependency_id="sha256:" + "5" * 64,
            python_executable_sha256=runner._python_executable_sha256(sys.executable)[1],
        ),
        descriptor=SimpleNamespace(descriptor=19),
        pass_fds=(19,),
        to_json=lambda: launch_json,
    )


def test_policy_is_closed_required_and_canonical() -> None:
    policy = {
        "schema": capsule.CONFIGURED_BOARD_LIVE_CAPSULE_POLICY_SCHEMA,
        "required": True,
        "control_paths": ["config/a.json", "docs/b.md"],
    }
    assert capsule.parse_configured_board_live_capsule_policy(policy) == (
        "config/a.json",
        "docs/b.md",
    )
    with pytest.raises(capsule.ConfiguredBoardLiveCapsuleError):
        capsule.parse_configured_board_live_capsule_policy({**policy, "unreviewed": True})
    with pytest.raises(capsule.ConfiguredBoardLiveCapsuleError):
        capsule.parse_configured_board_live_capsule_policy(
            {**policy, "control_paths": list(reversed(policy["control_paths"]))}
        )
    with pytest.raises(capsule.ConfiguredBoardLiveCapsuleError):
        capsule.parse_configured_board_live_capsule_policy({**policy, "required": False})


def test_admission_binds_board_source_controls_and_quack(
    tmp_path: Path,
    quack_projection: tuple[
        extension_projection.ConfiguredBoardExtensionPin,
        Path,
    ],
) -> None:
    root, raw_paths = _seed(tmp_path)
    paths = tuple(sorted(raw_paths))
    projection_pin, projection_home = quack_projection
    admission = _admission(root, paths, projection_pin)

    assert admission.board_namespace == "test-board-v1"
    assert admission.configuration_root == cid_for_dag_json({"configuration": "root"})
    assert admission.source_head == _git(root, "rev-parse", "HEAD")
    assert admission.source_tree == _git(root, "rev-parse", "HEAD^{tree}")
    assert tuple(item["path"] for item in admission.control_artifacts) == paths
    assert admission.database_authority["authority_mode"] == "quack"
    assert admission.database_authority["failover_policy"] == "fail_closed"
    assert admission.quack_extension_projection == projection_pin
    assert os.environ[extension_projection.CONFIGURED_BOARD_EXTENSION_DIRECTORY_ENV] == str(
        projection_home / ".duckdb/extensions"
    )
    assert "TEST_QUACK_TOKEN" in admission.to_json()
    assert "secret-value" not in admission.to_json()
    assert capsule.parse_configured_board_live_capsule_admission(admission.to_json()) == admission
    duplicate = admission.to_json().replace(
        '"max_lanes":2',
        '"max_lanes":2,"max_lanes":2',
    )
    with pytest.raises(
        capsule.ConfiguredBoardLiveCapsuleError,
        match="repeats JSON key",
    ):
        capsule.parse_configured_board_live_capsule_admission(duplicate)

    forged = json.loads(admission.to_json())
    forged["max_lanes"] = 3
    with pytest.raises(
        capsule.ConfiguredBoardLiveCapsuleError,
        match="identity drifted",
    ):
        capsule.parse_configured_board_live_capsule_admission(forged)

    malformed = json.loads(admission.to_json())
    malformed["configuration_root"] = "b" + "a" * 58
    malformed_without_identity = dict(malformed)
    malformed_without_identity.pop("admission_cid")
    malformed["admission_cid"] = capsule._cid(malformed_without_identity)
    with pytest.raises(
        capsule.ConfiguredBoardLiveCapsuleError,
        match="canonical CIDv1 DAG-JSON",
    ):
        capsule.parse_configured_board_live_capsule_admission(malformed)

    assert admission.admission_cid.startswith("baguqeera")


def test_admission_rejects_dirty_or_head_divergent_controls(
    tmp_path: Path,
    quack_projection: tuple[
        extension_projection.ConfiguredBoardExtensionPin,
        Path,
    ],
) -> None:
    root, raw_paths = _seed(tmp_path)
    paths = tuple(sorted(raw_paths))
    projection_pin, _projection_home = quack_projection
    (root / "untracked.py").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(
        capsule.ConfiguredBoardLiveCapsuleError,
        match="clean accepted checkout",
    ):
        _admission(root, paths, projection_pin)

    (root / "untracked.py").unlink()
    (root / "docs/plan.md").write_text("changed\n", encoding="utf-8")
    with pytest.raises(
        capsule.ConfiguredBoardLiveCapsuleError,
        match="clean accepted checkout",
    ):
        _admission(root, paths, projection_pin)


def test_verification_rechecks_capsule_source_and_control_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    quack_projection: tuple[
        extension_projection.ConfiguredBoardExtensionPin,
        Path,
    ],
) -> None:
    root, raw_paths = _seed(tmp_path)
    paths = tuple(sorted(raw_paths))
    pin = _pin(root)
    projection_pin, projection_home = quack_projection
    admission = _admission(root, paths, projection_pin)
    native_launch = _native_launch()
    monkeypatch.setattr(
        capsule,
        "verify_agent_implementation_sealed_control_plane",
        lambda _pin, descriptor: f"/proc/self/fd/{descriptor}",
    )
    monkeypatch.setattr(
        capsule,
        "verify_agent_supervisor_native_dependency_sealed_fd",
        lambda launch: f"/proc/self/fd/{launch.descriptor.descriptor}",
    )

    assert (
        capsule.verify_configured_board_live_capsule(
            admission,
            control_plane_pin=pin,
            control_plane_descriptor=9,
            native_dependency_launch=native_launch,
            repo_root=root,
            expected_board_namespace="test-board-v1",
            expected_config_path="config/scheduler.json",
        )
        == admission
    )

    monkeypatch.setenv(
        extension_projection.CONFIGURED_BOARD_EXTENSION_DIRECTORY_ENV,
        str(projection_home / ".duckdb/foreign"),
    )
    with pytest.raises(
        capsule.ConfiguredBoardLiveCapsuleError,
        match="projection environment is invalid",
    ):
        capsule.verify_configured_board_live_capsule(
            admission,
            control_plane_pin=pin,
            control_plane_descriptor=9,
            native_dependency_launch=native_launch,
            repo_root=root,
        )
    monkeypatch.setenv(
        extension_projection.CONFIGURED_BOARD_EXTENSION_DIRECTORY_ENV,
        str(projection_home / ".duckdb/extensions"),
    )

    (root / "docs/plan.md").write_text("drift\n", encoding="utf-8")
    with pytest.raises(capsule.ConfiguredBoardLiveCapsuleError):
        capsule.verify_configured_board_live_capsule(
            admission,
            control_plane_pin=pin,
            control_plane_descriptor=9,
            native_dependency_launch=native_launch,
            repo_root=root,
        )


def test_prediction_or_completion_fields_cannot_enter_admission(
    tmp_path: Path,
    quack_projection: tuple[
        extension_projection.ConfiguredBoardExtensionPin,
        Path,
    ],
) -> None:
    root, raw_paths = _seed(tmp_path)
    projection_pin, _projection_home = quack_projection
    admission = _admission(root, tuple(sorted(raw_paths)), projection_pin)
    payload = admission.as_dict()
    payload["worker_approved"] = True
    with pytest.raises(
        capsule.ConfiguredBoardLiveCapsuleError,
        match="fields are noncanonical",
    ):
        capsule.parse_configured_board_live_capsule_admission(payload)


def test_scheduler_runner_and_daemon_preserve_one_live_capsule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    quack_projection: tuple[
        extension_projection.ConfiguredBoardExtensionPin,
        Path,
    ],
) -> None:
    root, raw_paths = _seed_handoff(tmp_path)
    paths = tuple(sorted(raw_paths))
    pin = _pin(root)
    projection_pin, _projection_home = quack_projection
    admission = _admission(root, paths, projection_pin)
    native_launch = _native_launch()
    repository_root = Path(__file__).resolve().parents[2]
    template = scheduler.load_configured_board(
        repository_root / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json",
        repo_root=repository_root,
    )
    payload = dict(template.payload)
    payload["plan_revision"] = "TEST-PLAN-R2"
    payload["provider"] = {"provider_id": "auto"}
    program = replace(
        template.resolved_database_program(),
        store_id="data/control.duckdb",
        store_generation="2",
        endpoint_secret_handle="env://TEST_QUACK_TOKEN",
        event_store_path="state/events",
        runtime_registry_path="state/registry",
        worktree_root="state/worktrees",
    )
    board = replace(
        template,
        config_path=root / "config/scheduler.json",
        repo_root=root,
        payload=payload,
        configuration_root=admission.configuration_root,
        configuration_revision=cid_for_dag_json({"configuration": "revision"}),
        taskboard_path="docs/plan.md",
        objectives_path="docs/plan.md",
        plan_path="docs/plan.md",
        validator_path="scripts/validate.py",
        task_prefix="TEST-",
        board_namespace="test-board-v1",
        merge_target_branch="main",
        max_lanes=2,
        strict_task_sharding=True,
        protected_paths=paths,
        live_capsule_control_paths=paths,
        runtime_paths={
            "root": "state",
            "state": "state/runtime",
            "worktrees": "state/worktrees",
            "merge_queue": "state/merge-queue",
            "logs": "state/logs",
            "generated_runtime_artifacts_are_completion_authority": False,
        },
        database_program=program,
    )

    def accept_control_plane(
        _pin_value: AgentImplementationControlPlanePin,
        descriptor: int,
    ) -> str:
        return f"/proc/self/fd/{descriptor}"

    monkeypatch.setattr(
        capsule,
        "verify_agent_implementation_sealed_control_plane",
        accept_control_plane,
    )
    monkeypatch.setattr(
        scheduler,
        "verify_agent_implementation_sealed_control_plane",
        accept_control_plane,
    )
    monkeypatch.setattr(
        runner,
        "verify_agent_implementation_sealed_control_plane",
        accept_control_plane,
    )
    for module in (capsule, scheduler, runner, implementation):
        monkeypatch.setattr(
            module,
            "verify_agent_supervisor_native_dependency_sealed_fd",
            lambda launch: f"/proc/self/fd/{launch.descriptor.descriptor}",
        )
    monkeypatch.setattr(
        runner,
        "_validate_plan_bound_accepted_tree",
        lambda **_kwargs: None,
    )

    plan = scheduler.configured_board_launch_plan(
        board,
        implement=True,
        detach=False,
        duration_seconds=1,
        stamp="20260828T-live-capsule",
        accepted_control_plane_pin=pin,
        accepted_control_plane_descriptor=17,
        native_dependency_launch=native_launch,
        configured_board_live_admission=admission,
    )
    plan_argv = list(plan["argv"])
    assert "--require-configured-board-live-capsule" in plan_argv
    admission_index = plan_argv.index("--configured-board-live-admission-json")
    scheduled_admission_json = plan_argv[admission_index + 1]
    assert scheduled_admission_json == admission.to_json()
    native_index = plan_argv.index("--configured-board-live-native-launch-json")
    assert plan_argv[native_index + 1] == native_launch.to_json()
    assert plan_argv[plan_argv.index("--configured-board-live-native-fd") + 1] == "19"

    parsed_runner = runner.build_arg_parser().parse_args(plan_argv)
    tracks = runner.tracks_from_parsed_args(parsed_runner)
    common_args = runner.common_args_from_parsed_args(parsed_runner)
    inherited_admission = capsule.parse_configured_board_live_capsule_admission(
        parsed_runner.configured_board_live_admission_json
    )
    monkeypatch.setattr(
        runner,
        "verify_configured_board_live_capsule",
        lambda value, **_kwargs: (
            value
            if isinstance(value, capsule.ConfiguredBoardLiveCapsuleAdmission)
            else capsule.parse_configured_board_live_capsule_admission(value)
        ),
    )
    monkeypatch.setattr(
        runner,
        "parse_native_dependency_launch_json",
        lambda value: (
            native_launch
            if value == native_launch.to_json()
            else (_ for _ in ()).throw(ValueError("forged native launch"))
        ),
    )
    monkeypatch.setattr(
        runner,
        "_plan_bound_repository_identity",
        lambda _root: (pin.source_head, pin.source_tree),
    )
    runner_popen: dict[str, object] = {}

    def capture_runner_popen(command: list[str], **kwargs: object) -> object:
        runner_popen["command"] = list(command)
        runner_popen.update(kwargs)
        return SimpleNamespace(pid=999_991)

    monkeypatch.setattr(runner.subprocess, "Popen", capture_runner_popen)
    runner.start_track(
        tracks[0],
        repo_root=root,
        common_args=common_args,
        python_executable=sys.executable,
        accepted_control_plane_pin=pin,
        accepted_control_plane_descriptor=17,
        native_dependency_launch=native_launch,
        configured_board_live_admission=inherited_admission,
        output=lambda _message: None,
    )
    supervisor_command = list(runner_popen["command"])
    assert supervisor_command[1:5] == ["-I", "-S", "-B", "-c"]
    assert runner_popen["pass_fds"] == (17, 19)
    assert scheduled_admission_json in supervisor_command
    assert native_launch.to_json() in supervisor_command

    supervisor_argv = supervisor_command[13:]
    parsed_supervisor = implementation.parse_args(supervisor_argv)
    verified_live_admissions: list[object] = []

    def verify_live_admission(value: object, **_kwargs: object) -> object:
        verified_live_admissions.append(value)
        return inherited_admission

    monkeypatch.setattr(
        implementation,
        "parse_accepted_control_plane_pin",
        lambda _value: pin,
    )
    monkeypatch.setattr(
        implementation,
        "verify_agent_implementation_sealed_control_plane",
        accept_control_plane,
    )
    monkeypatch.setattr(
        implementation,
        "parse_native_dependency_launch_json",
        lambda value: (
            native_launch
            if value == native_launch.to_json()
            else (_ for _ in ()).throw(ValueError("forged native launch"))
        ),
    )
    monkeypatch.setattr(
        implementation,
        "verify_configured_board_live_capsule",
        verify_live_admission,
    )
    supervisor_config = implementation.supervisor_config_from_args(
        parsed_supervisor,
        repo_root=root,
    )
    assert supervisor_config.configured_board_live_admission == inherited_admission
    assert supervisor_config.native_dependency_launch is native_launch
    assert verified_live_admissions

    supervisor = object.__new__(implementation.PortalImplementationSupervisor)
    supervisor.config = supervisor_config
    daemon_command = supervisor._build_daemon_command()
    assert daemon_command[1:5] == ["-I", "-S", "-B", "-c"]
    assert scheduled_admission_json in daemon_command
    assert native_launch.to_json() in daemon_command
    daemon_popen: dict[str, object] = {}

    def capture_daemon_popen(command: list[str], **kwargs: object) -> object:
        daemon_popen["command"] = list(command)
        daemon_popen.update(kwargs)
        return SimpleNamespace(pid=os.getpid())

    supervisor.ensure_managed_daemon_pid_file = lambda: {}
    monkeypatch.setattr(
        implementation.subprocess,
        "Popen",
        capture_daemon_popen,
    )
    supervisor._start_daemon()
    assert daemon_popen["pass_fds"] == (17, 19)
    assert daemon_popen["command"] == daemon_command


def test_sealed_coordinator_environment_is_offline_and_secret_bounded(
    monkeypatch: pytest.MonkeyPatch,
    quack_projection: tuple[
        extension_projection.ConfiguredBoardExtensionPin,
        Path,
    ],
) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    board = scheduler.load_configured_board(
        repository_root / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json",
        repo_root=repository_root,
    )
    monkeypatch.setenv("SAWM_QUACK_TOKEN", "quack-secret")
    monkeypatch.setenv("UNRELATED_SECRET", "must-not-cross")
    _projection_pin, projection_home = quack_projection
    extension_directory = projection_home / ".duckdb/extensions"
    environment = scheduler._sealed_coordinator_environment(
        board,
        extension_directory=extension_directory,
    )

    assert environment["SAWM_QUACK_TOKEN"] == "quack-secret"
    assert "UNRELATED_SECRET" not in environment
    assert environment["IPFS_DATASETS_AUTO_INSTALL"] == "0"
    assert environment["IPFS_DATASETS_AUTO_INSTALL_TEST_DEPS"] == "0"
    assert environment["IPFS_KIT_AUTO_INSTALL_DEPS"] == "0"
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"
    assert environment["PATH"] == os.pathsep.join(
        (
            str(Path(board.payload["provider"]["primary_executable"]).parent),
            "/usr/local/bin",
            "/usr/bin",
            "/bin",
        )
    )
    assert environment[extension_projection.CONFIGURED_BOARD_EXTENSION_DIRECTORY_ENV] == str(
        extension_directory
    )


def test_operational_live_capsule_missing_or_forged_is_zero_popen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    quack_projection: tuple[
        extension_projection.ConfiguredBoardExtensionPin,
        Path,
    ],
) -> None:
    root, raw_paths = _seed_handoff(tmp_path)
    projection_pin, _projection_home = quack_projection
    admission = _admission(root, tuple(sorted(raw_paths)), projection_pin)
    pin = _pin(root)
    native_launch = _native_launch()
    track = runner.SupervisorTrack(
        name="test-board-v1",
        script_path=Path(runner.PLAN_BOUND_ACCEPTED_ENTRY_PATH),
        log_path=Path("state/runner.log"),
        supervisor_pid_path=Path("state/runner.pid"),
        daemon_pid_path=Path("state/daemon.pid"),
    )
    common = (
        "--state-schema-revision",
        runner.DATASETS_AUTHORITATIVE_OPERATIONAL_SCHEMA_REVISION,
    )
    popen_calls: list[object] = []
    monkeypatch.setattr(
        runner.subprocess,
        "Popen",
        lambda *args, **kwargs: popen_calls.append((args, kwargs)),
    )

    with pytest.raises(ValueError, match="native dependency launch"):
        runner.start_track(
            track,
            repo_root=root,
            common_args=common,
            accepted_control_plane_pin=pin,
            accepted_control_plane_descriptor=17,
            configured_board_live_admission=None,
            output=lambda _message: None,
        )

    for require in (False, True):
        with pytest.raises(ValueError, match="requires"):
            runner.run_supervisor_tracks(
                (track,),
                repo_root=root,
                common_args=common,
                duration_seconds=0,
                accepted_control_plane_pin=pin,
                accepted_control_plane_descriptor=17,
                require_configured_board_live_capsule=require,
                configured_board_live_admission=None,
                output=lambda _message: None,
            )

    forged = replace(admission, max_lanes=admission.max_lanes + 1)
    monkeypatch.setattr(
        capsule,
        "verify_agent_implementation_sealed_control_plane",
        lambda _pin_value, descriptor: f"/proc/self/fd/{descriptor}",
    )
    monkeypatch.setattr(
        runner,
        "verify_agent_supervisor_native_dependency_sealed_fd",
        lambda launch: f"/proc/self/fd/{launch.descriptor.descriptor}",
    )
    monkeypatch.setattr(
        capsule,
        "verify_agent_supervisor_native_dependency_sealed_fd",
        lambda launch: f"/proc/self/fd/{launch.descriptor.descriptor}",
    )
    with pytest.raises(ValueError):
        runner.run_supervisor_tracks(
            (track,),
            repo_root=root,
            common_args=common,
            duration_seconds=0,
            accepted_control_plane_pin=pin,
            accepted_control_plane_descriptor=17,
            native_dependency_launch=native_launch,
            require_configured_board_live_capsule=True,
            configured_board_live_admission=forged,
            output=lambda _message: None,
        )
    forged_native = SimpleNamespace(
        **{
            **vars(native_launch),
            "accepted_authorization_id": "sha256:" + "9" * 64,
        }
    )
    with pytest.raises(ValueError):
        runner.start_track(
            track,
            repo_root=root,
            common_args=common,
            accepted_control_plane_pin=pin,
            accepted_control_plane_descriptor=17,
            native_dependency_launch=forged_native,
            configured_board_live_admission=admission,
            output=lambda _message: None,
        )
    assert popen_calls == []
