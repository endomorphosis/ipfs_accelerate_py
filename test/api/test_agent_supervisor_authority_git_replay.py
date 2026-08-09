"""Fail-closed Git metadata replay for sealed authority validation."""

from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.deterministic_repair_forest import (
    DCR_ARTIFACT_PATH,
    write_repair_forest,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_daemon as daemon_module,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    _AUTHORITY_GIT_MOUNT_BOOTSTRAP,
    AuthorityGitReplayError,
    PortalTask,
    PortalTaskState,
    TodoImplementationDaemon,
    _authority_git_replay_plan,
    _authority_git_seal_mounts,
)
from test.api.test_agent_supervisor_dcr_forest import (
    LifecycleFixture as _LifecycleFixture,
)
from test.api.test_agent_supervisor_dcr_forest import (
    _git as _forest_git,
)
from test.api.test_agent_supervisor_dcr_forest import (
    _make_workspace as _make_forest_workspace,
)


def _git(root: Path, *arguments: str) -> str:
    environment = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
    }
    completed = subprocess.run(
        ("git", "-c", "protocol.file.allow=always", *arguments),
        cwd=root,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.strip()


def _init_repository(root: Path, *, label: str) -> None:
    root.mkdir(parents=True)
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.name", "Authority Git Replay")
    _git(root, "config", "user.email", "authority-git@example.invalid")
    (root / "tracked.txt").write_text(label + "\n", encoding="utf-8")
    _git(root, "add", "tracked.txt")
    _git(root, "commit", "-qm", f"seed {label}")


def _linked_workspace(tmp_path: Path) -> tuple[Path, Path]:
    nested_source = tmp_path / "nested-source"
    _init_repository(nested_source, label="nested")
    source = tmp_path / "source"
    _init_repository(source, label="root")
    _git(source, "submodule", "add", "-q", str(nested_source), "nested")
    _git(source, "commit", "-qam", "add configured nested root")
    linked = tmp_path / "linked"
    _git(source, "worktree", "add", "-q", "-b", "validation", str(linked))
    _git(linked, "submodule", "update", "--init", "--recursive")
    return source, linked


def _plan(workspace: Path):
    return _authority_git_replay_plan(
        workspace,
        lifecycle_subject=_git(workspace, "rev-parse", "HEAD"),
    )


def _real_forest_cli_fixture(
    tmp_path: Path, *, linked_outer: bool = False
) -> _LifecycleFixture:
    workspace = _make_forest_workspace(tmp_path)
    accelerator = workspace / "external/ipfs_accelerate"
    repository_root = Path(__file__).resolve().parents[2]
    module_files = (
        "ipfs_accelerate_py/agent_supervisor/analysis/"
        "deterministic_repair_forest.py",
        "ipfs_accelerate_py/agent_supervisor/autonomous_repair/no_llm_policy.py",
        "ipfs_accelerate_py/agent_supervisor/autonomous_repair/root_ownership.py",
    )
    for relative in module_files:
        for destination_root in (accelerator, workspace):
            destination = destination_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(repository_root / relative, destination)
    forest_test = accelerator / ("test/api/test_agent_supervisor_dcr_forest.py")
    forest_test.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        repository_root / "test/api/test_agent_supervisor_dcr_forest.py",
        forest_test,
    )
    for relative in (
        "ipfs_accelerate_py/__init__.py",
        "ipfs_accelerate_py/agent_supervisor/__init__.py",
        "ipfs_accelerate_py/agent_supervisor/analysis/__init__.py",
        "ipfs_accelerate_py/agent_supervisor/autonomous_repair/__init__.py",
    ):
        for destination_root in (accelerator, workspace):
            path = destination_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")
    todo = workspace.joinpath(
        *Path(daemon_module.AUTHORITY_VALIDATION_DCR_TODO_PATH).parts
    )
    todo_text = todo.read_text(encoding="utf-8")
    todo.write_text(
        todo_text.replace(
            "- Acceptance: bind the real repository forest.\n",
            "- Acceptance: bind the real repository forest.\n"
            "- Validation: "
            + "; ".join(daemon_module.AUTHORITY_VALIDATION_DCR_RAW_COMMANDS)
            + "\n",
            1,
        ),
        encoding="utf-8",
    )
    _forest_git(accelerator, "add", "ipfs_accelerate_py", "test/api")
    _forest_git(accelerator, "commit", "-m", "install sealed forest CLI")
    if linked_outer:
        accelerator_origin = Path(
            _forest_git(accelerator, "remote", "get-url", "origin")
        )
        _forest_git(
            accelerator_origin,
            "config",
            "receive.denyCurrentBranch",
            "updateInstead",
        )
        _forest_git(accelerator, "push", "origin", "HEAD:main")
    _forest_git(
        workspace,
        "add",
        "external/ipfs_accelerate",
        "ipfs_accelerate_py",
        "implementation_plan",
    )
    _forest_git(
        workspace,
        "commit",
        "-m",
        "DCR-011: pin landed provider implementation",
    )
    if linked_outer:
        linked_workspace = tmp_path / "LinkedAuthorityWorkspace"
        _forest_git(
            workspace,
            "worktree",
            "add",
            "-b",
            "authority-linked-base",
            str(linked_workspace),
            "HEAD",
        )
        _forest_git(
            linked_workspace,
            "submodule",
            "update",
            "--init",
            "--recursive",
        )
        _forest_git(workspace, "switch", "--detach")
        workspace = linked_workspace
    subject = _forest_git(workspace, "rev-parse", "HEAD")
    branch = "implementation/dcr-011-provider"
    _forest_git(workspace, "switch", "-c", branch)
    artifact = workspace.joinpath(*Path(DCR_ARTIFACT_PATH).parts)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    manifest = write_repair_forest(artifact, workspace)
    return _LifecycleFixture(
        workspace=workspace,
        manifest=manifest,
        subject=subject,
        branch=branch,
    )


def _exact_context(
    workspace: Path, *, scope: str = "post_merge"
) -> tuple[str, dict[str, str], dict[str, str]]:
    command_set = (
        daemon_module.AUTHORITY_VALIDATION_DCR_RAW_COMMANDS
        if scope == "post_merge"
        else daemon_module.AUTHORITY_VALIDATION_DCR_COMMANDS
    )
    values = _git(
        workspace,
        "rev-parse",
        "HEAD",
        "HEAD^{tree}",
        "--path-format=absolute",
        "--git-dir",
        "--git-common-dir",
    ).splitlines()
    head, tree, git_dir, common_dir = values
    plan_body = {
        "schema": (
            "ipfs_accelerate_py.agent_supervisor." "authority-validation-command-plan@2"
        ),
        "authority_profile": "dcr011_forest@1",
        "task_id": "DCR-011",
        "canonical_task_cid": daemon_module.AUTHORITY_VALIDATION_DCR_TASK_CID,
        "scope": scope,
        "commands": list(command_set),
        "target_commit": head,
        "target_tree": tree,
        "git_common_anchor": common_dir,
        "git_dir": git_dir,
    }
    environment = {
        daemon_module._AUTHORITY_VALIDATION_SCOPE_ENV: scope,
        daemon_module._AUTHORITY_VALIDATION_PROFILE_ENV: ("dcr011_forest@1"),
        daemon_module._AUTHORITY_VALIDATION_COMMANDS_ENV: json.dumps(
            list(command_set),
            ensure_ascii=True,
            separators=(",", ":"),
        ),
        daemon_module._AUTHORITY_VALIDATION_TASK_ENV: "DCR-011",
        daemon_module._AUTHORITY_VALIDATION_TASK_CID_ENV: (
            daemon_module.AUTHORITY_VALIDATION_DCR_TASK_CID
        ),
        daemon_module._AUTHORITY_VALIDATION_PLAN_ENV: (
            daemon_module.content_identity(plan_body)
        ),
        daemon_module._AUTHORITY_VALIDATION_TARGET_COMMIT_ENV: head,
        daemon_module._AUTHORITY_VALIDATION_TARGET_TREE_ENV: tree,
        daemon_module._AUTHORITY_VALIDATION_GIT_COMMON_ANCHOR_ENV: common_dir,
        daemon_module._AUTHORITY_VALIDATION_GIT_DIR_ENV: git_dir,
        daemon_module._AUTHORITY_VALIDATION_PRODUCER_TRUST_ENV: "1",
    }
    return (
        command_set[1],
        environment,
        {
            "target_commit": head,
            "repository_tree_id": f"git-tree:{tree}",
        },
    )


def _not_requested_receipt_fixture(tmp_path: Path):
    workspace = str((tmp_path / "workspace").resolve())
    common = str((tmp_path / "common.git").resolve())
    git_dir = str((Path(common) / "worktrees/candidate").resolve())
    target = "1" * 40
    tree = "2" * 40
    command = daemon_module.AUTHORITY_VALIDATION_DCR_RAW_COMMANDS[0]
    plan_body = {
        "schema": (
            "ipfs_accelerate_py.agent_supervisor." "authority-validation-command-plan@2"
        ),
        "authority_profile": "dcr011_forest@1",
        "task_id": "DCR-011",
        "canonical_task_cid": daemon_module.AUTHORITY_VALIDATION_DCR_TASK_CID,
        "scope": "post_merge",
        "commands": list(daemon_module.AUTHORITY_VALIDATION_DCR_RAW_COMMANDS),
        "target_commit": target,
        "target_tree": tree,
        "git_common_anchor": common,
        "git_dir": git_dir,
    }
    plan_id = daemon_module.content_identity(plan_body)
    binding_body = {
        "schema": (
            "ipfs_accelerate_py.agent_supervisor."
            "authority-validation-command-binding@2"
        ),
        "authority_profile": "dcr011_forest@1",
        "task_id": "DCR-011",
        "canonical_task_cid": daemon_module.AUTHORITY_VALIDATION_DCR_TASK_CID,
        "scope": "post_merge",
        "plan_id": plan_id,
        "expected_target_commit": target,
        "expected_target_tree": tree,
        "expected_git_common_anchor": common,
        "expected_git_dir": git_dir,
        "ordinal": 0,
        "validation_id": "",
        "command": command,
        "raw_command": command,
        "guarded_argv": daemon_module.validation_shell_command(command),
    }
    binding = {
        **binding_body,
        "command_binding_id": daemon_module.content_identity(binding_body),
    }
    binding["command_binding_cache_id"] = daemon_module.content_identity(
        {
            "schema": (
                "ipfs_accelerate_py.agent_supervisor."
                "authority-validation-command-cache-key@1"
            ),
            "workspace_path": workspace,
            "plan_id": plan_id,
            "command_binding_id": binding["command_binding_id"],
        }
    )
    replay_id = daemon_module.content_identity(
        {
            "schema": (
                "ipfs_accelerate_py.agent_supervisor."
                "authority-git-metadata-not-requested@1"
            ),
            "workspace": workspace,
            "mode": "not_requested",
        }
    )
    contract = {
        "available": True,
        "contract_id": "test-contract",
        "docker_endpoint": "unix:///run/docker.sock",
        "image_id": "sha256:" + "a" * 64,
        "gpu_uuid": "GPU-test",
        "typescript_validation_toolchain": {},
    }
    body = {
        "schema": (
            "ipfs_accelerate_py.agent_supervisor."
            "authority-validation-isolation-receipt@3"
        ),
        "contract_id": contract["contract_id"],
        "backend": "docker-local-cuda",
        "docker_endpoint": contract["docker_endpoint"],
        "base_image_id": contract["image_id"],
        "image_id": contract["image_id"],
        "gpu_uuid": contract["gpu_uuid"],
        "gpu_requested": True,
        "network_mode": "none",
        "host_filesystem": "workspace_only_read_only",
        "workspace_path": workspace,
        "workspace_read_only": True,
        "git_metadata_read_only": False,
        "git_metadata_drift_detected": False,
        "git_metadata_replay": {
            "schema": (
                "ipfs_accelerate_py.agent_supervisor." "authority-git-metadata-replay@2"
            ),
            "mode": "not_requested",
            "workspace_path": workspace,
            "configured_roots": [],
            "root_identities": [],
            "external_mounts": [],
            "external_mount_count": 0,
            "observation_count": 0,
            "observation_identities": [],
            "preflight_budget": {
                "schema": (
                    "ipfs_accelerate_py.agent_supervisor."
                    "authority-git-replay-budget@1"
                ),
                "source_bytes": 0,
                "metadata_entries": 0,
            },
            "preflight_id": replay_id,
            "postflight_id": replay_id,
            "drift_detected": False,
            "read_only": True,
            "symlinks_allowed": False,
            "projection_scope": "none",
            "raw_common_config_exposed": False,
            "raw_refs_exposed": False,
            "unrelated_objects_exposed": False,
        },
        "git_metadata_sealed_mounts": [],
        "git_sealed_projection": {
            "schema": (
                "ipfs_accelerate_py.agent_supervisor."
                "authority-git-sealed-projection@2"
            ),
            "source_bytes": 0,
            "sealed_bytes": 0,
            "aggregate_bytes": 0,
            "byte_limit": daemon_module.AUTHORITY_VALIDATION_GIT_SNAPSHOT_MAX_BYTES,
            "setup_elapsed_milliseconds": 0,
            "setup_time_limit_seconds": (
                daemon_module.AUTHORITY_VALIDATION_GIT_SNAPSHOT_MAX_SECONDS
            ),
            "metadata_entries": 0,
            "metadata_entry_limit": (
                daemon_module.AUTHORITY_VALIDATION_GIT_MAX_METADATA_ENTRIES
            ),
            "mount_count": 0,
            "completed": True,
            "copy_mode": "not_requested",
            "hardlinks_used": False,
            "object_scope": "none",
            "raw_common_config_exposed": False,
            "raw_refs_exposed": False,
            "unrelated_objects_exposed": False,
            "object_packs": [],
            "object_pack_count": 0,
            "git_version": "not_requested",
            "transport": {
                "schema": (
                    "ipfs_accelerate_py.agent_supervisor."
                    "authority-git-derived-image@1"
                ),
                "mode": "not_requested",
            },
        },
        "git_setup_reclaimer": {
            "schema": (
                "ipfs_accelerate_py.agent_supervisor."
                "authority-git-projection-reclaimer@2"
            ),
            "mode": "not_requested",
        },
        "git_postflight": {
            "schema": (
                "ipfs_accelerate_py.agent_supervisor." "authority-git-postflight@1"
            ),
            "mode": "not_requested",
            "elapsed_milliseconds": 0,
            "time_limit_seconds": (
                daemon_module.AUTHORITY_VALIDATION_GIT_SNAPSHOT_MAX_SECONDS
            ),
            "completed": True,
        },
        "git_projection_cleanup": {
            "schema": (
                "ipfs_accelerate_py.agent_supervisor."
                "authority-validation-runtime-cleanup@1"
            ),
            "container_attempted": True,
            "container_succeeded": True,
            "derived_image_attempted": False,
            "derived_image_succeeded": True,
            "derived_image_id": "",
            "derived_cleanup_lease_id": "",
            "derived_cleanup_journal_path": "",
            "derived_cleanup_quiet_milliseconds": 0,
            "derived_cleanup_elapsed_milliseconds": 0,
            "derived_cleanup_time_limit_milliseconds": 0,
            "derived_cleanup_journal_released": True,
            "projection_attempted": False,
            "projection_succeeded": True,
            "succeeded": True,
        },
        "authority_command_binding": binding,
        "git_mount_bootstrap": {
            "schema": (
                "ipfs_accelerate_py.agent_supervisor." "authority-git-mount-bootstrap@1"
            ),
            "record_count": 0,
            "manifest_id": daemon_module.content_identity([]),
            "records": [],
            "runs_before_validation_command": True,
            "source_swap_fails_closed": True,
            "derived_layer_manifest_id": "",
            "derived_layer_record_count": 0,
            "exact_tree_verified": False,
        },
        "git_safe_directory_config": {
            "mounted": False,
            "path": "",
            "sha256": "",
            "configured_roots": [],
            "read_only": True,
            "required_for_git_discovery": False,
            "synthetic_metadata_owned_by_container_uid": False,
        },
        "private_pid_namespace": True,
        "cgroup_process_limit": daemon_module.AUTHORITY_VALIDATION_PIDS_LIMIT,
        "memory_limit_bytes": daemon_module.AUTHORITY_VALIDATION_MEMORY_LIMIT_BYTES,
        "tmpfs_limit_bytes": daemon_module.AUTHORITY_VALIDATION_TMPFS_LIMIT_BYTES,
        "cpu_limit": daemon_module.AUTHORITY_VALIDATION_CPU_LIMIT,
        "timeout_limit_seconds": daemon_module.AUTHORITY_VALIDATION_TIMEOUT_LIMIT_SECONDS,
        "capabilities_dropped": "all",
        "no_new_privileges": True,
        "container_root_read_only": True,
        "container_log_driver": "none",
        "typescript_validation_toolchain": {},
        "output_limit_bytes": daemon_module.AUTHORITY_VALIDATION_OUTPUT_LIMIT_BYTES,
        "output_limit_exceeded": False,
        "output_bounded": True,
        "storage_bounded": True,
        "cpu_bounded": True,
        "container_removed": True,
        "process_tree_quiesced": True,
    }
    receipt = daemon_module._authority_validation_build_isolation_receipt(body)
    result = {
        "command": command,
        "raw_command": command,
        "ordinal": 0,
        "validation_id": "",
        "cache_key": "b" * 64,
        "authority_validation_command_binding": binding,
    }
    validated = {
        "target_commit": target,
        "repository_tree_id": f"git-tree:{tree}",
    }
    return receipt, contract, result, validated


def test_replay_plan_projects_only_linked_and_recursive_configured_git_metadata(
    tmp_path: Path,
) -> None:
    source, linked = _linked_workspace(tmp_path)

    plan = _plan(linked)
    replay = plan.receipt()

    assert replay["configured_roots"] == [".", "nested"]
    assert replay["preflight_id"] == replay["postflight_id"]
    assert replay["drift_detected"] is False
    assert plan.plan_id == _plan(linked).plan_id
    assert plan.mounts
    common_dir = Path(
        _git(linked, "rev-parse", "--path-format=absolute", "--git-common-dir")
    )
    git_dir = Path(_git(linked, "rev-parse", "--path-format=absolute", "--git-dir"))
    sources = {mount.source for mount in plan.mounts}
    assert sources == {common_dir}
    assert git_dir != common_dir
    assert source / ".git" == common_dir
    assert all(mount.source == mount.destination for mount in plan.mounts)
    assert all(mount.source.is_absolute() for mount in plan.mounts)
    assert all(str(mount.source).startswith(str(tmp_path)) for mount in plan.mounts)
    assert {mount.purpose for mount in plan.mounts} == {
        "synthetic-root-common-metadata"
    }
    assert replay["raw_common_config_exposed"] is False
    assert replay["raw_refs_exposed"] is False
    assert replay["unrelated_objects_exposed"] is False
    assert not any("unreviewed" in str(mount.source) for mount in plan.mounts)


def test_tree_closure_fails_closed_on_deep_chain_without_recursion() -> None:
    objects: dict[str, SimpleNamespace] = {}
    child = f"{1000:040x}"
    objects[child] = SimpleNamespace(object_type="tree", raw=b"")
    for ordinal in reversed(range(400)):
        oid = f"{ordinal + 1:040x}"
        objects[oid] = SimpleNamespace(
            object_type="tree",
            raw=b"40000 d\0" + bytes.fromhex(child),
        )
        child = oid
    assert (
        daemon_module._authority_git_tree_inventory(objects, child, oid_length=40)
        is None
    )


def test_tree_closure_fails_closed_on_shared_dag_expansion() -> None:
    objects: dict[str, SimpleNamespace] = {}
    blob_oid = "f" * 40
    objects[blob_oid] = SimpleNamespace(object_type="blob", raw=b"value")
    child = f"{900:040x}"
    objects[child] = SimpleNamespace(
        object_type="tree",
        raw=b"100644 f\0" + bytes.fromhex(blob_oid),
    )
    for ordinal in reversed(range(18)):
        oid = f"{ordinal + 500:040x}"
        child_bytes = bytes.fromhex(child)
        objects[oid] = SimpleNamespace(
            object_type="tree",
            raw=(b"40000 a\0" + child_bytes + b"40000 b\0" + child_bytes),
        )
        child = oid
    assert (
        daemon_module._authority_git_tree_inventory(objects, child, oid_length=40)
        is None
    )


@pytest.mark.parametrize("daemon_available", (True, False))
def test_docker_cleanup_lease_is_durable_and_quiet_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    daemon_available: bool,
) -> None:
    cleanup_root = tmp_path / "docker-cleanup"
    cleanup_root.mkdir(mode=0o700)
    monkeypatch.setattr(
        daemon_module,
        "_authority_git_docker_cleanup_parent",
        lambda: cleanup_root,
    )
    body = daemon_module._authority_git_docker_cleanup_lease_body(
        base_image_id="sha256:" + "a" * 64,
        authority_plan_id="plan",
        command_binding_id="binding",
        projection_manifest_id="projection",
    )
    lease, journal = daemon_module._authority_git_publish_docker_cleanup_lease(body)
    assert journal.exists()
    assert daemon_module._authority_git_read_docker_cleanup_lease(journal) == lease

    clock = [100.0]
    deadlines: list[float] = []

    def fake_bounded(command, **kwargs):
        deadlines.append(kwargs["deadline"])
        clock[0] += 0.05
        return daemon_module._BoundedSubprocessResult(
            args=tuple(command),
            returncode=0 if daemon_available else 1,
            stdout=b"",
            stderr=b"" if daemon_available else b"daemon unavailable",
            timed_out=False,
            output_overflow=False,
            reaped=True,
        )

    monkeypatch.setattr(daemon_module, "_run_bounded_subprocess", fake_bounded)
    monkeypatch.setattr(
        daemon_module,
        "time",
        SimpleNamespace(
            monotonic=lambda: clock[0],
            sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
        ),
    )
    cleaned = daemon_module._authority_git_cleanup_docker_lease(
        lease,
        journal,
        docker_prefix=("docker",),
        docker_environment={},
        deadline=130.0,
    )
    assert cleaned is daemon_available
    assert bool(journal.exists()) is (not daemon_available)
    assert deadlines and set(deadlines) == {130.0}


def test_docker_cleanup_quiet_window_catches_late_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cleanup_root = tmp_path / "docker-cleanup"
    cleanup_root.mkdir(mode=0o700)
    monkeypatch.setattr(
        daemon_module,
        "_authority_git_docker_cleanup_parent",
        lambda: cleanup_root,
    )
    body = daemon_module._authority_git_docker_cleanup_lease_body(
        base_image_id="sha256:" + "b" * 64,
        authority_plan_id="plan",
        command_binding_id="binding",
        projection_manifest_id="projection",
    )
    lease, journal = daemon_module._authority_git_publish_docker_cleanup_lease(body)
    clock = [200.0]
    image_queries = [0]
    removed: list[tuple[str, ...]] = []

    def fake_bounded(command, **kwargs):
        argv = tuple(command)
        clock[0] += 0.2
        stdout = b""
        if (
            "image" in argv
            and "ls" in argv
            and any(str(item).startswith("label=") for item in argv)
        ):
            image_queries[0] += 1
            if image_queries[0] == 2:
                stdout = ("sha256:" + "c" * 64 + "\n").encode("ascii")
        if "rm" in argv:
            removed.append(argv)
        return daemon_module._BoundedSubprocessResult(
            args=argv,
            returncode=0,
            stdout=stdout,
            stderr=b"",
            timed_out=False,
            output_overflow=False,
            reaped=True,
        )

    monkeypatch.setattr(daemon_module, "_run_bounded_subprocess", fake_bounded)
    monkeypatch.setattr(
        daemon_module,
        "time",
        SimpleNamespace(
            monotonic=lambda: clock[0],
            sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
        ),
    )
    assert daemon_module._authority_git_cleanup_docker_lease(
        lease,
        journal,
        docker_prefix=("docker",),
        docker_environment={},
        deadline=230.0,
    )
    assert removed
    assert image_queries[0] > 2
    assert not journal.exists()


def test_docker_cleanup_pending_live_owner_is_reclaimed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cleanup_root = tmp_path / "docker-cleanup"
    cleanup_root.mkdir(mode=0o700)
    monkeypatch.setattr(
        daemon_module,
        "_authority_git_docker_cleanup_parent",
        lambda: cleanup_root,
    )
    body = daemon_module._authority_git_docker_cleanup_lease_body(
        base_image_id="sha256:" + "d" * 64,
        authority_plan_id="plan",
        command_binding_id="binding",
        projection_manifest_id="projection",
    )
    lease, journal = daemon_module._authority_git_publish_docker_cleanup_lease(body)
    clock = [300.0]
    daemon_available = [False]

    def fake_bounded(command, **kwargs):
        clock[0] += 0.05
        return daemon_module._BoundedSubprocessResult(
            args=tuple(command),
            returncode=0 if daemon_available[0] else 1,
            stdout=b"",
            stderr=b"" if daemon_available[0] else b"daemon unavailable",
            timed_out=False,
            output_overflow=False,
            reaped=True,
        )

    monkeypatch.setattr(daemon_module, "_run_bounded_subprocess", fake_bounded)
    monkeypatch.setattr(
        daemon_module,
        "time",
        SimpleNamespace(
            monotonic=lambda: clock[0],
            sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
        ),
    )
    assert not daemon_module._authority_git_cleanup_docker_lease(
        lease,
        journal,
        docker_prefix=("docker",),
        docker_environment={},
        deadline=330.0,
    )
    pending = daemon_module._authority_git_read_docker_cleanup_lease(journal)
    assert pending is not None
    assert pending["phase"] == "cleanup_pending"

    daemon_available[0] = True
    receipt = daemon_module._authority_git_reclaim_docker_cleanup_leases(
        docker_prefix=("docker",),
        docker_environment={},
        deadline=330.0,
    )
    assert receipt["active"] == 0
    assert receipt["reclaimed"] == 1
    assert not journal.exists()


def test_docker_cleanup_reclaimer_is_serialized_by_root_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cleanup_root = tmp_path / "docker-cleanup"
    cleanup_root.mkdir(mode=0o700)
    monkeypatch.setattr(
        daemon_module,
        "_authority_git_docker_cleanup_parent",
        lambda: cleanup_root,
    )
    docker_calls: list[tuple[str, ...]] = []

    def unexpected_docker(command, **_kwargs):
        docker_calls.append(tuple(command))
        raise AssertionError("cleanup must not cross a held serialization fence")

    monkeypatch.setattr(daemon_module, "_run_bounded_subprocess", unexpected_docker)
    root_descriptor, lock_descriptor = daemon_module._authority_git_docker_cleanup_lock(
        deadline=time.monotonic() + 1.0
    )
    try:
        with pytest.raises(daemon_module.AuthorityGitReplayError):
            daemon_module._authority_git_reclaim_docker_cleanup_leases(
                docker_prefix=("docker",),
                docker_environment={},
                deadline=time.monotonic() + 0.05,
            )
    finally:
        daemon_module._authority_git_docker_cleanup_unlock(
            root_descriptor, lock_descriptor
        )
    assert not docker_calls


def test_docker_cleanup_reclaimer_batches_indeterminate_tombstones_under_one_quiet_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cleanup_root = tmp_path / "docker-cleanup"
    cleanup_root.mkdir(mode=0o700)
    monkeypatch.setattr(
        daemon_module,
        "_authority_git_docker_cleanup_parent",
        lambda: cleanup_root,
    )
    journals: list[Path] = []
    for ordinal in range(3):
        body = daemon_module._authority_git_docker_cleanup_lease_body(
            base_image_id="sha256:" + f"{ordinal + 1:x}" * 64,
            authority_plan_id=f"plan-{ordinal}",
            command_binding_id=f"binding-{ordinal}",
            projection_manifest_id=f"projection-{ordinal}",
        )
        lease, journal = daemon_module._authority_git_publish_docker_cleanup_lease(body)
        in_flight = daemon_module._authority_git_set_docker_cleanup_phase(
            lease,
            journal,
            expected_phases=frozenset({"building"}),
            next_phase="commit_in_flight",
        )
        assert in_flight is not None
        tombstone = daemon_module._authority_git_mark_docker_cleanup_pending(
            in_flight, journal
        )
        assert tombstone is not None
        assert tombstone["phase"] == "indeterminate_commit"
        journals.append(journal)

    clock = [400.0]
    deadlines: list[float] = []
    commands: list[tuple[str, ...]] = []

    def fake_bounded(command, **kwargs):
        argv = tuple(command)
        commands.append(argv)
        deadlines.append(kwargs["deadline"])
        clock[0] += 0.05
        return daemon_module._BoundedSubprocessResult(
            args=argv,
            returncode=0,
            stdout=b"",
            stderr=b"",
            timed_out=False,
            output_overflow=False,
            reaped=True,
        )

    monkeypatch.setattr(daemon_module, "_run_bounded_subprocess", fake_bounded)
    monkeypatch.setattr(
        daemon_module,
        "time",
        SimpleNamespace(
            monotonic=lambda: clock[0],
            sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
        ),
    )
    receipt = daemon_module._authority_git_reclaim_docker_cleanup_leases(
        docker_prefix=("docker",),
        docker_environment={},
        deadline=430.0,
    )
    assert receipt["schema"].endswith("docker-cleanup-reclaimer@2")
    assert receipt["active"] == 3
    assert receipt["reclaimed"] == 0
    assert receipt["retained_indeterminate"] == 3
    assert type(receipt["elapsed_milliseconds"]) is int
    assert 2000 <= receipt["elapsed_milliseconds"] < 3000
    assert receipt["time_limit_milliseconds"] == 30000
    assert receipt["quiet_milliseconds"] == 2000
    assert clock[0] - 400.0 < 3.0
    assert deadlines and set(deadlines) == {430.0}
    assert any(
        f"label={daemon_module._AUTHORITY_GIT_DOCKER_CLEANUP_LABEL_KEY}" in argv
        for argv in commands
    )
    assert any("reference=ipfs-accelerate-authority-git:*" in argv for argv in commands)
    assert all(journal.exists() for journal in journals)


def test_docker_cleanup_reclaimer_preserves_unknown_labeled_resource(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cleanup_root = tmp_path / "docker-cleanup"
    cleanup_root.mkdir(mode=0o700)
    monkeypatch.setattr(
        daemon_module,
        "_authority_git_docker_cleanup_parent",
        lambda: cleanup_root,
    )
    body = daemon_module._authority_git_docker_cleanup_lease_body(
        base_image_id="sha256:" + "e" * 64,
        authority_plan_id="plan",
        command_binding_id="binding",
        projection_manifest_id="projection",
    )
    lease, journal = daemon_module._authority_git_publish_docker_cleanup_lease(body)
    pending = daemon_module._authority_git_mark_docker_cleanup_pending(lease, journal)
    assert pending is not None
    unknown_lease_id = "f" * 64
    container_id = "a" * 12
    removals: list[tuple[str, ...]] = []

    def fake_bounded(command, **_kwargs):
        argv = tuple(command)
        stdout = b""
        if argv[1:3] == ("container", "ls"):
            stdout = (container_id + "\n").encode("ascii")
        elif argv[1:3] == ("container", "inspect"):
            stdout = (
                json.dumps(
                    {
                        daemon_module._AUTHORITY_GIT_DOCKER_CLEANUP_LABEL_KEY: (
                            unknown_lease_id
                        )
                    },
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8")
        if "rm" in argv:
            removals.append(argv)
        return daemon_module._BoundedSubprocessResult(
            args=argv,
            returncode=0,
            stdout=stdout,
            stderr=b"",
            timed_out=False,
            output_overflow=False,
            reaped=True,
        )

    monkeypatch.setattr(daemon_module, "_run_bounded_subprocess", fake_bounded)
    with pytest.raises(daemon_module.AuthorityGitReplayError):
        daemon_module._authority_git_reclaim_docker_cleanup_leases(
            docker_prefix=("docker",),
            docker_environment={},
            deadline=time.monotonic() + 2.0,
        )
    assert not removals
    assert journal.exists()


def test_docker_cleanup_reclaimer_leaves_all_live_active_phases_untouched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cleanup_root = tmp_path / "docker-cleanup"
    cleanup_root.mkdir(mode=0o700)
    monkeypatch.setattr(
        daemon_module,
        "_authority_git_docker_cleanup_parent",
        lambda: cleanup_root,
    )
    phases: list[tuple[Path, str]] = []
    for ordinal, target_phase in enumerate(("building", "commit_in_flight", "runtime")):
        body = daemon_module._authority_git_docker_cleanup_lease_body(
            base_image_id="sha256:" + f"{ordinal + 4:x}" * 64,
            authority_plan_id=f"plan-{ordinal}",
            command_binding_id=f"binding-{ordinal}",
            projection_manifest_id=f"projection-{ordinal}",
        )
        lease, journal = daemon_module._authority_git_publish_docker_cleanup_lease(body)
        current = lease
        if target_phase in {"commit_in_flight", "runtime"}:
            transitioned = daemon_module._authority_git_transition_docker_cleanup_phase(
                current,
                journal,
                expected_phase="building",
                next_phase="commit_in_flight",
                deadline=time.monotonic() + 2.0,
            )
            assert transitioned is not None
            current = transitioned
        if target_phase == "runtime":
            transitioned = daemon_module._authority_git_transition_docker_cleanup_phase(
                current,
                journal,
                expected_phase="commit_in_flight",
                next_phase="runtime",
                deadline=time.monotonic() + 2.0,
            )
            assert transitioned is not None
        phases.append((journal, target_phase))

    def unexpected_docker(command, **_kwargs):
        raise AssertionError(f"live active resource was queried: {command}")

    monkeypatch.setattr(daemon_module, "_run_bounded_subprocess", unexpected_docker)
    receipt = daemon_module._authority_git_reclaim_docker_cleanup_leases(
        docker_prefix=("docker",),
        docker_environment={},
        deadline=time.monotonic() + 2.0,
    )
    assert receipt["active"] == 3
    assert receipt["reclaimed"] == 0
    assert receipt["retained_indeterminate"] == 0
    for journal, phase in phases:
        current = daemon_module._authority_git_read_docker_cleanup_lease(journal)
        assert current is not None
        assert current["phase"] == phase


def test_replay_plan_ignores_unreviewed_nonsemantic_common_config(
    tmp_path: Path,
) -> None:
    _source, linked = _linked_workspace(tmp_path)
    before = _plan(linked)

    _git(linked, "config", "authority-test.drift", "true")
    after = _plan(linked)

    assert before.plan_id == after.plan_id


def test_replay_plan_ignores_unrelated_additive_git_object(
    tmp_path: Path,
) -> None:
    _source, linked = _linked_workspace(tmp_path)
    before = _plan(linked)
    unrelated = tmp_path / "unrelated-object"
    unrelated.write_text("not reachable from the reviewed heads\n", encoding="utf-8")

    _git(linked, "hash-object", "-w", str(unrelated))
    after = _plan(linked)

    assert before.plan_id == after.plan_id


def test_sealed_projection_does_not_mutate_reviewed_source_metadata(
    tmp_path: Path,
) -> None:
    _source, linked = _linked_workspace(tmp_path)
    _git(linked, "gc", "--prune=now")
    loose_payload = tmp_path / "source-loose-object"
    loose_payload.write_text("source identity sentinel\n", encoding="utf-8")
    loose_oid = _git(linked, "hash-object", "-w", str(loose_payload))
    plan = _plan(linked)
    common_dir = plan.mounts[0].source

    def source_regular_identities() -> dict[Path, tuple[int, ...]]:
        identities: dict[Path, tuple[int, ...]] = {}
        for directory, directory_names, filenames in os.walk(
            common_dir, followlinks=False
        ):
            directory_names[:] = sorted(directory_names)
            for filename in sorted(filenames):
                path = Path(directory) / filename
                value = path.lstat()
                assert not stat.S_ISLNK(value.st_mode)
                if not stat.S_ISREG(value.st_mode):
                    continue
                identities[path] = (
                    int(value.st_dev),
                    int(value.st_ino),
                    int(value.st_nlink),
                    int(value.st_mode),
                    int(value.st_size),
                    int(value.st_mtime_ns),
                    int(value.st_ctime_ns),
                )
        return identities

    before = source_regular_identities()
    loose_path = common_dir / "objects" / loose_oid[:2] / loose_oid[2:]
    pack_paths = {
        path
        for path in before
        if path.parent == common_dir / "objects" / "pack"
        and path.suffix in {".pack", ".idx"}
    }
    index_paths = {path for path in before if path.name == "index"}
    assert loose_path in before
    assert {path.suffix for path in pack_paths} == {".pack", ".idx"}
    assert index_paths

    with tempfile.TemporaryDirectory(dir=tmp_path) as temporary:
        sealed, snapshot, records = _authority_git_seal_mounts(plan, Path(temporary))
        assert snapshot["hardlinks_used"] is False
        assert records
        assert all(
            item.bind_source.lstat().st_ino != item.reviewed_source.lstat().st_ino
            for item in sealed
        )
        projected_regulars = {
            (int(path.lstat().st_dev), int(path.lstat().st_ino))
            for path in sealed[0].bind_source.rglob("*")
            if path.is_file() and not path.is_symlink()
        }
        source_regulars = {(identity[0], identity[1]) for identity in before.values()}
        assert projected_regulars
        assert projected_regulars.isdisjoint(source_regulars)

    assert source_regular_identities() == before


@pytest.mark.parametrize(
    ("kind", "reason"),
    (
        (
            "symlink",
            "authority_validation_git_metadata_symlink_forbidden",
        ),
        (
            "missing",
            "authority_validation_gitdir_pointer_unresolved",
        ),
        (
            "comma",
            "authority_validation_gitdir_pointer_unresolved",
        ),
        (
            "control",
            "authority_validation_gitdir_pointer_unresolved",
        ),
    ),
)
def test_replay_plan_rejects_unsafe_gitdir_near_misses(
    tmp_path: Path,
    kind: str,
    reason: str,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    marker = workspace / ".git"
    if kind == "symlink":
        metadata = tmp_path / "metadata"
        metadata.mkdir()
        marker.symlink_to(metadata, target_is_directory=True)
    elif kind == "missing":
        marker.write_text("gitdir: ../missing\n", encoding="utf-8")
    elif kind == "comma":
        metadata = tmp_path / "metadata,unsafe"
        metadata.mkdir()
        marker.write_text(f"gitdir: {metadata}\n", encoding="utf-8")
    else:
        marker.write_text("gitdir: ../metadata\nunreviewed-control\n", encoding="utf-8")

    with pytest.raises(AuthorityGitReplayError) as captured:
        _authority_git_replay_plan(workspace)

    assert captured.value.reason == reason


def test_replay_plan_rejects_symlinked_configured_root(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    _init_repository(workspace, label="workspace")
    (workspace / ".gitmodules").write_text(
        '[submodule "nested"]\n\tpath = nested\n\turl = ../nested-source\n',
        encoding="utf-8",
    )
    _git(workspace, "add", ".gitmodules")
    _git(workspace, "commit", "-qm", "declare nested root")
    nested_source = tmp_path / "nested-source"
    _init_repository(nested_source, label="nested")
    (workspace / "nested").symlink_to(nested_source, target_is_directory=True)

    with pytest.raises(AuthorityGitReplayError) as captured:
        _plan(workspace)

    assert captured.value.reason == (
        "authority_validation_git_metadata_symlink_forbidden"
    )


def test_replay_plan_rejects_hidden_worktree_gitmodules_bytes(
    tmp_path: Path,
) -> None:
    _source, linked = _linked_workspace(tmp_path)
    _git(linked, "update-index", "--assume-unchanged", ".gitmodules")
    with (linked / ".gitmodules").open("a", encoding="utf-8") as stream:
        stream.write(
            '[submodule "unreviewed"]\n'
            "\tpath = unreviewed\n"
            "\turl = ../unreviewed\n"
        )

    with pytest.raises(AuthorityGitReplayError) as captured:
        _plan(linked)

    assert captured.value.reason == "authority_validation_gitmodules_unreviewed"


def test_replay_plan_rejects_configured_path_that_is_not_a_gitlink(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    _init_repository(workspace, label="workspace")
    ordinary = workspace / "ordinary"
    ordinary.mkdir()
    (ordinary / "tracked.txt").write_text("ordinary\n", encoding="utf-8")
    (workspace / ".gitmodules").write_text(
        '[submodule "ordinary"]\n\tpath = ordinary\n\turl = ../ordinary\n',
        encoding="utf-8",
    )
    _git(workspace, "add", ".gitmodules", "ordinary/tracked.txt")
    _git(workspace, "commit", "-qm", "declare false submodule")
    _git(ordinary, "init", "-q", "-b", "main")

    with pytest.raises(AuthorityGitReplayError) as captured:
        _plan(workspace)

    assert captured.value.reason == (
        "authority_validation_git_configured_root_not_gitlink"
    )


def test_replay_plan_rejects_child_head_that_misses_parent_gitlink(
    tmp_path: Path,
) -> None:
    _source, linked = _linked_workspace(tmp_path)
    nested = linked / "nested"
    _git(nested, "config", "user.name", "Authority Git Replay")
    _git(nested, "config", "user.email", "authority-git@example.invalid")
    (nested / "tracked.txt").write_text("advanced\n", encoding="utf-8")
    _git(nested, "commit", "-qam", "advance child only")

    with pytest.raises(AuthorityGitReplayError) as captured:
        _plan(linked)

    assert captured.value.reason == (
        "authority_validation_git_configured_root_identity_mismatch"
    )


@pytest.mark.parametrize(
    ("prefix", "reason"),
    (
        ("", "authority_validation_git_external_alternate_forbidden"),
        (" ", "authority_validation_git_alternate_unresolved"),
    ),
)
def test_replay_plan_rejects_external_or_rewritten_object_alternate(
    tmp_path: Path,
    prefix: str,
    reason: str,
) -> None:
    _source, linked = _linked_workspace(tmp_path)
    external = tmp_path / "external-objects"
    _init_repository(external, label="external")
    common = Path(
        _git(linked, "rev-parse", "--path-format=absolute", "--git-common-dir")
    )
    alternates = common / "objects" / "info" / "alternates"
    alternates.write_text(
        prefix + str(external / ".git" / "objects") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(AuthorityGitReplayError) as captured:
        _plan(linked)

    assert captured.value.reason == reason


def test_mount_bootstrap_rejects_an_interposed_source_swap(
    tmp_path: Path,
) -> None:
    expected = tmp_path / "expected"
    replacement = tmp_path / "replacement"
    expected.write_text("expected\n", encoding="utf-8")
    replacement.write_text("replacement\n", encoding="utf-8")
    value = expected.lstat()
    record = {
        "destination": str(expected),
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "mode_type": value.st_mode & 0o170000,
        "size": int(value.st_size),
        "sha256": hashlib.sha256(expected.read_bytes()).hexdigest(),
    }
    payload = base64.b64encode(
        json.dumps([record], sort_keys=True).encode("utf-8")
    ).decode("ascii")
    held = tmp_path / "held"
    expected.rename(held)
    replacement.rename(expected)
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                _AUTHORITY_GIT_MOUNT_BOOTSTRAP,
                payload,
                sys.executable,
                "-c",
                "raise SystemExit(91)",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    finally:
        expected.rename(replacement)
        held.rename(expected)

    assert completed.returncode == 75
    assert "authority-git-mount-identity-mismatch" in completed.stderr


def test_linked_replay_argv_seals_sources_before_tmpfs_and_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _source, linked = _linked_workspace(tmp_path)
    plan = _plan(linked)
    target_commit = _git(linked, "rev-parse", "HEAD")
    target_tree = _git(linked, "rev-parse", "HEAD^{tree}")
    contract = {
        "available": True,
        "docker_path": "/usr/bin/docker",
        "docker_endpoint": "unix:///run/docker.sock",
        "image_id": "sha256:" + ("a" * 64),
        "gpu_uuid": "GPU-11111111-2222-3333-4444-555555555555",
        "contract_id": "linked-replay-contract",
    }
    monkeypatch.setattr(
        TodoImplementationDaemon,
        "_authority_validation_isolation_contract",
        staticmethod(lambda: contract),
    )
    monkeypatch.setattr(
        daemon_module,
        "_authority_git_replay_plan",
        lambda _workspace: plan,
    )
    popen_calls: list[list[str]] = []

    class CompletedDockerProcess:
        pid = 424242
        returncode = 0

        def __init__(self) -> None:
            self.stdout = io.BytesIO(b"validated\n")

        def poll(self) -> int:
            return self.returncode

    def fake_popen(command: list[str], **_kwargs: object):
        popen_calls.append(list(command))
        return CompletedDockerProcess()

    def fake_run(command: list[str], **_kwargs: object):
        if command and Path(str(command[0])).name == "git":
            return subprocess.CompletedProcess(
                command,
                0,
                f"{target_commit}\n{target_tree}\n".encode("ascii"),
                b"",
            )
        return subprocess.CompletedProcess(command, 0, "")

    monkeypatch.setattr(daemon_module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(daemon_module.subprocess, "run", fake_run)

    dcr012_command = (
        "python3 -m external.ipfs_accelerate.ipfs_accelerate_py."
        "agent_supervisor.analysis.deterministic_repair_analyzer_health "
        "validate --workspace . --forest data/agent_supervisor/"
        "deterministic_contract_repair/forest.json --artifact data/"
        "agent_supervisor/deterministic_contract_repair/analyzer-health.json "
        "--max-bytes 1048576"
    )
    generic_task_cid = daemon_module.content_identity(
        {"task": "DCR-012", "revision": 1}
    )
    generic_plan_body = {
        "schema": (
            "ipfs_accelerate_py.agent_supervisor." "authority-validation-command-plan@2"
        ),
        "authority_profile": "generic_workspace_only@1",
        "task_id": "DCR-012",
        "canonical_task_cid": generic_task_cid,
        "scope": "pre_merge",
        "commands": [dcr012_command],
        "target_commit": target_commit,
        "target_tree": target_tree,
        "git_common_anchor": "",
        "git_dir": "",
    }
    result = TodoImplementationDaemon._authority_validation_command_runner(
        spec=SimpleNamespace(
            command=dcr012_command,
            raw_command=dcr012_command,
            ordinal=0,
            validation_id="",
        ),
        workspace_path=linked,
        timeout_seconds=10,
        environment={
            daemon_module._AUTHORITY_VALIDATION_TASK_ENV: "DCR-012",
            daemon_module._AUTHORITY_VALIDATION_TASK_CID_ENV: generic_task_cid,
            daemon_module._AUTHORITY_VALIDATION_SCOPE_ENV: "pre_merge",
            daemon_module._AUTHORITY_VALIDATION_PROFILE_ENV: (
                "generic_workspace_only@1"
            ),
            daemon_module._AUTHORITY_VALIDATION_COMMANDS_ENV: json.dumps(
                [dcr012_command], separators=(",", ":")
            ),
            daemon_module._AUTHORITY_VALIDATION_PLAN_ENV: (
                daemon_module.content_identity(generic_plan_body)
            ),
            daemon_module._AUTHORITY_VALIDATION_TARGET_COMMIT_ENV: target_commit,
            daemon_module._AUTHORITY_VALIDATION_TARGET_TREE_ENV: target_tree,
        },
    )

    assert result["returncode"] == 0
    assert len(popen_calls) == 1
    argv = popen_calls[0]
    mount_arguments = [item for item in argv if item.startswith("--mount=")]
    assert mount_arguments[0] == (
        f"--mount=type=bind,src={linked},dst={linked},readonly"
    )
    assert len(mount_arguments) == 1
    assert all("readonly" in item for item in mount_arguments)
    tmpfs_index = next(
        index
        for index, item in enumerate(argv)
        if item.startswith(
            f"--tmpfs={daemon_module.AUTHORITY_VALIDATION_SCRATCH_PATH}:"
        )
    )
    assert all(argv.index(item) < tmpfs_index for item in mount_arguments)
    image_index = argv.index(contract["image_id"])
    assert argv[image_index + 1 : image_index + 4] == [
        "/usr/bin/python",
        "-c",
        _AUTHORITY_GIT_MOUNT_BOOTSTRAP,
    ]
    receipt = result["authority_validation_isolation_receipt"]
    assert receipt["host_filesystem"] == "workspace_only_read_only"
    assert receipt["git_metadata_replay"]["mode"] == "not_requested"
    assert receipt["git_safe_directory_config"]["mounted"] is False
    assert receipt["git_mount_bootstrap"]["records"] == []
    assert receipt["git_metadata_sealed_mounts"] == []


def test_dcr012_and_board_drift_cannot_request_dcr011_projection(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    _init_repository(workspace, label="workspace")
    forest_command = daemon_module.AUTHORITY_VALIDATION_DCR_RAW_COMMANDS[1]
    for environment in (
        {
            daemon_module._AUTHORITY_VALIDATION_TASK_ENV: "DCR-012",
            daemon_module._AUTHORITY_VALIDATION_SCOPE_ENV: "post_merge",
        },
        {
            daemon_module._AUTHORITY_VALIDATION_TASK_ENV: "DCR-011",
            daemon_module._AUTHORITY_VALIDATION_TASK_CID_ENV: "board-drifted",
            daemon_module._AUTHORITY_VALIDATION_SCOPE_ENV: "post_merge",
        },
    ):
        result = TodoImplementationDaemon._authority_validation_command_runner(
            spec=SimpleNamespace(
                command=forest_command,
                raw_command=forest_command,
                ordinal=1,
                validation_id="",
            ),
            workspace_path=workspace,
            timeout_seconds=10,
            environment=environment,
        )
        assert result["returncode"] == 78
        assert result["reason"] in {
            "authority_validation_git_projection_not_authorized",
            "authority_validation_dcr_plan_binding_mismatch",
        }
        assert "authority_validation_isolation_receipt" not in result


@pytest.mark.parametrize(
    "mutation",
    (
        "drop_top_level",
        "add_top_level",
        "empty_replay",
        "cleanup_failed",
        "ordinal_mode_mismatch",
        "command_binding_drift",
    ),
)
def test_strict_receipt_verifier_rejects_self_consistent_forgery(
    tmp_path: Path,
    mutation: str,
) -> None:
    receipt, contract, result, validated = _not_requested_receipt_fixture(tmp_path)
    assert daemon_module._authority_validation_isolation_receipt_valid(
        receipt,
        contract=contract,
        result=result,
        validated_tree_identity=validated,
    )
    forged = copy.deepcopy(receipt)
    if mutation == "drop_top_level":
        forged.pop("git_postflight")
    elif mutation == "add_top_level":
        forged["unreviewed"] = True
    elif mutation == "empty_replay":
        forged["git_metadata_replay"] = {}
    elif mutation == "cleanup_failed":
        forged["git_projection_cleanup"]["succeeded"] = False
    elif mutation == "ordinal_mode_mismatch":
        forged["authority_command_binding"]["ordinal"] = 1
    else:
        forged["authority_command_binding"]["scope"] = "pre_merge"
    forged_body = {key: forged[key] for key in forged if key != "receipt_id"}
    forged["receipt_id"] = daemon_module.content_identity(forged_body)
    assert not daemon_module._authority_validation_isolation_receipt_valid(
        forged,
        contract=contract,
        result=result,
        validated_tree_identity=validated,
    )


def test_strict_receipt_rejects_self_consistent_noncanonical_oid_width(
    tmp_path: Path,
) -> None:
    receipt, contract, result, validated = _not_requested_receipt_fixture(tmp_path)
    forged = copy.deepcopy(receipt)
    binding = forged["authority_command_binding"]
    binding["expected_target_commit"] = "1" * 41
    binding["expected_target_tree"] = "2" * 41
    plan_body = {
        "schema": (
            "ipfs_accelerate_py.agent_supervisor." "authority-validation-command-plan@2"
        ),
        "authority_profile": binding["authority_profile"],
        "task_id": binding["task_id"],
        "canonical_task_cid": binding["canonical_task_cid"],
        "scope": binding["scope"],
        "commands": list(daemon_module.AUTHORITY_VALIDATION_DCR_RAW_COMMANDS),
        "target_commit": binding["expected_target_commit"],
        "target_tree": binding["expected_target_tree"],
        "git_common_anchor": binding["expected_git_common_anchor"],
        "git_dir": binding["expected_git_dir"],
    }
    binding["plan_id"] = daemon_module.content_identity(plan_body)
    binding_body = {
        key: binding[key]
        for key in binding
        if key not in {"command_binding_id", "command_binding_cache_id"}
    }
    binding["command_binding_id"] = daemon_module.content_identity(binding_body)
    binding["command_binding_cache_id"] = daemon_module.content_identity(
        {
            "schema": (
                "ipfs_accelerate_py.agent_supervisor."
                "authority-validation-command-cache-key@1"
            ),
            "workspace_path": forged["workspace_path"],
            "plan_id": binding["plan_id"],
            "command_binding_id": binding["command_binding_id"],
        }
    )
    forged_body = {key: forged[key] for key in forged if key != "receipt_id"}
    forged["receipt_id"] = daemon_module.content_identity(forged_body)
    forged_result = copy.deepcopy(result)
    forged_result["authority_validation_command_binding"] = dict(binding)
    forged_validated = copy.deepcopy(validated)
    forged_validated["target_commit"] = binding["expected_target_commit"]
    forged_validated["repository_tree_id"] = (
        f"git-tree:{binding['expected_target_tree']}"
    )
    assert not daemon_module._authority_validation_isolation_receipt_valid(
        forged,
        contract=contract,
        result=forged_result,
        validated_tree_identity=forged_validated,
    )


@pytest.mark.timeout(180)
def test_exact_forest_cli_runs_all_lifecycle_states_in_sealed_image(
    tmp_path: Path,
) -> None:
    contract = TodoImplementationDaemon._authority_validation_isolation_contract()
    if contract.get("available") is not True:
        pytest.skip(str(contract.get("reason") or "sealed image unavailable"))
    fixture = _real_forest_cli_fixture(tmp_path, linked_outer=True)
    secret = tmp_path / "dangling-secret.txt"
    secret.write_text("provider secret must not enter projection\n", encoding="utf-8")
    dangling_oid = _git(fixture.workspace, "hash-object", "-w", str(secret))
    _git(
        fixture.workspace,
        "config",
        "remote.private.url",
        "https://credential.invalid/private-token",
    )
    observed_modes: list[str] = []
    last_result: dict[str, object] = {}
    for phase in ("captured", "artifact_carried", "integrated", "todo_completed"):
        command, environment, validated_tree = _exact_context(fixture.workspace)
        result = TodoImplementationDaemon._authority_validation_command_runner(
            spec=SimpleNamespace(
                command=command,
                raw_command=command,
                ordinal=1,
                validation_id="",
            ),
            workspace_path=fixture.workspace,
            timeout_seconds=120,
            environment=environment,
        )
        assert result["returncode"] == 0, result.get("output")
        receipt = result["authority_validation_isolation_receipt"]
        root = receipt["git_metadata_replay"]["root_identities"][0]
        observed_modes.append(root["lifecycle"]["mode"])
        assert dangling_oid not in root["object_ids"]
        assert receipt["git_sealed_projection"]["raw_common_config_exposed"] is False
        result.update(
            {
                "command": command,
                "raw_command": command,
                "ordinal": 1,
                "validation_id": "",
                "cache_key": "a" * 64,
            }
        )
        assert daemon_module._authority_validation_isolation_receipt_valid(
            receipt,
            contract=contract,
            result=result,
            validated_tree_identity=validated_tree,
        )
        last_result = result
        if phase == "captured":
            fixture.carry()
        elif phase == "artifact_carried":
            fixture.merge()
        elif phase == "integrated":
            fixture.complete_todo()
    assert observed_modes == [
        "captured",
        "artifact_carried",
        "integrated",
        "todo_completed",
    ]
    last_receipt = last_result["authority_validation_isolation_receipt"]
    cleanup = last_receipt["git_projection_cleanup"]
    transport = last_receipt["git_sealed_projection"]["transport"]
    assert cleanup["derived_cleanup_journal_released"] is True
    assert cleanup["derived_cleanup_quiet_milliseconds"] >= 2000
    assert 2000 <= cleanup["derived_cleanup_elapsed_milliseconds"] <= 30000
    assert cleanup["derived_cleanup_time_limit_milliseconds"] == 30000
    assert not Path(cleanup["derived_cleanup_journal_path"]).exists()
    for docker_arguments in (
        (
            "container",
            "ls",
            "--all",
            "--filter",
            f"label={transport['cleanup_label']}",
            "--format",
            "{{.ID}}",
        ),
        (
            "image",
            "ls",
            "--all",
            "--no-trunc",
            "--quiet",
            "--filter",
            f"label={transport['cleanup_label']}",
        ),
        (
            "image",
            "ls",
            "--all",
            "--no-trunc",
            "--filter",
            f"reference={transport['cleanup_image_tag']}",
            "--format",
            "{{.ID}}",
        ),
    ):
        listed = subprocess.run(
            [str(daemon_module.AUTHORITY_VALIDATION_DOCKER_PATH), *docker_arguments],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            check=False,
        )
        assert listed.returncode == 0, listed.stderr
        assert not listed.stdout.strip()

    forged = copy.deepcopy(last_result["authority_validation_isolation_receipt"])
    forged["git_metadata_replay"] = {}
    forged_body = {key: forged[key] for key in forged if key != "receipt_id"}
    forged["receipt_id"] = daemon_module.content_identity(forged_body)
    assert not daemon_module._authority_validation_isolation_receipt_valid(
        forged,
        contract=contract,
        result=last_result,
        validated_tree_identity={
            "target_commit": last_result["authority_validation_command_binding"][
                "expected_target_commit"
            ],
            "repository_tree_id": (
                "git-tree:"
                + last_result["authority_validation_command_binding"][
                    "expected_target_tree"
                ]
            ),
        },
    )

    current_pid = os.getpid()
    current_ticks = daemon_module._authority_process_start_ticks(current_pid)
    assert current_ticks is not None
    lease_mutations = (
        ("pid", current_pid + 1),
        ("process_start_ticks", current_ticks + 1),
        ("created_unix_ns", time.time_ns() + 1_000_000_000),
        ("created_unix_ns", 1),
        ("mount_source", f"/proc/{current_pid}/fd/01/metadata"),
        ("mount_source", f"/proc/{current_pid + 1}/fd/3/metadata"),
    )
    for lease_key, forged_value in lease_mutations:
        forged = copy.deepcopy(last_result["authority_validation_isolation_receipt"])
        lease = forged["git_sealed_projection"]["projection_lease"]
        original_lease_size = len(
            (daemon_module.canonical_json(dict(lease)) + "\n").encode("utf-8")
        )
        lease[lease_key] = forged_value
        lease_body = {key: lease[key] for key in lease if key != "manifest_id"}
        lease["manifest_id"] = daemon_module.content_identity(lease_body)
        mutated_lease_size = len(
            (daemon_module.canonical_json(dict(lease)) + "\n").encode("utf-8")
        )
        lease_size_delta = mutated_lease_size - original_lease_size
        forged["git_sealed_projection"]["sealed_bytes"] += lease_size_delta
        forged["git_sealed_projection"]["aggregate_bytes"] += lease_size_delta
        forged_body = {key: forged[key] for key in forged if key != "receipt_id"}
        forged["receipt_id"] = daemon_module.content_identity(forged_body)
        assert not daemon_module._authority_validation_isolation_receipt_valid(
            forged,
            contract=contract,
            result=last_result,
            validated_tree_identity={
                "target_commit": last_result["authority_validation_command_binding"][
                    "expected_target_commit"
                ],
                "repository_tree_id": (
                    "git-tree:"
                    + last_result["authority_validation_command_binding"][
                        "expected_target_tree"
                    ]
                ),
            },
        ), lease_key

    def rehash_projected_receipt(
        forged_receipt: dict[str, object],
        forged_result: dict[str, object],
    ) -> None:
        replay = forged_receipt["git_metadata_replay"]
        for root in replay["root_identities"]:
            lifecycle = root["lifecycle"]
            root["lifecycle_id"] = daemon_module.content_identity(lifecycle)
            root["object_manifest_id"] = daemon_module.content_identity(
                {
                    "schema": (
                        "ipfs_accelerate_py.agent_supervisor."
                        "authority-git-object-closure@1"
                    ),
                    "head": root["head"],
                    "tree": root["tree"],
                    "shallow_boundary": root["shallow_boundary"],
                    "object_ids": root["object_ids"],
                    "object_type_counts": root["object_type_counts"],
                    "uncompressed_bytes": root["object_source_bytes"],
                    "lifecycle_id": root["lifecycle_id"],
                }
            )
        for pack in forged_receipt["git_sealed_projection"]["object_packs"]:
            expected_oids = sorted(
                {
                    oid
                    for root in replay["root_identities"]
                    if root["common_dir"] == pack["common_dir"]
                    for oid in root["object_ids"]
                }
            )
            pack["object_count"] = len(expected_oids)
            pack["object_set_id"] = daemon_module.content_identity(expected_oids)
        plan_body = {
            "schema": (
                "ipfs_accelerate_py.agent_supervisor."
                "authority-git-metadata-replay-plan@2"
            ),
            "workspace": replay["workspace_path"],
            "roots": [
                (
                    replay["workspace_path"]
                    if relative == "."
                    else str(Path(replay["workspace_path"]) / relative)
                )
                for relative in replay["configured_roots"]
            ],
            "root_identities": replay["root_identities"],
            "mounts": replay["external_mounts"],
            "observation_identities": replay["observation_identities"],
            "preflight_budget": replay["preflight_budget"],
        }
        replay["preflight_id"] = daemon_module.content_identity(plan_body)
        replay["postflight_id"] = replay["preflight_id"]
        binding = forged_receipt["authority_command_binding"]
        plan = {
            "schema": (
                "ipfs_accelerate_py.agent_supervisor."
                "authority-validation-command-plan@2"
            ),
            "authority_profile": binding["authority_profile"],
            "task_id": binding["task_id"],
            "canonical_task_cid": binding["canonical_task_cid"],
            "scope": binding["scope"],
            "commands": list(daemon_module.AUTHORITY_VALIDATION_DCR_RAW_COMMANDS),
            "target_commit": binding["expected_target_commit"],
            "target_tree": binding["expected_target_tree"],
            "git_common_anchor": binding["expected_git_common_anchor"],
            "git_dir": binding["expected_git_dir"],
        }
        binding["plan_id"] = daemon_module.content_identity(plan)
        binding_body = {
            key: binding[key]
            for key in binding
            if key not in {"command_binding_id", "command_binding_cache_id"}
        }
        binding["command_binding_id"] = daemon_module.content_identity(binding_body)
        binding["command_binding_cache_id"] = daemon_module.content_identity(
            {
                "schema": (
                    "ipfs_accelerate_py.agent_supervisor."
                    "authority-validation-command-cache-key@1"
                ),
                "workspace_path": forged_receipt["workspace_path"],
                "plan_id": binding["plan_id"],
                "command_binding_id": binding["command_binding_id"],
            }
        )
        forged_result["authority_validation_command_binding"] = dict(binding)
        forged_body = {
            key: forged_receipt[key] for key in forged_receipt if key != "receipt_id"
        }
        forged_receipt["receipt_id"] = daemon_module.content_identity(forged_body)

    forged = copy.deepcopy(last_result["authority_validation_isolation_receipt"])
    forged_result = copy.deepcopy(last_result)
    root = forged["git_metadata_replay"]["root_identities"][0]
    original_source_bytes = root["object_source_bytes"]
    fake_oid = next(
        f"{value:040x}"
        for value in range(1, 1000)
        if f"{value:040x}" not in root["object_ids"]
    )
    root["object_ids"] = sorted([*root["object_ids"], fake_oid])
    root["object_count"] += 1
    root["object_type_counts"]["tree"] += 1
    root["object_source_bytes"] += 1
    forged["git_sealed_projection"]["source_bytes"] += 1
    forged["git_sealed_projection"]["aggregate_bytes"] += 1
    rehash_projected_receipt(forged, forged_result)
    assert root["object_source_bytes"] == original_source_bytes + 1
    assert not daemon_module._authority_validation_isolation_receipt_valid(
        forged,
        contract=contract,
        result=forged_result,
        validated_tree_identity={
            "target_commit": forged["authority_command_binding"][
                "expected_target_commit"
            ],
            "repository_tree_id": (
                "git-tree:"
                + forged["authority_command_binding"]["expected_target_tree"]
            ),
        },
    )

    forged = copy.deepcopy(last_result["authority_validation_isolation_receipt"])
    forged_result = copy.deepcopy(last_result)
    root = forged["git_metadata_replay"]["root_identities"][0]
    root_pack = next(
        pack
        for pack in forged["git_sealed_projection"]["object_packs"]
        if pack["common_dir"] == root["common_dir"]
    )
    parsed_pack = daemon_module.verify_exact_git_pack_base64(
        root_pack["pack_base64"],
        root_pack["index_base64"],
        object_format=root["object_format"],
        limits=daemon_module.AUTHORITY_VALIDATION_GIT_PACK_LIMITS,
    )
    alternate_tree = next(
        oid
        for oid, value in parsed_pack.items()
        if value.object_type == "tree" and oid != root["tree"]
    )
    root["tree"] = alternate_tree
    root["lifecycle"]["transition_records"][-1]["tree"] = alternate_tree
    forged["authority_command_binding"]["expected_target_tree"] = alternate_tree
    rehash_projected_receipt(forged, forged_result)
    assert not daemon_module._authority_validation_isolation_receipt_valid(
        forged,
        contract=contract,
        result=forged_result,
        validated_tree_identity={
            "target_commit": forged["authority_command_binding"][
                "expected_target_commit"
            ],
            "repository_tree_id": f"git-tree:{alternate_tree}",
        },
    )

    forged = copy.deepcopy(last_result["authority_validation_isolation_receipt"])
    transport = forged["git_sealed_projection"]["transport"]
    cleanup_lease = transport["cleanup_lease"]
    cleanup_lease["authority_plan_id"] = "forged-plan"
    seed_keys = (
        "schema",
        "uid",
        "pid",
        "process_start_ticks",
        "created_unix_ns",
        "docker_endpoint",
        "base_image_id",
        "authority_plan_id",
        "command_binding_id",
        "projection_manifest_id",
    )
    forged_lease_id = hashlib.sha256(
        daemon_module.canonical_json(
            {key: cleanup_lease[key] for key in seed_keys}
        ).encode("utf-8")
    ).hexdigest()
    cleanup_lease.update(
        {
            "lease_id": forged_lease_id,
            "cleanup_label_value": forged_lease_id,
            "image_tag": f"ipfs-accelerate-authority-git:{forged_lease_id}",
            "preflight_container_name": (
                f"ipfs-authority-git-preflight-{forged_lease_id[:32]}"
            ),
            "builder_container_name": (
                f"ipfs-authority-git-builder-{forged_lease_id[:32]}"
            ),
        }
    )
    cleanup_lease_body = {
        key: cleanup_lease[key] for key in cleanup_lease if key != "manifest_id"
    }
    cleanup_lease["manifest_id"] = daemon_module.content_identity(cleanup_lease_body)
    forged_journal = (
        daemon_module._authority_git_docker_cleanup_parent() / f"{forged_lease_id}.json"
    )
    transport.update(
        {
            "cleanup_lease_id": forged_lease_id,
            "cleanup_label": (
                f"{daemon_module._AUTHORITY_GIT_DOCKER_CLEANUP_LABEL_KEY}="
                f"{forged_lease_id}"
            ),
            "cleanup_image_tag": (f"ipfs-accelerate-authority-git:{forged_lease_id}"),
            "cleanup_journal_path": str(forged_journal),
        }
    )
    transport_body = {key: transport[key] for key in transport if key != "transport_id"}
    transport["transport_id"] = daemon_module.content_identity(transport_body)
    forged_cleanup = forged["git_projection_cleanup"]
    forged_cleanup["derived_cleanup_lease_id"] = forged_lease_id
    forged_cleanup["derived_cleanup_journal_path"] = str(forged_journal)
    forged_body = {key: forged[key] for key in forged if key != "receipt_id"}
    forged["receipt_id"] = daemon_module.content_identity(forged_body)
    assert not daemon_module._authority_validation_isolation_receipt_valid(
        forged,
        contract=contract,
        result=last_result,
        validated_tree_identity={
            "target_commit": forged["authority_command_binding"][
                "expected_target_commit"
            ],
            "repository_tree_id": (
                "git-tree:"
                + forged["authority_command_binding"]["expected_target_tree"]
            ),
        },
    )


@pytest.mark.timeout(120)
def test_sealed_projection_excludes_blobs_refs_and_raw_config(
    tmp_path: Path,
) -> None:
    contract = TodoImplementationDaemon._authority_validation_isolation_contract()
    if contract.get("available") is not True:
        pytest.skip(str(contract.get("reason") or "sealed image unavailable"))
    fixture = _real_forest_cli_fixture(tmp_path, linked_outer=True)
    ordinary_oid = _git(fixture.workspace, "hash-object", ".gitignore")
    secret_file = tmp_path / "unreachable-secret"
    secret_file.write_text("unreachable secret object\n", encoding="utf-8")
    secret_oid = _git(fixture.workspace, "hash-object", "-w", str(secret_file))
    secret_commit = _git(
        fixture.workspace,
        "-c",
        "user.name=Authority Git Replay",
        "-c",
        "user.email=authority-git@example.invalid",
        "commit-tree",
        _git(fixture.workspace, "rev-parse", "HEAD^{tree}"),
        "-m",
        "private cross-lane commit",
    )
    _git(fixture.workspace, "branch", "private-cross-lane", secret_commit)
    _git(
        fixture.workspace,
        "config",
        "remote.private.url",
        "https://credential.invalid/private-token",
    )
    budget = daemon_module._AuthorityGitSetupBudget.begin()
    plan = _authority_git_replay_plan(
        fixture.workspace,
        budget=budget,
        lifecycle_subject=fixture.subject,
    )
    root_oids = set(plan.root_identities[0]["object_ids"])
    assert {ordinary_oid, secret_oid, secret_commit}.isdisjoint(root_oids)
    projection, manifest, projection_descriptor = (
        daemon_module._authority_git_create_projection(fixture.workspace, budget=budget)
    )
    cleanup_guard = daemon_module._AuthorityValidationCleanupGuard()
    cleanup_guard.register_projection(
        projection, manifest, descriptor=projection_descriptor
    )
    try:
        sealed, _snapshot, _records = _authority_git_seal_mounts(
            plan,
            Path(str(manifest["mount_source"])).parent,
            budget=budget,
            projection_descriptor=projection_descriptor,
        )
        manifest = daemon_module._authority_git_finalize_projection(
            projection,
            projection_descriptor,
            manifest,
            budget=budget,
        )
        cleanup_guard.register_projection(projection, manifest)
        common_dir = plan.mounts[0].destination
        metadata = sealed[0].bind_source
        assert b"credential.invalid" not in b"".join(
            path.read_bytes() for path in metadata.rglob("config") if path.is_file()
        )
        docker_environment = {
            "DOCKER_HOST": str(contract["docker_endpoint"]),
            "DOCKER_CONFIG": "/nonexistent/ipfs-accelerate-docker-config",
            "HOME": "/nonexistent/ipfs-accelerate-docker-home",
            "PATH": os.defpath,
        }
        docker_prefix = (
            str(contract["docker_path"]),
            "--host",
            str(contract["docker_endpoint"]),
        )
        docker_cleanup_reclaimer = (
            daemon_module._authority_git_reclaim_docker_cleanup_leases(
                docker_prefix=docker_prefix,
                docker_environment=docker_environment,
                deadline=(
                    time.monotonic()
                    + daemon_module._AUTHORITY_GIT_DOCKER_CLEANUP_SECONDS
                ),
            )
        )
        derived_image, transport, layer_manifest = (
            daemon_module._authority_git_build_derived_image(
                projection_descriptor,
                manifest,
                common_dir,
                base_image_id=str(contract["image_id"]),
                docker_prefix=docker_prefix,
                docker_environment=docker_environment,
                budget=budget,
                cleanup_guard=cleanup_guard,
                authority_plan_id="confidentiality-test-plan",
                command_binding_id="confidentiality-test-binding",
                docker_cleanup_reclaimer=docker_cleanup_reclaimer,
            )
        )
        assert transport["derived_layers"] == [
            *transport["base_layers"],
            transport["added_layer"],
        ]
        assert cleanup_guard.cleanup_projection_now()
        payload = base64.b64encode(
            daemon_module.canonical_json(layer_manifest).encode("utf-8")
        ).decode("ascii")
        script = r"""
set -eu
workspace="$1"
ordinary="$2"
secret="$3"
private_commit="$4"
test -z "$(git -C "$workspace" status --porcelain=v1)"
for oid in "$ordinary" "$secret" "$private_commit"; do
  if git -C "$workspace" cat-file -e "$oid" 2>/dev/null; then
    exit 41
  fi
done
test -z "$(git -C "$workspace" config --get-regexp '^remote\.' || true)"
test -z "$(git -C "$workspace" show-ref || true)"
git -C "$workspace" rev-parse HEAD >/dev/null
""".strip()
        completed = subprocess.run(
            [
                str(contract["docker_path"]),
                "--host",
                str(contract["docker_endpoint"]),
                "run",
                "--rm",
                "--pull=never",
                "--network=none",
                "--read-only",
                "--cap-drop=ALL",
                "--security-opt=no-new-privileges:true",
                "--log-driver=none",
                f"--user={os.getuid()}:{os.getgid()}",
                f"--workdir={fixture.workspace}",
                (
                    f"--mount=type=bind,src={fixture.workspace},"
                    f"dst={fixture.workspace},readonly"
                ),
                (
                    f"--tmpfs={daemon_module.AUTHORITY_VALIDATION_SCRATCH_PATH}:"
                    "rw,noexec,nosuid,nodev,size=64m,mode=1777"
                ),
                "--env=GIT_CONFIG_NOSYSTEM=1",
                "--env=GIT_CONFIG_GLOBAL=/dev/null",
                "--env=GIT_NO_LAZY_FETCH=1",
                "--env=GIT_NO_REPLACE_OBJECTS=1",
                "--env=GIT_OPTIONAL_LOCKS=0",
                derived_image,
                "/usr/bin/python",
                "-c",
                daemon_module._AUTHORITY_GIT_DERIVED_LAYER_VERIFY,
                payload,
                "/bin/bash",
                "--noprofile",
                "--norc",
                "-c",
                script,
                "sealed-confidentiality",
                str(fixture.workspace),
                ordinary_oid,
                secret_oid,
                secret_commit,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=120,
            check=False,
            env=docker_environment,
        )
        assert completed.returncode == 0, completed.stdout
    finally:
        assert cleanup_guard.cleanup_all()["succeeded"]


@pytest.mark.timeout(300)
def test_linked_committed_carrier_runs_exact_ordered_authority_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The decisive handoff really runs pytest then the forest CLI.

    Unlike the diagnostic lifecycle test above, this is a release canary: an
    unavailable sealed contract is a failure, never a skip.  It exercises the
    clean committed-carrier helper and therefore proves that proposal-bound
    evidence is replaced by the empty-ID, uncached two-result authority run.
    """

    contract = TodoImplementationDaemon._authority_validation_isolation_contract()
    assert contract.get("available") is True, contract
    fixture = _real_forest_cli_fixture(tmp_path, linked_outer=True)
    outer_workspace = Path(
        _git(
            fixture.workspace,
            "rev-parse",
            "--path-format=absolute",
            "--git-common-dir",
        )
    ).parent
    _git(outer_workspace, "switch", "main")
    carrier = fixture.carry()
    carrier_tree = _git(fixture.workspace, "rev-parse", "HEAD^{tree}")
    state_dir = tmp_path / "committed-carrier-state"
    daemon = TodoImplementationDaemon(
        todo_path=outer_workspace.joinpath(
            *Path(daemon_module.AUTHORITY_VALIDATION_DCR_TODO_PATH).parts
        ),
        state_path=state_dir / "task-state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=outer_workspace,
        validation_cache_dir=state_dir / "validation-cache",
        merge_queue_dir=state_dir / "merge-queue",
        task_header_prefix="## DCR-",
        implementation_protected_paths=(
            daemon_module.AUTHORITY_VALIDATION_DCR_TODO_PATH,
        ),
        worktree_submodule_paths=(
            "Mcp-Plus-Plus",
            "external/ipfs_accelerate",
            "external/ipfs_datasets",
            "external/ipfs_kit",
            "swissknife",
        ),
        manual_completion_authority_task_ids=("DCR-011",),
        manual_completion_authority_epoch_id="sealed-dcr011-carrier-test",
    )
    task = PortalTask(
        task_id="DCR-011",
        title="Materialize one current multi-root forest and overlay identity",
        status="todo",
        completion="artifact",
        priority="P0",
        track="deterministic-contract-repair",
        validation=list(daemon_module.AUTHORITY_VALIDATION_DCR_RAW_COMMANDS),
    )
    revalidation_store = daemon._synchronize_manual_completion_revocation_generation(
        task_statuses={"DCR-011": "todo"},
    )
    assert revalidation_store["available"] is True, revalidation_store
    monkeypatch.setattr(
        daemon,
        "_identity_for_task",
        lambda _task: SimpleNamespace(
            canonical_task_cid=(daemon_module.AUTHORITY_VALIDATION_DCR_TASK_CID),
            canonical_task_key="dcr-011-linked-carrier-key",
            board_namespace="dcr011-linked-carrier",
            semantic_fingerprint="d" * 64,
            short_id="d" * 12,
        ),
    )
    daemon._manual_completion_authority_revalidation_task_ids = frozenset({"DCR-011"})
    daemon._manual_completion_authority_hard_blocked_task_ids = frozenset()
    monkeypatch.setattr(
        daemon,
        "_refresh_manual_completion_authority_guard",
        lambda: {"available": True, "_tasks": (task,)},
    )
    log_path = state_dir / "committed-carrier-validation.log"
    precommit = {
        "attempted": True,
        "passed": True,
        "returncode": 0,
        "manual_completion_authority_revalidation": False,
        "results": [
            {
                "validation_id": "declared:" + ("1" * 64),
                "proposal_bound": True,
            }
        ],
    }
    precommit_id = daemon._manual_completion_revalidation_evidence_id(precommit)
    daemon._trusted_manual_completion_revalidation_evidence_ids.add(precommit_id)

    result = daemon._run_committed_dcr011_authority_validation(
        workspace_path=fixture.workspace,
        task=task,
        state=PortalTaskState.load(daemon.state_path),
        log_path=log_path,
        baseline_ref=fixture.subject,
        implementation_commit=carrier,
        precommit_validation_result=precommit,
    )

    assert result["passed"] is True, result
    assert result["target_commit"] == carrier
    assert result["target_tree"] == carrier_tree
    assert precommit_id not in (
        daemon._trusted_manual_completion_revalidation_evidence_ids
    )
    assert result["manual_completion_authority_revalidation"] is True
    assert result["manual_completion_authority_force_uncached"] is True
    results = result["results"]
    assert [item["ordinal"] for item in results] == [0, 1]
    assert [item["validation_id"] for item in results] == ["", ""]
    assert [item["cache_hit"] for item in results] == [False, False]
    receipts = [item["authority_validation_isolation_receipt"] for item in results]
    assert receipts[0]["host_filesystem"] == "workspace_only_read_only"
    assert receipts[0]["git_metadata_replay"]["mode"] == "not_requested"
    assert receipts[1]["host_filesystem"] == (
        "workspace_and_identity_checked_git_metadata_read_only"
    )
    assert receipts[1]["git_metadata_replay"]["mode"] == "dcr_forest_exact"
    assert all(
        receipt["tmpfs_limit_bytes"]
        == daemon_module.AUTHORITY_VALIDATION_TMPFS_LIMIT_BYTES
        for receipt in receipts
    )
    assert all(
        receipt["authority_command_binding"]["expected_target_commit"] == carrier
        and receipt["authority_command_binding"]["expected_target_tree"] == carrier_tree
        for receipt in receipts
    )
    root = receipts[1]["git_metadata_replay"]["root_identities"][0]
    assert root["lifecycle"]["mode"] == "artifact_carried"
    final_evidence_id = daemon._manual_completion_revalidation_evidence_id(result)
    retained_bytes = (
        daemon._trusted_manual_completion_revalidation_evidence_bytes_by_id[
            final_evidence_id
        ]
    )
    assert retained_bytes == len(
        daemon_module._manual_completion_authority_evidence_bytes(result)
    )
    assert retained_bytes > 0
    assert retained_bytes <= (
        daemon_module.MAX_RETAINED_MANUAL_COMPLETION_AUTHORITY_EVIDENCE_ITEM_BYTES
    )
    assert daemon._trusted_manual_completion_revalidation_evidence_total_bytes == (
        retained_bytes
    )
    log_text = log_path.read_text(encoding="utf-8")
    assert daemon_module.AUTHORITY_VALIDATION_DCR_RAW_COMMANDS[0] in log_text
    assert "[validation passed]" in log_text

    # Persist the exact two-receipt capability through DuckDB, prove a peer
    # generic consumer cannot claim it, then run the real merge callback and
    # durable completion publication in this producer process.
    monkeypatch.setattr(
        daemon,
        "_bundle_work_order_for_task",
        lambda _task: None,
    )
    monkeypatch.setattr(
        daemon,
        "_current_completion_task_cids",
        lambda _task_ids, **_kwargs: (
            {"DCR-011": daemon_module.AUTHORITY_VALIDATION_DCR_TASK_CID},
            {},
        ),
    )
    request, _queued = daemon._enqueue_merge_candidate(
        branch_name=fixture.branch,
        implementation_commit=carrier,
        baseline_ref=fixture.subject,
        worktree_path=fixture.workspace,
        task=task,
        attempt=1,
        changed_submodule_paths=[],
        validation_result=dict(result),
    )
    queued_proof = request.metadata["validation_proof"]
    assert (
        queued_proof["manual_completion_authority_full_evidence_id"]
        == final_evidence_id
    )
    assert request.metadata["manual_completion_authority_rotation_binding_id"]
    assert daemon.merge_queue.dequeue(consumer_id="peer-generic") is None
    assert daemon.merge_queue.get(request.request_id).status == "pending"  # type: ignore[union-attr]

    try:
        train_result = daemon._consume_one_merge_candidate(
            allowed_request_ids=(request.request_id,)
        )
        handoff_summary = {
            key: train_result.get(key)  # type: ignore[union-attr]
            for key in (
                "status",
                "reason",
                "merged",
                "integrated",
                "accepted",
                "acceptance_pending",
                "completion_authoritative",
                "failure_count",
            )
        }
        if isinstance(train_result.get("merge_result"), dict):  # type: ignore[union-attr]
            handoff_summary["merge_result"] = {
                key: train_result["merge_result"].get(key)  # type: ignore[index]
                for key in (
                    "attempted",
                    "merged",
                    "already_merged",
                    "returncode",
                    "reason",
                    "completion_skipped",
                    "completion_pending_durability",
                )
            }
        assert daemon._merge_train_authority_handoff_complete(
            train_result,
            request_id=request.request_id,
        ), handoff_summary
        durable_request = daemon.merge_queue.get(request.request_id)
        assert durable_request is not None
        assert durable_request.status == "completed"
        callback_result = train_result["merge_result"]  # type: ignore[index]
        assert (
            callback_result.get("merged") is True
            or callback_result.get("already_merged") is True
        )
        assert callback_result["todo_update_result"]["completion_receipts"]
        # The raw post-merge validation result is authorized only while the
        # completion CAS/persistence call is on the stack.  The pre-merge
        # retained handoff is the sole surviving process-local capability.
        assert daemon._trusted_manual_completion_revalidation_evidence_ids == {
            final_evidence_id
        }
        assert set(daemon._trusted_manual_completion_revalidation_evidence_by_id) == {
            final_evidence_id
        }
        assert daemon._trusted_manual_completion_revalidation_evidence_total_bytes == (
            retained_bytes
        )
    finally:
        daemon._discard_retained_manual_completion_authority_evidence(final_evidence_id)
    assert daemon._trusted_manual_completion_revalidation_evidence_ids == set()
    assert daemon._trusted_manual_completion_revalidation_evidence_by_id == {}
    assert daemon._trusted_manual_completion_revalidation_evidence_bytes_by_id == {}
    assert daemon._trusted_manual_completion_revalidation_evidence_total_bytes == 0


@pytest.mark.parametrize("outcome", ("success", "failure", "exception"))
def test_post_merge_authority_token_is_ephemeral_and_journal_redacted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
) -> None:
    daemon = TodoImplementationDaemon(
        todo_path=tmp_path / "todo.md",
        state_path=tmp_path / "state.json",
        strategy_path=tmp_path / "strategy.json",
        events_path=tmp_path / "events.jsonl",
        repo_root=tmp_path,
        manual_completion_authority_task_ids=("DCR-011",),
    )
    task = PortalTask(
        task_id="DCR-011",
        title="Ephemeral post-merge authority",
        status="todo",
        completion="artifact",
        priority="P0",
        track="test",
        validation=["true"],
    )
    monkeypatch.setattr(daemon, "_load_tasks", lambda: [task])
    monkeypatch.setattr(
        daemon,
        "_identity_for_task",
        lambda _task: SimpleNamespace(
            canonical_task_cid="cid-dcr-011",
            canonical_task_key="key-dcr-011",
            board_namespace="test-board",
        ),
    )
    evidence = {
        "attempted": True,
        "passed": True,
        "returncode": 0,
        "manual_completion_authority_context_id": "context-1",
        "results": [],
    }
    expected_identity = {
        "schema": (
            "ipfs_accelerate_py.agent_supervisor." "manual-completion-validated-tree@1"
        ),
        "target_commit": "1" * 40,
        "repository_tree_id": "git-tree:" + ("2" * 40),
    }
    evidence_id = daemon._manual_completion_revalidation_evidence_id(evidence)
    structural_calls: list[dict[str, object]] = []

    def structural_check(*_args, **kwargs):
        structural_calls.append(dict(kwargs))
        assert kwargs["_pending_producer_evidence"] is True
        assert kwargs["expected_validated_tree_identity"] == expected_identity
        assert kwargs["authority_evidence"] is evidence
        assert evidence_id not in (
            daemon._trusted_manual_completion_revalidation_evidence_ids
        )
        return None

    monkeypatch.setattr(
        daemon,
        "_manual_completion_authority_rejection",
        structural_check,
    )

    def mutation(*_args, **kwargs):
        assert (
            kwargs["manual_completion_authority_expected_tree_identity"]
            == expected_identity
        )
        assert daemon._trusted_manual_completion_revalidation_evidence_ids == {
            evidence_id
        }
        assert daemon._trusted_manual_completion_revalidation_evidence_by_id == {}
        expectation = daemon._completion_callback_expectation(
            ["DCR-011"],
            manual_completion_authority_evidence=evidence,
        )
        assert expectation["manual_completion_authority_evidence_id"] == (evidence_id)
        assert "manual_completion_authority_evidence" not in expectation
        if outcome == "exception":
            raise RuntimeError("injected completion mutation exception")
        return {
            "updated": outcome == "success",
            "durable": outcome == "success",
            "reason": "updated" if outcome == "success" else "injected_failure",
        }

    monkeypatch.setattr(daemon, "_mark_tasks_completed_in_todo", mutation)
    invoke = lambda: daemon._mark_post_merge_completion_with_ephemeral_authority(
        ["DCR-011"],
        primary_task_id="DCR-011",
        completion_reason="single_task",
        expected_task_cids={"DCR-011": "cid-dcr-011"},
        expected_target_commit="1" * 40,
        completion_intent=None,
        authority_evidence=evidence,
        expected_validated_tree_identity=expected_identity,
    )
    if outcome == "exception":
        with pytest.raises(
            RuntimeError, match="injected completion mutation exception"
        ):
            invoke()
    else:
        result = invoke()
        assert result["durable"] is (outcome == "success")
    assert len(structural_calls) == 1
    assert daemon._trusted_manual_completion_revalidation_evidence_ids == set()
    assert daemon._trusted_manual_completion_revalidation_evidence_by_id == {}
    assert daemon._trusted_manual_completion_revalidation_evidence_bytes_by_id == {}
    assert daemon._trusted_manual_completion_revalidation_evidence_total_bytes == 0


def test_post_merge_structural_authority_rejects_older_commit_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = TodoImplementationDaemon(
        todo_path=tmp_path / "todo.md",
        state_path=tmp_path / "state.json",
        strategy_path=tmp_path / "strategy.json",
        events_path=tmp_path / "events.jsonl",
        repo_root=tmp_path,
        manual_completion_authority_task_ids=("TEST-001",),
    )
    task = PortalTask(
        task_id="TEST-001",
        title="Exact post-merge identity",
        status="todo",
        completion="artifact",
        priority="P0",
        track="test",
        validation=["true"],
    )
    task_cid = "cid-test-001"
    monkeypatch.setattr(
        daemon,
        "_identity_for_task",
        lambda _task: SimpleNamespace(canonical_task_cid=task_cid),
    )
    daemon._manual_completion_authority_revalidation_task_ids = frozenset({"TEST-001"})
    daemon._manual_completion_authority_hard_blocked_task_ids = frozenset()
    monkeypatch.setattr(
        daemon,
        "_refresh_manual_completion_authority_guard",
        lambda: {"available": True, "_tasks": (task,)},
    )
    monkeypatch.setattr(
        daemon, "_manual_completion_authority_policy_id", lambda: "context-1"
    )
    monkeypatch.setattr(daemon, "_authority_validation_isolation_contract", lambda: {})
    monkeypatch.setattr(
        daemon_module,
        "_authority_validation_isolation_receipt_valid",
        lambda *_args, **_kwargs: True,
    )
    old_commit = "1" * 40
    old_tree = "2" * 40
    old_identity = {
        "schema": (
            "ipfs_accelerate_py.agent_supervisor." "manual-completion-validated-tree@1"
        ),
        "target_commit": old_commit,
        "repository_tree_id": f"git-tree:{old_tree}",
    }
    plan = daemon._manual_completion_validation_plan_binding(task)
    binding = {
        "authority_profile": "generic_workspace_only@1",
        "task_id": "TEST-001",
        "canonical_task_cid": task_cid,
        "scope": "post_merge",
        "plan_id": "plan-1",
        "expected_target_commit": old_commit,
        "expected_target_tree": old_tree,
        "expected_git_common_anchor": "/git/common",
        "expected_git_dir": "/git/common/worktrees/candidate",
    }
    result_record = {
        "command": "true",
        "raw_command": "true",
        "ordinal": 0,
        "validation_id": "",
        "returncode": 0,
        "cache_hit": False,
        "cache_key": "a" * 64,
        "timed_out": False,
        "infrastructure_failure": False,
        "validation_result_digest": "b" * 64,
        "authority_validation_isolation_receipt": {
            "workspace_path": "/workspace",
            "authority_command_binding": binding,
        },
    }
    evidence = {
        "attempted": True,
        "passed": True,
        "returncode": 0,
        "manual_completion_authority_context_id": "context-1",
        "manual_completion_authority_revalidation": True,
        "manual_completion_authority_force_uncached": True,
        "manual_completion_authority_task_id": "TEST-001",
        "manual_completion_authority_task_cid": task_cid,
        "manual_completion_authority_validation_plan_id": plan["validation_plan_id"],
        "manual_completion_authority_declared_validation_commands": ["true"],
        "manual_completion_authority_executed_validation_commands": ["true"],
        "manual_completion_authority_revocation_generation": (
            daemon._manual_completion_authority_revocation_generation
        ),
        "manual_completion_authority_validated_tree_identity": old_identity,
        "manual_completion_authority_validated_tree_id": (
            daemon_module.content_identity(old_identity)
        ),
        "manual_completion_authority_validation_result_count": 1,
        "results": [result_record],
    }
    assert (
        daemon._manual_completion_authority_rejection(
            ["TEST-001"],
            authority_context_id="context-1",
            authority_evidence=evidence,
            expected_validated_tree_identity=old_identity,
            _pending_producer_evidence=True,
        )
        is None
    )
    newer_identity = {
        **old_identity,
        "target_commit": "3" * 40,
        "repository_tree_id": "git-tree:" + ("4" * 40),
    }
    rejection = daemon._manual_completion_authority_rejection(
        ["TEST-001"],
        authority_context_id="context-1",
        authority_evidence=evidence,
        expected_validated_tree_identity=newer_identity,
        _pending_producer_evidence=True,
    )
    assert rejection is not None
    assert rejection["reason"] == ("manual_completion_authority_revalidation_required")
    assert rejection["manual_completion_authority_evidence_valid"] is False
    assert daemon._trusted_manual_completion_revalidation_evidence_ids == set()


def test_protected_recovery_exception_revokes_ephemeral_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = TodoImplementationDaemon(
        todo_path=tmp_path / "todo.md",
        state_path=tmp_path / "state.json",
        strategy_path=tmp_path / "strategy.json",
        events_path=tmp_path / "events.jsonl",
        repo_root=tmp_path,
    )
    evidence = {
        "attempted": True,
        "passed": True,
        "returncode": 0,
        "manual_completion_authority_context_id": "context-1",
        "results": [],
    }
    evidence_id = daemon._manual_completion_revalidation_evidence_id(evidence)

    def fail_after_replay_authority_mint(*, _ephemeral_evidence_id_sink: list[str]):
        minted = daemon._mint_ephemeral_manual_completion_authority_evidence(evidence)
        _ephemeral_evidence_id_sink.append(minted)
        assert minted == evidence_id
        assert daemon._trusted_manual_completion_revalidation_evidence_ids == {
            evidence_id
        }
        raise RuntimeError("injected protected callback replay exception")

    monkeypatch.setattr(
        daemon,
        "_recover_protected_checkout_mutation_implementation",
        fail_after_replay_authority_mint,
    )
    with pytest.raises(RuntimeError, match="protected callback replay exception"):
        daemon._recover_protected_checkout_mutation()
    assert daemon._trusted_manual_completion_revalidation_evidence_ids == set()
    assert daemon._trusted_manual_completion_revalidation_evidence_by_id == {}
    assert daemon._trusted_manual_completion_revalidation_evidence_bytes_by_id == {}
    assert daemon._trusted_manual_completion_revalidation_evidence_total_bytes == 0


@pytest.mark.parametrize("fresh_passes", (True, False))
def test_reconciled_candidate_enqueues_only_fresh_committed_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fresh_passes: bool,
) -> None:
    fixture = _real_forest_cli_fixture(tmp_path, linked_outer=True)
    carrier = fixture.carry()
    state_dir = tmp_path / "reconciliation-state"
    daemon = TodoImplementationDaemon(
        todo_path=fixture.workspace.joinpath(
            *Path(daemon_module.AUTHORITY_VALIDATION_DCR_TODO_PATH).parts
        ),
        state_path=state_dir / "task-state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=fixture.workspace,
        worktree_root=tmp_path / "unused-worktrees",
        merge_queue_dir=state_dir / "merge-queue",
        validation_cache_dir=state_dir / "validation-cache",
        manual_completion_authority_task_ids=("DCR-011",),
        manual_completion_authority_epoch_id="reconciled-carrier-test",
    )
    task = PortalTask(
        task_id="DCR-011",
        title="Materialize one current multi-root forest and overlay identity",
        status="todo",
        completion="artifact",
        priority="P0",
        track="deterministic-contract-repair",
        outputs=[daemon_module.AUTHORITY_VALIDATION_DCR_FOREST_ARTIFACT],
        validation=list(daemon_module.AUTHORITY_VALIDATION_DCR_RAW_COMMANDS),
    )
    identity = SimpleNamespace(
        canonical_task_cid=daemon_module.AUTHORITY_VALIDATION_DCR_TASK_CID,
        canonical_task_key="dcr-011-sealed-reconciliation",
        board_namespace="dcr011-test",
        semantic_fingerprint="d" * 64,
        short_id="d" * 12,
    )
    monkeypatch.setattr(daemon, "_identity_for_task", lambda _task: identity)
    monkeypatch.setattr(
        daemon,
        "_prepare_worktree_for_validation",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        daemon,
        "_require_implementation_protected_snapshot",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(
        daemon,
        "_implementation_protected_path_violation",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(
        daemon,
        "_finalize_implementation_protected_path_fence",
        lambda **_kwargs: {},
    )
    proposal = SimpleNamespace(
        accepted=True,
        proposal=SimpleNamespace(proposal_id="reconciled-proposal"),
    )
    monkeypatch.setattr(
        daemon,
        "_validate_implementation_patch",
        lambda *_args, **_kwargs: proposal,
    )
    calls: list[str] = []

    def proposal_validation(*_args, **kwargs):
        assert kwargs["proposal_validation"] is proposal
        calls.append("proposal")
        return {
            "attempted": True,
            "passed": True,
            "returncode": 0,
            "producer": "proposal-bound",
            "results": [{"validation_id": "declared:" + ("2" * 64)}],
        }

    monkeypatch.setattr(daemon, "_run_validation_commands", proposal_validation)
    monkeypatch.setattr(
        daemon,
        "_apply_implementation_failure_review",
        lambda **kwargs: dict(kwargs["validation_result"]),
    )
    monkeypatch.setattr(
        daemon,
        "_restore_and_verify_post_validation_candidate",
        lambda *_args, **kwargs: dict(kwargs["validation_result"]),
    )
    monkeypatch.setattr(
        daemon,
        "_validated_existing_worktree_commit",
        lambda *_args, **_kwargs: {
            "committed": True,
            "commit": carrier,
            "submodule_results": [],
        },
    )
    monkeypatch.setattr(
        daemon,
        "_current_completion_task_cids",
        lambda *_args, **_kwargs: (
            {"DCR-011": daemon_module.AUTHORITY_VALIDATION_DCR_TASK_CID},
            "",
        ),
    )

    def fresh_committed_validation(**kwargs):
        assert kwargs["implementation_commit"] == carrier
        assert kwargs["baseline_ref"] == fixture.subject
        assert kwargs["precommit_validation_result"]["producer"] == ("proposal-bound")
        calls.append("fresh")
        return {
            "attempted": True,
            "passed": fresh_passes,
            "returncode": 0 if fresh_passes else 1,
            "reason": (
                "fresh_committed_authority_passed"
                if fresh_passes
                else "noncanonical_committed_candidate"
            ),
            "producer": "fresh-empty-id",
            "results": (
                [
                    {"ordinal": 0, "validation_id": ""},
                    {"ordinal": 1, "validation_id": ""},
                ]
                if fresh_passes
                else []
            ),
        }

    monkeypatch.setattr(
        daemon,
        "_run_committed_dcr011_authority_validation",
        fresh_committed_validation,
    )

    def enqueue(**kwargs):
        assert kwargs["implementation_commit"] == carrier
        assert kwargs["validation_result"]["producer"] == "fresh-empty-id"
        calls.append("enqueue")
        return {"merged": False, "queued": True, "reason": "queued"}

    monkeypatch.setattr(daemon, "_enqueue_validated_worktree", enqueue)
    monkeypatch.setattr(
        daemon,
        "_restore_ephemeral_worktree_paths_for_commit",
        lambda *_args, **_kwargs: None,
    )

    result = daemon.reconcile_validated_worktree_candidate(
        worktree_path=fixture.workspace,
        branch_name=fixture.branch,
        task=task,
        baseline_ref=fixture.subject,
        candidate_commit=carrier,
        recovery_key="dcr011-canonical-carrier",
    )

    if fresh_passes:
        assert calls == ["proposal", "fresh", "enqueue"]
        assert result["merge_result"]["queued"] is True
        assert result["validation_result"]["producer"] == "fresh-empty-id"
    else:
        assert calls == ["proposal", "fresh"]
        assert result["merge_result"].get("queued") is not True
        assert result["validation_result"]["reason"] == (
            "noncanonical_committed_candidate"
        )


def test_dcr011_cid_slice_rejects_stale_same_bundle_dcr012_expansion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    todo_path = tmp_path / "todo.md"
    todo_path.write_text("# slice fixture\n", encoding="utf-8")
    daemon = TodoImplementationDaemon(
        todo_path=todo_path,
        state_path=tmp_path / "state.json",
        strategy_path=tmp_path / "strategy.json",
        events_path=tmp_path / "events.jsonl",
        repo_root=tmp_path,
        execution_slice_task_cids=("cid-dcr-011",),
    )
    primary = PortalTask(
        task_id="DCR-011",
        title="Forest carrier",
        status="todo",
        completion="artifact",
        priority="P0",
        track="deterministic-contract-repair",
    )
    sibling = PortalTask(
        task_id="DCR-012",
        title="Historical analyzer",
        status="todo",
        completion="artifact",
        priority="P0",
        track="deterministic-contract-repair",
    )
    task_cids = {
        "DCR-011": "cid-dcr-011",
        "DCR-012": "cid-dcr-012",
    }
    monkeypatch.setattr(
        daemon,
        "_identity_for_task",
        lambda task: SimpleNamespace(canonical_task_cid=task_cids[task.task_id]),
    )
    monkeypatch.setattr(daemon, "_load_tasks", lambda: [primary, sibling])
    monkeypatch.setattr(
        daemon,
        "_task_metadata_value",
        lambda _task, *keys: ("dcr/same-bundle" if "bundle" in keys else ""),
    )
    monkeypatch.setattr(
        daemon,
        "_load_todo_vector_context",
        lambda _task: {
            "aggregate_primary": True,
            "covered_packet_task_ids": ["DCR-012"],
            "record": {
                "bundle_key": "dcr/same-bundle",
                "goal_packet_key": "dcr/packet",
            },
            "covered_packet_records": {"DCR-012": {"bundle_key": "dcr/same-bundle"}},
            "index_path": tmp_path / "stale-todo-vector.json",
        },
    )

    assert daemon._task_matches_execution_slice_identity(primary) is True
    assert daemon._task_matches_execution_slice_identity(sibling) is False
    assert daemon._task_in_execution_slice(primary) is False
    with pytest.raises(
        daemon_module.ImplementationRetryDeferred,
        match="execution_slice_bundle_expansion_forbidden",
    ):
        daemon._bundle_work_order_for_task(primary)


def test_committed_postflight_exception_leaves_no_trusted_evidence_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "candidate"
    _init_repository(workspace, label="baseline")
    baseline = _git(workspace, "rev-parse", "HEAD")
    (workspace / "carrier.txt").write_text("carrier\n", encoding="utf-8")
    _git(workspace, "add", "carrier.txt")
    _git(workspace, "commit", "-qm", "generic committed candidate")
    implementation = _git(workspace, "rev-parse", "HEAD")
    tree = _git(workspace, "rev-parse", "HEAD^{tree}")
    state_dir = tmp_path / "state"
    daemon = TodoImplementationDaemon(
        todo_path=tmp_path / "todo.md",
        state_path=state_dir / "task-state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=workspace,
        manual_completion_authority_task_ids=("TEST-001",),
    )
    task = PortalTask(
        task_id="TEST-001",
        title="Generic authority candidate",
        status="todo",
        completion="artifact",
        priority="P0",
        track="test",
        validation=["true"],
    )
    monkeypatch.setattr(
        daemon,
        "_identity_for_task",
        lambda _task: SimpleNamespace(canonical_task_cid="cid-test-001"),
    )
    daemon._manual_completion_authority_revalidation_task_ids = frozenset({"TEST-001"})
    proof_budget = {"output_overflow": False}
    proof_calls = ["before", "after"]

    def checkout_proof(*_args, **_kwargs):
        phase = proof_calls.pop(0)
        return {
            "passed": True,
            "phase": phase,
            "observation_id": "observation-same",
        }

    monkeypatch.setattr(
        daemon,
        "_exact_post_merge_tracked_checkout_proof",
        checkout_proof,
    )
    monkeypatch.setattr(
        daemon_module._TrackedCheckoutProofBudget,
        "receipt",
        lambda _budget: dict(proof_budget),
    )

    def validation(*_args, **kwargs):
        assert kwargs["_publish_authority_evidence"] is False
        identity = {
            "schema": (
                "ipfs_accelerate_py.agent_supervisor."
                "manual-completion-validated-tree@1"
            ),
            "target_commit": implementation,
            "repository_tree_id": f"git-tree:{tree}",
        }
        result = {
            "attempted": True,
            "passed": True,
            "returncode": 0,
            "manual_completion_authority_context_id": "test-context",
            "manual_completion_authority_validated_tree_identity": identity,
            "results": [
                {
                    "ordinal": 0,
                    "authority_validation_command_binding": {
                        "expected_target_commit": implementation,
                        "expected_target_tree": tree,
                        "authority_profile": "generic_workspace_only@1",
                    },
                }
            ],
        }
        # Simulate a future producer regression; the helper must demote this
        # raw token before any postflight work.
        daemon._trusted_manual_completion_revalidation_evidence_ids.add(
            daemon._manual_completion_revalidation_evidence_id(result)
        )
        return result

    monkeypatch.setattr(daemon, "_run_validation_commands", validation)
    monkeypatch.setattr(
        daemon_module,
        "_committed_candidate_authority_attestation_valid",
        lambda _evidence: True,
    )
    monkeypatch.setattr(
        daemon,
        "_manual_completion_authority_rejection",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("injected postflight failure")
        ),
    )

    result = daemon._run_committed_dcr011_authority_validation(
        workspace_path=workspace,
        task=task,
        state=PortalTaskState(),
        log_path=state_dir / "validation.log",
        baseline_ref=baseline,
        implementation_commit=implementation,
        precommit_validation_result={"passed": True},
    )

    assert result["passed"] is False
    assert (
        result["committed_candidate_authority_validation"]["authority_rejection"][
            "reason"
        ]
        == "committed_candidate_authority_postflight_exception"
    )
    assert daemon._trusted_manual_completion_revalidation_evidence_ids == set()
    assert daemon._trusted_manual_completion_revalidation_evidence_by_id == {}


def test_retained_committed_evidence_is_canonically_byte_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = TodoImplementationDaemon(
        todo_path=tmp_path / "todo.md",
        state_path=tmp_path / "state.json",
        strategy_path=tmp_path / "strategy.json",
        events_path=tmp_path / "events.jsonl",
        repo_root=tmp_path,
    )
    first = {"record": "a" * 80}
    second = {"record": "b" * 80}
    first_size = len(daemon_module._manual_completion_authority_evidence_bytes(first))
    second_size = len(daemon_module._manual_completion_authority_evidence_bytes(second))
    monkeypatch.setattr(
        daemon_module,
        "MAX_RETAINED_MANUAL_COMPLETION_AUTHORITY_EVIDENCE_ITEM_BYTES",
        max(first_size, second_size) + 8,
    )
    monkeypatch.setattr(
        daemon_module,
        "MAX_RETAINED_MANUAL_COMPLETION_AUTHORITY_EVIDENCE_TOTAL_BYTES",
        max(first_size, second_size) + 8,
    )

    assert (
        daemon._retain_manual_completion_authority_evidence("first", first)
        == first_size
    )
    assert (
        daemon._retain_manual_completion_authority_evidence("second", second)
        == second_size
    )
    assert set(daemon._trusted_manual_completion_revalidation_evidence_by_id) == {
        "second"
    }
    assert daemon._trusted_manual_completion_revalidation_evidence_ids == {"second"}
    assert daemon._trusted_manual_completion_revalidation_evidence_total_bytes == (
        second_size
    )

    with pytest.raises(ValueError, match="per-record byte cap"):
        daemon._retain_manual_completion_authority_evidence(
            "oversized", {"record": "z" * 200}
        )
    assert set(daemon._trusted_manual_completion_revalidation_evidence_by_id) == {
        "second"
    }


def test_authority_handoff_requires_terminal_acceptance_and_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = TodoImplementationDaemon(
        todo_path=tmp_path / "todo.md",
        state_path=tmp_path / "state.json",
        strategy_path=tmp_path / "strategy.json",
        events_path=tmp_path / "events.jsonl",
        repo_root=tmp_path,
    )
    monkeypatch.setattr(
        daemon,
        "_todo_completion_is_durable",
        lambda _result: True,
    )
    complete = {
        "request_id": "request-exact",
        "status": "merged",
        "integrated": True,
        "accepted": True,
        "acceptance_pending": False,
        "merged": True,
        "merge_result": {
            "merged": True,
            "todo_update_result": {
                "completion_receipts": [
                    {
                        "task_id": "DCR-011",
                        "canonical_task_cid": "cid-dcr-011",
                    }
                ]
            },
        },
    }
    assert daemon._merge_train_authority_handoff_complete(
        complete,
        request_id="request-exact",
    )
    integrated_pending = copy.deepcopy(complete)
    integrated_pending.update(
        {
            "status": "integrated_pending_validation",
            "accepted": False,
            "acceptance_pending": True,
            "completion_authoritative": False,
        }
    )
    assert not daemon._merge_train_authority_handoff_complete(
        integrated_pending,
        request_id="request-exact",
    )
    missing_receipt = copy.deepcopy(complete)
    missing_receipt["merge_result"]["todo_update_result"]["completion_receipts"] = []
    assert not daemon._merge_train_authority_handoff_complete(
        missing_receipt,
        request_id="request-exact",
    )


def test_rotation_binding_rejects_deleted_or_edited_queue_metadata() -> None:
    evidence = {
        "manual_completion_authority_declared_validation_commands": [
            "pytest exact",
            "forest exact",
        ],
        "manual_completion_authority_validated_tree_identity": {
            "target_commit": "2" * 40,
            "repository_tree_id": "git-tree:" + "3" * 40,
        },
        "committed_candidate_authority_validation": {
            "parent_commit": "1" * 40,
            "subject": daemon_module.AUTHORITY_VALIDATION_DCR_CARRIER_SUBJECT,
            "changed_paths": [daemon_module.AUTHORITY_VALIDATION_DCR_FOREST_ARTIFACT],
            "lifecycle_mode": "artifact_carried",
        },
    }
    arguments = {
        "branch_name": "candidate/dcr-011",
        "baseline_ref": "1" * 40,
        "implementation_commit": "2" * 40,
        "candidate_tree": "3" * 40,
        "target_repository_id": "repository-exact",
        "target_branch": "main",
        "task_id": "DCR-011",
        "canonical_task_cid": daemon_module.AUTHORITY_VALIDATION_DCR_TASK_CID,
        "canonical_task_key": "task-key-exact",
        "completion_task_cids": {
            "DCR-011": daemon_module.AUTHORITY_VALIDATION_DCR_TASK_CID
        },
        "todo_path": "/workspace/todo.md",
        "evidence": evidence,
    }
    binding_id = daemon_module._manual_completion_authority_rotation_binding_id(
        **arguments
    )
    assert daemon_module._manual_completion_authority_rotation_binding_valid(
        queued_binding_id=binding_id,
        queued_full_evidence_id="full-evidence-exact",
        **arguments,
    )
    assert not daemon_module._manual_completion_authority_rotation_binding_valid(
        queued_binding_id="",
        queued_full_evidence_id="full-evidence-exact",
        **arguments,
    )
    assert not daemon_module._manual_completion_authority_rotation_binding_valid(
        queued_binding_id=binding_id,
        queued_full_evidence_id="full-evidence-exact",
        **{**arguments, "baseline_ref": "4" * 40},
    )


def test_authority_handoff_retries_exact_request_and_revokes_capability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = TodoImplementationDaemon(
        todo_path=tmp_path / "todo.md",
        state_path=tmp_path / "state.json",
        strategy_path=tmp_path / "strategy.json",
        events_path=tmp_path / "events.jsonl",
        repo_root=tmp_path,
    )
    task = PortalTask(
        task_id="TEST-001",
        title="Bounded authority handoff",
        status="todo",
        completion="artifact",
        priority="P0",
        track="test",
        validation=["true"],
    )
    evidence = {"committed_candidate_authority_validation": {"passed": True}}
    evidence_id = daemon._manual_completion_revalidation_evidence_id(evidence)
    daemon._retain_manual_completion_authority_evidence(
        evidence_id,
        evidence,
    )
    request = SimpleNamespace(request_id="request-exact")
    monkeypatch.setattr(
        daemon,
        "_reject_protected_merge_candidate",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(daemon, "_mark_active_phase", lambda *_a, **_k: None)
    monkeypatch.setattr(
        daemon,
        "_release_pooled_worktree_lease",
        lambda *_a, **_k: {"released": False, "attempted": False},
    )
    monkeypatch.setattr(
        daemon,
        "_enqueue_merge_candidate",
        lambda **_kwargs: (request, {"queued": True}),
    )
    monkeypatch.setattr(
        daemon.merge_queue,
        "get",
        lambda _request_id: SimpleNamespace(status="pending"),
    )
    monkeypatch.setattr(daemon_module.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        daemon,
        "_todo_completion_is_durable",
        lambda _result: True,
    )
    terminal = {
        "request_id": "request-exact",
        "status": "merged",
        "integrated": True,
        "accepted": True,
        "acceptance_pending": False,
        "merge_result": {
            "merged": True,
            "todo_update_result": {
                "completion_receipts": [
                    {
                        "task_id": "TEST-001",
                        "canonical_task_cid": "cid-test-001",
                    }
                ]
            },
        },
    }
    calls: list[tuple[str, ...]] = []

    def consume(*, allowed_request_ids=None):
        calls.append(tuple(allowed_request_ids or ()))
        return terminal if len(calls) == 3 else None

    monkeypatch.setattr(daemon, "_consume_one_merge_candidate", consume)
    result = daemon._enqueue_validated_worktree(
        state=PortalTaskState(),
        task=task,
        attempt=1,
        branch_name="candidate/test-001",
        baseline_ref="1" * 40,
        worktree_path=tmp_path,
        implementation_commit="2" * 40,
        commit_result={},
        validation_result=evidence,
    )

    assert calls == [
        ("request-exact",),
        ("request-exact",),
        ("request-exact",),
    ]
    assert result["merged"] is True
    assert evidence_id not in (
        daemon._trusted_manual_completion_revalidation_evidence_ids
    )
    assert daemon._trusted_manual_completion_revalidation_evidence_by_id == {}


def test_projection_creator_budget_failure_removes_exact_initial_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projection_parent = tmp_path / "projection-parent"
    projection_parent.mkdir(mode=0o700)
    monkeypatch.setattr(
        daemon_module,
        "_authority_git_projection_parent",
        lambda: projection_parent,
    )

    class FailingBudget:
        @staticmethod
        def check(_detail: str) -> None:
            return None

        @staticmethod
        def add_sealed(_value: int, detail: str) -> None:
            raise AuthorityGitReplayError(
                "authority_validation_git_snapshot_byte_limit", detail
            )

    with pytest.raises(AuthorityGitReplayError) as raised:
        daemon_module._authority_git_create_projection(
            tmp_path,
            budget=FailingBudget(),
        )

    assert raised.value.reason == ("authority_validation_git_snapshot_byte_limit")
    assert list(projection_parent.iterdir()) == []


def test_projection_creator_first_lstat_failure_removes_only_empty_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projection_parent = tmp_path / "projection-parent"
    projection_parent.mkdir(mode=0o700)
    monkeypatch.setattr(
        daemon_module,
        "_authority_git_projection_parent",
        lambda: projection_parent,
    )
    original_lstat = Path.lstat

    def fail_first_lstat(path: Path):
        if path.parent == projection_parent and path.name.startswith(
            daemon_module.AUTHORITY_VALIDATION_GIT_PROJECTION_PREFIX
        ):
            raise OSError("injected first lstat failure")
        return original_lstat(path)

    monkeypatch.setattr(Path, "lstat", fail_first_lstat)
    with pytest.raises(OSError, match="injected first lstat failure"):
        daemon_module._authority_git_create_projection(
            tmp_path,
            budget=daemon_module._AuthorityGitSetupBudget.begin(),
        )
    assert list(projection_parent.iterdir()) == []


def test_projection_creator_first_lstat_failure_never_deletes_injected_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projection_parent = tmp_path / "projection-parent"
    projection_parent.mkdir(mode=0o700)
    monkeypatch.setattr(
        daemon_module,
        "_authority_git_projection_parent",
        lambda: projection_parent,
    )
    original_lstat = Path.lstat

    def inject_before_failure(path: Path):
        if path.parent == projection_parent and path.name.startswith(
            daemon_module.AUTHORITY_VALIDATION_GIT_PROJECTION_PREFIX
        ):
            (path / "unreviewed-entry").write_text(
                "do not delete\n",
                encoding="utf-8",
            )
            raise OSError("injected first lstat failure")
        return original_lstat(path)

    monkeypatch.setattr(Path, "lstat", inject_before_failure)
    with pytest.raises(AuthorityGitReplayError) as raised:
        daemon_module._authority_git_create_projection(
            tmp_path,
            budget=daemon_module._AuthorityGitSetupBudget.begin(),
        )
    assert raised.value.reason == (
        "authority_validation_git_projection_creator_cleanup_failed"
    )
    [residue] = list(projection_parent.iterdir())
    assert (residue / "unreviewed-entry").read_text(encoding="utf-8") == (
        "do not delete\n"
    )


def test_finalize_publish_updates_guard_before_later_budget_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projection_parent = tmp_path / "projection-parent"
    projection_parent.mkdir(mode=0o700)
    monkeypatch.setattr(
        daemon_module,
        "_authority_git_projection_parent",
        lambda: projection_parent,
    )
    projection, manifest, descriptor = daemon_module._authority_git_create_projection(
        tmp_path,
        budget=daemon_module._AuthorityGitSetupBudget.begin(),
    )
    guard = daemon_module._AuthorityValidationCleanupGuard()
    guard.register_projection(projection, manifest, descriptor=descriptor)
    initial_manifest_id = manifest["manifest_id"]
    metadata = projection / "metadata"
    metadata.mkdir(mode=0o700)
    (metadata / "HEAD").write_text("0" * 40 + "\n", encoding="ascii")
    os.chmod(metadata / "HEAD", 0o400)
    os.chmod(metadata, 0o500)

    class FinalizeBudget:
        @staticmethod
        def check(_detail: str) -> None:
            return None

        @staticmethod
        def add_sealed(_value: int, detail: str) -> None:
            raise AuthorityGitReplayError(
                "authority_validation_git_snapshot_byte_limit", detail
            )

    with pytest.raises(AuthorityGitReplayError) as raised:
        daemon_module._authority_git_finalize_projection(
            projection,
            descriptor,
            manifest,
            budget=FinalizeBudget(),
            publish_callback=lambda published: guard.register_projection(
                projection,
                published,
                descriptor=descriptor,
            ),
        )

    assert raised.value.reason == ("authority_validation_git_snapshot_byte_limit")
    assert guard.projection_manifest_id != initial_manifest_id
    assert guard.cleanup_projection_now() is True
    assert not projection.exists()
