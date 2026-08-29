from __future__ import annotations

import hashlib
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
    AgentSupervisorNativeDependencyPin,
    parse_agent_supervisor_native_dependency_pin,
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
_SYNTHETIC_HTTPFS_EXTENSION = b"synthetic-httpfs-extension-v1\x00\x01"
_SYNTHETIC_HTTPFS_INFO = b'{"extension":"httpfs","synthetic":true}\n'
_ProjectionFixture = tuple[
    extension_projection.ConfiguredBoardExtensionPin,
    extension_projection.ConfiguredBoardExtensionSetPin,
    Path,
]


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
) -> Iterator[
    tuple[
        extension_projection.ConfiguredBoardExtensionPin,
        extension_projection.ConfiguredBoardExtensionSetPin,
        Path,
    ]
]:
    sources = tmp_path / "quack-extension-sources"
    sources.mkdir()
    source_paths: dict[str, tuple[Path, Path]] = {}
    pins: dict[str, extension_projection.ConfiguredBoardExtensionPin] = {}
    for name, payload, metadata in (
        ("httpfs", _SYNTHETIC_HTTPFS_EXTENSION, _SYNTHETIC_HTTPFS_INFO),
        ("quack", _SYNTHETIC_QUACK_EXTENSION, _SYNTHETIC_QUACK_INFO),
    ):
        extension = sources / f"{name}.duckdb_extension"
        info = sources / f"{name}.duckdb_extension.info"
        extension.write_bytes(payload)
        info.write_bytes(metadata)
        source_paths[name] = (extension, info)
        pins[name] = extension_projection.inspect_configured_board_extension_sources(
            extension,
            info,
            name=name,
            engine_version="v1.5.5",
            platform="linux_amd64",
        )
    set_pin = extension_projection.build_configured_board_extension_set_pin(
        pins,
        versions={"httpfs": "test-httpfs-v1", "quack": "test-quack-v1"},
    )
    projection_parent = tmp_path / "private-extension-projection"
    projection_parent.mkdir(mode=0o700)
    home = extension_projection.project_configured_board_extension_set_home(
        pins,
        sources=source_paths,
        parent=projection_parent,
    )
    extension_directory = home / ".duckdb/extensions"
    monkeypatch.setenv(
        extension_projection.CONFIGURED_BOARD_EXTENSION_DIRECTORY_ENV,
        str(extension_directory.resolve(strict=True)),
    )
    monkeypatch.setenv(
        extension_projection.CONFIGURED_BOARD_EXTENSION_SET_PIN_ENV,
        set_pin.to_json(),
    )
    try:
        yield pins["quack"], set_pin, home
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


def _write_canonical_json(path: Path, payload: object) -> bytes:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    path.write_bytes(raw)
    return raw


def _commit_controls(root: Path, message: str) -> None:
    _git(root, "add", ".")
    _git(
        root,
        "-c",
        "user.name=Capsule Test",
        "-c",
        "user.email=capsule@example.invalid",
        "commit",
        "-m",
        message,
    )


def _native_pin() -> AgentSupervisorNativeDependencyPin:
    repository_root = Path(__file__).resolve().parents[2]
    seal = json.loads(
        (
            repository_root
            / "config/semantic_addressed_world_model_dependencies.seal.json"
        ).read_text(encoding="utf-8")
    )
    return parse_agent_supervisor_native_dependency_pin(
        seal["configured_board_native_dependency"]["pin"]
    )


def _native_authorization(
    pin: AgentSupervisorNativeDependencyPin,
) -> dict[str, object]:
    body: dict[str, object] = {
        "schema": (
            "semantic-addressed-world-model/"
            "native-dependency-launch-authorization@1"
        ),
        "board_namespace": "test-board-v1",
        "plan_revision": "TEST-PLAN-R2",
        "status": "accepted",
        "scope": "configured-board-live-control-plane",
        "dependency_id": pin.dependency_id,
        "payload_sha256": pin.payload_sha256,
        "python_executable_sha256": pin.python_executable_sha256,
        "authority_basis": (
            "operator-owned protected control inside the accepted immutable "
            "source capsule"
        ),
        "inspection_is_authority": False,
        "authorization_may_claim_task_completion": False,
    }
    encoded = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    body["authorization_id"] = "sha256:" + hashlib.sha256(encoded).hexdigest()
    return body


def _seed(
    tmp_path: Path,
    extension_set_pin: extension_projection.ConfiguredBoardExtensionSetPin,
) -> tuple[Path, tuple[str, ...]]:
    root = tmp_path / "repository"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "branch", "-M", "main")
    native_pin = _native_pin()
    authorization = _native_authorization(native_pin)
    authorization_raw = json.dumps(
        authorization,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    pins = extension_set_pin.pins
    versions = extension_set_pin.versions
    quack_pin = pins["quack"]
    httpfs_pin = pins["httpfs"]
    quack_source = "/accepted/quack.duckdb_extension"
    quack_info = "/accepted/quack.duckdb_extension.info"
    httpfs_source = "/accepted/httpfs.duckdb_extension"
    httpfs_info = "/accepted/httpfs.duckdb_extension.info"
    owner_pins = {
        "pinned_httpfs_extension": {
            "path": httpfs_source,
            "sha256": httpfs_pin.payload_sha256.removeprefix("sha256:"),
            "size": httpfs_pin.payload_size,
            "info_path": httpfs_info,
            "info_sha256": httpfs_pin.info_sha256.removeprefix("sha256:"),
            "info_size": httpfs_pin.info_size,
            "version": versions["httpfs"],
            "network_install_allowed": False,
            "unsigned_extension_allowed": False,
        },
        "pinned_extension": {
            "path": quack_source,
            "sha256": quack_pin.payload_sha256.removeprefix("sha256:"),
            "size": quack_pin.payload_size,
            "info_path": quack_info,
            "info_sha256": quack_pin.info_sha256.removeprefix("sha256:"),
            "info_size": quack_pin.info_size,
            "version": versions["quack"],
            "network_install_allowed": False,
            "unsigned_extension_allowed": False,
        },
    }
    dependency_seal_path = "config/dependencies.seal.json"
    authorization_path = "config/native.authorization.json"
    paths = (
        "config/scheduler.json",
        dependency_seal_path,
        authorization_path,
        "docs/plan.md",
        "scripts/validate.py",
    )
    controls: dict[str, object] = {
        "config/scheduler.json": {
            "board_namespace": "test-board-v1",
            "plan_revision": "TEST-PLAN-R2",
            "task_prefix": "TEST-",
            "merge_target_branch": "main",
            "max_lanes": 2,
            "strict_task_sharding": True,
            "database_program": {
                "authority_mode": "quack",
                "task_source_kind": "duckdb",
                "schema_revision": "datasets-authoritative-operational-v1",
                "failover_policy": "fail_closed",
                "store_id": "data/control.duckdb",
                "store_generation": "2",
                "endpoint_secret_handle": "env://TEST_QUACK_TOKEN",
                "quack_endpoint": "quack:127.0.0.1:29992",
            },
            "dependency_seal_path": dependency_seal_path,
            "quack_owner": owner_pins,
        },
        dependency_seal_path: {
            "schema": "semantic-addressed-world-model/dependency-seal@1",
            "board_namespace": "test-board-v1",
            "plan_revision": "TEST-PLAN-R2",
            "status": "sealed",
            "configured_board_native_dependency": {
                "schema": (
                    "semantic-addressed-world-model/"
                    "configured-board-native-dependency@1"
                ),
                "source_path": "/accepted/_duckdb.so",
                "acceptance": {
                    "schema": (
                        "semantic-addressed-world-model/"
                        "native-dependency-authorization-reference@1"
                    ),
                    "path": authorization_path,
                    "sha256": "sha256:"
                    + hashlib.sha256(authorization_raw).hexdigest(),
                    "size": len(authorization_raw),
                    "authorization_id": authorization["authorization_id"],
                },
                "pin": native_pin.as_dict(),
                "sealed_memfd_required": True,
                "ambient_site_import_allowed": False,
                "ambient_loader_environment_allowed": False,
            },
            "configured_board_quack_projection": {
                "schema": (
                    "semantic-addressed-world-model/"
                    "configured-board-quack-projection@1"
                ),
                "source_path": quack_source,
                "info_path": quack_info,
                "pin": quack_pin.as_dict(),
                "load_policy": "local_load_only",
                "network_install_allowed": False,
                "unsigned_extension_allowed": False,
            },
            "httpfs_extension_pin": {
                **owner_pins["pinned_httpfs_extension"],
            },
        },
        authorization_path: authorization,
    }
    for relative in paths:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if relative in controls:
            target.write_text(
                json.dumps(
                    controls[relative],
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                ),
                encoding="utf-8",
            )
        else:
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


def _seed_handoff(
    tmp_path: Path,
    extension_set_pin: extension_projection.ConfiguredBoardExtensionSetPin,
) -> tuple[Path, tuple[str, ...]]:
    root, paths = _seed(tmp_path, extension_set_pin)
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
    extension_set_pin: extension_projection.ConfiguredBoardExtensionSetPin,
) -> capsule.ConfiguredBoardLiveCapsuleAdmission:
    native_pin = _native_pin()
    authorization = _native_authorization(native_pin)
    config_raw = (root / "config/scheduler.json").read_bytes()
    return capsule.build_configured_board_live_capsule_admission(
        repo_root=root,
        board_namespace="test-board-v1",
        plan_revision="TEST-PLAN-R2",
        task_prefix="TEST-",
        config_path="config/scheduler.json",
        configuration_root=cid_for_dag_json(
            {"bytes_sha256": hashlib.sha256(config_raw).hexdigest()}
        ),
        control_paths=paths,
        control_plane_pin=_pin(root),
        native_authorization_id=str(authorization["authorization_id"]),
        native_dependency_id=native_pin.dependency_id,
        native_python_executable_sha256=native_pin.python_executable_sha256,
        quack_extension_projection=quack_projection_pin,
        extension_set_pin=extension_set_pin,
        database_authority=_authority(),
        max_lanes=2,
        strict_task_sharding=True,
    )


def _land_test_merge(root: Path) -> tuple[str, str, str]:
    baseline = _git(root, "rev-parse", "HEAD")
    _git(root, "checkout", "-q", "-b", "implementation/test-source")
    output = root / "src/accepted.py"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("ACCEPTED = True\n", encoding="utf-8")
    _git(root, "add", output.relative_to(root).as_posix())
    _git(
        root,
        "-c",
        "user.name=Capsule Test",
        "-c",
        "user.email=capsule@example.invalid",
        "commit",
        "-q",
        "-m",
        "implementation",
    )
    implementation = _git(root, "rev-parse", "HEAD")
    _git(root, "checkout", "-q", "main")
    _git(
        root,
        "-c",
        "user.name=Capsule Test",
        "-c",
        "user.email=capsule@example.invalid",
        "merge",
        "--no-ff",
        "-q",
        "-m",
        "accepted implementation",
        implementation,
    )
    return baseline, implementation, _git(root, "rev-parse", "HEAD")


def _source_transition_authority(
    root: Path,
    admission: capsule.ConfiguredBoardLiveCapsuleAdmission,
    *,
    baseline: str,
    implementation: str,
    merge_commit: str,
) -> dict[str, object]:
    task_cid = "sha256:" + "7" * 64
    attempt_id = "attempt:test-source:1"
    implementation_tree = _git(root, "rev-parse", f"{implementation}^{{tree}}")
    merge_tree = _git(root, "rev-parse", f"{merge_commit}^{{tree}}")
    diff = subprocess.run(
        [
            "git",
            "diff-tree",
            "--no-commit-id",
            "--name-status",
            "-r",
            "-z",
            baseline,
            merge_commit,
        ],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    proof = {
        "passed": True,
        "implementation_commit": implementation,
        "integration_commit": merge_commit,
        "integration_ref": merge_commit,
        "target_branch": "main",
    }
    invariant = {"passed": True, "repository_ref": merge_commit}
    database_attempt_binding: dict[str, object] = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "database-portal-attempt-binding@1"
        ),
        "interface": "DatabasePortalExecutionBridge@1",
        "attempt_id": attempt_id,
        "claim_id": "claim:test-source:1",
        "task_cid": task_cid,
        "task_alias": "TEST-001",
        "goal_cid": "goal:test-source",
        "plan_cid": "plan:test-source",
        "task_revision": 1,
        "fencing_token": 1,
        "fence_epoch": 1,
        "lease_id": "lease:test-source:1",
        "task_body_digest": "sha256:" + "3" * 64,
        "projection_seed_digest": "sha256:" + "4" * 64,
        "projection_immutable_digest": "sha256:" + "5" * 64,
        "authoritative_task_store": "duckdb",
        "projection_authority": False,
    }
    database_attempt_binding["binding_id"] = "sha256:" + hashlib.sha256(
        json.dumps(
            database_attempt_binding,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    transition: dict[str, object] = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/accepted-source-transition@1"
        ),
        "board_namespace": "test-board-v1",
        "configured_board_admission_cid": admission.admission_cid,
        "task_alias": "TEST-001",
        "database_task_cid": task_cid,
        "attempt_id": attempt_id,
        "attempt_number": 1,
        "portal_attempt_number": 1,
        "claim_id": "claim:test-source:1",
        "fencing_token": 1,
        "database_attempt_binding": database_attempt_binding,
        "canonical_task_cid": "baguqeera" + "a" * 48,
        "canonical_task_key": "task/v1/" + "1" * 64,
        "request_id": "request:test-source:1",
        "merge_request_digest": "sha256:" + "6" * 64,
        "merge_request_dedupe_key": "9" * 64,
        "target_repository_id": capsule.checkout_repository_id(root),
        "baseline_ref": baseline,
        "implementation_commit": implementation,
        "implementation_tree": implementation_tree,
        "merge_commit": merge_commit,
        "merge_tree": merge_tree,
        "target_branch": "main",
        "changed_path_diff_sha256": (
            "sha256:" + hashlib.sha256(diff).hexdigest()
        ),
        "integration_commit_proof": proof,
        "declared_output_invariant": invariant,
        "portal_event_log_sha256": "sha256:" + "8" * 64,
        "authority": (
            "database_completion_cas_after_portal_and_git_verification"
        ),
        "task_completion_authority": False,
        "worker_self_approval": False,
    }
    transition["transition_cid"] = "sha256:" + hashlib.sha256(
        json.dumps(
            transition,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    completion = {
        "operation": "database_complete",
        "validation": {
            "outcome": "passed",
            "task_cid": task_cid,
            "attempt_id": attempt_id,
            "accepted_source_transition": transition,
        },
    }
    return {
        "task_cid": task_cid,
        "task_alias": "TEST-001",
        "status": "completed",
        "revision": 2,
        "completion_receipt": completion,
        "transition": transition,
        "store_generation": 2,
        "database_uuid": "00000000-0000-4000-8000-000000000002",
    }


def _native_launch() -> SimpleNamespace:
    native_pin = _native_pin()
    authorization = _native_authorization(native_pin)
    launch_json = json.dumps(
        {
            "schema": "test-native-launch@1",
            "accepted_authorization_id": authorization["authorization_id"],
            "dependency_id": native_pin.dependency_id,
            "descriptor": 19,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return SimpleNamespace(
        accepted_authorization_id=authorization["authorization_id"],
        pin=native_pin,
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
    quack_projection: _ProjectionFixture,
) -> None:
    projection_pin, extension_set_pin, projection_home = quack_projection
    root, raw_paths = _seed(tmp_path, extension_set_pin)
    paths = tuple(sorted(raw_paths))
    admission = _admission(root, paths, projection_pin, extension_set_pin)

    assert admission.board_namespace == "test-board-v1"
    config_raw = (root / "config/scheduler.json").read_bytes()
    assert admission.configuration_root == cid_for_dag_json(
        {"bytes_sha256": hashlib.sha256(config_raw).hexdigest()}
    )
    assert admission.source_head == _git(root, "rev-parse", "HEAD")
    assert admission.source_tree == _git(root, "rev-parse", "HEAD^{tree}")
    assert tuple(item["path"] for item in admission.control_artifacts) == paths
    assert admission.database_authority["authority_mode"] == "quack"
    assert admission.database_authority["failover_policy"] == "fail_closed"
    assert admission.quack_extension_projection == projection_pin
    assert admission.extension_set_pin == extension_set_pin
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


def test_accepted_source_exact_pin_needs_no_transition_authority(
    tmp_path: Path,
    quack_projection: _ProjectionFixture,
) -> None:
    projection_pin, extension_set_pin, _projection_home = quack_projection
    root, raw_paths = _seed(tmp_path, extension_set_pin)
    admission = _admission(
        root, tuple(sorted(raw_paths)), projection_pin, extension_set_pin
    )

    receipt = capsule.verify_configured_board_accepted_source(
        admission,
        repo_root=root,
        transition_loader=lambda *_args: (_ for _ in ()).throw(
            AssertionError("exact source must not query transition authority")
        ),
    )

    assert receipt["kind"] == "exact"
    assert receipt["source_head"] == receipt["current_head"]
    assert receipt["task_completion_authority"] is False
    assert receipt["merge_commits"] == []


def test_accepted_source_admits_only_exact_database_receipted_merge(
    tmp_path: Path,
    quack_projection: _ProjectionFixture,
) -> None:
    projection_pin, extension_set_pin, _projection_home = quack_projection
    root, raw_paths = _seed(tmp_path, extension_set_pin)
    admission = _admission(
        root, tuple(sorted(raw_paths)), projection_pin, extension_set_pin
    )
    baseline, implementation, merge_commit = _land_test_merge(root)
    authority = _source_transition_authority(
        root,
        admission,
        baseline=baseline,
        implementation=implementation,
        merge_commit=merge_commit,
    )

    receipt = capsule.verify_configured_board_accepted_source(
        admission,
        repo_root=root,
        transition_loader=lambda requested, _scheduler: (
            authority
            if requested == merge_commit
            else (_ for _ in ()).throw(AssertionError("unexpected merge"))
        ),
    )

    assert receipt["kind"] == "accepted_supervisor_merge_successor"
    assert receipt["source_head"] == baseline
    assert receipt["current_head"] == merge_commit
    assert receipt["merge_commits"] == [merge_commit]
    assert receipt["implementation_commits"] == [implementation]
    assert receipt["task_aliases"] == ["TEST-001"]
    assert receipt["database_task_cids"] == [authority["task_cid"]]
    assert receipt["task_completion_authority"] is False

    forged = json.loads(json.dumps(authority))
    forged["transition"]["target_branch"] = "foreign"
    forged["completion_receipt"]["validation"][
        "accepted_source_transition"
    ] = forged["transition"]
    with pytest.raises(
        capsule.ConfiguredBoardLiveCapsuleError,
        match="canonical source transition is inconsistent",
    ):
        capsule.verify_configured_board_accepted_source(
            admission,
            repo_root=root,
            transition_loader=lambda *_args: forged,
        )


def test_accepted_source_rejects_unreceipted_direct_descendant(
    tmp_path: Path,
    quack_projection: _ProjectionFixture,
) -> None:
    projection_pin, extension_set_pin, _projection_home = quack_projection
    root, raw_paths = _seed(tmp_path, extension_set_pin)
    admission = _admission(
        root, tuple(sorted(raw_paths)), projection_pin, extension_set_pin
    )
    output = root / "src/unreceipted.py"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("UNRECEIPTED = True\n", encoding="utf-8")
    _commit_controls(root, "unreceipted direct commit")

    with pytest.raises(
        capsule.ConfiguredBoardLiveCapsuleError,
        match="non-supervisor merge",
    ):
        capsule.verify_configured_board_accepted_source(
            admission,
            repo_root=root,
            transition_loader=lambda *_args: (_ for _ in ()).throw(
                AssertionError("non-merge must fail before authority lookup")
            ),
        )


def test_admission_rejects_dirty_or_head_divergent_controls(
    tmp_path: Path,
    quack_projection: _ProjectionFixture,
) -> None:
    projection_pin, extension_set_pin, _projection_home = quack_projection
    root, raw_paths = _seed(tmp_path, extension_set_pin)
    paths = tuple(sorted(raw_paths))
    (root / "untracked.py").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(
        capsule.ConfiguredBoardLiveCapsuleError,
        match="clean accepted checkout",
    ):
        _admission(root, paths, projection_pin, extension_set_pin)

    (root / "untracked.py").unlink()
    (root / "docs/plan.md").write_text("changed\n", encoding="utf-8")
    with pytest.raises(
        capsule.ConfiguredBoardLiveCapsuleError,
        match="clean accepted checkout",
    ):
        _admission(root, paths, projection_pin, extension_set_pin)


def test_verification_rechecks_capsule_source_and_control_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    quack_projection: _ProjectionFixture,
) -> None:
    projection_pin, extension_set_pin, projection_home = quack_projection
    root, raw_paths = _seed(tmp_path, extension_set_pin)
    paths = tuple(sorted(raw_paths))
    pin = _pin(root)
    admission = _admission(root, paths, projection_pin, extension_set_pin)
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

    forged_payload = admission.as_dict()
    forged_payload["max_lanes"] = admission.max_lanes + 1
    forged_payload["admission_cid"] = cid_for_dag_json(
        {
            key: value
            for key, value in forged_payload.items()
            if key != "admission_cid"
        },
        for_identity=True,
    )
    forged_admission = capsule.parse_configured_board_live_capsule_admission(
        forged_payload
    )
    with pytest.raises(
        capsule.ConfiguredBoardLiveCapsuleError,
        match="scheduler identity differs from admission",
    ):
        capsule.verify_configured_board_live_capsule(
            forged_admission,
            control_plane_pin=pin,
            control_plane_descriptor=9,
            native_dependency_launch=native_launch,
            repo_root=root,
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
    monkeypatch.setenv(
        extension_projection.CONFIGURED_BOARD_EXTENSION_SET_PIN_ENV,
        "{}",
    )
    with pytest.raises(
        capsule.ConfiguredBoardLiveCapsuleError,
        match="extension projection is invalid",
    ):
        capsule.verify_configured_board_live_capsule(
            admission,
            control_plane_pin=pin,
            control_plane_descriptor=9,
            native_dependency_launch=native_launch,
            repo_root=root,
        )
    monkeypatch.setenv(
        extension_projection.CONFIGURED_BOARD_EXTENSION_SET_PIN_ENV,
        extension_set_pin.to_json(),
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


@pytest.mark.parametrize(
    ("tamper_kind", "expected_error"),
    (
        ("native_authorization", "native authorization was not admitted"),
        ("quack_projection", "Quack projection differs from protected authority"),
        ("httpfs_pin", "extension set differs from protected authority"),
    ),
)
def test_each_birth_rejects_forged_protected_native_or_quack_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    quack_projection: _ProjectionFixture,
    tamper_kind: str,
    expected_error: str,
) -> None:
    projection_pin, extension_set_pin, _projection_home = quack_projection
    root, raw_paths = _seed(tmp_path, extension_set_pin)
    dependency_path = root / "config/dependencies.seal.json"
    dependency = json.loads(dependency_path.read_text(encoding="utf-8"))
    if tamper_kind == "native_authorization":
        authorization_path = root / "config/native.authorization.json"
        authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
        authorization["status"] = "rejected"
        authorization_raw = _write_canonical_json(
            authorization_path,
            authorization,
        )
        reference = dependency["configured_board_native_dependency"]["acceptance"]
        reference["sha256"] = "sha256:" + hashlib.sha256(
            authorization_raw
        ).hexdigest()
        reference["size"] = len(authorization_raw)
    elif tamper_kind == "quack_projection":
        alternate_source = tmp_path / "alternate-quack.duckdb_extension"
        alternate_info = tmp_path / "alternate-quack.duckdb_extension.info"
        alternate_source.write_bytes(b"adversarial-quack-bytes")
        alternate_info.write_bytes(b'{"adversarial":true}\n')
        alternate_pin = extension_projection.inspect_configured_board_extension_sources(
            alternate_source,
            alternate_info,
            name="quack",
            engine_version=projection_pin.engine_version,
            platform=projection_pin.platform,
        )
        dependency["configured_board_quack_projection"]["pin"] = (
            alternate_pin.as_dict()
        )
    else:
        dependency["httpfs_extension_pin"]["sha256"] = "9" * 64
    _write_canonical_json(dependency_path, dependency)
    _commit_controls(root, f"forge {tamper_kind}")

    admission = _admission(
        root,
        tuple(sorted(raw_paths)),
        projection_pin,
        extension_set_pin,
    )
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
    with pytest.raises(
        capsule.ConfiguredBoardLiveCapsuleError,
        match=expected_error,
    ):
        capsule.verify_configured_board_live_capsule(
            admission,
            control_plane_pin=_pin(root),
            control_plane_descriptor=9,
            native_dependency_launch=native_launch,
            repo_root=root,
        )


def test_scheduler_inner_native_reauth_and_dependency_snapshot_are_zero_popen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    quack_projection: _ProjectionFixture,
) -> None:
    _projection_pin, extension_set_pin, _projection_home = quack_projection
    root, raw_paths = _seed(tmp_path, extension_set_pin)
    board = SimpleNamespace(
        dependency_seal_path="config/dependencies.seal.json",
        protected_paths=tuple(sorted(raw_paths)),
        live_capsule_control_paths=tuple(sorted(raw_paths)),
        repo_root=root,
        board_namespace="test-board-v1",
        payload={"plan_revision": "TEST-PLAN-R2"},
        path=lambda relative: root / relative,
    )
    snapshot = scheduler._configured_board_dependency_seal_snapshot(board)
    native_launch = _native_launch()
    popen_calls: list[object] = []
    monkeypatch.setattr(
        scheduler,
        "verify_agent_supervisor_native_dependency_sealed_fd",
        lambda launch: f"/proc/self/fd/{launch.descriptor.descriptor}",
    )
    with monkeypatch.context() as birth_context:
        birth_context.setattr(
            scheduler.subprocess,
            "Popen",
            lambda *args, **kwargs: popen_calls.append((args, kwargs)),
        )
        scheduler._authenticate_configured_board_native_dependency_launch(
            board,
            dependency_seal_snapshot=snapshot,
            launch=native_launch,
        )
        forged_launch = SimpleNamespace(
            **{
                **vars(native_launch),
                "accepted_authorization_id": "sha256:" + "9" * 64,
            }
        )
        with pytest.raises(scheduler.ConfiguredBoardError, match="unauthorized"):
            scheduler._authenticate_configured_board_native_dependency_launch(
                board,
                dependency_seal_snapshot=snapshot,
                launch=forged_launch,
            )

    dependency_path = root / "config/dependencies.seal.json"
    dependency = json.loads(dependency_path.read_text(encoding="utf-8"))
    dependency["configured_board_quack_projection"]["load_policy"] = (
        "substituted"
    )
    _write_canonical_json(dependency_path, dependency)
    _commit_controls(root, "substitute dependency seal")
    with pytest.raises(
        scheduler.ConfiguredBoardError,
        match="differs from live admission",
    ):
        scheduler._configured_board_dependency_seal_snapshot(
            board,
            expected_artifact=snapshot.artifact,
        )
    assert popen_calls == []


def test_scheduler_builds_one_exact_httpfs_quack_set_from_protected_controls() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    board = scheduler.load_configured_board(
        repository_root
        / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json",
        repo_root=repository_root,
    )
    snapshot = scheduler._configured_board_dependency_seal_snapshot(board)
    extension_set_pin, pins, sources = (
        scheduler._configured_board_extension_set_projection(
            board,
            dependency_seal_snapshot=snapshot,
        )
    )

    assert tuple(pins) == ("httpfs", "quack")
    assert tuple(sources) == ("httpfs", "quack")
    assert extension_set_pin.versions == {
        "httpfs": "827222f",
        "quack": "c154811",
    }
    assert extension_set_pin.pins == pins
    assert extension_set_pin.set_id == (
        "sha256:52801228bfb51f2201d4dca02206c2aca81ace0640656709f8053780048b5633"
    )


def test_prediction_or_completion_fields_cannot_enter_admission(
    tmp_path: Path,
    quack_projection: _ProjectionFixture,
) -> None:
    projection_pin, extension_set_pin, _projection_home = quack_projection
    root, raw_paths = _seed(tmp_path, extension_set_pin)
    admission = _admission(
        root,
        tuple(sorted(raw_paths)),
        projection_pin,
        extension_set_pin,
    )
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
    quack_projection: _ProjectionFixture,
) -> None:
    projection_pin, extension_set_pin, _projection_home = quack_projection
    root, raw_paths = _seed_handoff(tmp_path, extension_set_pin)
    paths = tuple(sorted(raw_paths))
    pin = _pin(root)
    admission = _admission(root, paths, projection_pin, extension_set_pin)
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
    accepted_source_receipt = capsule.verify_configured_board_accepted_source(
        inherited_admission,
        repo_root=root,
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
        "verify_configured_board_accepted_source",
        lambda _value, **_kwargs: accepted_source_receipt,
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
    supervisor.board_namespace = "test-board-v1"
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
    quack_projection: _ProjectionFixture,
) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    board = scheduler.load_configured_board(
        repository_root / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json",
        repo_root=repository_root,
    )
    monkeypatch.setenv("SAWM_QUACK_TOKEN", "quack-secret")
    monkeypatch.setenv("UNRELATED_SECRET", "must-not-cross")
    monkeypatch.setenv(
        extension_projection.CONFIGURED_BOARD_EXTENSION_DIRECTORY_ENV,
        "/tmp/hostile-ambient-extension-directory",
    )
    monkeypatch.setenv(
        extension_projection.CONFIGURED_BOARD_EXTENSION_SET_PIN_ENV,
        '{"hostile":"ambient"}',
    )
    _projection_pin, extension_set_pin, projection_home = quack_projection
    extension_directory = projection_home / ".duckdb/extensions"
    environment = scheduler._sealed_coordinator_environment(
        board,
        extension_directory=extension_directory,
        extension_set_pin=extension_set_pin,
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
    assert environment[extension_projection.CONFIGURED_BOARD_EXTENSION_SET_PIN_ENV] == (
        extension_set_pin.to_json()
    )


def test_inner_scheduler_authenticates_inherited_launch_before_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    board = SimpleNamespace()
    preflight_calls: list[object] = []
    monkeypatch.setattr(scheduler, "load_configured_board", lambda *_args, **_kwargs: board)
    monkeypatch.setattr(
        scheduler,
        "_sealed_configured_control_plane_required",
        lambda _board: True,
    )
    monkeypatch.setattr(
        scheduler,
        "preflight_configured_board",
        lambda value: preflight_calls.append(value) or {"valid": True},
    )

    result = scheduler.main(
        [
            "--repo-root",
            str(tmp_path),
            "--config",
            str(tmp_path / "scheduler.json"),
            "--accepted-control-plane-pin-json",
            "{}",
            "launch",
        ]
    )

    assert result == 2
    assert preflight_calls == []


def test_operational_live_capsule_missing_or_forged_is_zero_popen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    quack_projection: _ProjectionFixture,
) -> None:
    projection_pin, extension_set_pin, _projection_home = quack_projection
    root, raw_paths = _seed_handoff(tmp_path, extension_set_pin)
    admission = _admission(
        root,
        tuple(sorted(raw_paths)),
        projection_pin,
        extension_set_pin,
    )
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
