"""EAAEF-055: container isolation contracts; never invoke a live engine."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.containers import checkpoint as checkpoint_mod
from ipfs_accelerate_py.agent_supervisor.containers.checkpoint import (
    CHECKPOINT_SCHEMA,
    CheckpointError,
    ContainerCheckpoint,
    recover,
)
from ipfs_accelerate_py.agent_supervisor.containers.contracts import (
    ContainerExecutionProfile,
    ContainerTrustError,
    IsolationPolicy,
    ResourceBounds,
)
from ipfs_accelerate_py.agent_supervisor.containers.oci_runner import (
    EngineAdmission,
    OciRunner,
    OciRunnerTrustError,
    build_oci_run_spec,
)


IMAGE_DIGEST = "sha256:" + ("a" * 64)
_RESOURCES = ResourceBounds(
    cpu_millicores=4000,
    ram_mib=8192,
    disk_mib=16384,
    timeout_seconds=7200,
    gpu_count=0,
)


def _profile(**changes: object) -> ContainerExecutionProfile:
    values: dict[str, object] = {
        "image_digest": IMAGE_DIGEST,
        "worktree_id": "worktree:eaaef-055",
        "task_id": "task:EAAEF-055",
        "authority_id": "authority:supervisor",
        "resources": _RESOURCES,
        "policy": IsolationPolicy(),
    }
    values.update(changes)
    return ContainerExecutionProfile(**values)  # type: ignore[arg-type]


def test_default_deny_isolation_without_live_engine() -> None:
    spec = build_oci_run_spec(_profile())
    assert spec.live_engine_invoked is False
    assert spec.docker_socket_mounted is False
    assert spec.privileged is False
    assert spec.network == "none"
    assert spec.read_only is True
    assert spec.cap_drop == ("ALL",)
    assert spec.no_new_privileges is True
    assert "--network=none" in spec.argv
    assert "--read-only" in spec.argv
    assert "--cap-drop=ALL" in spec.argv
    assert "--security-opt=no-new-privileges" in spec.argv
    assert "--privileged" not in spec.argv
    assert not any(
        "docker.sock" in part and not part.startswith("--host=") for part in spec.argv
    )
    assert spec.cpu_millicores == 4000
    assert spec.ram_mib == 8192
    assert spec.disk_mib == 16384
    assert spec.timeout_seconds == 7200
    runner = OciRunner()
    planned = runner.build_spec(_profile())
    assert planned.live_engine_invoked is False
    assert planned.docker_socket_mounted is False
    assert isinstance(EngineAdmission().rootful_fallback_admitted, bool)


def test_isolation_policy_rejects_docker_socket_and_privileged() -> None:
    with pytest.raises(ContainerTrustError, match="docker.sock"):
        IsolationPolicy(docker_socket_mounted=True)
    with pytest.raises(ContainerTrustError, match="privileged"):
        IsolationPolicy(privileged=True)
    profile = _profile()
    with pytest.raises(OciRunnerTrustError, match="docker.sock"):
        build_oci_run_spec(
            profile,
            extra_mounts=(
                {
                    "source": "/var/run/docker.sock",
                    "target": "/var/run/docker.sock",
                    "read_only": True,
                    "kind": "other",
                },
            ),
        )
    with pytest.raises(OciRunnerTrustError, match="privileged"):
        build_oci_run_spec(profile, privileged=True)
    policy = IsolationPolicy()
    assert policy.docker_socket_mounted is False
    assert policy.privileged is False
    assert policy.no_new_privileges is True
    assert policy.read_only_base is True


def test_checkpoint_contracts_exist_without_starting_containers() -> None:
    assert checkpoint_mod.CHECKPOINT_SCHEMA == CHECKPOINT_SCHEMA
    assert CHECKPOINT_SCHEMA.endswith("@1")
    checkpoint = ContainerCheckpoint(
        attempt_id="attempt:1",
        worktree_id="worktree:eaaef-055",
        fence_token=1,
        lane_id="lane:0",
        owner_alive=False,
    )
    assert checkpoint.to_dict()["schema"] == CHECKPOINT_SCHEMA
    recovered = recover(checkpoint, next_fence=2)
    assert recovered.fence_token == 2
    assert recovered.owner_alive is True
    live = ContainerCheckpoint(
        attempt_id="attempt:1",
        worktree_id="worktree:eaaef-055",
        fence_token=2,
        lane_id="lane:0",
        owner_alive=True,
    )
    with pytest.raises(CheckpointError, match="live owner"):
        recover(live, next_fence=3)
    with pytest.raises(CheckpointError, match="later fence"):
        recover(checkpoint, next_fence=1)
    assert not hasattr(recover, "start_container")
    assert build_oci_run_spec(_profile()).live_engine_invoked is False
