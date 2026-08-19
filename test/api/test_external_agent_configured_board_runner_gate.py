from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from ipfs_accelerate_py.agent_implementation_route import (
    AgentImplementationControlPlanePin,
)
from ipfs_accelerate_py.agent_supervisor.runtime import (
    configured_board_scheduler as scheduler,
)
from ipfs_accelerate_py.agent_supervisor.runtime import (
    multi_supervisor_runner as runner,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_supervisor,
)


def _sha(token: str) -> str:
    return "sha256:" + token * 64


def _pin() -> AgentImplementationControlPlanePin:
    return AgentImplementationControlPlanePin(
        schema="ipfs_accelerate_py.agent_supervisor.accepted-control-plane@2",
        runner_path="/sealed/runner.py",
        runner_sha256=_sha("1"),
        capsule_root="/sealed",
        capsule_id=_sha("2"),
        source_head="3" * 40,
        source_tree="4" * 40,
        archive_sha256=_sha("5"),
    )


def _database_payload() -> dict[str, object]:
    return {
        "board_namespace": "external-agent-autonomous-execution-fabric-v1",
        "configured_board_live_seal": {
            "worker_network_authorization_policy": {
                "schema": (
                    "ipfs_accelerate_py/agent-supervisor/"
                    "eaaef-worker-network-dispatch-policy@1"
                ),
                "authorization_schema": (
                    "ipfs_accelerate_py/eaaef-worker-network-authorization@1"
                ),
                "verifier_interface": "verify_worker_network_authorization@1",
                "artifact_path_authority": "verified_invocation_profile_dir",
                "artifact_relative_path_template": (
                    "network-authorizations/<sha256(invocation_id)>/<provider>.json"
                ),
                "dynamic_caller_path_allowed": False,
                "expected_artifact_cid_required": True,
                "expected_worker_principal_did_required": True,
                "expected_provider_principal_did_required": True,
                "control_plane_capsule_binding_required": True,
                "task_plan_source_worktree_effect_binding_required": True,
                "container_and_lease_binding_required": True,
                "create_start_restart_reverification_required": True,
                "supported_providers": ["codex", "grok"],
                "child_propagation_status": "unavailable_fail_closed",
            }
        },
        "bootstrap_database_program": {
            "authority_mode": "embedded",
            "task_source_kind": "duckdb",
            "store_id": "data/eaaef/run-v1/control.duckdb",
            "store_generation": "run-v1",
            "schema_revision": (
                "datasets-authoritative-eaaef-operational-control-plane@2"
            ),
            "failover_policy": "fail_closed",
        },
        "database_program": {
            "authority_mode": "quack",
            "task_source_kind": "duckdb",
            "endpoint_secret_handle": "secret-handle:eaaef-quack",
            "quack_endpoint": "quack:127.0.0.1:19494",
            "store_id": "eaaef-control",
            "store_generation": "run-v1",
            "schema_revision": (
                "datasets-authoritative-eaaef-operational-control-plane@2"
            ),
            "failover_policy": "fail_closed",
        },
        "operational_command_fabric": {
            "schema": (
                "ipfs_accelerate_py/agent-supervisor/"
                "eaaef-signed-command-fabric-profile@2"
            ),
            "transport_kind": "signed_command_fabric",
            "board_namespace": "external-agent-autonomous-execution-fabric-v1",
            "shard_id": "control-shard-0",
            "ingress_endpoint": "quack:127.0.0.1:19494",
            "ingress_secret_handle": "secret-handle:eaaef-ingress",
            "projection_endpoint": "quack:127.0.0.1:19495",
            "projection_secret_handle": "secret-handle:eaaef-projection",
            "store_id": "eaaef-control",
            "store_generation": "run-v1",
            "schema_revision": (
                "datasets-authoritative-eaaef-operational-control-plane@2"
            ),
            "owner_qualification_schema": (
                "ipfs_accelerate_py/agent-supervisor/"
                "eaaef-quack-owner-qualification@1"
            ),
            "command_envelope_schema": (
                "ipfs_accelerate_py/agent-supervisor/authorized-state-command@1"
            ),
            "state_command_schema": (
                "ipfs_accelerate_py/agent-supervisor/state-command@1"
            ),
            "ingress_relation": "command_inbox",
            "ingress_append_only": True,
            "ingress_accepts_signed_envelopes_only": True,
            "operational_database_private": True,
            "operational_tables_remotely_exposed": False,
            "one_mutable_owner": True,
            "owner_verifies_signed_envelopes": True,
            "projection_read_only": True,
            "projection_append_allowed": False,
            "atomic_plan_r2_required": True,
            "direct_file_fallback": False,
            "failover_policy": "fail_closed",
            "child_adapter_status": "implemented_unqualified_fail_closed",
        },
    }


def test_eaaef_uses_plan_bound_slices_not_legacy_hash_sharding() -> None:
    board = SimpleNamespace(
        board_namespace="external-agent-autonomous-execution-fabric-v1",
        payload={"schema": scheduler.EAAEF_SCHEDULER_SCHEMA},
    )
    assert scheduler._plan_bound_profile(board) is True
    assert scheduler._eaaef_plan_bound_profile(board) is True


def test_eaaef_legacy_quack_profile_cannot_substitute_for_command_fabric() -> None:
    payload = _database_payload()
    operational, forbidden, command_fabric, network_policy = (
        runner._validated_eaaef_database_programs(payload)
    )
    with pytest.raises(
        ValueError,
        match="child_adapter_unavailable.*network_authorization_propagation_unavailable",
    ):
        runner._assert_eaaef_operational_child_profile(
            common_args=operational.cli_args(),
            track_args=("--plan-bound-dispatch",),
            operational=operational,
            command_fabric=command_fabric,
            worker_network_policy=network_policy,
            worker_principal_did="did:key:zWorker",
            provider_principal_did="did:key:zProviderService",
            forbidden_bootstrap_paths=forbidden,
        )
    assert operational.authority_mode == "quack"
    assert forbidden == ("data/eaaef/run-v1/control.duckdb",)
    assert not set(forbidden).intersection(operational.cli_args())

    embedded_args = list(operational.cli_args())
    embedded_args[
        embedded_args.index("quack")
    ] = "embedded"
    with pytest.raises(ValueError, match="operational Quack"):
        runner._assert_eaaef_operational_child_profile(
            common_args=embedded_args,
            track_args=(),
            operational=operational,
            command_fabric=command_fabric,
            worker_network_policy=network_policy,
            worker_principal_did="did:key:zWorker",
            provider_principal_did="did:key:zProviderService",
            forbidden_bootstrap_paths=forbidden,
        )


def test_eaaef_database_roles_reject_direct_file_fallback() -> None:
    payload = _database_payload()
    operational = payload["database_program"]
    assert isinstance(operational, dict)
    operational["store_id"] = "data/eaaef/run-v1/control.duckdb"
    with pytest.raises(ValueError, match="no direct-file fallback"):
        runner._validated_eaaef_database_programs(payload)


@pytest.mark.parametrize(
    "profile_name",
    ["bootstrap_database_program", "database_program"],
)
def test_eaaef_database_roles_require_exact_operational_profile_v2(
    profile_name: str,
) -> None:
    payload = _database_payload()
    profile = payload[profile_name]
    assert isinstance(profile, dict)
    profile["schema_revision"] = "datasets-authoritative-operational-v1"

    with pytest.raises(ValueError, match="exact operational profile @2"):
        runner._validated_eaaef_database_programs(payload)


def test_eaaef_profile_v2_requires_the_live_seal_gate() -> None:
    assert runner._configured_board_live_seal_required(
        (
            "--state-schema-revision",
            runner.EAAEF_OPERATIONAL_SCHEMA_REVISION,
        )
    )


def test_supervisor_loader_accepts_closed_scheduler_v2_only(
    tmp_path: Path,
) -> None:
    (tmp_path / "config").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/tasks.md").write_text("# Tasks\n", encoding="utf-8")
    (tmp_path / "docs/objectives.md").write_text("# Objectives\n", encoding="utf-8")
    payload: dict[str, object] = {
        "schema": (
            "ipfs_accelerate_py.agent_supervisor."
            "external_agent_autonomous_execution_fabric.scheduler_config@2"
        ),
        "taskboard_path": "docs/tasks.md",
        "objectives_path": "docs/objectives.md",
        "task_prefix": "EAAEF-",
        "board_namespace": "external-agent-autonomous-execution-fabric-v1",
        "merge_target_branch": "integration/eaaef-v1",
        "max_lanes": 5,
        "max_restarts": 3,
        "max_task_attempts": 3,
        "implementation_timeout_seconds": 120,
        "validation_max_workers": 1,
        "poll_interval_seconds": 1,
        "daemon_interval_seconds": 1,
        "check_interval_seconds": 1,
        "stale_seconds": 60,
        "worktree_submodule_paths": [],
        "protected_paths": [],
    }
    path = tmp_path / "config/eaaef.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = implementation_supervisor.load_supervisor_scheduler_config(
        path, repo_root=tmp_path
    )
    assert loaded["schema"].endswith("scheduler_config@2")

    payload["schema"] = str(payload["schema"]).replace("@2", "@3")
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        implementation_supervisor.SupervisorSchedulerConfigError,
        match="@1/@2",
    ):
        implementation_supervisor.load_supervisor_scheduler_config(
            path, repo_root=tmp_path
        )


def test_start_track_requires_live_ticket_before_any_popen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    track = runner.SupervisorTrack(
        name="eaaef-lane-0",
        script_path=tmp_path / "entry.py",
        log_path=tmp_path / "lane.log",
        supervisor_pid_path=tmp_path / "lane.pid",
        daemon_pid_path=tmp_path / "daemon.pid",
        extra_args=("--plan-bound-dispatch",),
    )
    (tmp_path / "entry.py").write_text("# entry\n", encoding="utf-8")
    calls: list[object] = []
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *args, **kwargs: calls.append(args))

    with pytest.raises(ValueError, match="NO-GO"):
        runner.start_track(
            track,
            repo_root=tmp_path,
            common_args=(
                "--state-schema-revision",
                runner.EAAEF_OPERATIONAL_SCHEMA_REVISION,
            ),
            accepted_control_plane_pin=_pin(),
        )
    assert calls == []


def test_plan_bound_birth_gate_reopens_ticket_after_parent_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pin = _pin()
    read_fd, write_fd = os.pipe()
    os.write(write_fd, runner.PLAN_BOUND_LAUNCH_GATE_SUCCESS)
    os.close(write_fd)
    monkeypatch.setattr(runner, "parse_accepted_control_plane_pin", lambda _value: pin)
    monkeypatch.setattr(
        runner,
        "verify_agent_implementation_sealed_control_plane",
        lambda *_args: "/proc/self/fd/99",
    )
    monkeypatch.setattr(
        runner,
        "_canonical_accepted_tree_root",
        lambda _path: tmp_path,
    )
    monkeypatch.setattr(
        runner,
        "build_sealed_control_plane_module_command",
        lambda **_kwargs: ["python", "sealed-bootstrap"],
    )
    monkeypatch.setattr(runner, "_validate_plan_bound_accepted_tree", lambda **_kwargs: None)
    birth_checks: list[str] = []

    def reject_swapped_ticket(**kwargs):
        birth_checks.append(kwargs["live_config"])
        raise ValueError("post-parent ticket swap")

    monkeypatch.setattr(
        runner,
        "_verify_eaaef_configured_board_birth",
        reject_swapped_ticket,
    )
    monkeypatch.setattr(
        runner.os,
        "execvpe",
        lambda *_args, **_kwargs: pytest.fail("child executed after ticket swap"),
    )
    child_args = [
        "--plan-bound-dispatch",
        "--plan-bound-source-head",
        pin.source_head,
        "--plan-bound-source-tree",
        pin.source_tree,
        "--plan-bound-accepted-tree-root",
        str(tmp_path),
        "--plan-revision-store-path",
        "state/plan-store",
        "--plan-bound-revision-cid",
        _sha("6"),
        "--plan-bound-slice-id",
        "slice-0",
        "--plan-bound-lane-id",
        "lane-0",
        "--state-dir",
        "state/lane-0",
    ]
    tokens = [
        str(read_fd),
        str(tmp_path),
        "pin-json",
        "99",
        "-",
        runner.EAAEF_CONFIGURED_BOARD_LIVE_SEAL_CONFIG_PATH,
        "--",
        "python",
        "sealed-bootstrap",
        *child_args,
    ]
    try:
        assert runner._run_plan_bound_launch_gate(tokens) == 78
    finally:
        try:
            os.close(read_fd)
        except OSError:
            pass
    assert birth_checks == [runner.EAAEF_CONFIGURED_BOARD_LIVE_SEAL_CONFIG_PATH]
