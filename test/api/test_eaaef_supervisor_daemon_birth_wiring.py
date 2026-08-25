from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import types
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from ipfs_accelerate_py.agent_supervisor.runtime import (
    eaaef_bootstrap_gateway as runtime,
)
from ipfs_accelerate_py.agent_supervisor.runtime import (
    multi_supervisor_runner as multi_runner,
)
from ipfs_accelerate_py.agent_supervisor.task_sources import duckdb_state
from ipfs_accelerate_py.agent_supervisor.task_sources.eaaef_operational_schema import (
    EAAEF_OPERATIONAL_PROFILE_ID,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    external_agent_container_dispatcher as container_dispatcher,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_daemon,
    implementation_supervisor,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_daemon_runner as daemon_runner,
)
from ipfs_accelerate_py.agent_supervisor.validation import (
    eaaef_lane_gateway_admission as lane,
)
from test.api.test_eaaef_bootstrap_gateway_launch import NOW_MS, _client
from test.api.test_eaaef_lane_gateway_runtime import _qualified_source_artifacts

from ipfs_accelerate_py import llm_router


def _birth_plan(
    artifacts: lane.VerifiedEAAEFLaneRuntimeSourceArtifacts,
    context: dict[str, object],
) -> daemon_runner.EAAEFImplementationDaemonBirthPlan:
    admission = artifacts.admission
    native_bindings = context["native_bindings"]
    assert isinstance(native_bindings, dict)
    return daemon_runner.EAAEFImplementationDaemonBirthPlan(
        board_namespace=str(admission["board_namespace"]),
        source_head=str(admission["source_head"]),
        source_tree=str(admission["source_tree"]),
        configuration_root=str(native_bindings["configuration_root"]),
        accepted_control_plane_capsule_id=str(
            native_bindings["accepted_control_plane_capsule_id"]
        ),
        accepted_control_plane_pin_cid=str(
            native_bindings["accepted_control_plane_pin_cid"]
        ),
        active_plan_root_cid=str(admission["active_plan_root_cid"]),
        active_plan_revision=int(admission["active_plan_revision"]),
        active_plan_revision_cid=str(admission["active_plan_revision_cid"]),
        slice_manifest_cid=str(admission["slice_manifest_cid"]),
        slice_id=str(admission["slice_id"]),
        lane_id=str(admission["lane_id"]),
        task_ids=tuple(admission["task_ids"]),
        task_cids=tuple(admission["task_cids"]),
        lane_session_id=str(admission["lane_session_id"]),
        lane_generation=int(admission["lane_generation"]),
        process_instance_id=str(admission["process_instance_id"]),
        process_birth_nonce=str(admission["process_birth_nonce"]),
        expected_process_uid=int(admission["expected_process_uid"]),
        expected_parent_pid=int(admission["expected_parent_pid"]),
        expected_parent_process_start_time_ticks=int(
            admission["expected_parent_process_start_time_ticks"]
        ),
        expected_executable_sha256=str(admission["expected_executable_sha256"]),
        launch_argv=tuple(sys.argv),
    )


def _reopened_child_birth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    daemon_runner.VerifiedEAAEFImplementationDaemonChildBirth,
    object,
    object,
    dict[str, object],
]:
    artifacts, command_secret, state_secret, context = _qualified_source_artifacts(
        tmp_path / "sources"
    )
    monkeypatch.setattr(runtime.time, "time_ns", lambda: NOW_MS * 1_000_000)
    plan = _birth_plan(artifacts, context)
    coordinates = (
        implementation_supervisor.prepare_eaaef_implementation_daemon_birth_source_coordinates(
            plan=plan,
            source_artifacts=artifacts,
            now_ms=NOW_MS,
        )
    )
    # Exercise the transport boundary: serialized coordinates are not
    # authority and must be parsed before the child can reopen signed sources.
    transported = json.loads(json.dumps(coordinates.to_dict()))
    parsed = lane.parse_eaaef_lane_runtime_dependency_source_coordinates(
        transported
    )
    child_birth = (
        daemon_runner.load_and_verify_eaaef_implementation_daemon_child_birth(
            tmp_path / "sources",
            plan=plan,
            source_coordinates=parsed,
            now_ms=NOW_MS,
        )
    )
    return child_birth, command_secret, state_secret, context


def _runtime_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    child_birth: daemon_runner.VerifiedEAAEFImplementationDaemonChildBirth,
    command_secret: object,
    state_secret: object,
    context: dict[str, object],
) -> runtime.EAAEFLaneRuntimeDependencyBundle:
    admission = child_birth.source_artifacts.admission
    sealed_clients = runtime.bind_eaaef_sealed_quack_client_descriptors(
        admission=admission,
        process_birth=child_birth.process_birth,
        command_descriptor=command_secret.descriptor,
        state_descriptor=state_secret.descriptor,
    )
    native_pin = context["native_pin"]
    descriptor = llm_router.AgentSupervisorNativeDependencyDescriptor(
        schema="ipfs_accelerate_py.agent_supervisor.native-dependency-descriptor@1",
        descriptor=999,
        st_dev=1,
        st_ino=1,
        st_mode=stat.S_IFREG | 0o500,
        st_uid=os.geteuid(),
        st_nlink=0,
        size_bytes=native_pin.size_bytes,
        payload_sha256=native_pin.payload_sha256,
        seals=15,
    )
    native_launch = llm_router.AgentSupervisorNativeDependencyLaunch(
        schema="ipfs_accelerate_py.agent_supervisor.native-dependency-launch@1",
        accepted_authorization_id=(
            child_birth.source_artifacts.native_admission.admission_cid
        ),
        pin=native_pin,
        descriptor=descriptor,
    )
    native_path = "/proc/self/fd/999"
    native_module = types.ModuleType("_duckdb")
    native_module.__file__ = native_path
    native_module.__version__ = native_pin.distribution_version

    class FixedTemplateConnection:
        """In-memory stand-in: records factory-owned SQL without executing it."""

        def __init__(self) -> None:
            self.statements: list[str] = []
            self.closed = False

        def execute(
            self, statement: str, *_args: object
        ) -> FixedTemplateConnection:
            self.statements.append(statement)
            return self

        def close(self) -> None:
            self.closed = True

    connections: list[FixedTemplateConnection] = []

    def connect(**_kwargs: object) -> FixedTemplateConnection:
        connection = FixedTemplateConnection()
        connections.append(connection)
        return connection

    native_module.connect = connect
    context["synthetic_connections"] = connections
    monkeypatch.setitem(sys.modules, "_duckdb", native_module)
    monkeypatch.setitem(sys.modules, "duckdb", native_module)
    monkeypatch.setattr(
        runtime,
        "verify_agent_supervisor_native_dependency_sealed_fd",
        lambda _launch: native_path,
    )
    monkeypatch.setattr(
        runtime.socket,
        "socket",
        lambda *_args, **_kwargs: pytest.fail(
            "synthetic wiring opened a dynamic-service socket"
        ),
    )

    journal_parent = tmp_path / "journal"
    journal_parent.mkdir(mode=0o700)
    return daemon_runner.build_eaaef_implementation_daemon_runtime_bundle(
        child_birth=child_birth,
        native_launch=native_launch,
        native_module=native_module,
        sealed_descriptors=sealed_clients,
        authorization_client=_client(
            dict(admission.operational_capability),
            context,
            clock_ms=lambda: NOW_MS,
        ),
        journal_parent_directory=journal_parent,
    )


def test_signed_parent_child_wiring_constructs_exact_daemon_without_live_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child_birth, command_secret, state_secret, context = _reopened_child_birth(
        tmp_path, monkeypatch
    )
    bundle: runtime.EAAEFLaneRuntimeDependencyBundle | None = None
    try:
        monkeypatch.setattr(
            implementation_supervisor.subprocess,
            "Popen",
            lambda *_args, **_kwargs: pytest.fail(
                "source-only EAAEF wiring spawned a process"
            ),
        )
        bundle = _runtime_bundle(
            tmp_path,
            monkeypatch,
            child_birth,
            command_secret,
            state_secret,
            context,
        )
        monkeypatch.setenv(
            "IPFS_ACCELERATE_AGENT_STATE_SCHEMA_REVISION",
            "poisoned-environment-revision",
        )
        signed_store_id = str(
            child_birth.source_artifacts.admission.operational_capability[
                "store_id"
            ]
        )
        path_calls: list[tuple[str, Path]] = []

        def observe_path_method(name: str, original: object) -> object:
            def observed(value: Path, *args: object, **kwargs: object) -> object:
                path_calls.append((name, value))
                return original(value, *args, **kwargs)

            return observed

        for method_name in ("open", "mkdir", "lstat"):
            monkeypatch.setattr(
                Path,
                method_name,
                observe_path_method(method_name, getattr(Path, method_name)),
            )
        monkeypatch.setattr(
            duckdb_state,
            "open_duckdb_connection",
            lambda *_args, **_kwargs: pytest.fail(
                "EAAEF constructor opened a direct DuckDB path"
            ),
        )
        monkeypatch.setattr(
            container_dispatcher.ExternalAgentContainerWorkerDispatcher,
            "run_provider",
            lambda *_args, **_kwargs: pytest.fail(
                "EAAEF constructor invoked a provider"
            ),
        )
        monkeypatch.setattr(
            container_dispatcher.ExternalAgentContainerWorkerDispatcher,
            "apply_effect",
            lambda *_args, **_kwargs: pytest.fail(
                "EAAEF constructor invoked an effect"
            ),
        )
        monkeypatch.setattr(
            container_dispatcher.ExternalAgentContainerWorkerDispatcher,
            "validate_effect",
            lambda *_args, **_kwargs: pytest.fail(
                "EAAEF constructor invoked validation"
            ),
        )
        daemon = (
            daemon_runner.build_eaaef_database_implementation_daemon_from_runtime_bundle(
                child_birth=child_birth,
                runtime_bundle=bundle,
                install_schema=False,
            )
        )
        admission = child_birth.source_artifacts.admission
        assert daemon.owner_session_id == admission["lane_session_id"]
        assert daemon.process_instance_id == admission["process_instance_id"]
        assert daemon.state_schema_revision == EAAEF_OPERATIONAL_PROFILE_ID
        assert daemon.state_schema_revision != "poisoned-environment-revision"
        assert daemon.database_path == Path(
            str(admission.operational_capability["store_id"])
        )
        assert daemon._quack_uri == admission.operational_capability["command_endpoint"]
        assert daemon._quack_command_gateway is bundle.gateway
        assert daemon.execution_callbacks_bound is True
        assert daemon.markdown_path is None
        assert daemon.state_path is None
        assert daemon.strategy_path is None
        assert daemon.events_path is None
        assert daemon.pid_path is None
        assert daemon.queue_path is None
        assert all(
            signed_store_id not in path.parts for _method, path in path_calls
        )
        daemon.close()
        assert bundle.gateway._dispatcher._transport._closed is True
        synthetic_connections = context["synthetic_connections"]
        assert isinstance(synthetic_connections, list)
        assert len(synthetic_connections) == 2
        assert all(connection.closed for connection in synthetic_connections)
    finally:
        if bundle is not None:
            bundle.close()
        command_secret.close()
        state_secret.close()


def test_constructor_failure_closes_exact_runtime_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child_birth, command_secret, state_secret, context = _reopened_child_birth(
        tmp_path, monkeypatch
    )
    bundle: runtime.EAAEFLaneRuntimeDependencyBundle | None = None
    closed = False
    try:
        bundle = _runtime_bundle(
            tmp_path,
            monkeypatch,
            child_birth,
            command_secret,
            state_secret,
            context,
        )
        original_close = runtime.EAAEFLaneRuntimeDependencyBundle.close

        def observed_close(
            value: runtime.EAAEFLaneRuntimeDependencyBundle,
        ) -> None:
            nonlocal closed
            closed = True
            original_close(value)

        monkeypatch.setattr(
            runtime.EAAEFLaneRuntimeDependencyBundle,
            "close",
            observed_close,
        )

        def fail_constructor(**_kwargs: object) -> None:
            raise RuntimeError("synthetic constructor failure")

        monkeypatch.setattr(
            implementation_daemon,
            "DatabaseImplementationDaemon",
            fail_constructor,
        )
        with pytest.raises(RuntimeError, match="synthetic constructor failure"):
            daemon_runner.build_eaaef_database_implementation_daemon_from_runtime_bundle(
                child_birth=child_birth,
                runtime_bundle=bundle,
                install_schema=False,
            )
        assert closed is True
        assert bundle.gateway._dispatcher._transport._closed is True
        synthetic_connections = context["synthetic_connections"]
        assert isinstance(synthetic_connections, list)
        assert all(connection.closed for connection in synthetic_connections)
    finally:
        if bundle is not None and not closed:
            bundle.close()
        command_secret.close()
        state_secret.close()


@pytest.mark.parametrize(
    ("field_name", "foreign"),
    [
        ("process_instance_id", "process:foreign-birth"),
        ("source_tree", "0" * 40),
        ("launch_argv", ("python", "--foreign-child")),
    ],
)
def test_parent_rejects_plan_or_birth_mismatch_before_runtime_factory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field_name: str,
    foreign: object,
) -> None:
    artifacts, command_secret, state_secret, context = _qualified_source_artifacts(
        tmp_path / "sources"
    )
    try:
        plan = replace(_birth_plan(artifacts, context), **{field_name: foreign})
        monkeypatch.setattr(
            runtime,
            "create_eaaef_lane_runtime_dependency_factory",
            lambda **_kwargs: pytest.fail("runtime factory ran before parent acceptance"),
        )
        with pytest.raises(
            implementation_supervisor.PlanBoundDispatchError,
            match="rejected the planned signed daemon birth",
        ):
            implementation_supervisor.prepare_eaaef_implementation_daemon_birth_source_coordinates(
                plan=plan,
                source_artifacts=artifacts,
                now_ms=NOW_MS,
            )
    finally:
        command_secret.close()
        state_secret.close()


def test_child_rejects_mapping_coordinates_and_actual_argv_mismatch_pre_factory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts, command_secret, state_secret, context = _qualified_source_artifacts(
        tmp_path / "sources"
    )
    try:
        monkeypatch.setattr(runtime.time, "time_ns", lambda: NOW_MS * 1_000_000)
        plan = _birth_plan(artifacts, context)
        coordinates = (
            implementation_supervisor.prepare_eaaef_implementation_daemon_birth_source_coordinates(
                plan=plan,
                source_artifacts=artifacts,
                now_ms=NOW_MS,
            )
        )
        with pytest.raises(
            daemon_runner.EAAEFImplementationDaemonBirthError,
            match="exact parsed source coordinates",
        ):
            daemon_runner.load_and_verify_eaaef_implementation_daemon_child_birth(
                tmp_path / "sources",
                plan=plan,
                source_coordinates=coordinates.to_dict(),
                now_ms=NOW_MS,
            )
        parsed = lane.parse_eaaef_lane_runtime_dependency_source_coordinates(
            coordinates.to_dict()
        )
        monkeypatch.setattr(sys, "argv", [*sys.argv, "--forged-after-signing"])
        monkeypatch.setattr(
            runtime,
            "create_eaaef_lane_runtime_dependency_factory",
            lambda **_kwargs: pytest.fail("runtime factory ran before argv join"),
        )
        with pytest.raises(
            daemon_runner.EAAEFImplementationDaemonBirthError,
            match="actual child UID|OS process birth",
        ):
            daemon_runner.load_and_verify_eaaef_implementation_daemon_child_birth(
                tmp_path / "sources",
                plan=plan,
                source_coordinates=parsed,
                now_ms=NOW_MS,
            )
    finally:
        command_secret.close()
        state_secret.close()


@pytest.mark.parametrize(
    "invalid_revision",
    ["", "datasets-authoritative-eaaef-operational-control-plane@1"],
)
def test_daemon_constructor_rejects_empty_or_mismatched_gateway_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_revision: str,
) -> None:
    child_birth, command_secret, state_secret, context = _reopened_child_birth(
        tmp_path, monkeypatch
    )
    bundle: runtime.EAAEFLaneRuntimeDependencyBundle | None = None
    try:
        bundle = _runtime_bundle(
            tmp_path,
            monkeypatch,
            child_birth,
            command_secret,
            state_secret,
            context,
        )
        bundle.gateway.capability.state_schema_revision = invalid_revision
        with pytest.raises(
            daemon_runner.EAAEFImplementationDaemonBirthError,
            match="signed store/schema authority",
        ):
            daemon_runner.build_eaaef_database_implementation_daemon_from_runtime_bundle(
                child_birth=child_birth,
                runtime_bundle=bundle,
                install_schema=False,
            )
    finally:
        if bundle is not None:
            bundle.close()
        command_secret.close()
        state_secret.close()


def test_live_eaaef_supervisor_and_daemon_launches_remain_pre_popen_no_go(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    program = multi_runner.DatabaseProgramConfig(
        authority_mode="quack",
        task_source_kind="duckdb",
        endpoint_secret_handle="secret-handle:eaaef:qualified",
        quack_endpoint="quack:127.0.0.1:19495",
        store_id="eaaef-control-run-v6",
        store_generation="generation:6",
        schema_revision=EAAEF_OPERATIONAL_PROFILE_ID,
        failover_policy="fail_closed",
    )
    with pytest.raises(
        ValueError,
        match="independently_signed_native_dependency_acceptance_absent",
    ):
        multi_runner._assert_eaaef_operational_child_profile(
            common_args=program.cli_args(),
            track_args=("--plan-bound-dispatch",),
            operational=program,
            command_fabric={"child_adapter_status": "admitted"},
            worker_network_policy={"child_propagation_status": "admitted"},
            worker_principal_did="did:key:zWorkerPrincipal",
            provider_principal_did="did:key:zProviderPrincipal",
            forbidden_bootstrap_paths=(),
        )

    supervisor = object.__new__(
        implementation_supervisor.PortalImplementationSupervisor
    )
    supervisor.config = SimpleNamespace(
        database_program=program,
        repo_root=tmp_path,
    )
    monkeypatch.setattr(
        implementation_supervisor.PortalImplementationSupervisor,
        "ensure_managed_daemon_pid_file",
        lambda _self: pytest.fail("EAAEF no-go repaired a PID file"),
    )
    monkeypatch.setattr(
        implementation_supervisor.PortalImplementationSupervisor,
        "_build_daemon_command",
        lambda _self: pytest.fail("EAAEF no-go built a child command"),
    )
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("EAAEF no-go spawned a child"),
    )
    with pytest.raises(
        implementation_supervisor.PlanBoundDispatchError,
        match=(
            "independently_signed_native_dependency_acceptance_absent.*"
            "independent_native_dependency_authority_verifier_absent.*"
            "accepted_signed_per_birth_lane_artifact_instances_absent.*"
            "accepted_signed_quack_client_qualification_instance_absent.*"
            "accepted_signed_container_service_qualification_instance_absent.*"
            "qualified_live_native_dependency_authority_absent.*"
            "qualified_live_container_service_authority_absent"
        ),
    ):
        supervisor._start_daemon()
