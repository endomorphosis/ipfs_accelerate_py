from __future__ import annotations

import pytest
from ipfs_accelerate_py.agent_supervisor.runtime import eaaef_bootstrap_gateway as runtime
from ipfs_accelerate_py.agent_supervisor.task_sources.eaaef_borrowed_transaction import (
    EAAEF_DAEMON_LANE_BINDING_SCHEMA,
    EAAEF_DEAD_LANE_RECOVERY_AUTHORITY_SCHEMA,
    EAAEF_TASK_OPERATION_AUTHORITY_SCHEMA,
)
from ipfs_accelerate_py.agent_supervisor.validation import (
    eaaef_bootstrap_gateway_launch as launch,
)


def _sha(token: str) -> str:
    return "sha256:" + token * 64


def _verified_capability() -> launch.VerifiedEAAEFBootstrapOperationalCapability:
    # Runtime projection tests use the validation module's closed constructor
    # token.  Production has no corresponding lane/proxy factory while the
    # exported blockers remain unresolved.
    return launch.VerifiedEAAEFBootstrapOperationalCapability(
        launch._VERIFIED_OPERATIONAL_CAPABILITY_TOKEN,
        {
            "gateway_binding_cid": _sha("1"),
            "owner_principal_did": "did:key:zEAAEFOwner",
            "owner_session_id": "session:eaaef-owner-v7",
            "owner_generation": 7,
            "fence_epoch": 11,
        },
    )


def _lane(
    *,
    lane_session_id: str = "session:eaaef-lane-0-v1",
    lane_generation: int = 1,
    process_instance_id: str = "process:eaaef-lane-0-v1",
) -> dict[str, object]:
    return {
        "schema": EAAEF_DAEMON_LANE_BINDING_SCHEMA,
        "gateway_binding_cid": _sha("1"),
        "owner_principal_did": "did:key:zEAAEFOwner",
        "owner_session_id": "session:eaaef-owner-v7",
        "owner_generation": 7,
        "lane_session_id": lane_session_id,
        "lane_generation": lane_generation,
        "process_instance_id": process_instance_id,
        "fence_epoch": 11,
    }


def _task_authority() -> dict[str, object]:
    return {
        "schema": EAAEF_TASK_OPERATION_AUTHORITY_SCHEMA,
        "task_cid": "task:eaaef:1",
        "claim_id": "claim:eaaef:1",
        "attempt_id": "attempt:eaaef:1",
        "attempt_number": 1,
        "lease_id": "lease:eaaef:task:1",
        "owner_session_id": "session:eaaef-lane-0-v1",
        "fencing_token": 13,
        "fence_epoch": 11,
        "daemon_lane_binding": _lane(),
    }


def test_runtime_proxy_is_exactly_marked_but_public_construction_is_no_go() -> None:
    assert runtime.EAAEF_BOOTSTRAP_EXECUTION_REPOSITORY_PROXY_INTERFACE == (
        "EAAEFBootstrapExecutionRepositoryProxy@2"
    )
    assert runtime.EAAEFBootstrapExecutionRepositoryProxy.EAAEF_INTERFACE == (
        runtime.EAAEF_BOOTSTRAP_EXECUTION_REPOSITORY_PROXY_INTERFACE
    )
    assert runtime.EAAEF_BOOTSTRAP_RUNTIME_GATEWAY_QUALIFICATION_STATUS == (
        "r1_source_verified_runtime_factory_implemented"
    )
    assert set(runtime.EAAEF_BOOTSTRAP_RUNTIME_GATEWAY_PRODUCTION_BLOCKERS) == {
        "signed_quack_client_factory_qualification_artifact_absent",
        "signed_dynamic_dispatcher_service_qualification_artifact_absent",
        "independently_signed_per_birth_lane_runtime_artifact_absent",
    }
    with pytest.raises(
        runtime.EAAEFBootstrapRuntimeGatewayNoGo,
        match="signed_quack_client_factory_qualification_artifact_absent",
    ):
        runtime.EAAEFBootstrapExecutionRepositoryProxy(
            verified_capability=_verified_capability(),
        )
    with pytest.raises(runtime.EAAEFBootstrapRuntimeGatewayNoGo):
        runtime.EAAEFBootstrapExecutionRepositoryProxy.from_unqualified_runtime()


def test_lane_and_task_authority_projections_are_closed_and_cross_bound() -> None:
    capability = _verified_capability()
    lane = runtime.eaaef_daemon_lane_binding_projection(_lane(), verified_capability=capability)
    assert set(lane) == {
        "schema",
        "gateway_binding_cid",
        "owner_principal_did",
        "owner_session_id",
        "owner_generation",
        "lane_session_id",
        "lane_generation",
        "process_instance_id",
        "fence_epoch",
    }
    authority = runtime.eaaef_task_operation_authority_projection(
        _task_authority(), verified_capability=capability
    )
    assert set(authority) == {
        "schema",
        "task_cid",
        "claim_id",
        "attempt_id",
        "attempt_number",
        "lease_id",
        "owner_session_id",
        "fencing_token",
        "fence_epoch",
        "daemon_lane_binding",
    }
    assert authority["daemon_lane_binding"] == dict(lane)

    crossed = _task_authority()
    crossed["owner_session_id"] = "session:another-lane"
    with pytest.raises(
        runtime.EAAEFBootstrapRuntimeGatewayError,
        match="differs from its daemon lane",
    ):
        runtime.eaaef_task_operation_authority_projection(crossed, verified_capability=capability)
    same_as_owner = _lane(lane_session_id="session:eaaef-owner-v7")
    with pytest.raises(
        runtime.EAAEFBootstrapRuntimeGatewayError,
        match="differs from the verified gateway owner",
    ):
        runtime.eaaef_daemon_lane_binding_projection(same_as_owner, verified_capability=capability)


def test_dead_lane_recovery_is_exact_bounded_and_read_only_shaped() -> None:
    capability = _verified_capability()
    arguments = runtime.eaaef_dead_lane_recovery_arguments(
        lane_bindings=[_lane()],
        limit=100,
        now_ms=1_800_000_000_000,
        verified_capability=capability,
    )
    assert set(arguments) == {"recovery_authority"}
    recovery = arguments["recovery_authority"]
    assert set(recovery) == {
        "schema",
        "purpose",
        "lane_bindings",
        "limit",
        "now_ms",
    }
    assert recovery["schema"] == EAAEF_DEAD_LANE_RECOVERY_AUTHORITY_SCHEMA
    assert recovery["purpose"] == "expired_lane_retirement"

    with pytest.raises(
        runtime.EAAEFBootstrapRuntimeGatewayError,
        match="duplicate lane_session_id",
    ):
        runtime.eaaef_dead_lane_recovery_arguments(
            lane_bindings=[
                _lane(),
                _lane(
                    lane_generation=8,
                    process_instance_id="process:eaaef-lane-0-v2",
                ),
            ],
            limit=100,
            now_ms=1_800_000_000_000,
            verified_capability=capability,
        )
    with pytest.raises(
        runtime.EAAEFBootstrapRuntimeGatewayError,
        match="one and five",
    ):
        runtime.eaaef_dead_lane_recovery_arguments(
            lane_bindings=[],
            limit=100,
            now_ms=1_800_000_000_000,
            verified_capability=capability,
        )


def test_proxy_surface_is_present_but_cannot_dispatch_before_qualification() -> None:
    methods = {
        "bind_daemon",
        "list_running_attempts",
        "list_expired_running_attempts",
        "record_event",
        "ensure_attempt",
        "get_attempt",
        "commit_phase",
        "commit_reconciled_attempt",
        "phase_history",
        "get_idempotent_result",
        "record_idempotent_result",
        "reserve_provider",
        "commit_provider",
        "reserve_effect",
        "commit_effect",
        "record_validation",
    }
    assert all(
        callable(getattr(runtime.EAAEFBootstrapExecutionRepositoryProxy, name, None))
        for name in methods
    )
    assert "_PROXY_FACTORY_TOKEN" not in vars(runtime)


def test_duck_typed_proxy_is_rejected() -> None:
    class Impostor:
        EAAEF_INTERFACE = runtime.EAAEF_BOOTSTRAP_EXECUTION_REPOSITORY_PROXY_INTERFACE
        INTERFACE = runtime.EAAEF_BOOTSTRAP_EXECUTION_REPOSITORY_PROXY_INTERFACE

        def list_expired_running_attempts(self, **_kwargs: object) -> list[object]:
            return []

    with pytest.raises(
        runtime.EAAEFBootstrapRuntimeGatewayError,
        match="exact EAAEF execution repository proxy",
    ):
        runtime.require_eaaef_bootstrap_execution_repository_proxy(Impostor())
