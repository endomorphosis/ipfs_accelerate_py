from __future__ import annotations

import pytest
from ipfs_accelerate_py.agent_supervisor.task_sources.eaaef_bootstrap_daemon_gateway import (
    EAAEF_BOOTSTRAP_DAEMON_ADMITTED_OPERATIONS,
    EAAEF_BOOTSTRAP_DAEMON_COMPONENT_OPERATIONS,
    EAAEF_BOOTSTRAP_DAEMON_EXCLUDED_OPERATIONS,
    EAAEF_BOOTSTRAP_DAEMON_MISSING_OPERATIONS,
    EAAEF_BOOTSTRAP_DAEMON_OPERATIONS,
    EAAEF_BOOTSTRAP_DAEMON_PRODUCTION_BLOCKER,
    EAAEF_BOOTSTRAP_DAEMON_QUALIFICATION_STATUS,
    EAAEFBootstrapDaemonCapability,
    EAAEFBootstrapDaemonGateway,
    EAAEFBootstrapDaemonGatewayError,
    EAAEFBootstrapDaemonOperationNoGo,
    eaaef_bootstrap_daemon_operation_dispositions,
    require_eaaef_bootstrap_daemon_gateway,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_daemon_gateway import (
    REQUIRED_QUACK_DAEMON_OPERATIONS,
    QuackDaemonCommandGateway,
    QuackDaemonGatewayCapability,
    QuackDaemonGatewayError,
    quack_daemon_operation_command_vocabulary,
)


def _sha(token: str) -> str:
    return "sha256:" + token * 64


def _capability(**overrides: object) -> EAAEFBootstrapDaemonCapability:
    values: dict[str, object] = {
        "board_namespace": "external-agent-autonomous-execution-fabric-v1",
        "shard_id": "eaaef-control-shard",
        "store_id": "eaaef-control-run-v5",
        "owner_principal_did": "did:key:zBootstrapOwner",
        "owner_generation": 3,
        "fence_epoch": 5,
        "authorization_policy_cid": _sha("a"),
        "command_fabric_qualification_cid": _sha("b"),
    }
    values.update(overrides)
    return EAAEFBootstrapDaemonCapability(**values)  # type: ignore[arg-type]


def _generic_capability() -> QuackDaemonGatewayCapability:
    return QuackDaemonGatewayCapability(
        board_namespace="external-agent-autonomous-execution-fabric-v1",
        shard_id="eaaef-control-shard",
        store_id="eaaef-control-run-v5",
        control_plane_schema_version="5",
        state_schema_revision="datasets-authoritative-operational-v1",
        command_endpoint="quack:127.0.0.1:19494",
        state_endpoint="quack:127.0.0.1:19495",
        owner_principal_did="did:key:zBootstrapOwner",
        owner_generation=3,
        fence_epoch=5,
        authorization_policy_cid=_sha("a"),
        command_fabric_qualification_cid=_sha("b"),
    )


def test_bootstrap_v1_is_exact_31_operation_subset_with_six_components() -> None:
    generic_before = dict(quack_daemon_operation_command_vocabulary())
    dispositions = eaaef_bootstrap_daemon_operation_dispositions()
    assert len(generic_before) == 39
    assert frozenset(generic_before) == REQUIRED_QUACK_DAEMON_OPERATIONS
    assert len(EAAEF_BOOTSTRAP_DAEMON_OPERATIONS) == 31
    assert len(EAAEF_BOOTSTRAP_DAEMON_EXCLUDED_OPERATIONS) == 8
    assert len(EAAEF_BOOTSTRAP_DAEMON_MISSING_OPERATIONS) == 29
    assert (
        EAAEF_BOOTSTRAP_DAEMON_QUALIFICATION_STATUS
        == "29_borrowed_transaction_handlers_missing_fail_closed"
    )
    assert EAAEF_BOOTSTRAP_DAEMON_ADMITTED_OPERATIONS == {
        "task.get",
        "task.ready",
    }
    assert (
        EAAEF_BOOTSTRAP_DAEMON_OPERATIONS | EAAEF_BOOTSTRAP_DAEMON_EXCLUDED_OPERATIONS
        == REQUIRED_QUACK_DAEMON_OPERATIONS
    )
    assert not (
        EAAEF_BOOTSTRAP_DAEMON_OPERATIONS & EAAEF_BOOTSTRAP_DAEMON_EXCLUDED_OPERATIONS
    )
    assert set(EAAEF_BOOTSTRAP_DAEMON_COMPONENT_OPERATIONS) == {
        "task",
        "coordination",
        "execution",
        "provider",
        "effect",
        "validation",
    }
    assert (
        frozenset().union(*EAAEF_BOOTSTRAP_DAEMON_COMPONENT_OPERATIONS.values())
        == EAAEF_BOOTSTRAP_DAEMON_OPERATIONS
    )
    assert frozenset(dispositions) == EAAEF_BOOTSTRAP_DAEMON_OPERATIONS
    assert {
        operation
        for operation, record in dispositions.items()
        if record["disposition"] == "admitted_owner_transaction"
    } == EAAEF_BOOTSTRAP_DAEMON_ADMITTED_OPERATIONS
    assert all(
        dispositions[operation]["disposition"] == "typed_no_go"
        and dispositions[operation]["reason_code"]
        and dispositions[operation]["borrowed_transaction_required"] is True
        for operation in EAAEF_BOOTSTRAP_DAEMON_MISSING_OPERATIONS
    )
    assert dict(quack_daemon_operation_command_vocabulary()) == generic_before


@pytest.mark.parametrize(
    ("field", "value", "detail"),
    [
        (
            "operations",
            EAAEF_BOOTSTRAP_DAEMON_OPERATIONS - {"task.get"},
            "missing=task.get",
        ),
        (
            "operations",
            EAAEF_BOOTSTRAP_DAEMON_OPERATIONS | {"merge.accept"},
            "extra=merge.accept",
        ),
        (
            "excluded_operations",
            EAAEF_BOOTSTRAP_DAEMON_EXCLUDED_OPERATIONS - {"task.list"},
            "missing=task.list",
        ),
        (
            "missing_borrowed_transaction_operations",
            EAAEF_BOOTSTRAP_DAEMON_MISSING_OPERATIONS - {"provider.reserve"},
            "missing=provider.reserve",
        ),
    ],
)
def test_bootstrap_capability_rejects_missing_extra_and_reclassified_operations(
    field: str,
    value: frozenset[str],
    detail: str,
) -> None:
    with pytest.raises(EAAEFBootstrapDaemonGatewayError, match=detail):
        _capability(**{field: value})


def test_bootstrap_gateway_rejects_excluded_and_missing_operations_typed() -> None:
    gateway = EAAEFBootstrapDaemonGateway(capability=_capability())
    gateway.require_operation("task.get")
    with pytest.raises(
        EAAEFBootstrapDaemonOperationNoGo,
        match=(
            "operation=provider.reserve;reason_code="
            "provider_reservation_before_container_launch_unqualified"
        ),
    ):
        gateway.require_operation("provider.reserve")
    for operation in EAAEF_BOOTSTRAP_DAEMON_EXCLUDED_OPERATIONS:
        with pytest.raises(
            EAAEFBootstrapDaemonGatewayError,
            match="explicitly excluded",
        ):
            gateway.disposition(operation)
    with pytest.raises(
        EAAEFBootstrapDaemonGatewayError,
        match="outside the closed 31-operation",
    ):
        gateway.disposition("authority.execute_sql")


def test_generic_and_bootstrap_capabilities_cannot_substitute_for_each_other() -> None:
    bootstrap = _capability()
    generic = _generic_capability()
    with pytest.raises(
        EAAEFBootstrapDaemonGatewayError,
        match="generic 39-operation capability cannot substitute",
    ):
        EAAEFBootstrapDaemonGateway(capability=generic)  # type: ignore[arg-type]
    with pytest.raises(QuackDaemonGatewayError, match="typed gateway capability"):
        QuackDaemonCommandGateway(
            capability=bootstrap,  # type: ignore[arg-type]
            task_source=object(),
            coordinator=object(),
            execution_repository=object(),
            merge_repository=object(),
            plan_repository=object(),
        )
    with pytest.raises(
        EAAEFBootstrapDaemonGatewayError,
        match="generic, Portal, and direct-file fallbacks",
    ):
        require_eaaef_bootstrap_daemon_gateway(
            object(), expected_capability_cid=bootstrap.content_id
        )


def test_bootstrap_gateway_is_zero_sidecar_noninjectable_and_production_false() -> None:
    capability = _capability()
    gateway = EAAEFBootstrapDaemonGateway(capability=capability)
    assert (
        require_eaaef_bootstrap_daemon_gateway(
            gateway, expected_capability_cid=capability.content_id
        )
        is gateway
    )
    evidence = gateway.evidence()
    assert evidence["production_admitted"] is False
    assert evidence["production_blockers"] == [
        EAAEF_BOOTSTRAP_DAEMON_PRODUCTION_BLOCKER
    ]
    assert len(evidence["missing_borrowed_transaction_operations"]) == 29
    forbidden = {
        "portal",
        "database_path",
        "connection",
        "execute",
        "execute_sql",
        "transaction",
        "owner_dispatcher",
        "dispatch",
    }
    assert all(not hasattr(gateway, name) for name in forbidden)
    for component_name in EAAEF_BOOTSTRAP_DAEMON_COMPONENT_OPERATIONS:
        component = getattr(gateway, component_name)
        assert component.gateway_binding_cid == capability.content_id
        assert (
            component.operations
            == EAAEF_BOOTSTRAP_DAEMON_COMPONENT_OPERATIONS[component_name]
        )
        assert all(not hasattr(component, name) for name in forbidden)
    with pytest.raises(
        EAAEFBootstrapDaemonGatewayError,
        match=EAAEF_BOOTSTRAP_DAEMON_PRODUCTION_BLOCKER,
    ):
        gateway.attach()
    with pytest.raises(
        EAAEFBootstrapDaemonGatewayError,
        match="enables unavailable authority: production_admitted",
    ):
        _capability(production_admitted=True)
    with pytest.raises(
        EAAEFBootstrapDaemonGatewayError,
        match="enables unavailable authority: direct_database_open,portal_fallback",
    ):
        _capability(direct_database_open=True, portal_fallback=True)
    with pytest.raises(
        EAAEFBootstrapDaemonGatewayError,
        match="opaque owner identity, not a database path",
    ):
        _capability(store_id="data/eaaef/control.duckdb")


def test_bootstrap_gateway_detects_capability_swap() -> None:
    capability = _capability()
    gateway = EAAEFBootstrapDaemonGateway(capability=capability)
    with pytest.raises(
        EAAEFBootstrapDaemonGatewayError,
        match="bound to another capability",
    ):
        require_eaaef_bootstrap_daemon_gateway(
            gateway,
            expected_capability_cid=_sha("f"),
        )
