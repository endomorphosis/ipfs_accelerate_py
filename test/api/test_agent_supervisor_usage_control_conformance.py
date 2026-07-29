"""ASI-169: Python / CLI / MCP usage-governance surface conformance."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.control.control_cli import (
    USAGE_CLI_COMMANDS,
    run_usage_cli,
    usage_cli_discovery_manifest,
)
from ipfs_accelerate_py.agent_supervisor.control.control_contracts import (
    SUPERVISOR_USAGE_ADMIN_AUTHORITY,
    SUPERVISOR_USAGE_BUDGET_AUTHORITY,
    SUPERVISOR_USAGE_CONTROL_REQUIREMENT_ID,
    SUPERVISOR_USAGE_CORRECTION_AUTHORITY,
    SUPERVISOR_USAGE_POLICY_AUTHORITY,
    SUPERVISOR_USAGE_READ_AUTHORITY,
    SUPERVISOR_USAGE_RESET_AUTHORITY,
    USAGE_CONTROL_MUTATION_OPERATIONS,
    USAGE_CONTROL_READ_OPERATIONS,
    SupervisorUsageControlOperation,
    discover_usage_control_catalog,
    usage_control_operations,
)
from ipfs_accelerate_py.agent_supervisor.control.control_plane import (
    ProviderUsageControl,
)
from ipfs_accelerate_py.mcp_server.tools.agent_supervisor_tools import (
    native_agent_supervisor_tools as native_tools,
)


class _Args:
    def __init__(self, **values: Any) -> None:
        self.__dict__.update(values)


class _RecordingToolManager:
    def __init__(self) -> None:
        self.definitions: list[dict[str, Any]] = []

    def register_tool(self, **definition: Any) -> None:
        self.definitions.append(definition)


def _controller() -> ProviderUsageControl:
    return ProviderUsageControl(
        catalog_revision_provider=lambda: "catalog-rev-conformance",
        supervisor_revision_provider=lambda: "supervisor-rev-conformance",
        policy_revision_provider=lambda: "policy-rev-conformance",
    )


def _read_auth() -> list[str]:
    return [SUPERVISOR_USAGE_READ_AUTHORITY]


def _admin_auth() -> list[str]:
    return [
        SUPERVISOR_USAGE_READ_AUTHORITY,
        SUPERVISOR_USAGE_ADMIN_AUTHORITY,
        SUPERVISOR_USAGE_BUDGET_AUTHORITY,
        SUPERVISOR_USAGE_POLICY_AUTHORITY,
        SUPERVISOR_USAGE_CORRECTION_AUTHORITY,
        SUPERVISOR_USAGE_RESET_AUTHORITY,
    ]


def _params_for(operation: SupervisorUsageControlOperation) -> dict[str, Any]:
    if operation in USAGE_CONTROL_READ_OPERATIONS:
        if operation is SupervisorUsageControlOperation.HEADROOM:
            return {"target_id": "scope:conformance"}
        if operation is SupervisorUsageControlOperation.RESERVATIONS:
            return {"target_id": "scope:conformance"}
        if operation is SupervisorUsageControlOperation.ROUTE_PREVIEW:
            return {
                "candidates": [
                    {
                        "binding_id": "binding:c",
                        "provider_id": "provider:c",
                        "scope_id": "scope:conformance",
                    }
                ]
            }
        return {}
    common = {
        "target_id": "scope:conformance",
        "lease_id": "lease:conformance",
        "fence": 1,
        "expected_effects": [operation.value],
        "actor": "operator:conformance",
        "source": "operator",
    }
    if operation is SupervisorUsageControlOperation.SET_BUDGET:
        return {
            **common,
            "budget": {"limits": [{"dimension": "requests", "ceiling": 10}]},
        }
    if operation is SupervisorUsageControlOperation.SET_POLICY:
        return {**common, "policy": {"mode": "observe"}}
    if operation is SupervisorUsageControlOperation.CORRECT:
        return {
            **common,
            "supersedes_event_id": "event:conformance",
            "units": {"entries": []},
        }
    if operation is SupervisorUsageControlOperation.RESET:
        return {**common}
    raise AssertionError(operation)


def test_discovery_populations_agree_across_python_cli_mcp() -> None:
    python_catalog = discover_usage_control_catalog()
    cli_manifest = usage_cli_discovery_manifest()
    mcp_manifest = native_tools.usage_mcp_discovery_manifest()

    assert python_catalog["requirement_id"] == SUPERVISOR_USAGE_CONTROL_REQUIREMENT_ID
    assert cli_manifest["requirement_id"] == SUPERVISOR_USAGE_CONTROL_REQUIREMENT_ID
    assert mcp_manifest["requirement_id"] == SUPERVISOR_USAGE_CONTROL_REQUIREMENT_ID
    assert set(python_catalog["operations"][0].keys())  # non-empty descriptors
    py_ops = {item["operation"] for item in python_catalog["operations"]}
    assert py_ops == set(usage_control_operations())
    assert set(cli_manifest["operations"]) == py_ops
    assert set(mcp_manifest["operations"]) == py_ops
    assert set(USAGE_CLI_COMMANDS.values()) >= py_ops | {"discover"}
    assert "discover" in USAGE_CLI_COMMANDS.values() or "usage-discover" in USAGE_CLI_COMMANDS

    manager = _RecordingToolManager()
    native_tools.register_native_agent_supervisor_usage_tools(manager)
    names = {item["name"] for item in manager.definitions}
    assert "agent_supervisor_usage" in names
    usage_tool = next(
        item for item in manager.definitions if item["name"] == "agent_supervisor_usage"
    )
    enum_ops = set(usage_tool["input_schema"]["properties"]["operation"]["enum"])
    assert py_ops.issubset(enum_ops)
    assert "discover" in enum_ops


def test_every_read_operation_is_schema_result_error_equivalent() -> None:
    controller = _controller()
    native_tools.set_provider_usage_control_service(controller)
    try:
        for operation in sorted(
            USAGE_CONTROL_READ_OPERATIONS, key=lambda item: item.value
        ):
            params = _params_for(operation)
            python_result = controller.execute(
                operation, authorities=_read_auth(), **params
            )
            assert python_result["success"] is True, (operation, python_result)
            assert python_result["operation"] == operation.value
            assert python_result["catalog_revision"] == "catalog-rev-conformance"
            assert python_result["usage_revision"]
            assert python_result["policy_revision"]
            assert python_result["supervisor_revision"]
            assert python_result["completion_authoritative"] is False

            # CLI adapter
            args = _Args(
                agent_usage_operation=operation.value,
                authorities_json=json.dumps(_read_auth()),
                target_id=params.get("target_id"),
                limit=params.get("limit", 50),
                cursor=None,
                parameters_json=json.dumps(
                    {k: v for k, v in params.items() if k != "target_id"}
                )
                if any(k != "target_id" for k in params)
                else None,
                output_json=True,
            )
            stream_out: list[str] = []
            stream_err: list[str] = []

            class _Stream:
                def __init__(self, sink: list[str]) -> None:
                    self._sink = sink

                def write(self, data: str) -> int:
                    self._sink.append(data)
                    return len(data)

            code = run_usage_cli(
                args,
                usage_control=controller,
                stdout=_Stream(stream_out),  # type: ignore[arg-type]
                stderr=_Stream(stream_err),  # type: ignore[arg-type]
            )
            assert code == 0, (operation, "".join(stream_err))
            cli_result = json.loads("".join(stream_out))
            assert cli_result["success"] is True
            assert cli_result["operation"] == python_result["operation"]
            assert cli_result["catalog_revision"] == python_result["catalog_revision"]
            assert cli_result["completion_authoritative"] is False

            mcp_result = asyncio.run(
                native_tools.agent_supervisor_usage(
                    operation.value,
                    authorities=_read_auth(),
                    target_id=params.get("target_id"),
                    parameters={
                        k: v for k, v in params.items() if k != "target_id"
                    },
                )
            )
            assert mcp_result["success"] is True
            assert mcp_result["operation"] == python_result["operation"]
            assert mcp_result["catalog_revision"] == python_result["catalog_revision"]
            assert mcp_result["completion_authoritative"] is False
    finally:
        native_tools.set_provider_usage_control_service(None)


def test_mutation_operations_share_guardrails_across_transports() -> None:
    controller = _controller()
    native_tools.set_provider_usage_control_service(controller)
    try:
        for operation in sorted(
            USAGE_CONTROL_MUTATION_OPERATIONS, key=lambda item: item.value
        ):
            # Fresh revision for each mutation.
            params = _params_for(operation)
            params["expected_usage_revision"] = controller.usage_revision()
            params["idempotency_key"] = f"idem:{operation.value}"
            python_result = controller.execute(
                operation, authorities=_admin_auth(), **params
            )
            assert python_result["success"] is True, (operation, python_result)
            assert python_result["audit"]["operation"] == operation.value
            assert python_result["audit"]["lease_id"] == "lease:conformance"
            assert python_result["completion_authoritative"] is False

            # Idempotent replay
            replay = controller.execute(
                operation, authorities=_admin_auth(), **params
            )
            assert replay["success"] is True
            assert "idempotency_replay" in replay.get("reason_codes", [])

            # Distinct authority required (read alone fails)
            denied_params = dict(params)
            denied_params["idempotency_key"] = f"denied:{operation.value}"
            denied_params["expected_usage_revision"] = controller.usage_revision()
            denied = controller.execute(
                operation, authorities=_read_auth(), **denied_params
            )
            assert denied["success"] is False
            assert denied["error_code"] in {
                "budget_authority_denied",
                "policy_authority_denied",
                "correction_authority_denied",
                "reset_authority_denied",
                "admin_denied",
            }

            mcp_params = {
                k: v
                for k, v in params.items()
                if k
                not in {
                    "target_id",
                    "idempotency_key",
                    "expected_usage_revision",
                }
            }
            mcp_params["expected_usage_revision"] = controller.usage_revision()
            mcp_params["idempotency_key"] = f"mcp:{operation.value}"
            mcp_result = asyncio.run(
                native_tools.agent_supervisor_usage(
                    operation.value,
                    authorities=_admin_auth(),
                    target_id=params["target_id"],
                    budget=True,
                    policy=True,
                    correction=True,
                    reset=True,
                    admin=True,
                    parameters=mcp_params,
                )
            )
            assert mcp_result["success"] is True, (operation, mcp_result)
            assert mcp_result["audit"]["operation"] == operation.value
    finally:
        native_tools.set_provider_usage_control_service(None)


def test_reads_cannot_reserve_refresh_probe_invoke_or_mutate() -> None:
    controller = _controller()
    before = controller.usage_revision()
    for operation in USAGE_CONTROL_READ_OPERATIONS:
        params = _params_for(operation)
        result = controller.execute(operation, authorities=_read_auth(), **params)
        assert result["success"] is True
        assert result.get("reserved", False) is False or "reserved" not in result or result["reserved"] is False
        if "invoked" in result:
            assert result["invoked"] is False
        if "probed" in result:
            assert result["probed"] is False
        if "refreshed" in result:
            assert result["refreshed"] is False
    assert controller.usage_revision() == before


def test_model_and_peer_sources_cannot_mutate_provider_truth() -> None:
    controller = _controller()
    rev = controller.usage_revision()
    for source, code in (
        ("model_output", "mutation_denied_model_output"),
        ("remote_peer", "mutation_denied_remote_peer"),
    ):
        result = controller.set_budget(
            "scope:x",
            authorities=_admin_auth(),
            expected_usage_revision=rev,
            idempotency_key=f"src:{source}",
            lease_id="lease",
            fence=1,
            expected_effects=["set_budget"],
            source=source,
            budget={"limits": [{"dimension": "requests", "ceiling": 1}]},
        )
        assert result["success"] is False
        assert result["error_code"] == code
