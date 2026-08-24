"""Parity and negative-path tests for federation CLI and MCP control adapters."""

from __future__ import annotations

import argparse
import io
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.federation import cli
from ipfs_accelerate_py.agent_supervisor.federation import contracts
from ipfs_accelerate_py.agent_supervisor.federation.control_service import (
    POST_ADMISSION_OPERATIONS,
)
from ipfs_accelerate_py.mcp.tools import agent_supervisor as federation_mcp
from test.api.causal_federation.test_control_service import command, service


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    cli.register_federation_cli(parser.add_subparsers(dest="root_command"))
    return parser


def _transport_payload(value: contracts.FederationCommand) -> str:
    return json.dumps(value.to_dict(), sort_keys=True)


def _without_transport_identity(record: dict[str, object]) -> dict[str, object]:
    copied = dict(record)
    copied.pop("schema")
    copied.pop("interface")
    return copied


def test_cli_and_mcp_publish_exact_post_admission_catalogs() -> None:
    cli_manifest = cli.federation_cli_discovery_manifest()
    mcp_manifest = federation_mcp.federation_control_mcp_discovery_manifest()
    expected = {operation.value for operation in POST_ADMISSION_OPERATIONS}

    assert set(cli_manifest["commands"].values()) == expected
    assert set(mcp_manifest["tools"].values()) == expected
    assert contracts.FederationOperation.CREATE.value not in cli_manifest["commands"].values()
    assert contracts.FederationOperation.CREATE.value not in mcp_manifest["tools"].values()
    assert cli_manifest["shell_out"] is False
    assert mcp_manifest["embedded_fallback"] is False


def test_cli_and_mcp_return_the_same_canonical_result_and_audit() -> None:
    value = command()
    control, authorizer, owner = service()
    parser = _parser()
    args = parser.parse_args(
        ["federation", "start", "--command-json", _transport_payload(value)]
    )
    stdout = io.StringIO()
    stderr = io.StringIO()

    cli_status = cli.run_federation_cli(
        args, service=control, stdout=stdout, stderr=stderr
    )
    cli_record = json.loads(stdout.getvalue())
    federation_mcp.configure_federation_control(service=control)
    try:
        mcp_record = federation_mcp.execute_federation_control(
            value.to_dict(), value.operation
        )
    finally:
        federation_mcp.configure_federation_control()

    assert cli_status == cli.FEDERATION_CLI_EXIT_SUCCESS
    assert not stderr.getvalue()
    assert _without_transport_identity(cli_record) == _without_transport_identity(mcp_record)
    assert cli_record["command"] == value.to_dict()
    assert cli_record["audit"]["command_cid"] == value.cid
    assert mcp_record["result"]["evidence_refs"] == [value.cid]
    assert len(authorizer.commands) == 2
    assert len(owner.calls) == 2


def test_cli_rejects_create_before_authorization_or_state_owner_dispatch() -> None:
    value = command(operation=contracts.FederationOperation.CREATE)
    control, authorizer, owner = service()
    parser = _parser()
    # ``start`` is intentionally used to prove that the decoded command is
    # checked, not inferred from an untrusted CLI name.
    args = parser.parse_args(
        ["federation", "start", "--command-json", _transport_payload(value)]
    )
    stderr = io.StringIO()

    status = cli.run_federation_cli(args, service=control, stdout=io.StringIO(), stderr=stderr)

    assert status == cli.FEDERATION_CLI_EXIT_INVALID
    assert "trigger gateway" in json.loads(stderr.getvalue())["message"]
    assert not authorizer.commands
    assert not owner.calls


@pytest.mark.parametrize(
    ("field", "payload"),
    [
        ("database_path", "/tmp/control.duckdb"),
        ("raw_credential", "Bearer secret-value-that-must-not-dispatch"),
    ],
)
def test_mcp_rejects_unsafe_payload_before_service_resolution(
    field: str, payload: str
) -> None:
    value = command().to_dict()
    value[field] = payload
    initial = federation_mcp.federation_control_service_resolution_count()
    federation_mcp.configure_federation_control()

    with pytest.raises(contracts.UnknownNormativeFieldError):
        federation_mcp.execute_federation_control(value, contracts.FederationOperation.START)

    assert federation_mcp.federation_control_service_resolution_count() == initial


def test_mcp_rejects_create_before_service_resolution() -> None:
    value = command(operation=contracts.FederationOperation.CREATE)
    initial = federation_mcp.federation_control_service_resolution_count()
    federation_mcp.configure_federation_control()

    with pytest.raises(federation_mcp.FederationControlMCPConfigurationError, match="trigger gateway"):
        federation_mcp.execute_federation_control(value.to_dict(), value.operation)

    assert federation_mcp.federation_control_service_resolution_count() == initial


def test_mcp_fails_closed_without_a_configured_service() -> None:
    value = command()
    federation_mcp.configure_federation_control()

    with pytest.raises(federation_mcp.FederationControlMCPConfigurationError):
        federation_mcp.execute_federation_control(value.to_dict(), value.operation)


def test_cli_fails_closed_without_a_configured_service() -> None:
    value = command()
    args = _parser().parse_args(
        ["federation", "start", "--command-json", _transport_payload(value)]
    )
    stderr = io.StringIO()

    status = cli.run_federation_cli(args, stdout=io.StringIO(), stderr=stderr)

    assert status == cli.FEDERATION_CLI_EXIT_INVALID
    assert "qualified FederationControlService" in json.loads(stderr.getvalue())["message"]


def test_mcp_registration_has_exact_closed_tool_population() -> None:
    class Host:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def register_tool(self, **kwargs: object) -> None:
            self.calls.append(dict(kwargs))

    host = Host()
    federation_mcp.register_federation_control_tools(host)

    assert {str(call["name"]) for call in host.calls} == {
        tool.__name__ for tool in federation_mcp.FEDERATION_CONTROL_OPERATION_TOOLS.values()
    }
    assert all(
        call["input_schema"]["x-federation-operation"] in {
            operation.value for operation in POST_ADMISSION_OPERATIONS
        }
        for call in host.calls
    )
