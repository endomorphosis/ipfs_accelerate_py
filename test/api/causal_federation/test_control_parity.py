"""Parity and negative-path tests for federation CLI and MCP control adapters."""

from __future__ import annotations

import argparse
import ast
import inspect
import io
import json

import pytest
from ipfs_accelerate_py.agent_supervisor.federation import cli, contracts
from ipfs_accelerate_py.agent_supervisor.federation.control_service import (
    POST_ADMISSION_OPERATIONS,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
    canonical_json_bytes,
)
from ipfs_accelerate_py.mcp.tools import agent_supervisor as federation_mcp
from test.api.causal_federation.test_control_service import command, service


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    cli.register_federation_cli(parser.add_subparsers(dest="root_command"))
    return parser


def _transport_payload(value: contracts.FederationCommand) -> str:
    return canonical_json_bytes(value.to_dict()).decode("utf-8")


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


@pytest.mark.parametrize("module", (cli, federation_mcp))
def test_control_adapters_have_no_filesystem_or_direct_database_surface(
    module: object,
) -> None:
    tree = ast.parse(inspect.getsource(module))
    forbidden_imports = {"duckdb", "pathlib", "sqlite3"}
    forbidden_calls = {
        "Path",
        "connect",
        "cursor",
        "open",
        "read_bytes",
        "read_text",
        "sql",
        "write_bytes",
        "write_text",
    }

    imported_roots = {
        node.module.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    imported_roots.update(
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    called_names.update(
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    )
    option_strings = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value.startswith("--")
    }

    assert not imported_roots.intersection(forbidden_imports)
    assert not called_names.intersection(forbidden_calls)
    assert not {option for option in option_strings if "file" in option.casefold()}


def test_cli_parser_exposes_only_inline_command_json() -> None:
    parser = _parser()

    with pytest.raises(SystemExit) as rejected:
        parser.parse_args(["federation", "start", "--command-file", "/tmp/untrusted-command.json"])

    assert rejected.value.code == 2


def test_cli_and_mcp_return_the_same_canonical_result_and_audit() -> None:
    value = command()
    control, authorizer, owner = service()
    parser = _parser()
    args = parser.parse_args(["federation", "start", "--command-json", _transport_payload(value)])
    stdout = io.StringIO()
    stderr = io.StringIO()

    cli_status = cli.run_federation_cli(args, service=control, stdout=stdout, stderr=stderr)
    cli_record = json.loads(stdout.getvalue())
    federation_mcp.configure_federation_control(service=control)
    try:
        mcp_record = federation_mcp.execute_federation_control(value.to_dict(), value.operation)
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
    args = parser.parse_args(["federation", "start", "--command-json", _transport_payload(value)])
    stderr = io.StringIO()

    status = cli.run_federation_cli(args, service=control, stdout=io.StringIO(), stderr=stderr)

    assert status == cli.FEDERATION_CLI_EXIT_INVALID
    assert "trigger gateway" in json.loads(stderr.getvalue())["message"]
    assert not authorizer.commands
    assert not owner.calls


@pytest.mark.parametrize(
    "payload",
    (
        '{"schema":',
        " " + _transport_payload(command()),
        "{}" + (" " * cli.FEDERATION_CLI_MAX_COMMAND_JSON_BYTES),
    ),
    ids=("malformed", "noncanonical", "oversized"),
)
def test_cli_rejects_malformed_noncanonical_or_oversized_inline_command_before_dispatch(
    payload: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    control, authorizer, owner = service()
    args = _parser().parse_args(["federation", "start", "--command-json", payload])
    stderr = io.StringIO()
    dispatched = False

    def fail_if_dispatched(*_args: object, **_kwargs: object) -> object:
        nonlocal dispatched
        dispatched = True
        raise AssertionError("malformed CLI command reached the control service")

    monkeypatch.setattr(cli, "execute_federation_command", fail_if_dispatched)
    status = cli.run_federation_cli(args, service=control, stdout=io.StringIO(), stderr=stderr)

    assert status == cli.FEDERATION_CLI_EXIT_INVALID
    assert json.loads(stderr.getvalue())["status"] == "invalid_request"
    assert dispatched is False
    assert not authorizer.commands
    assert not owner.calls


@pytest.mark.parametrize(
    ("field", "payload"),
    [
        ("database_path", "/tmp/control.duckdb"),
        ("raw_credential", "Bearer secret-value-that-must-not-dispatch"),
    ],
)
def test_mcp_rejects_unsafe_payload_before_service_resolution(field: str, payload: str) -> None:
    value = command().to_dict()
    value[field] = payload
    initial = federation_mcp.federation_control_service_resolution_count()
    federation_mcp.configure_federation_control()

    with pytest.raises(contracts.UnknownNormativeFieldError):
        federation_mcp.execute_federation_control(value, contracts.FederationOperation.START)

    assert federation_mcp.federation_control_service_resolution_count() == initial


def test_mcp_rejects_malformed_command_before_service_resolution() -> None:
    initial = federation_mcp.federation_control_service_resolution_count()
    factory_calls: list[contracts.FederationCommand] = []

    def factory(value: contracts.FederationCommand) -> object:
        factory_calls.append(value)
        raise AssertionError("malformed MCP command reached service resolution")

    federation_mcp.configure_federation_control(service_factory=factory)
    try:
        with pytest.raises(contracts.FederationContractError):
            federation_mcp.execute_federation_control(
                {"schema": contracts.FederationCommand.SCHEMA},
                contracts.FederationOperation.START,
            )
    finally:
        federation_mcp.configure_federation_control()

    assert federation_mcp.federation_control_service_resolution_count() == initial
    assert not factory_calls


def test_mcp_rejects_create_before_service_resolution() -> None:
    value = command(operation=contracts.FederationOperation.CREATE)
    initial = federation_mcp.federation_control_service_resolution_count()
    federation_mcp.configure_federation_control()

    with pytest.raises(
        federation_mcp.FederationControlMCPConfigurationError, match="trigger gateway"
    ):
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
        call["input_schema"]["x-federation-operation"]
        in {operation.value for operation in POST_ADMISSION_OPERATIONS}
        for call in host.calls
    )
