"""Parity and negative-path tests for federation CLI and MCP control adapters."""

from __future__ import annotations

import argparse
import ast
import inspect
import io
import json
import os
import subprocess
import sys
from dataclasses import replace

import anyio
import pytest
from ipfs_accelerate_py.agent_supervisor.federation import cli, contracts
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
    canonical_json_bytes,
)
from ipfs_accelerate_py.mcp.tools import agent_supervisor as federation_mcp
from ipfs_accelerate_py.mcp_server.hierarchical_tool_manager import (
    HierarchicalToolManager,
)
from ipfs_accelerate_py.mcp_server.server import configure_agent_supervisor_tools
from test.api.causal_federation.test_contracts import sample_binding
from test.api.causal_federation.test_control_service import command, service
from test.api.causal_federation.test_trigger import (
    gateway_for,
    sample_authentication,
    sample_request,
)

from ipfs_accelerate_py import cli as product_cli


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    cli.register_federation_cli(parser.add_subparsers(dest="root_command"))
    return parser


def _transport_payload(value: contracts.ClosedContract) -> str:
    return canonical_json_bytes(value.to_dict()).decode("utf-8")


def _create_transport() -> tuple[cli.FederationCreateTransport, object]:
    request = sample_request()
    transport = cli.FederationCreateTransport(
        request=request,
        authentication=sample_authentication(request),
    )
    gateway, _budgets, _store = gateway_for(request)
    return transport, gateway


def _large_binding() -> contracts.FederationBinding:
    def identities(prefix: str) -> tuple[str, ...]:
        return tuple(
            f"{prefix}:{index:03d}:" + ("a" * 300) for index in range(256)
        )

    return sample_binding(
        repository_ids=identities("repo"),
        repository_tree_ids=identities("tree"),
        semantic_state_roots=identities("semantic"),
    )


def _large_command() -> contracts.FederationCommand:
    return replace(command(), binding=_large_binding())


def _without_transport_identity(record: dict[str, object]) -> dict[str, object]:
    copied = dict(record)
    copied.pop("schema")
    copied.pop("interface")
    return copied


def _assert_transport_error(record: dict[str, object], *, invalid: bool) -> None:
    assert record["status"] == ("invalid_request" if invalid else "unavailable")
    assert record["code"] == (
        cli.FEDERATION_TRANSPORT_INVALID_CODE
        if invalid
        else cli.FEDERATION_TRANSPORT_UNAVAILABLE_CODE
    )
    assert record["message"] == (
        cli.FEDERATION_TRANSPORT_INVALID_MESSAGE
        if invalid
        else cli.FEDERATION_TRANSPORT_UNAVAILABLE_MESSAGE
    )


def test_cli_and_mcp_publish_the_exact_closed_federation_catalog() -> None:
    cli_manifest = cli.federation_cli_discovery_manifest()
    mcp_manifest = federation_mcp.federation_control_mcp_discovery_manifest()
    expected = {operation.value for operation in contracts.FederationOperation}

    assert set(cli_manifest["commands"].values()) == expected
    assert set(mcp_manifest["tools"].values()) == expected
    assert cli_manifest["request_schemas"] == mcp_manifest["request_schemas"]
    assert cli_manifest["dispatch"] == mcp_manifest["dispatch"]
    assert cli_manifest["max_canonical_bytes"] == mcp_manifest["max_canonical_bytes"]
    assert cli_manifest["max_canonical_bytes"] == (
        cli.FEDERATION_CONTROL_MAX_CANONICAL_BYTES
    )
    assert cli_manifest["create_via_trigger_gateway"] == "FederationControlGateway"
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


@pytest.mark.parametrize("option", ("--command-file", "--request-file"))
def test_cli_parser_exposes_only_inline_canonical_json(option: str) -> None:
    with pytest.raises(SystemExit) as rejected:
        _parser().parse_args(
            ["federation", "start", option, "/tmp/untrusted-command.json"]
        )

    assert rejected.value.code == 2


def test_real_unified_cli_publishes_cold_federation_help() -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.getcwd() + (
        os.pathsep + environment["PYTHONPATH"]
        if environment.get("PYTHONPATH")
        else ""
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "ipfs_accelerate_py.cli",
            "agent",
            "federation",
            "--help",
        ],
        cwd=os.getcwd(),
        env=environment,
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "federation.create" in completed.stdout
    assert "federation.start" in completed.stdout
    assert "--command-file" not in completed.stdout
    assert "--request-file" not in completed.stdout


def test_real_unified_cli_discovery_starts_no_process_or_database() -> None:
    script = r'''
import json
import subprocess
import sys
from ipfs_accelerate_py import cli

starts = 0
def forbidden_popen(*args, **kwargs):
    global starts
    starts += 1
    raise AssertionError("cold federation discovery started a process")
subprocess.Popen = forbidden_popen
try:
    cli.main(["agent", "federation", "--help"])
except SystemExit as exc:
    exit_code = int(exc.code or 0)
from ipfs_accelerate_py.mcp.tools import agent_supervisor as federation_mcp
print(json.dumps({
    "exit_code": exit_code,
    "process_starts": starts,
    "duckdb_modules": sorted(
        name for name in sys.modules if name == "duckdb" or name.startswith("duckdb.")
    ),
    "service_resolutions": federation_mcp.federation_control_service_resolution_count(),
    "gateway_resolutions": federation_mcp.federation_control_gateway_resolution_count(),
}, sort_keys=True))
'''
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=os.getcwd(),
        capture_output=True,
        text=True,
        timeout=45,
        check=True,
    )
    observation = json.loads(completed.stdout.splitlines()[-1])

    assert observation == {
        "duckdb_modules": [],
        "exit_code": 0,
        "gateway_resolutions": 0,
        "process_starts": 0,
        "service_resolutions": 0,
    }


def test_real_unified_cli_dispatches_to_the_injected_federation_service(
    capsys: pytest.CaptureFixture[str],
) -> None:
    value = command()
    control, authorizer, owner = service()

    status = product_cli.main(
        [
            "agent",
            "federation",
            "start",
            "--command-json",
            _transport_payload(value),
            "--output-json",
        ],
        federation_control_service=control,
    )
    captured = capsys.readouterr()

    assert status == cli.FEDERATION_CLI_EXIT_SUCCESS
    assert json.loads(captured.out)["command"] == value.to_dict()
    assert not captured.err
    assert len(authorizer.commands) == 1
    assert len(owner.calls) == 1


def test_real_unified_cli_dispatches_create_to_the_authenticated_gateway(
    capsys: pytest.CaptureFixture[str],
) -> None:
    transport, gateway = _create_transport()

    status = product_cli.main(
        [
            "agent",
            "federation",
            "create",
            "--request-json",
            _transport_payload(transport),
            "--output-json",
        ],
        federation_gateway=gateway,
    )
    captured = capsys.readouterr()
    record = json.loads(captured.out)

    assert status == cli.FEDERATION_CLI_EXIT_SUCCESS
    assert record["identity"]["record_id"] == f"federation:{transport.request.cid}"
    assert record["authentication_evidence_ref"] == transport.authentication.cid
    assert "authentication" not in record
    assert not captured.err


@pytest.mark.parametrize(
    "payload",
    (
        '{"schema":',
        " " + _transport_payload(command()),
    ),
    ids=("malformed", "noncanonical"),
)
def test_cli_rejects_malformed_or_noncanonical_json_before_dispatch(
    payload: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _parser().parse_args(
        ["federation", "start", "--command-json", payload]
    )
    control, authorizer, owner = service()
    dispatched = False

    def forbidden_dispatch(*_args: object, **_kwargs: object) -> object:
        nonlocal dispatched
        dispatched = True
        raise AssertionError("invalid command reached service dispatch")

    monkeypatch.setattr(cli, "execute_federation_command", forbidden_dispatch)
    stderr = io.StringIO()
    status = cli.run_federation_cli(
        args,
        service=control,
        stdout=io.StringIO(),
        stderr=stderr,
    )

    assert status == cli.FEDERATION_CLI_EXIT_INVALID
    _assert_transport_error(json.loads(stderr.getvalue()), invalid=True)
    assert not dispatched
    assert not authorizer.commands
    assert not owner.calls


def test_malformed_mcp_command_is_rejected_before_service_resolution() -> None:
    service_before = federation_mcp.federation_control_service_resolution_count()
    factory_calls: list[object] = []

    def forbidden_factory(decoded: object) -> object:
        factory_calls.append(decoded)
        raise AssertionError("malformed command reached service resolution")

    federation_mcp.configure_federation_control(
        service_factory=forbidden_factory,
    )
    try:
        record = federation_mcp.execute_federation_control(
            {"schema": contracts.FederationCommand.SCHEMA},
            contracts.FederationOperation.START,
        )
    finally:
        federation_mcp.configure_federation_control()

    _assert_transport_error(record, invalid=True)
    assert federation_mcp.federation_control_service_resolution_count() == service_before
    assert not factory_calls


def test_cli_mcp_and_python_share_post_admission_contract_and_large_bound() -> None:
    value = _large_command()
    encoded = canonical_json_bytes(value.to_dict())
    assert len(encoded) == 241_623
    assert len(encoded) < cli.FEDERATION_CONTROL_MAX_CANONICAL_BYTES
    assert cli.decode_federation_control_request(value.to_dict(), value.operation) == value

    control, authorizer, owner = service()
    args = _parser().parse_args(
        ["federation", "start", "--command-json", encoded.decode("utf-8")]
    )
    stdout = io.StringIO()
    stderr = io.StringIO()
    cli_status = cli.run_federation_cli(
        args,
        service=control,
        stdout=stdout,
        stderr=stderr,
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
    assert len(authorizer.commands) == 2
    assert len(owner.calls) == 2


def test_bound_admits_a_large_structurally_valid_create_transport() -> None:
    request = sample_request(binding=_large_binding())
    transport = cli.FederationCreateTransport(
        request=request,
        authentication=sample_authentication(request),
    )
    encoded = canonical_json_bytes(transport.to_dict())

    assert len(encoded) > 3 * 241_623
    assert len(encoded) < cli.FEDERATION_CONTROL_MAX_CANONICAL_BYTES
    assert cli.decode_federation_control_request(
        transport.to_dict(), contracts.FederationOperation.CREATE
    ) == transport


def test_cli_mcp_and_python_share_authenticated_create_gateway_contract() -> None:
    transport, gateway = _create_transport()
    payload = _transport_payload(transport)
    args = _parser().parse_args(
        ["federation", "create", "--request-json", payload]
    )
    stdout = io.StringIO()
    stderr = io.StringIO()

    cli_status = cli.run_federation_cli(
        args,
        gateway=gateway,
        stdout=stdout,
        stderr=stderr,
    )
    cli_record = json.loads(stdout.getvalue())
    federation_mcp.configure_federation_control(gateway=gateway)
    try:
        mcp_record = federation_mcp.execute_federation_control(
            transport.to_dict(),
            contracts.FederationOperation.CREATE,
        )
    finally:
        federation_mcp.configure_federation_control()
    python_record = cli.federation_create_response_record(
        transport,
        gateway.create(transport.request, transport.authentication),
    )

    assert cli_status == cli.FEDERATION_CLI_EXIT_SUCCESS
    assert not stderr.getvalue()
    assert _without_transport_identity(cli_record) == _without_transport_identity(mcp_record)
    assert _without_transport_identity(cli_record) == _without_transport_identity(python_record)
    assert "authentication" not in cli_record
    assert cli_record["authentication_evidence_ref"] == transport.authentication.cid


def test_real_hierarchical_manager_registers_discovers_and_invokes_tools() -> None:
    manager = HierarchicalToolManager()
    federation_mcp.register_federation_control_tools(manager)
    expected = {
        tool.__name__
        for tool in federation_mcp.FEDERATION_CONTROL_OPERATION_TOOLS.values()
    }
    assert {item["name"] for item in manager.list_tools("agent_supervisor")} == expected
    assert manager.get_tool_schema(
        "agent_supervisor", "federation_create"
    )["input_schema"]["x-federation-operation"] == "federation.create"

    value = command()
    control, _authorizer, _owner = service()
    create_transport, gateway = _create_transport()
    federation_mcp.configure_federation_control(
        service=control,
        gateway=gateway,
    )

    async def invoke() -> tuple[dict[str, object], dict[str, object]]:
        return (
            await manager.dispatch(
                "agent_supervisor",
                "federation_start",
                {"request": value.to_dict()},
            ),
            await manager.dispatch(
                "agent_supervisor",
                "federation_create",
                {"request": create_transport.to_dict()},
            ),
        )

    try:
        post_record, create_record = anyio.run(invoke)
    finally:
        federation_mcp.configure_federation_control()
    assert post_record["command"] == value.to_dict()
    assert create_record["identity"]["record_id"] == (
        f"federation:{create_transport.request.cid}"
    )


def test_canonical_mcp_server_lazy_category_includes_federation_tools() -> None:
    manager = HierarchicalToolManager()
    service_before = federation_mcp.federation_control_service_resolution_count()
    gateway_before = federation_mcp.federation_control_gateway_resolution_count()

    configure_agent_supervisor_tools(manager)
    assert "agent_supervisor" in manager.list_categories()
    names = {item["name"] for item in manager.list_tools("agent_supervisor")}

    assert "federation_create" in names
    assert "federation_start" in names
    assert federation_mcp.federation_control_service_resolution_count() == service_before
    assert federation_mcp.federation_control_gateway_resolution_count() == gateway_before


@pytest.mark.parametrize(
    ("field", "payload"),
    (
        ("caller_selected_/tmp/secret.duckdb", "/tmp/secret.duckdb"),
        ("caller_selected_secret", "Bearer secret-value-that-must-not-leak"),
    ),
)
def test_unknown_contract_fields_are_redacted_before_service_resolution(
    field: str,
    payload: str,
) -> None:
    value = command().to_dict()
    value[field] = payload
    service_before = federation_mcp.federation_control_service_resolution_count()
    factory_calls: list[object] = []

    def forbidden_factory(decoded: object) -> object:
        factory_calls.append(decoded)
        raise AssertionError("invalid command reached service resolution")

    federation_mcp.configure_federation_control(service_factory=forbidden_factory)
    try:
        record = federation_mcp.execute_federation_control(
            value,
            contracts.FederationOperation.START,
        )
    finally:
        federation_mcp.configure_federation_control()

    _assert_transport_error(record, invalid=True)
    assert field not in json.dumps(record)
    assert payload not in json.dumps(record)
    assert federation_mcp.federation_control_service_resolution_count() == service_before
    assert not factory_calls


def test_unknown_create_fields_are_redacted_before_gateway_resolution() -> None:
    transport, _gateway = _create_transport()
    value = transport.to_dict()
    value["caller_/tmp/private-key"] = "Bearer gateway-secret-material"
    gateway_before = federation_mcp.federation_control_gateway_resolution_count()
    factory_calls: list[object] = []

    def forbidden_factory(decoded: object) -> object:
        factory_calls.append(decoded)
        raise AssertionError("invalid CREATE reached gateway resolution")

    federation_mcp.configure_federation_control(gateway_factory=forbidden_factory)
    try:
        record = federation_mcp.execute_federation_control(
            value,
            contracts.FederationOperation.CREATE,
        )
    finally:
        federation_mcp.configure_federation_control()

    _assert_transport_error(record, invalid=True)
    serialized = json.dumps(record)
    assert "private-key" not in serialized
    assert "gateway-secret" not in serialized
    assert federation_mcp.federation_control_gateway_resolution_count() == gateway_before
    assert not factory_calls


def test_cli_redacts_unknown_field_name_and_value_before_dispatch() -> None:
    value = command().to_dict()
    value["caller_/tmp/database.duckdb"] = "Bearer cli-secret-material"
    args = _parser().parse_args(
        [
            "federation",
            "start",
            "--command-json",
            canonical_json_bytes(value).decode("utf-8"),
        ]
    )
    control, authorizer, owner = service()
    stderr = io.StringIO()

    status = cli.run_federation_cli(
        args,
        service=control,
        stdout=io.StringIO(),
        stderr=stderr,
    )
    record = json.loads(stderr.getvalue())

    assert status == cli.FEDERATION_CLI_EXIT_INVALID
    _assert_transport_error(record, invalid=True)
    assert "database.duckdb" not in stderr.getvalue()
    assert "cli-secret" not in stderr.getvalue()
    assert not authorizer.commands
    assert not owner.calls


def test_oversized_payloads_fail_before_service_or_gateway_resolution() -> None:
    oversized = {
        "schema": contracts.FederationCommand.SCHEMA,
        "padding": "x" * cli.FEDERATION_CONTROL_MAX_CANONICAL_BYTES,
    }
    service_before = federation_mcp.federation_control_service_resolution_count()
    gateway_before = federation_mcp.federation_control_gateway_resolution_count()
    calls: list[object] = []

    def forbidden_factory(decoded: object) -> object:
        calls.append(decoded)
        raise AssertionError("oversized payload reached authority resolution")

    federation_mcp.configure_federation_control(
        service_factory=forbidden_factory,
        gateway_factory=forbidden_factory,
    )
    try:
        post = federation_mcp.execute_federation_control(
            oversized,
            contracts.FederationOperation.START,
        )
        create = federation_mcp.execute_federation_control(
            oversized,
            contracts.FederationOperation.CREATE,
        )
    finally:
        federation_mcp.configure_federation_control()

    _assert_transport_error(post, invalid=True)
    _assert_transport_error(create, invalid=True)
    assert federation_mcp.federation_control_service_resolution_count() == service_before
    assert federation_mcp.federation_control_gateway_resolution_count() == gateway_before
    assert not calls


@pytest.mark.parametrize("authority", ("service", "owner", "gateway"))
def test_cli_redacts_service_owner_and_gateway_exceptions(
    authority: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret_error = RuntimeError(
        "/tmp/private/control.duckdb Bearer transport-secret-material"
    )
    stderr = io.StringIO()
    if authority in {"service", "owner"}:
        value = command()
        args = _parser().parse_args(
            [
                "federation",
                "start",
                "--command-json",
                _transport_payload(value),
            ]
        )
        control, _authorizer, owner = service()

        def explode_owner(*_args: object, **_kwargs: object) -> object:
            raise secret_error

        if authority == "service":
            monkeypatch.setattr(cli, "execute_federation_command", explode_owner)
        else:
            monkeypatch.setattr(owner, "execute_federation_command", explode_owner)
        status = cli.run_federation_cli(
            args,
            service=control,
            stdout=io.StringIO(),
            stderr=stderr,
        )
    else:
        transport, gateway = _create_transport()
        args = _parser().parse_args(
            [
                "federation",
                "create",
                "--request-json",
                _transport_payload(transport),
            ]
        )

        def explode_gateway(*_args: object, **_kwargs: object) -> object:
            raise secret_error

        monkeypatch.setattr(gateway, "create", explode_gateway)
        status = cli.run_federation_cli(
            args,
            gateway=gateway,
            stdout=io.StringIO(),
            stderr=stderr,
        )

    record = json.loads(stderr.getvalue())
    assert status == cli.FEDERATION_CLI_EXIT_FAILED
    _assert_transport_error(record, invalid=False)
    assert "/tmp/private" not in stderr.getvalue()
    assert "transport-secret" not in stderr.getvalue()


@pytest.mark.parametrize(
    "authority",
    ("service", "service_factory", "owner", "gateway", "gateway_factory"),
)
def test_mcp_redacts_factory_owner_and_gateway_exceptions(
    authority: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret_error = RuntimeError(
        "/tmp/private/control.duckdb Bearer mcp-secret-material"
    )

    def explode(*_args: object, **_kwargs: object) -> object:
        raise secret_error

    if authority in {"gateway", "gateway_factory"}:
        transport, _gateway = _create_transport()
        if authority == "gateway_factory":
            federation_mcp.configure_federation_control(gateway_factory=explode)
        else:
            monkeypatch.setattr(_gateway, "create", explode)
            federation_mcp.configure_federation_control(gateway=_gateway)
        operation = contracts.FederationOperation.CREATE
        payload = transport.to_dict()
    else:
        value = command()
        operation = value.operation
        payload = value.to_dict()
        if authority == "service":
            control, _authorizer, _owner = service()
            monkeypatch.setattr(
                federation_mcp,
                "execute_federation_command",
                explode,
            )
            federation_mcp.configure_federation_control(service=control)
        elif authority == "service_factory":
            federation_mcp.configure_federation_control(service_factory=explode)
        else:
            control, _authorizer, owner = service()
            monkeypatch.setattr(owner, "execute_federation_command", explode)
            federation_mcp.configure_federation_control(service=control)
    try:
        record = federation_mcp.execute_federation_control(payload, operation)
    finally:
        federation_mcp.configure_federation_control()

    _assert_transport_error(record, invalid=False)
    serialized = json.dumps(record)
    assert "/tmp/private" not in serialized
    assert "mcp-secret" not in serialized


@pytest.mark.parametrize("operation", ("post_admission", "create"))
def test_missing_authority_fails_closed_with_stable_unavailable_record(
    operation: str,
) -> None:
    federation_mcp.configure_federation_control()
    if operation == "create":
        transport, _gateway = _create_transport()
        payload = transport.to_dict()
        selected = contracts.FederationOperation.CREATE
    else:
        value = command()
        payload = value.to_dict()
        selected = value.operation

    record = federation_mcp.execute_federation_control(payload, selected)

    _assert_transport_error(record, invalid=False)


def test_cli_missing_service_fails_closed_without_configuration_details() -> None:
    value = command()
    args = _parser().parse_args(
        ["federation", "start", "--command-json", _transport_payload(value)]
    )
    stderr = io.StringIO()

    status = cli.run_federation_cli(
        args,
        stdout=io.StringIO(),
        stderr=stderr,
    )
    record = json.loads(stderr.getvalue())

    assert status == cli.FEDERATION_CLI_EXIT_FAILED
    _assert_transport_error(record, invalid=False)
    assert "FederationControlService" not in stderr.getvalue()
