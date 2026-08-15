"""LPC-130 API gates: Python / CLI / MCP LogicOperationCatalog@1 parity.

Channels must agree on names, schemas, status, authority, failure codes, and
opt-in. Installation is not an ordinary verify operation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import anyio
import pytest

from ipfs_datasets_py.logic import cli as logic_cli
from ipfs_datasets_py.logic.verification_api import (
    FORMAL_VERIFICATION_MCP_PARITY_INTERFACE,
    GOAL_TACTICIAN_CLI_MCP_INTERFACE,
    GOAL_TACTICIAN_OPERATIONS,
    GOAL_TACTICIAN_TOOL_TO_OPERATION,
    LOGIC_VERIFICATION_API_INTERFACE,
    LOGIC_VERIFICATION_RESPONSE_SCHEMA,
    STABLE_OPERATIONS,
    VerificationAuthority,
    VerificationStatus,
    get_verification_api,
    invoke_goal_tactician,
    list_stable_features,
)
from ipfs_datasets_py.mcp_server.tools import logic_verification as lv


LOGIC_OPERATION_CATALOG_INTERFACE = "LogicOperationCatalog@1"

ENVELOPE_KEYS = frozenset(
    {
        "status",
        "authority",
        "operation",
        "result",
        "assumptions",
        "bounds",
        "translations",
        "witnesses",
        "unsupported_features",
        "diagnostics",
        "cache",
        "interface",
    }
)

CLI_TO_OPERATION: dict[str, str] = {
    "list-features": "list_features",
    "list-families": "list_logic_families",
    "list-providers": "list_providers",
    "provider-capabilities": "provider_capabilities",
    "compile": "compile_verification_artifact",
    "check": "check",
    "monitor": "monitor",
    "portfolio": "run_portfolio",
    "counterexample": "explain_counterexample",
    "verify-receipt": "verify_receipt",
    "advise": "advise",
    "attest-receipt": "attest_receipt",
    "probe-provider": "probe_provider",
    "install-provider": "install_provider",
}

OPERATION_TO_CLI = {operation: command for command, operation in CLI_TO_OPERATION.items()}
OPERATION_TO_MCP = {
    operation: tool
    for tool, operation in lv.TOOL_TO_OPERATION.items()
    if tool != "verification_capabilities"
}

STATUS_VALUES = frozenset(item.value for item in VerificationStatus)
AUTHORITY_VALUES = frozenset(item.value for item in VerificationAuthority)

CATALOG_NOTE_RELATIVE = Path(
    "data/agent_supervisor/logic_platform_canonicalization/notes/operation_catalog.md"
)


def _catalog_note_path() -> Path:
    here = Path(__file__).resolve()
    candidates = [
        here.parents[2] / CATALOG_NOTE_RELATIVE,
        Path.cwd() / CATALOG_NOTE_RELATIVE,
    ]
    for parent in here.parents:
        candidates.append(parent / CATALOG_NOTE_RELATIVE)
    for path in candidates:
        if path.is_file():
            return path
    return here.parents[2] / CATALOG_NOTE_RELATIVE


def _run(coro):
    return anyio.run(lambda: coro)


def _cli(argv: list[str]) -> dict[str, Any]:
    ns = logic_cli.create_parser().parse_args(argv)
    data = anyio.run(logic_cli._run_async, ns)
    assert isinstance(data, dict)
    return data


def _assert_envelope(payload: dict[str, Any], *, operation: str | None = None) -> None:
    missing = ENVELOPE_KEYS - set(payload)
    assert not missing, f"missing envelope keys: {sorted(missing)}"
    assert payload["interface"] == LOGIC_VERIFICATION_API_INTERFACE
    schema = payload.get("schema_version")
    assert schema is None or isinstance(schema, str)
    if isinstance(schema, str) and schema:
        assert schema == LOGIC_VERIFICATION_RESPONSE_SCHEMA or schema.startswith(
            "logic-verification-response"
        )
    assert isinstance(payload["status"], str) and payload["status"] in STATUS_VALUES
    assert isinstance(payload["authority"], str) and payload["authority"] in AUTHORITY_VALUES
    assert isinstance(payload["result"], dict)
    assert isinstance(payload["assumptions"], list)
    assert isinstance(payload["bounds"], dict)
    assert isinstance(payload["translations"], list)
    assert isinstance(payload["witnesses"], list)
    assert isinstance(payload["unsupported_features"], list)
    assert isinstance(payload["diagnostics"], list)
    assert isinstance(payload["cache"], dict)
    if operation is not None:
        assert payload["operation"] == operation


def _assert_channel_agreement(
    python_payload: dict[str, Any],
    cli_payload: dict[str, Any],
    mcp_payload: dict[str, Any],
    *,
    operation: str,
    compare_result_keys: tuple[str, ...] = (),
) -> None:
    for payload, channel in (
        (python_payload, "python"),
        (cli_payload, "cli"),
        (mcp_payload, "mcp"),
    ):
        _assert_envelope(payload, operation=operation)

    assert python_payload["status"] == cli_payload["status"] == mcp_payload["status"]
    assert (
        python_payload["authority"]
        == cli_payload["authority"]
        == mcp_payload["authority"]
    )
    assert (
        set(python_payload["unsupported_features"])
        == set(cli_payload["unsupported_features"])
        == set(mcp_payload["unsupported_features"])
    )

    if compare_result_keys:
        for key in compare_result_keys:
            assert (
                python_payload["result"].get(key)
                == cli_payload["result"].get(key)
                == mcp_payload["result"].get(key)
            ), key


def test_catalog_interfaces_and_evidence_note() -> None:
    assert LOGIC_OPERATION_CATALOG_INTERFACE == "LogicOperationCatalog@1"
    assert LOGIC_VERIFICATION_API_INTERFACE == "LogicVerificationAPI@1"
    assert logic_cli.LOGIC_VERIFICATION_CLI_INTERFACE == "LogicVerificationCLI@1"
    assert lv.LOGIC_VERIFICATION_MCP_INTERFACE == "LogicVerificationMCP@1"
    assert FORMAL_VERIFICATION_MCP_PARITY_INTERFACE == "FormalVerificationMCPParity@1"
    note = _catalog_note_path()
    assert note.is_file(), f"missing catalog evidence note at {note}"
    text = note.read_text(encoding="utf-8")
    assert "LogicOperationCatalog@1" in text
    assert "Installation is not an ordinary verify operation" in text
    assert "STABLE_OPERATIONS" in text
    assert "install_without_opt_in" in text
    assert "requires_opt_in" in text


def test_channel_name_maps_cover_stable_operations() -> None:
    mcp_ops = {
        op
        for tool, op in lv.TOOL_TO_OPERATION.items()
        if tool != "verification_capabilities"
    }
    cli_ops = set(CLI_TO_OPERATION.values())
    assert set(STABLE_OPERATIONS) <= mcp_ops
    assert set(STABLE_OPERATIONS) <= cli_ops

    for operation in STABLE_OPERATIONS:
        assert operation in OPERATION_TO_MCP
        assert operation in OPERATION_TO_CLI
        tool = OPERATION_TO_MCP[operation]
        schema = lv.get_tool_schema(tool)
        assert schema is not None
        assert schema["python_operation"] == operation
        assert schema["returns"]["envelope"] == "logic-verification-response/v1"
        assert schema["interface"] == lv.LOGIC_VERIFICATION_MCP_INTERFACE


def test_opt_in_surface_agrees_across_feature_catalog_and_tools() -> None:
    features = {item.feature_id: item for item in list_stable_features()}
    opt_in = {
        name
        for name, item in features.items()
        if name in STABLE_OPERATIONS and item.requires_opt_in
    }
    assert opt_in == {"probe_provider", "install_provider", "attest_receipt"}

    install_schema = lv.TOOL_SCHEMAS["verification_install_provider"]
    assert install_schema["parameters"]["properties"]["allow_install"]["default"] is False
    probe_schema = lv.TOOL_SCHEMAS["verification_probe_provider"]
    assert "provider_id" in probe_schema["parameters"]["required"]

    parser = logic_cli.create_parser()
    install_help = parser._subparsers._group_actions[0].choices[  # type: ignore[attr-defined]
        "install-provider"
    ].format_help()
    assert "--allow-install" in install_help


def test_list_providers_parity_python_cli_mcp() -> None:
    api = get_verification_api(reset=True)
    py = api.list_providers().to_dict()
    cli = _cli(["list-providers"])
    mcp = _run(lv.verification_list_providers())

    _assert_channel_agreement(
        py,
        cli,
        mcp,
        operation="list_providers",
        compare_result_keys=("count",),
    )
    assert py["status"] == "declarative"
    assert py["authority"] == "declarative"
    py_ids = {item["provider_id"] for item in py["result"]["providers"]}
    cli_ids = {item["provider_id"] for item in cli["result"]["providers"]}
    mcp_ids = {item["provider_id"] for item in mcp["result"]["providers"]}
    assert py_ids == cli_ids == mcp_ids
    assert mcp["mcp_interface"] == lv.LOGIC_VERIFICATION_MCP_INTERFACE
    assert cli["mcp_interface"] == lv.LOGIC_VERIFICATION_MCP_INTERFACE


def test_list_features_and_capabilities_parity() -> None:
    api = get_verification_api(reset=True)
    py = api.list_features().to_dict()
    cli = _cli(["list-features"])
    mcp = _run(lv.verification_list_features())

    _assert_channel_agreement(
        py,
        cli,
        mcp,
        operation="list_features",
    )
    assert set(py["result"]["operations"]) >= set(STABLE_OPERATIONS)
    assert set(cli["result"]["operations"]) >= set(STABLE_OPERATIONS)
    assert set(mcp["result"]["operations"]) >= set(STABLE_OPERATIONS)

    meta_cli = _cli(["verification-capabilities"])
    meta_mcp = _run(lv.verification_capabilities())
    assert meta_cli["success"] is True
    assert meta_mcp["success"] is True
    assert meta_cli["cli_interface"] == logic_cli.LOGIC_VERIFICATION_CLI_INTERFACE
    assert meta_mcp["mcp_interface"] == lv.LOGIC_VERIFICATION_MCP_INTERFACE
    assert set(meta_mcp["operations"]) >= set(STABLE_OPERATIONS)
    assert set(meta_mcp["tools"]) == set(lv.TOOL_NAMES)


def test_provider_capabilities_failure_code_parity() -> None:
    api = get_verification_api(reset=True)
    py = api.provider_capabilities("not-a-backend").to_dict()
    cli = _cli(["provider-capabilities", "--provider-id", "not-a-backend"])
    mcp = _run(lv.verification_provider_capabilities(provider_id="not-a-backend"))

    _assert_channel_agreement(py, cli, mcp, operation="provider_capabilities")
    assert py["status"] == "unsupported"
    assert "provider:not-a-backend" in py["unsupported_features"]
    assert "provider:not-a-backend" in cli["unsupported_features"]
    assert "provider:not-a-backend" in mcp["unsupported_features"]


def test_install_provider_is_not_ordinary_verify_across_channels(monkeypatch) -> None:
    monkeypatch.delenv("IPFS_DATASETS_PY_MCP_ALLOW_PROVIDER_INSTALLS", raising=False)
    api = get_verification_api(reset=True)

    # Denied opt-in: Python + CLI share install_without_opt_in; MCP may add host policy.
    py_denied = api.install_provider("z3").to_dict()
    cli_denied = _cli(["install-provider", "z3"])
    mcp_denied = _run(lv.verification_install_provider("z3"))

    for payload in (py_denied, cli_denied, mcp_denied):
        _assert_envelope(payload, operation="install_provider")
        assert payload["authority"] == "none"
        assert payload["status"] == "unsupported"

    assert "install_without_opt_in" in py_denied["unsupported_features"]
    assert "install_without_opt_in" in cli_denied["unsupported_features"]
    # MCP without allow_install still uses facade opt-in denial (not host gate).
    assert "install_without_opt_in" in mcp_denied["unsupported_features"]

    # Dry-run planning is declarative installer work, not verify success.
    py_plan = api.install_provider("z3", dry_run=True).to_dict()
    cli_plan = _cli(["install-provider", "z3", "--dry-run"])
    mcp_plan = _run(lv.verification_install_provider("z3", dry_run=True))

    _assert_channel_agreement(
        py_plan,
        cli_plan,
        mcp_plan,
        operation="install_provider",
        compare_result_keys=("install_attempted", "status"),
    )
    assert py_plan["status"] == "declarative"
    assert py_plan["authority"] == "none"
    assert py_plan["result"]["install_attempted"] is False

    # MCP live mutation requires host operator gate in addition to allow_install.
    mcp_host_denied = _run(
        lv.verification_install_provider("z3", allow_install=True)
    )
    assert mcp_host_denied["status"] == "unsupported"
    assert "mcp_provider_install_operator_policy" in mcp_host_denied["unsupported_features"]
    assert mcp_host_denied["authority"] == "none"

    # verify_receipt remains a distinct ordinary verify path.
    py_receipt = api.verify_receipt(None).to_dict()
    cli_receipt = _cli(
        ["verify-receipt", "--receipt", json.dumps({"receipt_id": "rcpt:parity"})]
    )
    mcp_receipt = _run(
        lv.verification_verify_receipt({"receipt_id": "rcpt:parity"})
    )
    for payload in (py_receipt, cli_receipt, mcp_receipt):
        _assert_envelope(payload, operation="verify_receipt")
        assert payload["operation"] != "install_provider"
        assert "install_without_opt_in" not in payload["unsupported_features"]
        assert "mcp_provider_install_operator_policy" not in payload["unsupported_features"]
        assert "offline_install" not in payload["unsupported_features"]


def test_advise_and_attest_parity_and_failure_codes() -> None:
    api = get_verification_api(reset=True)
    request = {"goal_text": "prove P -> Q"}

    py = api.advise(request, provider="static").to_dict()
    cli = _cli(
        [
            "advise",
            "--request",
            json.dumps(request),
            "--provider",
            "static",
        ]
    )
    mcp = _run(lv.verification_advise(request, provider="static"))
    _assert_channel_agreement(py, cli, mcp, operation="advise")
    assert py["status"] == "succeeded"
    assert py["authority"] == "advisory"

    py_bad = api.advise(request, provider="not-real").to_dict()
    cli_bad = _cli(
        [
            "advise",
            "--request",
            json.dumps(request),
            "--provider",
            "not-real",
        ]
    )
    mcp_bad = _run(lv.verification_advise(request, provider="not-real"))
    _assert_channel_agreement(py_bad, cli_bad, mcp_bad, operation="advise")
    assert py_bad["status"] == "unsupported"
    assert "advisor:not-real" in py_bad["unsupported_features"]

    py_attest = api.attest_receipt({"receipt_id": "r"}, backend_mode="disabled").to_dict()
    cli_attest = _cli(
        [
            "attest-receipt",
            "--receipt",
            json.dumps({"receipt_id": "r"}),
            "--backend-mode",
            "disabled",
        ]
    )
    mcp_attest = _run(
        lv.verification_attest_receipt({"receipt_id": "r"}, backend_mode="disabled")
    )
    _assert_channel_agreement(py_attest, cli_attest, mcp_attest, operation="attest_receipt")
    assert py_attest["status"] == "unavailable"
    assert "attestation_backend" in py_attest["unsupported_features"]


def test_compile_unsupported_target_failure_code_parity() -> None:
    api = get_verification_api(reset=True)
    artifact = {"obligation_id": "obl:parity", "statement": "true"}
    py = api.compile_verification_artifact(artifact, target="not-a-target").to_dict()
    cli = _cli(
        [
            "compile",
            "--artifact",
            json.dumps(artifact),
            "--target",
            "not-a-target",
        ]
    )
    mcp = _run(lv.verification_compile(artifact, target="not-a-target"))
    _assert_channel_agreement(
        py, cli, mcp, operation="compile_verification_artifact"
    )
    assert py["status"] == "unsupported"
    assert "compile_target:not-a-target" in py["unsupported_features"]


def test_list_logic_families_parity() -> None:
    api = get_verification_api(reset=True)
    py = api.list_logic_families().to_dict()
    cli = _cli(["list-families"])
    mcp = _run(lv.verification_list_logic_families())
    _assert_channel_agreement(
        py,
        cli,
        mcp,
        operation="list_logic_families",
        compare_result_keys=("count",),
    )
    assert py["status"] == "declarative"


def test_supervisor_only_controls_not_exposed_on_channels() -> None:
    forbidden = (
        "admit_goal",
        "mutate_supervisor",
        "force_complete",
        "lease_steal",
        "supervisor_mutate",
    )
    public_names = set(STABLE_OPERATIONS) | set(lv.TOOL_NAMES) | set(CLI_TO_OPERATION)
    for name in forbidden:
        assert name not in public_names
        assert f"verification_{name}" not in lv.TOOL_NAMES

    # Goal tactician still refuses supervisor mutation with a shared failure code.
    refused = invoke_goal_tactician(
        "execute_proof_plan",
        {
            "plan_id": "plan:forbidden",
            "steps": [{"step_id": "s", "obligation_id": "o", "statement": "x"}],
            "controls": {"mutate_supervisor": True},
        },
    ).to_dict()
    assert refused["status"] == "invalid"
    assert "supervisor_only_control" in refused["unsupported_features"]


def test_goal_tactician_additive_does_not_break_verification_mcp() -> None:
    assert set(GOAL_TACTICIAN_OPERATIONS).isdisjoint(set(STABLE_OPERATIONS))
    for tool_name in GOAL_TACTICIAN_TOOL_TO_OPERATION:
        assert tool_name not in lv.TOOL_NAMES
    mapped = {
        op
        for tool, op in lv.TOOL_TO_OPERATION.items()
        if tool != "verification_capabilities"
    }
    for operation in STABLE_OPERATIONS:
        assert operation in mapped
    assert GOAL_TACTICIAN_CLI_MCP_INTERFACE == "GoalTacticianCLIMCP@1"


def test_probe_provider_parity_status_and_authority() -> None:
    api = get_verification_api(reset=True)
    py = api.probe_provider("z3").to_dict()
    cli = _cli(["probe-provider", "z3"])
    mcp = _run(lv.verification_probe_provider("z3"))
    _assert_channel_agreement(py, cli, mcp, operation="probe_provider")
    assert py["status"] in {
        "succeeded",
        "unavailable",
        "unsupported",
        "error",
        "declarative",
    }
    # Probe never claims semantic verification authority.
    assert py["authority"] == "none"
    assert "available" in py["result"]


def test_counterexample_and_portfolio_envelope_parity() -> None:
    api = get_verification_api(reset=True)
    witness = {
        "kind": "model",
        "model": {"x": "1"},
        "summary": "x assigned 1",
        "tool_id": "z3",
        "property_id": "property:parity",
    }
    py_cex = api.explain_counterexample(witness).to_dict()
    cli_cex = _cli(["counterexample", "--witness", json.dumps(witness)])
    mcp_cex = _run(lv.verification_explain_counterexample(witness))
    _assert_channel_agreement(py_cex, cli_cex, mcp_cex, operation="explain_counterexample")
    assert py_cex["status"] in {"succeeded", "invalid", "partial", "unavailable", "error"}

    obligation = {
        "obligation_id": "obl:channel-parity",
        "property_kind": "satisfiability",
        "statement": "(assert true)",
        "assumption_ids": ["a:parity"],
    }
    # Match MCP default (execute=True) so planning + selection semantics agree.
    py_port = api.run_portfolio(obligation).to_dict()
    cli_port = _cli(["portfolio", "--obligation", json.dumps(obligation)])
    mcp_port = _run(lv.verification_portfolio(obligation=obligation))
    _assert_channel_agreement(py_port, cli_port, mcp_port, operation="run_portfolio")
    assert py_port["status"] in STATUS_VALUES
    assert py_port["authority"] in AUTHORITY_VALUES
    assert py_port["result"].get("executed") is True
