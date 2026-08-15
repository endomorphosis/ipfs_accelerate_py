"""LPC-130 API gate: LogicOperationCatalog@1 Python / CLI / MCP parity.

Live channel agreement for operation names, response envelope, status,
authority, failure codes, and opt-in install boundary. Complements the unit
catalog tests in ``ipfs_datasets_py/tests/unit/logic/test_channel_parity.py``.
"""

from __future__ import annotations

from typing import Any

import anyio
import pytest

from ipfs_datasets_py.logic import cli as logic_cli
from ipfs_datasets_py.logic.verification_api import (
    FeatureAvailability,
    LOGIC_VERIFICATION_API_INTERFACE,
    STABLE_OPERATIONS,
    VerificationAuthority,
    VerificationStatus,
    get_verification_api,
    list_stable_features,
)
from ipfs_datasets_py.mcp_server.tools import logic_verification as lv


ENVELOPE_KEYS = {
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

STATUS_VALUES = {item.value for item in VerificationStatus}
AUTHORITY_VALUES = {item.value for item in VerificationAuthority}


def _assert_envelope(payload: dict[str, Any], *, operation: str | None = None) -> None:
    missing = ENVELOPE_KEYS - set(payload)
    assert not missing, f"missing envelope keys: {sorted(missing)}"
    assert payload["interface"] == LOGIC_VERIFICATION_API_INTERFACE
    assert payload["status"] in STATUS_VALUES
    assert payload["authority"] in AUTHORITY_VALUES
    assert isinstance(payload["result"], dict)
    assert isinstance(payload["unsupported_features"], list)
    if operation is not None:
        assert payload["operation"] == operation


def _run(coro):
    return anyio.run(lambda: coro)


def _cli(argv: list[str]) -> dict[str, Any]:
    ns = logic_cli.create_parser().parse_args(argv)
    data = anyio.run(logic_cli._run_async, ns)
    assert isinstance(data, dict)
    return data


def _semantic_core(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract channel-neutral identity fields for cross-channel comparison."""

    return {
        "status": payload["status"],
        "authority": payload["authority"],
        "operation": payload["operation"],
        "unsupported_features": sorted(payload["unsupported_features"]),
        "interface": payload["interface"],
    }


def test_catalog_interfaces_and_mcp_schema_identity() -> None:
    assert LOGIC_VERIFICATION_API_INTERFACE == "LogicVerificationAPI@1"
    assert logic_cli.LOGIC_VERIFICATION_CLI_INTERFACE == "LogicVerificationCLI@1"
    assert lv.LOGIC_VERIFICATION_MCP_INTERFACE == "LogicVerificationMCP@1"
    for tool in lv.TOOL_NAMES:
        schema = lv.TOOL_SCHEMAS[tool]
        assert schema["returns"]["envelope"] == "logic-verification-response/v1"
        assert schema["python_operation"] == lv.TOOL_TO_OPERATION[tool]


def test_stable_operations_projected_on_all_channels() -> None:
    mapped_mcp = set(lv.TOOL_TO_OPERATION.values())
    for operation in STABLE_OPERATIONS:
        assert operation in mapped_mcp

    parser = logic_cli.create_parser()
    choices = parser._subparsers._group_actions[0].choices  # type: ignore[attr-defined]
    for command in (
        "list-families",
        "list-providers",
        "provider-capabilities",
        "compile",
        "check",
        "monitor",
        "portfolio",
        "counterexample",
        "verify-receipt",
        "attest-receipt",
        "advise",
        "probe-provider",
        "install-provider",
    ):
        assert command in choices

    feature_ids = {item.feature_id for item in list_stable_features()}
    assert set(STABLE_OPERATIONS) <= feature_ids


def test_list_providers_agreement_python_cli_mcp() -> None:
    api = get_verification_api(reset=True)
    py = api.list_providers().to_dict()
    mcp = _run(lv.verification_list_providers())
    cli = _cli(["list-providers"])

    for payload in (py, mcp, cli):
        _assert_envelope(payload, operation="list_providers")

    assert _semantic_core(py) == _semantic_core(mcp) == _semantic_core(cli)
    assert py["result"]["count"] == mcp["result"]["count"] == cli["result"]["count"]
    py_ids = sorted(item["provider_id"] for item in py["result"]["providers"])
    mcp_ids = sorted(item["provider_id"] for item in mcp["result"]["providers"])
    cli_ids = sorted(item["provider_id"] for item in cli["result"]["providers"])
    assert py_ids == mcp_ids == cli_ids


def test_list_logic_families_agreement_python_cli_mcp() -> None:
    api = get_verification_api(reset=True)
    py = api.list_logic_families().to_dict()
    mcp = _run(lv.verification_list_logic_families())
    cli = _cli(["list-families"])

    for payload in (py, mcp, cli):
        _assert_envelope(payload, operation="list_logic_families")

    assert _semantic_core(py) == _semantic_core(mcp) == _semantic_core(cli)
    assert py["result"]["count"] == mcp["result"]["count"] == cli["result"]["count"]


def test_list_features_opt_in_flags_and_channel_agreement() -> None:
    api = get_verification_api(reset=True)
    py = api.list_features().to_dict()
    mcp = _run(lv.verification_list_features())
    cli = _cli(["list-features"])

    for payload in (py, mcp, cli):
        _assert_envelope(payload, operation="list_features")
        assert set(payload["result"]["operations"]) >= set(STABLE_OPERATIONS)

    assert _semantic_core(py) == _semantic_core(mcp) == _semantic_core(cli)

    by_id = {
        item["feature_id"]: item
        for item in py["result"]["features"]
        if isinstance(item, dict) and "feature_id" in item
    }
    # Prefer structured feature descriptors when present; fall back to list_stable_features.
    if not by_id:
        by_id = {
            item.feature_id: {
                "availability": item.availability.value,
                "requires_opt_in": item.requires_opt_in,
                "authority_ceiling": item.authority_ceiling.value,
            }
            for item in list_stable_features()
        }

    for opt_in_op in ("probe_provider", "install_provider", "attest_receipt"):
        descriptor = by_id[opt_in_op]
        availability = descriptor.get("availability")
        if hasattr(availability, "value"):
            availability = availability.value
        assert availability == FeatureAvailability.OPT_IN.value
        assert descriptor.get("requires_opt_in") is True


def test_install_boundary_agreement_and_not_ordinary_verify() -> None:
    api = get_verification_api(reset=True)
    py = api.install_provider("z3").to_dict()
    mcp = _run(lv.verification_install_provider("z3"))
    cli = _cli(["install-provider", "z3"])

    for payload in (py, mcp, cli):
        _assert_envelope(payload, operation="install_provider")
        assert payload["status"] == "unsupported"
        assert payload["authority"] == "none"
        assert "install_without_opt_in" in payload["unsupported_features"]

    assert _semantic_core(py) == _semantic_core(mcp) == _semantic_core(cli)

    planned = api.install_provider("z3", allow_install=True, dry_run=True).to_dict()
    _assert_envelope(planned, operation="install_provider")
    assert planned["status"] == "declarative"
    assert planned["authority"] == "none"
    assert planned["result"]["install_attempted"] is False

    # Installation must remain distinct from ordinary verify operations.
    assert lv.TOOL_TO_OPERATION["verification_install_provider"] != "check"
    assert lv.TOOL_TO_OPERATION["verification_install_provider"] != "verify_receipt"
    check_schema = lv.TOOL_SCHEMAS["verification_check"]
    install_schema = lv.TOOL_SCHEMAS["verification_install_provider"]
    assert check_schema["python_operation"] == "check"
    assert install_schema["python_operation"] == "install_provider"


def test_mcp_install_host_policy_code_is_transport_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("IPFS_DATASETS_PY_MCP_ALLOW_PROVIDER_INSTALLS", raising=False)
    denied = _run(lv.verification_install_provider("z3", allow_install=True))
    assert denied["status"] == "unsupported"
    assert "mcp_provider_install_operator_policy" in denied["unsupported_features"]
    # Facade-owned code path without host gate still uses install_without_opt_in.
    py = get_verification_api(reset=True).install_provider("z3").to_dict()
    assert "install_without_opt_in" in py["unsupported_features"]


def test_unknown_provider_failure_code_agreement() -> None:
    api = get_verification_api(reset=True)
    py = api.provider_capabilities("not-a-backend").to_dict()
    mcp = _run(lv.verification_provider_capabilities(provider_id="not-a-backend"))
    cli = _cli(["provider-capabilities", "--provider-id", "not-a-backend"])

    for payload in (py, mcp, cli):
        _assert_envelope(payload, operation="provider_capabilities")
        assert payload["status"] == "unsupported"
        assert "provider:not-a-backend" in payload["unsupported_features"]

    assert _semantic_core(py) == _semantic_core(mcp) == _semantic_core(cli)


def test_supervisor_only_controls_not_exposed_on_datasets_channels() -> None:
    forbidden = {
        "admit_goal",
        "close_plan",
        "mutate_supervisor",
        "force_complete",
        "lease_steal",
        "rewrite_event_log",
        "bypass_resource_policy",
        "promote_proof_authority",
        "supervisor_mutate",
        "supervisor_only",
    }
    tool_surface = set(lv.TOOL_NAMES) | set(lv.TOOL_TO_OPERATION.values())
    cli_surface = set(
        logic_cli.create_parser()._subparsers._group_actions[0].choices  # type: ignore[attr-defined]
    )
    for name in forbidden:
        assert name not in STABLE_OPERATIONS
        assert name not in tool_surface
        assert name not in cli_surface
        assert name.replace("_", "-") not in cli_surface

    api = get_verification_api(reset=True)
    refused = api.execute_proof_plan(
        {
            "plan_id": "plan:parity-forbidden",
            "steps": [{"step_id": "s1", "obligation_id": "o1"}],
            "controls": {"admit_goal": True, "mutate_supervisor": True},
        }
    ).to_dict()
    _assert_envelope(refused, operation="execute_proof_plan")
    assert refused["status"] == "invalid"
    assert "supervisor_only_control" in refused["unsupported_features"]


def test_verification_capabilities_meta_surface() -> None:
    meta = _run(lv.verification_capabilities())
    assert meta["success"] is True
    assert meta["status"] == "declarative"
    assert meta["mcp_interface"] == lv.LOGIC_VERIFICATION_MCP_INTERFACE
    assert meta["cli_interface"] == logic_cli.LOGIC_VERIFICATION_CLI_INTERFACE
    assert meta["python_interface"] == LOGIC_VERIFICATION_API_INTERFACE
    assert set(meta["operations"]) >= set(STABLE_OPERATIONS)
    assert set(meta["tool_to_operation"].values()) >= set(STABLE_OPERATIONS)

    cli_meta = _cli(["verification-capabilities"])
    assert cli_meta["cli_interface"] == logic_cli.LOGIC_VERIFICATION_CLI_INTERFACE
    assert set(cli_meta["operations"]) >= set(STABLE_OPERATIONS)
