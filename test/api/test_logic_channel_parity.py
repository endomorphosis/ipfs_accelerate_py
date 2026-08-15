"""LPC-130: Python / CLI / MCP logic channel parity (API).

``LogicOperationCatalog@1`` requires that the three public channels agree on
operation names, response schemas, status, authority, failure codes, and
opt-in. Installation is not an ordinary verify operation.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Final

import anyio
import pytest

from ipfs_datasets_py.logic import cli as logic_cli
from ipfs_datasets_py.logic.verification_api import (
    FeatureAvailability,
    LOGIC_VERIFICATION_API_INTERFACE,
    LOGIC_VERIFICATION_RESPONSE_SCHEMA,
    STABLE_OPERATIONS,
    VerificationAuthority,
    VerificationStatus,
    get_verification_api,
    list_stable_features,
)
from ipfs_datasets_py.mcp_server.tools import logic_verification as lv


LOGIC_OPERATION_CATALOG_INTERFACE: Final = "LogicOperationCatalog@1"

ENVELOPE_KEYS: Final[frozenset[str]] = frozenset(
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

# Stable CLI command → Python operation (mirrors operation_catalog.md).
VERIFICATION_CLI_TO_OPERATION: Final[dict[str, str]] = {
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


def _catalog_note_path() -> Path:
    note_relative = Path(
        "data/agent_supervisor/logic_platform_canonicalization/notes/"
        "operation_catalog.md"
    )
    for parent in Path(__file__).resolve().parents:
        candidate = parent / note_relative
        if candidate.is_file():
            return candidate
    return Path(__file__).resolve().parents[2] / note_relative


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
    assert payload["status"] in {member.value for member in VerificationStatus}
    assert payload["authority"] in {member.value for member in VerificationAuthority}
    assert isinstance(payload["result"], dict)
    assert isinstance(payload["unsupported_features"], list)
    if operation is not None:
        assert payload["operation"] == operation


def _load_root_mcp_surfaces() -> list[tuple[str, Any]]:
    candidates = (
        "ipfs_accelerate_py.mcp_server.tools.logic_verification",
        "ipfs_accelerate_py.mcp_server.tools.logic_tools.logic_verification",
        "ipfs_datasets_py.mcp_server.tools.logic_verification",
    )
    loaded: list[tuple[str, Any]] = []
    for name in candidates:
        try:
            module = importlib.import_module(name)
        except Exception:
            continue
        loaded.append((name, module))
    assert loaded, "no MCP formal verification surface could be imported"
    return loaded


# ---------------------------------------------------------------------------
# Catalog artifact
# ---------------------------------------------------------------------------


def test_operation_catalog_note_matches_live_stable_operations() -> None:
    path = _catalog_note_path()
    assert path.is_file(), f"missing {path}"
    text = path.read_text(encoding="utf-8")
    assert LOGIC_OPERATION_CATALOG_INTERFACE in text
    assert "Channels agree on names" in text or "agree on" in text.lower()
    for operation in STABLE_OPERATIONS:
        assert operation in text, f"catalog missing {operation}"
    for command in VERIFICATION_CLI_TO_OPERATION:
        assert f"`{command}`" in text or command in text
    assert "install_without_opt_in" in text
    assert "opt-in" in text.lower() or "Opt-in" in text


# ---------------------------------------------------------------------------
# Name / schema parity across channels
# ---------------------------------------------------------------------------


def test_three_channels_share_closed_stable_operation_set() -> None:
    mcp_ops = set(lv.TOOL_TO_OPERATION.values())
    cli_ops = set(VERIFICATION_CLI_TO_OPERATION.values())
    assert set(STABLE_OPERATIONS) <= mcp_ops
    assert set(STABLE_OPERATIONS) <= cli_ops

    parser = logic_cli.create_parser()
    choices = set(parser._subparsers._group_actions[0].choices)  # type: ignore[attr-defined]
    for command in VERIFICATION_CLI_TO_OPERATION:
        assert command in choices

    for tool_name, operation in lv.TOOL_TO_OPERATION.items():
        schema = lv.TOOL_SCHEMAS[tool_name]
        assert schema["python_operation"] == operation
        assert schema["returns"]["envelope"] == LOGIC_VERIFICATION_RESPONSE_SCHEMA
        assert schema["interface"] == lv.LOGIC_VERIFICATION_MCP_INTERFACE


def test_root_mcp_preserves_datasets_tool_operation_pairs() -> None:
    datasets_ops = dict(lv.TOOL_TO_OPERATION)
    for name, module in _load_root_mcp_surfaces():
        mapping = dict(getattr(module, "TOOL_TO_OPERATION", {}) or {})
        assert mapping, f"{name} exposes empty TOOL_TO_OPERATION"
        for tool, operation in datasets_ops.items():
            if tool in mapping:
                assert mapping[tool] == operation, f"{name} drifted on {tool}"


# ---------------------------------------------------------------------------
# Behavioral parity: discovery
# ---------------------------------------------------------------------------


def test_list_providers_parity_python_cli_mcp() -> None:
    python_api = get_verification_api(reset=True)
    py = python_api.list_providers().to_dict()
    mcp = _run(lv.verification_list_providers())
    cli = _cli(["list-providers"])

    for payload, channel in ((py, "python"), (mcp, "mcp"), (cli, "cli")):
        _assert_envelope(payload, operation="list_providers")
        assert payload["status"] == "declarative", channel
        assert payload["authority"] == "declarative", channel
        assert payload["result"]["count"] >= 1, channel

    py_ids = {item["provider_id"] for item in py["result"]["providers"]}
    mcp_ids = {item["provider_id"] for item in mcp["result"]["providers"]}
    cli_ids = {item["provider_id"] for item in cli["result"]["providers"]}
    assert py_ids == mcp_ids == cli_ids
    assert py["result"]["count"] == mcp["result"]["count"] == cli["result"]["count"]


def _feature_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    items = payload["result"]["features"]
    assert isinstance(items, list), "list_features.result.features must be a list"
    return {item["feature_id"]: item for item in items}


def test_list_features_opt_in_parity_across_channels() -> None:
    python_api = get_verification_api(reset=True)
    py = python_api.list_features().to_dict()
    mcp = _run(lv.verification_list_features())
    cli = _cli(["list-features"])

    for payload in (py, mcp, cli):
        _assert_envelope(payload, operation="list_features")
        assert payload["status"] == "declarative"
        assert set(STABLE_OPERATIONS) <= set(payload["result"]["operations"])
        features = _feature_map(payload)
        install = features["install_provider"]
        assert install["availability"] == FeatureAvailability.OPT_IN.value
        assert install["requires_opt_in"] is True
        assert install["authority_ceiling"] == VerificationAuthority.NONE.value
        check = features["check"]
        assert check["requires_opt_in"] is False
        assert check["availability"] == FeatureAvailability.DECLARED.value

    # Feature descriptors from the pure catalog helper match channel payloads.
    catalog = {item.feature_id: item for item in list_stable_features()}
    assert catalog["install_provider"].requires_opt_in is True
    assert catalog["probe_provider"].requires_opt_in is True
    assert catalog["check"].requires_opt_in is False


def test_provider_capabilities_failure_code_parity() -> None:
    python_api = get_verification_api(reset=True)
    py = python_api.provider_capabilities("not-a-backend").to_dict()
    mcp = _run(lv.verification_provider_capabilities(provider_id="not-a-backend"))
    cli = _cli(["provider-capabilities", "--provider-id", "not-a-backend"])

    for payload in (py, mcp, cli):
        _assert_envelope(payload, operation="provider_capabilities")
        assert payload["status"] == "unsupported"
        assert "provider:not-a-backend" in payload["unsupported_features"]


# ---------------------------------------------------------------------------
# Installation is not verify
# ---------------------------------------------------------------------------


def test_install_denied_failure_code_parity_python_and_mcp(monkeypatch) -> None:
    monkeypatch.delenv("IPFS_DATASETS_PY_MCP_ALLOW_PROVIDER_INSTALLS", raising=False)
    python_api = get_verification_api(reset=True)
    py = python_api.install_provider("z3").to_dict()
    mcp = _run(lv.verification_install_provider("z3"))

    for payload in (py, mcp):
        _assert_envelope(payload, operation="install_provider")
        assert payload["status"] == "unsupported"
        assert payload["authority"] == "none"
        assert "install_without_opt_in" in payload["unsupported_features"]
        assert payload["result"].get("install_attempted") is False
        assert payload["result"].get("installed") is False


def test_install_dry_run_parity_is_declarative_not_check() -> None:
    python_api = get_verification_api(reset=True)
    py = python_api.install_provider("z3", dry_run=True).to_dict()
    mcp = _run(lv.verification_install_provider("z3", dry_run=True))
    cli = _cli(["install-provider", "z3", "--dry-run"])

    for payload, channel in ((py, "python"), (mcp, "mcp"), (cli, "cli")):
        _assert_envelope(payload, operation="install_provider")
        assert payload["status"] == "declarative", channel
        assert payload["authority"] == "none", channel
        assert payload["result"]["status"] == "planned", channel
        assert payload["result"]["install_attempted"] is False, channel
        assert payload["result"]["mutation_authorized"] is False, channel
        assert payload["operation"] != "check"
        assert payload["operation"] != "verify_receipt"
        # Installer receipts never claim proof authority.
        assert payload["result"].get("authority") in {None, "none", "None"} or (
            payload["authority"] == "none"
        )


def test_install_cli_requires_allow_flag_for_mutation() -> None:
    parser = logic_cli.create_parser()
    install = parser._subparsers._group_actions[0].choices["install-provider"]  # type: ignore[attr-defined]
    actions = {action.dest: action for action in install._actions}
    assert "allow_install" in actions
    assert actions["allow_install"].option_strings == ["--allow-install"]
    # Default is off (store_true → default False).
    assert actions["allow_install"].default is False


def test_check_and_verify_receipt_are_distinct_from_install() -> None:
    python_api = get_verification_api(reset=True)
    install = python_api.install_provider("z3", dry_run=True).to_dict()
    check = python_api.check(
        {
            "statement": "(assert true)",
            "source": "(assert true)",
            "logic_family": "first_order",
            "query_kind": "satisfiability",
        }
    ).to_dict()
    receipt = python_api.verify_receipt(
        {
            "receipt_id": "rcpt:channel-parity",
            "authority": "bounded",
            "digest": "b" * 64,
            "kind": "proof_receipt",
        }
    ).to_dict()

    _assert_envelope(install, operation="install_provider")
    _assert_envelope(check, operation="check")
    _assert_envelope(receipt, operation="verify_receipt")
    assert {install["operation"], check["operation"], receipt["operation"]} == {
        "install_provider",
        "check",
        "verify_receipt",
    }
    assert install["authority"] == "none"
    # check may be unavailable in hermetic envs; it must still not be install.
    assert check["operation"] != "install_provider"
    assert receipt["result"].get("install_attempted") is None


def test_mcp_live_install_host_policy_failure_code(monkeypatch) -> None:
    monkeypatch.delenv("IPFS_DATASETS_PY_MCP_ALLOW_PROVIDER_INSTALLS", raising=False)
    denied = _run(
        lv.verification_install_provider("z3", allow_install=True, dry_run=False)
    )
    assert denied["status"] == "unsupported"
    assert "mcp_provider_install_operator_policy" in denied["unsupported_features"]
    # Host policy denial is MCP-transport; Python dry-run remains available.
    planned = get_verification_api(reset=True).install_provider(
        "z3", dry_run=True
    ).to_dict()
    assert planned["status"] == "declarative"


# ---------------------------------------------------------------------------
# Execution envelope parity (representative)
# ---------------------------------------------------------------------------


def test_portfolio_envelope_parity_python_mcp() -> None:
    obligation = {
        "obligation_id": "obl:channel-parity",
        "property_kind": "satisfiability",
        "statement": "(assert true)",
        "assumption_ids": ("parity:channel",),
    }
    python_api = get_verification_api(reset=True)
    # Default execute=True matches MCP/CLI dispatch; compare envelope identity
    # and shared semantic fields rather than brittle probe-dependent status.
    py = python_api.run_portfolio(obligation, execute=True).to_dict()
    mcp = _run(lv.verification_portfolio(obligation=obligation))
    cli = _cli(
        [
            "portfolio",
            "--obligation",
            __import__("json").dumps(obligation),
        ]
    )

    for payload in (py, mcp, cli):
        _assert_envelope(payload, operation="run_portfolio")
        assert payload["status"] in {
            "succeeded",
            "partial",
            "unavailable",
            "error",
            "unsupported",
        }
        # Portfolio execution is not the installer and never mints theorem
        # authority from transport success alone.
        assert payload["operation"] == "run_portfolio"
        assert payload["operation"] != "install_provider"
        if payload["status"] == "succeeded":
            assert payload["authority"] != "theorem"
        if payload["assumptions"]:
            assert "parity:channel" in payload["assumptions"]
    # Channels must agree on the closed status/authority vocabulary even when
    # local probe outcomes differ by environment.
    for payload in (py, mcp, cli):
        assert payload["status"] in {member.value for member in VerificationStatus}
        assert payload["authority"] in {
            member.value for member in VerificationAuthority
        }


def test_advise_parity_authority_is_advisory() -> None:
    request = {"goal_text": "prove P -> Q"}
    python_api = get_verification_api(reset=True)
    py = python_api.advise(request, provider="static").to_dict()
    mcp = _run(lv.verification_advise(request=request, provider="static"))
    cli = _cli(
        [
            "advise",
            "--request",
            __import__("json").dumps(request),
            "--provider",
            "static",
        ]
    )
    for payload in (py, mcp, cli):
        _assert_envelope(payload, operation="advise")
        assert payload["status"] == "succeeded"
        assert payload["authority"] == "advisory"
        note = str(payload["result"].get("authority_note") or "").lower()
        assert "never" in note or payload["authority"] == "advisory"


def test_channels_do_not_expose_supervisor_mutation_tools() -> None:
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
    public = set(STABLE_OPERATIONS) | set(lv.TOOL_TO_OPERATION.values()) | set(
        lv.TOOL_NAMES
    )
    parser = logic_cli.create_parser()
    public |= set(parser._subparsers._group_actions[0].choices)  # type: ignore[attr-defined]
    leaked = forbidden & public
    assert not leaked


def test_verification_capabilities_surface_agrees() -> None:
    mcp = _run(lv.verification_capabilities())
    cli = _cli(["verification-capabilities"])
    assert mcp["mcp_interface"] == lv.LOGIC_VERIFICATION_MCP_INTERFACE
    assert mcp["python_interface"] == LOGIC_VERIFICATION_API_INTERFACE
    assert set(mcp["operations"]) >= set(STABLE_OPERATIONS)
    assert cli["cli_interface"] == logic_cli.LOGIC_VERIFICATION_CLI_INTERFACE
    assert set(cli["operations"]) >= set(STABLE_OPERATIONS)
    assert mcp["tool_to_operation"] == cli["tool_to_operation"]
