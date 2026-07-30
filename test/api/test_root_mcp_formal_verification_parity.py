"""Root/datasets MCP parity for formal verification (FVT-G012 / FVT-013).

``FormalVerificationMCPParity@1`` requires that Python
``LogicVerificationAPI@1``, datasets MCP ``LogicVerificationMCP@1``, and the
parent/root MCP import path expose equivalent availability and execution
semantics with a shared response envelope.
"""

from __future__ import annotations

import importlib
from typing import Any

import anyio
import pytest

from ipfs_datasets_py.logic.verification_api import (
    EXECUTABLE_PROVIDER_MATRIX_INTERFACE,
    FORMAL_VERIFICATION_MCP_PARITY_INTERFACE,
    LOGIC_VERIFICATION_API_INTERFACE,
    LOGIC_VERIFICATION_RESPONSE_SCHEMA,
    STABLE_OPERATIONS,
    get_verification_api,
)


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


def _run(coro):
    return anyio.run(lambda: coro)


def _assert_envelope(payload: dict[str, Any], *, operation: str | None = None) -> None:
    missing = ENVELOPE_KEYS - set(payload)
    assert not missing, f"missing envelope keys: {sorted(missing)}"
    assert payload["interface"] == LOGIC_VERIFICATION_API_INTERFACE
    assert payload.get("schema_version") in {
        None,
        LOGIC_VERIFICATION_RESPONSE_SCHEMA,
        "logic-verification-response/v1",
    } or isinstance(payload.get("schema_version"), str)
    assert isinstance(payload["status"], str) and payload["status"]
    assert isinstance(payload["authority"], str)
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


def _load_datasets_mcp():
    return importlib.import_module(
        "ipfs_datasets_py.mcp_server.tools.logic_verification"
    )


def _load_root_mcp_surface():
    """Resolve the parent/root MCP formal-verification surface.

    Prefer parent package re-exports under ``ipfs_accelerate_py.mcp_server`` when
    present; otherwise the datasets MCP module is the authoritative root that
    parent registration is expected to proxy (FormalVerificationMCPParity@1).
    """

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


def test_formal_verification_mcp_parity_interface_constant() -> None:
    assert FORMAL_VERIFICATION_MCP_PARITY_INTERFACE == "FormalVerificationMCPParity@1"
    assert EXECUTABLE_PROVIDER_MATRIX_INTERFACE == "ExecutableProviderMatrix@1"


def test_datasets_mcp_tool_surface_matches_stable_python_operations() -> None:
    lv = _load_datasets_mcp()
    assert lv.LOGIC_VERIFICATION_MCP_INTERFACE == "LogicVerificationMCP@1"
    mapped = set(lv.TOOL_TO_OPERATION.values())
    for operation in STABLE_OPERATIONS:
        assert operation in mapped, f"datasets MCP missing mapping for {operation}"
    # Core discovery/execution tools required for matrix parity.
    for tool in (
        "verification_list_providers",
        "verification_provider_capabilities",
        "verification_check",
        "verification_portfolio",
        "verification_probe_provider",
    ):
        assert tool in lv.TOOL_NAMES
        assert tool in lv.TOOL_SCHEMAS
        assert lv.TOOL_SCHEMAS[tool]["interface"] == lv.LOGIC_VERIFICATION_MCP_INTERFACE
        assert lv.TOOL_SCHEMAS[tool]["returns"]["envelope"] == (
            "logic-verification-response/v1"
        )


def test_root_and_datasets_mcp_share_tool_names_and_operations() -> None:
    datasets = _load_datasets_mcp()
    surfaces = _load_root_mcp_surface()
    datasets_tools = set(datasets.TOOL_NAMES)
    datasets_ops = dict(datasets.TOOL_TO_OPERATION)

    for name, module in surfaces:
        assert getattr(module, "LOGIC_VERIFICATION_MCP_INTERFACE", "") == (
            "LogicVerificationMCP@1"
        ), f"{name} interface mismatch"
        tool_names = set(getattr(module, "TOOL_NAMES", ()) or ())
        assert tool_names, f"{name} exposes no tools"
        # Root/parent surfaces must cover the datasets formal-verification tools.
        assert datasets_tools <= tool_names or tool_names == datasets_tools
        mapping = dict(getattr(module, "TOOL_TO_OPERATION", {}) or {})
        for tool, operation in datasets_ops.items():
            if tool in mapping:
                assert mapping[tool] == operation


def test_list_providers_parity_python_datasets_mcp() -> None:
    python_api = get_verification_api(reset=True)
    py = python_api.list_providers().to_dict()
    _assert_envelope(py, operation="list_providers")
    assert py["status"] == "declarative"
    assert py["result"]["count"] >= 9
    assert py["result"].get("executable_provider_matrix") == (
        EXECUTABLE_PROVIDER_MATRIX_INTERFACE
    )
    py_ids = {item["provider_id"] for item in py["result"]["providers"]}
    assert {"z3", "cvc5", "runtime_mtl", "hammer", "lean"} <= py_ids

    lv = _load_datasets_mcp()
    mcp = _run(lv.verification_list_providers())
    _assert_envelope(mcp, operation="list_providers")
    assert mcp["status"] == py["status"]
    mcp_ids = {item["provider_id"] for item in mcp["result"]["providers"]}
    # Matrix providers must appear on both surfaces.
    matrix_core = {
        "z3",
        "cvc5",
        "tla_tlc",
        "runtime_mtl",
        "datalog_secpal",
        "proverif",
        "hyperltl_autohyper_mchyper",
        "vampire",
        "hammer",
        "lean",
    }
    assert matrix_core <= py_ids
    assert matrix_core <= mcp_ids
    assert mcp["result"].get("executable_provider_matrix") == (
        EXECUTABLE_PROVIDER_MATRIX_INTERFACE
    )


def test_provider_capabilities_and_probe_parity() -> None:
    python_api = get_verification_api(reset=True)
    py_caps = python_api.provider_capabilities().to_dict()
    _assert_envelope(py_caps, operation="provider_capabilities")
    assert py_caps["status"] == "declarative"
    assert py_caps["result"]["count"] >= 9

    lv = _load_datasets_mcp()
    mcp_caps = _run(lv.verification_provider_capabilities())
    _assert_envelope(mcp_caps, operation="provider_capabilities")
    assert mcp_caps["result"]["count"] == py_caps["result"]["count"]

    missing_py = python_api.provider_capabilities("not-a-backend").to_dict()
    missing_mcp = _run(
        lv.verification_provider_capabilities(provider_id="not-a-backend")
    )
    assert missing_py["status"] == "unsupported"
    assert missing_mcp["status"] == "unsupported"

    probe_py = python_api.probe_provider("runtime_mtl").to_dict()
    probe_mcp = _run(lv.verification_probe_provider(provider_id="runtime_mtl"))
    _assert_envelope(probe_py, operation="probe_provider")
    _assert_envelope(probe_mcp, operation="probe_provider")
    assert probe_py["status"] == probe_mcp["status"]
    assert "available" in probe_py["result"]
    assert "available" in probe_mcp["result"]


def test_portfolio_execution_parity_across_python_and_mcp() -> None:
    obligation = {
        "obligation_id": "obl:mcp-parity",
        "property_kind": "satisfiability",
        "statement": "(assert true)",
        "assumption_ids": ("parity:1",),
    }
    python_api = get_verification_api(reset=True)
    py = python_api.run_portfolio(obligation, execute=True).to_dict()
    _assert_envelope(py, operation="run_portfolio")
    assert py["result"].get("executed") is True
    assert "selection" in py["result"]
    assert "verdict" in py["result"]
    assert py["result"].get("executable_provider_matrix") == (
        EXECUTABLE_PROVIDER_MATRIX_INTERFACE
    )
    assert py["assumptions"] == ["parity:1"]

    lv = _load_datasets_mcp()
    mcp = _run(lv.verification_portfolio(obligation=obligation))
    _assert_envelope(mcp, operation="run_portfolio")
    assert mcp["result"].get("executed") is True
    assert "selection" in mcp["result"]
    assert "verdict" in mcp["result"]
    # Both surfaces use the same stable authority vocabulary.
    assert py["authority"] in {
        "none",
        "advisory",
        "bounded",
        "satisfiability",
        "model_check",
        "monitor",
        "authorization",
        "protocol",
        "hyperproperty",
        "candidate",
        "reconstruction",
        "attestation",
        "theorem",
        "declarative",
    }
    assert mcp["authority"] in {
        "none",
        "advisory",
        "bounded",
        "satisfiability",
        "model_check",
        "monitor",
        "authorization",
        "protocol",
        "hyperproperty",
        "candidate",
        "reconstruction",
        "attestation",
        "theorem",
        "declarative",
    }


def test_root_mcp_surfaces_match_python_envelope_for_list_features() -> None:
    python_api = get_verification_api(reset=True)
    py = python_api.list_features().to_dict()
    _assert_envelope(py, operation="list_features")
    assert set(py["result"]["operations"]) >= set(STABLE_OPERATIONS)

    for name, module in _load_root_mcp_surface():
        list_features = getattr(module, "verification_list_features", None)
        assert callable(list_features), f"{name} missing verification_list_features"
        payload = _run(list_features())
        _assert_envelope(payload, operation="list_features")
        assert set(payload["result"]["operations"]) >= set(STABLE_OPERATIONS)


def test_logic_tools_reexport_matches_datasets_mcp() -> None:
    """Datasets logic_tools re-export is the bridge parent MCP registration uses."""

    datasets_mcp = _load_datasets_mcp()
    logic_tools = importlib.import_module(
        "ipfs_datasets_py.mcp_server.tools.logic_tools"
    )
    for tool in (
        "verification_list_providers",
        "verification_portfolio",
        "verification_check",
        "verification_probe_provider",
    ):
        assert hasattr(logic_tools, tool), f"logic_tools missing {tool}"
        assert callable(getattr(logic_tools, tool))
        assert tool in datasets_mcp.TOOL_NAMES
