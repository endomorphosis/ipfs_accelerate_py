"""SCA-612/613 / SCAEV181MCPRUNTIME: live MCP++ conformance and runtime identity."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.mcp_live_conformance import (
    MCP_LIVE_CONFORMANCE_INTERFACE,
    RUNTIME_SERVICE_AUTHORITY_SCHEMA,
    RUNTIME_SERVICE_IDENTITY_INTERFACE,
    SCAEV181_EVIDENCE_TERM,
    ConformanceTerminalState,
    McpLiveConformanceError,
    TransportKind,
    build_runtime_service_identity,
    load_runtime_service_authority,
    run_mcp_live_conformance,
    run_scaev181_mcp_runtime_gate,
    write_service_identity_receipt,
)
from ipfs_accelerate_py.mcp_server.mcplusplus.idl_registry import (
    IDL_IDENTITY_PROFILE,
    is_pseudo_interface_cid,
)


def _repo_root() -> Path:
    # test/api/this_file -> test -> ipfs_accelerate -> external -> workspace root
    here = Path(__file__).resolve()
    for candidate in (here.parents[4], here.parents[3], Path.cwd()):
        if (candidate / "config" / "swissknife_runtime_service_authority.json").is_file():
            return candidate
    return here.parents[4]


def test_run_mcp_live_conformance_passes_with_zero_model_calls() -> None:
    report = run_mcp_live_conformance()
    payload = report.to_dict()

    assert report.passed is True
    assert report.model_calls == 0
    assert report.evidence_term == SCAEV181_EVIDENCE_TERM
    assert payload["interface"] == MCP_LIVE_CONFORMANCE_INTERFACE
    assert report.interface_profile == IDL_IDENTITY_PROFILE
    assert not is_pseudo_interface_cid(report.interface_cid)
    assert report.interface_cid.startswith("b")

    package = report.package_results["ipfs_accelerate_py"]
    assert package["status"] == ConformanceTerminalState.PASSED.value

    list_receipts = [r for r in report.receipts if r.operation == "tools/list"]
    call_receipts = [
        r
        for r in report.receipts
        if r.method == "tools/call"
        and r.terminal_state == ConformanceTerminalState.PASSED.value
    ]
    unknown_receipts = [
        r
        for r in report.receipts
        if r.terminal_state == ConformanceTerminalState.REFUTED.value
        and r.operation.startswith("__sca_unknown")
    ]
    direct_receipts = [
        r for r in report.receipts if r.transport == TransportKind.DIRECT_IMPORT.value
    ]

    assert len(list_receipts) == 1
    assert list_receipts[0].mediated is True
    assert list_receipts[0].details["tool_count"] >= 1

    assert len(call_receipts) == 1
    assert call_receipts[0].mediated is True
    assert call_receipts[0].details.get("model_invoked") is False

    assert len(unknown_receipts) == 1
    assert "unknown_operation_fail_closed" in unknown_receipts[0].reason_codes

    assert len(direct_receipts) == 1
    assert direct_receipts[0].mediated is False
    assert direct_receipts[0].details["satisfies_mcp_mediation"] is False

    for receipt in report.receipts:
        # Every receipt binds the five required identity axes.
        assert receipt.request_identity.startswith("b")
        assert receipt.schema_identity.startswith("b")
        assert receipt.handler_identity.startswith("b")
        assert receipt.effect_identity.startswith("b")
        assert receipt.transport_identity.startswith("b")
        assert not is_pseudo_interface_cid(receipt.interface_cid)
        assert receipt.model_calls == 0


def test_direct_import_cannot_satisfy_mediation_policy() -> None:
    report = run_mcp_live_conformance()
    assert report.to_dict()["policies"]["direct_call_satisfies_mediation"] is False
    direct = [
        r for r in report.receipts if r.transport == TransportKind.DIRECT_IMPORT.value
    ][0]
    assert direct.mediated is False
    assert direct.terminal_state == ConformanceTerminalState.REFUTED.value


def test_runtime_service_authority_manifest_is_present_and_valid() -> None:
    root = _repo_root()
    authority = load_runtime_service_authority(repo_root=root)
    assert authority["schema"] == RUNTIME_SERVICE_AUTHORITY_SCHEMA
    assert authority["service_id"] == "swissknife-mcp-runtime-v1"
    assert authority["evidence_term"] == SCAEV181_EVIDENCE_TERM
    assert authority["policies"]["health_is_liveness_only"] is True
    assert authority["policies"]["mixed_checkout_state_roots_allowed"] is False
    assert authority["policies"]["pseudo_cid_allowed"] is False
    assert not is_pseudo_interface_cid(authority["configuration"]["cid"])
    assert not is_pseudo_interface_cid(authority["state"]["cid"])
    assert "ipfs_accelerate_py.mcp_server.mcplusplus.idl_registry" in authority[
        "modules"
    ]
    assert (
        "ipfs_accelerate_py.agent_supervisor.analysis.mcp_live_conformance"
        in authority["modules"]
    )


def test_build_runtime_service_identity_binds_modules_commit_config_state() -> None:
    root = _repo_root()
    identity = build_runtime_service_identity(repo_root=root)
    payload = identity.to_dict()

    assert identity.passed is True
    assert payload["interface"] == RUNTIME_SERVICE_IDENTITY_INTERFACE
    assert payload["evidence_term"] == SCAEV181_EVIDENCE_TERM
    assert len(identity.commit) == 40
    assert len(identity.tree) == 40
    assert not is_pseudo_interface_cid(identity.configuration_cid)
    assert not is_pseudo_interface_cid(identity.state_cid)
    assert not is_pseudo_interface_cid(identity.authority_cid)
    assert len(identity.modules) >= 2
    for module in identity.modules:
        assert Path(module["path"]).is_file()
        assert str(module["digest"]).startswith("sha256:")
        assert int(module["byte_length"]) > 0
    assert "modules_bound" in identity.reason_codes
    assert "configuration_cid_bound" in identity.reason_codes
    assert "state_cid_bound" in identity.reason_codes
    assert payload["policies"]["health_is_liveness_only"] is True


def test_mixed_commit_root_fails_closed() -> None:
    root = _repo_root()
    with pytest.raises(McpLiveConformanceError) as exc:
        build_runtime_service_identity(
            repo_root=root,
            expected_commit="0" * 40,
        )
    assert exc.value.reason_code == "commit_root_mismatch"


def test_service_identity_json_matches_live_receipt(tmp_path: Path) -> None:
    root = _repo_root()
    identity = build_runtime_service_identity(repo_root=root)
    out = tmp_path / "service-identity.json"
    written = write_service_identity_receipt(identity, path=out, repo_root=root)
    assert written == out
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["passed"] is True
    assert loaded["service_id"] == identity.service_id
    assert loaded["commit"] == identity.commit
    assert loaded["tree"] == identity.tree
    assert loaded["configuration_cid"] == identity.configuration_cid
    assert loaded["state_cid"] == identity.state_cid
    assert "receipt_cid" in loaded
    assert not is_pseudo_interface_cid(loaded["receipt_cid"])

    # Checked-in runtime receipt must exist and share schema/service id.
    checked_in = root / (
        "data/agent_supervisor/swissknife_contract_assurance/runtime/"
        "service-identity.json"
    )
    assert checked_in.is_file()
    committed = json.loads(checked_in.read_text(encoding="utf-8"))
    assert committed["schema"] == loaded["schema"]
    assert committed["service_id"] == loaded["service_id"]
    assert committed["configuration_cid"] == loaded["configuration_cid"]
    assert committed["state_cid"] == loaded["state_cid"]
    assert committed["passed"] is True


def test_scaev181_gate_combines_conformance_and_identity() -> None:
    root = _repo_root()
    gate = run_scaev181_mcp_runtime_gate(repo_root=root, write_identity=False)
    assert gate["evidence_term"] == SCAEV181_EVIDENCE_TERM
    assert gate["passed"] is True
    assert gate["model_calls"] == 0
    assert gate["conformance"]["passed"] is True
    assert gate["service_identity"]["passed"] is True
    assert not is_pseudo_interface_cid(gate["conformance"]["interface_cid"])
    assert "scaev181_mcp_runtime" in gate["reason_codes"]
