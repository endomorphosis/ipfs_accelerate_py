"""DCR-091: live initialize/list/call/logic equivalence for all servers.

Acceptance:
* Accelerate, datasets, and kit are required from one manifest.
* Process-local proof cannot substitute for MCP reachability.
* Discovery/transport errors never appear as empty success.
* Required service/profile/tool is conformant or typed unsupported.
* Runtime model calls remain 0.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.live_service_conformance import (
    DEFAULT_LIVE_CONFORMANCE_PATH,
    DCR_LIVE_CONFORMANCE_EVIDENCE,
    DCR_TASK_ID,
    LIVE_MCP_CONFORMANCE_INTERFACE,
    LiveConformanceResult,
    LiveServiceConformanceError,
    LiveThreeServiceConformance,
    LogicRouteEquivalence,
    ReachabilityStatus,
    assess_live_services,
    materialize_live_conformance,
)
from ipfs_accelerate_py.agent_supervisor.analysis.mcp_live_observer import (
    MCP_PLUS_PROFILES_A_F,
    REQUIRED_SERVICE_ROLES,
    SAFE_TOOLS_CALL,
)


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parents[4], here.parents[3], Path.cwd()):
        if (candidate / "config" / "deterministic_contract_repair_services.json").is_file():
            return candidate
    return here.parents[4]


@pytest.fixture(scope="module")
def live_result() -> LiveConformanceResult:
    return assess_live_services(
        repo_root=_repo_root(),
        include_loopback_probes=True,
        stable_process_identity=True,
        require_hermetic_precondition=True,
    )


def test_interfaces_and_symbols() -> None:
    assert LIVE_MCP_CONFORMANCE_INTERFACE == "LiveMcpConformance@1"
    assert LiveConformanceResult.INTERFACE == LIVE_MCP_CONFORMANCE_INTERFACE
    assert LiveThreeServiceConformance.INTERFACE == "LiveThreeServiceConformance@1"
    assert LogicRouteEquivalence.INTERFACE == "LogicRouteEquivalence@1"
    assert DCR_TASK_ID == "DCR-091"
    assert DCR_LIVE_CONFORMANCE_EVIDENCE == "dcr/live-service-conformance@1"
    assert callable(assess_live_services)
    assert set(SAFE_TOOLS_CALL) == set(REQUIRED_SERVICE_ROLES)


def test_all_three_services_required_and_observed(live_result: LiveConformanceResult) -> None:
    assert set(live_result.roles_observed) == set(REQUIRED_SERVICE_ROLES)
    assert live_result.three_service.all_roles_required is True
    assert set(live_result.three_service.roles) == set(REQUIRED_SERVICE_ROLES)
    assert set(live_result.three_service.role_status) == set(REQUIRED_SERVICE_ROLES)
    for role in REQUIRED_SERVICE_ROLES:
        status = live_result.three_service.role_status[role]
        assert status["safe_tool"] == SAFE_TOOLS_CALL[role]
        assert status["conformant"] is True
        assert status["initialize_ok"] is True
        assert status["tools_list_ok"] is True
        assert status["tools_call_ok"] is True
        assert status["fail_closed_ok"] is True
        assert status["profiles_ok"] is True
        assert status["profile_count"] == len(MCP_PLUS_PROFILES_A_F)


def test_process_local_cannot_substitute_for_mcp_reachability(
    live_result: LiveConformanceResult,
) -> None:
    for role, reach in live_result.reachability.items():
        assert reach != ReachabilityStatus.PROCESS_LOCAL_ONLY.value, role
        assert reach in {
            ReachabilityStatus.MCP_IN_PROCESS.value,
            ReachabilityStatus.MCP_LOOPBACK.value,
            ReachabilityStatus.TYPED_UNSUPPORTED.value,
        }
    # Constructor fail-closed: process-local-only + passed is rejected.
    with pytest.raises(LiveServiceConformanceError):
        LiveConformanceResult(
            passed=True,
            service_id="x",
            three_service=live_result.three_service,
            logic_equivalence=live_result.logic_equivalence,
            hermetic_precondition_ok=True,
            transcript_cid="sha256:" + "0" * 64,
            roles_observed=tuple(REQUIRED_SERVICE_ROLES),
            reachability={
                role: ReachabilityStatus.PROCESS_LOCAL_ONLY.value
                for role in REQUIRED_SERVICE_ROLES
            },
            reason_codes=("bad",),
        )


def test_no_empty_success_from_discovery_or_transport(
    live_result: LiveConformanceResult,
) -> None:
    assert live_result.three_service.empty_success_violations == ()
    assert "no_empty_success_from_errors" in live_result.three_service.reason_codes


def test_logic_route_equivalence(live_result: LiveConformanceResult) -> None:
    logic = live_result.logic_equivalence
    assert logic.tool
    assert logic.canonically_equivalent is True
    assert "logic_cec_prove_canonically_equivalent" in logic.reason_codes
    payload = logic.to_dict()
    assert payload["content_id"].startswith("sha256:")


def test_live_conformance_passes_with_zero_model_calls(
    live_result: LiveConformanceResult,
) -> None:
    assert live_result.passed is True
    assert live_result.runtime_model_calls == 0
    assert live_result.hermetic_precondition_ok is True
    assert live_result.transcript_cid
    assert "live_conformance_passed" in live_result.reason_codes
    payload = live_result.to_dict()
    assert payload["interface"] == LIVE_MCP_CONFORMANCE_INTERFACE
    assert payload["runtime_model_calls"] == 0
    assert payload["passed"] is True
    assert payload["content_id"].startswith("sha256:")


def test_materialize_live_conformance(tmp_path: Path) -> None:
    dest = tmp_path / "live-conformance.json"
    payload = materialize_live_conformance(
        repo_root=_repo_root(),
        destination=dest,
        stable_process_identity=True,
    )
    assert dest.is_file()
    on_disk = json.loads(dest.read_text(encoding="utf-8"))
    assert on_disk["interface"] == LIVE_MCP_CONFORMANCE_INTERFACE
    assert on_disk["task_id"] == DCR_TASK_ID
    assert on_disk["runtime_model_calls"] == 0
    assert on_disk["result"]["passed"] is True
    assert set(on_disk["result"]["roles_observed"]) == set(REQUIRED_SERVICE_ROLES)
    assert payload["result"]["passed"] is True


def test_default_artifact_path() -> None:
    assert DEFAULT_LIVE_CONFORMANCE_PATH.endswith("live-conformance.json")
