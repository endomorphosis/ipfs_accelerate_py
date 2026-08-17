"""DCR-092: SwissKnife desktop/browser mediation contract repair e2e.

Acceptance:
* Real source/policy alignment becomes conformant on a new epoch.
* Raw proxy mutation is denied; mutations use GovernedMcpMediator.
* Every model/provider counter remains zero.
* Disposable fixture + loopback only.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.desktop_contract_repair_e2e import (
    DEFAULT_DESKTOP_E2E_PATH,
    DCR_DESKTOP_E2E_EVIDENCE,
    DCR_TASK_ID,
    DESKTOP_CONTRACT_REPAIR_E2E_INTERFACE,
    DesktopContractRepairE2E,
    DesktopContractRepairError,
    GovernedMutationAssertion,
    RepairPhase,
    materialize_desktop_e2e,
    run_desktop_contract_repair_e2e,
)
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.operators.transport_repairs import (
    GOVERNED_MCP_MEDIATOR_INTERFACE,
    GOVERNED_MUTATION_ROUTE,
    MethodEffectClass,
)


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parents[4], here.parents[3], Path.cwd()):
        if (candidate / "config" / "deterministic_contract_repair_services.json").is_file():
            return candidate
    return here.parents[4]


@pytest.fixture(scope="module")
def e2e() -> DesktopContractRepairE2E:
    return run_desktop_contract_repair_e2e(repo_root=_repo_root())


def test_interfaces_and_symbols() -> None:
    assert DESKTOP_CONTRACT_REPAIR_E2E_INTERFACE == "DesktopContractRepairE2E@1"
    assert DesktopContractRepairE2E.INTERFACE == DESKTOP_CONTRACT_REPAIR_E2E_INTERFACE
    assert GovernedMutationAssertion.INTERFACE == "GovernedMutationAssertion@1"
    assert DCR_TASK_ID == "DCR-092"
    assert DCR_DESKTOP_E2E_EVIDENCE == "dcr/desktop-contract-repair-e2e@1"
    assert callable(run_desktop_contract_repair_e2e)
    assert GOVERNED_MCP_MEDIATOR_INTERFACE == "GovernedMcpMediator@1"


def test_e2e_passes_with_zero_model_and_provider_counters(e2e: DesktopContractRepairE2E) -> None:
    assert e2e.passed is True
    assert e2e.runtime_model_calls == 0
    assert e2e.provider_calls == 0
    assert e2e.live_precondition_ok is True
    payload = e2e.to_dict()
    assert payload["runtime_model_calls"] == 0
    assert payload["provider_calls"] == 0
    assert payload["passed"] is True
    assert payload["mediator_interface"] == GOVERNED_MCP_MEDIATOR_INTERFACE
    assert payload["governed_mutation_route"] == GOVERNED_MUTATION_ROUTE


def test_epoch_advances_after_repair(e2e: DesktopContractRepairE2E) -> None:
    assert e2e.epoch_before != e2e.epoch_after
    assert e2e.source_diff["before"]["policy_id"] != e2e.source_diff["after"]["policy_id"]
    assert e2e.source_diff["destructive_production_tools"] is False
    assert e2e.source_diff["write_authority_granted"] is False
    with pytest.raises(DesktopContractRepairError):
        DesktopContractRepairE2E(
            passed=True,
            fixture_id=e2e.fixture_id,
            original_counterexample=e2e.original_counterexample,
            source_diff=e2e.source_diff,
            phase_receipts=e2e.phase_receipts,
            mutation_assertions=e2e.mutation_assertions,
            browser_trace=e2e.browser_trace,
            epoch_before=e2e.epoch_before,
            epoch_after=e2e.epoch_before,  # same epoch must fail when passed
            graph_proof_roots=e2e.graph_proof_roots,
            rollback_replay=e2e.rollback_replay,
            live_precondition_ok=True,
            reason_codes=("bad",),
        )


def test_raw_proxy_mutations_denied(e2e: DesktopContractRepairE2E) -> None:
    mutate = [
        a
        for a in e2e.mutation_assertions
        if a.effect_class == MethodEffectClass.MUTATE.value
        and a.service_path.startswith("/mcp/services/")
    ]
    assert mutate
    for assertion in mutate:
        assert assertion.allowed is False
        assert assertion.raw_proxy_denied is True
        assert "governed" in assertion.decision or "reject" in assertion.decision


def test_all_repair_phases_recorded(e2e: DesktopContractRepairE2E) -> None:
    phases = {item["phase"] for item in e2e.phase_receipts}
    for phase in RepairPhase:
        assert phase.value in phases
    assert all(item["ok"] for item in e2e.phase_receipts)
    assert all(item["runtime_model_calls"] == 0 for item in e2e.phase_receipts)


def test_rollback_replay_restores_and_realigns(e2e: DesktopContractRepairE2E) -> None:
    assert e2e.rollback_replay["rollback_ok"] is True
    assert e2e.rollback_replay["replay_ok"] is True
    assert e2e.rollback_replay["inverse_epoch"] == e2e.epoch_before


def test_browser_trace_and_counterexample(e2e: DesktopContractRepairE2E) -> None:
    assert e2e.original_counterexample["kind"]
    assert e2e.browser_trace
    events = {item.get("event") for item in e2e.browser_trace}
    assert "counterexample_observed" in events
    assert "policy_preview_applied" in events


def test_materialize_desktop_e2e(tmp_path: Path) -> None:
    dest = tmp_path / "desktop-e2e.json"
    payload = materialize_desktop_e2e(repo_root=_repo_root(), destination=dest)
    assert dest.is_file()
    on_disk = json.loads(dest.read_text(encoding="utf-8"))
    assert on_disk["interface"] == DESKTOP_CONTRACT_REPAIR_E2E_INTERFACE
    assert on_disk["task_id"] == DCR_TASK_ID
    assert on_disk["result"]["passed"] is True
    assert on_disk["runtime_model_calls"] == 0
    assert payload["result"]["passed"] is True


def test_default_artifact_path() -> None:
    assert DEFAULT_DESKTOP_E2E_PATH.endswith("desktop-e2e.json")
