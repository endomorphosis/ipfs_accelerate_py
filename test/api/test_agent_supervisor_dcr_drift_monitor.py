"""DCR-104: continuous drift detection and evidence invalidation.

Acceptance:
* Relevant drift reopens exactly affected state.
* Irrelevant changes reuse reconstructed evidence.
* Two unchanged scans are a no-op with zero model/provider calls.
* Scans cannot auto-weaken contracts or add operator semantics.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.autonomous_repair.drift_monitor import (
    AFFECTED_EVIDENCE_CLOSURE_INTERFACE,
    CONTRACT_DRIFT_MONITOR_INTERFACE,
    DEFAULT_DRIFT_POLICY_PATH,
    DCR_DRIFT_EVIDENCE,
    DCR_TASK_ID,
    PROOF_INVALIDATION_INTERFACE,
    ContractDriftMonitor,
    InvalidationAction,
    materialize_drift_policy,
    run_drift_monitor_suite,
)


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parents[4], here.parents[3], Path.cwd()):
        if (candidate / "config" / "deterministic_contract_repair_services.json").is_file():
            return candidate
    return here.parents[4]


@pytest.fixture(scope="module")
def suite() -> dict:
    return run_drift_monitor_suite(repo_root=_repo_root())


def test_interfaces_and_symbols() -> None:
    assert CONTRACT_DRIFT_MONITOR_INTERFACE == "ContractDriftMonitor@1"
    assert PROOF_INVALIDATION_INTERFACE == "ProofInvalidation@1"
    assert AFFECTED_EVIDENCE_CLOSURE_INTERFACE == "AffectedEvidenceClosure@1"
    assert DCR_TASK_ID == "DCR-104"
    assert DCR_DRIFT_EVIDENCE == "dcr/contract-drift-monitor@1"
    assert callable(run_drift_monitor_suite)
    assert ContractDriftMonitor.INTERFACE == CONTRACT_DRIFT_MONITOR_INTERFACE


def test_suite_passes_zero_llm(suite: dict) -> None:
    assert suite["passed"] is True
    assert suite["runtime_model_calls"] == 0
    assert suite["provider_calls"] == 0
    assert "two_unchanged_scans_noop" in suite["reason_codes"]
    assert "relevant_drift_reopens_affected" in suite["reason_codes"]


def test_two_unchanged_scans_are_noop(suite: dict) -> None:
    assert suite["scan_unchanged_a"]["noop"] is True
    assert suite["scan_unchanged_b"]["noop"] is True
    assert suite["scan_unchanged_a"]["runtime_model_calls"] == 0
    assert suite["scan_unchanged_b"]["runtime_model_calls"] == 0


def test_relevant_policy_drift_invalidates_closure(suite: dict) -> None:
    relevant = suite["scan_relevant"]
    assert relevant["noop"] is False
    closure = set(relevant["closure"]["closure"])
    assert "policy" in closure
    assert "canary" in closure  # transitive
    assert any(inv["reopened"] for inv in relevant["invalidations"])
    for inv in relevant["invalidations"]:
        assert inv["contract_weakened"] is False
        assert inv["operator_semantics_added"] is False
        assert inv["health_from_stale_receipt"] is False


def test_irrelevant_change_is_noop_or_reuse(suite: dict) -> None:
    irrelevant = suite["scan_irrelevant"]
    # Unmonitored path should not produce invalidations.
    assert irrelevant["noop"] is True or not irrelevant["invalidations"]


def test_monitor_closure_mapping() -> None:
    mon = ContractDriftMonitor(repo_root=_repo_root())
    mon.observe_baseline()
    closure = mon.build_closure(
        ("config/deterministic_contract_repair_policy.json",)
    )
    assert "policy" in closure.seed_evidence
    assert "canary" in closure.closure
    assert "release" in closure.closure


def test_materialize_drift_policy(tmp_path: Path) -> None:
    dest = tmp_path / "drift-policy.json"
    payload = materialize_drift_policy(repo_root=_repo_root(), destination=dest)
    assert dest.is_file()
    on_disk = json.loads(dest.read_text(encoding="utf-8"))
    assert on_disk["interface"] == CONTRACT_DRIFT_MONITOR_INTERFACE
    assert on_disk["task_id"] == DCR_TASK_ID
    assert on_disk["result"]["passed"] is True
    assert on_disk["result"]["policy"]["forbid_auto_weaken"] is True
    assert payload["result"]["passed"] is True


def test_default_path() -> None:
    assert DEFAULT_DRIFT_POLICY_PATH.endswith("drift-policy.json")
