"""DCR-103: publish deterministic repair release and operator policy.

Acceptance:
* Fresh verification reproduces pins/evidence with zero model/provider calls.
* Unresolved typed gaps and auto-safe boundary are named.
* Compatibility claims do not exceed live evidence.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.evaluation.dcr_release import (
    DEFAULT_OPS_DOC_PATH,
    DEFAULT_RELEASE_PATH,
    DETERMINISTIC_REPAIR_RELEASE_INTERFACE,
    DCR_RELEASE_EVIDENCE,
    DCR_TASK_ID,
    OPERATOR_POLICY_ROOT_INTERFACE,
    DeterministicRepairRelease,
    materialize_release,
    publish_deterministic_repair_release,
    verify_release,
)


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parents[4], here.parents[3], Path.cwd()):
        if (candidate / "config" / "deterministic_contract_repair_services.json").is_file():
            return candidate
    return here.parents[4]


@pytest.fixture(scope="module")
def release() -> DeterministicRepairRelease:
    return publish_deterministic_repair_release(repo_root=_repo_root())


def test_interfaces_and_symbols() -> None:
    assert DETERMINISTIC_REPAIR_RELEASE_INTERFACE == "DeterministicRepairRelease@1"
    assert OPERATOR_POLICY_ROOT_INTERFACE == "OperatorPolicyRoot@1"
    assert DeterministicRepairRelease.INTERFACE == DETERMINISTIC_REPAIR_RELEASE_INTERFACE
    assert DCR_TASK_ID == "DCR-103"
    assert DCR_RELEASE_EVIDENCE == "dcr/deterministic-repair-release@1"
    assert callable(verify_release)
    assert callable(publish_deterministic_repair_release)


def test_release_passes_and_zero_llm(release: DeterministicRepairRelease) -> None:
    assert release.passed is True
    assert release.runtime_model_calls == 0
    assert release.provider_calls == 0
    assert release.compatibility_claims.get("exceeds_live_evidence") is False
    assert release.auto_safe_boundary == "auto_safe"
    check = verify_release(release, repo_root=_repo_root())
    assert check["ok"] is True
    assert check["errors"] == []


def test_pins_and_evidence_present(release: DeterministicRepairRelease) -> None:
    assert release.pins.get("monorepo_head")
    assert release.pins.get("ipfs_accelerate")
    assert release.pins.get("policy_sha256", "").startswith("sha256:")
    for key in (
        "hermetic_conformance",
        "live_conformance",
        "desktop_e2e",
        "adversarial",
        "fixed_point",
        "benchmark",
        "shadow",
        "canary",
        "policy",
    ):
        assert key in release.evidence_cids
        assert release.evidence_cids[key].startswith("sha256:")


def test_unresolved_typed_named(release: DeterministicRepairRelease) -> None:
    assert release.unresolved_typed
    statuses = {row["status"] for row in release.unresolved_typed}
    assert "unsupported" in statuses or "review_required" in statuses


def test_operator_policy_root(release: DeterministicRepairRelease) -> None:
    assert release.operator_policy.INTERFACE == OPERATOR_POLICY_ROOT_INTERFACE
    assert release.operator_policy.mode == "auto_safe"
    assert release.operator_policy.allowlisted_operators
    assert release.operator_policy.always_abstain_families


def test_materialize_release_and_ops_doc(tmp_path: Path) -> None:
    rel = tmp_path / "release.json"
    ops = tmp_path / "deterministic-contract-repair-operations.md"
    payload = materialize_release(
        repo_root=_repo_root(),
        release_destination=rel,
        ops_destination=ops,
    )
    assert rel.is_file()
    assert ops.is_file()
    on_disk = json.loads(rel.read_text(encoding="utf-8"))
    assert on_disk["interface"] == DETERMINISTIC_REPAIR_RELEASE_INTERFACE
    assert on_disk["result"]["passed"] is True
    text = ops.read_text(encoding="utf-8")
    assert "Deterministic Contract Repair Operations" in text
    assert "Rollback procedure" in text
    assert "Unresolved typed gaps" in text
    assert payload["result"]["passed"] is True


def test_default_paths() -> None:
    assert DEFAULT_RELEASE_PATH.endswith("release.json")
    assert DEFAULT_OPS_DOC_PATH.endswith("deterministic-contract-repair-operations.md")
