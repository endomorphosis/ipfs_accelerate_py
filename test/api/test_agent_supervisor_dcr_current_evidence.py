"""DCR-010 tests for byte-bound current evidence reconciliation."""

from __future__ import annotations

from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.analysis.deterministic_repair_current_state import (
    CURRENT_IMPLEMENTATION_EVIDENCE_INTERFACE,
    CurrentComponentStatus,
    CurrentEvidenceComponentSpec,
    reconcile_current_evidence,
)


def write(root: Path, relative: str, text: str) -> str:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return relative


def synthetic_paths(root: Path, *, legacy: bool = True) -> None:
    gate = "allow_legacy_residual: bool = True\nlegacy_worker_prompt_residual\nimplementation_disposition_cid("
    daemon = "pre_implementation_provider_gate(allow_legacy_residual=True)\n"
    if not legacy:
        gate = "allow_legacy_residual: bool = False\n"
        daemon = "pre_implementation_provider_gate(allow_legacy_residual=False)\n"
    write(
        root,
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/pre_implementation_provider_gate.py",
        gate,
    )
    write(
        root,
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/pre_implementation_kernel.py",
        "planner_available: bool = True\ndoctor_available: bool = True\n",
    )
    write(
        root,
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py",
        daemon,
    )


def test_current_component_is_byte_bound_and_repeatable(tmp_path: Path) -> None:
    source = write(tmp_path, "planner.py", "class Planner:\n    pass\n# wired:doctor\n")
    synthetic_paths(tmp_path)
    spec = CurrentEvidenceComponentSpec(
        component_id="planner",
        family="Planner",
        paths=(source,),
        required_markers=("class Planner",),
        wiring_markers=("wired:doctor",),
    )
    one = reconcile_current_evidence(tmp_path, (spec,), commit_id="commit:fixture")
    two = reconcile_current_evidence(tmp_path, (spec,), commit_id="commit:fixture")
    assert CURRENT_IMPLEMENTATION_EVIDENCE_INTERFACE == "CurrentImplementationEvidence@1"
    assert one.to_dict() == two.to_dict()
    assert one.components[0].status is CurrentComponentStatus.IMPLEMENTED_CURRENT
    assert one.components[0].file_digests[source].startswith("sha256:")
    assert one.repair_ready is False
    assert one.to_dict()["provider_or_llm_invoked"] is False


@pytest.mark.parametrize(
    ("spec", "expected"),
    (
        (
            CurrentEvidenceComponentSpec("missing", "WPD", ("missing.py",)),
            CurrentComponentStatus.INCOMPLETE,
        ),
        (
            CurrentEvidenceComponentSpec(
                "incomplete", "SCA", ("component.py",), required_markers=("required",)
            ),
            CurrentComponentStatus.INCOMPLETE,
        ),
        (
            CurrentEvidenceComponentSpec(
                "unwired", "RPR", ("component.py",), wiring_markers=("wire",)
            ),
            CurrentComponentStatus.UNWIRED,
        ),
        (
            CurrentEvidenceComponentSpec(
                "conflict", "Doctor", ("component.py",), conflict_markers=("<<<<<<<",)
            ),
            CurrentComponentStatus.CONFLICTING,
        ),
    ),
)
def test_every_component_gets_exactly_one_fail_closed_status(
    tmp_path: Path, spec: CurrentEvidenceComponentSpec, expected: CurrentComponentStatus
) -> None:
    write(tmp_path, "component.py", "plain\n<<<<<<< conflict\n")
    synthetic_paths(tmp_path, legacy=False)
    row = reconcile_current_evidence(
        tmp_path, (spec,), dirty_overlay_identity="overlay:fixture"
    ).components[0]
    assert row.status is expected
    assert row.status.value in {item.value for item in CurrentComponentStatus}


def test_stale_digest_and_synthetic_planner_doctor_legacy_path_are_reported(tmp_path: Path) -> None:
    source = write(tmp_path, "live.py", "def live(): pass\n")
    synthetic_paths(tmp_path, legacy=True)
    spec = CurrentEvidenceComponentSpec(
        component_id="live-wiring",
        family="live-wiring",
        paths=(source,),
        expected_digests={source: "sha256:old"},
    )
    evidence = reconcile_current_evidence(tmp_path, (spec,))
    assert evidence.components[0].status is CurrentComponentStatus.STALE
    finding = evidence.synthetic_planner_doctor_path
    assert finding.detected
    assert finding.status is CurrentComponentStatus.CONFLICTING
    assert {
        "legacy_residual_allowance",
        "live_daemon_enables_legacy_residual",
        "legacy_residual_packet_cid",
        "synthetic_planner_availability",
        "synthetic_doctor_availability",
        "synthetic_disposition_cid",
    }.issubset(finding.flags)
    assert evidence.snapshot_kind == "dirty_overlay"


def test_mutually_exclusive_snapshot_identities_reject(tmp_path: Path) -> None:
    synthetic_paths(tmp_path, legacy=False)
    with pytest.raises(ValueError, match="mutually exclusive"):
        reconcile_current_evidence(
            tmp_path, (), commit_id="commit:1", dirty_overlay_identity="overlay:1"
        )
