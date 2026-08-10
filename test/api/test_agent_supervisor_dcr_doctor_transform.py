"""DCR-052: Doctor transform selection, impact bounding, no prose bodies."""

from __future__ import annotations

from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.analysis.mcp_contract_graph import (
    ContractAuthority,
    ConsumerPathInput,
    SourceSpan,
    StageEndpoint,
    build_mcp_contract_graph,
)
from ipfs_accelerate_py.agent_supervisor.planning.dcr_doctor_transform import (
    DOCTOR_TRANSFORM_PROPOSAL_INTERFACE,
    REPAIR_OPERATOR_INTERFACE,
    DoctorTransformDisposition,
    DoctorTransformProposal,
    materialize_doctor_transforms,
    prove_impact,
    synthesize_transform,
)
from ipfs_accelerate_py.agent_supervisor.planning.deterministic_doctor_synthesis import (
    DoctorTransformProposal as SynthesisExportedProposal,
    prove_impact as synthesis_prove_impact,
    synthesize_transform as synthesis_synthesize_transform,
)
from ipfs_accelerate_py.agent_supervisor.sca_doctor_bridge import (
    DoctorDiagnosisDisposition,
    diagnose_contract_failure,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.worker_doctor_bridge import (
    synthesize_transform_from_diagnosis,
)


def _endpoint(stage: str, **kw) -> StageEndpoint:
    return StageEndpoint(
        stage=stage,
        stable_key=kw.get("key", stage),
        label=stage,
        authority=kw.get("authority", ContractAuthority.REVIEWED_DECLARATION),
        owning_root=kw.get("owning_root", "swissknife"),
        payload={},
        source_refs=(f"src/{stage}.ts",),
        span=SourceSpan(
            path=f"src/{stage}.ts",
            root_id=kw.get("owning_root", "swissknife"),
        ),
    )


def _partial_graph(snapshot_id: str = "snap:dcr052"):
    endpoints = (
        _endpoint("ui_action", key="ui"),
        _endpoint("descriptor", key="d"),
        _endpoint("orb_idl", key="o"),
        _endpoint(
            "mcp_method_schema",
            key="m",
            owning_root="Mcp-Plus-Plus",
        ),
        _endpoint(
            "mediator",
            key="med",
            authority=ContractAuthority.POLICY,
        ),
        _endpoint("route", key="r"),
    )
    return build_mcp_contract_graph(
        snapshot_id=snapshot_id,
        consumers=(
            ConsumerPathInput(
                consumer_id="c:dcr052",
                package="ipfs_accelerate_py",
                operation="tools.call.echo",
                owning_root="swissknife",
                transport="stdio",
                profile="mcp++/default",
                aliases=("echo",),
                declaration={
                    "method": "tools/call",
                    "tool": "echo",
                    "schema_root": "schemas/echo.json",
                    "input_schema": {"type": "object"},
                },
                endpoints=endpoints,
            ),
        ),
    )


def test_interfaces_exported() -> None:
    assert DOCTOR_TRANSFORM_PROPOSAL_INTERFACE == "DoctorTransformProposal@1"
    assert REPAIR_OPERATOR_INTERFACE == "RepairOperator@1"
    assert SynthesisExportedProposal is DoctorTransformProposal
    assert synthesis_synthesize_transform is synthesize_transform
    assert synthesis_prove_impact is prove_impact


def test_synthesize_transform_selects_registered_operator_without_bodies() -> None:
    diagnosis = diagnose_contract_failure(
        _partial_graph(), require_shared_epoch=False
    )
    assert diagnosis.disposition is DoctorDiagnosisDisposition.DIAGNOSED
    receipt = synthesize_transform(diagnosis)
    assert receipt.disposition is DoctorTransformDisposition.PROPOSED
    assert receipt.proposal is not None
    assert isinstance(receipt.proposal, DoctorTransformProposal)
    assert receipt.grants_transform_authority is False
    assert receipt.proposal.grants_write_authority is False
    assert receipt.proposal.grants_transform_authority is False
    assert receipt.runtime_model_calls == 0
    assert "body_free" in receipt.reason_codes or "authority_not_granted" in (
        receipt.proposal.reason_codes
    )
    # No prose source bodies in arguments.
    for value in receipt.proposal.arguments.values():
        assert "\n" not in value
        assert "def " not in value


def test_non_actionable_diagnosis_abstains() -> None:
    diagnosis = diagnose_contract_failure(
        _partial_graph("snap:dcr052-bad"),
        {"schema": "not-live", "evidence_term": "wrong"},
        require_shared_epoch=True,
    )
    receipt = synthesize_transform(diagnosis)
    assert receipt.disposition is DoctorTransformDisposition.ABSTAIN_REVIEW
    assert receipt.proposal is None
    assert receipt.grants_transform_authority is False
    assert "no_transform" in receipt.reason_codes


def test_prove_impact_bounds_without_granting_authority() -> None:
    diagnosis = diagnose_contract_failure(
        _partial_graph("snap:dcr052-impact"), require_shared_epoch=False
    )
    synth = synthesize_transform(diagnosis)
    assert synth.proposal is not None
    proved = prove_impact(synth)
    assert proved.disposition is DoctorTransformDisposition.PROPOSED
    assert proved.impact is not None
    assert proved.grants_transform_authority is False
    assert proved.runtime_model_calls == 0
    assert "impact_bounded" in proved.reason_codes


def test_worker_bridge_wrapper_and_materialize(tmp_path: Path) -> None:
    diagnosis = diagnose_contract_failure(
        _partial_graph("snap:dcr052-worker"), require_shared_epoch=False
    )
    wrapped = synthesize_transform_from_diagnosis(diagnosis)
    assert wrapped.grants_transform_authority is False
    dest = tmp_path / "doctor-transforms.json"
    payload = materialize_doctor_transforms(diagnosis=diagnosis, destination=dest)
    assert dest.is_file()
    assert payload["runtime_model_calls"] == 0
    assert payload["interface"] == DOCTOR_TRANSFORM_PROPOSAL_INTERFACE
