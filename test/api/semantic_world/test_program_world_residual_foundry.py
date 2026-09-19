"""SAWM-030 residual-intelligence corpus, training, and calibration."""

from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.self_improvement.program_world_residual_foundry import (
    FAMILIES,
    ProgramWorldCalibrationMonitor,
    ProgramWorldResidualFoundry,
    ProgramWorldTaskFamily,
    admit_program_world_checkpoint,
    prepare_program_world_training,
)


def test_missing_corpus_is_training_unavailable() -> None:
    result = prepare_program_world_training("call_ranking")
    assert result["training_unavailable"] is True
    assert result["completion_authority"] is False


def test_admitted_family_checkpoint_is_proposal_only() -> None:
    foundry = ProgramWorldResidualFoundry(
        families={
            name: ProgramWorldTaskFamily(name=name, corpus_admitted=True, split_frozen=True)
            for name in FAMILIES
        }
    )
    admission = admit_program_world_checkpoint("call_ranking", foundry=foundry)
    assert admission.admitted is True
    assert admission.proposal_only is True
    assert admission.completion_authority is False


def test_drift_abstains_from_promotion() -> None:
    foundry = ProgramWorldResidualFoundry(
        families={
            "call_ranking": ProgramWorldTaskFamily(
                name="call_ranking", corpus_admitted=True, split_frozen=True
            )
        },
        monitor=ProgramWorldCalibrationMonitor(drift=True),
    )
    admission = admit_program_world_checkpoint("call_ranking", foundry=foundry)
    assert admission.admitted is False
    assert admission.reason_code == "drift_or_ood"
