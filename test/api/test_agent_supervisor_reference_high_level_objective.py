"""PCPR-060: submit the Accelerate reference high-level objective."""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.assurance.reference_high_level_objective import (
    CLOSED_RELEASE_OUTCOMES,
    CURRENT_HEAD_NON_PROMOTION_VERDICT_CID,
    HERMETIC_CANDIDATE_SUITES,
    INTERFACE,
    OPERATOR_BLOCKING_TASK_ID,
    OBJECTIVE_KIND,
    OutcomeProbe,
    PCPR_060_GOAL_ID,
    PCPR_060_TASK_ID,
    PINNED_IDEA_DIGEST,
    PINNED_OBJECTIVE_CID,
    REFERENCE_OBJECTIVE_IDEA,
    SCHEMA,
    SEALED_PATH,
    SEALED_PYTHON,
    ReferenceHighLevelObjectiveError,
    admit_supervisor_objective_intent,
    current_head_static_probes,
    idea_digest,
    pcpr_060_receipt_promotion,
    platform_objective_catalog,
    qualify_current_head_reference_high_level_objective,
    qualify_reference_high_level_objective,
    refuse_idea_digest_remint,
    refuse_objective_remint,
    render_declared_reference_objective,
    verify_reference_objective_files,
)


_PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def test_closed_vocabularies_match_pcpr_060_requirements() -> None:
    assert PCPR_060_TASK_ID == "PCPR-060"
    assert PCPR_060_GOAL_ID == "PCPR-G700"
    assert INTERFACE == "AccelerateReferenceHighLevelObjective@1"
    assert SCHEMA == "ipfs_accelerate_py/assurance/reference-high-level-objective@1"
    assert OBJECTIVE_KIND == "declared_reference_high_level_objective"
    assert OPERATOR_BLOCKING_TASK_ID == (
        "pcpr-060-operator-live-objective-materialization"
    )
    assert "typed formal-logic API" in REFERENCE_OBJECTIVE_IDEA
    assert "release_candidate_qualified" in CLOSED_RELEASE_OUTCOMES
    assert "rnd_non_promoted" not in CLOSED_RELEASE_OUTCOMES
    assert HERMETIC_CANDIDATE_SUITES[-1].endswith(
        "test_agent_supervisor_reference_high_level_objective.py"
    )
    assert SEALED_PATH == "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin"
    assert SEALED_PYTHON == "/usr/bin/python3.12"
    intent = admit_supervisor_objective_intent()
    assert intent["interface"] == "SupervisorObjectiveIntent@1"
    assert intent["objective_id"] == "PCPR-G700"
    assert intent["idea_digest"] == idea_digest()
    assert intent["idea_digest"] == PINNED_IDEA_DIGEST
    document = render_declared_reference_objective()
    assert document["objective_kind"] == OBJECTIVE_KIND
    assert document["live"] is False
    assert document["applied"] is False
    assert document["submitted_live"] is False
    assert document["release_claim"] is False
    assert document["closed_release_outcome"] is None
    assert document["materialization"]["admitted"] is False
    assert document["materialization"]["duckdb_or_quack_state_written"] is False
    assert document["start"]["performed"] is False
    assert document["context_pack"]["constructed"] is False
    assert document["storage"]["stored"] is False
    assert document["caller"]["authenticated"] is False
    assert document["operator_blocking_task"]["status"] == "typed_blocked"
    assert document["objective_cid"] == PINNED_OBJECTIVE_CID
    assert document["idea_digest"] == PINNED_IDEA_DIGEST
    assert refuse_objective_remint(PINNED_OBJECTIVE_CID) == PINNED_OBJECTIVE_CID
    assert refuse_idea_digest_remint(PINNED_IDEA_DIGEST) == PINNED_IDEA_DIGEST
    with pytest.raises(ReferenceHighLevelObjectiveError, match="remints"):
        refuse_objective_remint(
            "baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        )


def test_current_head_is_rnd_non_promoted_and_not_a_release() -> None:
    verdict = qualify_current_head_reference_high_level_objective()
    assert verdict.promotion_status == "rnd_non_promoted"
    assert verdict.supervisor_disposition == "supervisor_non_promoted"
    assert verdict.closed_release_outcome is None
    assert verdict.release_claim is False
    assert verdict.completion_authoritative is False
    assert verdict.contracts_frozen is False
    assert verdict.duckdb_or_quack_state_written is False
    assert verdict.sibling_source_required is False
    assert verdict.live_objective_submission is False
    assert verdict.live_objective_submission_evidence_kind == "unavailable"
    assert verdict.live_materialization is False
    assert verdict.live_materialization_evidence_kind == "unavailable"
    assert verdict.live_start is False
    assert verdict.live_authenticated_caller is False
    assert verdict.operator_blocking_task == OPERATOR_BLOCKING_TASK_ID
    assert verdict.simulated_results_represented_as_live is False
    assert verdict.this_task_created_competing_authority is False
    assert verdict.promotion_status not in CLOSED_RELEASE_OUTCOMES
    assert verdict.verdict_cid.startswith("baguqeera")
    assert verdict.verdict_cid == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID
    assert verdict.objective_cid == PINNED_OBJECTIVE_CID
    assert verdict.idea_digest == PINNED_IDEA_DIGEST
    assert verdict.blockers == ()
    section = pcpr_060_receipt_promotion(verdict)
    assert section["promotion_status"] == "rnd_non_promoted"
    assert section["closed_release_outcome"] is None
    assert section["release_claim"] is False
    assert section["live_objective_submission"] is False
    assert section["live_materialization"] is False
    assert section["duckdb_or_quack_state_written"] is False
    assert section["verdict_cid"] == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID


def test_static_probes_show_declared_objective_constraints() -> None:
    probes = {item.probe_id: item for item in current_head_static_probes()}
    assert probes["objective_files_match_generator"].present is True
    assert probes["pyproject_reference_high_level_objective_table"].present is True
    assert probes["supervisor_objective_intent_admitted"].present is True
    assert probes["idea_digest_matches_pin"].present is True
    assert probes["direct_interface_source_present"].present is True
    assert probes["live_materialization_is_unavailable"].present is True
    assert probes["start_not_performed"].present is True
    assert probes["duckdb_or_quack_not_written"].present is True
    assert probes["context_pack_deferred_to_pcpr_061"].present is True
    assert probes["storage_deferred_to_pcpr_062"].present is True
    assert probes["pcpr_057_identities_not_reminted"].present is True
    assert probes["pcpr_041_intent_vector_not_reminted"].present is True
    assert probes["operator_blocking_task_emitted"].present is True
    assert probes["no_closed_release_outcome"].present is True
    assert probes["compatibility_identities_reminted"].present is False
    assert probes["simulated_results_represented_as_live"].present is False
    assert probes["direct_database_bypass_used"].present is False
    assert probes["live_objective_submission"].evidence_kind == "unavailable"
    assert probes["live_materialization"].evidence_kind == "unavailable"
    for probe in probes.values():
        assert probe.live is False
        assert probe.simulated_represented_as_live is False


def test_committed_files_match_generator() -> None:
    verified = verify_reference_objective_files()
    assert verified["ok"] is True
    assert verified["missing"] == []
    catalog = platform_objective_catalog()
    assert catalog["interface"] == "PlatformReferenceHighLevelObjective@1"
    assert catalog["release_claim"] is False
    assert catalog["closed_release_outcome"] is None
    assert catalog["live_objective_submission"] is False
    assert catalog["live_materialization"] is False
    assert catalog["objective_cid"] == PINNED_OBJECTIVE_CID
    assert catalog["idea_digest"] == PINNED_IDEA_DIGEST
    assert catalog["components"]["ipfs_accelerate_py"]["status"] == "observed"
    assert catalog["components"]["ipfs_datasets_py"]["binding"]["status"] == "observed"
    assert catalog["components"]["ipfs_kit_py"]["binding"]["status"] == "observed"
    pyproject = (_PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'interface = "AccelerateReferenceHighLevelObjective@1"' in pyproject


def test_simulated_live_probe_is_rejected() -> None:
    with pytest.raises(ReferenceHighLevelObjectiveError, match="simulated"):
        qualify_reference_high_level_objective(
            (
                OutcomeProbe(
                    probe_id="bogus",
                    present=False,
                    evidence_kind="simulated",
                    live=False,
                    simulated_represented_as_live=True,
                    reason="must fail",
                ),
            ),
            objective_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            idea_digest_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            intent_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            catalog_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        )


def test_live_claim_without_measured_live_evidence_is_rejected() -> None:
    with pytest.raises(ReferenceHighLevelObjectiveError, match="measured_live"):
        qualify_reference_high_level_objective(
            (
                OutcomeProbe(
                    probe_id="bogus",
                    present=True,
                    evidence_kind="measured",
                    live=True,
                    simulated_represented_as_live=False,
                    reason="must fail",
                ),
            ),
            objective_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            idea_digest_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            intent_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            catalog_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        )
