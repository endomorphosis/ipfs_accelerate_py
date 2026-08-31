"""PCPR-002 fail-closed canonical supervisor-contract freeze."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.validation.canonical_supervisor_contract_freeze import (
    CANONICAL_CONTRACT_CATALOG,
    CLOSED_RELEASE_OUTCOMES,
    COMPETING_AUTHORITY_PROHIBITIONS,
    CONTRACT_FREEZE_INTERFACE,
    CURRENT_HEAD_ACCELERATOR_COMMIT,
    CURRENT_HEAD_DATASETS_COMMIT,
    CURRENT_HEAD_KIT_COMMIT,
    CURRENT_HEAD_NON_PROMOTION_VERDICT_CID,
    CURRENT_HEAD_UNAVAILABLE_VERDICT_CID,
    HERMETIC_CANDIDATE_SUITES,
    PCPR_002_GOAL_ID,
    PCPR_002_TASK_ID,
    PLACEHOLDER_SCHEMA,
    PUBLIC_SUPERVISOR_SURFACES,
    REQUIRED_CONTRACT_NAMES,
    REQUIRED_FREEZE_SURFACES,
    SURFACE_CONTRACT_MAP,
    CanonicalSupervisorContractFreezeError,
    CompetingAuthorityObservation,
    ContractObservation,
    OperatorOverride,
    QualificationPrerequisite,
    SurfaceObservation,
    current_head_competing_authorities,
    current_head_contract_observations,
    current_head_operator_override,
    current_head_pcpr_002_current_tree_binding,
    current_head_pcpr_002_receipt_promotion,
    current_head_pcpr_002_receipt_sections,
    current_head_qualification_prerequisite,
    current_head_surface_observations,
    freeze_canonical_supervisor_contracts,
    pcpr_002_current_tree_binding,
    pcpr_002_receipt_promotion,
    qualify_current_head_without_supervisor_promotion,
    validate_pcpr_002_outer_receipt,
)


def _clone_surface(item: SurfaceObservation, **changes: object) -> SurfaceObservation:
    payload = item.to_mapping()
    payload.update(changes)
    return SurfaceObservation(**payload)


def _live_promoted_qualification() -> QualificationPrerequisite:
    return QualificationPrerequisite(
        task_id="PCPR-001",
        promotion_status="supervisor_promoted",
        supervisor_disposition="supervisor_promoted",
        live_qualification_evidence_kind="measured_live",
        verdict_cid="baguqeeralivepcpr001supervisorpromotedqualificationaaaaaaa",
        live_campaign_identity="pcpr-live-cohort",
        missed_live_cohort_count=0,
        missed_target_count=0,
        reason="fixture live promotion; not current-head evidence",
    )


def _normative_surfaces() -> tuple[SurfaceObservation, ...]:
    return tuple(
        SurfaceObservation(
            surface=surface,
            contract_name=SURFACE_CONTRACT_MAP[surface],
            schema=f"ipfs_accelerate_py/agent-supervisor/{surface}@1",
            authority="ipfs_accelerate_py"
            if surface != "contextpack"
            else "ipfs_datasets_py",
            evidence_kind="measured_live",
            freeze_claimed=False,
            reason="fixture normative surface; not current-head evidence",
        )
        for surface in REQUIRED_FREEZE_SURFACES
    )


def _live_absent_competing() -> tuple[CompetingAuthorityObservation, ...]:
    return tuple(
        CompetingAuthorityObservation(
            name=name,
            present=False,
            evidence_kind="measured_live",
            scope="repository",
            reason="fixture live inventory; not current-head evidence",
        )
        for name in COMPETING_AUTHORITY_PROHIBITIONS
    )


def _live_contracts() -> tuple[ContractObservation, ...]:
    return tuple(
        ContractObservation(
            name=item["name"],
            schema=f"ipfs_accelerate_py/agent-supervisor/{item['name'].lower()}@1"
            if item["schema"] == PLACEHOLDER_SCHEMA
            else item["schema"],
            authority=item["authority"],
            evidence_kind="measured_live",
            freeze_claimed=False,
            reason="fixture",
            normative_task=item["normative_task"],
            freeze_task=item["freeze_task"],
        )
        for item in CANONICAL_CONTRACT_CATALOG
    )


def _no_override() -> OperatorOverride:
    return OperatorOverride(
        present=False,
        evidence_kind="measured",
        reason="fixture",
    )


def test_closed_vocabularies_match_pcpr_002_requirements() -> None:
    assert PCPR_002_TASK_ID == "PCPR-002"
    assert PCPR_002_GOAL_ID == "PCPR-G130"
    assert CONTRACT_FREEZE_INTERFACE == "CanonicalSupervisorContractFreeze@1"
    assert list(REQUIRED_FREEZE_SURFACES) == [
        "objective",
        "task",
        "event",
        "contextpack",
        "state_machine",
        "receipt",
    ]
    assert REQUIRED_FREEZE_SURFACES == PUBLIC_SUPERVISOR_SURFACES
    assert len(REQUIRED_CONTRACT_NAMES) == 14
    assert len(CANONICAL_CONTRACT_CATALOG) == 14
    assert len(COMPETING_AUTHORITY_PROHIBITIONS) == 10
    assert SURFACE_CONTRACT_MAP["objective"] == "SupervisorObjectiveIntent"
    assert SURFACE_CONTRACT_MAP["contextpack"] == "SupervisorContextPack"
    assert SURFACE_CONTRACT_MAP["state_machine"] == "TaskStateTransition"
    assert "release_candidate_qualified" in CLOSED_RELEASE_OUTCOMES
    assert "non_promoted_supervisor_unqualified" in CLOSED_RELEASE_OUTCOMES
    assert "rnd_non_promoted" not in CLOSED_RELEASE_OUTCOMES
    assert "supervisor_non_promoted" not in CLOSED_RELEASE_OUTCOMES
    assert all(item["freeze_task"] == "PCPR-002" for item in CANONICAL_CONTRACT_CATALOG)
    assert all(
        item["disposition"] == "baseline_recorded_not_frozen"
        for item in CANONICAL_CONTRACT_CATALOG
    )
    assert HERMETIC_CANDIDATE_SUITES[-1].endswith(
        "test_agent_supervisor_canonical_supervisor_contract_freeze.py"
    )


def test_current_head_is_rnd_non_promoted_and_not_a_freeze_or_release() -> None:
    verdict = qualify_current_head_without_supervisor_promotion()
    assert verdict.promotion_status == "rnd_non_promoted"
    assert verdict.supervisor_disposition == "supervisor_non_promoted"
    assert verdict.closed_release_outcome is None
    assert verdict.release_claim is False
    assert verdict.completion_authoritative is False
    assert verdict.contracts_frozen is False
    assert verdict.competing_authorities_prohibited_by_freeze is False
    assert verdict.continuation_requires_bounded_operator_approval is True
    assert verdict.duckdb_or_quack_state_written is False
    assert verdict.promotion_status not in CLOSED_RELEASE_OUTCOMES
    assert verdict.verdict_cid == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID
    assert verdict.qualification.task_id == "PCPR-001"
    assert verdict.qualification.verdict_cid == CURRENT_HEAD_UNAVAILABLE_VERDICT_CID
    assert verdict.qualification.promotion_status == "rnd_non_promoted"
    assert verdict.qualification.live_qualification_evidence_kind == "unavailable"
    assert verdict.qualification.missed_live_cohort_count == 13
    assert verdict.qualification.missed_target_count == 10
    assert "qualification_not_supervisor_promoted" in verdict.freeze_refused_reasons
    assert "live_qualification_unavailable" in verdict.freeze_refused_reasons
    assert "public_surfaces_not_all_normative" in verdict.freeze_refused_reasons
    assert "competing_authority_inventory_unavailable" in verdict.freeze_refused_reasons
    assert all(item.frozen is False for item in verdict.surfaces)
    assert all(item.present is None for item in verdict.competing_authorities)
    section = current_head_pcpr_002_receipt_promotion()
    assert section["promotion_status"] == "rnd_non_promoted"
    assert section["closed_release_outcome"] is None
    assert section["release_claim"] is False
    assert section["contracts_frozen"] is False
    assert section["duckdb_or_quack_state_written"] is False
    assert section["verdict_cid"] == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID


def test_current_head_receipt_sections_are_rnd_non_promoted_and_not_a_release() -> None:
    sections = current_head_pcpr_002_receipt_sections()
    assert sections["promotion_status"] == "rnd_non_promoted"
    assert sections["closed_release_outcome"] is None
    assert sections["release_claim"] is False
    assert sections["contracts_frozen"] is False
    assert sections["verdict_cid"] == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID
    assert sections["qualification_prerequisite"]["task_id"] == "PCPR-001"
    assert sections["qualification_prerequisite"]["verdict_cid"] == (
        CURRENT_HEAD_UNAVAILABLE_VERDICT_CID
    )
    assert [item["surface"] for item in sections["public_surfaces"]["catalog"]] == list(
        REQUIRED_FREEZE_SURFACES
    )
    assert sections["public_surfaces"]["frozen"] is False
    assert sections["public_surfaces"]["all_surfaces_freeze_eligible"] is False
    assert len(sections["shared_contracts"]["catalog"]) == 14
    assert sections["shared_contracts"]["frozen"] is False
    assert sections["shared_contracts"]["normative_stabilization_task"] == "PCPR-040"
    assert sections["competing_authorities"]["inventory_task"] == "PCPR-003"
    assert sections["competing_authorities"]["prohibited_by_freeze"] is False
    assert sections["competing_authorities"]["repository_live_inventory"] is False
    assert sections["operator_override"]["present"] is False
    assert sections["operator_override"]["can_freeze_contracts"] is False
    assert sections["negative_results"]["closed_release_outcome_not_emitted"] is True
    assert sections["negative_results"]["direct_database_bypass_not_used"] is True
    assert sections["negative_results"]["non_promoted_qualification_cannot_freeze"] is True


def test_outer_receipt_validator_accepts_generated_non_promotion_receipt() -> None:
    sections = current_head_pcpr_002_receipt_sections()
    payload = {
        "task_id": PCPR_002_TASK_ID,
        "status": "implemented",
        "completion_authoritative": False,
        "release_claim": False,
        "qualification_verdict": sections["qualification_verdict"],
        "qualification_prerequisite": sections["qualification_prerequisite"],
        "current_tree_binding": current_head_pcpr_002_current_tree_binding(),
        "acceptance": {
            "named_receipt_exists": True,
            "promotion_status": "rnd_non_promoted",
            "closed_release_outcome": None,
            "release_claim": False,
            "contracts_frozen": False,
        },
    }
    checked = validate_pcpr_002_outer_receipt(payload)
    assert checked["valid"] is True
    assert checked["promotion_status"] == "rnd_non_promoted"
    assert checked["closed_release_outcome"] is None
    assert checked["release_claim"] is False
    assert checked["contracts_frozen"] is False
    assert checked["verdict_cid"] == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID


def test_outer_receipt_validator_rejects_closed_release_outcome() -> None:
    sections = current_head_pcpr_002_receipt_sections()
    forged = {
        "task_id": PCPR_002_TASK_ID,
        "status": "implemented",
        "qualification_verdict": dict(sections["qualification_verdict"]),
        "acceptance": {
            "promotion_status": "release_candidate_qualified",
            "closed_release_outcome": "release_candidate_qualified",
            "release_claim": True,
        },
    }
    with pytest.raises(CanonicalSupervisorContractFreezeError, match="closed PCPR release"):
        validate_pcpr_002_outer_receipt(forged)
    forged_status = {
        "task_id": PCPR_002_TASK_ID,
        "status": "non_promoted_supervisor_unqualified",
        "qualification_verdict": sections["qualification_verdict"],
    }
    with pytest.raises(CanonicalSupervisorContractFreezeError, match="closed PCPR release"):
        validate_pcpr_002_outer_receipt(forged_status)
    forged_verdict = {
        "task_id": PCPR_002_TASK_ID,
        "status": "implemented",
        "qualification_verdict": {
            **sections["qualification_verdict"],
            "closed_release_outcome": "non_promoted_unmeasured",
        },
    }
    with pytest.raises(CanonicalSupervisorContractFreezeError, match="must be null"):
        validate_pcpr_002_outer_receipt(forged_verdict)
    forged_freeze = {
        "task_id": PCPR_002_TASK_ID,
        "status": "implemented",
        "qualification_verdict": {
            **sections["qualification_verdict"],
            "contracts_frozen": True,
        },
        "qualification_prerequisite": sections["qualification_prerequisite"],
    }
    with pytest.raises(CanonicalSupervisorContractFreezeError, match="cannot freeze"):
        validate_pcpr_002_outer_receipt(forged_freeze)


def test_duckdb_or_quack_write_is_rejected() -> None:
    with pytest.raises(CanonicalSupervisorContractFreezeError, match="DuckDB or Quack"):
        freeze_canonical_supervisor_contracts(
            qualification=current_head_qualification_prerequisite(),
            surfaces=current_head_surface_observations(),
            contracts=current_head_contract_observations(),
            competing_authorities=current_head_competing_authorities(),
            operator_override=current_head_operator_override(),
            duckdb_or_quack_state_written=True,
        )


def test_simulated_freeze_claim_is_rejected() -> None:
    surfaces = list(current_head_surface_observations())
    surfaces[0] = _clone_surface(
        surfaces[0], evidence_kind="simulated", freeze_claimed=True
    )
    with pytest.raises(CanonicalSupervisorContractFreezeError, match="simulated freeze"):
        freeze_canonical_supervisor_contracts(
            qualification=current_head_qualification_prerequisite(),
            surfaces=surfaces,
            contracts=current_head_contract_observations(),
            competing_authorities=current_head_competing_authorities(),
            operator_override=current_head_operator_override(),
        )


def test_simulated_competing_authority_absence_is_rejected() -> None:
    competing = list(current_head_competing_authorities())
    competing[0] = CompetingAuthorityObservation(
        name=competing[0].name,
        present=False,
        evidence_kind="simulated",
        scope="repository",
        reason="fixture",
    )
    with pytest.raises(CanonicalSupervisorContractFreezeError, match="simulated absence"):
        freeze_canonical_supervisor_contracts(
            qualification=current_head_qualification_prerequisite(),
            surfaces=current_head_surface_observations(),
            contracts=current_head_contract_observations(),
            competing_authorities=competing,
            operator_override=current_head_operator_override(),
        )


def test_hermetic_qualification_cannot_freeze() -> None:
    qualification = QualificationPrerequisite(
        task_id="PCPR-001",
        promotion_status="supervisor_promoted",
        supervisor_disposition="supervisor_promoted",
        live_qualification_evidence_kind="measured_hermetic",
        verdict_cid="baguqeerahermeticpcpr001cannotfreezecidplaceholderaaaaaa",
        live_campaign_identity="hermetic",
        reason="hermetic",
    )
    verdict = freeze_canonical_supervisor_contracts(
        qualification=qualification,
        surfaces=_normative_surfaces(),
        contracts=_live_contracts(),
        competing_authorities=_live_absent_competing(),
        operator_override=_no_override(),
    )
    assert verdict.contracts_frozen is False
    assert verdict.promotion_status == "rnd_non_promoted"
    assert "qualification_not_supervisor_promoted" in verdict.freeze_refused_reasons
    assert verdict.release_claim is False


def test_unavailable_metrics_are_not_represented_as_zero() -> None:
    competing = current_head_competing_authorities()
    assert all(item.present is None for item in competing)
    assert all(item.evidence_kind == "unavailable" for item in competing)
    verdict = qualify_current_head_without_supervisor_promotion()
    assert verdict.qualification.live_campaign_identity == ""
    assert all(item.frozen is False for item in verdict.surfaces)


def test_non_promoted_qualification_cannot_freeze_even_with_normative_surfaces() -> None:
    verdict = freeze_canonical_supervisor_contracts(
        qualification=current_head_qualification_prerequisite(),
        surfaces=_normative_surfaces(),
        contracts=_live_contracts(),
        competing_authorities=_live_absent_competing(),
        operator_override=_no_override(),
    )
    assert verdict.contracts_frozen is False
    assert verdict.promotion_status == "rnd_non_promoted"
    assert verdict.supervisor_disposition == "supervisor_non_promoted"
    assert "qualification_not_supervisor_promoted" in verdict.freeze_refused_reasons
    assert verdict.closed_release_outcome is None


def test_typed_blocked_qualification_stays_typed_blocked() -> None:
    qualification = QualificationPrerequisite(
        task_id="PCPR-001",
        promotion_status="typed_blocked",
        supervisor_disposition="supervisor_non_promoted",
        live_qualification_evidence_kind="unavailable",
        verdict_cid="baguqeerablockedpcpr001qualificationcidplaceholderaaaaaa",
        reason="blocked cohort",
        missed_live_cohort_count=1,
        missed_target_count=10,
    )
    verdict = freeze_canonical_supervisor_contracts(
        qualification=qualification,
        surfaces=current_head_surface_observations(),
        contracts=current_head_contract_observations(),
        competing_authorities=current_head_competing_authorities(),
        operator_override=current_head_operator_override(),
    )
    assert verdict.promotion_status == "typed_blocked"
    assert verdict.contracts_frozen is False
    assert verdict.release_claim is False


def test_typed_unavailable_qualification_stays_typed_unavailable() -> None:
    qualification = QualificationPrerequisite(
        task_id="PCPR-001",
        promotion_status="typed_unavailable",
        supervisor_disposition="supervisor_non_promoted",
        live_qualification_evidence_kind="unavailable",
        verdict_cid="baguqeeraunavailablepcpr001qualificationcidplaceholderaa",
        reason="missing live environment",
        missed_live_cohort_count=13,
        missed_target_count=10,
    )
    verdict = freeze_canonical_supervisor_contracts(
        qualification=qualification,
        surfaces=current_head_surface_observations(),
        contracts=current_head_contract_observations(),
        competing_authorities=current_head_competing_authorities(),
        operator_override=current_head_operator_override(),
    )
    assert verdict.promotion_status == "typed_unavailable"
    assert verdict.contracts_frozen is False
    assert verdict.closed_release_outcome is None


def test_promoted_but_placeholder_schemas_cannot_freeze() -> None:
    verdict = freeze_canonical_supervisor_contracts(
        qualification=_live_promoted_qualification(),
        surfaces=current_head_surface_observations(),
        contracts=current_head_contract_observations(),
        competing_authorities=_live_absent_competing(),
        operator_override=_no_override(),
    )
    assert verdict.contracts_frozen is False
    assert verdict.promotion_status == "rnd_non_promoted"
    assert "public_surfaces_not_all_normative" in verdict.freeze_refused_reasons
    assert any("not_yet_normative" in item for item in verdict.freeze_refused_reasons)


def test_promoted_without_live_competing_inventory_cannot_freeze() -> None:
    verdict = freeze_canonical_supervisor_contracts(
        qualification=_live_promoted_qualification(),
        surfaces=_normative_surfaces(),
        contracts=_live_contracts(),
        competing_authorities=current_head_competing_authorities(),
        operator_override=_no_override(),
    )
    assert verdict.contracts_frozen is False
    assert verdict.promotion_status == "rnd_non_promoted"
    assert "competing_authority_inventory_unavailable" in verdict.freeze_refused_reasons


def test_complete_live_evidence_freezes_without_release_claim() -> None:
    verdict = freeze_canonical_supervisor_contracts(
        qualification=_live_promoted_qualification(),
        surfaces=_normative_surfaces(),
        contracts=_live_contracts(),
        competing_authorities=_live_absent_competing(),
        operator_override=_no_override(),
    )
    assert verdict.promotion_status == "supervisor_promoted"
    assert verdict.supervisor_disposition == "supervisor_promoted"
    assert verdict.contracts_frozen is True
    assert verdict.competing_authorities_prohibited_by_freeze is True
    assert verdict.continuation_requires_bounded_operator_approval is False
    assert verdict.closed_release_outcome is None
    assert verdict.release_claim is False
    assert verdict.completion_authoritative is False
    assert verdict.promotion_status not in CLOSED_RELEASE_OUTCOMES
    assert all(item.frozen is True for item in verdict.surfaces)
    assert verdict.freeze_refused_reasons == ()
    assert verdict.blockers == ()
    section = pcpr_002_receipt_promotion(verdict)
    assert section["promotion_status"] == "supervisor_promoted"
    assert section["contracts_frozen"] is True
    assert section["closed_release_outcome"] is None
    assert section["release_claim"] is False
    assert section["duckdb_or_quack_state_written"] is False


def test_competing_authority_present_is_typed_blocked() -> None:
    competing = list(_live_absent_competing())
    competing[0] = CompetingAuthorityObservation(
        name=competing[0].name,
        present=True,
        evidence_kind="measured_live",
        scope="repository",
        reason="second supervisor family observed",
    )
    verdict = freeze_canonical_supervisor_contracts(
        qualification=_live_promoted_qualification(),
        surfaces=_normative_surfaces(),
        contracts=_live_contracts(),
        competing_authorities=competing,
        operator_override=_no_override(),
    )
    assert verdict.promotion_status == "typed_blocked"
    assert verdict.contracts_frozen is False
    assert "competing_authority_present" in verdict.freeze_refused_reasons
    assert any("present" in item for item in verdict.blockers)
    assert verdict.release_claim is False


def test_operator_override_cannot_freeze() -> None:
    verdict = freeze_canonical_supervisor_contracts(
        qualification=_live_promoted_qualification(),
        surfaces=_normative_surfaces(),
        contracts=_live_contracts(),
        competing_authorities=_live_absent_competing(),
        operator_override=OperatorOverride(
            present=True,
            evidence_kind="measured",
            reason="operator asked to continue; freeze still forbidden",
        ),
    )
    assert verdict.contracts_frozen is False
    assert verdict.promotion_status == "rnd_non_promoted"
    assert "operator_override_cannot_freeze" in verdict.freeze_refused_reasons
    assert verdict.operator_override.present is True
    assert verdict.operator_override.to_mapping()["can_freeze_contracts"] is False
    assert verdict.operator_override.to_mapping()["can_close_release"] is False


def test_claimed_freeze_without_qualification_is_typed_blocked() -> None:
    surfaces = [
        _clone_surface(item, freeze_claimed=True)
        for item in current_head_surface_observations()
    ]
    verdict = freeze_canonical_supervisor_contracts(
        qualification=current_head_qualification_prerequisite(),
        surfaces=surfaces,
        contracts=current_head_contract_observations(),
        competing_authorities=current_head_competing_authorities(),
        operator_override=current_head_operator_override(),
    )
    assert verdict.contracts_frozen is False
    assert verdict.promotion_status == "typed_blocked"
    assert any("claimed_freeze_without_qualification" in item for item in verdict.blockers)


def test_unknown_or_duplicate_surface_fails_closed() -> None:
    surfaces = current_head_surface_observations()
    with pytest.raises(CanonicalSupervisorContractFreezeError, match="duplicate"):
        freeze_canonical_supervisor_contracts(
            qualification=current_head_qualification_prerequisite(),
            surfaces=(*surfaces, surfaces[0]),
            contracts=current_head_contract_observations(),
            competing_authorities=current_head_competing_authorities(),
            operator_override=current_head_operator_override(),
        )
    extra = surfaces[1:] + (
        SurfaceObservation(
            surface="secret_surface",
            contract_name="Secret",
            schema="secret",
            authority="ipfs_accelerate_py",
            evidence_kind="measured",
        ),
    )
    with pytest.raises(CanonicalSupervisorContractFreezeError, match="unknown"):
        freeze_canonical_supervisor_contracts(
            qualification=current_head_qualification_prerequisite(),
            surfaces=extra,
            contracts=current_head_contract_observations(),
            competing_authorities=current_head_competing_authorities(),
            operator_override=current_head_operator_override(),
        )


def test_wrong_qualification_task_fails_closed() -> None:
    with pytest.raises(CanonicalSupervisorContractFreezeError, match="PCPR-001"):
        freeze_canonical_supervisor_contracts(
            qualification=QualificationPrerequisite(
                task_id="PCPR-000",
                promotion_status="rnd_non_promoted",
                supervisor_disposition="supervisor_non_promoted",
                live_qualification_evidence_kind="unavailable",
                verdict_cid="baguqeerawrongtaskqualificationcidplaceholderaaaaaaaaaa",
            ),
            surfaces=current_head_surface_observations(),
            contracts=current_head_contract_observations(),
            competing_authorities=current_head_competing_authorities(),
            operator_override=current_head_operator_override(),
        )


def test_simulated_qualification_cannot_promote() -> None:
    with pytest.raises(CanonicalSupervisorContractFreezeError, match="simulated qualification"):
        freeze_canonical_supervisor_contracts(
            qualification=QualificationPrerequisite(
                task_id="PCPR-001",
                promotion_status="supervisor_promoted",
                supervisor_disposition="supervisor_promoted",
                live_qualification_evidence_kind="simulated",
                verdict_cid="baguqeerasimulatedpcpr001qualificationcidplaceholderaaa",
                live_campaign_identity="sim",
            ),
            surfaces=_normative_surfaces(),
            contracts=_live_contracts(),
            competing_authorities=_live_absent_competing(),
            operator_override=_no_override(),
        )


def test_live_promotion_requires_campaign_identity() -> None:
    with pytest.raises(CanonicalSupervisorContractFreezeError, match="live_campaign_identity"):
        freeze_canonical_supervisor_contracts(
            qualification=QualificationPrerequisite(
                task_id="PCPR-001",
                promotion_status="supervisor_promoted",
                supervisor_disposition="supervisor_promoted",
                live_qualification_evidence_kind="measured_live",
                verdict_cid="baguqeeramissinglivecampaignidentitycidplaceholderaaaaa",
            ),
            surfaces=_normative_surfaces(),
            contracts=_live_contracts(),
            competing_authorities=_live_absent_competing(),
            operator_override=_no_override(),
        )


def test_current_tree_binding_is_measured_and_not_a_release() -> None:
    binding = current_head_pcpr_002_current_tree_binding()
    assert binding["evidence_kind"] == "measured"
    assert binding["origin_main_is_ancestor"] is True
    assert binding["accelerator_origin_main_is_ancestor"] is True
    assert binding["accelerator_pre_change_commit"] == binding["accelerator_gitlink"]
    assert binding["accelerator_gitlink"] == CURRENT_HEAD_ACCELERATOR_COMMIT
    assert binding["datasets_commit"] == CURRENT_HEAD_DATASETS_COMMIT
    assert binding["kit_commit"] == CURRENT_HEAD_KIT_COMMIT
    assert binding["datasets_commit"] == binding["datasets_gitlink"]
    assert binding["kit_commit"] == binding["kit_gitlink"]
    assert binding["accelerator_post_change_commit"] == "pending nested commit after admission"
    assert all(outcome not in binding["outer_subject"] for outcome in CLOSED_RELEASE_OUTCOMES)
    assert "closed_release_outcome" not in binding


def test_current_tree_binding_rejects_non_ancestor_and_gitlink_mismatch() -> None:
    kwargs = dict(current_head_pcpr_002_current_tree_binding())
    kwargs.pop("outer_repository")
    kwargs.pop("owning_repository_for_receipts")
    kwargs.pop("accelerator_post_change_commit")
    kwargs.pop("accelerator_post_change_tree")
    kwargs.pop("evidence_kind")
    kwargs["origin_main_is_ancestor"] = False
    with pytest.raises(CanonicalSupervisorContractFreezeError, match="origin_main_is_ancestor"):
        pcpr_002_current_tree_binding(**kwargs)
    kwargs = dict(current_head_pcpr_002_current_tree_binding())
    kwargs.pop("outer_repository")
    kwargs.pop("owning_repository_for_receipts")
    kwargs.pop("accelerator_post_change_commit")
    kwargs.pop("accelerator_post_change_tree")
    kwargs.pop("evidence_kind")
    kwargs["accelerator_gitlink"] = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    with pytest.raises(CanonicalSupervisorContractFreezeError, match="accelerator_gitlink"):
        pcpr_002_current_tree_binding(**kwargs)
    sections = current_head_pcpr_002_receipt_sections()
    forged = {
        "task_id": PCPR_002_TASK_ID,
        "status": "implemented",
        "qualification_verdict": sections["qualification_verdict"],
        "current_tree_binding": {
            **current_head_pcpr_002_current_tree_binding(),
            "evidence_kind": "simulated",
        },
    }
    with pytest.raises(CanonicalSupervisorContractFreezeError, match="measured"):
        validate_pcpr_002_outer_receipt(forged)


def test_contextpack_schema_is_observed_but_not_frozen() -> None:
    surfaces = current_head_surface_observations()
    contextpack = next(item for item in surfaces if item.surface == "contextpack")
    assert contextpack.schema == "ipfs_datasets_py.proof_context.context_pack"
    assert contextpack.authority == "ipfs_datasets_py"
    assert contextpack.freeze_claimed is False
    objective = next(item for item in surfaces if item.surface == "objective")
    assert objective.schema == PLACEHOLDER_SCHEMA
    verdict = qualify_current_head_without_supervisor_promotion()
    frozen_contextpack = next(item for item in verdict.surfaces if item.surface == "contextpack")
    assert frozen_contextpack.frozen is False
    assert frozen_contextpack.freeze_eligible is False
