"""PCPR-003 fail-closed legacy-bypass and false-authority inventory."""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.validation.canonical_supervisor_contract_freeze import (
    COMPETING_AUTHORITY_PROHIBITIONS,
    CompetingAuthorityObservation,
)
from ipfs_accelerate_py.agent_supervisor.validation.legacy_bypass_and_false_authority_inventory import (
    CLOSED_RELEASE_OUTCOMES,
    CURRENT_HEAD_ACCELERATOR_COMMIT,
    CURRENT_HEAD_DATASETS_COMMIT,
    CURRENT_HEAD_KIT_COMMIT,
    CURRENT_HEAD_NON_PROMOTION_VERDICT_CID,
    CURRENT_HEAD_UNAVAILABLE_VERDICT_CID,
    HERMETIC_CANDIDATE_SUITES,
    INVENTORY_INTERFACE,
    PATH_CLASS_CATALOG,
    PCPR_003_GOAL_ID,
    PCPR_003_TASK_ID,
    REQUIRED_PATH_IDS,
    LegacyBypassInventoryError,
    PathObservation,
    competing_authorities_from_paths,
    current_head_path_observations,
    current_head_pcpr_003_current_tree_binding,
    current_head_pcpr_003_receipt_promotion,
    current_head_pcpr_003_receipt_sections,
    inventory_legacy_bypass_and_false_authority_paths,
    pcpr_003_current_tree_binding,
    pcpr_003_receipt_promotion,
    qualify_current_head_inventory,
    scan_path_catalog,
    validate_pcpr_003_outer_receipt,
)


def _unavailable_paths() -> tuple[PathObservation, ...]:
    return tuple(
        PathObservation(
            path_id=item.path_id,
            category=item.category,
            repository=item.repository,
            remediating_task=item.remediating_task,
            disposition=item.disposition,
            authority_claim=item.authority_claim,
            migration_impact=item.migration_impact,
            present=None,
            evidence_kind="unavailable",
            hit_count=0,
            sample_hits=(),
            source_identity={"repository": item.repository},
            reason="fixture unavailable; not current-head evidence",
        )
        for item in PATH_CLASS_CATALOG
    )


def _unavailable_competing() -> tuple[CompetingAuthorityObservation, ...]:
    return tuple(
        CompetingAuthorityObservation(
            name=name,
            present=None,
            evidence_kind="unavailable",
            scope="repository",
            reason="fixture unavailable; not current-head evidence",
        )
        for name in COMPETING_AUTHORITY_PROHIBITIONS
    )


def test_closed_vocabularies_match_pcpr_003_requirements() -> None:
    assert PCPR_003_TASK_ID == "PCPR-003"
    assert PCPR_003_GOAL_ID == "PCPR-G130"
    assert INVENTORY_INTERFACE == "LegacyBypassAndFalseAuthorityInventory@1"
    assert len(REQUIRED_PATH_IDS) == 22
    assert len(PATH_CLASS_CATALOG) == 22
    assert len(set(REQUIRED_PATH_IDS)) == 22
    assert REQUIRED_PATH_IDS[0] == "datasets_import_time_auto_install"
    assert "accelerate_pseudo_cid" in REQUIRED_PATH_IDS
    assert "accelerate_legacy_mock_coordinator" in REQUIRED_PATH_IDS
    assert "markdown_as_live_authority" in REQUIRED_PATH_IDS
    assert len(COMPETING_AUTHORITY_PROHIBITIONS) == 10
    assert "release_candidate_qualified" in CLOSED_RELEASE_OUTCOMES
    assert "rnd_non_promoted" not in CLOSED_RELEASE_OUTCOMES
    assert HERMETIC_CANDIDATE_SUITES[-1].endswith(
        "test_agent_supervisor_legacy_bypass_and_false_authority_inventory.py"
    )
    remediations = {item.remediating_task for item in PATH_CLASS_CATALOG}
    assert "PCPR-010" in remediations
    assert "PCPR-030" in remediations
    assert "PCPR-032" in remediations


def test_current_head_is_rnd_non_promoted_and_not_a_release() -> None:
    verdict = qualify_current_head_inventory()
    assert verdict.promotion_status == "rnd_non_promoted"
    assert verdict.supervisor_disposition == "supervisor_non_promoted"
    assert verdict.closed_release_outcome is None
    assert verdict.release_claim is False
    assert verdict.completion_authoritative is False
    assert verdict.contracts_frozen is False
    assert verdict.competing_authorities_prohibited is False
    assert verdict.live_runtime_inventory is False
    assert verdict.duckdb_or_quack_state_written is False
    assert verdict.catalog_complete is True
    assert verdict.this_task_created_competing_authority is False
    assert verdict.promotion_status not in CLOSED_RELEASE_OUTCOMES
    assert verdict.verdict_cid == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID
    assert len(verdict.paths) == 22
    assert [item.path_id for item in verdict.paths] == list(REQUIRED_PATH_IDS)
    section = current_head_pcpr_003_receipt_promotion()
    assert section["promotion_status"] == "rnd_non_promoted"
    assert section["closed_release_outcome"] is None
    assert section["release_claim"] is False
    assert section["contracts_frozen"] is False
    assert section["live_runtime_inventory"] is False
    assert section["duckdb_or_quack_state_written"] is False
    assert section["verdict_cid"] == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID


def test_current_head_scan_measures_known_false_authority_paths() -> None:
    paths = {item.path_id: item for item in current_head_path_observations()}
    mock = paths["accelerate_legacy_mock_coordinator"]
    assert mock.evidence_kind == "measured"
    assert mock.present is True
    assert mock.hit_count >= 1
    assert mock.disposition == "quarantine"
    assert mock.remediating_task == "PCPR-030"
    cid_path = paths["accelerate_pseudo_cid"]
    assert cid_path.present is True
    assert cid_path.evidence_kind == "measured"
    hardware = paths["accelerate_fabricated_hardware"]
    assert hardware.present is True
    git_deps = paths["accelerate_mutable_git_branch_deps"]
    assert git_deps.present is True
    python_floor = paths["accelerate_python_floor_metadata_drift"]
    assert python_floor.present is False
    assert python_floor.evidence_kind == "measured"
    assert python_floor.remediating_task == "PCPR-036"
    duckdb = paths["accelerate_direct_duckdb_task_state"]
    assert duckdb.present is True
    assert duckdb.category == "competing_authority"
    launch = paths["prebuilt_complete_launch_plan_injection"]
    assert launch.present is True
    assert launch.disposition == "keep_internal_not_public_requirement"


def test_current_head_receipt_sections_are_rnd_non_promoted_and_not_a_release() -> None:
    sections = current_head_pcpr_003_receipt_sections()
    assert sections["promotion_status"] == "rnd_non_promoted"
    assert sections["closed_release_outcome"] is None
    assert sections["release_claim"] is False
    assert sections["contracts_frozen"] is False
    assert sections["live_runtime_inventory"] is False
    assert sections["verdict_cid"] == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID
    assert sections["qualification_prerequisite"]["task_id"] == "PCPR-001"
    assert sections["qualification_prerequisite"]["verdict_cid"] == (
        CURRENT_HEAD_UNAVAILABLE_VERDICT_CID
    )
    catalog = sections["path_inventory"]["catalog"]
    assert [item["path_id"] for item in catalog] == list(REQUIRED_PATH_IDS)
    assert sections["path_inventory"]["exhaustive_of_unlisted_runtime_bypasses"] is False
    assert sections["path_inventory"]["catalog_complete_relative_to_closed_pcpr_list"] is True
    for item in catalog:
        assert "authority_analysis" in item
        assert "migration_impact" in item
        assert "disposition" in item
        assert "source_identity" in item
        assert item["evidence_kind"] in {"measured", "unavailable"}
    competing = sections["competing_authorities"]
    assert competing["inventory_task"] == "PCPR-003"
    assert competing["prohibited_by_inventory"] is False
    assert competing["repository_live_inventory"] is False
    assert competing["this_task_created_competing_authority"] is False
    assert len(competing["observations"]) == 10
    assert sections["negative_results"]["closed_release_outcome_not_emitted"] is True
    assert sections["negative_results"]["direct_database_bypass_not_used"] is True
    assert sections["negative_results"]["simulated_presence_cannot_count_as_live"] is True


def test_outer_receipt_validator_accepts_generated_non_promotion_receipt() -> None:
    sections = current_head_pcpr_003_receipt_sections()
    payload = {
        "task_id": PCPR_003_TASK_ID,
        "status": "implemented",
        "completion_authoritative": False,
        "release_claim": False,
        "qualification_verdict": sections["qualification_verdict"],
        "qualification_prerequisite": sections["qualification_prerequisite"],
        "current_tree_binding": current_head_pcpr_003_current_tree_binding(),
        "acceptance": {
            "named_receipt_exists": True,
            "promotion_status": "rnd_non_promoted",
            "closed_release_outcome": None,
            "release_claim": False,
            "contracts_frozen": False,
        },
    }
    checked = validate_pcpr_003_outer_receipt(payload)
    assert checked["valid"] is True
    assert checked["promotion_status"] == "rnd_non_promoted"
    assert checked["closed_release_outcome"] is None
    assert checked["release_claim"] is False
    assert checked["contracts_frozen"] is False
    assert checked["verdict_cid"] == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID


def test_outer_receipt_validator_rejects_closed_release_outcome() -> None:
    sections = current_head_pcpr_003_receipt_sections()
    forged = {
        "task_id": PCPR_003_TASK_ID,
        "status": "implemented",
        "qualification_verdict": dict(sections["qualification_verdict"]),
        "acceptance": {
            "promotion_status": "release_candidate_qualified",
            "closed_release_outcome": "release_candidate_qualified",
            "release_claim": True,
        },
    }
    with pytest.raises(LegacyBypassInventoryError, match="closed PCPR release"):
        validate_pcpr_003_outer_receipt(forged)
    forged_status = {
        "task_id": PCPR_003_TASK_ID,
        "status": "non_promoted_supervisor_unqualified",
        "qualification_verdict": sections["qualification_verdict"],
    }
    with pytest.raises(LegacyBypassInventoryError, match="closed PCPR release"):
        validate_pcpr_003_outer_receipt(forged_status)
    forged_verdict = {
        "task_id": PCPR_003_TASK_ID,
        "status": "implemented",
        "qualification_verdict": {
            **sections["qualification_verdict"],
            "closed_release_outcome": "non_promoted_unmeasured",
        },
    }
    with pytest.raises(LegacyBypassInventoryError, match="must be null"):
        validate_pcpr_003_outer_receipt(forged_verdict)
    forged_freeze = {
        "task_id": PCPR_003_TASK_ID,
        "status": "implemented",
        "qualification_verdict": {
            **sections["qualification_verdict"],
            "contracts_frozen": True,
        },
    }
    with pytest.raises(LegacyBypassInventoryError, match="cannot freeze"):
        validate_pcpr_003_outer_receipt(forged_freeze)
    forged_live = {
        "task_id": PCPR_003_TASK_ID,
        "status": "implemented",
        "qualification_verdict": {
            **sections["qualification_verdict"],
            "live_runtime_inventory": True,
        },
    }
    with pytest.raises(LegacyBypassInventoryError, match="live runtime inventory"):
        validate_pcpr_003_outer_receipt(forged_live)
    forged_promote = {
        "task_id": PCPR_003_TASK_ID,
        "status": "implemented",
        "qualification_verdict": {
            **sections["qualification_verdict"],
            "promotion_status": "supervisor_promoted",
        },
    }
    with pytest.raises(LegacyBypassInventoryError, match="cannot promote"):
        validate_pcpr_003_outer_receipt(forged_promote)


def test_duckdb_or_quack_write_is_rejected() -> None:
    with pytest.raises(LegacyBypassInventoryError, match="DuckDB or Quack"):
        inventory_legacy_bypass_and_false_authority_paths(
            paths=_unavailable_paths(),
            competing_authorities=_unavailable_competing(),
            duckdb_or_quack_state_written=True,
        )


def test_live_runtime_inventory_claim_is_rejected() -> None:
    with pytest.raises(LegacyBypassInventoryError, match="live runtime inventory"):
        inventory_legacy_bypass_and_false_authority_paths(
            paths=_unavailable_paths(),
            competing_authorities=_unavailable_competing(),
            live_runtime_inventory=True,
        )


def test_simulated_presence_is_rejected() -> None:
    paths = list(_unavailable_paths())
    first = paths[0]
    paths[0] = PathObservation(
        path_id=first.path_id,
        category=first.category,
        repository=first.repository,
        remediating_task=first.remediating_task,
        disposition=first.disposition,
        authority_claim=first.authority_claim,
        migration_impact=first.migration_impact,
        present=True,
        evidence_kind="simulated",
        hit_count=1,
        sample_hits=({"relpath": "fixture.py", "line": 1, "needle": "mock"},),
        source_identity={"repository": first.repository},
        reason="fixture",
    )
    with pytest.raises(LegacyBypassInventoryError, match="simulated observations"):
        inventory_legacy_bypass_and_false_authority_paths(
            paths=paths,
            competing_authorities=_unavailable_competing(),
        )


def test_simulated_absence_is_rejected() -> None:
    paths = list(_unavailable_paths())
    first = paths[0]
    paths[0] = PathObservation(
        path_id=first.path_id,
        category=first.category,
        repository=first.repository,
        remediating_task=first.remediating_task,
        disposition=first.disposition,
        authority_claim=first.authority_claim,
        migration_impact=first.migration_impact,
        present=False,
        evidence_kind="simulated",
        hit_count=0,
        sample_hits=(),
        source_identity={"repository": first.repository},
        reason="fixture",
    )
    with pytest.raises(LegacyBypassInventoryError, match="simulated absence"):
        inventory_legacy_bypass_and_false_authority_paths(
            paths=paths,
            competing_authorities=_unavailable_competing(),
        )


def test_estimated_values_are_rejected() -> None:
    paths = list(_unavailable_paths())
    first = paths[0]
    paths[0] = PathObservation(
        path_id=first.path_id,
        category=first.category,
        repository=first.repository,
        remediating_task=first.remediating_task,
        disposition=first.disposition,
        authority_claim=first.authority_claim,
        migration_impact=first.migration_impact,
        present=True,
        evidence_kind="estimated",
        hit_count=1,
        sample_hits=({"relpath": "fixture.py", "line": 1, "needle": "x"},),
        source_identity={"repository": first.repository},
        reason="fixture",
    )
    with pytest.raises(LegacyBypassInventoryError, match="estimated"):
        inventory_legacy_bypass_and_false_authority_paths(
            paths=paths,
            competing_authorities=_unavailable_competing(),
        )


def test_measured_live_static_inventory_is_rejected() -> None:
    paths = list(_unavailable_paths())
    first = paths[0]
    paths[0] = PathObservation(
        path_id=first.path_id,
        category=first.category,
        repository=first.repository,
        remediating_task=first.remediating_task,
        disposition=first.disposition,
        authority_claim=first.authority_claim,
        migration_impact=first.migration_impact,
        present=True,
        evidence_kind="measured_live",
        hit_count=1,
        sample_hits=({"relpath": "fixture.py", "line": 1, "needle": "x"},),
        source_identity={"repository": first.repository},
        reason="fixture",
    )
    with pytest.raises(LegacyBypassInventoryError, match="measured_live"):
        inventory_legacy_bypass_and_false_authority_paths(
            paths=paths,
            competing_authorities=_unavailable_competing(),
        )


def test_unavailable_metrics_are_not_represented_as_zero() -> None:
    verdict = inventory_legacy_bypass_and_false_authority_paths(
        paths=_unavailable_paths(),
        competing_authorities=_unavailable_competing(),
    )
    assert verdict.promotion_status == "typed_unavailable"
    assert verdict.measured_present_count == 0
    assert verdict.unavailable_count == 22
    assert all(item.present is None for item in verdict.paths)
    assert all(item.hit_count == 0 for item in verdict.paths)
    assert all(item.evidence_kind == "unavailable" for item in verdict.competing_authorities)
    assert all(item.present is None for item in verdict.competing_authorities)
    assert verdict.closed_release_outcome is None
    assert verdict.release_claim is False


def test_unavailable_present_boolean_is_rejected() -> None:
    paths = list(_unavailable_paths())
    first = paths[0]
    paths[0] = PathObservation(
        path_id=first.path_id,
        category=first.category,
        repository=first.repository,
        remediating_task=first.remediating_task,
        disposition=first.disposition,
        authority_claim=first.authority_claim,
        migration_impact=first.migration_impact,
        present=False,
        evidence_kind="unavailable",
        hit_count=0,
        sample_hits=(),
        source_identity={"repository": first.repository},
        reason="fixture",
    )
    with pytest.raises(LegacyBypassInventoryError, match="must not record a boolean"):
        inventory_legacy_bypass_and_false_authority_paths(
            paths=paths,
            competing_authorities=_unavailable_competing(),
        )


def test_this_task_created_competing_authority_is_typed_blocked() -> None:
    competing = list(_unavailable_competing())
    competing[0] = CompetingAuthorityObservation(
        name=competing[0].name,
        present=True,
        evidence_kind="measured",
        scope="this_task",
        reason="fixture this-task competing authority",
    )
    verdict = inventory_legacy_bypass_and_false_authority_paths(
        paths=_unavailable_paths(),
        competing_authorities=competing,
    )
    assert verdict.promotion_status == "typed_blocked"
    assert "this_task_created_competing_authority" in verdict.blockers
    assert verdict.release_claim is False
    assert verdict.this_task_created_competing_authority is True


def test_unknown_or_duplicate_path_fails_closed() -> None:
    paths = _unavailable_paths()
    with pytest.raises(LegacyBypassInventoryError, match="duplicate"):
        inventory_legacy_bypass_and_false_authority_paths(
            paths=(*paths, paths[0]),
            competing_authorities=_unavailable_competing(),
        )
    extra = paths[1:] + (
        PathObservation(
            path_id="secret_bypass",
            category="bypass",
            repository="ipfs_accelerate_py",
            remediating_task="PCPR-003",
            disposition="remove",
            authority_claim="secret",
            migration_impact="secret",
            present=None,
            evidence_kind="unavailable",
            hit_count=0,
            sample_hits=(),
            source_identity={"repository": "ipfs_accelerate_py"},
        ),
    )
    with pytest.raises(LegacyBypassInventoryError, match="unknown"):
        inventory_legacy_bypass_and_false_authority_paths(
            paths=extra,
            competing_authorities=_unavailable_competing(),
        )


def test_simulated_competing_authority_absence_is_rejected() -> None:
    competing = list(_unavailable_competing())
    competing[0] = CompetingAuthorityObservation(
        name=competing[0].name,
        present=False,
        evidence_kind="simulated",
        scope="repository",
        reason="fixture",
    )
    with pytest.raises(LegacyBypassInventoryError, match="simulated absence"):
        inventory_legacy_bypass_and_false_authority_paths(
            paths=_unavailable_paths(),
            competing_authorities=competing,
        )


def test_scan_path_catalog_on_fixture_tree_is_measured_not_live() -> None:
    root = Path(__file__).resolve().parents[2]
    observations = scan_path_catalog(accelerate_root=root)
    by_id = {item.path_id: item for item in observations}
    assert by_id["accelerate_legacy_mock_coordinator"].present is True
    assert by_id["accelerate_legacy_mock_coordinator"].evidence_kind == "measured"
    assert by_id["accelerate_pseudo_cid"].present is True
    datasets = by_id["datasets_import_time_auto_install"]
    assert datasets.evidence_kind in {"measured", "unavailable"}
    if datasets.evidence_kind == "unavailable":
        assert datasets.present is None
        assert datasets.hit_count == 0
    competing = competing_authorities_from_paths(observations)
    assert len(competing) == 10
    duckdb = next(item for item in competing if item.name == "direct_duckdb_writes")
    if duckdb.evidence_kind == "measured":
        assert duckdb.present is True
    assert all(item.evidence_kind != "measured_live" for item in competing)
    assert all(item.evidence_kind != "simulated" for item in competing)


def test_current_tree_binding_is_measured_and_not_a_release() -> None:
    binding = current_head_pcpr_003_current_tree_binding()
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
    kwargs = dict(current_head_pcpr_003_current_tree_binding())
    kwargs.pop("outer_repository")
    kwargs.pop("owning_repository_for_receipts")
    kwargs.pop("accelerator_post_change_commit")
    kwargs.pop("accelerator_post_change_tree")
    kwargs.pop("evidence_kind")
    kwargs["origin_main_is_ancestor"] = False
    with pytest.raises(LegacyBypassInventoryError, match="origin_main_is_ancestor"):
        pcpr_003_current_tree_binding(**kwargs)
    kwargs = dict(current_head_pcpr_003_current_tree_binding())
    kwargs.pop("outer_repository")
    kwargs.pop("owning_repository_for_receipts")
    kwargs.pop("accelerator_post_change_commit")
    kwargs.pop("accelerator_post_change_tree")
    kwargs.pop("evidence_kind")
    kwargs["accelerator_gitlink"] = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    with pytest.raises(LegacyBypassInventoryError, match="accelerator_gitlink"):
        pcpr_003_current_tree_binding(**kwargs)
    sections = current_head_pcpr_003_receipt_sections()
    forged = {
        "task_id": PCPR_003_TASK_ID,
        "status": "implemented",
        "qualification_verdict": sections["qualification_verdict"],
        "current_tree_binding": {
            **current_head_pcpr_003_current_tree_binding(),
            "evidence_kind": "simulated",
        },
    }
    with pytest.raises(LegacyBypassInventoryError, match="measured"):
        validate_pcpr_003_outer_receipt(forged)


def test_receipt_promotion_rejects_supervisor_promoted() -> None:
    verdict = inventory_legacy_bypass_and_false_authority_paths(
        paths=_unavailable_paths(),
        competing_authorities=_unavailable_competing(),
    )
    mutated = {
        **verdict.to_mapping(),
        "promotion_status": "supervisor_promoted",
    }
    # Build via dataclass replacement would still go through pcpr_003_receipt_promotion
    # only after constructing InventoryVerdict; use a shallow copy of the real verdict
    # by calling the helper with a forged mapping-level check on outer validator.
    forged = {
        "task_id": PCPR_003_TASK_ID,
        "status": "implemented",
        "qualification_verdict": {
            **pcpr_003_receipt_promotion(verdict),
            "promotion_status": "supervisor_promoted",
            "verdict_cid": current_head_pcpr_003_receipt_promotion()["verdict_cid"],
        },
    }
    with pytest.raises(LegacyBypassInventoryError, match="cannot promote"):
        validate_pcpr_003_outer_receipt(forged)
    del mutated


def test_every_path_has_required_inventory_fields() -> None:
    verdict = qualify_current_head_inventory()
    for item in verdict.paths:
        mapping = item.to_mapping()
        assert mapping["path_id"]
        assert mapping["authority_analysis"]
        assert mapping["disposition"]
        assert mapping["migration_impact"]
        assert mapping["source_identity"]["repository"]
        assert mapping["evidence_kind"] in {"measured", "unavailable"}
        if mapping["evidence_kind"] == "unavailable":
            assert mapping["present"] is None
            assert mapping["hit_count"] == 0
        if mapping["present"] is True:
            assert mapping["hit_count"] >= 1
            assert mapping["sample_hits"]
            assert mapping["source_identity"].get("primary_relpath")
